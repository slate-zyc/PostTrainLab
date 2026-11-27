"""
准备 E1 的清洗前训练组，并沿用 v1 的清洗后训练组。

输入
    现有 v1 数据清单、Hugging Face 原始训练样本和固定评测拆分。
输出
    清洗前训练用 JSONL 和对照数据记录。

运行方式
    python scripts/prepare_sft_ablation.py
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from random import Random

from posttrainlab.data.schema import Message, validate_dialogue
from posttrainlab.data.sources import PROJECT_DIR, read_jsonl, read_yaml, write_json, write_jsonl
from posttrainlab.data.split import PromptIndex, prompt_text


# ============================================================
# 对照数据：保留原文与训练内重复，仅保护固定评测集
# ============================================================

def prepare_uncleaned(manifest_path: Path, output_dir: Path, threshold: float, samples: int | None, seed: int) -> None:
    """从清洗后训练组的同一来源建立清洗前对照组。

    输入
        manifest_path 是数据清单，output_dir 保存对照产物。
        threshold 使用数据阶段的近似去重阈值保护验证与测试 Prompt。
        samples 可限制实验子集条数，seed 固定该子集的选择。
    输出
        返回 None，保存同源的 uncleaned.jsonl、cleaned.jsonl 和 data.json。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sft = manifest["datasets"]["sft"]
    raw = sft["raw_sources"]["train"]
    held_out = PromptIndex(threshold)
    for split in ("validation", "test"):
        for row in read_jsonl(PROJECT_DIR / sft["splits"][split]["file"]):
            held_out.add(prompt_text(row), row["id"])

    rows = []
    invalid_dialogue = evaluation_overlap = 0
    for number, row in enumerate(read_jsonl(PROJECT_DIR / raw["file"]), start=1):
        messages = row["messages"]
        try:
            # 跳过正文清洗器，只确认消息可以渲染并保存为 UTF-8。
            validate_dialogue([Message.model_construct(**message) for message in messages], "assistant")
            for message in messages:
                message["content"].encode("utf-8")
        except (ValueError, AttributeError):
            invalid_dialogue += 1
            continue

        # 索引只含固定验证和测试题，训练组内部重复在清洗前对照中保留。
        if held_out.find(prompt_text(row)):
            evaluation_overlap += 1
            continue
        rows.append({
            "id": f"{raw['dataset_id']}:{raw['split']}:{number}",
            "messages": messages,
            "metadata": {"source_split": raw["split"], "source_row": number},
        })

    # 先固定一份原始实验子集，再以原始行号从清洗结果取同一批样本。
    eligible = len(rows)
    if samples is not None:
        rows = Random(seed).sample(rows, samples)
    row_ids = {row["id"] for row in rows}
    cleaned = [row for row in read_jsonl(PROJECT_DIR / sft["splits"]["train"]["file"]) if row["id"] in row_ids]
    if not cleaned:
        raise ValueError("原始实验子集没有对应的清洗后训练样本")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "uncleaned.jsonl", rows)
    write_jsonl(output_dir / "cleaned.jsonl", cleaned)
    write_json(output_dir / "data.json", {
        "data_version": manifest["version"],
        "source": raw,
        "raw_rows": raw["rows"],
        "eligible_rows": eligible,
        "uncleaned_rows": len(rows),
        "cleaned_rows": len(cleaned),
        "seed": seed,
        "invalid_dialogue": invalid_dialogue,
        "evaluation_overlap": evaluation_overlap,
        "cleaned": sft["splits"]["train"],
        "held_out": {split: sft["splits"][split] for split in ("validation", "test")},
    })
    print(f"清洗前训练组 {len(rows)} 条，已写入 {output_dir}", flush=True)


def main() -> None:
    """解析 E1 对照数据的版本清单和输出目录。

    输入
        --manifest 指定数据清单，--samples 指定原始实验子集大小，--output-dir 指定产物目录。
    输出
        返回 None，写入对照样本与处理记录。
    """
    parser = argparse.ArgumentParser(description="准备 SFT 清洗前后对照数据")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/manifests/dataset_v1.json")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = read_yaml(PROJECT_DIR / "configs/data_sft.yaml")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_dir or PROJECT_DIR / "outputs/sft_ablation" / stamp
    prepare_uncleaned(args.manifest, output_dir, config["near_duplicate_threshold"], args.samples, config["seed"])


if __name__ == "__main__":
    main()
