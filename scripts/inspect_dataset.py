"""
候选数据抽查入口，从本地原始缓存提取样例和字段统计。

输入
    configs/data_sources.yaml 和按该配置保存的 Hugging Face 抽样 JSONL。
输出
    候选数据各自的 samples.jsonl 与 summary.json，保存在 outputs/data_quality/candidates/。

运行方式
    python scripts/inspect_dataset.py --dataset-id HuggingFaceH4/ultrachat_200k
"""

import argparse
import statistics
from collections import Counter

from posttrainlab.data.sources import PROJECT_DIR, SOURCES_CONFIG, read_raw_sample, read_yaml, write_json, write_jsonl


def text_length(value) -> int:
    """计算候选字段中正文的字符数。

    输入
        value 可以是对话字符串、由消息字典组成的列表，或空字段。
    输出
        返回正文字数。消息列表按 content 的长度求和。
    """
    # 数据卡中的内容字段分别为字符串或消息列表。
    if isinstance(value, list):
        return sum(len(message["content"] or "") for message in value)
    return len(value or "")


# ============================================================
# 主函数：读取候选配置 → 抽样 → 保存审查材料
# ============================================================

def main() -> None:
    """按候选配置汇总字段、字符长度和来源信息。

    输入
        --dataset-id 可指定一个候选。默认遍历候选目录。
    输出
        返回 None。每个候选生成一份抽查样本和一份摘要。
    """
    config = read_yaml(SOURCES_CONFIG)
    parser = argparse.ArgumentParser(description="抽样审查 Hugging Face 候选数据")
    parser.add_argument("--dataset-id", choices=config["sources"])
    args = parser.parse_args()

    dataset_ids = [args.dataset_id] if args.dataset_id else list(config["sources"])
    for dataset_id in dataset_ids:
        source = config["sources"][dataset_id]
        rows, provenance = read_raw_sample(
            dataset_id, source["train_split"], config["sample_size"],
            config["sample_seed"], config["shuffle_buffer"], source,
        )

        # 完整样本以 JSONL 保存，方便在本地逐条审查。终端显示结构和统计摘要。
        audit_rows = [
            {"sample": index, **{field: row.get(field) for field in source["audit_fields"]}}
            for index, row in enumerate(rows, start=1)
        ]
        field_counts = Counter(field for row in rows for field in row)  # 一个字段出现一次就计入一条样本
        content_fields = [field for field in ("messages", "text", "chosen", "rejected") if field in source["audit_fields"]]
        content_chars = [sum(text_length(row[field]) for field in content_fields) for row in audit_rows]
        output = PROJECT_DIR / "outputs" / "data_quality" / "candidates" / dataset_id.replace("/", "--")
        write_jsonl(output / "samples.jsonl", audit_rows)
        write_json(output / "summary.json", {
            **provenance,
            "fields": dict(sorted(field_counts.items())),
            "sample_content_chars": {
                "min": min(content_chars), "median": statistics.median(content_chars), "max": max(content_chars),
            },
            "language_metadata": dict(Counter(row["lang"] for row in rows if row.get("lang"))),  # 统计上游已有的语言标签
        })
        print(f"{dataset_id}: {len(rows)} 条，字段 {', '.join(sorted(field_counts))}", flush=True)


if __name__ == "__main__":
    main()
