"""
将上游数据转成统一结构，再按正文 token 长度过滤。

输入
    SFT 或偏好数据配置、原始训练与测试缓存、固定版本的 Qwen 分词器。
输出
    data/interim/ 中的标准化 JSONL、拒收记录和中间清单。
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from filelock import FileLock
from pydantic import ValidationError
from transformers import AutoTokenizer

from posttrainlab.data.schema import SFTExample, convert_record, first_user
from posttrainlab.data.sources import (
    PROJECT_DIR, SOURCES_CONFIG, read_raw_sample, read_yaml, write_json, write_jsonl,
)


# ============================================================
# 格式校验、类别与长度
# ============================================================

TASK_RULES = [
    ("translation", r"\btranslat(?:e|ion)\b"),
    ("summarization", r"\bsummari[sz](?:e|ation)\b|\bsummary\b"),
    ("code", r"\b(?:python|javascript|sql|code|programming|function)\b"),
    ("math", r"\b(?:equation|calculate|algebra|probability|arithmetic)\b"),
    ("writing", r"\b(?:write|story|poem|draft|rewrite|essay)\b"),
    ("explanation", r"\b(?:explain|describe|define|what is|how does)\b"),
]


def task_category(prompt: str) -> str:
    """按首条用户消息的关键词粗分任务类型。

    输入
        prompt 是用户消息正文。
    输出
        返回第一条匹配规则的类别，例如 code 或 translation。其余归为 other。
    """
    for category, pattern in TASK_RULES:  # 按定义顺序尝试，所以更具体的规则排在前面
        if re.search(pattern, prompt, re.I):
            return category
    return "other"


def message_texts(example) -> list[str]:
    """收集一条样本里的待分词正文。

    输入
        example 是已标准化的 SFTExample 或 PreferenceExample。
    输出
        返回正文字符串列表。SFT 包含完整对话，偏好数据包含共同上文和两份回答。
    """
    if isinstance(example, SFTExample):
        return [message.content for message in example.messages]
    return [message.content for message in example.prompt + example.chosen + example.rejected]


def token_counts(example, lengths: dict[str, int]) -> dict[str, int]:
    """根据已测得的正文长度，计算单条样本的长度统计。

    输入
        example 是标准化样本。lengths 把消息正文映射到其 token 数。
    输出
        返回含 prompt、answer、total 的计数字典。偏好样本还包含 chosen 和 rejected。
        SFT 对所有 Assistant 回合的回答求和。偏好数据以较长回答计算总长。
    """
    # 这里统计正文 token。应用 Chat Template 后另行统计完整训练序列。
    if isinstance(example, SFTExample):
        context = sum(lengths[message.content] for message in example.messages if message.role != "assistant")
        answer = sum(lengths[message.content] for message in example.messages if message.role == "assistant")
        return {"prompt": context, "answer": answer, "total": context + answer}

    context = sum(lengths[message.content] for message in example.prompt)
    chosen = lengths[example.chosen[0].content]
    rejected = lengths[example.rejected[0].content]
    longest = max(chosen, rejected)
    return {"prompt": context, "answer": longest, "total": context + longest, "chosen": chosen, "rejected": rejected}


def clean_rows(rows: list[dict], dataset_id: str, split: str, kind: str, tokenizer, max_tokens: int):
    """校验原始记录并按正文长度筛选训练样本。

    输入
        rows 是来自同一上游拆分的原始记录。dataset_id、split 用于记录来源。
        kind 选择数据格式，tokenizer 计算正文长度，max_tokens 是正文 token 上限。
    输出
        返回 (accepted, rejects, stats)。accepted 是通过校验和长度过滤的 JSON 记录。
        rejects 记录原始行号和原因，stats 保存过滤计数和所有格式有效样本的原始长度。

    关键步骤
        先分清结构问题和文本问题，再批量分词。分词前去掉重复正文，
        共用 Prompt 的偏好对可以直接复用同一个长度。
    """
    # --- 步骤 1: 标准化字段，记录校验失败的样本 ---
    valid = []
    rejects = []
    for row_number, row in enumerate(rows, start=1):
        try:
            valid.append(convert_record(row, dataset_id, split, row_number, kind))
        except ValidationError as error:
            issue = error.errors(include_url=False, include_input=False, include_context=False)[0]
            reason = issue["msg"].removeprefix("Value error, ") if issue["type"] == "value_error" else issue["type"]
            rejects.append({"source": dataset_id, "split": split, "row": row_number, "reason": reason})
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error) if type(error) is ValueError else "invalid_source_fields"
            rejects.append({"source": dataset_id, "split": split, "row": row_number, "reason": reason})

    # --- 步骤 2: 分批统计不同正文的 token 数 ---
    # 用正文作字典的键，偏好对的共享 Prompt 只需计数一次。
    texts = list(dict.fromkeys(text for example in valid for text in message_texts(example)))
    lengths = {}
    for start in range(0, len(texts), 256):
        chunk = texts[start:start + 256]
        encoded = tokenizer(chunk, add_special_tokens=False, return_length=True, verbose=False)  # 只统计正文长度
        lengths.update(zip(chunk, encoded["length"], strict=True))

    # --- 步骤 3: 按整条样本的长度筛选，并保留筛选前计数 ---
    accepted = []
    before_lengths = []
    for example in valid:
        measured = token_counts(example, lengths)
        before_lengths.append(measured)
        if measured["total"] > max_tokens:
            rejects.append({
                "source": dataset_id, "split": split, "row": example.metadata["source_row"],
                "reason": "too_long", "total_tokens": measured["total"],
            })
            continue
        example.metadata["tokens"] = measured
        example.metadata["task_category"] = task_category(first_user(example))
        accepted.append(example.model_dump(mode="json"))

    counts = Counter(item["reason"] for item in rejects)
    return accepted, rejects, {"raw_rows": len(rows), "schema_valid_rows": len(valid), "reasons": dict(counts),
                                "before_tokens": before_lengths}


# ============================================================
# 阶段入口：本地原始数据 → 中间数据
# ============================================================

def prepare_main(kind: str) -> None:
    """运行一种数据类型的标准化和过滤。

    输入
        kind 为 sft 或 preference。--config 可选择相应的 YAML。
        YAML 指定需要读取的上游样本、分词器和长度上限。
    输出
        返回 None。将标准化训练和测试缓存、过滤记录及清单写入 data/interim/<kind>/。
    """
    parser = argparse.ArgumentParser(description=f"标准化和清洗 {kind} 数据")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / f"data_{kind}.yaml")
    config = read_yaml(parser.parse_args().config)
    catalog = read_yaml(SOURCES_CONFIG)
    dataset_id = config["dataset_id"]
    source = catalog["sources"][dataset_id]
    if source["kind"] != kind:
        parser.error(f"此脚本需要 {kind} 数据源")

    snapshots = {}
    for split in ("train", "test"):
        snapshots[split] = read_raw_sample(
            dataset_id, source[f"{split}_split"], config[f"{split}_rows"],
            config["seed"], catalog["shuffle_buffer"], source, config.get(f"{split}_file"),
        )

    # 加载固定版本的 Tokenizer，统一全部样本的长度计数口径。
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_id"], revision=config["tokenizer_revision"], token=False, trust_remote_code=False,
    )
    output = PROJECT_DIR / "data" / "interim" / kind
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": kind, "config": config, "splits": {}}
    with FileLock(str(output / "prepare.lock")):
        for split, (rows, provenance) in snapshots.items():
            accepted, rejects, stats = clean_rows(
                rows, dataset_id, provenance["split"], kind, tokenizer, config["max_content_tokens"],
            )
            clean_path = output / f"{split}.jsonl"
            reject_path = output / f"{split}_rejected.jsonl"
            write_jsonl(clean_path, accepted)
            write_jsonl(reject_path, rejects)
            manifest["splits"][split] = {
                "provenance": provenance,  # 保留上游拆分、固定提交和缓存路径，方便追到原样本
                "rows": len(accepted),
                **stats,
            }
            print(f"{kind}/{split}: {len(rows)} → {len(accepted)} 条，清洗拒收 {len(rejects)} 条", flush=True)
        write_json(output / "manifest.json", manifest)
