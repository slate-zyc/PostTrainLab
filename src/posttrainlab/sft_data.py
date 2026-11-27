"""
构造 Qwen2.5 SFT 的 Completion-only 标签和训练数据。

输入
    项目统一的多轮对话、Qwen2.5 Tokenizer 和最大序列长度。
输出
    含 input_ids、attention_mask、labels 的 Hugging Face Dataset 和统计记录。
"""

from pathlib import Path
from random import Random

from datasets import Dataset

from posttrainlab.data.sources import read_jsonl


IGNORE_INDEX = -100
TURN_END = "<|im_end|>"


# ============================================================
# 多轮对话：Chat Template → Assistant 字符区间 → Token 标签
# ============================================================

def encode_sft_example(messages: list[dict], tokenizer, max_length: int) -> tuple[dict, bool]:
    """将一条对话转换为带回答标签的模型输入。

    输入
        messages 是可选 system 与多轮 user、assistant 消息。
        tokenizer 使用 Qwen2.5 的官方 Chat Template，max_length 限制实际输入长度。
    输出
        返回 (features, truncated)。features 包含输入 ID、注意力掩码和标签。
        每轮 Assistant 正文及结束符有标签，角色头、用户和系统消息设为 -100。
    """
    # --- 步骤 1: 渲染整段对话，再用 Assistant 生成提示确定每轮回答的起点 ---
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    answer_spans = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
            answer = message["content"] + TURN_END
            if not rendered.startswith(prefix + answer):
                raise ValueError("Qwen2.5 Chat Template 的 Assistant 边界与预期不一致")
            answer_spans.append((len(prefix), len(prefix) + len(answer)))

    # --- 步骤 2: 一次分词并右截断，避免单独分词时边界的 token 发生变化 ---
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,  # Chat Template 已加入角色和结束标记
        return_offsets_mapping=True,  # 将每个 token 映射回渲染文本中的字符区间
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")

    # --- 步骤 3: 只监督回答区间中的 token，截断时保留实际出现的结束符 ---
    encoded["labels"] = [
        token_id if any(start <= left < right <= end for start, end in answer_spans) else IGNORE_INDEX
        for token_id, (left, right) in zip(encoded["input_ids"], offsets, strict=True)
    ]
    return dict(encoded), offsets[-1][1] < len(rendered)


# ============================================================
# 文件读取：固定抽样顺序 → 编码 → 统计可训练样本
# ============================================================

def load_sft_dataset(
    path: Path, tokenizer, max_length: int, limit: int | None = None, seed: int | None = None,
) -> tuple[Dataset, dict]:
    """从统一 JSONL 建立训练或验证用的 Dataset。

    输入
        path 是现有 JSONL，tokenizer 和 max_length 用于编码。
        limit 限制可用样本条数，seed 用于训练样本的固定抽样顺序。
    输出
        返回 (dataset, stats)。stats 记录来源、检查条数、截断、跳过和实际 token 数。
    """
    rows = list(read_jsonl(path))
    if seed is not None:
        Random(seed).shuffle(rows)  # 用相同顺序取前缀，保证不同规模实验的样本嵌套

    features = []
    scanned = truncated = no_answer = input_tokens = answer_tokens = 0
    for row in rows:
        encoded, was_truncated = encode_sft_example(row["messages"], tokenizer, max_length)
        scanned += 1
        truncated += was_truncated
        supervised = sum(label != IGNORE_INDEX for label in encoded["labels"])
        if not supervised:
            no_answer += 1  # 长 Prompt 占满输入时跳过，避免一个 batch 完全没有监督
            continue
        features.append(encoded)
        input_tokens += len(encoded["input_ids"])
        answer_tokens += supervised
        if len(features) == limit:
            break

    if not features or (limit is not None and len(features) < limit):
        raise ValueError(f"{path} 有 {len(features)} 条可训练样本，要求 {limit or '至少 1'} 条")
    stats = {
        "file": str(path),
        "source_rows": len(rows),
        "scanned_rows": scanned,
        "rows": len(features),
        "truncated_rows": truncated,
        "skipped_without_answer": no_answer,
        "input_tokens": input_tokens,
        "supervised_tokens": answer_tokens,
    }
    return Dataset.from_list(features), stats
