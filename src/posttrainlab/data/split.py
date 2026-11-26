"""
按首条用户消息去重，并划分训练、验证和测试数据。

输入
    已标准化的 SFT、Preference 记录，按官方 train 与 test 拆分保存。
输出
    两类数据各自的 train、validation、test，以及每条重复匹配的记录。
"""

import random
from collections import defaultdict

from rapidfuzz import fuzz

from posttrainlab.data.schema import normalize_prompt


# ============================================================
# Prompt 精确匹配和保守的近似匹配
# ============================================================

def prompt_text(row: dict) -> str:
    """提取最终去重使用的首条用户消息。

    输入
        row 是标准化后的 SFT 记录，或含 prompt 的偏好记录。
    输出
        返回第一条 role 为 user 的正文。
    """
    messages = row["messages"] if "messages" in row else row["prompt"]
    return next(message["content"] for message in messages if message["role"] == "user")


def signature(text: str) -> list[tuple[str, str]]:
    """挑选近似去重时用于缩小候选范围的文本首尾。

    输入
        text 是规范化后的 Prompt。
    输出
        返回首尾各 16 字符的索引标记。短于 32 字符时返回空列表，
        此类 Prompt 只参加精确去重。
    """
    # 候选须共享首部或尾部。短问题只做精确匹配，减少误删。
    if len(text) < 32:
        return []
    return [("prefix", text[:16]), ("suffix", text[-16:])]


class PromptIndex:
    """在两类数据之间复用已保留问题的索引。

    输入
        threshold 是近似匹配分数的下限。
    输出
        可查找精确或近似重复、记录已保留样本的索引实例。
    """

    def __init__(self, threshold: float):
        """保存阈值并初始化精确索引和首尾候选索引。

        输入
            threshold 使用 RapidFuzz 的 0–100 分制。
        输出
            返回 None。self.exact 负责精确查找，self.buckets 负责缩小近似候选。
        """
        self.threshold = threshold
        self.exact: dict[str, str] = {}
        self.buckets: dict[tuple[str, str], list[str]] = defaultdict(list)

    def find(self, text: str) -> tuple[str, str, float] | None:
        """检查一条 Prompt 是否与已保留样本重复。

        输入
            text 是待比较的原始用户消息。
        输出
            命中时返回 (原因, 已保留样本 ID, 分数)，没有命中时返回 None。
            先找规范化文本完全相同的样本，再检查共用首部或尾部的候选。
        """
        normalized = normalize_prompt(text)
        if normalized in self.exact:
            return "exact_prompt_duplicate", self.exact[normalized], 100.0
        # 同一个候选可能同时匹配首部和尾部，先合并再计算相似度。
        candidates = dict.fromkeys(
            other for anchor in signature(normalized) for other in self.buckets.get(anchor, [])
        )
        for other in candidates:
            score = fuzz.ratio(normalized, other, score_cutoff=self.threshold)  # 相似度范围为 0–100
            if score:
                return "near_prompt_duplicate", self.exact[other], round(score, 2)
        return None

    def add(self, text: str, example_id: str) -> None:
        """把已确认保留的用户消息放进索引。

        输入
            text 是原始用户消息，example_id 是对应的样本 ID。
        输出
            返回 None。更新规范化文本的精确映射以及首尾候选列表。
        """
        normalized = normalize_prompt(text)
        self.exact[normalized] = example_id
        for anchor in signature(normalized):
            self.buckets[anchor].append(normalized)


# ============================================================
# 跨数据集去重和训练/验证划分
# ============================================================

def split_datasets(inputs: dict, configs: dict) -> tuple[dict, list[dict]]:
    """联合保留测试题、剔除重复问题，再划分训练与验证集。

    输入
        inputs[kind][split] 是 SFT 和 Preference 的官方 train、test 样本列表。
        configs 包含近似去重阈值、随机种子和验证集比例。
    输出
        返回 (outputs, duplicates)。outputs 按数据类型和新拆分存放样本，
        duplicates 记录被剔除样本、匹配对象、原因和分数。

    实现思路
        先处理所有测试样本，训练样本撞到已保留的测试问题时予以剔除。
        再用固定种子从每类去重后的官方训练数据中划出验证集。
    """
    kinds = ("sft", "preference")
    threshold = min(configs[kind]["near_duplicate_threshold"] for kind in kinds)  # 跨数据源使用统一阈值，取两个设置中较低的值
    index = PromptIndex(threshold)
    outputs = {kind: {"train": [], "test": []} for kind in kinds}
    duplicates = []

    # 先收两个官方测试集，再收训练集。撞到测试题的训练记录会被移除。
    for split in ("test", "train"):
        for kind in kinds:
            for row in inputs[kind][split]:
                text = prompt_text(row)
                duplicate = index.find(text)
                if duplicate:
                    reason, matched_id, score = duplicate
                    duplicates.append({
                        "kind": kind, "split": split, "id": row["id"], "reason": reason,
                        "matched_id": matched_id, "score": score,
                    })
                    continue

                index.add(text, row["id"])
                outputs[kind][split].append(row)

    for kind in kinds:
        train = outputs[kind]["train"]
        config = configs[kind]
        random.Random(config["seed"]).shuffle(train)  # 固定验证集划分
        # 在数据量允许时保证验证集至少一条，并为训练集至少留一条。
        count = min(len(train) - 1, max(1, round(len(train) * config["validation_ratio"])))
        outputs[kind]["validation"] = train[:count]
        outputs[kind]["train"] = train[count:]
        if not all(outputs[kind].values()):
            raise ValueError(f"{kind} 划分后有空集合，请检查数据规模与过滤规则")

    return outputs, duplicates
