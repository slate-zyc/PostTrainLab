"""
训练数据结构定义，规范化对话文本并检查 SFT 和偏好对。

输入
    Hugging Face 原始记录、数据集标识、上游拆分和缓存行号。
输出
    已校验的 SFTExample 或 PreferenceExample 对象，可导出为项目统一的 JSONL。
"""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, StrictStr, field_validator, model_validator


# ============================================================
# 文本与对话检查
# ============================================================

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
PLACEHOLDERS = {"n/a", "none", "null", "[removed]", "[deleted]", "[placeholder]"}


class Message(BaseModel):
    """一轮会话中的角色和正文。

    输入
        role 为 system、user 或 assistant，content 为文本字符串。
    输出
        Pydantic 实例。正文在实例化时完成换行与首尾空白清理。
    """
    role: Literal["system", "user", "assistant"]
    content: StrictStr

    @field_validator("content")  # 校验 Message 的 content 字段，同时返回清理后的正文
    @classmethod
    def clean_text(cls, text: str) -> str:
        """清理消息正文并筛掉明显无效文本。

        输入
            text 是 Pydantic 已检查为字符串的 content 字段。
        输出
            返回清理后的文本，保留正文里的代码缩进和 Markdown。
            空文本、异常字符和占位内容会触发带过滤原因的 ValueError。
        """
        # 统一换行和首尾空白，保留正文里的缩进、代码块和 Markdown。
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"\A<(?:html|body)>\s*|\s*</(?:html|body)>\Z", "", text, flags=re.I).strip()
        if not text:
            raise ValueError("empty_message")
        if CONTROL_CHARS.search(text) or text.count("\ufffd") >= 3:
            raise ValueError("invalid_character")
        if text.casefold() in PLACEHOLDERS or not any(char.isalnum() for char in text):
            raise ValueError("placeholder_or_symbols")
        return text


def validate_dialogue(messages: list[Message], last_role: str) -> None:
    """核对对话角色序列和结束角色。

    输入
        messages 是已校验的 Message 列表。last_role 指定期望的最后角色。
        SFT 对话以 assistant 结束，偏好对的共同上文以 user 结束。
    输出
        校验成功返回 None，角色顺序有误时抛出 ValueError。
    """
    roles = [message.role for message in messages]
    if roles[:1] == ["system"]:
        roles = roles[1:]
    # 去掉可选的 system 后，生成 user、assistant 交替的期望序列。
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(roles))]
    if not roles or roles != expected or roles[-1] != last_role:
        raise ValueError("invalid_message_order")


def normalize_prompt(text: str) -> str:
    """为文本比较生成统一的 Prompt 表示。

    输入
        text 是待比较的原文。
    输出
        返回 Unicode 兼容归一化、大小写折叠及合并空白后的文本，用于精确和近似去重。
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


# ============================================================
# SFT 和 Preference 数据结构
# ============================================================

class SFTExample(BaseModel):
    """统一的 SFT 样本结构。

    输入
        id、来源、语言、完整多轮 messages 和来源 metadata。
    输出
        已校验的 SFTExample。model_dump() 可转为 JSONL 所需结构。
    """
    id: str
    source: str
    language: str
    messages: list[Message]
    metadata: dict

    @model_validator(mode="after")  # messages 内的字段解析完成后再检查角色序列
    def check_dialogue(self):
        """检查 SFT 样本是否具有完整的问答轮次。

        输入
            self 是各字段已解析完成的 SFTExample。
        输出
            返回通过校验的实例。对话必须以 assistant 的回答结束。
        """
        validate_dialogue(self.messages, "assistant")
        return self


class PreferenceExample(BaseModel):
    """统一的偏好对结构，显式保存共同上文。

    输入
        id、来源、共同 Prompt、chosen、rejected 和来源 metadata。
    输出
        已校验的 PreferenceExample，两个回答各包含一条 assistant 消息。
    """
    id: str
    source: str
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    metadata: dict

    @model_validator(mode="after")
    def check_pair(self):
        """检查共同 Prompt 和两个候选回答。

        输入
            self 是字段解析完成的 PreferenceExample。
        输出
            返回通过校验的实例。非法角色或归一化后相同的回答会触发 ValueError。
        """
        validate_dialogue(self.prompt, "user")
        for response in (self.chosen, self.rejected):
            if len(response) != 1 or response[0].role != "assistant":
                raise ValueError("invalid_preference_answer")
        if normalize_prompt(self.chosen[0].content) == normalize_prompt(self.rejected[0].content):
            raise ValueError("identical_preference_answers")
        return self


def first_user(example: SFTExample | PreferenceExample) -> str:
    """提取样本中的第一条用户消息。

    输入
        example 是通过结构校验的 SFT 或偏好样本。
    输出
        返回首条 user 消息的正文，用于任务类别统计。
    """
    messages = example.messages if isinstance(example, SFTExample) else example.prompt
    return next(message.content for message in messages if message.role == "user")


# ============================================================
# 官方数据格式 → 项目数据格式
# ============================================================

def convert_record(row: dict, dataset_id: str, split: str, row_number: int, kind: str):
    """把上游原始样本转成项目的统一训练结构。

    输入
        row 是官方数据中的单条记录，dataset_id、split 和 row_number 用于追溯来源。
        kind 指定 sft 或 preference，决定解析 messages 还是 chosen、rejected。
    输出
        返回对应的 Pydantic 样本。偏好分支的上文不一致时抛出 ValueError。

    偏好样本示例
        chosen   = [共同 Prompt, 较优回答]
        rejected = [共同 Prompt, 较差回答]
        清洗时拆成 prompt、chosen、rejected 三个字段，并保留上游评分。
    """
    metadata = {"source_split": split, "source_row": row_number, "source_prompt_id": row.get("prompt_id")}
    example_id = f"{dataset_id}:{split}:{row_number}"  # 通过上游拆分和缓存行号定位原始样本
    if kind == "sft":
        return SFTExample.model_validate({
            "id": example_id,
            "source": dataset_id,
            "language": "en",
            "messages": row["messages"],
            "metadata": metadata,
        })

    chosen = [Message.model_validate(message) for message in row["chosen"]]
    rejected = [Message.model_validate(message) for message in row["rejected"]]
    # UltraFeedback 两条分支都包含上文。先核对共同 Prompt，再分别取最后一句回答。
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("preference_prompt_mismatch")
    metadata.update({
        "language": "en",
        "preference_source": "model_judgment",
        "score_chosen": row.get("score_chosen"),
        "score_rejected": row.get("score_rejected"),
    })
    return PreferenceExample.model_validate({
        "id": example_id,
        "source": dataset_id,
        "prompt": chosen[:-1],
        "chosen": chosen[-1:],
        "rejected": rejected[-1:],
        "metadata": metadata,
    })
