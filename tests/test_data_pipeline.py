"""
数据处理的离线测试，覆盖格式、长度筛选、偏好对齐和跨集合重复。

输入
    函数中构造的少量对话和偏好对，用可预测的词数代替真实分词。
输出
    unittest 的通过或失败结果。

运行方式
    python -m unittest discover -s tests -p test_data_pipeline.py
"""

import unittest

from posttrainlab.data.prepare import clean_rows
from posttrainlab.data.schema import convert_record, normalize_prompt
from posttrainlab.data.split import prompt_text, split_datasets


# ============================================================
# 测试样例和分词替身
# ============================================================

def message(role: str, content: str) -> dict:
    """构造供测试使用的一条消息。

    输入
        role 为角色，content 为正文。
    输出
        返回含 role 和 content 的字典。
    """
    return {"role": role, "content": content}


def sft_row(prompt: str, answer: str) -> dict:
    """构造一条原始 SFT 测试样本。

    输入
        prompt 是问题，answer 是回答。
    输出
        返回含 user 和 assistant 两条消息的 messages 字典。
    """
    return {"messages": [message("user", prompt), message("assistant", answer)]}


def preference_row(prompt: str, chosen: str, rejected: str) -> dict:
    """构造包含共同 Prompt 的偏好对样例。

    输入
        prompt 是两条分支共用的问题，chosen、rejected 分别是候选回答。
    输出
        返回两条完整对话分支，格式与原始偏好记录一致。
    """
    common = [message("user", prompt)]
    return {"chosen": common + [message("assistant", chosen)], "rejected": common + [message("assistant", rejected)]}


def sample_tokenizer(texts, **parameters):
    """用空格分词模拟 Tokenizer 的长度返回值。

    输入
        texts 是文本列表，parameters 接收与实际调用一致的分词选项。
    输出
        返回带 length 列表的字典，每个长度等于对应文本的词数。
    """
    # 用可预测的长度验证阈值，真实分词器在数据准备脚本中使用。
    return {"length": [len(text.split()) for text in texts]}


# ============================================================
# 验证格式、清洗和跨集合泄漏防护
# ============================================================

class DataPipelineTest(unittest.TestCase):
    """数据处理规则的单元测试集合。

    输入
        测试自行构造对话、偏好对和拆分配置。
    输出
        由 unittest 报告格式校验、长度过滤、去重及拆分是否符合预期。
    """
    def test_sft_keeps_indentation_and_filters_empty_or_long_answer(self):
        """验证正文清理保留缩进，空回答和超长样本会被筛掉。

        输入
            带 CRLF 与缩进的有效样本，以及空回答和超长回答样本。
        输出
            通过断言核对保留数、换行、缩进和两个拒收原因。
        """
        rows = [
            sft_row(" Explain Python. ", "Line 1\r\n    return 1"),
            sft_row("Name a color.", ""),
            sft_row("List the steps.", "one two three four five six seven eight nine ten"),
        ]
        accepted, rejects, stats = clean_rows(rows, "sft-fixture", "train", "sft", sample_tokenizer, 8)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["messages"][-1]["content"], "Line 1\n    return 1")
        self.assertEqual({item["reason"] for item in rejects}, {"empty_message", "too_long"})
        self.assertEqual(stats["schema_valid_rows"], 2)

    def test_preference_requires_matching_prompt_and_distinct_answers(self):
        """验证偏好对能够正确拆分并拒收不匹配的分支。

        输入
            合法偏好对、两分支 Prompt 不同的偏好对，以及两个回答相同的偏好对。
        输出
            通过断言核对共同 Prompt、保留的回答和两个拒收原因。
        """
        valid = preference_row("What is LoRA?", "It trains adapters.", "It retrains the entire model.")
        mismatch = preference_row("What is LoRA?", "An adapter method.", "An optimizer.")
        mismatch["rejected"][0] = message("user", "What is AdamW?")
        identical = preference_row("Name a color.", "Red.", "Red.")
        accepted, rejects, _ = clean_rows(
            [valid, mismatch, identical], "preference-fixture", "train_prefs", "preference", sample_tokenizer, 45,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(accepted[0]["prompt"]), 1)
        self.assertEqual(len(accepted[0]["chosen"]), 1)
        self.assertEqual(
            {item["reason"] for item in rejects}, {"preference_prompt_mismatch", "identical_preference_answers"},
        )

    def test_test_prompts_are_protected_and_splits_are_repeatable(self):
        """验证测试样本优先保留，固定配置复跑时划分一致。

        输入
            两类样本的官方 train、test，其中偏好 train 含与 SFT 测试题重复及近似的问题。
        输出
            通过断言核对去重原因、匹配对象、复跑结果和最终 Prompt 的唯一性。
        """
        shared = "Explain how gradient accumulation changes the effective batch size."
        near = shared.replace("size.", "size?")
        inputs = {
            "sft": {
                "test": [convert_record(sft_row(shared, "It accumulates gradients."), "sft", "test", 1, "sft").model_dump()],
                "train": [convert_record(sft_row(prompt, "A useful answer."), "sft", "train", index, "sft").model_dump()
                          for index, prompt in enumerate(["Name an ocean.", "Explain derivatives.", "Summarize this text."], 1)],
            },
            "preference": {
                "test": [convert_record(preference_row("Name a moon.", "Europa", "Mars"), "pref", "test", 1, "preference").model_dump()],
                "train": [convert_record(preference_row(prompt, "A", "B"), "pref", "train", index, "preference").model_dump()
                          for index, prompt in enumerate([shared, near, "Name a planet.", "Write a poem.", "Define gravity."], 1)],
            },
        }
        config = {"seed": 42, "validation_ratio": 0.33, "near_duplicate_threshold": 97}
        configs = {"sft": config, "preference": config}
        outputs, duplicates = split_datasets(inputs, configs)
        self.assertEqual((outputs, duplicates), split_datasets(inputs, configs))
        self.assertEqual({row["reason"] for row in duplicates}, {"exact_prompt_duplicate", "near_prompt_duplicate"})
        self.assertTrue(all(row["matched_id"] == "sft:test:1" for row in duplicates))
        prompts = [normalize_prompt(prompt_text(row)) for parts in outputs.values()
                   for split in parts.values() for row in split]
        self.assertEqual(len(prompts), len(set(prompts)))
