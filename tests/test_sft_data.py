"""
验证真实 Qwen2.5 Chat Template 下的多轮标签、截断、补位和因果移位。

输入
    固定版本的 Qwen Tokenizer 和测试构造的多轮对话。
输出
    unittest 测试结果。首次运行会从 Hugging Face 获取 Tokenizer。
"""

import unittest

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Qwen2Config, Qwen2ForCausalLM

from posttrainlab.data.sources import PROJECT_DIR, read_yaml
from posttrainlab.sft_data import IGNORE_INDEX, TURN_END, encode_sft_example


class SFTDataTest(unittest.TestCase):
    """用两轮对话定位回答边界，以及截断、补位后的移位目标。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载固定的 Qwen Tokenizer 和两轮样例。

        输入
            cls 是 unittest 当前测试类。
        输出
            返回 None，设置类级 Tokenizer 和样例供各用例复用。
        """
        model = read_yaml(PROJECT_DIR / "configs/sft.yaml")["model"]
        cls.tokenizer = AutoTokenizer.from_pretrained(model["name_or_path"], revision=model["revision"], token=False)
        cls.messages = [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Say Green."},
            {"role": "assistant", "content": "Green"},
            {"role": "user", "content": "Now say 蓝色."},
            {"role": "assistant", "content": "蓝色"},
        ]

    def test_only_assistant_and_turn_end_are_supervised(self) -> None:
        """核对两轮回答的首 token 和结束符参与训练。

        输入
            设置了 system 的两轮对话，其中用户文本也包含回答中的词。
        输出
            确认有效标签解码只得到两条回答以及对应结束标记。
        """
        features, truncated = encode_sft_example(self.messages, self.tokenizer, 512)
        expected = f"Green{TURN_END}蓝色{TURN_END}"
        supervised = [label for label in features["labels"] if label != IGNORE_INDEX]
        self.assertFalse(truncated)
        self.assertEqual(self.tokenizer.decode(supervised), expected)
        first = next(index for index, label in enumerate(features["labels"]) if label != IGNORE_INDEX)
        prompt_ids = self.tokenizer.apply_chat_template(
            self.messages[:2], tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        self.assertEqual(first, len(prompt_ids))
        self.assertEqual(features["input_ids"][:first], prompt_ids)

    def test_truncation_and_padding_keep_labels_aligned(self) -> None:
        """核对回答中间截断时的标签和动态补位。

        输入
            同一条两轮对话，另构造一条只保留首个回答 token 的截断样例。
        输出
            确认截断样本的首个回答 token 仍保留，补位标签设为 -100。
        """
        full, _ = encode_sft_example(self.messages, self.tokenizer, 512)
        first = next(index for index, label in enumerate(full["labels"]) if label != IGNORE_INDEX)
        short, truncated = encode_sft_example(self.messages, self.tokenizer, first + 1)
        self.assertTrue(truncated)
        self.assertEqual(short["labels"], full["labels"][: first + 1])
        self.assertEqual(short["labels"][-1], short["input_ids"][-1])
        batch = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX)([short, full])
        self.assertTrue(batch["attention_mask"][0, first + 1 :].eq(0).all())
        self.assertTrue(batch["labels"][0, first + 1 :].eq(IGNORE_INDEX).all())

    def test_first_answer_uses_previous_logit(self) -> None:
        """用独立的交叉熵核对首个回答 token 的 causal shift。

        输入
            只保留第一个回答 token 的 Qwen Tokenizer 输出及单层 Qwen2。
        输出
            确认模型损失等于用回答前一位置 logits 计算的交叉熵。
        """
        features, _ = encode_sft_example(self.messages, self.tokenizer, 512)
        first = next(index for index, label in enumerate(features["labels"]) if label != IGNORE_INDEX)
        batch = {key: torch.tensor([values[: first + 1]]) for key, values in features.items()}
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1,
        )).eval()
        with torch.inference_mode():
            output = model(**batch)
        expected = cross_entropy(output.logits[:, first - 1].float(), batch["labels"][:, first])
        torch.testing.assert_close(output.loss, expected)
