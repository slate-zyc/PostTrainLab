import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

IGNORE_INDEX = -100


def read_sft_fixture(path: Path) -> list[tuple[str, str]]:
    """读取冒烟测试用的单轮 user/assistant 样本。"""
    samples = []
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 格式无效") from error
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"{path}:{line_number}: 需要一组 user/assistant 消息")

            contents = []
            for message, role in zip(messages, ("user", "assistant"), strict=True):
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_number}: {role} 内容不能为空")
                if message.get("role") != role:
                    raise ValueError(f"{path}:{line_number}: 预期 {role} 消息")
                contents.append(content.strip())
            samples.append((contents[0], contents[1]))

    if not samples:
        raise ValueError(f"{path}: 没有可用的 SFT 样本")
    return samples


def build_smoke_batch(
    samples: list[tuple[str, str]],
) -> tuple[dict[str, torch.Tensor], int]:
    """用仅适用于测试样本的小词表构造输入和回答标签。"""
    pad_id, eos_id, user_id, assistant_id = range(4)
    words = sorted({word for sample in samples for text in sample for word in text.split()})
    vocabulary = {word: index for index, word in enumerate(words, start=4)}
    prompts = []
    answers = []
    for prompt, answer in samples:
        prompt_ids = [vocabulary[word] for word in prompt.split()]
        answer_ids = [vocabulary[word] for word in answer.split()]
        prompts.append([user_id, *prompt_ids, assistant_id])
        answers.append([*answer_ids, eos_id])
    lengths = [len(prompt) + len(answer) for prompt, answer in zip(prompts, answers, strict=True)]
    input_ids = torch.full((len(samples), max(lengths)), pad_id, dtype=torch.long)
    expected_labels = torch.full_like(input_ids, IGNORE_INDEX)
    for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
        input_ids[row, :lengths[row]] = torch.tensor(prompt + answer)
        expected_labels[row, len(prompt):lengths[row]] = torch.tensor(answer)

    positions = torch.arange(input_ids.shape[1])
    attention_mask = positions < torch.tensor(lengths).unsqueeze(1)
    completion_mask = positions >= torch.tensor([len(prompt) for prompt in prompts]).unsqueeze(1)
    if attention_mask.all():
        raise ValueError("测试样本应有不同的序列长度，以覆盖 padding 屏蔽")
    labels = input_ids.masked_fill(~(attention_mask & completion_mask), IGNORE_INDEX)
    torch.testing.assert_close(labels, expected_labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask.long(),
        "labels": labels,
    }, len(words) + 4


def run_sft_smoke(fixture: Path, device: torch.device, seed: int, learning_rate: float) -> None:
    torch.manual_seed(seed)
    samples = read_sft_fixture(fixture)
    batch, vocab_size = build_smoke_batch(samples)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    supervised_tokens = batch["labels"].ne(IGNORE_INDEX).sum().item()
    print(f"PASS SFT data and labels | samples={len(samples)} | supervised_tokens={supervised_tokens}")

    model_config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        use_cache=False,
    )
    base_model = Qwen2ForCausalLM(model_config)
    # 恢复 Adapter 时使用完全相同的初始 Base 权重。
    base_weights = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config).to(device)
    parameters = dict(model.named_parameters())
    trainable = {name: param for name, param in parameters.items() if param.requires_grad}
    if not trainable or any(".lora_" not in name for name in trainable):
        raise RuntimeError("预期只有 LoRA 参数可训练")
    before_step = {name: param.detach().clone() for name, param in parameters.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=0.0)

    model.train()
    output = model(**batch)
    # Causal LM 的位置 t 预测 t+1。第一个回答 token 的标签因此对应其前一位 logits。
    shifted_loss = F.cross_entropy(
        output.logits.detach()[:, :-1, :].reshape(-1, vocab_size),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("训练损失不是有限值")
    torch.testing.assert_close(output.loss, shifted_loss)
    output.loss.backward()
    optimizer.step()

    changed = {name for name, param in parameters.items() if not torch.equal(before_step[name], param)}
    if not changed or not changed.issubset(trainable):
        raise RuntimeError("预期至少一项 LoRA 参数更新，并且 Base 参数全部保持不变")
    print(f"PASS SFT loss and update | loss={output.loss.item():.6f} | adapter_tensors_updated={len(changed)}")

    inference_inputs = {name: batch[name] for name in ("input_ids", "attention_mask")}
    model.eval()
    with TemporaryDirectory(prefix="posttrainlab-smoke-") as adapter_dir, torch.inference_mode():
        model.save_pretrained(adapter_dir)
        restored_base = Qwen2ForCausalLM(model_config)
        restored_base.load_state_dict(base_weights)
        restored = PeftModel.from_pretrained(restored_base, adapter_dir).to(device).eval()
        restored_parameters = dict(restored.named_parameters())
        for name, parameter in trainable.items():
            torch.testing.assert_close(restored_parameters[name], parameter, rtol=0, atol=0)
        torch.testing.assert_close(
            model(**inference_inputs).logits,
            restored(**inference_inputs).logits,
        )
    print("PASS LoRA save and reload | logits match")
