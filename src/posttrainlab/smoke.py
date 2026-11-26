"""
Tiny SFT 冒烟测试，逐步检查训练掩码、LoRA 更新和重载。

输入
    单轮 user/assistant JSONL、随机种子、学习率和运行设备。
输出
    终端验证结果。Adapter 在临时目录保存和重载。

核心流程
    先把 Prompt 和回答拼成同一条 token 序列，再对回答 token 计算损失。
    使用随机初始化的 Tiny Qwen2 跑一次更新，比较训练前后和重载的权重。
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM


# ============================================================
# 常量定义
# ============================================================

# Prompt 给模型提供上下文。它和补位 token 的标签设为 -100，交叉熵会跳过这些位置。
IGNORE_INDEX = -100

# 先给补位、结束和角色标记预留 ID，普通词从后面开始编号。
PAD_TOKEN_ID, EOS_TOKEN_ID, USER_TOKEN_ID, ASSISTANT_TOKEN_ID = range(4)
FIRST_WORD_ID = ASSISTANT_TOKEN_ID + 1


# ============================================================
# 数据加载：JSONL → 单轮问题和回答
# ============================================================

def read_sft_fixture(path: Path) -> list[tuple[str, str]]:
    """逐行读取测试样本，将单轮对话拆成问题和回答。

    输入
        path 是 JSONL 文件路径。每行的 messages 包含一个 user 和一个 assistant 消息。
    输出
        返回 [(问题文本, 回答文本), ...]，正文已清除首尾空白。
        格式或内容有问题时，错误中包含文件路径和行号。
    """
    samples: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 格式无效") from error

            # 每条测试样本固定包含一个 user 消息和一个 assistant 消息。
            user_message, assistant_message = record["messages"]
            roles = (user_message["role"], assistant_message["role"])
            if roles != ("user", "assistant"):
                raise ValueError(f"{path}:{line_number}: 消息应依次为 user、assistant")

            prompt = user_message["content"].strip()
            answer = assistant_message["content"].strip()
            if not prompt or not answer:
                raise ValueError(f"{path}:{line_number}: 问题和回答都需要有内容")
            samples.append((prompt, answer))

    if not samples:
        raise ValueError(f"{path}: 没有可用的 SFT 样本")
    return samples


# ============================================================
# 数据预处理：问题和回答 → token、attention_mask 和 labels
# ============================================================

def build_smoke_batch(
    samples: list[tuple[str, str]],
) -> tuple[dict[str, torch.Tensor], int]:
    """将测试样本转成 Tiny Qwen2 使用的 batch。

    输入
        samples 是 (问题, 回答) 列表。样本需要不同长度，以覆盖 padding 测试。
    输出
        返回 (batch, vocab_size)。batch 的每个张量形状为 [样本数, 补齐后的长度]。
        input_ids 保存 token ID，attention_mask 记录实际内容位置。
        labels 在回答和 EOS 处保留原值，Prompt 与 padding 均设为 -100。

    验证思路
        用切片写出期望 labels，再与根据两个 mask 构造的 labels 逐项对照。
    """
    # --- 步骤 1: 用样本文本构造小词表 ---
    # 这里用空格切词，排序后给词分配稳定的 ID，便于复跑。正式 SFT 使用模型 Tokenizer。
    words: set[str] = set()
    for prompt, answer in samples:
        words.update(prompt.split())
        words.update(answer.split())
    vocabulary = {word: index for index, word in enumerate(sorted(words), start=FIRST_WORD_ID)}

    # --- 步骤 2: 拼接 Prompt 和回答 ---
    # assistant 标记放在 Prompt 末尾，提示接下来开始回答。EOS 是回答结束的学习目标。
    prompts = []
    answers = []
    for prompt, answer in samples:
        prompt_ids = [vocabulary[word] for word in prompt.split()]
        answer_ids = [vocabulary[word] for word in answer.split()]
        prompts.append([USER_TOKEN_ID, *prompt_ids, ASSISTANT_TOKEN_ID])
        answers.append([*answer_ids, EOS_TOKEN_ID])

    # --- 步骤 3: 右侧补齐，并单独记录预期标签 ---
    lengths = [len(prompt) + len(answer) for prompt, answer in zip(prompts, answers, strict=True)]
    input_ids = torch.full((len(samples), max(lengths)), PAD_TOKEN_ID, dtype=torch.long)
    expected_labels = torch.full_like(input_ids, IGNORE_INDEX)
    for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
        input_ids[row, :lengths[row]] = torch.tensor(prompt + answer)
        expected_labels[row, len(prompt):lengths[row]] = torch.tensor(answer)

    # --- 步骤 4: 用两个 mask 生成 labels，并核对回答边界 ---
    # 预期标签按区间切片写入，用来独立检查掩码。
    positions = torch.arange(input_ids.shape[1])
    attention_mask = positions < torch.tensor(lengths).unsqueeze(1)  # 每条样本的有效 token 位置
    completion_mask = positions >= torch.tensor([len(prompt) for prompt in prompts]).unsqueeze(1)  # 从首个回答 token 开始
    if attention_mask.all():
        raise ValueError("测试样本应有不同的序列长度，以覆盖 padding 屏蔽")

    labels = input_ids.masked_fill(~(attention_mask & completion_mask), IGNORE_INDEX)  # 只保留有效的回答和 EOS 标签
    torch.testing.assert_close(labels, expected_labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask.long(),
        "labels": labels,
    }, len(vocabulary) + FIRST_WORD_ID


# ============================================================
# 训练与验证：计算 loss → 更新 LoRA → 保存和重载
# ============================================================

def run_sft_smoke(fixture: Path, device: torch.device, seed: int, learning_rate: float) -> None:
    """运行一次 Tiny Qwen2 的 LoRA 训练并核对重载结果。

    输入
        fixture 是测试样本路径，device 指定 CPU 或 CUDA。
        seed 用于复现随机权重，learning_rate 是 AdamW 的学习率。
    输出
        返回 None。打印数据掩码、损失、更新和重载的检验结果。

    验证思路
        首先手算移位后的交叉熵，确认模型使用了相同的标签。
        更新后比较全部参数，重载时使用训练前保存的 Base 权重。
    """
    # --- 步骤 1: 固定随机种子并准备 batch ---
    torch.manual_seed(seed)
    samples = read_sft_fixture(fixture)
    batch, vocab_size = build_smoke_batch(samples)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    supervised_tokens = batch["labels"].ne(IGNORE_INDEX).sum().item()
    print(f"PASS SFT data and labels | samples={len(samples)} | supervised_tokens={supervised_tokens}")

    # --- 步骤 2: 初始化 Tiny Qwen2 并添加 LoRA ---
    # 只建一层，方便在 CPU 或显存较小的 GPU 上快速验证。
    model_config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,  # Query 头数
        num_key_value_heads=2,  # 每个 KV 头对应两个 Query 头
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        bos_token_id=USER_TOKEN_ID,
        use_cache=False,  # 训练时关闭用于加速生成的 KV Cache
    )
    base_model = Qwen2ForCausalLM(model_config)
    # Adapter 保存的是增量。重载时要配回同一份初始 Base 权重。
    base_weights = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 按因果语言模型加载 LoRA
        r=4,  # 低秩矩阵的中间维度
        lora_alpha=8,  # 普通 LoRA 按 alpha / r 缩放增量，这里是 2
        target_modules=["q_proj", "v_proj"],  # 在注意力的 Query 和 Value 投影上注入 LoRA
    )
    model = get_peft_model(base_model, lora_config).to(device)  # 注入 LoRA，并冻结原模型权重

    # --- 步骤 3: 计算训练 loss，并独立核对 causal shift ---
    parameters = dict(model.named_parameters())
    trainable = {name: param for name, param in parameters.items() if param.requires_grad}
    if not trainable or any(".lora_" not in name for name in trainable):
        raise RuntimeError("预期只有 LoRA 参数可训练")

    # clone() 复制当前数值，便于 optimizer.step() 之后逐项比较。
    before_step = {name: param.detach().clone() for name, param in parameters.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=0.0)  # 关闭权重衰减，专注验证梯度更新
    model.train()
    output = model(**batch)  # 传入 labels 后，模型会同时返回 logits 和 loss

    # 第 t 位的 logits 预测第 t+1 位的 label，因此首个回答 token 用前一位 logits 预测。
    # 模型会自动移位。这里手算一遍交叉熵，核对模型返回的 loss。
    shifted_loss = F.cross_entropy(
        output.logits.detach()[:, :-1, :].reshape(-1, vocab_size),  # 去掉末位 logits，再展平样本和时间维
        batch["labels"][:, 1:].reshape(-1),  # 从第二个 token 开始，与前一位 logits 对齐
        ignore_index=IGNORE_INDEX,  # 跳过标签为 -100 的位置
    )
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("训练损失不是有限值")
    torch.testing.assert_close(output.loss, shifted_loss)

    # --- 步骤 4: 更新一次，检查变化是否局限于 LoRA 参数 ---
    output.loss.backward()  # 反向传播，累积可训练参数的梯度
    optimizer.step()  # 用梯度更新交给优化器的 LoRA 参数
    changed = {name for name, param in parameters.items() if not torch.equal(before_step[name], param)}
    if not changed or not changed.issubset(trainable):
        raise RuntimeError("预期至少一项 LoRA 参数更新，并且 Base 参数全部保持不变")
    print(f"PASS SFT loss and update | loss={output.loss.item():.6f} | adapter_tensors_updated={len(changed)}")

    # --- 步骤 5: 给同一份 Base 加载 Adapter，对照权重和推理输出 ---
    inference_inputs = {name: batch[name] for name in ("input_ids", "attention_mask")}
    model.eval()  # 切换评估模式，例如关闭 Dropout
    # 使用临时目录，测试结束后自动清理 Adapter 文件。
    with TemporaryDirectory(prefix="posttrainlab-smoke-") as adapter_dir, torch.inference_mode():  # 关闭梯度跟踪
        model.save_pretrained(adapter_dir)  # 保存 Adapter 权重和配置
        restored_base = Qwen2ForCausalLM(model_config)
        restored_base.load_state_dict(base_weights)  # 恢复训练前保存的 Base 权重
        restored = PeftModel.from_pretrained(restored_base, adapter_dir).to(device).eval()
        restored_parameters = dict(restored.named_parameters())
        for name, parameter in trainable.items():
            torch.testing.assert_close(restored_parameters[name], parameter, rtol=0, atol=0)  # 参数要求完全相等
        torch.testing.assert_close(
            model(**inference_inputs).logits,
            restored(**inference_inputs).logits,
        )
    print("PASS LoRA save and reload | logits match")
