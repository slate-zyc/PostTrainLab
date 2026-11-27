"""
在正式 SFT 数据上训练 Qwen2.5 LoRA，保存可重载的 Adapter 和训练记录。

输入
    SFT YAML 配置、数据版本清单、目标运行目录和可选短跑参数。
输出
    独立目录内的 Adapter、Tokenizer、最佳验证检查点、运行信息和损失曲线。

核心流程
    从清单取数据并编码 → 加载 Base 并注入 LoRA → 用 Trainer 训练和验证 → 保存最佳 Adapter。
"""

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from posttrainlab.data.sources import PROJECT_DIR, write_json
from posttrainlab.sft_data import IGNORE_INDEX, load_sft_dataset

matplotlib.use("Agg")  # 支持在无图形界面的机器上保存曲线
from matplotlib import pyplot as plt


# ============================================================
# SFT 主流程：准备数据 → 加载 Base → 注入 LoRA → 训练与保存
# ============================================================

def train_sft(
    config: dict,
    output_dir: Path,
    device: str,
    train_samples: int | None,
    eval_samples: int | None,
    max_steps: int,
    train_file: Path | None = None,
) -> None:
    """从预训练 Base 出发执行一次正式的 Completion-only LoRA SFT。

    输入
        config 是 SFT YAML，output_dir 是独立运行目录，device 是 cpu 或 cuda。
        train_samples、eval_samples 限制使用条数，max_steps 为 -1 时按 epoch 训练。
        train_file 可替换训练集，验证集固定从数据清单读取。
    输出
        返回 None。保存最佳验证损失对应的 Adapter、Tokenizer 和本次运行记录。
    """
    # --- 步骤 1: 读取配置，构造训练和验证数据 ---
    git_info = git_snapshot()
    model_config = config["model"]
    training = config["training"]
    data_config = config["data"]
    model_id = model_config["name_or_path"]
    revision = model_config["revision"]
    seed = training["seed"]
    set_seed(seed)  # 固定 LoRA 初始化和训练采样的随机状态

    manifest_path = PROJECT_DIR / data_config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_info = manifest["datasets"]["sft"]
    splits = dataset_info["splits"]
    train_path = train_file or PROJECT_DIR / splits["train"]["file"]  # 对照实验可以替换训练文件
    validation_path = PROJECT_DIR / splits["validation"]["file"]  # 各组共用清单里的验证集
    max_length = data_config["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,  # Tokenizer 和 Base 权重使用同一提交
        token=False,
        trust_remote_code=False,
    )
    tokenizer.truncation_side = "right"  # 保留序列开头，从右侧截去过长部分
    tokenizer.padding_side = "right"  # 补位放在有效序列末尾
    train_data, train_stats = load_sft_dataset(train_path, tokenizer, max_length, train_samples, seed)
    eval_data, eval_stats = load_sft_dataset(validation_path, tokenizer, max_length, eval_samples)
    print(f"训练样本 {len(train_data)} 条，验证样本 {len(eval_data)} 条", flush=True)

    # --- 步骤 2: 按设备选择精度，加载 Base ---
    use_cuda = device == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.float32
    if use_cuda:
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        torch.cuda.reset_peak_memory_stats()  # 从加载本轮模型前开始统计 CUDA 峰值

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,  # 复跑和重载 Adapter 时使用同一版本的 Base
        dtype=dtype,
        token=False,
        trust_remote_code=False,
    )
    base.config.use_cache = False  # 训练按整段序列计算 logits，关闭用于逐 token 生成的 KV Cache

    # --- 步骤 3: 注入 LoRA，统计可训练参数 ---
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 按自回归语言模型配置 PEFT
        revision=revision,  # 将 Base 提交写入 Adapter 配置
        **config["lora"],  # rank、缩放、Dropout 和目标层统一从 YAML 读取
    )
    model = get_peft_model(base, lora)  # 冻结 Base 权重，在目标层添加可训练的低秩矩阵
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()

    # --- 步骤 4: 配置批次补位、Trainer 和检查点 ---
    output_dir.mkdir(parents=True, exist_ok=False)  # 每次运行使用独立目录
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,  # 按当前批次的最长序列动态补位
        label_pad_token_id=IGNORE_INDEX,  # 补位标签为 -100，交叉熵忽略这些位置
        pad_to_multiple_of=8 if use_cuda else None,  # CUDA 批次的序列长度再对齐到 8 的倍数
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        **training,  # 学习率、训练轮数、batch 大小和梯度累积取自 YAML
        max_steps=max_steps,  # 正数用于限制优化器更新步数

        # 精度与数据加载
        use_cpu=not use_cuda,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,  # CUDA 设备不支持 BF16 时使用 FP16
        activation_checkpointing_kwargs={"use_reentrant": False},  # 开启激活重计算时使用非重入模式
        dataloader_num_workers=0,  # 在主进程加载批次
        dataloader_pin_memory=use_cuda,  # CUDA 训练使用锁页内存传输批次
        optim="adamw_torch",

        # 验证与检查点
        eval_strategy="epoch",
        eval_on_start=True,  # 更新前先测一次验证集，记录 Base 的起点
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,  # 训练结束时载入验证损失最低的已保存检查点
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # 验证损失越低越好
        prediction_loss_only=True,  # 验证只收集损失，省去跨批次保留大词表 logits
        label_names=["labels"],  # 告知 Trainer 哪一列是模型的监督目标

        # 训练日志
        logging_first_step=True,
        include_num_input_tokens_seen="non_padding",  # token 计数排除批次补位
        report_to="none",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,  # 保存 Adapter 时一并保存这份 Tokenizer
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    # --- 步骤 5: 记录本次运行的配置和数据来源 ---
    run = {
        "run_id": output_dir.name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "max_steps": max_steps,
        "git": git_info,
        "data_version": manifest["version"],
        "data_sources": dataset_info["raw_sources"],
        "data_preprocessing": dataset_info["config"],
        "adapter_dir": str(output_dir / "adapter"),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if use_cuda else None,
        "train": train_stats,
        "validation": eval_stats,
        "device": torch.cuda.get_device_name() if use_cuda else "cpu",
        "model_dtype": str(dtype),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets")},
        "cuda_build": torch.version.cuda,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    write_json(output_dir / "run.json", run)  # 先保存配置，便于核查中断的运行

    # --- 步骤 6: 执行训练和验证 ---
    started = perf_counter()
    result = trainer.train()  # 执行梯度累积、参数更新及验证，最后加载最佳检查点
    elapsed = perf_counter() - started

    # --- 步骤 7: 保存最佳 Adapter、训练状态和损失曲线 ---
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))  # 保存 LoRA 权重、Adapter 配置和 Tokenizer
    trainer.save_state()  # 保存 Trainer 的更新步数、最佳检查点和日志状态

    history = trainer.state.log_history
    metrics = {
        "training": result.metrics,
        "steps": trainer.state.global_step,
        "training_seconds": elapsed,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if use_cuda else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "history": history,
    }
    write_json(output_dir / "metrics.json", metrics)
    plot_loss(history, output_dir / "loss.png")
    print(f"SFT Adapter 已保存至 {adapter_dir}", flush=True)


# ============================================================
# 辅助函数：绘制损失曲线和记录代码版本
# ============================================================

def plot_loss(history: list[dict], path: Path) -> None:
    """将训练日志中的训练和验证损失绘制到同一份图中。

    输入
        history 是 Trainer 的 log_history，每项包含对应的更新步数。
        path 是 PNG 图片路径。
    输出
        返回 None，生成训练损失和验证损失曲线。
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("loss", "Train"), ("eval_loss", "Validation")):
        rows = [row for row in history if key in row]
        axis.plot([row["step"] for row in rows], [row[key] for row in rows], marker=".", label=label)
    axis.set(xlabel="Optimizer step", ylabel="Completion token loss")
    axis.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def git_snapshot() -> dict:
    """读取代码来源及工作区状态，用于复现实验。

    输入
        使用当前项目根路径和 PATH 中的 Git。
    输出
        返回 commit 和 dirty。无法获取对应状态时记录 None。
    """
    executable = shutil.which("git")
    if executable is None:
        return {"commit": None, "dirty": None}
    command = [executable, "-C", str(PROJECT_DIR)]
    commit = subprocess.run([*command, "rev-parse", "HEAD"], capture_output=True, encoding="utf-8")
    changes = subprocess.run(
        [*command, "--no-optional-locks", "status", "--porcelain"],
        capture_output=True, encoding="utf-8",
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(changes.stdout.strip()) if changes.returncode == 0 else None,
    }
