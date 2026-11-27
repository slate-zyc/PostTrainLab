"""
真实 SFT 训练入口，读取 YAML 并启动 Qwen2.5 LoRA 训练。

输入
    configs/sft.yaml、数据清单，以及可选的样本数、训练步数和运行设备。
输出
    独立运行目录中的 Adapter、Tokenizer、损失日志、曲线和运行记录。

运行方式
    python scripts/train_sft.py --config configs/sft.yaml
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from posttrainlab.data.sources import PROJECT_DIR, read_yaml
from posttrainlab.sft import train_sft


def main() -> None:
    """解析本次训练的配置和运行参数。

    输入
        --config 指定 YAML，--device 选择运行设备。
        --train-samples、--eval-samples 和 --max-steps 可缩小运行规模。
        --train-file 可指定对照实验的训练 JSONL，验证集仍来自配置中的清单。
    输出
        返回 None，训练结果保存在 --output-dir 或带 UTC 时间戳的默认目录。
    """
    parser = argparse.ArgumentParser(description="训练 Qwen2.5 SFT LoRA Adapter")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/sft.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = read_yaml(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_dir or PROJECT_DIR / config["output_dir"] / stamp
    train_sft(config, output_dir.resolve(), device, args.train_samples, args.eval_samples, args.max_steps, args.train_file)


if __name__ == "__main__":
    main()
