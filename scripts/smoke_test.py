"""
Tiny SFT 冒烟测试入口，用内置样本验证一次 LoRA 更新。

输入
    configs/smoke.yaml、配置引用的单轮对话 JSONL 和运行设备。
输出
    终端打印环境、回答掩码、权重更新和 Adapter 重载结果。

运行方式
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cpu
"""

import argparse
import platform
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from posttrainlab.smoke import run_sft_smoke


# ============================================================
# 主函数：加载配置 → 选择设备 → 运行测试
# ============================================================

def main() -> None:
    """解析运行配置，选择设备并调用 Tiny SFT 验证。

    输入
        --config 指定测试 YAML，--device 可选 auto、cpu 或 cuda。
    输出
        返回 None。测试通过时打印通过信息，测试失败时由实际检查报错。
    """
    # --- 步骤 1: 获取本次运行的配置 ---
    default_config = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"  # parents[1] 是项目根目录
    parser = argparse.ArgumentParser(description="验证 PostTrainLab 的 Tiny SFT 链路")
    parser.add_argument("--config", type=Path, default=default_config, help="测试使用的 YAML 配置")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="运行设备")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))  # 把 YAML 文本解析成 Python 字典
    seed = config["seed"]
    learning_rate = config["learning_rate"]
    # 样本路径以 YAML 所在目录为基准，从其他目录启动也能找到同一份文件。
    fixture = config_path.parent / config["fixture"]

    # --- 步骤 2: 确定设备并记录版本 ---
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    dependencies = ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "PyYAML")
    print(f"Python {platform.python_version()} | device={device_name} | seed={seed} | lr={learning_rate}")
    print(" | ".join(f"{name}={version(name)}" for name in dependencies))
    if device_name == "cuda":
        print(f"GPU={torch.cuda.get_device_name()} | CUDA build={torch.version.cuda}")

    # --- 步骤 3: 执行一次 Tiny SFT 验证 ---
    run_sft_smoke(fixture, torch.device(device_name), seed, learning_rate)
    print("PASS PostTrainLab Tiny SFT smoke test")


if __name__ == "__main__":
    main()
