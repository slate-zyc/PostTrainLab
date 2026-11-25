import argparse
import math
import platform
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from posttrainlab.smoke import run_sft_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 PostTrainLab 的 Tiny SFT 链路")
    default_config = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("配置文件须包含 YAML 映射")
    fixture_setting = config.get("fixture")
    if not isinstance(fixture_setting, str) or not fixture_setting.strip():
        parser.error("配置须包含非空的 fixture 路径")
    seed = config.get("seed")
    learning_rate = config.get("learning_rate")
    if type(seed) is not int or not 0 <= seed < 2**64:
        parser.error("seed 须为 0 至 2**64 - 1 的整数")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        parser.error("learning_rate 须为有限的正数")
    fixture = config_path.parent / fixture_setting

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch 未检测到可用的 CUDA 设备")

    dependencies = ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "PyYAML")
    print(f"Python {platform.python_version()} | device={device_name} | seed={seed} | lr={learning_rate}")
    print(" | ".join(f"{name}={version(name)}" for name in dependencies))
    if device_name == "cuda":
        print(f"GPU={torch.cuda.get_device_name()} | CUDA build={torch.version.cuda}")
    run_sft_smoke(fixture, torch.device(device_name), seed, learning_rate)
    print("PASS PostTrainLab Tiny SFT smoke test")


if __name__ == "__main__":
    main()
