"""
数据发布入口，合并 SFT 和 Preference 中间数据，生成划分与报告。

输入
    两份数据 YAML 配置和 data/interim/ 下的清洗结果。
输出
    Train、Validation、Test 文件、数据清单、数据卡和质量报告。

运行方式
    python scripts/analyze_data.py --config configs/data_sft.yaml
"""

import argparse
from pathlib import Path

from posttrainlab.data.report import build_dataset_version
from posttrainlab.data.sources import PROJECT_DIR, read_yaml


# ============================================================
# 主函数：读取两份配置 → 发布数据版本
# ============================================================

def main() -> None:
    """从命令行读取两份数据配置并运行发布流程。

    输入
        --config 指定 SFT 配置，--preference-config 指定偏好配置。
    输出
        返回 None。构建数据版本，在终端打印版本号和各集合条数。
    """
    parser = argparse.ArgumentParser(description="划分并分析 SFT 与 Preference 数据")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "data_sft.yaml")
    parser.add_argument("--preference-config", type=Path, default=PROJECT_DIR / "configs" / "data_preference.yaml")
    args = parser.parse_args()
    # 两份配置一起交给发布流程，便于在不同数据源之间统一检查重复 Prompt。
    configs = {"sft": read_yaml(args.config), "preference": read_yaml(args.preference_config)}
    result = build_dataset_version(PROJECT_DIR, configs)
    print(f"数据版本 {result['version']}")
    for kind, dataset in result["datasets"].items():
        counts = {split: detail["rows"] for split, detail in dataset["splits"].items()}
        print(f"{kind}: {counts}")


if __name__ == "__main__":
    main()
