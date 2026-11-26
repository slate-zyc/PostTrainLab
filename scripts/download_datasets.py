"""
数据下载入口，从固定版本的 Hugging Face 官方仓库抽取原始样本。

输入
    候选数据目录和 SFT、Preference 的抽样配置。
输出
    写入 data/raw/ 下的 JSONL 缓存，并打印每份文件的位置。

运行方式
    python scripts/download_datasets.py --mode audit
    python scripts/download_datasets.py --mode train
"""

import argparse

from posttrainlab.data.sources import PROJECT_DIR, SOURCES_CONFIG, download_raw_sample, read_yaml


# ============================================================
# 主函数：按抽查或训练用途下载
# ============================================================

def main() -> None:
    """根据用途选择数据集、上游拆分和抽样规模。

    输入
        --mode audit 下载候选抽查样本，--mode train 下载入选数据的训练和测试子集。
        --dataset-id 可指定本次处理的数据集。
    输出
        返回 None。下载缺失的缓存，逐一打印对应的 JSONL 路径。
    """
    catalog = read_yaml(SOURCES_CONFIG)
    parser = argparse.ArgumentParser(description="下载 Hugging Face 原始数据")
    parser.add_argument("--mode", choices=("audit", "train"), default="audit")
    parser.add_argument("--dataset-id", choices=catalog["sources"])     # choices 默认检查的是字典的键，而不是值
    args = parser.parse_args()
    buffer_size = catalog["shuffle_buffer"]

    # 抽查默认覆盖全部候选。训练模式只下载入选的 SFT、Preference 数据。
    if args.mode == "audit":
        for dataset_id, source in catalog["sources"].items():
            if args.dataset_id and args.dataset_id != dataset_id:
                continue
            path = download_raw_sample(
                dataset_id, source["train_split"], catalog["sample_size"],
                catalog["sample_seed"], buffer_size, source,
            )
            print(path, flush=True)
        return

    # 训练模式从各自的配置读取条数和种子，抽样模式使用候选目录中的统一配置。
    stage_configs = [read_yaml(PROJECT_DIR / "configs" / f"data_{kind}.yaml") for kind in ("sft", "preference")]
    if args.dataset_id and args.dataset_id not in {config["dataset_id"] for config in stage_configs}:
        parser.error("训练模式只使用当前入选的数据集")
    for config in stage_configs:
        dataset_id = config["dataset_id"]
        if args.dataset_id and args.dataset_id != dataset_id:
            continue
        source = catalog["sources"][dataset_id]
        for split in ("train", "test"):
            path = download_raw_sample(
                dataset_id, source[f"{split}_split"], config[f"{split}_rows"],
                config["seed"], buffer_size, source, config.get(f"{split}_file"),
            )
            print(path, flush=True)


if __name__ == "__main__":
    main()
