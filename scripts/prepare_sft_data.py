"""
SFT 数据准备入口，保留多轮对话的 messages 结构。

输入
    configs/data_sft.yaml 指定的 UltraChat 原始缓存与 Qwen 分词器。
输出
    data/interim/sft/ 下的标准化样本、过滤记录和中间清单。

运行方式
    python scripts/prepare_sft_data.py --config configs/data_sft.yaml
"""

from posttrainlab.data.prepare import prepare_main


if __name__ == "__main__":
    prepare_main("sft")
