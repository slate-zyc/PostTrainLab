"""
偏好数据准备入口，从两条回答分支拆出共同 Prompt。

输入
    configs/data_preference.yaml 指定的 UltraFeedback 原始缓存与 Qwen 分词器。
输出
    data/interim/preference/ 下的 Prompt、chosen、rejected、过滤记录和中间清单。

运行方式
    python scripts/prepare_preference_data.py --config configs/data_preference.yaml
"""

from posttrainlab.data.prepare import prepare_main


if __name__ == "__main__":
    prepare_main("preference")
