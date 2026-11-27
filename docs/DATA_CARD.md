# PostTrainLab 数据卡

数据版本：`v1`。

| 用途 | 数据源 | 固定版本 | 许可证 |
| --- | --- | --- | --- |
| sft | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a) | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT |
| preference | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/tree/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85) | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT |

- UltraChat 提供模型生成的英文多轮对话，按 `messages` 保留训练序列。
- UltraFeedback 的偏好标签来自模型评分。每条记录拆为共同 Prompt、chosen 与 rejected。
- 用流式缓冲抽取工程子集，原始文件的提交、上游拆分、条数和本地路径见数据清单。
- 验证集取自官方训练拆分，测试集来自独立的官方测试拆分。共用 Prompt 索引用于去除抽样子集之间的重叠。
- 长度阈值用于正文 token 筛选。正式训练应用 Chat Template 后再检查实际输入长度。
- sft 训练子集固定从官方分片 `data/train_sft-00000-of-00003-a3ecf92756993583.parquet` 抽取。

## SFT 训练中的输入长度

`configs/sft.yaml` 将应用 Qwen Chat Template 后的整条序列限制到 512 token。每个 Assistant 回答及其仍在序列中的结束标记参与损失。无回答标签的截断样本被跳过。v1 训练集的 5,649 条中有 5,101 条可训练，验证集的 628 条中有 569 条可计算损失。版本清单仍记录清洗后、套模板前的样本条数。
