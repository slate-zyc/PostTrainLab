# 数据候选对比

| 数据源 | 任务与字段 | 数据卡标注 | 首版用途 |
| --- | --- | --- | --- |
| [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 英文多轮指令，`prompt_id`、`messages` | MIT，模型生成对话 | 首选 SFT，抽查用户与助手轮次、空回答和长度 |
| [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 英文偏好对，`chosen`、`rejected` 与两个分数 | MIT，模型评审 | 首选 Reward Model 与 DPO，抽查上文对齐和标签合理性 |
| [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | `helpful-base` 的人类偏好对，以整段对话字符串保存 | MIT，人类偏好 | 偏好数据对照，抽查对话分隔符及共同前缀 |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 多语言带树结构的消息，`parent_id`、`role`、`lang`、`rank` | Apache-2.0，人类贡献与标注 | 多语言候选。转成 SFT 前需重建会话路径 |

选择 UltraChat 和 UltraFeedback 作为首版候选，两份数据来源的官方训练与测试划分独立保留。实际抽样以 `configs/data_sources.yaml` 固定的 revision 为准，结构统计和待复核样本由 `inspect_dataset.py` 生成。

## 首批结构核对

- UltraChat 的首批 120 条中有 119 条通过对话结构与文本检查，1 条包含异常字符。通过的样本含 2–7 个 Assistant 回合，适合核对多轮回答的标签位置。
- UltraFeedback 的首批 120 条中有 119 条通过共同 Prompt 与偏好对检查，1 条的两份回答相同。通过的 119 条中，99 条 chosen 分数更高，20 条分数持平。清洗保留来源的 chosen/rejected 标签和原始分数，便于单独复核持平项。
- HH-RLHF 的 helpful-base 已抽取 120 条，每条含 chosen、rejected 两个对话字符串。转换该数据需要先按对话角色解析，两份首版训练子集使用上表中的优先来源。

以上核对记录用于筛查结构问题。样本文本与逐条复核入口位于 `outputs/data_quality/candidates/`。

## 正文 Token 抽样

四个候选各读取 120 条。首选数据用固定版本的 Qwen2.5-0.5B Tokenizer 试跑清洗。长度统计覆盖格式有效的样本。

| 数据 | 有效样本 | 正文总长 P50 | 正文总长 P95 | 正文 ≤512 | 清洗后保留 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 119 | 991 | 2001.9 | 10 | 101 |
| Preference | 119 | 424 | 953.1 | 80 | 119 |

SFT 的多轮对话通常超过起步配置的 512 token。套用 Chat Template 后，要在截断时保留有效回答标签，并单独观察首个回答 token 的对齐。清洗预览使用 1536 个正文 token 作为上限。
