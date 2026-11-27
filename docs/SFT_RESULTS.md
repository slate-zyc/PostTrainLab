# SFT 实验记录

## 运行条件

- 模型固定为 Qwen2.5-0.5B Base，训练配置见 [configs/sft.yaml](../configs/sft.yaml)。
- 本机使用 RTX 3060 Laptop 6GB，BF16 LoRA。每个小批 2 条、累积 4 次，序列上限 512 token，随机种子 42，每组训练 1 个 epoch。
- 五组使用同一份 v1 验证集，取模板编码后前 128 条带回答标签的样本。训练前 Base 在这组上的验证损失为 1.438390。

## E1 清洗前后与等量对照

从同一份 UltraChat 原始缓存开始，排除与验证、测试 Prompt 重叠的记录后，共有 7,371 条可供抽样。固定选择其中 384 条，从 v1 清洗结果按原始行号取交集，得到 293 条清洗后样本，保留率为 76.3%。套用 512 token 训练上限后，分别保留 336 条和 257 条有回答标签的样本。再从清洗前组固定取 257 条，与清洗后组等量比较。

| 训练组 | 样本 | 监督 token | 更新步数 | 验证损失 | 训练与验证耗时 | CUDA 峰值分配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 清洗后 | 257 | 86,459 | 33 | 1.409741 | 63.2 秒 | 3,930 MiB |
| 清洗前完整组 | 336 | 118,882 | 42 | 1.405187 | 67.1 秒 | 3,930 MiB |
| 清洗前等量组 | 257 | 91,049 | 33 | 1.411172 | 60.4 秒 | 3,930 MiB |

完整清洗前组验证损失最低。按样本数和更新步数配平后，清洗后组比清洗前组低 0.001431。两组仍有 4,590 个监督 token 的差异。以上比较来自固定验证子集和一次随机种子的实验。

运行目录为 `outputs/models/sft/v1-e1-cleaned/`、`outputs/models/sft/v1-e1-uncleaned/`、`outputs/models/sft/v1-e1-uncleaned-matched/`。对照数据和排除统计见 `outputs/sft_ablation/v1-protocol/`。

## E2 数据规模

两组由同一固定随机顺序取样，2K 组包含在 5K 组中。

| 训练组 | 样本 | 监督 token | 更新步数 | 验证损失 | 训练与验证耗时 | CUDA 峰值分配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 2,000 | 670,206 | 250 | 1.378731 | 335.2 秒 | 3,930 MiB |
| 5K | 5,000 | 1,683,619 | 625 | 1.371078 | 891.0 秒 | 3,930 MiB |

5K 的验证损失比 2K 低 0.007653，训练与验证耗时约为 2.66 倍，两组的已分配显存峰值相同。5K 是本轮验证损失最低的清洗后模型，后续 RM 的 SFT 初始化候选为 `outputs/models/sft/v1-e2-clean-5000/adapter/`。当前 v1 在 512 token 下有 5,101 条可训练样本，10K 对照需要扩展数据版本。

两组的运行记录、最佳检查点和损失曲线分别在 `outputs/models/sft/v1-e2-clean-2000/` 和 `outputs/models/sft/v1-e2-clean-5000/`。

## 固定测试题的回答对比

从 v1 的独立 SFT 测试拆分按种子 42 选取 20 条首轮提问。Base 和五份 Adapter 使用同一个 Tokenizer、Chat Template、贪心解码和最多 192 个新增 token，共保存 120 条回答。六套输出均覆盖全部题目。完整回答和参考文本在 `outputs/sft_evaluation/v1-e1-e2/answers.jsonl`，生成设置在同目录的 `evaluation.json`。

| 测试样本 | 核查项 | 实际表现 |
| --- | --- | --- |
| `test_sft:14` | 两位摄影师的角色对话 | Base 先作引导性开场，五份 Adapter 均从摄影师的台词开始。 |
| `test_sft:173` | Canny 阈值 100、保存 PNG、Matplotlib 并排展示 | E1 清洗后、清洗前完整组与 E2 2K 组均生成了这三部分。5K 组用 OpenCV 弹窗展示图像，缺少要求的 Matplotlib 并排图和 PNG 写入。 |
| `test_sft:156` | 根据游记判断是否见到海牛 | 原文写明未见到。Base 转向游览建议，五份 Adapter 都以肯定回答开头，部分回答随后又提到没有看到。 |
| `test_sft:60` | 给任意两个或更多整数计算最大公约数 | 六个回答都把输入固定为两个整数，遗漏多整数的处理要求。 |

5K 组的验证损失最低。定点案例仍可看到事实抽取和多项约束遗漏。每份训练曲线只在第 0 步与 epoch 结束时测验证损失，训练损失按配置中的日志间隔记录。

## 复现设置

1. 按 [README](../README.md) 准备 v1 数据并运行 `prepare_sft_ablation.py --samples 384`。
2. E1 将生成的 `cleaned.jsonl` 或 `uncleaned.jsonl` 传给 `train_sft.py --train-file`。清洗前等量组再传 `--train-samples 257`。
3. E2 通过 `train_sft.py --train-samples 2000` 和 `--train-samples 5000` 取样。五组均使用 `--eval-samples 128 --device cuda` 和独立输出目录。
4. 用 `evaluate_models.py --adapter` 依次传入五份 Adapter，使用 `--count 20 --seed 42 --max-new-tokens 192 --batch-size 20` 生成对照。
