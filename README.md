# PostTrainLab

PostTrainLab 是一个面向通用指令遵循和回答质量的小模型后训练实验项目。默认模型为 Qwen2.5-0.5B，首版覆盖数据处理、SFT、Reward Model、DPO、PPO 和统一评测。

## 项目进度

已完成 v1 数据准备、离线 Tiny SFT 验证和 Qwen2.5-0.5B LoRA 训练。本机完成 E1 清洗对照和 E2 的 2K／5K 数据规模对照。

## 环境与运行

在项目根目录使用 Python 3.11 独立环境。锁定依赖已在 Windows x86_64 上验证。

```powershell
conda create -n posttrainlab python=3.11 pip -y
conda activate posttrainlab
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130"
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

```powershell
python scripts/smoke_test.py
python scripts/smoke_test.py --device cpu
python scripts/smoke_test.py --device cuda
```

默认自动选择可用设备。通过时输出数据与标签检查、loss 和参数更新检查、Adapter 重载检查。`--config` 可指定其他 YAML 文件。配置中的 `fixture` 相对于该配置文件所在目录解析。 自定义样本使用单轮 user/assistant 对话，并至少包含两种序列长度，以覆盖 padding。

## 数据准备

当前数据版本为 `v1`。数据来源、固定提交和处理范围见[数据卡](docs/DATA_CARD.md)。

按 `configs/data_sources.yaml` 中固定的 Hugging Face revision 抽查候选数据。原始样本缓存在 `data/raw/`，文件名使用本地版本、拆分和抽样参数，例如 `v1-train_sft-n120-s42-b512.jsonl`。更换上游提交或指定分片时，递增对应数据源的 `cache_version`。

```powershell
python scripts/download_datasets.py --mode audit
python scripts/inspect_dataset.py
```

选定首版来源后，下载训练子集和官方测试子集，再执行格式转换、清洗、去重和划分。两个准备入口读取本地数据，使用 Qwen Tokenizer 统计正文长度。

```powershell
python scripts/download_datasets.py --mode train
python scripts/prepare_sft_data.py --config configs/data_sft.yaml
python scripts/prepare_preference_data.py --config configs/data_preference.yaml
python scripts/analyze_data.py --config configs/data_sft.yaml
```

清单保存在 `data/manifests/dataset_v1.json`，记录每份 JSONL 的路径、条数和上游来源。报告和待复核样本保存在 `outputs/data_quality/`。数据阶段的离线单元测试可用下面的命令运行。

```powershell
python -m unittest discover -s tests -p test_data_pipeline.py
```

## SFT 训练与评测

从固定版本的 Qwen2.5-0.5B Base 开始训练。训练入口从数据清单读取 v1 的 SFT 训练集和验证集，应用 Qwen Chat Template。每轮 Assistant 正文及结束符参与损失。训练默认右截断到 512 token。

```powershell
python -m unittest discover -s tests -v
python scripts/train_sft.py --config configs/sft.yaml --device cuda --train-samples 16 --eval-samples 8 --max-steps 2 --output-dir outputs/models/sft/local-check
python scripts/evaluate_models.py --config configs/sft.yaml --adapter outputs/models/sft/local-check/adapter --output-dir outputs/sft_evaluation/local-check
```

首次运行从 Hugging Face 获取 Base 和 Tokenizer。省略样本与步数上限时，训练使用全部保留样本和配置指定的 epoch。每次训练指定新的 `--output-dir`，或省略该参数以使用默认时间戳目录。

每次训练保存 `adapter/`、`run.json`、`metrics.json` 和 `loss.png`。评测入口从独立测试集固定抽取 20 条首轮 Prompt，保存 Base、各 Adapter 和参考回答到 `answers.jsonl`。在 `--adapter` 后传入多个 Adapter 路径即可复用同一组 Prompt。

### SFT 对照实验

本机实测指标、输出目录和案例对照见 [SFT 实验记录](docs/SFT_RESULTS.md)。

- **E1 清洗前后。** 运行 `python scripts/prepare_sft_ablation.py --samples 384 --output-dir outputs/sft_ablation/v1-protocol`。在同一批原始行中生成 `uncleaned.jsonl` 和 `cleaned.jsonl`。用相同配置分别通过 `--train-file` 训练，再给较大组传入 `--train-samples` 以匹配较小组实际进入训练器的条数。
- **E2 数据规模。** 使用默认清洗后训练集，分别传入 `--train-samples 2000` 和 `--train-samples 5000`。两次运行使用同一份 SFT 配置和固定验证集。小规模组是大规模组可训练样本的前缀。

## 实验主线

1. 筛选、清洗并划分公开指令和偏好数据。
2. 完成 SFT，在训练好的 SFT 骨干上初始化独立的 Reward Model。
3. 从同一 SFT checkpoint 完成 DPO 和 PPO，保存并验证模型。
4. 在统一评测集上比较 Base、SFT、DPO 和真实 PPO，记录消融实验、人工盲测和失败案例。

算法原理、模型权重关系、设计理由和验证记录见[学习笔记](学习笔记.md)。
