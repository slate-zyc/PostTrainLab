# PostTrainLab

PostTrainLab 是一个面向通用指令遵循和回答质量的小模型后训练实验项目。默认模型为 Qwen2.5-0.5B，首版覆盖数据处理、SFT、Reward Model、DPO、PPO 和统一评测。

## 项目进度

第一阶段提供本地 Tiny SFT Smoke Test。它使用内置 JSONL 和随机初始化的 Tiny Qwen2 验证一次 LoRA 更新。可离线运行。正式数据处理、训练和评测将逐阶段实现。

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

## 实验主线

1. 筛选、清洗并划分公开指令和偏好数据。
2. 完成 SFT，在训练好的 SFT 骨干上初始化独立的 Reward Model。
3. 从同一 SFT checkpoint 完成 DPO 和 PPO，保存并验证模型。
4. 在统一评测集上比较 Base、SFT、DPO 和真实 PPO，记录消融实验、人工盲测和失败案例。

算法原理、模型权重关系、设计理由和验证记录见[学习笔记](学习笔记.md)。
