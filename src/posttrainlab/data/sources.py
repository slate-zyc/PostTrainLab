"""
数据来源和文件读写，统一 Hugging Face 抽样、缓存与本地清单。

输入
    数据源 YAML、固定提交、上游拆分和抽样参数，或已有本地文件路径。
输出
    data/raw/ 下的原始 JSONL、来源记录，以及供其他模块使用的文件读写函数。
"""

import json
import re
import tempfile     # 创建和管理临时文件、临时目录
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Iterator, Any    # 类型注解

import yaml
from datasets import load_dataset
from filelock import FileLock       # 文件锁，实现文件级别的进程互斥，避免两个进程同时下载和写入同一个缓存文件
from httpx2 import TransportError   # 捕获网络传输过程中发生的错误


# ============================================================
# 路径和文件读写
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]                   # 项目根目录
SOURCES_CONFIG = PROJECT_DIR / "configs" / "data_sources.yaml"      # 下载源数据相关配置


def read_yaml(path: Path) -> dict:
    """读取一份项目 YAML 配置。

    输入
        path 是 UTF-8 YAML 文件路径。
    输出
        返回解析结果。项目的配置文件最外层约定为字典。
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[Any]:
    """逐行读取 JSONL，每次交出一条原始记录。

    输入
        path 是 UTF-8 JSONL 文件，文件的每行是独立的 JSON 值。
    输出
        生成器按文件顺序产生解析结果。解析失败时在错误中附带文件和行号。
        这里一般返回的是 Iterator[dict]
    """
    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            try:
                yield json.loads(line)      # yield 惰性迭代，逐个返回解析结果，从而避免一次性将所有数据加载到内存中
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 格式无效") from error


def write_text(path: Path, text: str) -> None:
    """在目标目录写入 UTF-8 文本。 主要作为中间函数。

    输入
        path 是目标路径，text 是待保存的完整内容。
    输出
        返回 None。原子写入（Atomic Write）：先写同目录临时文件，再替换目标，进程中断时可避免留下半个目标文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 创建具有唯一文件名的临时文件
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)  # 关闭后保留临时文件，供下一步 replace 使用
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            # 二进制模式的 write() 接收 bytes，而不是 str，所以使用 text.encode("utf-8") 将字符串转换为 UTF-8 编码的字节数据
            temporary.write(text.encode("utf-8"))  # NamedTemporaryFile 默认以二进制模式打开
        temporary_path.replace(path)               # 用临时文件替换目标文件（临时文件被重命名为目标文件）
    finally:    # 清理临时文件
        temporary_path.unlink(missing_ok=True)     # 写入或替换失败：临时文件可能还存在，需要删除；替换成功：文件不存在时不要报错


def write_json(path: Path, value) -> None:
    """将配置、清单或统计结果格式化为 JSON 文件。

    输入
        path 是输出文件路径，value 是可序列化的 Python 对象。
    输出
        返回 None。文件采用两格缩进并直接保存中文。
    """
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    # json.dumps() 返回的是 str，不是文件对象.   eg: '{"name": "\\u5f20\\u4e09", "age": 20}'


def write_jsonl(path: Path, rows) -> None:
    """把若干记录写成每行一个 JSON 对象的文件。

    输入
        path 是目标路径，rows 是可迭代的记录序列。
    输出
        返回 None。文件中每条记录占一行，后续可逐行读取。
    """
    # default=str: 当 json.dumps() 遇到无法直接序列化的 Python 对象时（eg: Path 对象），调用 str() 将其转换为字符串
    text = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    write_text(path, text)


# ============================================================
# 原始数据：Hugging Face 流式抽样 → 固定本地 JSONL
# ============================================================

def raw_sample_path(
    dataset_id: str, split: str, count: int, seed: int, buffer_size: int, source: dict, source_file: str | None,
) -> Path:
    """根据抽样条件确定原始缓存文件名。 中间函数

    用处：不下载整个数据集，只下载里面的部分数据文件。这个函数用于生成一个具有唯一标识的原始数据缓存文件路径。

    输入
        dataset_id 和 split 指定数据集与官方拆分。
        count、seed、buffer_size 指定抽样条件。source 提供本地缓存版本，
        source_file 可指定官方 Parquet 分片。
            Hugging Face官方仓库中可找到，形如：test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet，
            一个 data 数据集下一般有多个 parquet 文件，可选择只下载指定的数据而不是整个数据集。
            UltraChat 的 train_sft 有 3 个 Parquet 分片。项目固定只从 1 个分片流式抽取，目的是减少读取多个分片时的网络抖动。
            UltraChat 的 test_sft 和 UltraFeedback 的 train_prefs、test_prefs 各自也都是 Parquet，但每个拆分只有一个文件，所以不必显示指定
    输出
        返回 data/raw/ 下的 Path。文件名保留本地版本、抽样参数和可选的分片编号。
    """
    tag = dataset_id.replace("/", "--")
    file_tag = ""
    if source_file:
        # re.sub() 用于去除文件名里面的哈希标识。
        # .stem 获取文件名主体，也就是去掉文件扩展名。
        # .removeprefix() 是 Python 字符串的方法，用于删除字符串开头指定的前缀。
        shard = re.sub(r"-[0-9a-f]{16,64}$", "", Path(source_file).stem).removeprefix(f"{split}-")  # 只保留分片编号
        file_tag = f"-{shard}"
    name = f"{source['cache_version']}-{split}-n{count}-s{seed}-b{buffer_size}{file_tag}.jsonl"
    return PROJECT_DIR / "data" / "raw" / tag / name


def download_raw_sample(
    dataset_id: str, split: str, count: int, seed: int, buffer_size: int, source: dict,
    source_file: str | None = None,
) -> Path:
    """从 Hugging Face 流式抽样并保存原始缓存。

    输入
        dataset_id、split、source 和 source_file 用于定位官方数据。
        count、seed 和 buffer_size 分别控制条数、随机性和洗牌缓冲区。
    输出
        返回缓存 JSONL 的 Path。缺失时读取并保存，缓存存在时复用已有文件。

    实现思路
        加锁后检查缓存，避免两个进程同时写入同一份原始样本。
        遇到连接中断或超时，等待后重新读取。
    """
    revision = source["revision"]
    raw_path = raw_sample_path(dataset_id, split, count, seed, buffer_size, source, source_file)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # 同一批原始样本只写一次。后续清洗、审查和复跑读取这一份。
    with FileLock(str(raw_path) + ".lock"):
        if not raw_path.exists():
            print(f"从 Hugging Face 读取 {dataset_id} / {split}，目标 {count} 条", flush=True)
            for attempt in range(3):    # 最多重试三次
                try:
                    if source_file:
                        # 指定同一提交的官方 Parquet 分片，避免同时解析多个分片。
                        # https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/train_sft-00000-of-00003-a3ecf92756993583.parquet
                        # main可替换为固定的 Commit SHA，这样即使数据集后续更新，仍然可以访问指定版本的 Parquet 文件
                        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{source_file}"
                        data = load_dataset("parquet", data_files={split: url}, split=split, streaming=True, token=False)
                    else:
                        data = load_dataset(
                            dataset_id,         # hugging face 仓库名
                            revision=revision,  # 使用配置固定的上游提交
                            data_dir=source.get("data_dir"),    # 若有，指定数据集下载子目录，而不是下载数据集所有子目录
                            split=split,        # 指定需要读取的数据拆分
                            streaming=True,     # 流式读取，返回的是一个 IterableDataset，可以逐条迭代
                            token=False,        # 不使用本地 Hugging Face Token（访问非公开数据集才需要）
                        )
                    # shuffle 从有限缓冲区随机取样，take 一到目标条数就结束迭代。
                    # 流式数据集一般很大，不能像普通数据集一样直接将所有数据打散，所以使用缓冲区随机抽样机制
                    # list() 将迭代器中的结果转换为 Python 列表，后续使用 len()
                    rows = list(data.shuffle(seed=seed, buffer_size=buffer_size).take(count))
                    if len(rows) != count:
                        raise ValueError(f"{split} 实际读取 {len(rows)} 条，预期 {count} 条")
                    write_jsonl(raw_path, rows)
                    break
                except (ConnectionError, TimeoutError, TransportError):
                    if attempt == 2:
                        raise
                    print(f"连接超时或中断，30 秒后重试（第 {attempt + 2} 次）", flush=True)
                    sleep(30)

    return raw_path


def read_raw_sample(
    dataset_id: str, split: str, count: int, seed: int, buffer_size: int, source: dict,
    source_file: str | None = None,
) -> tuple[list[dict], dict]:
    """读取已经下载的原始样本并整理来源信息。

    输入
        与下载时一致的数据集、拆分、固定提交、抽样参数和可选官方分片。
    输出
        返回 (rows, metadata)。metadata 记录官方提交、许可证、本地路径、
        上游拆分、条数、抽样配置和本地缓存写入时间。
    """
    revision = source["revision"]
    raw_path = raw_sample_path(dataset_id, split, count, seed, buffer_size, source, source_file)
    if not raw_path.is_file():
        raise FileNotFoundError(f"缺少 {raw_path}，请先运行 scripts/download_datasets.py")
    rows = list(read_jsonl(raw_path))       # list[dict]
    if len(rows) != count:
        raise ValueError(f"{raw_path} 有 {len(rows)} 条，预期 {count} 条")

    metadata = {
        "dataset_id": dataset_id,
        "url": f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}",
        "revision": revision,
        "license": source["license"],
        "data_dir": source.get("data_dir"),
        "upstream_file": source_file,
        "split": split,
        "rows": count,
        "seed": seed,
        "shuffle_buffer": buffer_size,
        "file": raw_path.relative_to(PROJECT_DIR).as_posix(),  # 清单用项目相对路径，便于在其他机器上复用
        "downloaded_at_utc": datetime.fromtimestamp(raw_path.stat().st_mtime, UTC).isoformat(),     # 文件最后一次修改时间
    }
    return rows, metadata
