"""
发布清洗完成的数据版本，生成统计、图表、数据卡和复核队列。

输入
    两类数据的中间 JSONL、来源清单及对应 YAML 配置。
输出
    data/processed/v1/ 的最终数据、data/manifests/ 的文件清单、
    docs/ 的数据卡和 outputs/data_quality/ 的质量报告。
"""

import csv
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from filelock import FileLock
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from posttrainlab.data.sources import read_jsonl, write_json, write_jsonl, write_text
from posttrainlab.data.split import split_datasets

KINDS = ("sft", "preference")
SPLITS = ("train", "validation", "test")
DATASET_VERSION = "v1"


# ============================================================
# 加载两份中间数据，核对清洗配置和记录条数
# ============================================================

def read_interim(root: Path, kind: str, config: dict):
    """加载一种数据类型的标准化结果。

    输入
        root 是项目目录，kind 为 sft 或 preference，config 是当前 YAML 解析结果。
    输出
        返回 (datasets, manifest, rejects)。datasets 按官方 train、test 存放记录，
        manifest 保存配置和上游来源，rejects 是对应的清洗拒收记录。

    核对规则
        读取时使用与写入相同的文件锁，再核对配置和中间文件条数。
    """
    directory = root / "data" / "interim" / kind
    with FileLock(str(directory / "prepare.lock")):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError(f"{kind} 配置已更新，请重新执行数据准备脚本")

        datasets = {}
        rejects = []
        for split in ("train", "test"):
            path = directory / f"{split}.jsonl"
            rows = list(read_jsonl(path))
            if len(rows) != manifest["splits"][split]["rows"]:
                raise ValueError(f"{path} 的条数与中间清单不符")
            datasets[split] = rows
            rejects.extend(read_jsonl(directory / f"{split}_rejected.jsonl"))
    return datasets, manifest, rejects


# ============================================================
# 文本报告与图表
# ============================================================

def length_summary(records: list[dict]) -> dict:
    """计算 Prompt、回答与总正文长度的分位数。

    输入
        records 是含 prompt、answer、total 计数的非空记录列表。
    输出
        返回三项长度各自的 p50、p90、p95 和最大值，单位为 token。
    """
    summary = {}
    for name in ("prompt", "answer", "total"):
        values = [record[name] for record in records]
        p50, p90, p95 = np.percentile(values, [50, 90, 95])  # 用分位数观察样本长度的长尾
        summary[name] = {
            "p50": round(float(p50), 1), "p90": round(float(p90), 1),
            "p95": round(float(p95), 1), "max": max(values),
        }
    return summary


def summarize(outputs: dict, manifests: dict, duplicates: list[dict]) -> dict:
    """汇总两种数据的保留情况、长度和任务分布。

    输入
        outputs 是最终划分，manifests 保存清洗前的条数和长度，
        duplicates 记录去重阶段移除的样本。
    输出
        返回各类数据的原始与保留条数、各集合大小、过滤原因、
        长度分位数以及任务、语言和来源分布。
    """
    summary = {}
    for kind in KINDS:
        upstream = manifests[kind]["splits"]
        rows = [row for split in SPLITS for row in outputs[kind][split]]
        raw_count = sum(stats["raw_rows"] for stats in upstream.values())
        reasons = Counter()
        for stats in upstream.values():
            reasons.update(stats["reasons"])
        reasons.update(item["reason"] for item in duplicates if item["kind"] == kind)
        summary[kind] = {
            "raw": raw_count,
            "schema_valid": sum(stats["schema_valid_rows"] for stats in upstream.values()),
            "kept": len(rows),
            "retention": len(rows) / raw_count,
            "splits": {split: len(outputs[kind][split]) for split in SPLITS},
            "reasons": dict(reasons),
            "categories": dict(Counter(row["metadata"]["task_category"] for row in rows)),
            "languages": dict(Counter(row.get("language", row["metadata"].get("language")) for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "before_lengths": length_summary([values for stats in upstream.values() for values in stats["before_tokens"]]),
            "after_lengths": length_summary([row["metadata"]["tokens"] for row in rows]),
            "content_within_512": sum(row["metadata"]["tokens"]["total"] <= 512 for row in rows),
        }
    return summary


def write_plots(directory: Path, outputs: dict, manifests: dict, summary: dict) -> None:
    """用相同的分箱对照清洗前后长度，并绘制任务类别分布。

    输入
        directory 是已创建的报告目录，outputs 和 manifests 提供两阶段的长度，
        summary 提供最终任务类别计数。
    输出
        返回 None。写入 length_distribution.png 与 category_distribution.png。
    """
    # Agg 可以在 Windows 终端或远程服务器直接出图。
    lengths = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(lengths)
    categories = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(categories)
    for index, kind in enumerate(KINDS, start=1):
        axis = lengths.add_subplot(1, 2, index)
        before = [values["total"] for stats in manifests[kind]["splits"].values() for values in stats["before_tokens"]]
        after = [row["metadata"]["tokens"]["total"] for split in SPLITS for row in outputs[kind][split]]
        upper = max(before)
        bins = [upper * step / 30 for step in range(31)]  # 31 个边界将长度划成 30 个区间，供前后两组共用
        axis.hist(before, bins=bins, alpha=0.45, label="Schema valid")
        axis.hist(after, bins=bins, alpha=0.60, label="Final")
        axis.set(title=kind, xlabel="Content tokens", ylabel="Rows")
        axis.legend()

        bar = categories.add_subplot(1, 2, index)
        values = sorted(summary[kind]["categories"].items(), key=lambda item: item[1])
        bar.barh([item[0] for item in values], [item[1] for item in values])
        bar.set(title=kind, xlabel="Rows")
    lengths.savefig(directory / "length_distribution.png", dpi=160)
    categories.savefig(directory / "category_distribution.png", dpi=160)


def write_reports(directory: Path, version: str, outputs: dict, manifests: dict, duplicates: list[dict]) -> dict:
    """保存数据质量统计、重复匹配结果和人工复核样本。

    输入
        directory 是报告位置，version 是数据版本。
        outputs 是最终样本，manifests 是清洗统计，duplicates 是去重记录。
    输出
        返回用于报告的统计字典，写入 Markdown、CSV、JSON、PNG，
        并从每类训练集中抽取最多 120 条供复核。
    """
    summary = summarize(outputs, manifests, duplicates)
    lines = [
        "# 数据质量报告", "", f"数据版本：`{version}`。", "",
        "| 数据 | 原始条数 | 格式有效 | Train | Validation | Test | 保留率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in KINDS:
        result = summary[kind]
        counts = result["splits"]
        lines.append(
            f"| {kind} | {result['raw']} | {result['schema_valid']} | {counts['train']} | "
            f"{counts['validation']} | {counts['test']} | {result['retention']:.1%} |"
        )

        csv_text = io.StringIO(newline="")  # 先在内存组成 CSV，再统一写到文件
        writer = csv.writer(csv_text)
        writer.writerow(["metric", "value"])
        writer.writerows((name, result[name]) for name in ("raw", "schema_valid", "kept"))
        writer.writerows(counts.items())
        writer.writerows((f"removed_{reason}", count) for reason, count in sorted(result["reasons"].items()))
        for stage in ("before", "after"):
            for part, distribution in result[f"{stage}_lengths"].items():
                writer.writerows((f"{stage}_{part}_tokens_{stat}", value) for stat, value in distribution.items())
        writer.writerow(["content_within_512", result["content_within_512"]])
        write_text(directory / f"{kind}_before_after.csv", csv_text.getvalue())

    lines += [
        "", "## 正文 Token 长度", "",
        "| 数据 | 统计阶段 | Prompt P50 | 回答 P50 | 总长 P50 | 总长 P95 | 总长最大值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in KINDS:
        for phase, label in (("before", "格式有效"), ("after", "最终保留")):
            stats = summary[kind][f"{phase}_lengths"]
            lines.append(
                f"| {kind} | {label} | {stats['prompt']['p50']} | {stats['answer']['p50']} | "
                f"{stats['total']['p50']} | {stats['total']['p95']} | {stats['total']['max']} |"
            )
    lines += [
        "", "## 过滤原因", "",
    ]
    for kind in KINDS:
        reasons = summary[kind]["reasons"]
        detail = "、".join(f"{key} {value} 条" for key, value in sorted(reasons.items()))
        lines.append(f"- {kind}：{detail or '本次没有过滤记录'}。")
    lines += [
        "", "## 清洗与划分", "",
        "- 首条 user 消息作为分组键，先保护官方 Test，再排除精确和近似重复。近似比较只覆盖共享首部或尾部的候选。",
        "- 只对正文计算 Qwen token 长度。SFT 的回答长度包含所有 Assistant 回合，偏好数据以较长的回答计算总长。",
        "- 任务类型采用关键词粗分类。样本来源和标签生成方式见数据卡。",
        "", "## 复核材料", "",
        "- `rejected_samples.jsonl` 记录过滤规则和原始行号，`duplicate_report.json` 记录匹配项。",
        "- `review_samples.jsonl` 为复核队列。请从指令清晰度、回答相关性、事实正确性、完整性与偏好合理性逐条检查。",
    ]
    write_text(directory / "DATA_QUALITY_REPORT.md", "\n".join(lines) + "\n")
    write_json(directory / "summary.json", summary)
    write_json(directory / "duplicate_report.json", {
        "exact": sum(row["reason"] == "exact_prompt_duplicate" for row in duplicates),
        "near": sum(row["reason"] == "near_prompt_duplicate" for row in duplicates),
        "matches": duplicates,
    })

    review = []
    for kind in KINDS:
        pool = outputs[kind]["train"]
        for row in random.Random(42).sample(pool, min(120, len(pool))):  # 固定抽样种子，便于复核同一批样本
            review.append({"kind": kind, "review": None, "sample": row})
    write_jsonl(directory / "review_samples.jsonl", review)
    write_plots(directory, outputs, manifests, summary)
    return summary


def write_data_card(path: Path, version: str, manifests: dict) -> None:
    """记录当前数据版本的来源、许可证和处理规则。

    输入
        path 是数据卡路径，version 是版本号，manifests 保存两个数据源的来源信息。
    输出
        返回 None。写入 Markdown 数据卡，附上固定提交和配置中指定的官方分片。
    """
    lines = ["# PostTrainLab 数据卡", "", f"数据版本：`{version}`。", "",
             "| 用途 | 数据源 | 固定版本 | 许可证 |", "| --- | --- | --- | --- |"]
    for kind in KINDS:
        source = manifests[kind]["splits"]["train"]["provenance"]
        lines.append(
            f"| {kind} | [{source['dataset_id']}]({source['url']}) | `{source['revision']}` | {source['license']} |"
        )
    lines += [
        "", "- UltraChat 提供模型生成的英文多轮对话，按 `messages` 保留训练序列。",
        "- UltraFeedback 的偏好标签来自模型评分。每条记录拆为共同 Prompt、chosen 与 rejected。",
        "- 用流式缓冲抽取工程子集，原始文件的提交、上游拆分、条数和本地路径见数据清单。",
        "- 验证集取自官方训练拆分，测试集来自独立的官方测试拆分。共用 Prompt 索引用于去除抽样子集之间的重叠。",
        "- 长度阈值用于正文 token 筛选。正式训练应用 Chat Template 后再检查实际输入长度。",
    ]
    for kind in KINDS:
        upstream_file = manifests[kind]["splits"]["train"]["provenance"].get("upstream_file")
        if upstream_file:
            lines.append(f"- {kind} 训练子集固定从官方分片 `{upstream_file}` 抽取。")
    write_text(path, "\n".join(lines) + "\n")


# ============================================================
# 生成稳定版本，最后发布清单
# ============================================================

def build_dataset_version(root: Path, configs: dict) -> dict:
    """从中间数据生成并发布 v1 数据。

    输入
        root 是项目目录，configs 同时包含 sft 和 preference 配置。
    输出
        返回数据清单，包含版本、分区文件相对路径、条数和上游来源。
        在目录中写入最终 JSONL、报告及数据卡。

    发布顺序
        先读取中间数据，联合去重并划分，再保存各集合和报告。
        最后写入清单，使训练入口可以按同一份清单定位文件。
    """
    report_dir = root / "outputs" / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(report_dir / "build.lock")):
        inputs = {}
        manifests = {}
        rejects = []
        for kind in KINDS:
            inputs[kind], manifests[kind], filtered = read_interim(root, kind, configs[kind])
            rejects.extend(filtered)
        # 两类数据共用一次去重，测试题优先保留。分区写入使用统一的 v1 目录。
        outputs, duplicates = split_datasets(inputs, configs)
        version = DATASET_VERSION
        data_dir = root / "data" / "processed" / version
        result = {"version": version, "datasets": {}}
        for kind in KINDS:
            splits = {}
            for split in SPLITS:
                rows = outputs[kind][split]
                path = data_dir / kind / f"{split}.jsonl"
                write_jsonl(path, rows)
                splits[split] = {
                    "file": path.relative_to(root).as_posix(), "rows": len(rows),
                }
            result["datasets"][kind] = {
                "splits": splits, "config": configs[kind],
                "raw_sources": {split: stats["provenance"] for split, stats in manifests[kind]["splits"].items()},
            }

        write_jsonl(report_dir / "rejected_samples.jsonl", rejects + duplicates)
        write_reports(report_dir, version, outputs, manifests, duplicates)
        write_data_card(root / "docs" / "DATA_CARD.md", version, manifests)
        write_json(root / "data" / "manifests" / "dataset_v1.json", result)
        return result
