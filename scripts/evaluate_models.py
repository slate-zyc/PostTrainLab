"""
在相同的 SFT 测试 Prompt 上比较 Base 和已保存的 LoRA Adapter。

输入
    SFT YAML、一个或多个 Adapter 目录，以及固定的抽样和生成参数。
输出
    独立目录中的 JSONL 回答和对照配置。

运行方式
    python scripts/evaluate_models.py --adapter outputs/models/sft/example/adapter
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from random import Random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrainlab.data.sources import PROJECT_DIR, read_jsonl, read_yaml, write_json, write_jsonl
from posttrainlab.sft_data import TURN_END


# ============================================================
# 固定测试样本与生成参数
# ============================================================

def select_cases(path: Path, count: int, seed: int) -> list[dict]:
    """从 SFT 测试拆分抽取固定的首轮提问和示范回答。

    输入
        path 是独立测试 JSONL，count 是条数，seed 固定抽样顺序。
    输出
        返回含样本 ID、类别、Prompt 消息和参考回答的字典列表。
    """
    rows = Random(seed).sample(list(read_jsonl(path)), count)
    cases = []
    for row in rows:
        messages = row["messages"]
        index = next(index for index, message in enumerate(messages) if message["role"] == "user")
        cases.append({
            "id": row["id"],
            "task_category": row["metadata"]["task_category"],
            "messages": messages[: index + 1],  # 保留 system 与首个 user，省去后续含参考答案的对话
            "reference": messages[index + 1]["content"],
        })
    return cases


@torch.inference_mode()
def generate_answers(model, tokenizer, cases: list[dict], max_new_tokens: int, batch_size: int) -> list[str]:
    """用同一组贪心解码参数生成所有 Prompt 的回答。

    输入
        model 是 Base 或激活了某个 Adapter 的模型，tokenizer 使用同一 Chat Template。
        cases 包含 Prompt，max_new_tokens 和 batch_size 控制生成规模。
    输出
        返回与 cases 顺序对应的文本列表，已移除输入 token 和特殊标记。
    """
    answers = []
    eos_ids = [tokenizer.convert_tokens_to_ids(TURN_END), tokenizer.eos_token_id]
    for start in range(0, len(cases), batch_size):
        prompts = [case["messages"] for case in cases[start : start + batch_size]]
        inputs = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True, padding=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
        # generate 返回输入和新 token，按 batch 补齐后的宽度切出回答。
        answers.extend(tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
    return answers


# ============================================================
# 主函数：固定同一 Base 和 Prompt，依次切换 Adapter
# ============================================================

def main() -> None:
    """读取固定测试集，依次生成 Base 和各 Adapter 的回答。

    输入
        --adapter 接收本次比较的 Adapter 目录。
        --config、--count、--seed 和 --max-new-tokens 指定评测条件。
    输出
        返回 None，保存 answers.jsonl 与 evaluation.json。
    """
    parser = argparse.ArgumentParser(description="比较 Base 和 SFT Adapter 的固定测试回答")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/sft.yaml")
    parser.add_argument("--adapter", type=Path, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = read_yaml(args.config)
    manifest_path = PROJECT_DIR / config["data"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_path = PROJECT_DIR / manifest["datasets"]["sft"]["splits"]["test"]["file"]
    cases = select_cases(test_path, args.count, args.seed)
    model_config = config["model"]
    model_id = model_config["name_or_path"]
    revision = model_config["revision"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=False, trust_remote_code=False)
    tokenizer.padding_side = "left"  # decoder-only 批量生成从末端有效 token 继续预测
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, token=False, trust_remote_code=False,
    ).to(device).eval()

    # 同一份 Base 先生成原始回答，再逐个重载训练产物。
    answers = generate_answers(base, tokenizer, cases, args.max_new_tokens, args.batch_size)
    for case, answer in zip(cases, answers, strict=True):
        case["base"] = answer
        case["adapters"] = []
    print(f"Base 已生成 {len(answers)} 条", flush=True)

    for adapter in args.adapter:
        model = PeftModel.from_pretrained(base, adapter).eval()
        answers = generate_answers(model, tokenizer, cases, args.max_new_tokens, args.batch_size)
        for case, answer in zip(cases, answers, strict=True):
            case["adapters"].append({"path": str(adapter), "answer": answer})
        base = model.unload()  # 释放当前 Adapter，保留同一份未合并的 Base
        print(f"{adapter} 已生成 {len(answers)} 条", flush=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_dir or PROJECT_DIR / "outputs/sft_evaluation" / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "answers.jsonl", cases)
    write_json(output / "evaluation.json", {
        "model": model_config,
        "data_version": manifest["version"],
        "test_file": str(test_path.relative_to(PROJECT_DIR)),
        "adapters": [str(adapter) for adapter in args.adapter],
        "seed": args.seed,
        "count": len(cases),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "batch_size": args.batch_size,
        "device": device,
    })
    print(f"回答对照已保存至 {output}", flush=True)


if __name__ == "__main__":
    main()
