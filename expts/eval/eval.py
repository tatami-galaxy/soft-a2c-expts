r"""Evaluate Reasoning Gym questions or a DeepScaleR/DeepMath holdout subset.

Run from the repository root, selecting an available GPU.

Reasoning Gym: evaluate 100 questions at a fixed task difficulty:
    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset reasoning_gym --model_name Qwen/Qwen3-1.7B \
        --task leg_counting --task_config '{"min_animals":5,"max_animals":5}' \
        --num_examples 100 --dataset_seed 1000000

Reasoning Gym: evaluate 10 questions PER TASK across all registered tasks:
    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset reasoning_gym --model_name /path/to/run/final \
        --task all --num_examples 10

DeepScaleR: evaluate 100 held-out problems using the checkpoint's saved split:
    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset deepscaler --model_name /path/to/run/final --num_examples 100

DeepScaleR: evaluate a base model on the same holdout subset:
    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset deepscaler --model_name Qwen/Qwen3-1.7B \
        --split_manifest /path/to/run/split_manifest.json --num_examples 100

DeepMath: evaluate a checkpoint or a base model on its saved holdout subset:
    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset deepmath --model_name /path/to/deepmath-run/final --num_examples 100

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
        --dataset deepmath --model_name Qwen/Qwen3-1.7B \
        --split_manifest /path/to/deepmath-run/split_manifest.json --num_examples 100

For both math datasets, num_examples selects the first N problems in the fixed holdout
order; it does not change the training split. Local final/ and checkpoint-N/
directories automatically resolve the training run's split manifest.

One response is sampled per question. Match generation settings across models;
MiniCPM5-2B-SFT uses --temperature 1.0 --top_k 0. Use --generation_seed to control
sampling, --max_new_tokens for the response budget, and --output_dir for a fresh
results directory. On a shared GPU, lower --gpu_memory_utilization from its
default of 0.8 to fit the available memory.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from statistics import mean

from ..config import dataset_module
from .config import parse_eval_config


def summarize(results, by_task=False):
    summary = {
        "num_examples": len(results),
        "num_full_credit": sum(row["score"] == 1.0 for row in results),
        "full_credit_rate": mean(row["score"] == 1.0 for row in results),
        "mean_reward": mean(row["score"] for row in results),
        "answer_extraction_rate": mean(bool(row["model_answer"]) for row in results),
        "mean_response_length": mean(row["response_length"] for row in results),
        "min_response_length": min(row["response_length"] for row in results),
        "max_response_length": max(row["response_length"] for row in results),
        "truncation_rate": mean(row["truncated"] for row in results),
        "thinking_complete_rate": mean(row["thinking_complete"] for row in results),
    }
    if "verifier_error" in results[0]:
        summary["verifier_failure_rate"] = mean(row["verifier_error"] is not None for row in results)
    if by_task:
        groups = defaultdict(list)
        for row in results:
            groups[row["metadata"]["source_dataset"]].append(row)
        summary["per_task"] = {name: summarize(rows) for name, rows in sorted(groups.items())}
        summary["macro_full_credit_rate"] = mean(row["full_credit_rate"] for row in summary["per_task"].values())
        summary["macro_mean_reward"] = mean(row["mean_reward"] for row in summary["per_task"].values())
    return summary


def main():
    run, data_config, args = parse_eval_config()
    module = dataset_module(run.dataset)
    entries, scorer, metadata = module.load_eval_data(data_config, args.num_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if run.dataset in {"deepscaler", "deepmath"}:
        copyfile(data_config.split_manifest, output_dir / "split_manifest.json")
    with (output_dir / "eval_set.jsonl").open("w") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    names = ["vllm", "transformers", "torch", "datasets"]
    names += ["reasoning-gym"] if run.dataset == "reasoning_gym" else ["math-verify", "latex2sympy2_extended", "sympy"]
    config = {
        "model_name": run.model_name, "dataset": run.dataset, "data": metadata,
        **asdict(args), "versions": {name: version(name) for name in names},
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {len(entries)} questions to {output_dir / 'eval_set.jsonl'}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=run.model_name, dtype="bfloat16", seed=args.generation_seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager, generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": tokenizer.apply_chat_template(
        module.make_prompt(entry["question"]), tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=True,
    )} for entry in entries]
    # Original row IDs keep each problem's sampling seed stable across subset sizes.
    sampling_params = [SamplingParams(
        n=1, max_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, min_p=0.0,
        seed=args.generation_seed + entry["example_id"], skip_special_tokens=False,
    ) for entry in entries]
    outputs = llm.generate(prompts, sampling_params)
    results = []
    with (output_dir / "results.jsonl").open("w") as file:
        for entry, output in zip(entries, outputs, strict=True):
            completion = output.outputs[0]
            result = {
                **entry, "response": completion.text,
                **module.score_response(scorer, completion.text, entry),
                "response_length": len(completion.token_ids),
                "truncated": completion.finish_reason == "length",
                "finish_reason": completion.finish_reason,
            }
            results.append(result)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize(results, by_task=run.dataset == "reasoning_gym" and data_config.task == "all")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
