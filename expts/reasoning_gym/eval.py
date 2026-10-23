"""Evaluate a model with vLLM on a fixed, reproducible Reasoning Gym question set.

From the repository root (select an available GPU):
    CUDA_VISIBLE_DEVICES=4 uv run python -m expts.reasoning_gym.eval \
        --task leg_counting --num_examples 100 --seed 1000000 \
        --task_config '{"min_animals": 5, "max_animals": 5}'

Use --model_name with a Hugging Face model ID or a full-model checkpoint directory.
One response is sampled per question. Use the same arguments and package versions
to regenerate the question set; generation_seed separately controls model sampling.
vLLM batches the prompts automatically. Select GPUs with CUDA_VISIBLE_DEVICES;
use --tensor_parallel_size to split a model across multiple GPUs.
uv run uses the local .venv and puts build tools such as ninja on PATH.
"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS
from vllm import LLM, SamplingParams

from .grpo.config import parse_eval_config
from .task_utils import extract_final_answer, make_prompt


def summarize(results):
    return {
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


def main():
    args = parse_eval_config()

    dataset = reasoning_gym.create_dataset(
        args.task, seed=args.seed, size=args.num_examples, **args.task_config
    )
    entries = list(dataset)
    output_dir = Path(args.output_dir)
    # A fresh directory prevents accidentally mixing results from different runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "eval_set.jsonl").open("w") as file:
        for index, entry in enumerate(entries):
            file.write(json.dumps({"example_id": index, **entry}, ensure_ascii=False) + "\n")
    config = {
        **asdict(args),
        "task_config": asdict(dataset.config),
        "system_prompt": SYSTEM_PROMPTS["default"],
        "versions": {
            name: version(name) for name in ("reasoning-gym", "vllm", "transformers", "torch")
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {len(entries)} questions to {output_dir / 'eval_set.jsonl'}", flush=True)

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        seed=args.generation_seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    # Pass enable_thinking explicitly so models whose templates default to False still think.
    prompts = [
        {"prompt_token_ids": tokenizer.apply_chat_template(
            make_prompt(entry["question"]), tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=True,
        )}
        for entry in entries
    ]
    # Give each question its own sampling seed rather than sharing one RNG stream.
    sampling_params = [
        SamplingParams(
            n=1, max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            min_p=0.0, seed=args.generation_seed + index, skip_special_tokens=False,
        )
        for index in range(len(entries))
    ]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    with (output_dir / "results.jsonl").open("w") as file:
        for index, (entry, output) in enumerate(zip(entries, outputs, strict=True)):
            completion = output.outputs[0]
            answer, thinking_done = extract_final_answer(completion.text)
            score = dataset.score_answer(answer, entry) if answer else 0.0
            result = {
                "example_id": index,
                **entry,
                "response": completion.text,
                "model_answer": answer,
                "score": float(score),
                "response_length": len(completion.token_ids),
                "truncated": completion.finish_reason == "length",
                "finish_reason": completion.finish_reason,
                "thinking_complete": thinking_done,
            }
            results.append(result)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
