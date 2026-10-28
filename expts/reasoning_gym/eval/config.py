"""Evaluation defaults, CLI options, and configuration resolution."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from transformers import HfArgumentParser


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-4B"
    task: str = "leg_counting"
    task_config: dict | str | None = field(default_factory=dict)
    seed: int = 1_000_000
    num_examples: int = 500
    generation_seed: int = 42
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int | None = None
    max_num_seqs: int = 64
    enforce_eager: bool = False
    output_dir: str | None = None


def parse_eval_config(argv=None):
    """Parse evaluation CLI overrides, validate inputs, and resolve the output path."""
    parser = HfArgumentParser(EvalConfig)
    (args,) = parser.parse_args_into_dataclasses(args=argv)
    if isinstance(args.task_config, str):
        try:
            args.task_config = json.loads(args.task_config)
        except json.JSONDecodeError as error:
            parser.error(f"Invalid --task_config JSON: {error}")
    if not isinstance(args.task_config, dict) or {"seed", "size"} & args.task_config.keys():
        parser.error("--task_config must be a JSON object without seed or size.")
    if min(args.num_examples, args.max_new_tokens, args.tensor_parallel_size, args.max_num_seqs) <= 0:
        parser.error("num_examples, max_new_tokens, tensor_parallel_size, and max_num_seqs must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1].")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("Sampling requires temperature > 0, 0 < top_p <= 1, and top_k >= 0.")

    model_label = args.model_name.rstrip("/").split("/")[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    args.output_dir = str(Path(args.output_dir or (
        f"results/reasoning_gym/eval/{model_label}-{args.task}-seed{args.seed}-{timestamp}"
    )))
    return args
