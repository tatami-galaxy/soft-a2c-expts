"""Training and evaluation defaults, CLI options, and configuration resolution."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from transformers import HfArgumentParser
from transformers.trainer_utils import SchedulerType
from trl import GRPOConfig


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-4B"
    task: str = "leg_counting"
    task_config: dict | str | None = field(default_factory=dict)
    wandb_project: str = "reasoning-gym"
    wandb_entity: str = "ujan-rl"


@dataclass
class TrainingConfig(GRPOConfig):
    """Our training defaults; all other options inherit TRL's defaults."""

    max_steps: int = 1000
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_generations: int = 8
    max_completion_length: int = 8192
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    learning_rate: float = 1e-5
    lr_scheduler_type: SchedulerType | str = "linear"
    bf16: bool = True
    gradient_checkpointing: bool = True
    loss_type: str = "dapo"
    beta: float = 0.0
    scale_rewards: str = "group"
    mask_truncated_completions: bool = True
    shuffle_dataset: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    logging_steps: float = 1
    log_completions: bool = False
    report_to: None | str | list[str] = "wandb"
    save_steps: float = 100
    save_total_limit: int | None = 2
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    # Explicitly enable thinking even for templates that default to False.
    chat_template_kwargs: dict | str | None = field(default_factory=lambda: {"enable_thinking": True})
    generation_kwargs: dict | str | None = field(default_factory=lambda: {"min_p": 0.0})


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


def parse_config(argv=None):
    """Parse training CLI overrides, validate inputs, and resolve dependent settings."""
    parser = HfArgumentParser((ExperimentConfig, TrainingConfig))
    experiment, args = parser.parse_args_into_dataclasses(args=argv)

    if isinstance(experiment.task_config, str):
        try:
            experiment.task_config = json.loads(experiment.task_config)
        except json.JSONDecodeError as error:
            parser.error(f"Invalid --task_config JSON: {error}")
    if not isinstance(experiment.task_config, dict) or {"seed", "size"} & experiment.task_config.keys():
        parser.error("--task_config must be a JSON object without seed or size; use --seed.")
    if args.max_steps <= 0:
        parser.error("Streaming training requires --max_steps > 0.")

    model_label = experiment.model_name.rstrip("/").split("/")[-1]
    if args.run_name is None:
        args.run_name = f"{model_label}-{experiment.task}-seed{args.seed}"
    # TrainingArguments resolves an omitted output_dir to trainer_output.
    if args.output_dir == "trainer_output":
        args.output_dir = f"/mnt/data/ujan/sac/reasoning_gym/{args.run_name}"

    # JSON CLI options override individual keys while retaining our defaults.
    for name in ("chat_template_kwargs", "generation_kwargs"):
        setattr(args, name, {**parser.get_default(name), **(getattr(args, name) or {})})
    if args.chat_template_kwargs["enable_thinking"] is not True:
        parser.error("Only thinking mode is supported; enable_thinking must be true.")
    args.model_init_kwargs = {
        "dtype": "bfloat16" if args.bf16 else "float32", **(args.model_init_kwargs or {})
    }
    return experiment, args


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
