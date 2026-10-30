"""Compose run, dataset, and method options without accepting unrelated flags."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from transformers import HfArgumentParser


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dataset: str = "reasoning_gym"
    wandb_project: str | None = None
    wandb_entity: str = "ujan-rl"


def dataset_module(name):
    if name == "reasoning_gym":
        from .data import reasoning_gym
        return reasoning_gym
    if name == "deepscaler":
        from .data import deepscaler
        return deepscaler
    if name == "deepmath":
        from .data import deepmath
        return deepmath
    raise ValueError(f"Unknown dataset: {name}")


def parse_options(config_class, argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--dataset", choices=("reasoning_gym", "deepscaler", "deepmath"), default="reasoning_gym")
    selected, _ = selector.parse_known_args(argv)
    module = dataset_module(selected.dataset)
    parser = HfArgumentParser((RunConfig, module.DataConfig, config_class), allow_abbrev=False)
    run, data, args = parser.parse_args_into_dataclasses(args=argv)
    module.validate_config(data, parser)
    return run, data, args, parser


def parse_training_config(argv=None, config_class=None, method="grpo"):
    if config_class is None:
        from .grpo.config import TrainingConfig
        config_class = TrainingConfig
    run, data, args, parser = parse_options(config_class, argv)
    if run.wandb_project is None:
        run.wandb_project = run.dataset.replace("_", "-")
    if args.max_steps <= 0:
        parser.error("Training requires --max_steps > 0.")
    model_label = Path(run.model_name).name
    label = data.task if run.dataset == "reasoning_gym" else run.dataset
    if args.run_name is None:
        args.run_name = f"{method}-{model_label}-{label}-seed{args.seed}"
    if args.output_dir == "trainer_output":
        args.output_dir = f"/mnt/data/ujan/sac/{run.dataset}/{args.run_name}"
    # A finite math dataset should reshuffle every epoch; the RG stream orders itself.
    if args.shuffle_dataset is None:
        args.shuffle_dataset = run.dataset in {"deepscaler", "deepmath"}
    for name in ("chat_template_kwargs", "generation_kwargs"):
        setattr(args, name, {**parser.get_default(name), **(getattr(args, name) or {})})
    if args.chat_template_kwargs["enable_thinking"] is not True:
        parser.error("Only thinking mode is supported; enable_thinking must be true.")
    args.model_init_kwargs = {
        "dtype": "bfloat16" if args.bf16 else "float32", **(args.model_init_kwargs or {})
    }
    return run, data, args


def run_metadata(model_name):
    """Find the run config beside a local final/ or checkpoint-N/ directory."""
    path = Path(model_name).parent / "run_config.json"
    return json.loads(path.read_text()) if path.is_file() else None
