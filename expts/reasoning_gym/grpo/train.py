"""Fine-tune all model parameters on streamed Reasoning Gym questions with TRL GRPO.

From the repository root (select an available GPU):
    CUDA_VISIBLE_DEVICES=4 uv run python -m expts.reasoning_gym.grpo.train \
        --task leg_counting --max_steps 100

MiniCPM5-2B-SFT uses the same thinking/answer format. Set its sampling parameters:
   CUDA_VISIBLE_DEVICES=6 uv run python -m expts.reasoning_gym.grpo.train \
    --task countdown --run-name MiniCPM5-2B-SFT_countdown_constant \
        --learning-rate 1e-6 --lr_scheduler_type constant \
            --model_name openbmb/MiniCPM5-2B-SFT --per-device-train-batch-size 8 \
                --gradient-accumulation-steps 2 

Both models use top_p=0.95 and min_p=0.0. Qwen sampling remains the default.
Override task defaults with --task_config '{"max_animals": 5}'.
Use --task all for uniform sampling over all registered tasks at default difficulty.
Set PYTHONHASHSEED=0 for reproducibility across processes (some tasks use sets).
Override the W&B account or team with --wandb_entity TEAM.
Override the default MODEL-TASK-seedSEED run name with --run_name NAME.
Standard GRPOConfig arguments are also accepted; see --help.
Tested with TRL 1.16.0 and reasoning-gym 0.1.25.
"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from datasets import IterableDataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer

from ..task_utils import (
    create_task_dataset, generate_examples, make_reward_function,
    resolved_task_config, task_sampling_config, validate_mixture_resume,
)
from .config import parse_config


def main():
    experiment, args = parse_config()
    set_seed(args.seed)

    dataset = create_task_dataset(
        experiment.task, experiment.task_config, seed=args.seed
    )
    config = {
        **asdict(experiment),
        "task_config": resolved_task_config(dataset),
        "task_sampling": task_sampling_config(experiment.task),
        "training": args.to_dict(),
        "versions": {name: version(name) for name in (
            "reasoning-gym", "trl", "transformers", "torch", "datasets"
        )},
    }
    checkpoint = args.resume_from_checkpoint
    if checkpoint is True:
        checkpoint = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
        if checkpoint is None:
            raise ValueError(f"No checkpoint found in {args.output_dir}")
    if checkpoint:
        validate_mixture_resume(checkpoint, config)
    train_dataset = IterableDataset.from_generator(
        generate_examples,
        gen_kwargs={"task": experiment.task, "task_config": experiment.task_config, "seed": args.seed},
    )
    tokenizer = AutoTokenizer.from_pretrained(experiment.model_name, padding_side="left")
    trainer = GRPOTrainer(
        model=experiment.model_name,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=make_reward_function(dataset, tokenizer),
    )

    run = None
    if trainer.accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if "wandb" in args.report_to:
            import wandb

            run = wandb.init(
                project=experiment.wandb_project,
                entity=experiment.wandb_entity,
                name=args.run_name,
                config=config,
                dir=str(output_dir),
            )

    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(Path(args.output_dir) / "final"))
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
