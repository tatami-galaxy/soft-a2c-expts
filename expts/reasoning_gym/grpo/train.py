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
Override the W&B account or team with --wandb_entity TEAM.
Override the default MODEL-TASK-seedSEED run name with --run_name NAME.
Standard GRPOConfig arguments are also accepted; see --help.
Tested with TRL 1.16.0 and reasoning-gym 0.1.25.
"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import reasoning_gym
from datasets import IterableDataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOTrainer

from ..task_utils import generate_examples, make_reward_function
from .config import parse_config


def main():
    experiment, args = parse_config()
    set_seed(args.seed)

    dataset = reasoning_gym.create_dataset(
        experiment.task, seed=args.seed, size=1, **experiment.task_config
    )
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
        config = {
            **asdict(experiment),
            "task_config": asdict(dataset.config),
            "training": args.to_dict(),
            "versions": {
                name: version(name)
                for name in ("reasoning-gym", "trl", "transformers", "torch", "datasets")
            },
        }
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
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(Path(args.output_dir) / "final"))
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
