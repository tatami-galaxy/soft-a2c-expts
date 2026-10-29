"""Full-parameter PPO on streamed Reasoning Gym questions.

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.reasoning_gym.ppo.train \
        --model_name Qwen/Qwen3-1.7B --task leg_counting --max_steps 100

Uses a separate critic, GAE, and GRPOTrainer's vLLM generation / clipped policy
loss, not TRL's PPOTrainer. Select ONE training GPU. For larger models, use
--vllm_mode server with a separately launched vLLM server.

Eight rollouts per prompt match the GRPO baseline. --num_generations 1 is also
supported with an even generation batch. --num_iterations controls rollout
reuse (default 1). --critic_warmup_steps freezes the actor for the first N
optimizer steps, counted within max_steps. MiniCPM sampling requires explicit
--temperature 1.0 --top_k 0, just as in the GRPO entry point.

Resume with --resume_from_checkpoint PATH and the original experiment settings.
Checkpoint directories contain both models; final/ contains only the policy
and works with expts.reasoning_gym.eval.eval.
"""

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import reasoning_gym
from datasets import IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from ..task_utils import generate_examples, make_reward_function
from .config import parse_config
from .trainer import PPOTrainer


def validate_resume(checkpoint, config):
    """Optimizer state restores learning rates; reject misleading CLI changes."""
    previous = json.loads((Path(checkpoint).parent / "run_config.json").read_text())
    for key in ("method", "model_name", "task", "task_config", "versions"):
        if previous.get(key) != config[key]:
            raise ValueError(f"Cannot resume with changed {key}.")
    old, new = previous["training"], config["training"]
    fields = (
        "seed", "data_seed", "per_device_train_batch_size", "gradient_accumulation_steps",
        "generation_batch_size", "steps_per_generation", "num_generations", "num_iterations",
        "learning_rate", "critic_learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio",
        "optim", "weight_decay", "max_grad_norm", "critic_max_grad_norm", "critic_warmup_steps",
        "gamma", "lam", "vf_coef", "cliprange_value", "whiten_advantages", "loss_type",
        "epsilon", "epsilon_high", "max_completion_length", "mask_truncated_completions",
        "temperature", "top_p", "top_k", "generation_kwargs", "chat_template_kwargs",
        "bf16", "model_init_kwargs",
    )
    changed = [key for key in fields if old.get(key) != new.get(key)]
    if changed:
        raise ValueError(f"Cannot resume with changed training settings: {', '.join(changed)}")
    if old["max_steps"] != new["max_steps"] and new["lr_scheduler_type"] != "constant":
        raise ValueError("Changing max_steps on resume requires a constant learning-rate schedule.")


def main():
    experiment, args = parse_config()
    set_seed(args.seed)
    dataset = reasoning_gym.create_dataset(
        experiment.task, seed=args.seed, size=1, **experiment.task_config
    )
    output_dir = Path(args.output_dir)
    config = {
        **asdict(experiment),
        "method": "ppo",
        "task_config": asdict(dataset.config),
        "training": args.to_dict(),
        "versions": {name: version(name) for name in (
            "reasoning-gym", "trl", "transformers", "torch", "datasets", "accelerate", "vllm"
        )},
    }
    checkpoint = args.resume_from_checkpoint
    if checkpoint is True:
        checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        if checkpoint is None:
            raise ValueError(f"No checkpoint found in {output_dir}")
    if checkpoint:
        validate_resume(checkpoint, config)
    elif (output_dir / "run_config.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Resume or choose a new output_dir.")

    train_dataset = IterableDataset.from_generator(
        generate_examples,
        gen_kwargs={"task": experiment.task, "task_config": experiment.task_config, "seed": args.seed},
    )
    tokenizer = AutoTokenizer.from_pretrained(experiment.model_name, padding_side="left")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        experiment.model_name, num_labels=1, **args.model_init_kwargs
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    trainer = PPOTrainer(
        model=experiment.model_name,
        value_model=value_model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=make_reward_function(dataset, tokenizer),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run = None
    try:
        if "wandb" in args.report_to:
            import wandb

            run = wandb.init(
                project=experiment.wandb_project, entity=experiment.wandb_entity,
                name=args.run_name, config=config, dir=str(output_dir),
            )
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "final"))
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
