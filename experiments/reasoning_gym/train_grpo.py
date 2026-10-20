"""Fine-tune all model parameters on streamed Reasoning Gym questions with TRL GRPO.

From the repository root (select an available GPU):
    CUDA_VISIBLE_DEVICES=4 uv run python -m experiments.reasoning_gym.train_grpo \
        --task leg_counting --max_steps 100

Override task defaults with --task_config '{"max_animals": 5}'.
Override the W&B account or team with --wandb_entity TEAM.
Standard GRPOConfig arguments are also accepted; see --help.
Tested with TRL 1.16.0 and reasoning-gym 0.1.25.
"""

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import count
from pathlib import Path

import reasoning_gym
from datasets import IterableDataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from trl import GRPOConfig, GRPOTrainer

from .task_utils import extract_final_answer, make_prompt


WANDB_PROJECT = "reasoning-gym"


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-4B"
    task: str = "leg_counting"
    task_config: str = "{}"
    enable_thinking: bool = True
    wandb_entity: str = "ujan-rl"


def generate_examples(task, task_config, seed):
    """Each example is reproducible from its seed and task configuration."""
    for example_id in count():
        dataset = reasoning_gym.create_dataset(
            task, seed=seed + example_id, size=1, **task_config
        )
        entry = dataset[0]
        yield {
            "prompt": make_prompt(entry["question"]),
            "entry": entry,
            "example_id": example_id,
        }


def make_reward_function(dataset, tokenizer, enable_thinking):
    def reasoning_reward(completion_ids, entry, log_metric=None, log_extra=None, **kwargs):
        answers, rewards, thinking_finished = [], [], []
        for token_ids, example in zip(completion_ids, entry, strict=True):
            # Decode tokens ourselves so TRL's response parsing cannot remove </think>.
            response = tokenizer.decode(token_ids, skip_special_tokens=False)
            answer, thinking_done = extract_final_answer(response, enable_thinking)
            reward = dataset.score_answer(answer, example) if answer else 0.0
            answers.append(answer)
            rewards.append(float(reward))
            thinking_finished.append(thinking_done)

        if log_metric:
            log_metric("full_credit_rate", sum(r == 1.0 for r in rewards) / len(rewards))
            log_metric("answer_extraction_rate", sum(bool(a) for a in answers) / len(answers))
            if enable_thinking:
                log_metric("thinking_complete_rate", sum(thinking_finished) / len(thinking_finished))
        if log_extra:
            log_extra("extracted_answer", answers)
            log_extra("reference_answer", [example["answer"] for example in entry])
        return rewards

    return reasoning_reward


def main():
    parser = HfArgumentParser((ExperimentConfig, GRPOConfig))
    parser.set_defaults(
        max_steps=1000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_completion_length=8192,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=True,
        loss_type="dapo",
        beta=0.0,
        scale_rewards="group",
        mask_truncated_completions=True,
        shuffle_dataset=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        logging_steps=1,
        log_completions=False,
        report_to="wandb",
        save_steps=100,
        save_total_limit=2,
        temperature=None,
        top_p=None,
        top_k=20,
    )
    experiment, args = parser.parse_args_into_dataclasses()
    task_config = json.loads(experiment.task_config)
    if not isinstance(task_config, dict) or {"seed", "size"} & task_config.keys():
        parser.error("--task_config must be a JSON object without seed or size; use --seed.")
    if args.max_steps <= 0:
        parser.error("Streaming training requires --max_steps > 0.")

    model_label = experiment.model_name.rstrip("/").split("/")[-1]
    args.run_name = f"{model_label}-{experiment.task}-seed{args.seed}"
    # HfArgumentParser initializes the Trainer's default output_dir before this point.
    if args.output_dir == "trainer_output":
        args.output_dir = f"results/reasoning_gym/{args.run_name}"
    args.chat_template_kwargs = {
        **(args.chat_template_kwargs or {}), "enable_thinking": experiment.enable_thinking
    }
    if args.temperature is None:
        args.temperature = 0.6 if experiment.enable_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if experiment.enable_thinking else 0.8
    args.generation_kwargs = {"min_p": 0.0, **(args.generation_kwargs or {})}
    args.model_init_kwargs = {
        "dtype": "bfloat16" if args.bf16 else "float32", **(args.model_init_kwargs or {})
    }
    set_seed(args.seed)

    dataset = reasoning_gym.create_dataset(
        experiment.task, seed=args.seed, size=1, **task_config
    )
    train_dataset = IterableDataset.from_generator(
        generate_examples,
        gen_kwargs={"task": experiment.task, "task_config": task_config, "seed": args.seed},
    )
    tokenizer = AutoTokenizer.from_pretrained(experiment.model_name, padding_side="left")
    trainer = GRPOTrainer(
        model=experiment.model_name,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=make_reward_function(dataset, tokenizer, experiment.enable_thinking),
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
                project=WANDB_PROJECT,
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
