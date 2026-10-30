# Reasoning Gym experiments

GRPO and PPO share task generation, thinking-mode prompts, answer extraction,
and task-specific scoring. Single-task training still uses `--task NAME` and
optional `--task_config` overrides.

## Uniform task training

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
    --task all --model_name Qwen/Qwen3-1.7B --max_steps 100

PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
    --task all --model_name Qwen/Qwen3-1.7B --max_steps 100
```

Choose a GPU with enough free memory; PPO additionally loads a trainable critic.
See [PPO settings](../ppo/README.md) for its options and memory considerations.

`all` includes every registered task except the composite wrapper: 105 tasks
in the installed reasoning-gym 0.1.25. Every new question independently selects
a task with equal probability, with replacement. All tasks use their library
defaults; `--task_config` must be empty in this mode.

GRPO/PPO then generate `num_generations` completions for each selected question.
For example, 16 completions with eight generations per question represent only
two independent task draws. PPO also supports `--num_generations 1`.

The stream uses Reasoning Gym's composite dataset in lazy chunks of 1024
questions. Chunk k starts at seed `seed + 1024*k`, avoiding overlapping
seed/index ranges between chunks. Task selection and question generation are
reproducible for fixed settings and library versions. Set `PYTHONHASHSEED=0`
before launching Python because some upstream generators iterate sets.
Duplicate question content can still occur naturally in finite task spaces.

`run_config.json` records the ordered task list, equal weights, resolved task
defaults, initial seeds, chunk scheme, Python hash seed, and package version.
Both trainers reject resuming a changed mixture. Generation/scoring failures
raise errors instead of silently skipping tasks or switching to another task.

Rewards use each entry's original task scorer, preserving partial credit.
Completion logs include the task name. Per-task metrics include mean reward,
full-credit rate, and `questions_per_generation`, which counts distinct question
IDs before rollout repetition. Counts include zeros for absent tasks and are
averaged over the logging interval (and training processes). They are not
cumulative run totals. Reward metrics are reported when that task is present.

Uniform question sampling does not imply equal token weight under DAPO.
Loss weighting and truncation handling are unchanged.

## Balanced evaluation

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
    --task all --model_name /path/to/run/final --num_examples 10
```

For `--task all`, `num_examples` means **examples per task**: the command above
evaluates 1050 questions. Each task gets a fixed seeded question set rather than
a random share of the evaluation budget. Results include the task in metadata,
per-task summaries, and macro-average full-credit rate and mean reward.
Keep task defaults, seeds, token budgets, and sampling settings matched across
models. MiniCPM5-2B-SFT uses explicit `--temperature 1.0 --top_k 0`.

Run the CPU checks with:

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 uv run python -m unittest discover -s tests -v
```
