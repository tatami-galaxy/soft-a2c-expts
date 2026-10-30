# Small LLM reasoning experiments

GRPO and PPO support `--dataset reasoning_gym` (the default),
`--dataset deepscaler`, and `--dataset deepmath`. Training methods live in `expts/grpo` and `expts/ppo`;
dataset configs, prompts, and scoring live in `expts/data`. Both methods use
`expts/train.py` for model setup, run metadata, and checkpoint handling.
`expts/eval` evaluates fixed questions with the same prompts and scorers.
The two finite math datasets share preparation and scoring in `expts/data/math_data.py`;
their source mappings and pinned revisions live in separate dataset modules.

Select a training GPU with `CUDA_VISIBLE_DEVICES`. PPO requires one visible
training GPU and adds a separate trainable critic. The training environment
targets TRL 1.16.0, Transformers 5.17.0, and vLLM 0.31.0. Each math dataset additionally
uses the dependencies in `requirements-math.txt`:

```sh
uv pip install -r requirements-math.txt
```

## Reasoning Gym

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
    --dataset reasoning_gym --task leg_counting \
    --task_config '{"min_animals":5,"max_animals":5}' \
    --model_name Qwen/Qwen3-1.7B --max_steps 100
```

Replace `expts.grpo.train` with `expts.ppo.train` to use PPO. `--task all`
retains uniform task sampling and balanced per-task evaluation. See
[Reasoning Gym details](expts/data/README.md) and [PPO details](expts/ppo/README.md).
`--dataset_seed` optionally separates procedural data generation from the
training seed; it defaults to `--seed`.

## DeepScaleR

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
    --dataset deepscaler --model_name Qwen/Qwen3-1.7B \
    --run_name grpo-qwen17-deepscaler --max_steps 100

CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
    --dataset deepscaler --model_name Qwen/Qwen3-1.7B \
    --run_name ppo-qwen17-deepscaler --max_steps 100
```

The [dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
revision defaults to `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`. Override it
with a full commit SHA using `--dataset_revision`. Only `problem` enters the
prompt; `answer` supplies the reference, and `solution` is unused.

The initial load audits the data and writes a split manifest under
`data/deepscaler/`. Preparation removes empty problems/answers, keeps one row
per whitespace-normalized problem, excludes groups with conflicting answer
strings, and excludes references Math-Verify cannot parse. This conservative
conflict policy can discard equivalent answers written differently. Every
excluded row and its reason are recorded for inspection. Duplicate handling
does not detect paraphrases or all near-duplicates.

Defaults are `--source_split train`, `--holdout_size 500`, and `--split_seed 42`.
The remaining finite training set reshuffles each epoch until `max_steps`.
Changing the optimizer seed does not change the holdout. The manifest records
the source revision, content checksum, row IDs, exclusions, and verifier
versions. It is copied into every training and evaluation output directory.
Use `--split_manifest PATH` to reuse a particular split; explicit conflicting
options are rejected. With an existing manifest, omitted split options inherit
its settings.

The model is instructed to finish thinking and return `\boxed{...}`. Only the
last boxed answer after `</think>` is scored. Nested LaTeX braces are supported.
Math-Verify gives a binary equivalence reward, with no format bonus or partial
credit. Parsing and comparison each have a three-second timeout. Missing or
invalid predictions receive zero; verifier errors are recorded separately.
Training and evaluation call the same scoring function. Truncated completions
are masked from training losses by default, as in the existing experiments;
evaluation reports their actual extracted-answer score and truncation status.

## DeepMath-103K

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
    --dataset deepmath --model_name Qwen/Qwen3-1.7B \
    --run_name grpo-qwen17-deepmath --max_steps 100

CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
    --dataset deepmath --model_name Qwen/Qwen3-1.7B \
    --run_name ppo-qwen17-deepmath --max_steps 100

CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
    --dataset deepmath --model_name /path/to/deepmath-run/final --num_examples 100
```

The [dataset](https://huggingface.co/datasets/zwhe99/DeepMath-103K) is pinned to
`5cf055d1fe3d7a2eb19719ac020211469736ae44`. It supplies `question` as the prompt
and `final_answer` as the reward reference. `topic` and `difficulty` are retained
in example/result metadata. The three `r1_solution_*` columns are excluded;
the loader projects only the four needed Parquet columns.

DeepMath uses the same answer format, Math-Verify reward, auditing, and seeded
holdout procedure described above. Its split manifests live under
`data/deepmath/`; they cannot be substituted for DeepScaleR manifests.
The defaults remain `--source_split train`, `--holdout_size 500`, and
`--split_seed 42`. Training samples the cleaned dataset with epoch shuffling,
without topic or difficulty filtering. Evaluation uses the saved holdout order,
not the original dataset's first or last rows.

## Evaluate a holdout subset

Both math datasets support the commands below; for DeepMath, use
`--dataset deepmath` and the corresponding DeepMath run/manifest path.

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
    --dataset deepscaler --model_name /path/to/run/final --num_examples 100
```

For a local `final/` or `checkpoint-N/`, evaluation automatically uses the
training run's `split_manifest.json`. For a base model, pass that manifest
explicitly to compare on precisely the same questions:

```sh
CUDA_VISIBLE_DEVICES=2 uv run python -m expts.eval.eval \
    --dataset deepscaler --model_name Qwen/Qwen3-1.7B \
    --split_manifest /path/to/run/split_manifest.json --num_examples 100
```

Evaluation takes the first N problems from the manifest's seeded holdout order.
Thus 100 examples are a subset of the corresponding 500-example evaluation;
`num_examples` never changes the training split. Requesting more than the
holdout size is an error. Sampling seeds use original row IDs and
`--generation_seed`, independent of subset size. Outputs contain the selected
questions, row IDs, responses, scores, verifier errors, and aggregate metrics.
This evaluator covers in-distribution holdouts; OOD evaluation is separate
future work.

For Reasoning Gym, evaluation uses `--task` and `--task_config`, and
`--num_examples` remains the number per task when `--task all` is selected.
`--dataset_seed` controls its generated questions (default 1000000 for eval).
The math datasets use `--split_seed` for partitioning instead. Dataset-specific
flags are accepted only for the selected dataset. Run `--dataset deepmath --help`
to see its options.

## Reproducibility and checks

`run_config.json` separates `data` settings from `training` settings and records
package versions. Output directories default to
`/mnt/data/ujan/sac/<dataset>/<run_name>`; use `--output_dir` to override.
W&B projects default to the dataset name (`reasoning-gym`, `deepscaler`, or `deepmath`).
Resume using `--resume_from_checkpoint PATH` and the original settings.
Data, scoring, and optimizer setting changes are rejected. Extending
`max_steps` requires a constant learning-rate schedule. Old Reasoning Gym run
metadata is recognized; use the new module paths to resume those checkpoints.
Buffered rollouts are regenerated on resume, so restarts during rollout reuse
are not bitwise identical to uninterrupted runs.

Qwen sampling defaults are temperature 0.6, top-p 0.95, and top-k 20.
For MiniCPM5-2B-SFT, explicitly set `--temperature 1.0 --top_k 0` for both
training and evaluation. Match model, rollout count, sampling, token budget,
and actor optimizer/rate when comparing methods; PPO and GRPO have different
optimizer defaults.

```sh
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
    uv run python -m unittest discover -s tests -v
```
