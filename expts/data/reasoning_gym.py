"""Reasoning Gym configuration, data, prompts, and task-specific rewards."""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import count
from random import Random

import reasoning_gym
from reasoning_gym.composite import CompositeDataset, DatasetSpec
from reasoning_gym.factory import DATASETS
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


MIXTURE_CHUNK_SIZE = 1024


@dataclass
class DataConfig:
    task: str = "leg_counting"
    task_config: dict | str | None = field(default_factory=dict)
    dataset_seed: int | None = None  # Training seed, or 1000000 for evaluation.


def validate_config(config, parser):
    if isinstance(config.task_config, str):
        try:
            config.task_config = json.loads(config.task_config)
        except json.JSONDecodeError as error:
            parser.error(f"Invalid --task_config JSON: {error}")
    if not isinstance(config.task_config, dict) or {"seed", "size"} & config.task_config.keys():
        parser.error("--task_config must be an object without seed or size; use --dataset_seed.")
    if config.task == "all" and config.task_config:
        parser.error("--task all uses task defaults; leave --task_config empty.")


def data_metadata(config, dataset):
    return {
        "task": config.task,
        "task_config": resolved_task_config(dataset),
        "task_sampling": task_sampling_config(config.task),
        "system_prompt": SYSTEM_PROMPTS["default"],
        "answer_format": "think_then_answer_tags",
    }


def load_training_data(config, seed):
    from datasets import IterableDataset

    seed = config.dataset_seed if config.dataset_seed is not None else seed
    dataset = create_task_dataset(config.task, config.task_config, seed)
    rows = IterableDataset.from_generator(generate_examples, gen_kwargs={
        "task": config.task, "task_config": config.task_config, "seed": seed,
    })
    return rows, dataset, data_metadata(config, dataset)


def load_eval_data(config, num_examples):
    seed = config.dataset_seed if config.dataset_seed is not None else 1_000_000
    dataset = create_task_dataset(config.task, config.task_config, seed, size=num_examples)
    entries = [{"example_id": i, **entry} for i, entry in enumerate(evaluation_entries(dataset))]
    metadata = data_metadata(config, dataset)
    metadata.pop("task_sampling")  # Evaluation does not use the training mixture stream.
    metadata["evaluation_sampling"] = "balanced_per_task" if config.task == "all" else "single_task"
    metadata["python_hash_seed"] = os.environ.get("PYTHONHASHSEED", "random")
    return entries, dataset, metadata


def score_response(dataset, response, entry):
    answer, thinking_done = extract_final_answer(response)
    return {
        "model_answer": answer,
        "score": float(dataset.score_answer(answer, entry)) if answer else 0.0,
        "thinking_complete": thinking_done,
    }


def create_task_dataset(task, task_config, seed, size=None):
    """Construct one task or an equally weighted mixture of every registered task."""
    if task != "all":
        return reasoning_gym.create_dataset(task, seed=seed, size=size or 1, **task_config)
    if task_config:
        raise ValueError("--task all uses each task's defaults; leave --task_config empty.")
    specs = [DatasetSpec(name=name, weight=1.0, config={}) for name in sorted(DATASETS) if name != "composite"]
    return reasoning_gym.create_dataset(
        "composite", seed=seed, size=size or MIXTURE_CHUNK_SIZE, datasets=specs
    )


def resolved_task_config(dataset):
    """Record actual child defaults, not just the empty overrides in DatasetSpec."""
    config = asdict(dataset.config)
    if isinstance(dataset, CompositeDataset):
        for spec in config["datasets"]:
            spec["config"] = asdict(dataset.datasets[spec["name"]].config)
    # Time-interval tasks have datetime.time defaults; tuples also need canonical lists
    # so an in-memory config compares equal to a checkpoint's JSON on resume.
    return json.loads(json.dumps(config, default=lambda value: value.isoformat()))


def task_sampling_config(task):
    if task != "all":
        return None
    return {
        "method": "uniform_per_question_v1",
        "chunk_size": MIXTURE_CHUNK_SIZE,
        "chunk_seed": "seed + chunk_index * chunk_size",
        # Some upstream tasks iterate Python sets when generating examples.
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
    }


def evaluation_entries(dataset):
    """All-task evaluation uses the same number of examples from each task."""
    if isinstance(dataset, CompositeDataset):
        return [entry for child in dataset.datasets.values() for entry in child]
    return list(dataset)


def make_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["default"]},
        {"role": "user", "content": question},
    ]


def extract_final_answer(response):
    """Only text after </think> counts; a response that never finishes thinking has no answer."""
    if "</think>" not in response:
        return None, False
    return extract_answer(response.rsplit("</think>", 1)[1]), True


def generate_examples(task, task_config, seed):
    """Each example is reproducible from its seed and task configuration."""
    dataset = None
    for example_id in count():
        if task == "all":
            index = example_id % MIXTURE_CHUNK_SIZE
            if index == 0:
                # Advance by the entire chunk, not one: seed+index ranges must not overlap.
                dataset = create_task_dataset(task, task_config, seed + example_id)
        else:
            dataset = create_task_dataset(task, task_config, seed + example_id)
            index = 0
        try:
            entry = dataset[index]
        except Exception as error:
            name = task
            if task == "all":
                name = Random(dataset.seed + index).choices(dataset.dataset_names, weights=dataset.weights, k=1)[0]
            raise RuntimeError(f"Failed to generate task {name!r} at example {example_id}; no example was skipped.") from error
        yield {
            "prompt": make_prompt(entry["question"]),
            "entry": entry,
            "example_id": example_id,
        }


def make_reward_function(dataset, tokenizer):
    def reasoning_reward(completion_ids, entry, example_id=None, log_metric=None, log_extra=None, **kwargs):
        answers, rewards, thinking_finished = [], [], []
        for token_ids, example in zip(completion_ids, entry, strict=True):
            # Decode tokens ourselves so TRL's response parsing cannot remove </think>.
            response = tokenizer.decode(token_ids, skip_special_tokens=False)
            try:
                result = score_response(dataset, response, example)
            except Exception as error:
                name = example["metadata"]["source_dataset"]
                raise RuntimeError(f"Failed to score task {name!r}.") from error
            answers.append(result["model_answer"])
            rewards.append(result["score"])
            thinking_finished.append(result["thinking_complete"])

        if log_metric:
            log_metric("full_credit_rate", sum(r == 1.0 for r in rewards) / len(rewards))
            log_metric("answer_extraction_rate", sum(bool(a) for a in answers) / len(answers))
            log_metric("thinking_complete_rate", sum(thinking_finished) / len(thinking_finished))
            if isinstance(dataset, CompositeDataset):
                indices = defaultdict(list)
                for i, example in enumerate(entry):
                    indices[example["metadata"]["source_dataset"]].append(i)
                for name in dataset.dataset_names:
                    rows = indices[name]
                    # Include zeros so averaging logging windows preserves task frequencies.
                    if example_id is not None:
                        log_metric(f"tasks/{name}/questions_per_generation", len({example_id[i] for i in rows}))
                    if rows:
                        log_metric(f"tasks/{name}/mean_reward", sum(rewards[i] for i in rows) / len(rows))
                        log_metric(f"tasks/{name}/full_credit_rate", sum(rewards[i] == 1.0 for i in rows) / len(rows))
        if log_extra:
            log_extra("extracted_answer", answers)
            log_extra("reference_answer", [example["answer"] for example in entry])
            if isinstance(dataset, CompositeDataset):
                log_extra("task", [example["metadata"]["source_dataset"] for example in entry])
        return rewards

    return reasoning_reward
