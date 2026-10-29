"""Reasoning Gym data, prompts, and rewards shared by the experiments."""

import json
import os
from collections import defaultdict
from dataclasses import asdict
from itertools import count
from pathlib import Path
from random import Random

import reasoning_gym
from reasoning_gym.composite import CompositeDataset, DatasetSpec
from reasoning_gym.factory import DATASETS
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


MIXTURE_CHUNK_SIZE = 1024


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


def validate_mixture_resume(checkpoint, config):
    path = Path(checkpoint).parent / "run_config.json"
    if not path.exists() and config["task"] != "all":
        return
    previous = json.loads(path.read_text())
    if previous.get("task") != "all" and config["task"] != "all":
        return
    for key in ("task", "task_config", "task_sampling"):
        if previous.get(key) != config.get(key):
            raise ValueError(f"Cannot resume with changed {key}.")
    if previous["versions"]["reasoning-gym"] != config["versions"]["reasoning-gym"]:
        raise ValueError("Cannot resume a task mixture with a different reasoning-gym version.")


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
            answer, thinking_done = extract_final_answer(response)
            try:
                reward = dataset.score_answer(answer, example) if answer else 0.0
            except Exception as error:
                name = example["metadata"]["source_dataset"]
                raise RuntimeError(f"Failed to score task {name!r}.") from error
            answers.append(answer)
            rewards.append(float(reward))
            thinking_finished.append(thinking_done)

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
