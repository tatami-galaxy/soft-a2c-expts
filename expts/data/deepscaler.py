"""DeepScaleR source fields and pinned revision, with shared math preparation."""

from datasets import load_dataset

from . import math_data
from .math_data import DataConfig, make_prompt, make_reward_function, score_response, validate_config


DATASET_ID = "agentica-org/DeepScaleR-Preview-Dataset"
DEFAULT_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"


def load_rows(revision, split):
    rows = load_dataset(DATASET_ID, revision=revision, split=split)
    return rows.select_columns(["problem", "answer"])


def _load(config, num_examples=None):
    return math_data.load_data(
        config, dataset_name="deepscaler", dataset_id=DATASET_ID,
        default_revision=DEFAULT_REVISION, load_rows=load_rows, num_examples=num_examples,
    )


def load_training_data(config, seed):
    return _load(config)


def load_eval_data(config, num_examples):
    return _load(config, num_examples)
