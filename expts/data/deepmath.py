"""DeepMath-103K question/answer mapping and metadata for RL and holdout evaluation."""

from datasets import load_dataset

from . import math_data
from .math_data import DataConfig, make_prompt, make_reward_function, score_response, validate_config


DATASET_ID = "zwhe99/DeepMath-103K"
DEFAULT_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"


def load_rows(revision, split):
    # Project Parquet columns while loading: the three long R1 solutions are unused.
    columns = ["question", "final_answer", "difficulty", "topic"]
    rows = load_dataset(DATASET_ID, revision=revision, split=split, columns=columns)
    # Also restrict cached datasets if the loader falls back to a full-column cache.
    return rows.select_columns(columns).rename_columns({"question": "problem", "final_answer": "answer"})


def _load(config, num_examples=None):
    return math_data.load_data(
        config, dataset_name="deepmath", dataset_id=DATASET_ID,
        default_revision=DEFAULT_REVISION, load_rows=load_rows, num_examples=num_examples,
    )


def load_training_data(config, seed):
    return _load(config)


def load_eval_data(config, num_examples):
    return _load(config, num_examples)
