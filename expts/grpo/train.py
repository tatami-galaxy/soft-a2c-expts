r"""Full-parameter GRPO on Reasoning Gym or finite math datasets.

Run from the repository root, selecting an available GPU:
    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
        --dataset reasoning_gym --task leg_counting --max_steps 100

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.grpo.train \
        --dataset deepmath --model_name Qwen/Qwen3-1.7B --max_steps 100

Use --dataset deepscaler for DeepScaleR with the same math dataset options.
"""

from ..config import parse_training_config
from ..train import train


def main():
    train(*parse_training_config(), method="grpo")


if __name__ == "__main__":
    main()
