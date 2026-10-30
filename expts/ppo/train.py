r"""Full-parameter PPO with a separate critic; select one training GPU.

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
        --dataset deepscaler --model_name Qwen/Qwen3-1.7B --max_steps 100

    CUDA_VISIBLE_DEVICES=2 uv run python -m expts.ppo.train \
        --dataset deepmath --model_name Qwen/Qwen3-1.7B --max_steps 100

Use --dataset reasoning_gym --task leg_counting for procedural tasks.
"""

from ..train import train
from .config import parse_config


def main():
    train(*parse_config(), method="ppo")


if __name__ == "__main__":
    main()
