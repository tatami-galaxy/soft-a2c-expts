"""GRPO training: python -m expts.grpo.train --dataset reasoning_gym --task leg_counting."""

from ..config import parse_training_config
from ..train import train


def main():
    train(*parse_training_config(), method="grpo")


if __name__ == "__main__":
    main()
