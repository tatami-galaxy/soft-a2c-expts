"""PPO training: python -m expts.ppo.train --dataset deepscaler."""

from ..train import train
from .config import parse_config


def main():
    train(*parse_config(), method="ppo")


if __name__ == "__main__":
    main()
