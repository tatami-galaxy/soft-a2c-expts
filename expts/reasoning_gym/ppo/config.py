"""Reference-free PPO settings using the same CLI and rollout defaults as GRPO."""

from dataclasses import dataclass

from transformers.trainer_utils import SchedulerType

from ..grpo.config import TrainingConfig, parse_config as parse_training_config


@dataclass
class PPOConfig(TrainingConfig):
    learning_rate: float = 1e-6
    lr_scheduler_type: SchedulerType | str = "constant"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gamma: float = 1.0
    lam: float = 1.0
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    critic_learning_rate: float = 1e-5
    critic_max_grad_norm: float = 10.0
    critic_warmup_steps: int = 0
    whiten_advantages: bool = True
    scale_rewards: str = "none"

    def __post_init__(self):
        self.beta = 0.0
        self.scale_rewards = "none"
        if self.num_generations < 1:
            raise ValueError("PPO requires num_generations >= 1.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("gamma and lam must be in [0, 1].")
        if min(self.vf_coef, self.cliprange_value, self.critic_max_grad_norm, self.critic_warmup_steps) < 0:
            raise ValueError("Value loss, clipping, and critic warmup settings must be nonnegative.")
        if self.critic_learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("critic_learning_rate and max_grad_norm must be positive.")
        if self.loss_type not in {"dapo", "bnpo"}:
            raise ValueError("PPO supports loss_type=dapo or bnpo.")
        if self.deepspeed or self.fsdp:
            raise ValueError("PPO currently supports one training process without DeepSpeed/FSDP.")
        if self.fp16:
            raise ValueError("Use bf16 or float32; the separate critic does not support fp16 scaling.")
        if self.use_vllm and self.vllm_mode == "server" and any(
            name in str(self.optim).lower() for name in ("8bit", "bnb", "paged")
        ):
            raise ValueError("The source PPO experiments found server-mode corruption with bitsandbytes; use adafactor or adamw_torch.")

        # GRPO rejects G=1 even though PPO does not need a within-prompt baseline.
        # Keep its other validation intact. This workaround requires an even rollout batch.
        requested = self.num_generations
        if requested == 1:
            batch_size = self.generation_batch_size or (
                self.per_device_train_batch_size * (self.steps_per_generation or self.gradient_accumulation_steps)
            )
            if batch_size % 2:
                raise ValueError("num_generations=1 currently requires an even generation batch size.")
        self.num_generations = max(2, requested)
        try:
            super().__post_init__()
        finally:
            self.num_generations = requested
        if self.world_size != 1 or self.n_gpu > 1:
            raise ValueError("PPO requires one training process and one visible training GPU.")


def parse_config(argv=None):
    return parse_training_config(argv, config_class=PPOConfig, method="ppo")
