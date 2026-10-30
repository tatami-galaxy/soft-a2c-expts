"""Dataset-independent GRPO training defaults."""

from dataclasses import dataclass, field

from transformers.trainer_utils import SchedulerType
from trl import GRPOConfig


@dataclass
class TrainingConfig(GRPOConfig):
    """Our training defaults; all other options inherit TRL's defaults."""

    max_steps: int = 1000
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_generations: int = 8
    max_completion_length: int = 8192
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    learning_rate: float = 1e-5
    lr_scheduler_type: SchedulerType | str = "linear"
    bf16: bool = True
    gradient_checkpointing: bool = True
    loss_type: str = "dapo"
    beta: float = 0.0
    scale_rewards: str = "group"
    mask_truncated_completions: bool = True
    shuffle_dataset: bool | None = None
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    logging_steps: float = 1
    log_completions: bool = False
    report_to: None | str | list[str] = "wandb"
    save_steps: float = 100
    save_total_limit: int | None = 2
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    # Explicitly enable thinking even for templates that default to False.
    chat_template_kwargs: dict | str | None = field(default_factory=lambda: {"enable_thinking": True})
    generation_kwargs: dict | str | None = field(default_factory=lambda: {"min_p": 0.0})

