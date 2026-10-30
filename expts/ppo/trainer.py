"""Dataset-independent PPO using TRL GRPO's generation and clipped policy loss.

Adapted from self-distill/train/ppo/train_ppo.py. A separate critic supplies
token-level GAE in place of group-relative advantages. Only one training process
is supported; vLLM can be colocated or served separately.
"""

from pathlib import Path

import torch
from trl import GRPOTrainer
from trl.trainer.utils import disable_dropout_in_model


VALUE_MODEL_FILE = "value_model.pt"


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(values, mask):
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean).square(), mask)
    return (values - mean) * torch.rsqrt(variance + 1e-8) * mask


def compute_gae(rewards, values, mask, gamma=1.0, lam=1.0):
    """Values describe the state BEFORE each completion token; padding is terminal.

    With gamma=lam=1 and a terminal reward R, returns are R at every real token.
    Returns are computed before advantage whitening.
    """
    values = values * mask
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, 0])
    for t in reversed(range(values.shape[1])):
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_advantage = (delta + gamma * lam * last_advantage) * mask[:, t]
        advantages[:, t] = last_advantage
    return advantages, (advantages + values) * mask


class PPOTrainer(GRPOTrainer):
    def __init__(self, *args, value_model, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("The separate PPO critic currently supports one training process.")
        # Keep the critic out of self.model: vLLM must receive policy weights only.
        self.value_model = value_model.to(self.accelerator.device)
        self.value_model.train()
        if self.args.disable_dropout:
            disable_dropout_in_model(self.value_model)
        if self.args.gradient_checkpointing:
            self.value_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs
            )
            self.value_model.enable_input_require_grads()
        self.value_model.config.use_cache = False
        self._raw_rewards = None

    def create_optimizer(self):
        if self.optimizer is None:
            super().create_optimizer()
            self.optimizer.add_param_group({
                "params": [p for p in self.value_model.parameters() if p.requires_grad],
                "lr": self.args.critic_learning_rate,
                "weight_decay": self.args.weight_decay,
            })
        return self.optimizer

    def _clip_grad_norm(self, model):
        policy_norm = super()._clip_grad_norm(model)
        norm = torch.nn.utils.clip_grad_norm_(
            self.value_model.parameters(), self.args.critic_max_grad_norm or float("inf"),
            error_if_nonfinite=True,
        )
        self._metrics["train"]["ppo/critic_grad_norm"].append(norm.item())
        return policy_norm

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        # Transformers clears self.model's gradients, but our critic is separate.
        self.value_model.zero_grad(set_to_none=True)
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            directory = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
            torch.save(self.value_model.state_dict(), directory / VALUE_MODEL_FILE)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / VALUE_MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume PPO without critic weights: {path}")
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        # Load in place so the optimizer's parameter references remain valid.
        self.value_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    @property
    def in_critic_warmup(self):
        return self.model.training and self.state.global_step < self.args.critic_warmup_steps

    def _get_per_token_values(self, input_ids, attention_mask, completion_length):
        backbone = getattr(self.value_model, self.value_model.base_model_prefix)
        hidden = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        # The last prompt position predicts the first completion token's value.
        hidden = hidden[:, -(completion_length + 1):-1]
        return self.value_model.score(hidden).squeeze(-1).float()

    def _values(self, batch):
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        return self._get_per_token_values(input_ids, attention_mask, batch["completion_ids"].shape[1])

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._raw_rewards = rewards
        return rewards

    def _generate_and_score_completions(self, inputs):
        self._raw_rewards = None
        output = super()._generate_and_score_completions(inputs)
        mask = output["completion_mask"].float()
        raw_rewards = self._raw_rewards
        self._raw_rewards = None
        if raw_rewards is None or raw_rewards.shape[0] != mask.shape[0]:
            raise RuntimeError("TRL's reward hook did not return rewards aligned with PPO completions.")
        if not torch.isfinite(raw_rewards).all():
            raise ValueError("PPO requires a finite score for every completion.")
        weights = self.reward_weights.to(raw_rewards.device).unsqueeze(0)
        rewards = (raw_rewards * weights).sum(dim=1)
        token_rewards = torch.zeros_like(mask)
        last_token = (mask.sum(dim=1).long() - 1).clamp(min=0)
        token_rewards.scatter_(1, last_token[:, None], rewards[:, None])
        token_rewards *= mask  # also drops fully masked, truncated completions

        # Generation batches can be much larger than training microbatches.
        with torch.no_grad():
            chunks = []
            for start in range(0, mask.shape[0], self.args.per_device_train_batch_size):
                stop = start + self.args.per_device_train_batch_size
                batch = {key: output[key][start:stop] for key in (
                    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask"
                )}
                chunks.append(self._values(batch))
            old_values = torch.cat(chunks) * mask
        advantages, returns = compute_gae(token_rewards, old_values, mask, self.args.gamma, self.args.lam)
        if self.args.whiten_advantages:
            advantages = masked_whiten(advantages, mask)
        output.update(advantages=advantages, returns=returns, old_values=old_values)

        # Completion tables need a scalar: show mean token GAE, not GRPO's discarded baseline.
        logged_advantages = self._logs["advantages"]
        for _ in range(mask.shape[0]):
            logged_advantages.pop()
        logged_advantages.extend((advantages.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).tolist())

        mode = "train" if self.model.training else "eval"
        for name, value in (
            ("value_mean", masked_mean(old_values, mask)),
            ("returns_mean", masked_mean(returns, mask)),
            ("advantage_mean", masked_mean(advantages, mask)),
        ):
            self._metrics[mode][f"ppo/{name}"].append(value.item())
        return output

    def _compute_loss(self, model, inputs):
        policy_loss = super()._compute_loss(model, inputs)
        if self.in_critic_warmup:
            # No actor gradients: even weight decay cannot change the actor during warmup.
            policy_loss = policy_loss.detach() * 0.0
        mask = inputs["completion_mask"]
        values = self._values(inputs)
        clipped_values = inputs["old_values"] + (values - inputs["old_values"]).clamp(
            -self.args.cliprange_value, self.args.cliprange_value
        )
        squared_error = (values - inputs["returns"]).square()
        clipped_error = (clipped_values - inputs["returns"]).square()
        value_losses = 0.5 * torch.maximum(squared_error, clipped_error)

        # Mirror TRL 1.16's policy normalizer; Trainer adds no outer accumulation division.
        accumulation = self.current_gradient_accumulation_steps if self.model.training else 1
        if self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"].clamp(min=1)
            if self.model.training:
                normalizer = normalizer * accumulation / self.args.steps_per_generation
            value_loss = (value_losses * mask).sum() / normalizer
        else:
            value_loss = masked_mean(value_losses, mask) / accumulation

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ppo/value_loss"].append(value_loss.detach().item())
        self._metrics[mode]["ppo/value_clip_fraction"].append(
            masked_mean((clipped_error > squared_error).float(), mask).item()
        )
        self._metrics[mode]["ppo/critic_warmup"].append(float(self.in_critic_warmup))
        loss = policy_loss + self.args.vf_coef * value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("PPO produced a non-finite actor/critic loss.")
        return loss
