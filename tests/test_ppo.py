"""CPU checks for PPO mathematics and its boundaries with the inherited trainer."""

import tempfile
import types
import unittest
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import patch

import torch
from trl import GRPOTrainer
from trl.trainer.base_trainer import _BaseTrainer
from transformers import TrainingArguments

from expts.reasoning_gym.ppo.config import PPOConfig, parse_config
from expts.reasoning_gym.ppo.trainer import PPOTrainer, compute_gae, masked_whiten
from expts.reasoning_gym.task_utils import generate_examples, make_reward_function


class ConfigTest(unittest.TestCase):
    def config(self, **kwargs):
        return PPOConfig(use_cpu=True, bf16=False, report_to="none", use_vllm=False, **kwargs)

    def test_single_generation_and_reference_free_settings(self):
        config = self.config(num_generations=1, beta=0.5, scale_rewards="group")
        self.assertEqual(config.num_generations, 1)
        self.assertEqual(config.generation_batch_size, 16)
        self.assertEqual(config.beta, 0)
        self.assertEqual(config.scale_rewards, "none")

    def test_invalid_settings(self):
        for kwargs in (
            {"num_generations": 0}, {"num_generations": 1, "gradient_accumulation_steps": 3},
            {"gamma": -1}, {"lam": 1.1}, {"critic_learning_rate": 0},
            {"loss_type": "grpo"}, {"critic_warmup_steps": -1}, {"max_grad_norm": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config(**kwargs)

    def test_cli_preserves_task_and_sampling_overrides(self):
        experiment, config = parse_config([
            "--use_cpu", "true", "--bf16", "false", "--report_to", "none", "--use_vllm", "false",
            "--task", "leg_counting", "--task_config", '{"max_animals": 5}',
            "--temperature", "1.0", "--top_k", "0",
        ])
        self.assertEqual(experiment.task_config, {"max_animals": 5})
        self.assertEqual(config.temperature, 1)
        self.assertEqual(config.top_k, 0)
        self.assertTrue(config.run_name.startswith("ppo-"))


class GAETest(unittest.TestCase):
    def test_terminal_returns_and_padding(self):
        values = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.2, 99.0], [9.0, 9.0, 9.0]])
        rewards = torch.tensor([[0., 0., 1.], [0., 0.5, 0.], [0., 0., 0.]])
        mask = torch.tensor([[1., 1., 1.], [1., 1., 0.], [0., 0., 0.]])
        advantages, returns = compute_gae(rewards, values, mask)
        expected = torch.tensor([[1., 1., 1.], [0.5, 0.5, 0.], [0., 0., 0.]])
        torch.testing.assert_close(returns, expected)
        torch.testing.assert_close(advantages, expected - values * mask)

    def test_discounted_bootstrapping(self):
        values = torch.tensor([[0.2, 0.4]])
        rewards = torch.tensor([[0., 1.]])
        advantages, returns = compute_gae(rewards, values, torch.ones_like(values), gamma=0.9, lam=0.5)
        torch.testing.assert_close(advantages, torch.tensor([[0.43, 0.6]]))
        torch.testing.assert_close(returns, torch.tensor([[0.63, 1.]]))

    def test_whitening_ignores_padding_and_handles_empty_mask(self):
        values = torch.tensor([[1., 3., 100.]])
        mask = torch.tensor([[1., 1., 0.]])
        torch.testing.assert_close(masked_whiten(values, mask), torch.tensor([[-1., 1., 0.]]))
        torch.testing.assert_close(masked_whiten(values, mask * 0), torch.zeros_like(values))


def trainer_stub(loss_type="dapo", accumulation=1, steps_per_generation=1):
    trainer = object.__new__(PPOTrainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.value_model = torch.nn.Linear(1, 1, bias=False)
    trainer.args = types.SimpleNamespace(
        critic_warmup_steps=0, cliprange_value=0.2, vf_coef=1.,
        steps_per_generation=steps_per_generation, per_device_train_batch_size=1,
        gamma=1., lam=1., whiten_advantages=False,
    )
    trainer.state = types.SimpleNamespace(global_step=0)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.loss_type = loss_type
    trainer.current_gradient_accumulation_steps = accumulation
    return trainer


class TrainerTest(unittest.TestCase):
    def test_values_align_to_preceding_token(self):
        class Echo(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

        trainer = trainer_stub()
        trainer.value_model = types.SimpleNamespace(
            base_model_prefix="model", model=Echo(), score=torch.nn.Identity()
        )
        values = trainer._get_per_token_values(
            torch.tensor([[0, 9, 2, 3, 4]]), torch.tensor([[0, 1, 1, 1, 1]]), 3
        )
        torch.testing.assert_close(values, torch.tensor([[9., 2., 3.]]))

    def test_rewards_use_raw_scores_and_value_forward_is_chunked(self):
        trainer = trainer_stub()
        trainer.reward_weights = torch.tensor([1.])
        output = {
            "prompt_ids": torch.ones(3, 2, dtype=torch.long),
            "prompt_mask": torch.ones(3, 2, dtype=torch.long),
            "completion_ids": torch.ones(3, 3, dtype=torch.long),
            "completion_mask": torch.tensor([[1., 1., 1.], [1., 1., 0.], [0., 0., 0.]]),
            "advantages": torch.zeros(3),
        }
        batches = []

        def generate(*args):
            trainer._raw_rewards = torch.tensor([[1.], [0.25], [1.]])
            trainer._logs = {"advantages": deque([0., 0., 0.])}
            return output

        def values(batch):
            batches.append(batch["completion_ids"].shape[0])
            return torch.full_like(batch["completion_mask"], 0.1)

        trainer._values = values
        with patch.object(GRPOTrainer, "_generate_and_score_completions", generate):
            result = trainer._generate_and_score_completions([])
        self.assertEqual(batches, [1, 1, 1])
        torch.testing.assert_close(result["returns"], torch.tensor([[1., 1., 1.], [0.25, 0.25, 0.], [0., 0., 0.]]))
        torch.testing.assert_close(result["advantages"][2], torch.zeros(3))
        torch.testing.assert_close(torch.tensor(list(trainer._logs["advantages"])), torch.tensor([0.9, 0.15, 0.]))

    def loss_gradient(self, masks, accumulation, steps, loss_type="dapo"):
        trainer = trainer_stub(loss_type, accumulation, steps)
        value = torch.tensor(0.2, requires_grad=True)
        trainer._values = lambda inputs: value.expand_as(inputs["completion_mask"])
        total_tokens = sum(mask.sum() for mask in masks)
        with patch.object(GRPOTrainer, "_compute_loss", lambda *args: torch.tensor(0.)):
            for mask in masks:
                inputs = {
                    "completion_mask": mask, "old_values": torch.full_like(mask, 0.2),
                    "returns": torch.ones_like(mask), "num_items_in_batch": total_tokens,
                }
                trainer._compute_loss(trainer.model, inputs).backward()
        return value.grad

    def test_dapo_accumulation_matches_full_batch_with_unequal_lengths(self):
        mask = torch.tensor([[1., 1., 1.], [1., 0., 0.]])
        full = self.loss_gradient([mask], accumulation=1, steps=1)
        accumulated = self.loss_gradient(list(mask.split(1)), accumulation=2, steps=2)
        torch.testing.assert_close(full, torch.tensor(-0.8))
        torch.testing.assert_close(full, accumulated)
        # A generation batch spanning two optimizer steps requires rescaling.
        rescaled = self.loss_gradient(list(mask.split(1)), accumulation=1, steps=2)
        torch.testing.assert_close(rescaled, full * 2)

    def test_bnpo_and_empty_mask(self):
        mask = torch.ones(2, 3)
        grad = self.loss_gradient(list(mask.split(1)), accumulation=2, steps=2, loss_type="bnpo")
        torch.testing.assert_close(grad, torch.tensor(-0.8))
        for loss_type in ("dapo", "bnpo"):
            grad = self.loss_gradient([mask * 0], accumulation=1, steps=1, loss_type=loss_type)
            torch.testing.assert_close(grad, torch.tensor(0.))

    def test_warmup_freezes_actor_even_with_weight_decay(self):
        trainer = trainer_stub()
        trainer.args.critic_warmup_steps = 1
        actor = trainer.model.weight
        critic = trainer.value_model.weight
        trainer._values = lambda inputs: critic.expand_as(inputs["completion_mask"])
        optimizer = torch.optim.AdamW([actor, critic], lr=0.01, weight_decay=0.1)
        inputs = {
            "completion_mask": torch.ones(1, 2), "returns": torch.ones(1, 2),
            "old_values": critic.detach().expand(1, 2), "num_items_in_batch": torch.tensor(2),
        }
        actor_before, critic_before = actor.detach().clone(), critic.detach().clone()
        with patch.object(GRPOTrainer, "_compute_loss", lambda *args: actor.sum()):
            trainer._compute_loss(trainer.model, inputs).backward()
            self.assertIsNone(actor.grad)
            optimizer.step()
            torch.testing.assert_close(actor, actor_before)
            self.assertFalse(torch.equal(critic, critic_before))
            optimizer.zero_grad(set_to_none=True)
            trainer.state.global_step = 1
            trainer._compute_loss(trainer.model, inputs).backward()
            self.assertIsNotNone(actor.grad)

    def test_critic_checkpoint_restores_in_place_and_requires_weights(self):
        trainer = trainer_stub()
        with tempfile.TemporaryDirectory() as directory:
            trainer.args.should_save = True
            trainer._get_output_dir = lambda trial: directory
            checkpoint = Path(directory) / "checkpoint-0"
            checkpoint.mkdir()
            before = trainer.value_model.weight.detach().clone()
            parameter = trainer.value_model.weight
            with patch.object(GRPOTrainer, "_save_checkpoint"):
                trainer._save_checkpoint(trainer.model, None)
            with torch.no_grad():
                parameter.add_(1)
            with patch.object(GRPOTrainer, "_load_from_checkpoint"):
                trainer._load_from_checkpoint(checkpoint)
            self.assertIs(trainer.value_model.weight, parameter)
            torch.testing.assert_close(parameter, before)
            with self.assertRaises(FileNotFoundError):
                trainer._load_from_checkpoint(Path(directory) / "missing")


class TaskTest(unittest.TestCase):
    def test_stream_reproducibility_and_answer_contract(self):
        import reasoning_gym

        stream = generate_examples("leg_counting", {}, 42)
        first, second = next(stream), next(stream)
        self.assertEqual(first, next(generate_examples("leg_counting", {}, 42)))
        self.assertEqual(second["entry"], reasoning_gym.create_dataset("leg_counting", seed=43, size=1)[0])
        tokenizer = types.SimpleNamespace(decode=lambda text, **kwargs: text)
        dataset = reasoning_gym.create_dataset("leg_counting", seed=42, size=1)
        reward = make_reward_function(dataset, tokenizer)
        answer = first["entry"]["answer"]
        responses = [f"</think><answer>{answer}</answer>", f"<answer>{answer}</answer>", "unfinished"]
        self.assertEqual(reward(responses, [first["entry"]] * 3), [1., 0., 0.])


class AccumulationLoopTest(unittest.TestCase):
    def test_transformers_does_not_divide_normalized_losses_again(self):
        class ProbeTrainer(_BaseTrainer):
            loss_is_scaled_for_ga = PPOTrainer.loss_is_scaled_for_ga

            def _send_telemetry(self):
                pass

            def compute_loss(self, model, inputs, **kwargs):
                return model.weight.sum() * inputs["weight"].sum()

        updates = []
        for accumulation in (1, 4):
            with tempfile.TemporaryDirectory() as directory:
                model = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.ones_(model.weight)
                args = TrainingArguments(
                    output_dir=directory, use_cpu=True, report_to="none", max_steps=1,
                    per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
                    optim="sgd", learning_rate=1., lr_scheduler_type="constant", max_grad_norm=0.,
                    save_strategy="no", logging_strategy="no", disable_tqdm=True,
                    remove_unused_columns=False,
                )
                trainer = ProbeTrainer(
                    model=model, args=args,
                    train_dataset=[{"weight": 1. / accumulation}] * accumulation,
                )
                trainer.train()
                updates.append(model.weight.detach().clone())
        torch.testing.assert_close(updates[0], torch.zeros(1, 1))
        torch.testing.assert_close(updates[1], updates[0])


if __name__ == "__main__":
    unittest.main()
