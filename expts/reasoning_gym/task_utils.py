"""Reasoning Gym data, prompts, and rewards shared by the experiments."""

from itertools import count

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


def make_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["default"]},
        {"role": "user", "content": question},
    ]


def extract_final_answer(response):
    """Only text after </think> counts; a response that never finishes thinking has no answer."""
    if "</think>" not in response:
        return None, False
    return extract_answer(response.rsplit("</think>", 1)[1]), True


def generate_examples(task, task_config, seed):
    """Each example is reproducible from its seed and task configuration."""
    for example_id in count():
        dataset = reasoning_gym.create_dataset(
            task, seed=seed + example_id, size=1, **task_config
        )
        entry = dataset[0]
        yield {
            "prompt": make_prompt(entry["question"]),
            "entry": entry,
            "example_id": example_id,
        }


def make_reward_function(dataset, tokenizer):
    def reasoning_reward(completion_ids, entry, log_metric=None, log_extra=None, **kwargs):
        answers, rewards, thinking_finished = [], [], []
        for token_ids, example in zip(completion_ids, entry, strict=True):
            # Decode tokens ourselves so TRL's response parsing cannot remove </think>.
            response = tokenizer.decode(token_ids, skip_special_tokens=False)
            answer, thinking_done = extract_final_answer(response)
            reward = dataset.score_answer(answer, example) if answer else 0.0
            answers.append(answer)
            rewards.append(float(reward))
            thinking_finished.append(thinking_done)

        if log_metric:
            log_metric("full_credit_rate", sum(r == 1.0 for r in rewards) / len(rewards))
            log_metric("answer_extraction_rate", sum(bool(a) for a in answers) / len(answers))
            log_metric("thinking_complete_rate", sum(thinking_finished) / len(thinking_finished))
        if log_extra:
            log_extra("extracted_answer", answers)
            log_extra("reference_answer", [example["answer"] for example in entry])
        return rewards

    return reasoning_reward
