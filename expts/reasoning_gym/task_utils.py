"""Prompt and answer conventions shared by training and evaluation."""

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
