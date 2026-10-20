"""Prompt and answer conventions shared by training and evaluation."""

from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


def make_prompt(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["default"]},
        {"role": "user", "content": question},
    ]


def extract_final_answer(response, enable_thinking):
    thinking_done = "</think>" in response
    if thinking_done:
        response = response.rsplit("</think>", 1)[1]
    elif enable_thinking:
        response = ""
    return extract_answer(response), thinking_done
