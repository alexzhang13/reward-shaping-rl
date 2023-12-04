import os
from pathlib import Path

import openai

openai.organization = os.getenv("OPENAI_PERSONAL_ORG_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

LLM_TO_COST = {
    "gpt-3.5-turbo": {
        "input": 0.0015 / 1e3,
        "output": 0.002 / 1e3,
    },
    "gpt-4": {"input": 0.03 / 1e3, "output": 0.06 / 1e3},
}

MAX_TOKENS = 500
model = "gpt-3.5-turbo"


def query_gpt(prompt: str):
    """
    Script for querying ChatGPT
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    # compute cost
    input_cost = response["usage"]["prompt_tokens"] * LLM_TO_COST[model]["input"]
    output_cost = response["usage"]["completion_tokens"] * LLM_TO_COST[model]["output"]
    total_cost = input_cost + output_cost

    # print("Total cost:", total_cost)
    # print("LLM RESPONSE\n", response["choices"][0]["message"]["content"])
    return total_cost, response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    test_str = "Print me a random word."
    response = query_gpt(test_str)
    print("[RESPONSE]:", response)
