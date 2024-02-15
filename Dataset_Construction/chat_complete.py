"""
use chatGPT or GPT4 to complete the chat
"""

import openai
import time


def request(
        model,
        messages,
        max_tokens=512,
        temperature=0,
        top_p=0.7,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        api_key=None
):
    if api_key is not None:
        openai.api_key = api_key
    if type(messages) is str:
        messages = [
            {
                "role": "user",
                "content": messages
            }
        ]
    if model == "gpt-35-turbo":
        model = "gpt-3.5-turbo"
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                timeout=2
            )
            break
        except Exception as e:
            print(str(e))
            print("Retrying...")
            time.sleep(2)

    generations = [gen['message']['content'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if len(_) != 0]
    return generations
