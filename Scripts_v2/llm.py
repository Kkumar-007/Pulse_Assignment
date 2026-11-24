"""One helper for every LLM call: Ollama's chat endpoint with JSON-schema output.

v1 sent hand-written Llama chat-template tokens to /api/generate, which wraps the
prompt in the model's own template again. /api/chat with plain system and user
messages avoids that, and `format` constrains the reply to valid JSON.
"""
import json
import time

import requests

from config import LLM_MODEL, OLLAMA_URL, REQUEST_TIMEOUT, RETRIES, SEED


def chat_json(system, user, schema, num_predict=256):
    """Return the model's reply parsed as JSON, or None if every attempt fails.

    Callers must still validate the content: the schema guarantees the shape,
    not that the answer is sensible.
    """
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "format": schema,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": SEED, "num_predict": num_predict},
    }
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"  LLM call failed (attempt {attempt}/{RETRIES}): {error}")
            time.sleep(2 * attempt)
    return None
