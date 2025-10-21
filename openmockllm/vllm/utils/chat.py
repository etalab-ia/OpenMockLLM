import asyncio
import json
import random
import time
import uuid
from typing import AsyncGenerator

import tiktoken
from faker import Faker

tokenizer = tiktoken.get_encoding("cl100k_base")
fake = Faker("fr_FR")
fake.seed_instance()


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def generate_random_response(user_message: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    if max_tokens is None:
        max_tokens = random.randint(100, 1000)
    prompt_token_count = count_tokens(user_message)

    base_paragraphs = 1 + temperature * 5
    prompt_factor = 1 + (prompt_token_count / 100) * 0.5
    num_paragraphs = int(base_paragraphs * min(prompt_factor, 2.0))

    base_target = max_tokens * 4
    adjusted_target = int(base_target * min(prompt_factor, 1.5))
    target_length = min(adjusted_target, 8000)

    response_parts = []
    current_length = 0

    while current_length < target_length and len(response_parts) < num_paragraphs:
        text = fake.paragraph(nb_sentences=random.randint(1, 7))
        response_parts.append(text)
        current_length += len(text)

    return "\n\n".join(response_parts)


def calculate_realistic_delay(completion_tokens: int, temperature: float = 0.7) -> float:
    tokens_per_second = 35 - (temperature * 10)

    base_delay = completion_tokens / tokens_per_second

    startup_delay = random.uniform(0.1, 0.3)

    variation = random.uniform(0.85, 1.15)

    total_delay = (base_delay + startup_delay) * variation

    return max(0.1, total_delay)


async def generate_stream_response(response_text: str, model: str, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    tokens_per_second = 35 - (temperature * 10)
    token_delay = 1.0 / tokens_per_second

    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    token_ids = tokenizer.encode(response_text)

    for token_id in token_ids:
        token_text = tokenizer.decode([token_id])

        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        await asyncio.sleep(token_delay)

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"

    yield "data: [DONE]\n\n"
