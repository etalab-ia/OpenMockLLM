import asyncio
import random
import time
import uuid
from typing import List, Optional

import tiktoken
from faker import Faker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock LLM API", version="1.0.0")

fake = Faker('fr_FR')
tokenizer = tiktoken.get_encoding("cl100k_base")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.get("/")
async def root():
    return {"message": "Mock LLM API - Compatible OpenAI", "status": "running"}


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def generate_random_response(user_message: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
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

    return '\n\n'.join(response_parts)


def calculate_realistic_delay(completion_tokens: int, temperature: float = 0.7) -> float:
    tokens_per_second = 35 - (temperature * 10)

    base_delay = completion_tokens / tokens_per_second

    startup_delay = random.uniform(0.1, 0.3)

    variation = random.uniform(0.85, 1.15)

    total_delay = (base_delay + startup_delay) * variation

    return max(0.1, total_delay)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming non supporté dans cette version mock")

    last_message = request.messages[-1].content if request.messages else ""

    simulated_response = generate_random_response(
        last_message,
        request.temperature,
        request.max_tokens
    )

    prompt_tokens = sum(count_tokens(msg.content) for msg in request.messages)
    completion_tokens = count_tokens(simulated_response)

    delay = calculate_realistic_delay(completion_tokens, request.temperature)
    # await asyncio.sleep(delay)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=simulated_response),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

    return response


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
