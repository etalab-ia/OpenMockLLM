from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool | None = False
    safe_prompt: bool | None = False
    random_seed: int | None = None


class ChatResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: Usage


class ChatStreamResponseChoice(BaseModel):
    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None


class ChatStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamResponseChoice]
