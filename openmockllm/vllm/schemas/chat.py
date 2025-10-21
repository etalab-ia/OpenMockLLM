from typing import Any, Dict, List, Optional, Union

from openmockllm.vllm.schemas.core import VllmBaseModel


class Message(VllmBaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatRequest(VllmBaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatResponseChoice(VllmBaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(VllmBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatResponseChoice]
    usage: Usage


class ChatStreamResponseChoice(VllmBaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatStreamResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamResponseChoice]
