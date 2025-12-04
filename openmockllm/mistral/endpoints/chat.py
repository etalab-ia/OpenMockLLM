import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from lorem_text import lorem

from openmockllm.logger import init_logger
from openmockllm.mistral.schemas.chat import (
    ChatResponse,
    ChatResponseChoice,
    ChatStreamResponse,
    ChatStreamResponseChoice,
    Message,
    Usage,
)
from openmockllm.security import check_api_key

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


def get_chat_content():
    """Generate lorem ipsum content"""
    return lorem.paragraphs(3)


async def generate_stream(request_id: str, model: str, content: str):
    """Generate streaming response chunks"""
    # First chunk with role
    chunk = ChatStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[ChatStreamResponseChoice(index=0, delta={"role": "assistant", "content": ""}, finish_reason=None)],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    # Split content into words for streaming
    words = content.split()
    for i, word in enumerate(words):
        chunk = ChatStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatStreamResponseChoice(
                    index=0,
                    delta={"content": word + (" " if i < len(words) - 1 else "")},
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    chunk = ChatStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[ChatStreamResponseChoice(index=0, delta={}, finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(request: Request, body: dict):
    """Handle chat completion requests with streaming and non-streaming support"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    model = body["model"]

    is_audio_request = any(
        item.get("type") == "input_audio"
        for message in body["messages"]
        for item in message.get("content", [])
    )

    if is_audio_request:
        response = ChatResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=get_chat_content()),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        return response
    elif body["stream"]:
        return StreamingResponse(
            generate_stream(request_id, model, get_chat_content()),
            media_type="text/event-stream",
        )
    else:
        response = ChatResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=get_chat_content()),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        return response