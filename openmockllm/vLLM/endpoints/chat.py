import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from lorem_text import lorem

from openmockllm.logging import init_logger
from openmockllm.vllm.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    ChatStreamResponse,
    ChatStreamResponseChoice,
    Message,
    Usage,
)

logger = init_logger(__name__)
router = APIRouter()


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


@router.post("/completions")
async def chat_completions(chat_request: ChatRequest, request: Request):
    """Handle chat completion requests with streaming and non-streaming support"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    model = chat_request.model
    content = get_chat_content()

    if chat_request.stream:
        return StreamingResponse(
            generate_stream(request_id, model, content),
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
                    message=Message(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        return response
