import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from openmockllm.logging import init_logger
from openmockllm.security import check_api_key
from openmockllm.vllm.exceptions import NotFoundError
from openmockllm.vllm.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    Message,
    Usage,
)
from openmockllm.vllm.utils.chat import generate_stream, get_chat_content

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(request: Request, body: ChatRequest):
    """Handle chat completion requests with streaming and non-streaming support"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    if body.model != request.app.state.model_name:
        raise NotFoundError(f"The model `{body.model}` does not exist.")
    content = get_chat_content()

    if body.stream:
        return StreamingResponse(generate_stream(request_id, body["model"], content), media_type="text/event-stream")
    else:
        response = ChatResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[ChatResponseChoice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        return response
