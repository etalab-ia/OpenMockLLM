import asyncio
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
from openmockllm.vllm.utils.chat import calculate_realistic_delay, count_tokens, generate_random_response, generate_stream_response

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(request: Request, body: ChatRequest):
    """Handle chat completion requests with streaming and non-streaming support"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Use the model from the request or fall back to the default
    model = body.model or request.app.state.model_name

    # Validate model if specified
    if body.model and body.model != request.app.state.model_name:
        raise NotFoundError(f"The model `{body.model}` does not exist.")
    last_message = body.messages[-1].content if body.messages else ""
    simulated_response = generate_random_response(last_message, body.temperature, body.max_tokens)

    if body.stream:
        return StreamingResponse(generate_stream_response(simulated_response, body.model, body.temperature), media_type="text/event-stream")
    else:
        prompt_tokens = sum(count_tokens(msg.content) for msg in body.messages)
        completion_tokens = count_tokens(simulated_response)

        delay = calculate_realistic_delay(completion_tokens, body.temperature)
        await asyncio.sleep(delay)

        response = ChatResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[ChatResponseChoice(index=0, message=Message(role="assistant", content=simulated_response), finish_reason="stop")],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
        )
        return response
