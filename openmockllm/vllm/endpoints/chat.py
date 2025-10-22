import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from openmockllm.logger import init_json_logger
from openmockllm.security import check_api_key
from openmockllm.vllm.exceptions import NotFoundError
from openmockllm.vllm.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    Message,
    Usage,
)
from openmockllm.vllm.utils.chat import (
    calculate_realistic_delay,
    count_tokens,
    generate_random_response,
    generate_stream_response,
    get_active_requests,
    increment_active_requests,
    decrement_active_requests,
)

logger = init_json_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(request: Request, body: ChatRequest):
    """Handle chat completion requests with streaming and non-streaming support"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    start_time = time.perf_counter()
    try:
        increment_active_requests()
        logger.info("beginning of request", extra={"active requests": get_active_requests()})

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
    finally:
        decrement_active_requests()
        execution_time = time.perf_counter() - start_time
        logger.info(
            "end of request",
            extra={
                "request message": last_message,
                "active requests": get_active_requests(),
                "execution_time_seconds": round(execution_time, 3),
                "request_id": request_id,
            },
        )
