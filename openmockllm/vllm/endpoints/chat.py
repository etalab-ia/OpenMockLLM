import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.utils import count_tokens, generate_unstreamed_chat_content
from openmockllm.vllm.schemas import ChatCompletionRequest
from openmockllm.vllm.schemas.chat import ChatResponse, ChatResponseChoice, Message, Usage
from openmockllm.vllm.utils.chat import check_max_context_length, extract_prompt, generate_stream

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(path="/chat/completions", dependencies=[Depends(dependency=check_api_key)])
async def chat_completions(request: Request, body: ChatCompletionRequest):
    # get content from messages
    prompt = "\n\n".join([extract_prompt(content=msg.content) for msg in body.messages])

    # check max context length
    check_max_context_length(prompt=prompt, max_context_length=request.app.state.max_context)

    if not body.stream:
        # generate response content
        content = await generate_unstreamed_chat_content(prompt=prompt, max_tokens=body.max_tokens)
        input_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(content)

        # create response
        response = ChatResponse(
            id="baf234d63e524e74b25c2d764b043bc2",
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[ChatResponseChoice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=input_tokens, completion_tokens=completion_tokens, total_tokens=input_tokens + completion_tokens),
        )
        return response

    else:
        return StreamingResponse(content=generate_stream(request=request, body=body), media_type="text/event-stream")
