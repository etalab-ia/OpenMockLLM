import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from mistralai.models import AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, CompletionEvent, UsageInfo
from mistralai.types.basemodel import Unset

from openmockllm.mistral.utils.chat import generate_stream
from openmockllm.security import check_api_key
from openmockllm.utils import count_tokens, generate_unstreamed_chat_content

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(path="/chat/completions", dependencies=[Depends(dependency=check_api_key)])
async def chat_completions(request: Request, body: ChatCompletionRequest) -> ChatCompletionResponse | CompletionEvent:
    if not body.stream:
        prompt = "\n\n".join([msg.content for msg in body.messages])
        max_tokens = None if isinstance(body.max_tokens, Unset) else body.max_tokens
        content = await generate_unstreamed_chat_content(prompt=prompt, max_tokens=max_tokens)
        input_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(content)
        response = ChatCompletionResponse(
            id="baf234d63e524e74b25c2d764b043bc2",
            object="chat.completion",
            created=int(time.time()),
            usage=UsageInfo(prompt_tokens=input_tokens, completion_tokens=completion_tokens, total_tokens=input_tokens + completion_tokens),
            model=request.app.state.model_name,
            choices=[ChatCompletionChoice(index=0, message=AssistantMessage(content=content, tool_calls=None), finish_reason="stop")],
        )
        return response

    else:
        return StreamingResponse(content=generate_stream(request=request, body=body), media_type="text/event-stream")
