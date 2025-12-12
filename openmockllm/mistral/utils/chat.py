from fastapi import Request
from mistralai.models import (
    ChatCompletionRequest,
    CompletionChunk,
    CompletionResponseStreamChoice,
    DeltaMessage,
)
from mistralai.types.basemodel import Unset

from openmockllm.utils import generate_stream_chat_content


async def generate_stream(request: Request, body: ChatCompletionRequest):
    """Generate streaming response chunks in SSE format"""

    prompt = "\n\n".join([msg.content for msg in body.messages])
    i = 0
    max_tokens = None if isinstance(body.max_tokens, Unset) else body.max_tokens
    async for chunk_text in generate_stream_chat_content(prompt=prompt, max_tokens=max_tokens):
        # Check if this is the final "[DONE]" chunk
        # The generator sends "[DONE]\n\n" as the final chunk
        if "[DONE]" in chunk_text:
            # Send final chunk with finish_reason
            chunk = CompletionChunk(
                id="baf234d63e524e74b25c2d764b043bc2",
                model=request.app.state.model_name,
                choices=[
                    CompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role=None, content=""),
                        finish_reason="stop",
                    )
                ],
            )
            # Format as SSE: data: <json>\n\n
            yield f"data: {chunk.model_dump_json()}\n\n"
            break

        # Regular content chunk
        role = "assistant" if i == 0 else None
        chunk = CompletionChunk(
            id="baf234d63e524e74b25c2d764b043bc2",
            model=request.app.state.model_name,
            choices=[
                CompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role=role, content=chunk_text),
                    finish_reason=None,
                )
            ],
        )
        # Format as SSE: data: <json>\n\n
        yield f"data: {chunk.model_dump_json()}\n\n"
        i += 1
