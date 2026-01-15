from faker import Faker
from fastapi import Request
import tiktoken

from openmockllm.settings import settings
from openmockllm.utils import generate_stream_chat_content
from openmockllm.vllm.schemas import ChatCompletionRequest
from openmockllm.vllm.schemas.chat import (
    ChatStreamResponse,
    ChatStreamResponseChoice,
    StreamDelta,
)

tokenizer = tiktoken.get_encoding(settings.tiktoken_encoder)
fake = Faker(settings.faker_langage)
fake.seed_instance(settings.faker_seed)


def extract_prompt(content: str | list | None) -> str:
    """
    Normalize vLLM message content to a plain text prompt.

    The SDK allows either:
    - a single string
    - a list of "content chunks" (e.g. text, input_audio, image_url, ...)
    """
    prompt = ""

    if isinstance(content, str):
        prompt = content

    if isinstance(content, list):
        prompt = ""
        for chunk in content:
            if "text" == chunk.type:
                prompt += chunk.text

    return prompt


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def check_max_context_length(prompt: str, max_context_length: int) -> int:
    return len(tokenizer.encode(prompt)) <= max_context_length


async def generate_stream(request: Request, body: ChatCompletionRequest):
    """Generate streaming response chunks in SSE format"""

    prompt = "\n\n".join([extract_prompt(content=msg.content) for msg in body.messages])
    i = 0

    async for chunk_text in generate_stream_chat_content(prompt=prompt, max_tokens=body.max_tokens):
        # Check if this is the final "[DONE]" chunk
        # The generator sends "[DONE]\n\n" as the final chunk
        if "[DONE]" in chunk_text:
            # Send final chunk with finish_reason
            chunk = ChatStreamResponse(
                id="baf234d63e524e74b25c2d764b043bc2",
                model=request.app.state.model_name,
                created=0,
                choices=[ChatStreamResponseChoice(index=i, delta=StreamDelta(role=None, content=""), finish_reason="stop")],
            )
            # Format as SSE: data: <json>\n\n
            yield f"data: {chunk.model_dump_json()}\n\n"
            break

        # Regular content chunk
        role = "assistant" if i == 0 else None
        chunk = ChatStreamResponse(
            id="baf234d63e524e74b25c2d764b043bc2",
            model=request.app.state.model_name,
            created=0,
            choices=[ChatStreamResponseChoice(index=i, delta=StreamDelta(role=role, content=chunk_text), finish_reason=None)],
        )
        # Format as SSE: data: <json>\n\n
        yield f"data: {chunk.model_dump_json()}\n\n"
        i += 1
