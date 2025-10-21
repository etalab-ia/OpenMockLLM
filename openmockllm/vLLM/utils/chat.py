import time

from lorem_text import lorem

from openmockllm.vllm.schemas.chat import ChatStreamResponse, ChatStreamResponseChoice


def get_chat_content() -> str:
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
