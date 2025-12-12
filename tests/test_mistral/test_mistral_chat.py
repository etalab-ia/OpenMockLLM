import pytest


def test_chat_completion_basic(mistral_client):
    """Test basic chat completion request"""
    response = mistral_client.chat.complete(
        model="openmockllm",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

    assert response is not None
    assert response.id is not None
    assert response.object == "chat.completion"
    assert response.model == "openmockllm"
    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0


def test_chat_completion_with_max_tokens(mistral_client):
    """Test chat completion with max_tokens parameter"""
    response = mistral_client.chat.complete(
        model="openmockllm",
        messages=[{"role": "user", "content": "Tell me a short story"}],
        max_tokens=50,
    )

    assert response is not None
    assert response.choices[0].message.content is not None
    assert response.usage.completion_tokens <= 50


def test_chat_completion_multiple_messages(mistral_client):
    """Test chat completion with multiple messages in conversation"""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    response = mistral_client.chat.complete(
        model="openmockllm",
        messages=messages,
    )

    assert response is not None
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_chat_completion_streaming_sync_basic(mistral_client):
    """Test streaming chat completion"""

    stream_response = mistral_client.chat.stream(model="openmockllm", messages=[{"role": "user", "content": "Hello, how are you?"}])

    for chunk in stream_response:
        assert chunk.data.choices[0].delta.content is not None


@pytest.mark.asyncio
async def test_chat_completion_async_streaming(mistral_client):
    """Test streaming chat completion"""
    stream = await mistral_client.chat.stream_async(model="openmockllm", messages=[{"role": "user", "content": "Hello, how are you?"}])

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if chunk.data.choices[0].finish_reason is not None:
            break

    assert len(chunks) > 0
    if chunks[0].data.choices[0].delta.role:
        assert chunks[0].data.choices[0].delta.role == "assistant"

    content_chunks = [c for c in chunks if c.data.choices[0].delta.content]
    assert len(content_chunks) > 0
