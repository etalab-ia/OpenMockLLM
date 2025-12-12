import pytest


@pytest.mark.asyncio
async def test_chat_completion_basic(mistral_client):
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


@pytest.mark.asyncio
async def test_chat_completion_with_max_tokens(mistral_client):
    """Test chat completion with max_tokens parameter"""
    response = mistral_client.chat.complete(
        model="openmockllm",
        messages=[{"role": "user", "content": "Tell me a short story"}],
        max_tokens=50,
    )

    assert response is not None
    assert response.choices[0].message.content is not None
    assert response.usage.completion_tokens <= 50


@pytest.mark.asyncio
async def test_chat_completion_multiple_messages(mistral_client):
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


@pytest.mark.asyncio
async def test_chat_completion_streaming(mistral_client):
    """Test streaming chat completion"""
    stream = mistral_client.chat.complete(
        model="openmockllm",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if chunk.data.choices[0].finish_reason is not None:
            break

    assert len(chunks) > 0
    # First chunk should have role
    if chunks[0].data.choices[0].delta.role:
        assert chunks[0].data.choices[0].delta.role == "assistant"
    # Verify we received content chunks
    content_chunks = [c for c in chunks if c.data.choices[0].delta.content]
    assert len(content_chunks) > 0
