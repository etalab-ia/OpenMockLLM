import pytest


def test_chat_completion_basic(vllm_client):
    """Test basic chat completion request"""
    response = vllm_client.chat.completions.create(
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


def test_chat_completion_with_max_tokens(vllm_client):
    """Test chat completion with max_tokens parameter"""
    response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Tell me a short story"}],
        max_tokens=50,
    )

    assert response is not None
    assert response.choices[0].message.content is not None
    assert response.usage.completion_tokens <= 50


def test_chat_completion_multiple_messages(vllm_client):
    """Test chat completion with multiple messages in conversation"""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=messages,
    )

    assert response is not None
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


def test_chat_completion_sync_streaming(vllm_client):
    """Test streaming chat completion"""

    stream_response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=True,
    )

    chunks = []
    for chunk in stream_response:
        chunks.append(chunk)
        assert chunk.choices is not None
        assert len(chunk.choices) > 0

    assert len(chunks) > 0

    # Verify first chunk has role
    if chunks[0].choices[0].delta.role:
        assert chunks[0].choices[0].delta.role == "assistant"

    # Verify we have content chunks
    content_chunks = [c for c in chunks if c.choices[0].delta.content]
    assert len(content_chunks) > 0

    # Verify final chunk has finish_reason
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_completion_async_streaming(vllm_async_client):
    """Test async streaming chat completion"""
    stream = await vllm_async_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        assert chunk.choices is not None
        assert len(chunk.choices) > 0
        if chunk.choices[0].finish_reason is not None:
            break

    assert len(chunks) > 0

    # Verify first chunk has role
    if chunks[0].choices[0].delta.role:
        assert chunks[0].choices[0].delta.role == "assistant"

    # Verify we have content chunks
    content_chunks = [c for c in chunks if c.choices[0].delta.content]
    assert len(content_chunks) > 0


def test_chat_completion_with_temperature(vllm_client):
    """Test chat completion with temperature parameter"""
    response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    assert response is not None
    assert response.choices[0].message.content is not None


def test_chat_completion_with_top_p(vllm_client):
    """Test chat completion with top_p parameter"""
    response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Hello"}],
        top_p=0.9,
    )

    assert response is not None
    assert response.choices[0].message.content is not None


def test_chat_completion_with_stop_sequences(vllm_client):
    """Test chat completion with stop sequences"""
    response = vllm_client.chat.completions.create(
        model="openmockllm",
        messages=[{"role": "user", "content": "Count to 10"}],
        stop=["5", "five"],
    )

    assert response is not None
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_chat_completion_async_basic(vllm_async_client):
    """Test basic async chat completion request"""
    response = await vllm_async_client.chat.completions.create(
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
