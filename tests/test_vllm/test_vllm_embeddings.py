import pytest


def test_embeddings_single_input(vllm_client):
    """Test creating embeddings with a single input string"""
    response = vllm_client.embeddings.create(
        model="openmockllm",
        input="Hello, world!",
    )

    assert response is not None
    assert response.object == "list"
    assert response.model == "openmockllm"
    assert response.data is not None
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    assert response.data[0].embedding is not None
    assert len(response.data[0].embedding) > 0
    assert response.usage is not None


def test_embeddings_multiple_inputs(vllm_client):
    """Test creating embeddings with multiple input strings"""
    inputs = [
        "Hello, world!",
        "This is a test",
        "Another sentence",
    ]

    response = vllm_client.embeddings.create(
        model="openmockllm",
        input=inputs,
    )

    assert response is not None
    assert response.object == "list"
    assert response.data is not None
    assert len(response.data) == 3

    # Check each embedding
    for i, embedding_data in enumerate(response.data):
        assert embedding_data.object == "embedding"
        assert embedding_data.index == i
        assert embedding_data.embedding is not None
        assert len(embedding_data.embedding) > 0


def test_embeddings_with_dimensions(vllm_client):
    """Test creating embeddings with specific dimensions"""
    response = vllm_client.embeddings.create(
        model="openmockllm",
        input="Hello, world!",
        dimensions=512,
    )

    assert response is not None
    assert len(response.data[0].embedding) == 512


@pytest.mark.asyncio
async def test_embeddings_async(vllm_async_client):
    """Test async embeddings creation"""
    response = await vllm_async_client.embeddings.create(
        model="openmockllm",
        input="Hello, world!",
    )

    assert response is not None
    assert response.object == "list"
    assert response.model == "openmockllm"
    assert response.data is not None
    assert len(response.data) == 1
    assert response.data[0].embedding is not None
