import base64

import pytest


def test_embeddings_single_string(tei_client):
    """Test embeddings generation for a single string"""
    response = tei_client.post("/v1/embeddings", json={"input": "Hello, world!", "model": "openmockllm"})

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert data["model"] == "openmockllm"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["index"] == 0
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) == 1024  # Default dimension from server
    assert data["usage"]["prompt_tokens"] == 0
    assert data["usage"]["total_tokens"] == 0


def test_embeddings_list_of_strings(tei_client):
    """Test embeddings generation for multiple strings"""
    texts = ["First text", "Second text", "Third text"]
    response = tei_client.post("/v1/embeddings", json={"input": texts, "model": "openmockllm"})

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 3

    for i, embedding_data in enumerate(data["data"]):
        assert embedding_data["object"] == "embedding"
        assert embedding_data["index"] == i
        assert isinstance(embedding_data["embedding"], list)
        assert len(embedding_data["embedding"]) == 1024  # Default dimension from server


def test_embeddings_with_custom_dimensions(tei_client):
    """Test embeddings with custom dimensions parameter"""
    response = tei_client.post("/v1/embeddings", json={"input": "Test text", "model": "openmockllm", "dimensions": 512})

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 512


def test_embeddings_encoding_format_float(tei_client):
    """Test embeddings with float encoding format (default)"""
    response = tei_client.post("/v1/embeddings", json={"input": "Test text", "model": "openmockllm", "encoding_format": "float"})

    assert response.status_code == 200
    data = response.json()

    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, list)
    assert all(isinstance(x, int | float) for x in embedding)


def test_embeddings_encoding_format_base64(tei_client):
    """Test embeddings with base64 encoding format"""
    response = tei_client.post("/v1/embeddings", json={"input": "Test text", "model": "openmockllm", "encoding_format": "base64"})

    assert response.status_code == 200
    data = response.json()

    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, str)
    # Verify it's valid base64
    try:
        decoded = base64.b64decode(embedding)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Invalid base64 encoding: {e}")


def test_embeddings_response_structure(tei_client):
    """Test that response follows OpenAI-compatible structure"""
    response = tei_client.post("/v1/embeddings", json={"input": ["Text 1", "Text 2"], "model": "openmockllm"})

    assert response.status_code == 200
    data = response.json()

    # Check top-level structure
    assert "object" in data
    assert "data" in data
    assert "model" in data
    assert "usage" in data

    # Check data array structure
    assert isinstance(data["data"], list)
    for item in data["data"]:
        assert "object" in item
        assert "embedding" in item
        assert "index" in item

    # Check usage structure
    assert "prompt_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


def test_embeddings_empty_batch_error(tei_client):
    """Test that empty input raises appropriate error"""
    response = tei_client.post("/v1/embeddings", json={"input": [], "model": "openmockllm"})

    assert response.status_code == 400  # Empty batch error
    data = response.json()
    assert "error" in data or "detail" in data


def test_embeddings_batch_size_limit(tei_client):
    """Test that exceeding max_client_batch_size raises error"""
    # Create a batch larger than max_client_batch_size (32)
    large_batch = [f"Text {i}" for i in range(35)]

    response = tei_client.post("/v1/embeddings", json={"input": large_batch, "model": "openmockllm"})

    assert response.status_code == 413  # Payload Too Large
    data = response.json()
    assert "error" in data or "message" in data


def test_embeddings_without_model(tei_client):
    """Test embeddings without specifying model (should use default)"""
    response = tei_client.post("/v1/embeddings", json={"input": "Test text"})

    assert response.status_code == 200
    data = response.json()

    # Should use default model from app state
    assert data["model"] == "openmockllm"
    assert len(data["data"]) == 1


def test_embeddings_default_dimensions(tei_client):
    """Test that default dimensions from app state are used"""
    response = tei_client.post("/v1/embeddings", json={"input": "Test text", "model": "openmockllm"})

    assert response.status_code == 200
    data = response.json()

    # Should use embedding_dimension from server default (1024)
    assert len(data["data"][0]["embedding"]) == 1024
