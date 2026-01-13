def test_info_basic(tei_client):
    """Test basic info endpoint"""
    response = tei_client.get("/info")

    assert response.status_code == 200
    data = response.json()

    # Check basic structure
    assert data is not None
    assert "model_id" in data
    assert "version" in data


def test_info_model_fields(tei_client):
    """Test that all required model info fields are present"""
    response = tei_client.get("/info")

    assert response.status_code == 200
    data = response.json()

    # Model info
    assert data["model_id"] == "openmockllm"
    assert data["model_dtype"] == "float16"

    # Router parameters
    assert data["max_concurrent_requests"] == 128
    assert data["max_input_length"] == 512
    assert data["max_batch_tokens"] == 16384
    assert data["max_client_batch_size"] == 32
    assert data["auto_truncate"] is False
    assert data["tokenization_workers"] == 4

    # Version info
    assert data["version"] == "1.8.2"

    # Optional fields can be None
    assert "model_sha" in data
    assert "sha" in data
    assert "docker_label" in data
    assert "max_batch_requests" in data
