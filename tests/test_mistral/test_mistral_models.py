import pytest


@pytest.mark.asyncio
async def test_list_models(mistral_client):
    """Test listing available models"""
    response = mistral_client.models.list()

    assert response is not None
    assert response.data is not None
    assert len(response.data) > 0

    # Verify model structure
    model = response.data[0]
    assert model.id == "openmockllm"
    assert model.name == "openmockllm"
    assert model.max_context_length == 128000
    assert model.description is not None
    assert model.aliases is not None
    assert len(model.aliases) > 0
    assert model.default_model_temperature is not None
    assert model.capabilities is not None
