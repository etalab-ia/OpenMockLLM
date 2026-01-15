def test_list_models(vllm_client):
    """Test listing available models"""
    response = vllm_client.models.list()

    assert response is not None
    assert response.object == "list"
    assert response.data is not None
    assert len(response.data) > 0

    # Check first model
    model = response.data[0]
    assert model.id is not None
    assert model.object == "model"
    assert model.created is not None
    assert model.owned_by is not None
