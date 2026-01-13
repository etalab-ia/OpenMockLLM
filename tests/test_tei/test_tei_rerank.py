def test_rerank_basic(tei_client):
    """Test basic reranking functionality"""
    response = tei_client.post(
        "/rerank",
        json={
            "query": "What is Deep Learning?",
            "texts": [
                "Deep Learning is a subset of machine learning",
                "Python is a programming language",
                "Neural networks are inspired by the human brain",
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 3

    # Check each rank object
    for rank in data:
        assert "index" in rank
        assert "score" in rank
        assert isinstance(rank["index"], int)
        assert isinstance(rank["score"], int | float)
        assert 0 <= rank["index"] < 3


def test_rerank_with_return_text(tei_client):
    """Test reranking with return_text=True"""
    texts = ["Deep Learning is a subset of machine learning", "Python is a programming language", "Neural networks are inspired by the human brain"]

    response = tei_client.post("/rerank", json={"query": "What is Deep Learning?", "texts": texts, "return_text": True})

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 3

    # Each result should include the text
    for rank in data:
        assert "text" in rank
        assert rank["text"] is not None
        assert rank["text"] in texts


def test_rerank_without_return_text(tei_client):
    """Test reranking with return_text=False (default)"""
    response = tei_client.post(
        "/rerank",
        json={
            "query": "What is Deep Learning?",
            "texts": ["Deep Learning is a subset of machine learning", "Python is a programming language"],
            "return_text": False,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Text should be null when return_text is False
    for rank in data:
        assert "text" in rank
        assert rank["text"] is None


def test_rerank_response_structure(tei_client):
    """Test that rerank response has correct structure"""
    response = tei_client.post(
        "/rerank",
        json={
            "query": "Machine learning query",
            "texts": ["Text about machine learning", "Text about cooking", "Text about sports"],
            "return_text": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)

    for rank in data:
        # Check required fields
        assert "index" in rank
        assert "score" in rank

        # Check types
        assert isinstance(rank["index"], int)
        assert isinstance(rank["score"], int | float)
        assert rank["index"] >= 0

        # Score should be a reasonable value
        assert -1.0 <= rank["score"] <= 1.0 or 0.0 <= rank["score"] <= 1.0


def test_rerank_scores_descending(tei_client):
    """Test that rerank scores are in descending order"""
    response = tei_client.post(
        "/rerank",
        json={
            "query": "What is Deep Learning?",
            "texts": [
                "Deep Learning is a subset of machine learning",
                "Python is a programming language",
                "Neural networks are inspired by the human brain",
                "Cooking is an art",
                "Sports are fun",
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Scores should be in descending order
    scores = [rank["score"] for rank in data]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"


def test_rerank_empty_texts_error(tei_client):
    """Test that empty texts list raises error"""
    response = tei_client.post("/rerank", json={"query": "What is Deep Learning?", "texts": []})

    assert response.status_code == 400  # Empty batch error
    data = response.json()
    assert "error" in data or "detail" in data


def test_rerank_batch_size_limit(tei_client):
    """Test that exceeding max_client_batch_size raises error"""
    # Create a batch larger than max_client_batch_size (32)
    large_batch = [f"Text {i} about various topics" for i in range(35)]

    response = tei_client.post("/rerank", json={"query": "What is Deep Learning?", "texts": large_batch})

    assert response.status_code == 413  # Payload Too Large
    data = response.json()
    assert "error" in data or "message" in data


def test_rerank_indices_match_input(tei_client):
    """Test that returned indices correspond to input text positions"""
    texts = ["First text about AI", "Second text about cooking", "Third text about sports"]

    response = tei_client.post("/rerank", json={"query": "Artificial Intelligence", "texts": texts, "return_text": True})

    assert response.status_code == 200
    data = response.json()

    # Verify that each index points to the correct original text
    for rank in data:
        index = rank["index"]
        text = rank["text"]
        assert text == texts[index], f"Index {index} should correspond to text: {texts[index]}"


def test_rerank_with_single_text(tei_client):
    """Test reranking with only one text"""
    response = tei_client.post("/rerank", json={"query": "What is Deep Learning?", "texts": ["Deep Learning is a subset of machine learning"]})

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["index"] == 0
    assert "score" in data[0]


def test_rerank_with_optional_parameters(tei_client):
    """Test reranking with optional parameters"""
    response = tei_client.post(
        "/rerank",
        json={
            "query": "Machine learning",
            "texts": [
                "Text about machine learning",
                "Text about cooking",
            ],
            "raw_scores": False,
            "return_text": True,
            "truncate": False,
            "truncation_direction": "Right",
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()

    assert len(data) == 2
    for rank in data:
        assert "index" in rank
        assert "score" in rank
        assert "text" in rank
