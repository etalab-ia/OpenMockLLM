from random import random


def generate_mock_rerank_scores(num_texts: int, query: str = "") -> list[tuple[int, float]]:
    """
    Generate mock reranking scores for a list of texts

    Args:
        num_texts: Number of texts to generate scores for
        query: The query string (used for seed but not actually processed)

    Returns:
        List of (index, score) tuples sorted by score in descending order
    """
    # Generate random scores for each text
    scores = [(i, random()) for i in range(num_texts)]

    # Sort by score in descending order (highest relevance first)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
