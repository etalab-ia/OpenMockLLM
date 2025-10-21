from random import random


def generate_mock_embedding(dimension: int = 1536) -> list[float]:
    """Generate a mock embedding vector"""
    return [random.random() for _ in range(dimension)]
