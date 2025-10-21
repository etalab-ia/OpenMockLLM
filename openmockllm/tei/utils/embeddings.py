import base64
from random import random
import struct


def generate_mock_embedding(dimension: int = 1024, encoding_format: str = "float") -> list[float] | str:
    """
    Generate a mock embedding vector

    Args:
        dimension: The dimension of the embedding vector
        encoding_format: Either "float" or "base64"

    Returns:
        List of floats or base64 encoded string
    """
    # Generate random floats
    embedding = [random() for _ in range(dimension)]

    if encoding_format == "base64":
        # Convert floats to base64 encoded string
        # Pack as little-endian floats
        packed = struct.pack(f"<{dimension}f", *embedding)
        return base64.b64encode(packed).decode("utf-8")

    return embedding
