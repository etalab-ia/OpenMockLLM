import random

from fastapi import APIRouter, Request

from openmockllm.logging import init_logger
from openmockllm.vllm.schemas.embeddings import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

logger = init_logger(__name__)
router = APIRouter()


def generate_mock_embedding(dimension: int = 1536) -> list[float]:
    """Generate a mock embedding vector"""
    return [random.random() for _ in range(dimension)]


@router.post("")
async def create_embeddings(embedding_request: EmbeddingRequest, request: Request):
    """Create embeddings for the input"""
    model = embedding_request.model

    # Handle both single string and list of strings
    if isinstance(embedding_request.input, str):
        inputs = [embedding_request.input]
    else:
        inputs = embedding_request.input

    # Generate mock embeddings
    embeddings_data = [
        EmbeddingData(object="embedding", index=i, embedding=generate_mock_embedding()) for i in range(len(inputs))
    ]

    response = EmbeddingResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )
    return response
