import random
import uuid

from fastapi import APIRouter, Depends, Request

from openmockllm.logging import init_logger
from openmockllm.mistral.schemas.embeddings import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from openmockllm.security import check_api_key

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["embeddings"])


def generate_mock_embedding(dimension: int = 1024) -> list[float]:
    """Generate a mock embedding vector (mistral uses 1024 dimensions)"""
    return [random.random() for _ in range(dimension)]


@router.post("/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(embedding_request: EmbeddingRequest, request: Request):
    """Create embeddings for the input"""
    model = embedding_request.model

    # Handle both single string and list of strings
    if isinstance(embedding_request.input, str):
        inputs = [embedding_request.input]
    else:
        inputs = embedding_request.input

    # Generate mock embeddings
    embeddings_data = [EmbeddingData(object="embedding", embedding=generate_mock_embedding(), index=i) for i in range(len(inputs))]

    response = EmbeddingResponse(
        id=f"embd-{uuid.uuid4().hex}",
        object="list",
        data=embeddings_data,
        model=model,
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )
    return response
