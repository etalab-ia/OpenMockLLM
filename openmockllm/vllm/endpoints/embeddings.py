from fastapi import APIRouter, Depends, Request

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.vllm.exceptions import NotFoundError
from openmockllm.vllm.schemas.embeddings import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from openmockllm.vllm.utils.embeddings import generate_mock_embedding

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: Request, body: EmbeddingRequest):
    """Create embeddings for the input"""
    # Use the model from the request or fall back to the default
    model = body.model or request.app.state.model_name

    # Validate model if specified
    if body.model and body.model != request.app.state.model_name:
        raise NotFoundError(f"The model `{body.model}` does not exist.")

    # Handle both single string and list of strings
    if isinstance(body.input, str):
        inputs = [body.input]
    else:
        inputs = body.input

    # Use dimensions from request or fall back to default
    dimensions = body.dimensions or request.app.state.embedding_dimension

    # Generate mock embeddings
    embeddings_data = [
        EmbeddingData(object="embedding", index=i, embedding=generate_mock_embedding(dimension=dimensions)) for i in range(len(inputs))
    ]

    response = EmbeddingResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )
    return response
