from fastapi import APIRouter, Depends, Request

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.tei.exceptions import EmptyBatchError, ValidationError
from openmockllm.tei.schemas.embeddings import (
    EncodingFormat,
    OpenAICompatEmbedding,
    OpenAICompatRequest,
    OpenAICompatResponse,
    OpenAICompatUsage,
)
from openmockllm.tei.utils.embeddings import generate_mock_embedding

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["Text Embeddings Inference"])


@router.post("/embeddings", dependencies=[Depends(check_api_key)])
async def openai_embed(request: Request, body: OpenAICompatRequest):
    """OpenAI compatible embeddings endpoint"""
    # Use the model from the request or fall back to the default
    model = body.model or request.app.state.model_name

    # Handle both single string and list of strings/integers
    if isinstance(body.input, str):
        inputs = [body.input]
    elif isinstance(body.input, list):
        if len(body.input) == 0:
            raise EmptyBatchError("Batch is empty")
        inputs = body.input
    else:
        inputs = [body.input]

    # Validate batch size
    max_batch_size = getattr(request.app.state, "max_client_batch_size", 32)
    if len(inputs) > max_batch_size:
        raise ValidationError(f"Batch size {len(inputs)} exceeds maximum {max_batch_size}", status_code=413)

    # Use dimensions from request or fall back to default
    dimensions = body.dimensions or getattr(request.app.state, "embedding_dimension", 1024)

    # Get encoding format
    encoding_format = body.encoding_format.value if isinstance(body.encoding_format, EncodingFormat) else body.encoding_format

    # Generate mock embeddings
    embeddings_data = [
        OpenAICompatEmbedding(
            object="embedding",
            index=i,
            embedding=generate_mock_embedding(dimension=dimensions, encoding_format=encoding_format),
        )
        for i in range(len(inputs))
    ]

    response = OpenAICompatResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=OpenAICompatUsage(prompt_tokens=0, total_tokens=0),
    )
    return response
