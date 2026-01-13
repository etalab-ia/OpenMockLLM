from fastapi import APIRouter, Depends, Request

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.tei.schemas import EmbeddingModel, Info, ModelType, ModelType2

logger = init_logger(__name__)
router = APIRouter(tags=["Text Embeddings Inference"])


@router.get("/info", dependencies=[Depends(check_api_key)])
async def get_model_info(request: Request):
    """Get model information"""
    # Get config from app state
    model_name = getattr(request.app.state, "model_name", "openmockllm")
    max_client_batch_size = getattr(request.app.state, "max_client_batch_size", 32)
    max_batch_tokens = getattr(request.app.state, "max_batch_tokens", 16384)
    auto_truncate = getattr(request.app.state, "auto_truncate", False)

    # Create model_type for embedding model
    model_type = ModelType(root=ModelType2(embedding=EmbeddingModel(pooling="cls")))

    # Create info response with mock values
    info = Info(
        # Model info
        model_id=model_name,
        model_sha=None,
        model_dtype="float16",
        model_type=model_type,
        max_concurrent_requests=128,
        max_input_length=512,
        max_batch_tokens=max_batch_tokens,
        max_client_batch_size=max_client_batch_size,
        max_batch_requests=None,
        auto_truncate=auto_truncate,
        tokenization_workers=4,
        version="1.8.2",
        sha=None,
        docker_label=None,
    )
    return info
