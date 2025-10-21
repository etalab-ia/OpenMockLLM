from fastapi import APIRouter, Response

from openmockllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter(tags=["Text Embeddings Inference"])


@router.get("/health")
async def health():
    """Health check endpoint"""
    logger.debug("Health check")
    return Response(status_code=200)
