from fastapi import APIRouter, Depends, Request

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.tei.exceptions import EmptyBatchError, ValidationError
from openmockllm.tei.schemas import Rank, RerankRequest, RerankResponse
from openmockllm.tei.utils.rerank import generate_mock_rerank_scores

logger = init_logger(__name__)
router = APIRouter(tags=["Text Embeddings Inference"])


@router.post("/rerank", dependencies=[Depends(check_api_key)])
async def rerank(request: Request, body: RerankRequest):
    """Rerank texts based on query relevance"""
    # Validate inputs
    if not body.texts or len(body.texts) == 0:
        raise EmptyBatchError("Batch is empty")

    # Validate batch size
    max_batch_size = getattr(request.app.state, "max_client_batch_size", 32)
    if len(body.texts) > max_batch_size:
        raise ValidationError(f"Batch size {len(body.texts)} exceeds maximum {max_batch_size}", status_code=413)

    # Generate mock reranking scores
    ranked_results = generate_mock_rerank_scores(len(body.texts), body.query)

    # Create response
    response: RerankResponse = [
        Rank(
            index=idx,
            score=score,
            text=body.texts[idx] if body.return_text else None,
        )
        for idx, score in ranked_results
    ]

    return response
