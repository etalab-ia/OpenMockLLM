import time

from fastapi import APIRouter, Depends, Request

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key
from openmockllm.vllm.schemas.models import Model, ModelsResponse

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", dependencies=[Depends(check_api_key)])
async def list_models(request: Request):
    """List available models"""
    # Get config from app state
    model_name = getattr(request.app.state, "model_name", "openmockllm")
    owned_by = getattr(request.app.state, "owned_by", "OpenMockLLM")

    models = ModelsResponse(
        object="list",
        data=[
            Model(
                id=model_name,
                object="model",
                created=int(time.time()),
                owned_by=owned_by,
            )
        ],
    )
    return models
