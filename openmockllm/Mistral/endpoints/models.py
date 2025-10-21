import time

from fastapi import APIRouter, Request

from openmockllm.logging import init_logger
from openmockllm.mistral.schemas.models import Model, ModelsResponse

logger = init_logger(__name__)
router = APIRouter()


@router.get("")
async def list_models(request: Request):
    """List available models"""
    # Get config from app state
    model_name = getattr(request.app.state, "model_name", "openmockllm")
    owned_by = getattr(request.app.state, "owned_by", "OpenMockLLM")

    models = ModelsResponse(
        object="list", data=[Model(id=model_name, object="model", created=int(time.time()), owned_by=owned_by)]
    )
    return models
