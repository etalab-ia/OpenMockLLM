import time

from fastapi import APIRouter, Depends, Request
from mistralai.models import BaseModelCard, ModelCapabilities, ModelList

from openmockllm.logger import init_logger
from openmockllm.security import check_api_key

logger = init_logger(__name__)
router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", dependencies=[Depends(check_api_key)])
async def list_models(request: Request):
    response = ModelList(
        data=[
            BaseModelCard(
                id=request.app.state.model_name,
                created=int(time.time()),
                name=request.app.state.model_name,
                description="Lorem ipsum dolor sit amet.",
                max_context_length=request.app.state.max_context,
                aliases=[f"{request.app.state.model_name}-latest"],
                deprecation=None,
                deprecation_replacement_model=None,
                default_model_temperature=0.3,
                capabilities=ModelCapabilities(),
            )
        ]
    )
    return response
