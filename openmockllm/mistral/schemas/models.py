from typing import List

from openmockllm.vllm.schemas.core import VllmBaseModel


class Model(VllmBaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(VllmBaseModel):
    object: str = "list"
    data: List[Model]
