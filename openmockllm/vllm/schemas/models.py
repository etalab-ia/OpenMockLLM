from pydantic import BaseModel


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    max_model_len: int


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[Model]
