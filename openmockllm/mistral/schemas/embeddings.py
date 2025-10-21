from pydantic import BaseModel
from typing import List, Union


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
