from pydantic import BaseModel
from typing import List, Union


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: str = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
