from typing import List, Optional, Union

from openmockllm.vllm.schemas.core import VllmBaseModel


class EmbeddingRequest(VllmBaseModel):
    # Required fields
    input: Union[str, List[str]]

    # Model selection
    model: Optional[str] = None

    # Encoding options
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None

    # User and priority
    user: Optional[str] = None
    priority: Optional[int] = 0

    # Token handling
    truncate_prompt_tokens: Optional[int] = None
    add_special_tokens: Optional[bool] = True


class EmbeddingData(VllmBaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(VllmBaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(VllmBaseModel):
    id: Optional[str] = None  # vLLM specific
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
