from openmockllm.vllm.schemas.core import VllmBaseModel


class EmbeddingRequest(VllmBaseModel):
    # Required fields
    input: str | list[str]

    # Model selection
    model: str | None = None

    # Encoding options
    encoding_format: str | None = "float"
    dimensions: int | None = None

    # User and priority
    user: str | None = None
    priority: int | None = 0

    # Token handling
    truncate_prompt_tokens: int | None = None
    add_special_tokens: bool | None = True


class EmbeddingData(VllmBaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(VllmBaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(VllmBaseModel):
    id: str | None = None  # vLLM specific
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
