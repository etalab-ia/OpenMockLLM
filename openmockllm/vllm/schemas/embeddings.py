from openmockllm.vllm.schemas.core import VllmBaseModel


class EmbeddingData(VllmBaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingUsage(VllmBaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(VllmBaseModel):
    id: str | None = None  # vLLM specific
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
