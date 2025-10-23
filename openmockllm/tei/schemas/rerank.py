from openmockllm.tei.schemas.core import TeiBaseModel, TruncationDirection


class RerankRequest(TeiBaseModel):
    """Rerank request"""

    query: str
    texts: list[str]
    raw_scores: bool = False
    return_text: bool = False
    truncate: bool | None = False
    truncation_direction: TruncationDirection = TruncationDirection.RIGHT


class Rank(TeiBaseModel):
    """Rank object in rerank response"""

    index: int
    score: float
    text: str | None = None


# Response is a list of Rank objects
RerankResponse = list[Rank]
