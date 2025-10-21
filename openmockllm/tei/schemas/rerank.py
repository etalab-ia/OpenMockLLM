from typing import List, Optional

from openmockllm.tei.schemas.core import TeiBaseModel, TruncationDirection


class RerankRequest(TeiBaseModel):
    """Rerank request"""

    query: str
    texts: List[str]
    raw_scores: bool = False
    return_text: bool = False
    truncate: Optional[bool] = False
    truncation_direction: TruncationDirection = TruncationDirection.RIGHT


class Rank(TeiBaseModel):
    """Rank object in rerank response"""

    index: int
    score: float
    text: Optional[str] = None


# Response is a list of Rank objects
RerankResponse = List[Rank]
