from enum import Enum
from typing import List, Optional, Union

from openmockllm.tei.schemas.core import TeiBaseModel


class EncodingFormat(str, Enum):
    """Encoding format for embeddings"""

    FLOAT = "float"
    BASE64 = "base64"


# Input types based on TEI OpenAPI schema
InputType = Union[str, List[int]]
Input = Union[InputType, List[InputType]]

# Embedding can be array of floats or base64 encoded string
Embedding = Union[List[float], str]


class OpenAICompatRequest(TeiBaseModel):
    """OpenAI compatible embedding request"""

    input: Input
    model: Optional[str] = None
    encoding_format: EncodingFormat = EncodingFormat.FLOAT
    dimensions: Optional[int] = None
    user: Optional[str] = None


class OpenAICompatEmbedding(TeiBaseModel):
    """OpenAI compatible embedding object"""

    object: str = "embedding"
    embedding: Embedding
    index: int


class OpenAICompatUsage(TeiBaseModel):
    """OpenAI compatible usage information"""

    prompt_tokens: int
    total_tokens: int


class OpenAICompatResponse(TeiBaseModel):
    """OpenAI compatible embedding response"""

    object: str = "list"
    data: List[OpenAICompatEmbedding]
    model: str
    usage: OpenAICompatUsage


class OpenAICompatErrorResponse(TeiBaseModel):
    """OpenAI compatible error response"""

    message: str
    code: int
    error_type: str
