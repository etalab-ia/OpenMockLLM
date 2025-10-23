from enum import Enum

from openmockllm.tei.schemas.core import TeiBaseModel


class EncodingFormat(str, Enum):
    """Encoding format for embeddings"""

    FLOAT = "float"
    BASE64 = "base64"


# Input types based on TEI OpenAPI schema
InputType = str | list[int]
Input = InputType | list[InputType]

# Embedding can be array of floats or base64 encoded string
Embedding = list[float] | str


class OpenAICompatRequest(TeiBaseModel):
    """OpenAI compatible embedding request"""

    input: Input
    model: str | None = None
    encoding_format: EncodingFormat = EncodingFormat.FLOAT
    dimensions: int | None = None
    user: str | None = None


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
    data: list[OpenAICompatEmbedding]
    model: str
    usage: OpenAICompatUsage


class OpenAICompatErrorResponse(TeiBaseModel):
    """OpenAI compatible error response"""

    message: str
    code: int
    error_type: str
