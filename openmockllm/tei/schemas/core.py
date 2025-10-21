from enum import Enum

from pydantic import BaseModel, ConfigDict


class TeiBaseModel(BaseModel):
    """Base model for TEI schemas"""

    model_config = ConfigDict(extra="allow")


class ErrorType(str, Enum):
    """Error types for TEI API"""

    UNHEALTHY = "Unhealthy"
    BACKEND = "Backend"
    OVERLOADED = "Overloaded"
    VALIDATION = "Validation"
    TOKENIZER = "Tokenizer"
    EMPTY = "Empty"


class TruncationDirection(str, Enum):
    """Truncation direction"""

    LEFT = "Left"
    RIGHT = "Right"
