from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openmockllm.logger import init_logger

logger = init_logger(__name__)


class ErrorResponse(BaseModel):
    """Error response schema matching TEI API"""

    error: str
    error_type: str


class TEIException(HTTPException):
    """Base exception for TEI API errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_type = error_type


class EmptyBatchError(TEIException):
    """400 Bad Request - Batch is empty"""

    def __init__(self, message: str = "Batch is empty"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_type="empty",
        )


class ValidationError(TEIException):
    """400/413 Bad Request - Validation error"""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            message=message,
            error_type="validation",
        )


class TokenizerError(TEIException):
    """422 Unprocessable Entity - Tokenization error"""

    def __init__(self, message: str = "Tokenization error"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            error_type="tokenizer",
        )


class BackendError(TEIException):
    """424 Failed Dependency - Backend/Inference error"""

    def __init__(self, message: str = "Inference failed"):
        super().__init__(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            message=message,
            error_type="backend",
        )


class OverloadedError(TEIException):
    """429 Too Many Requests - Model is overloaded"""

    def __init__(self, message: str = "Model is overloaded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=message,
            error_type="overloaded",
        )


class UnhealthyError(TEIException):
    """503 Service Unavailable - Service is unhealthy"""

    def __init__(self, message: str = "unhealthy"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
            error_type="unhealthy",
        )


async def tei_exception_handler(request: Request, exc: TEIException) -> JSONResponse:
    """Handle TEI exceptions and return proper error response"""
    logger.error(f"TEIException: {exc.error_type} - {exc.detail} (status: {exc.status_code})")

    error_response = ErrorResponse(
        error=exc.detail,
        error_type=exc.error_type,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {type(exc).__name__} - {str(exc)}")

    error_response = ErrorResponse(
        error=str(exc),
        error_type="backend",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )
