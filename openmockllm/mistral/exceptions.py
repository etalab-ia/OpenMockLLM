from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openmockllm.logging import init_logger

logger = init_logger(__name__)


class ErrorResponse(BaseModel):
    """Error response schema matching mistral API"""

    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: int


class MistralException(HTTPException):
    """Base exception for mistral API errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        param: str | None = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_type = error_type
        self.param = param


class BadRequestError(MistralException):
    """400 Bad Request"""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            error_type="BadRequestError",
            param=param,
        )


class NotFoundError(MistralException):
    """404 Not Found"""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            error_type="NotFoundError",
            param=param,
        )


class InternalServerError(MistralException):
    """500 Internal Server Error"""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_type="InternalServerError",
            param=param,
        )


async def mistral_exception_handler(request: Request, exc: MistralException) -> JSONResponse:
    """Handle mistral exceptions and return proper error response"""
    logger.error(f"MistralException: {exc.error_type} - {exc.detail} (status: {exc.status_code})")

    error_response = ErrorResponse(
        message=exc.detail,
        type=exc.error_type,
        code=exc.status_code,
        param=exc.param,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {type(exc).__name__} - {str(exc)}")

    error_response = ErrorResponse(
        message=str(exc),
        type="InternalServerError",
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )
