from typing import Annotated, Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from openmockllm.exceptions import InvalidAPIKeyException, InvalidAuthenticationSchemeException


class Settings:
    """Simple settings class to store API key configuration"""

    api_key: Optional[str] = None


settings = Settings()
auth_scheme = HTTPBearer(scheme_name="API key", auto_error=False)


def check_api_key(
    api_key: Annotated[Optional[HTTPAuthorizationCredentials], Depends(auth_scheme)] = None,
):
    """Check API key if configured, otherwise allow access"""
    # If no API key is configured in settings, allow access
    if not settings.api_key:
        return None

    # If API key is configured but not provided in request
    if not api_key:
        raise InvalidAPIKeyException(detail="API key required but not provided")

    # Check authentication scheme
    if api_key.scheme != "Bearer":
        raise InvalidAuthenticationSchemeException()

    # Validate the API key
    if api_key.credentials != settings.api_key:
        raise InvalidAPIKeyException()

    return api_key.credentials
