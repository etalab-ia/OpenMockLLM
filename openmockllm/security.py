from typing import Annotated, Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from openmockllm.exceptions import InvalidAPIKeyException, InvalidAuthenticationSchemeException
from openmockllm.settings import settings

auth_scheme = HTTPBearer(scheme_name="API key", auto_error=False)


def check_api_key(
    api_key: Annotated[Optional[HTTPAuthorizationCredentials], Depends(auth_scheme)] = None,
):
    """Check API key if configured, otherwise allow access"""
    if not settings.api_key:
        return None

    if not api_key:
        raise InvalidAPIKeyException(detail="API key required but not provided")

    if api_key.scheme != "Bearer":
        raise InvalidAuthenticationSchemeException()

    if api_key.credentials != settings.api_key:
        raise InvalidAPIKeyException()

    return api_key.credentials
