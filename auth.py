"""Authentication utilities for Qdrant Proxy admin API.

Provides dependency injection compatible auth functions for FastAPI.
"""

import base64
import logging
from typing import Optional

from config import settings
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


def verify_admin_auth(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin API key from Authorization header.

    Supports both Bearer token and Basic auth formats:
    - Bearer token: Authorization: Bearer <QDRANT_PROXY_ADMIN_KEY>
    - Basic auth: Authorization: Basic base64(admin:<QDRANT_PROXY_ADMIN_KEY>)

    Usage as FastAPI dependency:
        @app.get("/admin/endpoint")
        async def admin_endpoint(_: bool = Depends(verify_admin_auth)):
            ...

    Raises:
        HTTPException(503): If admin key not configured
        HTTPException(401): If authorization fails
    """
    if not settings.qdrant_proxy_admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API not configured. Set QDRANT_PROXY_ADMIN_KEY environment variable.",
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Try Bearer token
    if authorization.startswith("Bearer "):
        token = authorization[7:]
        if token == settings.qdrant_proxy_admin_key:
            return True

    # Try Basic auth
    if authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization[6:]).decode("utf-8")
            _, password = decoded.split(":", 1)
            if password == settings.qdrant_proxy_admin_key:
                return True
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid admin API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_admin_key_param(key: str) -> bool:
    """Verify admin API key provided as query/path parameter.

    Used for endpoints that need key verification via query param.

    Args:
        key: The API key to verify

    Returns:
        True if key is valid

    Raises:
        HTTPException(401): If key is invalid
    """
    if key != settings.qdrant_proxy_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
    return True
