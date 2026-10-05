"""
JWT Authentication & Authorization.

Architecture:
  Tokens are self-signed HS256 JWTs containing:
    - sub:        user_id
    - tenant_id:  the tenant this user belongs to (authoritative — never trust request body)
    - exp:        expiry timestamp
    - iat:        issued-at timestamp

Security properties:
  - tenant_id comes from the VERIFIED JWT payload, not the request body.
    A malicious client cannot escalate to another tenant by sending a
    different tenant_id in the JSON body — it will be ignored.
  - Expiry is enforced by python-jose; clock skew tolerance is zero (strict).
  - Tokens are HS256 (symmetric) — upgrade to RS256 (asymmetric) when
    adding an external identity provider (Auth0, Cognito, etc.).

Usage:
  # In a FastAPI route:
  current_user: dict = Depends(get_current_user)
  tenant_id = current_user["tenant_id"]

  # Generate a token (dev/testing endpoint):
  token = create_access_token(user_id="usr_99823", tenant_id="corp_acme_01")
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from config import get_settings

logger = logging.getLogger(__name__)

# FastAPI HTTPBearer extracts the raw JWT from the Authorization: Bearer <token> header
_http_bearer = HTTPBearer(auto_error=True)


# ── Token Generation ──────────────────────────────────────────────────────────

def create_access_token(user_id: str, tenant_id: str) -> str:
    """
    Mint a signed JWT access token.

    Args:
        user_id:    Unique user identifier.
        tenant_id:  Tenant this user belongs to (embedded in token — authoritative).

    Returns:
        Signed JWT string.
    """
    settings = get_settings()
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload: Dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": expire,
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    logger.debug(
        "[Auth] Token created | user=%s | tenant=%s | expires=%s",
        user_id,
        tenant_id,
        expire.isoformat(),
    )
    return token


# ── Token Validation (FastAPI Dependency) ─────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_http_bearer),
) -> Dict[str, str]:
    """
    FastAPI dependency that validates the Bearer JWT and extracts user context.

    Returns:
        {"user_id": str, "tenant_id": str}

    Raises:
        401 Unauthorized — token missing, malformed, expired, or tampered.
    """
    settings = get_settings()
    token = credentials.credentials

    _credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials — token is invalid or expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError:
        logger.warning("[Auth] Rejected expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        logger.warning("[Auth] JWT validation failed: %s", exc)
        raise _credentials_error

    user_id: str | None = payload.get("sub")
    tenant_id: str | None = payload.get("tenant_id")

    if not user_id or not tenant_id:
        logger.warning("[Auth] Token missing required claims: %s", list(payload.keys()))
        raise _credentials_error

    return {"user_id": user_id, "tenant_id": tenant_id}
