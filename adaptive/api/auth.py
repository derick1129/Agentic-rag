"""Authentication utilities for the API."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from jwt.exceptions import (
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidTokenError,
)

from adaptive.config import Settings, get_settings
from adaptive.interfaces import RequestContext

logger = logging.getLogger(__name__)

_BEARER_HEADER_PARTS = 2
_DEV_PLACEHOLDER_TOKENS = frozenset({"dev-jwt"})


def _extract_bearer_token(request: Request, auth_mode: str) -> str | None:
    """Extract Bearer token from Authorization header if present."""
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header:
        return None

    parts = auth_header.strip().split()
    if len(parts) == _BEARER_HEADER_PARTS and parts[0].lower() == "bearer":
        return parts[1]

    if auth_mode == "production":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'.",
        )
    return None


def _parse_acl_claim(acl_claim: object, default_acl: frozenset[str]) -> frozenset[str]:
    """Parse ACL claim into a frozen set of strings."""
    if acl_claim is None:
        return default_acl
    if isinstance(acl_claim, (list, tuple, set, frozenset)):
        return frozenset(str(x).strip() for x in acl_claim if str(x).strip())
    if isinstance(acl_claim, str):
        return frozenset(x.strip() for x in acl_claim.split(",") if x.strip())
    return default_acl


def _authenticate_development(token: str | None, settings: Settings) -> RequestContext:
    """Authenticate request in development mode."""
    # If no token provided or static dev token, return dev defaults
    if not token or token in _DEV_PLACEHOLDER_TOKENS:
        return RequestContext(
            tenant_id=settings.auth_dev_tenant_id,
            subject_id=settings.auth_dev_subject_id,
            acl=settings.auth_dev_acl_set,
        )

    # If a JWT was passed, attempt to parse its claims (without requiring strict prod signatures)
    try:
        payload = jwt.decode(
            token,
            key=settings.auth_jwt_secret,
            algorithms=[settings.auth_jwt_algorithm],
            options={"verify_signature": False},
        )
        tenant_id = str(payload.get("tenant_id") or settings.auth_dev_tenant_id).strip()
        subject_id = str(
            payload.get("sub") or payload.get("subject_id") or settings.auth_dev_subject_id
        ).strip()
        acl = _parse_acl_claim(payload.get("acl"), settings.auth_dev_acl_set)
        return RequestContext(tenant_id=tenant_id, subject_id=subject_id, acl=acl)
    except Exception:
        # Fallback to static dev identity
        return RequestContext(
            tenant_id=settings.auth_dev_tenant_id,
            subject_id=settings.auth_dev_subject_id,
            acl=settings.auth_dev_acl_set,
        )


def _authenticate_production(token: str | None, settings: Settings) -> RequestContext:
    """Authenticate request in production mode using strict JWT validation."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    try:
        payload = jwt.decode(
            token,
            key=settings.auth_jwt_secret,
            algorithms=[settings.auth_jwt_algorithm],
            audience=settings.auth_jwt_audience,
            issuer=settings.auth_jwt_issuer,
            options={
                "require": ["exp", "iss", "aud", "sub"],
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_nbf": True,
            },
        )
    except ExpiredSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token has expired: {e}",
        ) from e
    except InvalidAudienceError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token audience: {e}",
        ) from e
    except InvalidIssuerError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token issuer: {e}",
        ) from e
    except ImmatureSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token not valid yet (nbf): {e}",
        ) from e
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        ) from e

    tenant_id = payload.get("tenant_id")
    if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required tenant_id claim",
        )

    subject_id = payload.get("sub")
    if not subject_id or not isinstance(subject_id, str) or not subject_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required sub claim",
        )

    acl = _parse_acl_claim(payload.get("acl"), frozenset())

    return RequestContext(
        tenant_id=tenant_id.strip(),
        subject_id=subject_id.strip(),
        acl=acl,
    )


_get_settings_dep: Any = Depends(get_settings)


def authenticate_request(
    request: Request,
    settings: Settings = _get_settings_dep,
) -> RequestContext:
    """Authenticate request and produce a frozen RequestContext.

    In development mode:
        Reads Authorization: Bearer <token> or uses dev defaults.
    In production mode:
        Validates JWT claims (iss, aud, exp, nbf, sub, tenant_id) using PyJWT.
    """
    if not isinstance(settings, Settings):
        settings = get_settings()

    token = _extract_bearer_token(request, settings.auth_mode)

    if settings.auth_mode == "development":
        return _authenticate_development(token, settings)
    if settings.auth_mode == "production":
        return _authenticate_production(token, settings)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unknown auth_mode: {settings.auth_mode}",
    )


async def get_request_context(
    x_tenant_id: str | None = Header(None),
    x_subject_id: str | None = Header(None),
    x_acl: str | None = Header(None),
    settings: Settings | None = None,
) -> RequestContext:
    """Extract request context from headers.

    Deprecated: Use authenticate_request instead. In production, logs a warning
    and raises a deprecation warning.
    """
    if settings is None:
        try:
            settings = get_settings()
        except Exception:
            settings = None

    if settings and settings.auth_mode == "production":
        logger.warning(
            "get_request_context is deprecated and insecure in production. "
            "Use authenticate_request instead."
        )
        warnings.warn(
            "get_request_context is deprecated in production; use authenticate_request instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    tenant_id = x_tenant_id or (settings.auth_dev_tenant_id if settings else "default")
    subject_id = x_subject_id or (settings.auth_dev_subject_id if settings else "anonymous")
    if x_acl is not None:
        acl = frozenset(a.strip() for a in x_acl.split(",") if a.strip())
    else:
        acl = settings.auth_dev_acl_set if settings else frozenset()

    return RequestContext(
        tenant_id=tenant_id,
        subject_id=subject_id,
        acl=acl,
    )


async def verify_api_key(x_api_key: str | None = Header(None)) -> None:
    """Verify API key if required."""
