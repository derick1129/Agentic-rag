"""Tests for JWT authentication boundary and RequestContext generation."""

from datetime import datetime, timedelta, timezone
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
import jwt

from adaptive.api.auth import authenticate_request, get_request_context
from adaptive.config import Settings
from adaptive.interfaces import RequestContext

TEST_SECRET = "test-secret-key-1234567890-32bytes!"


def make_settings(
    auth_mode: str = "production",
    jwt_secret: str = TEST_SECRET,
    jwt_algorithm: str = "HS256",
    jwt_audience: str = "adaptive-rag",
    jwt_issuer: str = "adaptive-rag-idp",
    dev_tenant_id: str = "acme",
    dev_subject_id: str = "user-1",
    dev_acl: str = "support,admin",
) -> Settings:
    """Helper to create Settings instance for testing."""
    return Settings(
        DATABASE_URL="postgresql://localhost/test",
        AUTH_MODE=auth_mode,
        AUTH_JWT_SECRET=jwt_secret,
        AUTH_JWT_ALGORITHM=jwt_algorithm,
        AUTH_JWT_AUDIENCE=jwt_audience,
        AUTH_JWT_ISSUER=jwt_issuer,
        AUTH_DEV_TENANT_ID=dev_tenant_id,
        AUTH_DEV_SUBJECT_ID=dev_subject_id,
        AUTH_DEV_ACL=dev_acl,
    )


def make_request(headers: dict[str, str] | None = None, method: str = "POST", path: str = "/v1/queries") -> Request:
    """Helper to create a Starlette Request with given headers."""
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw_headers,
    }
    return Request(scope)


def create_jwt(
    payload: dict,
    secret: str = TEST_SECRET,
    algorithm: str = "HS256",
) -> str:
    """Helper to create a signed JWT."""
    return jwt.encode(payload, secret, algorithm=algorithm)


def test_development_returns_request_context():
    """Test dev mode returns RequestContext with dev defaults when Bearer dev-jwt is provided."""
    settings = make_settings(auth_mode="development")
    request = make_request(headers={"authorization": "Bearer dev-jwt"})

    context = authenticate_request(request, settings=settings)

    assert isinstance(context, RequestContext)
    assert context.tenant_id == "acme"
    assert context.subject_id == "user-1"
    assert context.acl == frozenset({"support", "admin"})


def test_development_without_token_returns_defaults():
    """Test dev mode returns RequestContext with dev defaults when no Authorization header is provided."""
    settings = make_settings(auth_mode="development")
    request = make_request()

    context = authenticate_request(request, settings=settings)

    assert isinstance(context, RequestContext)
    assert context.tenant_id == "acme"
    assert context.subject_id == "user-1"
    assert context.acl == frozenset({"support", "admin"})


def test_development_with_jwt_payload_extracts_claims():
    """Test dev mode parses payload claims if a structured JWT is provided."""
    settings = make_settings(auth_mode="development")
    token = create_jwt({
        "tenant_id": "custom-dev-tenant",
        "sub": "custom-dev-user",
        "acl": ["custom-role"],
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    context = authenticate_request(request, settings=settings)

    assert context.tenant_id == "custom-dev-tenant"
    assert context.subject_id == "custom-dev-user"
    assert context.acl == frozenset({"custom-role"})


def test_production_rejects_missing_authorization_header():
    """Test production mode rejects requests without Authorization header (401)."""
    settings = make_settings(auth_mode="production")
    request = make_request()

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_malformed_authorization_header():
    """Test production mode rejects malformed Authorization header (401)."""
    settings = make_settings(auth_mode="production")
    request = make_request(headers={"authorization": "NotBearer token"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_expired_token():
    """Test production mode rejects expired token (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "user-42",
        "tenant_id": "tenant-99",
        "exp": now - timedelta(hours=1),
        "nbf": now - timedelta(hours=2),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_invalid_signature():
    """Test production mode rejects token signed with wrong secret (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt(
        {
            "iss": settings.auth_jwt_issuer,
            "aud": settings.auth_jwt_audience,
            "sub": "user-42",
            "tenant_id": "tenant-99",
            "exp": now + timedelta(hours=1),
        },
        secret="completely-wrong-secret-key-32bytes!",
    )
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_malformed_token():
    """Test production mode rejects malformed/corrupted token string (401)."""
    settings = make_settings(auth_mode="production")
    request = make_request(headers={"authorization": "Bearer this.is.notavalidtoken"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_wrong_audience():
    """Test production mode rejects token with wrong audience (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": "wrong-audience",
        "sub": "user-42",
        "tenant_id": "tenant-99",
        "exp": now + timedelta(hours=1),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_wrong_issuer():
    """Test production mode rejects token with wrong issuer (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": "untrusted-issuer",
        "aud": settings.auth_jwt_audience,
        "sub": "user-42",
        "tenant_id": "tenant-99",
        "exp": now + timedelta(hours=1),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_missing_tenant_id():
    """Test production mode rejects token without tenant_id claim (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "user-42",
        "exp": now + timedelta(hours=1),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_missing_sub():
    """Test production mode rejects token without sub claim (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "tenant_id": "tenant-99",
        "exp": now + timedelta(hours=1),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_production_rejects_immature_token_nbf():
    """Test production mode rejects token with nbf in the future (401)."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "user-42",
        "tenant_id": "tenant-99",
        "exp": now + timedelta(hours=1),
        "nbf": now + timedelta(minutes=30),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc_info:
        authenticate_request(request, settings=settings)

    assert exc_info.value.status_code == 401


def test_valid_jwt_in_production_produces_request_context():
    """Test valid JWT in production successfully produces frozen RequestContext."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "user-valid-123",
        "tenant_id": "corp-tenant",
        "acl": ["billing", "engineering"],
        "exp": now + timedelta(hours=1),
        "nbf": now - timedelta(minutes=5),
    })
    request = make_request(headers={"authorization": f"Bearer {token}"})

    context = authenticate_request(request, settings=settings)

    assert isinstance(context, RequestContext)
    assert context.tenant_id == "corp-tenant"
    assert context.subject_id == "user-valid-123"
    assert context.acl == frozenset({"billing", "engineering"})

    # Ensure RequestContext is frozen
    with pytest.raises(Exception):
        context.tenant_id = "hacked"


def test_production_ignores_untrusted_header_injection():
    """Test production mode ignores x-tenant-id header injection and uses token claims strictly."""
    settings = make_settings(auth_mode="production")
    now = datetime.now(timezone.utc)
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "user-legit",
        "tenant_id": "tenant-legit",
        "exp": now + timedelta(hours=1),
    })
    # Untrusted user tries to inject malicious tenant in x-tenant-id header
    request = make_request(headers={
        "authorization": f"Bearer {token}",
        "x-tenant-id": "evil-tenant",
        "x-subject-id": "evil-user",
        "x-acl": "super-admin",
    })

    context = authenticate_request(request, settings=settings)

    assert context.tenant_id == "tenant-legit"
    assert context.subject_id == "user-legit"
    assert "super-admin" not in context.acl


def test_fastapi_dependency_integration():
    """Test authenticate_request used directly as a FastAPI dependency in an app."""
    settings = make_settings(auth_mode="production")
    app = FastAPI()

    def auth_dep(request: Request) -> RequestContext:
        return authenticate_request(request, settings=settings)

    @app.get("/test-endpoint")
    def endpoint(context: RequestContext = Depends(auth_dep)):
        return {"tenant_id": context.tenant_id, "subject_id": context.subject_id}

    client = TestClient(app)

    # 1. Without header -> 401
    resp = client.get("/test-endpoint")
    assert resp.status_code == 401

    # 2. With valid token -> 200
    token = create_jwt({
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "test-sub",
        "tenant_id": "test-ten",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    })
    resp = client.get("/test-endpoint", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == {"tenant_id": "test-ten", "subject_id": "test-sub"}


@pytest.mark.asyncio
async def test_get_request_context_deprecated_in_production(caplog):
    """Test get_request_context emits a warning in production mode."""
    settings = make_settings(auth_mode="production")

    with pytest.deprecated_call():
        ctx = await get_request_context(
            x_tenant_id="custom-tenant",
            x_subject_id="custom-sub",
            x_acl="read,write",
            settings=settings,
        )

    assert ctx.tenant_id == "custom-tenant"
    assert "deprecated" in caplog.text.lower()
