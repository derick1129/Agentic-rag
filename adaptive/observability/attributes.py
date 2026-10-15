"""Safe, stable attributes for telemetry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry controls with sensitive payloads disabled by default."""

    enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "adaptive-rag"
    redact_prompts: bool = True
    redact_documents: bool = True
    redact_credentials: bool = True
    capture_content: bool = False


_PROMPT_KEYS = ("prompt", "query", "completion", "input", "output", "answer")
_DOCUMENT_KEYS = ("document", "retrieved", "content", "context", "text")
_CREDENTIAL_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "credential",
)
_CAPTURE_CONTENT_KEYS = frozenset({"query", "answer", "retrieved_context"})


def query_hash(query: str) -> str:
    """Return a non-reversible correlation key for a query."""

    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def scope_id(tenant_id: str, acl: frozenset[str] | set[str] | tuple[str, ...] = frozenset()) -> str:
    """Return a stable tenant/ACL identifier without exposing scope values."""

    value = tenant_id + "\0" + ",".join(sorted(acl))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _is_sensitive(key: str, config: TelemetryConfig) -> bool:
    lowered = key.lower()
    if any(part in lowered for part in _CREDENTIAL_KEYS):
        return True
    if lowered.endswith(".hash") or lowered in {"query_hash", "scope_id"}:
        return False
    if config.capture_content and lowered in _CAPTURE_CONTENT_KEYS:
        return False
    return (config.redact_prompts and any(part in lowered for part in _PROMPT_KEYS)) or (
        config.redact_documents and any(part in lowered for part in _DOCUMENT_KEYS)
    )


def _primitive(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return json.dumps(value, sort_keys=True, default=str)


def normalize_attributes(
    attributes: dict[str, Any] | None, config: TelemetryConfig | None = None
) -> dict[str, Any]:
    """Filter sensitive keys and coerce values to OpenTelemetry-safe types."""

    config = config or TelemetryConfig()
    if not attributes:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in attributes.items():
        name = str(key)
        if _is_sensitive(name, config):
            continue
        if isinstance(value, (list, tuple, set)):
            normalized[name] = [_primitive(item) for item in value]
        else:
            normalized[name] = _primitive(value)
    return normalized
