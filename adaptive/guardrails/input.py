"""Validation and prompt-injection checks at the query boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from adaptive.interfaces import GuardrailAction, GuardrailResult, RequestContext


class InputRequest(BaseModel):
    """User input plus server-derived identity; identity is never read from query text."""

    query: str
    context: RequestContext
    payload: dict[str, Any] = Field(default_factory=dict)


_INJECTION_MARKERS = (
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"reveal\s+(?:the\s+)?(?:system|hidden)\s+prompt",
    r"disregard\s+(?:your\s+)?(?:instructions|policy)",
    r"you\s+are\s+now\s+(?:the\s+)?system",
    r"print\s+(?:your\s+)?(?:secrets|credentials)",
)


def _result(
    action: GuardrailAction,
    code: str,
    message: str,
    **details: int | str,
) -> GuardrailResult:
    safe_details: dict[str, Any] = {"code": code, **details}
    return GuardrailResult(
        action=action,
        reason=message,
        details=safe_details,
        span_attributes={
            "guardrail.name": "input",
            "guardrail.status": action.value,
            "guardrail.action": action.value,
            "guardrail.code": code,
        },
    )


def coerce_request(request: InputRequest | Mapping[str, Any]) -> InputRequest | None:
    if isinstance(request, InputRequest):
        return request
    if isinstance(request, Mapping):
        try:
            return InputRequest.model_validate(request)
        except Exception:
            return None
    return None


def _validate_payload(payload: dict[str, Any], max_payload_bytes: int) -> GuardrailResult | None:
    try:
        payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return _result(
            GuardrailAction.DENY, "invalid_payload", "Request payload is not serializable."
        )
    if payload_bytes > max_payload_bytes:
        return _result(
            GuardrailAction.DENY,
            "payload_too_large",
            "Request payload exceeds the configured size limit.",
            limit=max_payload_bytes,
        )
    return None


def check_input(
    request: InputRequest | Mapping[str, Any],
    *,
    max_query_chars: int = 8_000,
    max_payload_bytes: int = 64_000,
) -> GuardrailResult:
    parsed = coerce_request(request)
    if parsed is None:
        return _result(
            GuardrailAction.DENY, "invalid_request", "Request identity or shape is invalid."
        )
    if not parsed.context.tenant_id.strip() or not parsed.context.subject_id.strip():
        return _result(
            GuardrailAction.DENY, "invalid_identity", "Authenticated tenant identity is required."
        )
    query = parsed.query.strip()
    if not query:
        return _result(GuardrailAction.DENY, "empty_query", "A non-empty query is required.")
    if len(query) > max_query_chars:
        return _result(
            GuardrailAction.DENY,
            "query_too_large",
            "Query exceeds the configured size limit.",
            limit=max_query_chars,
        )
    payload_error = _validate_payload(parsed.payload, max_payload_bytes)
    if payload_error is not None:
        return payload_error
    lowered = query.casefold()
    matched = next((marker for marker in _INJECTION_MARKERS if re.search(marker, lowered)), None)
    return (
        _result(
            GuardrailAction.DENY,
            "prompt_injection",
            "Prompt-injection language is not permitted.",
            marker=matched,
        )
        if matched
        else _result(GuardrailAction.ALLOW, "input_valid", "Input passed validation.")
    )
