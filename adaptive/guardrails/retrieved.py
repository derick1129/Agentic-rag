"""Treat retrieved documents and tool output as untrusted observations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from adaptive.interfaces import Chunk, GuardrailAction, GuardrailResult, RequestContext, ToolResult


def _chunk_value(chunk: Chunk | dict[str, Any], key: str, default: Any = None) -> Any:
    return getattr(chunk, key, chunk.get(key, default) if isinstance(chunk, dict) else default)


def _chunks(value: Any) -> list[Chunk | dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [item for item in value if isinstance(item, (Chunk, dict))]
    return []


from adaptive.guardrails.sanitize import frame_retrieved_content


def check_tool_result(result: ToolResult, context: RequestContext) -> GuardrailResult:
    raw_chunks = result.diagnostics.get("chunks", result.diagnostics.get("retrieved_chunks", []))
    chunks = _chunks(raw_chunks)
    declared_tenant = result.diagnostics.get("tenant_id")
    if declared_tenant is not None and declared_tenant != context.tenant_id:
        return _deny("tenant_mismatch", "Tool result tenant does not match authenticated scope.")
    for chunk in chunks:
        if _chunk_value(chunk, "tenant_id") != context.tenant_id:
            return _deny("tenant_mismatch", "Retrieved content crossed tenant scope.")
        acl = frozenset(_chunk_value(chunk, "acl", frozenset()))
        if not context.can_access(acl):
            return _deny("acl_mismatch", "Retrieved content is not authorized for this subject.")
    return GuardrailResult(
        action=GuardrailAction.ALLOW,
        reason="Tool result passed tenant and ACL checks.",
        details={
            "code": "retrieved_content_approved",
            "framed_content": frame_retrieved_content(chunks, result.text),
        },
        span_attributes={
            "guardrail.name": "retrieved",
            "guardrail.status": "allow",
            "guardrail.action": "allow",
            "guardrail.chunk_count": len(chunks),
        },
    )


def _deny(code: str, reason: str) -> GuardrailResult:
    return GuardrailResult(
        action=GuardrailAction.DENY,
        reason=reason,
        details={"code": code},
        span_attributes={
            "guardrail.name": "retrieved",
            "guardrail.status": "deny",
            "guardrail.action": "deny",
            "guardrail.code": code,
        },
    )
