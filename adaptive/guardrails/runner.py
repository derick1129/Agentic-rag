"""Orchestrates guardrails and exposes trace-ready machine-readable outcomes."""

from __future__ import annotations

from typing import Any, Iterable

from adaptive.guardrails.input import InputRequest, check_input
from adaptive.guardrails.output import check_output
from adaptive.guardrails.post_generation_nli import NLIValidator
from adaptive.guardrails.retrieved import check_tool_result
from adaptive.interfaces import (
    Answer,
    GuardrailAction,
    GuardrailResult,
    RequestContext,
    ToolResult,
)


class GuardrailRunner:
    def __init__(
        self,
        *,
        max_query_chars: int = 8_000,
        max_payload_bytes: int = 64_000,
        groundedness_threshold: float = 0.55,
        citation_coverage_threshold: float = 1.0,
        nli_validator: NLIValidator | None = None,
    ) -> None:
        self.max_query_chars = max_query_chars
        self.max_payload_bytes = max_payload_bytes
        self.groundedness_threshold = groundedness_threshold
        self.citation_coverage_threshold = citation_coverage_threshold
        self.nli_validator = nli_validator or NLIValidator(threshold=groundedness_threshold)

    async def check_input(self, request: InputRequest | dict[str, Any]) -> GuardrailResult:
        return check_input(
            request,
            max_query_chars=self.max_query_chars,
            max_payload_bytes=self.max_payload_bytes,
        )

    async def check_tool_result(
        self, result: ToolResult, context: RequestContext
    ) -> GuardrailResult:
        return check_tool_result(result, context)

    async def check_output(
        self,
        answer: Answer,
        evidence: Iterable[Any],
        context: RequestContext,
    ) -> GuardrailResult:
        return check_output(
            answer,
            evidence,
            context,
            groundedness_threshold=self.groundedness_threshold,
            citation_coverage_threshold=self.citation_coverage_threshold,
        )

    async def check_post_generation(
        self,
        answer: Answer | str,
        evidence: Iterable[Any],
        context: RequestContext,
        **kwargs: Any,
    ) -> GuardrailResult:
        if isinstance(answer, Answer):
            answer_text = answer.text
            citations = list(answer.citations)
        else:
            answer_text = str(answer)
            citations = list(kwargs.get("citations", []))

        evidence_list = list(evidence)
        tenant_id = kwargs.get("tenant_id", context.tenant_id if context else None)

        kwargs_copy = dict(kwargs)
        kwargs_copy.pop("citations", None)
        kwargs_copy.pop("tenant_id", None)

        nli_result = self.nli_validator.validate(
            answer=answer_text,
            citations=citations,
            evidence_chunks=evidence_list,
            tenant_id=tenant_id,
            **kwargs_copy,
        )

        status_action_map = {
            "grounded": GuardrailAction.ALLOW,
            "partial": GuardrailAction.QUALIFY,
            "ungrounded": GuardrailAction.DENY,
            "refused": GuardrailAction.DENY,
        }
        action = status_action_map.get(nli_result.status, GuardrailAction.DENY)

        if nli_result.status == "ungrounded":
            reason = "ungrounded"
        elif nli_result.status == "grounded":
            reason = "grounded"
        elif nli_result.status == "partial":
            reason = "partial"
        elif nli_result.status == "refused":
            reason = nli_result.reason or "refused"
        else:
            reason = nli_result.reason or nli_result.status

        span_attributes = {
            "guardrail.name": "post_generation_nli",
            "guardrail.status": action.value,
            "guardrail.action": action.value,
            "guardrail.post_generation.action": action.value,
            "answer.nli.status": nli_result.status,
            "answer.nli.claim_violations_count": len(nli_result.claim_violations),
            "answer.nli.evidence_coverage": nli_result.evidence_coverage,
            **nli_result.span_attributes,
        }

        details = {
            "code": f"nli_{nli_result.status}",
            "status": nli_result.status,
            "claim_violations": nli_result.claim_violations,
            "evidence_coverage": nli_result.evidence_coverage,
            "reason": nli_result.reason,
            "span_attributes": nli_result.span_attributes,
        }

        return GuardrailResult(
            action=action,
            reason=reason,
            details=details,
            span_attributes=span_attributes,
        )

