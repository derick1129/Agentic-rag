"""Groundedness, citation, sensitive-data, refusal, and qualification checks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from adaptive.interfaces import Answer, GuardrailAction, GuardrailResult, RequestContext, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterable

_MIN_CLAIM_WORDS = 2
_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{8,}\d)(?!\w)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "secret": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16})\b"),
}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "within",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
}


def _evidence_text(evidence: Iterable[Any]) -> tuple[str, set[str]]:
    texts: list[str] = []
    ids: set[str] = set()
    for item in evidence:
        if isinstance(item, ToolResult):
            texts.append(item.text)
            texts.extend(str(row) for row in item.rows)
            for citation in item.citations:
                if isinstance(citation, dict) and citation.get("chunk_id"):
                    ids.add(str(citation["chunk_id"]))
            chunks = item.diagnostics.get("chunks", item.diagnostics.get("retrieved_chunks", []))
            for chunk in chunks if isinstance(chunks, list) else []:
                text = getattr(
                    chunk, "text", chunk.get("text", "") if isinstance(chunk, dict) else ""
                )
                texts.append(text)
                chunk_id = getattr(
                    chunk, "id", chunk.get("id") if isinstance(chunk, dict) else None
                )
                if chunk_id:
                    ids.add(str(chunk_id))
        elif isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(str(item.get("text", "")))
            if item.get("id"):
                ids.add(str(item["id"]))
    return " ".join(texts), ids


def check_output(
    answer: Answer,
    evidence: Iterable[Any],
    context: RequestContext,
    *,
    groundedness_threshold: float = 0.55,
    citation_coverage_threshold: float = 1.0,
) -> GuardrailResult:
    del context  # Scope is verified before evidence reaches this boundary.
    evidence_text, evidence_ids = _evidence_text(evidence)
    if not evidence_text.strip():
        return _output(
            GuardrailAction.REFUSE,
            "insufficient_evidence",
            "No approved evidence supports an answer.",
        )
    pii_types = [name for name, pattern in _PII_PATTERNS.items() if pattern.search(answer.text)]
    unsupported = _unsupported_claims(answer.text, evidence_text, groundedness_threshold)
    if pii_types:
        return _output(
            GuardrailAction.DENY,
            "pii_detected",
            "Answer contains configured sensitive data.",
            pii_types=pii_types,
            unsupported_claims=unsupported,
        )
    if unsupported:
        return _output(
            GuardrailAction.QUALIFY,
            "unsupported_claims",
            "Some claims are not supported by approved evidence.",
            unsupported_claims=unsupported,
        )
    citations = {
        str(c.get("chunk_id"))
        for c in answer.citations
        if isinstance(c, dict) and c.get("chunk_id")
    }
    if not citations:
        return _output(
            GuardrailAction.QUALIFY,
            "missing_citations",
            "Answer needs citations for its evidence.",
            citation_coverage=0.0,
        )
    coverage = len(citations & evidence_ids) / len(citations) if citations else 0.0
    if coverage < citation_coverage_threshold:
        return _output(
            GuardrailAction.QUALIFY,
            "citation_coverage",
            "Citations do not cover approved evidence.",
            citation_coverage=coverage,
        )
    return _output(
        GuardrailAction.ALLOW,
        "output_valid",
        "Answer passed output guardrails.",
        citation_coverage=coverage,
    )


def _unsupported_claims(answer: str, evidence: str, threshold: float) -> list[str]:
    evidence_words = {
        word for word in re.findall(r"[a-z0-9]+", evidence.casefold()) if word not in _STOPWORDS
    }
    claims: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        words = {
            word for word in re.findall(r"[a-z0-9]+", sentence.casefold()) if word not in _STOPWORDS
        }
        if len(words) >= _MIN_CLAIM_WORDS and (
            not (words & evidence_words) or len(words & evidence_words) / len(words) < threshold
        ):
            claims.append(sentence)
    return claims


def _output(
    action: GuardrailAction,
    code: str,
    reason: str,
    **details: list[str] | float,
) -> GuardrailResult:
    return GuardrailResult(
        action=action,
        reason=reason,
        details={"code": code, **details},
        span_attributes={
            "guardrail.name": "output",
            "guardrail.status": action.value,
            "guardrail.action": action.value,
            "guardrail.output.action": action.value,
            "guardrail.code": code,
        },
    )
