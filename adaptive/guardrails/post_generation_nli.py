"""Post-generation claim-to-evidence checker.

The default implementation uses token overlap as a heuristic. An injected
provider can perform actual natural language inference.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

_MIN_LONG_SUFFIX_LEN = 4
_MIN_SHORT_SUFFIX_LEN = 3
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
    "with",
    "at",
    "by",
    "from",
    "it",
    "this",
    "that",
    "be",
    "as",
    "into",
    "their",
    "its",
    "our",
    "your",
    "we",
    "they",
    "you",
    "have",
    "has",
    "had",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
}


@runtime_checkable
class NLIProvider(Protocol):
    """Pluggable provider interface for external NLI models."""

    def check_entailment(self, claim: str, evidence: str) -> bool: ...


class NLIResult(BaseModel):
    """Result of post-generation NLI validation."""

    status: Literal["grounded", "partial", "ungrounded", "refused"]
    claim_violations: list[str] = Field(default_factory=list)
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    span_attributes: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    model_config = {"frozen": True}


def _normalize_word(word: str) -> str:
    w = word.casefold()
    if len(w) > _MIN_LONG_SUFFIX_LEN and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > _MIN_SHORT_SUFFIX_LEN and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > _MIN_LONG_SUFFIX_LEN and w.endswith("ing"):
        return w[:-3]
    if len(w) > _MIN_SHORT_SUFFIX_LEN and w.endswith("ed"):
        return w[:-2]
    return w


def _extract_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [_normalize_word(w) for w in words if w not in _STOPWORDS]


def _extract_chunk_dicts(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalizes evidence chunks from models, dicts, ToolResults, or strings."""
    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            meta = item.get("metadata")
            status = (
                item.get("status")
                or item.get("document_status")
                or (meta.get("status") if isinstance(meta, dict) else None)
            )
            result.append(
                {
                    "id": str(item.get("id") or item.get("chunk_id") or ""),
                    "text": str(item.get("text", "")),
                    "tenant_id": item.get("tenant_id"),
                    "status": str(status).lower() if status else "active",
                }
            )
        elif hasattr(item, "diagnostics") and hasattr(item, "text"):
            # ToolResult-like object
            chunks = getattr(item, "diagnostics", {}).get(
                "chunks", getattr(item, "diagnostics", {}).get("retrieved_chunks", [])
            )
            if isinstance(chunks, list) and chunks:
                result.extend(_extract_chunk_dicts(chunks))
            elif getattr(item, "text", ""):
                result.append(
                    {
                        "id": "",
                        "text": getattr(item, "text", ""),
                        "tenant_id": None,
                        "status": "active",
                    }
                )
        elif hasattr(item, "text") and hasattr(item, "id"):
            # Chunk or Pydantic BaseModel
            meta = getattr(item, "metadata", {})
            status = (
                getattr(item, "status", None)
                or getattr(item, "document_status", None)
                or (meta.get("status") if isinstance(meta, dict) else None)
            )
            result.append(
                {
                    "id": str(getattr(item, "id", "")),
                    "text": str(getattr(item, "text", "")),
                    "tenant_id": getattr(item, "tenant_id", None),
                    "status": str(status).lower() if status else "active",
                }
            )
        elif isinstance(item, str):
            result.append(
                {
                    "id": "",
                    "text": item,
                    "tenant_id": None,
                    "status": "active",
                }
            )
    return result


class NLIValidator:
    """Claim checker; uses overlap heuristics unless an NLI provider is injected."""

    def __init__(
        self,
        threshold: float = 0.55,
        min_evidence_words: int = 3,
        max_claims_per_check: int = 10,
        provider: NLIProvider | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_evidence_words = min_evidence_words
        self.max_claims_per_check = max_claims_per_check
        self.provider = provider
        self.tenant_id = tenant_id

    def _split_claims(self, text: str) -> list[str]:
        raw = text.strip()
        if not raw:
            return []
        segments = re.split(r"(?:(?<=[.!?])\s+|\n+|;\s*)", raw)
        claims: list[str] = []
        for segment in segments:
            # Strip list numbers, bullets, and citation tags like [1] or [source 1]
            cleaned = re.sub(r"^(\d+[\.\)]\s*|[-*•]\s*)", "", segment.strip()).strip()
            cleaned = re.sub(r"\[(?:\d+|source\s*[^\]]*)\]", "", cleaned).strip()
            if cleaned:
                claims.append(cleaned)
            if len(claims) >= self.max_claims_per_check:
                break
        return claims[: self.max_claims_per_check]

    def _match_claim(self, claim: str, chunk_text: str) -> bool:
        if self.provider is not None:
            return self.provider.check_entailment(claim, chunk_text)

        # Check chunk length
        chunk_tokens = _extract_tokens(chunk_text)
        if len(chunk_tokens) < self.min_evidence_words:
            return False

        # Numeric entity guard: numbers mentioned in claim must appear in chunk
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
        if claim_numbers:
            chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk_text))
            if not claim_numbers.issubset(chunk_numbers):
                return False

        claim_tokens = _extract_tokens(claim)
        if not claim_tokens:
            return False

        overlap = set(claim_tokens) & set(chunk_tokens)
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / len(claim_tokens)

        required_count = min(len(claim_tokens), self.min_evidence_words)
        return overlap_ratio >= self.threshold and overlap_count >= required_count

    def _resolve_cited_chunks(
        self,
        valid_citations: list[dict[str, Any]],
        evidence_chunks: list[Any],
        effective_tenant: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        normalized_chunks = _extract_chunk_dicts(evidence_chunks)
        chunks_by_id = {c["id"]: c for c in normalized_chunks if c["id"]}
        valid_cited_chunks: list[dict[str, Any]] = []
        blocked_found = False

        for cit in valid_citations:
            cid = str(cit.get("chunk_id") or cit.get("id"))
            cit_status = str(cit.get("status") or "").lower()
            cit_tenant = cit.get("tenant_id")
            if cit_status in ("deleted", "replaced") or (
                effective_tenant and cit_tenant and cit_tenant != effective_tenant
            ):
                blocked_found = True
                continue

            chunk = chunks_by_id.get(cid)
            if chunk is None:
                continue
            chk_status = chunk.get("status", "active")
            chk_tenant = chunk.get("tenant_id")
            if chk_status in ("deleted", "replaced") or (
                effective_tenant and chk_tenant and chk_tenant != effective_tenant
            ):
                blocked_found = True
                continue
            valid_cited_chunks.append(chunk)

        return valid_cited_chunks, blocked_found

    def validate(
        self,
        answer: object,
        citations: list[dict[str, Any]],
        evidence_chunks: list[Any],
        tenant_id: str | None = None,
        **kwargs: object,
    ) -> NLIResult:
        del kwargs
        # Extract text from answer if Answer object
        answer_text = getattr(answer, "text", answer)
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)

        claims = self._split_claims(answer_text)

        # 1. Citations empty check and reject citations with empty/missing chunk_id/id
        valid_citations = [
            c for c in citations if isinstance(c, dict) and bool(c.get("chunk_id") or c.get("id"))
        ]
        if not valid_citations:
            return self._result(
                status="refused",
                claim_violations=claims,
                coverage=0.0,
                reason="no_citations",
            )

        # 2. Reject empty/blocked citations for deleted, replaced, or cross-tenant chunks
        effective_tenant = tenant_id or self.tenant_id
        valid_cited_chunks, blocked_found = self._resolve_cited_chunks(
            valid_citations, evidence_chunks, effective_tenant
        )

        # 7. Fail closed on empty/blocked evidence: default ungrounded
        if not valid_cited_chunks:
            reason: str | None = "blocked_citations" if blocked_found else "empty_evidence"
            return self._result(
                status="ungrounded",
                claim_violations=claims,
                coverage=0.0,
                reason=reason,
            )

        if not claims:
            return self._result(
                status="ungrounded",
                claim_violations=[],
                coverage=0.0,
                reason="no_claims",
            )

        # 3, 4. Match each claim against valid evidence chunks
        supported_claims: list[str] = []
        unsupported_claims: list[str] = []

        for claim in claims:
            matched = any(self._match_claim(claim, chunk["text"]) for chunk in valid_cited_chunks)
            if matched:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)

        coverage = len(supported_claims) / len(claims)

        # 5. No claim matches evidence chunk -> ungrounded
        if not supported_claims:
            return self._result(
                status="ungrounded",
                claim_violations=unsupported_claims,
                coverage=0.0,
                reason="ungrounded",
            )

        # 6. Most claims covered (>= threshold) -> grounded; otherwise partial
        status: Literal["grounded", "partial"] = (
            "grounded" if coverage >= self.threshold else "partial"
        )
        result_reason = None if status == "grounded" else "partial_coverage"

        return self._result(
            status=status,
            claim_violations=unsupported_claims,
            coverage=coverage,
            reason=result_reason,
        )

    def _result(
        self,
        status: Literal["grounded", "partial", "ungrounded", "refused"],
        claim_violations: list[str],
        coverage: float,
        reason: str | None = None,
    ) -> NLIResult:
        span_attributes: dict[str, Any] = {
            "answer.nli.status": status,
            "answer.nli.method": "provider" if self.provider is not None else "heuristic_overlap",
            "answer.nli.claim_violations_count": len(claim_violations),
            "answer.nli.evidence_coverage": coverage,
        }
        if reason:
            span_attributes["answer.nli.reason"] = reason

        return NLIResult(
            status=status,
            claim_violations=claim_violations,
            evidence_coverage=coverage,
            span_attributes=span_attributes,
            reason=reason,
        )
