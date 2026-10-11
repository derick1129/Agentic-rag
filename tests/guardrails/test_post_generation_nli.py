from __future__ import annotations

import pytest

from adaptive.guardrails.post_generation_nli import NLIResult, NLIValidator
from adaptive.interfaces import Chunk


def sample_chunk(
    chunk_id: str = "c1",
    text: str = "Refunds are available within 30 days of purchase.",
    tenant_id: str = "tenant-1",
    status: str = "active",
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "tenant_id": tenant_id,
        "status": status,
        "acl": ["public"],
    }


def test_ground_answer_supported_by_evidence():
    validator = NLIValidator(threshold=0.55)
    answer = "The refund window is 30 days."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1", "document_id": "d1", "text": "Refunds are available within 30 days."}],
        evidence_chunks=evidence,
    )
    assert isinstance(result, NLIResult)
    assert result.status == "grounded"
    assert result.evidence_coverage >= 0.55
    assert result.claim_violations == []
    assert result.span_attributes["answer.nli.status"] == "grounded"
    assert result.span_attributes["answer.nli.claim_violations_count"] == 0


def test_partial_answer_some_claims_unsupported():
    validator = NLIValidator(threshold=0.55)
    answer = "Refunds are available within 30 days. We also offer free international luxury cruises."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
    )
    assert result.status == "partial"
    assert 0.0 < result.evidence_coverage < 0.55
    assert len(result.claim_violations) == 1
    assert "cruises" in result.claim_violations[0]
    assert result.span_attributes["answer.nli.status"] == "partial"


def test_ungrounded_answer_all_claims_unsupported():
    validator = NLIValidator(threshold=0.55)
    answer = "We specialize in quantum teleportation and interstellar space flights."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
    )
    assert result.status == "ungrounded"
    assert result.evidence_coverage == 0.0
    assert len(result.claim_violations) == 1
    assert result.span_attributes["answer.nli.status"] == "ungrounded"


def test_numeric_entity_claim_mismatch_is_ungrounded():
    validator = NLIValidator(threshold=0.55)
    # The evidence specifies 30 days, but the answer claims 90 days.
    answer = "The refund window is 90 days."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
    )
    assert result.status == "ungrounded"
    assert result.evidence_coverage == 0.0
    assert len(result.claim_violations) == 1


def test_empty_citations_returns_refused():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[],
        evidence_chunks=evidence,
    )
    assert result.status == "refused"
    assert result.reason == "no_citations"
    assert result.evidence_coverage == 0.0
    assert result.span_attributes["answer.nli.status"] == "refused"


def test_empty_citation_dict_returns_refused():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": ""}, {}],
        evidence_chunks=evidence,
    )
    assert result.status == "refused"
    assert result.reason in ("no_citations", "empty_citations")
    assert result.evidence_coverage == 0.0


def test_blocked_citation_deleted_chunk():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    evidence = [sample_chunk(status="deleted")]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
        tenant_id="tenant-1",
    )
    assert result.status == "ungrounded"
    assert result.reason == "blocked_citations"
    assert result.evidence_coverage == 0.0
    assert len(result.claim_violations) == 1


def test_blocked_citation_replaced_chunk():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    evidence = [sample_chunk(status="replaced")]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
        tenant_id="tenant-1",
    )
    assert result.status == "ungrounded"
    assert result.reason == "blocked_citations"
    assert result.evidence_coverage == 0.0


def test_blocked_citation_tenant_mismatch():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    evidence = [sample_chunk(tenant_id="tenant-evil")]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
        tenant_id="tenant-1",
    )
    assert result.status == "ungrounded"
    assert result.reason == "blocked_citations"
    assert result.evidence_coverage == 0.0


def test_fail_closed_on_empty_evidence():
    validator = NLIValidator()
    answer = "The refund window is 30 days."
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=[],
        tenant_id="tenant-1",
    )
    assert result.status == "ungrounded"
    assert result.evidence_coverage == 0.0


def test_supports_pydantic_chunk_models():
    validator = NLIValidator(threshold=0.55)
    chunk = Chunk(
        id="chunk-pydantic-1",
        document_id="doc-1",
        tenant_id="tenant-1",
        version=1,
        ordinal=0,
        text="Refunds are available within 30 days.",
        acl=frozenset({"public"}),
    )
    result = validator.validate(
        answer="The refund window is 30 days.",
        citations=[{"chunk_id": "chunk-pydantic-1"}],
        evidence_chunks=[chunk],
        tenant_id="tenant-1",
    )
    assert result.status == "grounded"


def test_custom_pluggable_nli_provider():
    class FakeNLIProvider:
        def __init__(self, entails: bool):
            self.entails = entails

        def check_entailment(self, claim: str, evidence: str) -> bool:
            return self.entails

    validator = NLIValidator(provider=FakeNLIProvider(entails=True))
    result = validator.validate(
        answer="Completely arbitrary sentence.",
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=[sample_chunk()],
    )
    assert result.status == "grounded"

    validator_denying = NLIValidator(provider=FakeNLIProvider(entails=False))
    result_denied = validator_denying.validate(
        answer="Completely arbitrary sentence.",
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=[sample_chunk()],
    )
    assert result_denied.status == "ungrounded"


def test_max_claims_per_check_cap():
    validator = NLIValidator(threshold=0.55, max_claims_per_check=3)
    # 5 sentences: first 3 match evidence, 4 and 5 do not
    answer = (
        "Refunds are available within 30 days. "
        "The refund window is 30 days. "
        "A refund is available in 30 days. "
        "Claim four is false and completely made up. "
        "Claim five is false and completely made up."
    )
    evidence = [sample_chunk()]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1"}],
        evidence_chunks=evidence,
    )
    # The first 3 claims are checked and all 3 match the 30-day refund evidence.
    assert result.status == "grounded"
    assert result.evidence_coverage == 1.0
