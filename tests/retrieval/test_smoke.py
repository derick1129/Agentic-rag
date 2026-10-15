"""Integration smoke test for retrieval + guardrail + auth without live model credentials."""

# ruff: noqa: ANN001, ANN201, ANN401, E501, PLR2004, F401

from __future__ import annotations

from typing import Any

import jwt
import pytest
from starlette.requests import Request

from adaptive.api.auth import authenticate_request
from adaptive.config import Settings
from adaptive.guardrails.post_generation_nli import NLIValidator
from adaptive.interfaces import Chunk, RequestContext, RetrievalQuery
from adaptive.retrieval.contracts import HashEmbeddingProvider, ScoredChunk
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.query_rewrite import QueryRewrite
from adaptive.retrieval.retrieval_grader import RetrievalGrader, RetrievalGraderBridge


class FakeStage:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits

    async def search(self, query: RetrievalQuery, limit: int) -> list[ScoredChunk]:
        return self.hits[:limit]


class FunctionReranker:
    def __init__(self, score_boost: float = 0.2) -> None:
        self.score_boost = score_boost

    async def rerank(
        self, query: Any, chunks: list[Any], limit: int | None = None
    ) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []
        for i, item in enumerate(chunks):
            c = item.chunk if isinstance(item, ScoredChunk) else item
            base = 0.85 if "refund" in getattr(c, "text", "").lower() else 0.4
            scored.append(ScoredChunk(chunk=c, score=round(base + (0.01 * (len(chunks) - i)), 4)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit] if limit else scored


@pytest.mark.asyncio
async def test_full_retrieval_guardrail_auth_smoke_path():
    # 1. Authenticate with dev JWT
    settings = Settings(
        DATABASE_URL="sqlite:///:memory:",
        AUTH_MODE="development",
        AUTH_DEV_TENANT_ID="acme",
        AUTH_DEV_SUBJECT_ID="user-smoke",
        AUTH_DEV_ACL="public",
    )
    dev_token = jwt.encode(
        {"tenant_id": "acme", "sub": "user-smoke", "acl": ["public"]},
        "dev-secret-key-that-is-32-bytes-long!",
        algorithm="HS256",
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/queries",
        "headers": [(b"authorization", f"Bearer {dev_token}".encode())],
    }
    request = Request(scope)
    context = authenticate_request(request, settings=settings)

    assert context.tenant_id == "acme"
    assert context.subject_id == "user-smoke"
    assert "public" in context.acl

    # 2. Insert chunks with different ACLs and tenants
    visible_chunk = Chunk(
        id="c-visible",
        document_id="doc-acme-1",
        tenant_id="acme",
        version=1,
        ordinal=0,
        text="Acme 30-day refund policy is active for all customers.",
        token_count=10,
        acl=frozenset({"public"}),
    )
    internal_chunk = Chunk(
        id="c-internal",
        document_id="doc-acme-2",
        tenant_id="acme",
        version=1,
        ordinal=0,
        text="Acme confidential billing rules: employees only.",
        token_count=8,
        acl=frozenset({"internal"}),
    )
    other_tenant_chunk = Chunk(
        id="c-globex",
        document_id="doc-globex-1",
        tenant_id="globex",
        version=1,
        ordinal=0,
        text="Globex refund policy: no refunds are allowed under any circumstance.",
        token_count=11,
        acl=frozenset({"public"}),
    )

    # 3. Setup HashEmbeddingProvider, FunctionReranker, stages and RetrievalGraderBridge
    embedder = HashEmbeddingProvider(dimensions=64)
    query_vectors = embedder.embed(["refund policy"])
    assert len(query_vectors) == 1
    assert len(query_vectors[0]) == 64

    retriever = HybridRetrieverService(
        bm25=FakeStage(
            [
                ScoredChunk(chunk=visible_chunk, score=0.6),
                ScoredChunk(chunk=internal_chunk, score=0.7),
            ]
        ),
        dense=FakeStage(
            [
                ScoredChunk(chunk=other_tenant_chunk, score=0.9),
                ScoredChunk(chunk=visible_chunk, score=0.65),
            ]
        ),
        reranker=FunctionReranker(),
    )

    bridge = RetrievalGraderBridge(
        retriever=retriever,
        grader=RetrievalGrader(grade_threshold=0.05),
        rewriter=QueryRewrite(language="en"),
        use_rewrite=True,
    )

    retrieval_query = RetrievalQuery(
        text="How much does the plan cost?",
        context=context,
        limit=5,
    )

    result = await bridge.retrieve(retrieval_query)

    # 4. Assert tenant and ACL isolation
    assert len(result.chunks) > 0
    for chunk in result.chunks:
        assert chunk.tenant_id == "acme", "Cross-tenant chunk leaked!"
        assert "public" in chunk.acl, "Protected ACL chunk leaked!"
        assert chunk.id != "c-internal"
        assert chunk.id != "c-globex"

    # 5. Assert RetrievalGrader diagnostics
    assert hasattr(result, "diagnostics")
    diag = result.diagnostics
    assert diag is not None
    assert diag.grader_accepted is True
    assert diag.reason == "reranker_outperformed"
    assert diag.grader_reranker_score is not None
    assert diag.grader_bm25_score is not None

    # 6. Assert Post-Generation NLI Validator
    validator = NLIValidator(threshold=0.55)
    supported_answer = "Acme offers a 30-day refund policy for all customers."
    nli_result = validator.validate(
        answer=supported_answer,
        citations=[{"chunk_id": "c-visible", "document_id": "doc-acme-1"}],
        evidence_chunks=result.chunks,
        tenant_id=context.tenant_id,
    )
    assert nli_result.status == "grounded"
    assert nli_result.evidence_coverage > 0.5

    # 7. Assert unsupported claim fails post-generation validation
    unsupported_answer = "Acme policy allows free upgrades to private jet subscriptions."
    nli_bad = validator.validate(
        answer=unsupported_answer,
        citations=[{"chunk_id": "c-visible", "document_id": "doc-acme-1"}],
        evidence_chunks=result.chunks,
        tenant_id=context.tenant_id,
    )
    assert nli_bad.status == "ungrounded"
