"""Tests for RetrievalGraderBridge."""

# ruff: noqa: ANN001, E501, E702, PLR2004

import pytest

from adaptive.interfaces import Chunk, RequestContext, RetrievalQuery
from adaptive.retrieval.contracts import ScoredChunk
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.retrieval_grader import RetrievalGrader, RetrievalGraderBridge


def make_chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        tenant_id="acme",
        version=1,
        ordinal=0,
        text=text,
        acl=frozenset({"public"}),
    )


@pytest.mark.asyncio
async def test_retrieve_with_grader_accepts_when_reranker_not_better():
    async def fail_bm25(query, limit):
        return []

    async def fail_dense(query, limit):
        return []

    retriever = HybridRetrieverService(bm25=fail_bm25, dense=fail_dense)
    gradered = RetrievalGraderBridge(retriever, use_rewrite=False)
    result = await gradered.retrieve(query="test", limit=3)
    assert result.chunks == []
    assert result.diagnostics.grader_accepted is True
    assert result.diagnostics.reason == "no_reranker_available"


@pytest.mark.asyncio
async def test_retrieve_accepts_when_reranker_outperforms():
    c1 = make_chunk("c1", "pricing plan info")
    bm25_chunk = ScoredChunk(chunk=c1, score=0.6)

    async def stage_bm25(query, limit):
        return [bm25_chunk]

    async def stage_dense(query, limit):
        return [bm25_chunk]

    class GoodReranker:
        async def rerank(self, query, candidates, limit):
            return [ScoredChunk(chunk=c1, score=0.85)]

    retriever = HybridRetrieverService(
        bm25=stage_bm25,
        dense=stage_dense,
        reranker=GoodReranker(),
    )
    bridge = RetrievalGraderBridge(
        retriever, grader=RetrievalGrader(grade_threshold=0.05), use_rewrite=True
    )
    context = RequestContext(tenant_id="acme", subject_id="u1", acl=frozenset({"public"}))
    result = await bridge.retrieve(
        RetrievalQuery(text="pricing plan", context=context, limit=3)
    )

    assert result.diagnostics.grader_accepted is True
    assert result.diagnostics.reason == "reranker_outperformed"
    assert result.diagnostics.grader_bm25_score == pytest.approx(0.6)
    assert result.diagnostics.grader_reranker_score == pytest.approx(0.85)
    assert result.diagnostics.dense_retry_attempted is False
    assert result.diagnostics.rewrite_performed is False
    assert len(result.chunks) == 1
    assert result.chunks[0].id == "c1"


@pytest.mark.asyncio
async def test_retrieve_triggers_rewrite_and_dense_retry_on_rejection():
    c1 = make_chunk("c1", "original refund")
    c_retry = make_chunk("c_retry", "retried refund policy")
    bm25_chunk = ScoredChunk(chunk=c1, score=0.80)

    async def stage_bm25(query, limit):
        return [bm25_chunk]

    dense_calls = []

    async def stage_dense(query, limit):
        dense_calls.append(query.text)
        if len(dense_calls) == 1:
            return [bm25_chunk]
        # On retry, return retried chunk
        return [ScoredChunk(chunk=c_retry, score=0.95)]

    class MediocreReranker:
        async def rerank(self, query, candidates, limit):
            # Only 0.02 above BM25, below the configured 0.05 threshold.
            return [ScoredChunk(chunk=c1, score=0.82)]

    retriever = HybridRetrieverService(
        bm25=stage_bm25,
        dense=stage_dense,
        reranker=MediocreReranker(),
    )
    bridge = RetrievalGraderBridge(
        retriever, grader=RetrievalGrader(grade_threshold=0.05), use_rewrite=True
    )
    context = RequestContext(tenant_id="acme", subject_id="u1", acl=frozenset({"public"}))
    orig_text = "How much does the plan cost?"
    result = await bridge.retrieve(query=orig_text, limit=3, context=context)

    # Dense was searched twice: initial and retried
    assert len(dense_calls) == 2
    assert dense_calls[0] == orig_text
    assert dense_calls[1] != orig_text  # rewritten query

    assert result.diagnostics.grader_accepted is False
    assert result.diagnostics.reason == "reranker_not_better"
    assert result.diagnostics.rewrite_performed is True
    assert result.diagnostics.dense_retry_attempted is True
    assert result.diagnostics.original_query == orig_text
    assert result.diagnostics.rewrite_query == dense_calls[1]
    assert [c.id for c in result.chunks] == ["c_retry"]


@pytest.mark.asyncio
async def test_retrieve_without_rewrite_does_not_retry():
    c1 = make_chunk("c1", "content")
    bm25_chunk = ScoredChunk(chunk=c1, score=0.8)

    async def stage_bm25(query, limit):
        return [bm25_chunk]

    dense_calls = []

    async def stage_dense(query, limit):
        dense_calls.append(query.text)
        return [bm25_chunk]

    class MediocreReranker:
        async def rerank(self, query, candidates, limit):
            return [ScoredChunk(chunk=c1, score=0.81)]

    retriever = HybridRetrieverService(
        bm25=stage_bm25,
        dense=stage_dense,
        reranker=MediocreReranker(),
    )
    bridge = RetrievalGraderBridge(
        retriever, grader=RetrievalGrader(grade_threshold=0.05), use_rewrite=False
    )
    context = RequestContext(tenant_id="acme", subject_id="u1", acl=frozenset({"public"}))
    result = await bridge.retrieve(
        RetrievalQuery(text="How much does the plan cost?", context=context, limit=3)
    )

    assert len(dense_calls) == 1
    assert result.diagnostics.grader_accepted is False
    assert result.diagnostics.dense_retry_attempted is False
    assert result.diagnostics.rewrite_performed is False
