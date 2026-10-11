"""Tests for RetrievalGrader."""

# ruff: noqa: ANN001, E501, E702, PLR2004

from adaptive.interfaces import Chunk
from adaptive.retrieval.contracts import ScoredChunk
from adaptive.retrieval.retrieval_grader import RetrievalGrader, RetrievalGraderResult


def make_chunk(chunk_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            tenant_id="acme",
            version=1,
            ordinal=0,
            text=f"Sample text for {chunk_id}",
            acl=frozenset({"public"}),
        ),
        score=score,
    )


def test_returns_accept_when_reranker_better_than_bm25():
    grader = RetrievalGrader(grade_threshold=0.05)
    bm25 = make_chunk("b1", score=0.7)
    reranked = make_chunk("r1", score=0.9)
    result = grader.assess(query="refund policy", bm25_top1=bm25, reranked_top1=reranked)

    assert isinstance(result, RetrievalGraderResult)
    assert result.accept is True
    assert result.reason == "reranker_outperformed"
    assert result.bm25_score == 0.7
    assert result.reranker_score == 0.9
    assert result.score == 0.2


def test_rejects_when_reranker_not_better():
    grader = RetrievalGrader(grade_threshold=0.05)
    bm25 = make_chunk("b1", score=0.8)
    reranked = make_chunk("r1", score=0.82)
    result = grader.assess(query="refund policy", bm25_top1=bm25, reranked_top1=reranked)

    assert result.accept is False
    assert result.reason == "reranker_not_better"
    assert result.bm25_score == 0.8
    assert result.reranker_score == 0.82


def test_rejects_when_reranker_lower_than_bm25():
    grader = RetrievalGrader(grade_threshold=0.05)
    bm25 = make_chunk("b1", score=0.8)
    reranked = make_chunk("r1", score=0.6)
    result = grader.assess(query="refund policy", bm25_top1=bm25, reranked_top1=reranked)

    assert result.accept is False
    assert result.reason == "reranker_not_better"


def test_accepts_when_no_reranker_available():
    grader = RetrievalGrader(grade_threshold=0.05)
    bm25 = make_chunk("b1", score=0.7)
    result = grader.assess(query="refund policy", bm25_top1=bm25, reranked_top1=None)

    assert result.accept is True
    assert result.reason == "no_reranker_available"
    assert result.bm25_score == 0.7
    assert result.reranker_score is None


def test_insufficient_candidates_when_both_none():
    grader = RetrievalGrader(grade_threshold=0.05)
    result = grader.assess(query="refund policy", bm25_top1=None, reranked_top1=None)

    assert result.accept is False
    assert result.reason == "insufficient_candidates"
    assert result.bm25_score is None
    assert result.reranker_score is None


def test_insufficient_candidates_when_bm25_none_and_reranker_present():
    grader = RetrievalGrader(grade_threshold=0.05)
    reranked = make_chunk("r1", score=0.9)
    result = grader.assess(query="refund policy", bm25_top1=None, reranked_top1=reranked)

    assert result.accept is False
    assert result.reason == "insufficient_candidates"
    assert result.bm25_score is None
    assert result.reranker_score == 0.9


def test_exact_threshold_boundary():
    grader = RetrievalGrader(grade_threshold=0.05)
    bm25 = make_chunk("b1", score=0.50)
    reranked = make_chunk("r1", score=0.55)
    result = grader.assess(query="boundary check", bm25_top1=bm25, reranked_top1=reranked)

    assert result.accept is True
    assert result.reason == "reranker_outperformed"


def test_custom_grade_threshold():
    grader = RetrievalGrader(grade_threshold=0.15)
    bm25 = make_chunk("b1", score=0.70)
    reranked = make_chunk("r1", score=0.80)
    result = grader.assess(query="custom threshold", bm25_top1=bm25, reranked_top1=reranked)

    assert result.accept is False
    assert result.reason == "reranker_not_better"
