"""Deterministic retrieval quality grader and hybrid retrieval bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive.retrieval.contracts import ScoredChunk


@dataclass(frozen=True)
class RetrievalGraderResult:
    """Result of evaluating reranked top-1 vs BM25 top-1 retrieval confidence."""

    accept: bool
    reason: str
    bm25_score: float | None = None
    reranker_score: float | None = None
    score: float = 0.0
    grade_threshold: float = 0.05


# Spec alias
GraderResult = RetrievalGraderResult


class RetrievalGrader:
    """Deterministic score-margin gate between BM25 top-1 and reranked top-1."""

    def __init__(self, grade_threshold: float = 0.05) -> None:
        self.grade_threshold = grade_threshold

    def assess(
        self,
        query: str,
        bm25_top1: ScoredChunk | None,
        reranked_top1: ScoredChunk | None,
    ) -> RetrievalGraderResult:
        if reranked_top1 is None and bm25_top1 is None:
            return RetrievalGraderResult(
                accept=False,
                reason="insufficient_candidates",
                bm25_score=None,
                reranker_score=None,
                score=0.0,
                grade_threshold=self.grade_threshold,
            )

        if reranked_top1 is None:
            return RetrievalGraderResult(
                accept=True,
                reason="no_reranker_available",
                bm25_score=float(bm25_top1.score),
                reranker_score=None,
                score=0.0,
                grade_threshold=self.grade_threshold,
            )

        if bm25_top1 is None:
            return RetrievalGraderResult(
                accept=False,
                reason="insufficient_candidates",
                bm25_score=None,
                reranker_score=float(reranked_top1.score),
                score=0.0,
                grade_threshold=self.grade_threshold,
            )

        bm25_score = float(bm25_top1.score)
        reranker_score = float(reranked_top1.score)
        margin = round(reranker_score - bm25_score, 6)

        if margin >= self.grade_threshold:
            return RetrievalGraderResult(
                accept=True,
                reason="reranker_outperformed",
                bm25_score=bm25_score,
                reranker_score=reranker_score,
                score=margin,
                grade_threshold=self.grade_threshold,
            )

        return RetrievalGraderResult(
            accept=False,
            reason="reranker_not_better",
            bm25_score=bm25_score,
            reranker_score=reranker_score,
            score=margin,
            grade_threshold=self.grade_threshold,
        )


__all__ = [
    "GraderResult",
    "RetrievalGrader",
    "RetrievalGraderResult",
]
