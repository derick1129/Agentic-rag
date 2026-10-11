"""Deterministic retrieval quality grader and hybrid retrieval bridge."""

# ruff: noqa: ANN401, C901, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive.interfaces import (
    Chunk,
    RequestContext,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResult,
)
from adaptive.retrieval.contracts import ScoredChunk
from adaptive.retrieval.hybrid import HybridRetrieverService, _search_stage
from adaptive.retrieval.query_rewrite import QueryRewrite


@dataclass(frozen=True)
class RetrievalGraderResult:
    """Result of evaluating reranked top-1 vs BM25 top-1 retrieval confidence."""

    accept: bool
    reason: str
    bm25_score: float | None = None
    reranker_score: float | None = None
    score: float = 0.0
    grade_threshold: float | None = None


# Spec alias
GraderResult = RetrievalGraderResult


class RetrievalGraderDiagnostics(RetrievalDiagnostics):
    """Diagnostics carrying retrieval grader and rewrite metrics."""

    grader_accepted: bool = False
    grader_bm25_score: float | None = None
    grader_reranker_score: float | None = None
    reason: str = ""
    rewrite_performed: bool = False
    rewrite_query: str | None = None
    original_query: str = ""
    dense_retry_attempted: bool = False


class RetrievalGrader:
    """Optional score-margin gate, disabled until scores are calibrated."""

    def __init__(self, grade_threshold: float | None = None) -> None:
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

        if self.grade_threshold is None:
            return RetrievalGraderResult(
                accept=True,
                reason="uncalibrated_scores",
                bm25_score=bm25_score,
                reranker_score=reranker_score,
                score=margin,
                grade_threshold=None,
            )

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


class RetrievalGraderBridge:
    """Wraps hybrid retrieval with deterministic grading and single rewrite/dense-retry on rejection."""

    def __init__(
        self,
        retriever: HybridRetrieverService,
        *,
        grader: RetrievalGrader | None = None,
        rewriter: QueryRewrite | None = None,
        use_rewrite: bool = True,
    ) -> None:
        self.retriever = retriever
        self.grader = grader or RetrievalGrader()
        self.rewriter = rewriter or QueryRewrite()
        self.use_rewrite = use_rewrite

    async def retrieve(
        self,
        query: str | RetrievalQuery,
        limit: int = 3,
        **kwargs: Any,
    ) -> RetrievalResult:
        if isinstance(query, str):
            context = kwargs.get("context")
            if context is None:
                context = RequestContext(tenant_id="default", subject_id="default")
            doc_ids = kwargs.get("document_ids")
            retrieval_query = RetrievalQuery(
                text=query,
                context=context,
                limit=limit,
                document_ids=doc_ids,
            )
        else:
            retrieval_query = query

        captured_bm25: list[ScoredChunk] = []
        original_bm25 = getattr(self.retriever, "bm25", None)

        class _Bm25Interceptor:
            def __init__(self, target: Any) -> None:
                self._target = target

            async def search(self, q: RetrievalQuery, limit: int) -> list[ScoredChunk]:
                res = await _search_stage(self._target, q, limit)
                captured_bm25.extend(res)
                return res

            async def __call__(self, q: RetrievalQuery, limit: int) -> list[ScoredChunk]:
                res = await _search_stage(self._target, q, limit)
                captured_bm25.extend(res)
                return res

        if original_bm25 is not None:
            self.retriever.bm25 = _Bm25Interceptor(original_bm25)

        try:
            result = await self.retriever.retrieve(retrieval_query)
        finally:
            if original_bm25 is not None:
                self.retriever.bm25 = original_bm25

        bm25_top1: ScoredChunk | None = None
        if captured_bm25:
            allowed_bm25 = [
                h
                for h in captured_bm25
                if self.retriever._allowed(self.retriever._as_chunk(h.chunk), retrieval_query)
            ]
            bm25_top1 = allowed_bm25[0] if allowed_bm25 else captured_bm25[0]

        has_reranker = getattr(self.retriever, "reranker", None) is not None
        reranked_top1: ScoredChunk | None = None
        if has_reranker and result.diagnostics.reranked and result.chunks:
            reranked_top1 = ScoredChunk(
                chunk=result.chunks[0],
                score=result.chunks[0].score,
            )

        if not has_reranker:
            grader_result = RetrievalGraderResult(
                accept=True,
                reason="no_reranker_available",
                bm25_score=float(bm25_top1.score) if bm25_top1 is not None else None,
                reranker_score=None,
                score=0.0,
                grade_threshold=self.grader.grade_threshold,
            )
        else:
            grader_result = self.grader.assess(
                query=retrieval_query.text,
                bm25_top1=bm25_top1,
                reranked_top1=reranked_top1,
            )

        rewrite_performed = False
        rewrite_query: str | None = None
        dense_retry_attempted = False
        final_chunks = result.chunks
        dense_count = result.diagnostics.dense_count

        if not grader_result.accept and self.use_rewrite:
            rewrite_res = self.rewriter.rewrite(retrieval_query.text)
            rewrite_performed = rewrite_res.rewrite_performed
            rewrite_query = rewrite_res.rewritten_query
            dense_retry_attempted = True

            retry_query = retrieval_query.model_copy(update={"text": rewrite_query})
            dense_stage_limit = self.retriever.stage_limit or max(retrieval_query.limit, 1)
            retry_dense_hits = await _search_stage(
                self.retriever.dense, retry_query, dense_stage_limit
            )
            dense_count += len(retry_dense_hits)

            retry_chunks: list[Chunk] = []
            tokens = 0
            for hit in retry_dense_hits:
                chunk = self.retriever._as_chunk(hit.chunk)
                if not self.retriever._allowed(chunk, retry_query):
                    continue
                count = chunk.token_count or len(chunk.text.split())
                if retry_chunks and tokens + count > self.retriever.context_token_budget:
                    continue
                if not retry_chunks and count > self.retriever.context_token_budget:
                    chunk = chunk.model_copy(
                        update={
                            "text": " ".join(
                                chunk.text.split()[: self.retriever.context_token_budget]
                            ),
                            "token_count": self.retriever.context_token_budget,
                        }
                    )
                    count = self.retriever.context_token_budget
                retry_chunks.append(
                    chunk.model_copy(
                        update={
                            "score": hit.score,
                            "score_breakdown": hit.score_breakdown or {"dense_retry": hit.score},
                        }
                    )
                )
                tokens += count
                if len(retry_chunks) >= retrieval_query.limit:
                    break

            if retry_chunks or not result.chunks:
                final_chunks = retry_chunks

        diagnostics = RetrievalGraderDiagnostics(
            bm25_count=result.diagnostics.bm25_count,
            dense_count=dense_count,
            fusion=result.diagnostics.fusion,
            reranked=result.diagnostics.reranked,
            final_count=len(final_chunks),
            latency_ms=result.diagnostics.latency_ms,
            grader_accepted=grader_result.accept,
            grader_bm25_score=grader_result.bm25_score,
            grader_reranker_score=grader_result.reranker_score,
            reason=grader_result.reason,
            rewrite_performed=rewrite_performed,
            rewrite_query=rewrite_query,
            original_query=retrieval_query.text,
            dense_retry_attempted=dense_retry_attempted,
        )

        return RetrievalResult(
            chunks=final_chunks,
            diagnostics=diagnostics,
            query=retrieval_query,
        )


__all__ = [
    "GraderResult",
    "RetrievalGrader",
    "RetrievalGraderBridge",
    "RetrievalGraderDiagnostics",
    "RetrievalGraderResult",
]
