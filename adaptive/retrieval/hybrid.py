"""Hybrid retrieval orchestration independent of a concrete search backend."""

# Backends are deliberately duck-typed so sync and async provider adapters can coexist.
# ruff: noqa: ANN401, PLR0913

from __future__ import annotations

import inspect
import time
from typing import Any

from adaptive.interfaces import Chunk, RetrievalDiagnostics, RetrievalQuery, RetrievalResult
from adaptive.retrieval.contracts import Reranker, RetrievalStage, ScoredChunk


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _search_stage(stage: Any, query: RetrievalQuery, limit: int) -> list[ScoredChunk]:
    if hasattr(stage, "search"):
        stage_hits: list[ScoredChunk] = await _maybe_await(stage.search(query, limit))
        return stage_hits
    if callable(stage):
        callable_hits: list[ScoredChunk] = await _maybe_await(stage(query, limit))
        return callable_hits
    return []


class HybridRetrieverService:
    def __init__(
        self,
        *,
        bm25: RetrievalStage,
        dense: RetrievalStage,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        context_token_budget: int = 4000,
        stage_limit: int | None = None,
    ) -> None:
        if rrf_k < 1 or context_token_budget < 1:
            raise ValueError("rrf_k and context_token_budget must be positive")  # noqa: TRY003
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.context_token_budget = context_token_budget
        self.stage_limit = stage_limit

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started = time.monotonic()
        stage_limit = self.stage_limit or max(query.limit, 1)
        bm25_hits, dense_hits = (
            await _search_stage(self.bm25, query, stage_limit),
            await _search_stage(self.dense, query, stage_limit),
        )
        fused: dict[str, ScoredChunk] = {}
        for stage_hits in (bm25_hits, dense_hits):
            for rank, hit in enumerate(stage_hits, start=1):
                chunk = self._as_chunk(hit.chunk)
                if not self._allowed(chunk, query):
                    continue
                contribution = 1.0 / (self.rrf_k + rank)
                current = fused.get(chunk.id)
                if current is None:
                    fused[chunk.id] = ScoredChunk(
                        chunk=chunk,
                        score=contribution,
                        score_breakdown={"rrf": contribution},
                    )
                else:
                    breakdown = dict(current.score_breakdown or {})
                    breakdown["rrf"] = breakdown.get("rrf", 0.0) + contribution
                    fused[chunk.id] = ScoredChunk(
                        chunk=current.chunk,
                        score=current.score + contribution,
                        score_breakdown=breakdown,
                    )
        candidates = sorted(fused.values(), key=lambda item: (-item.score, item.chunk.id))
        reranked = False
        if self.reranker is not None and candidates:
            candidates = await _maybe_await(self.reranker.rerank(query, candidates, query.limit))
            reranked = True
        selected: list[Chunk] = []
        tokens = 0
        for candidate in candidates:
            chunk = self._as_chunk(candidate.chunk)
            count = chunk.token_count or len(chunk.text.split())
            if selected and tokens + count > self.context_token_budget:
                continue
            if not selected and count > self.context_token_budget:
                chunk = chunk.model_copy(
                    update={
                        "text": " ".join(chunk.text.split()[: self.context_token_budget]),
                        "token_count": self.context_token_budget,
                    }
                )
                count = self.context_token_budget
            selected.append(
                chunk.model_copy(
                    update={
                        "score": candidate.score,
                        "score_breakdown": candidate.score_breakdown or {},
                    }
                )
            )
            tokens += count
            if len(selected) >= query.limit:
                break
        diagnostics = RetrievalDiagnostics(
            bm25_count=len(bm25_hits),
            dense_count=len(dense_hits),
            reranked=reranked,
            final_count=len(selected),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return RetrievalResult(chunks=selected, diagnostics=diagnostics, query=query)

    @staticmethod
    def _as_chunk(value: Chunk | Any) -> Chunk:
        return value if isinstance(value, Chunk) else Chunk(**value.model_dump())

    @staticmethod
    def _allowed(chunk: Chunk, query: RetrievalQuery) -> bool:
        return (
            chunk.tenant_id == query.context.tenant_id
            and query.context.can_access(chunk.acl)
            and (query.document_ids is None or chunk.document_id in query.document_ids)
        )
