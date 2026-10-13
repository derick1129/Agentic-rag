"""Fallback re-ranker that tries primary then fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive.retrieval.contracts import Reranker

if TYPE_CHECKING:
    from adaptive.interfaces import RetrievalQuery
    from adaptive.retrieval.contracts import ScoredChunk


class FallbackReranker(Reranker):
    """Re-ranker that tries a primary re-ranker, then falls back to a secondary re-ranker."""

    def __init__(
        self,
        primary: Reranker,
        fallback: Reranker,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def rerank(
        self, query: RetrievalQuery, candidates: list[ScoredChunk], limit: int
    ) -> list[ScoredChunk]:
        try:
            return await self.primary.rerank(query, candidates, limit)
        except Exception:
            # Fall back to the secondary re-ranker
            return await self.fallback.rerank(query, candidates, limit)
