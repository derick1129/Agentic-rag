"""Stable seams between retrieval orchestration and search providers."""

# These runtime imports are intentional: the provider is a usable offline implementation.
# ruff: noqa: TC001, TC003, UP012

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Protocol

from adaptive.interfaces import Chunk, ChunkDraft, RetrievalQuery


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned by one retrieval stage or a reranker."""

    chunk: Chunk | ChunkDraft
    score: float
    score_breakdown: dict[str, float] | None = None


@dataclass(frozen=True)
class IndexWriteResult:
    indexed_ids: list[str]
    failed_ids: list[str]


class EmbeddingProvider(Protocol):
    """Provider seam; implementations may be local or hosted."""

    @property
    def dimensions(self) -> int: ...

    def embed(
        self, texts: Sequence[str], *, input_type: str | None = None
    ) -> list[list[float]] | Awaitable[list[list[float]]]: ...


class IndexWriter(Protocol):
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk | ChunkDraft],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> IndexWriteResult: ...

    def delete_document_version(self, document_id: str, version: int) -> None: ...


class RetrievalStage(Protocol):
    async def search(self, query: RetrievalQuery, limit: int) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: RetrievalQuery, candidates: list[ScoredChunk], limit: int
    ) -> list[ScoredChunk]: ...


class HashEmbeddingProvider:
    """Small deterministic embedding provider for offline tests and local demos."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")  # noqa: TRY003
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            values = []
            for index in range(self._dimensions):
                digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
                values.append((int.from_bytes(digest[:8], "big") / 2**63) - 1.0)
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors
