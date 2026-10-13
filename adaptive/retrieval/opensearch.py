"""OpenSearch index writer and query-stage adapters."""

# The OpenSearch client is intentionally injected as a provider-specific seam.
# ruff: noqa: ANN401, PLR2004, TC003

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Sequence
from typing import Any

from adaptive.interfaces import Chunk, ChunkDraft, RetrievalQuery
from adaptive.retrieval.contracts import (
    EmbeddingProvider,
    IndexWriteResult,
    ScoredChunk,
)


async def _resolve_awaitable(
    awaitable: Awaitable[list[list[float]]],
) -> list[list[float]]:
    return await awaitable


class OpenSearchIndexWriter:
    """Indexes the exact chunks used by both BM25 and dense retrieval."""

    def __init__(
        self,
        client: Any,
        *,
        index_name: str = "adaptive-chunks-v1",
        embedding_provider: EmbeddingProvider | None = None,
        refresh: bool = False,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.embedding_provider = embedding_provider
        self.refresh = refresh
        self._tenant_by_document_version: dict[tuple[str, int], str] = {}
        self._tenant_by_document: dict[str, str] = {}
        self._chunk_ids_by_document_version: dict[tuple[str, int], list[str]] = {}

    @staticmethod
    def document_version_id(tenant_id: str, document_id: str, version: int) -> str:
        return hashlib.sha256(f"{tenant_id}:{document_id}:{version}".encode()).hexdigest()

    @staticmethod
    def chunk_id(chunk: Chunk | ChunkDraft) -> str:
        return hashlib.sha256(
            f"{chunk.tenant_id}:{chunk.document_id}:{chunk.version}:{chunk.id}".encode()
        ).hexdigest()

    def ensure_index(self) -> None:
        indices = self.client.indices() if callable(self.client.indices) else self.client.indices
        if not indices.exists(index=self.index_name):
            indices.create(index=self.index_name, body=self.index_body())

    def index_body(self) -> dict[str, Any]:
        dimensions = self.embedding_provider.dimensions if self.embedding_provider else 1536
        return {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": dimensions},
                    "tenant_id": {"type": "keyword"},
                    "acl": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "version": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "active": {"type": "boolean"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk | ChunkDraft],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> IndexWriteResult:
        if not chunks:
            return IndexWriteResult([], [])
        self.ensure_index()
        if embeddings is None and self.embedding_provider is not None:
            raw_embeddings = self.embedding_provider.embed([chunk.text for chunk in chunks])
            if inspect.isawaitable(raw_embeddings):
                embeddings = asyncio.run(_resolve_awaitable(raw_embeddings))
            else:
                embeddings = raw_embeddings
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embedding count must match chunk count")  # noqa: TRY003
        body: list[dict[str, Any]] = []
        indexed_ids: list[str] = []
        for position, chunk in enumerate(chunks):
            indexed_id = self.chunk_id(chunk)
            indexed_ids.append(indexed_id)
            self._tenant_by_document_version[(chunk.document_id, chunk.version)] = chunk.tenant_id
            self._tenant_by_document[chunk.document_id] = chunk.tenant_id
            self._chunk_ids_by_document_version.setdefault(
                (chunk.document_id, chunk.version), []
            ).append(indexed_id)
            body.extend(
                [
                    {"update": {"_index": self.index_name, "_id": indexed_id}},
                    {
                        "doc": self._source(chunk, embeddings[position] if embeddings else None),
                        "doc_as_upsert": True,
                    },
                ]
            )
        response = self.client.bulk(body=body, index=self.index_name, refresh=self.refresh)
        if response.get("errors"):
            failed = [
                indexed_ids[index]
                for index, item in enumerate(response.get("items", []))
                if next(iter(item.values())).get("status", 200) >= 300
            ]
            return IndexWriteResult([item for item in indexed_ids if item not in failed], failed)
        return IndexWriteResult(indexed_ids, [])

    def delete_document_version(self, document_id: str, version: int) -> None:
        tenant_id = self._tenant_by_document_version.get(
            (document_id, version), self._tenant_by_document.get(document_id, "")
        )
        filters = [
            {"term": {"tenant_id": tenant_id}},
            {"term": {"document_id": document_id}},
            {"term": {"version": version}},
        ]
        if hasattr(self.client, "delete_by_query"):
            self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"bool": {"filter": filters}}},
                refresh=self.refresh,
            )
            return
        for indexed_id in self._chunk_ids_by_document_version.get((document_id, version), []):
            self.client.delete(index=self.index_name, id=indexed_id, refresh=self.refresh)

    @staticmethod
    def _source(chunk: Chunk | ChunkDraft, embedding: Sequence[float] | None) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "tenant_id": chunk.tenant_id,
            "version": chunk.version,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "heading_path": chunk.heading_path,
            "page_number": chunk.page_number,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata,
            "acl": sorted(chunk.acl),
            "embedding": list(embedding) if embedding is not None else None,
            "active": True,
        }


class OpenSearchStage:
    """One filtered OpenSearch retrieval stage, parameterized by query kind."""

    def __init__(
        self,
        client: Any,
        *,
        index_name: str,
        kind: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.kind = kind
        self.embedding_provider = embedding_provider

    async def search(self, query: RetrievalQuery, limit: int) -> list[ScoredChunk]:
        must = [{"match": {"text": query.text}}] if self.kind == "bm25" else [{"match_all": {}}]
        filters: list[dict[str, Any]] = [
            {"term": {"tenant_id": query.context.tenant_id}},
            {"term": {"active": True}},
        ]
        filters.extend({"term": {"acl": acl}} for acl in query.context.acl)
        if query.document_ids:
            filters.append({"terms": {"document_id": query.document_ids}})
        query_body: dict[str, Any] = {
            "size": limit,
            "query": {"bool": {"must": must, "filter": filters}},
        }
        if self.kind == "dense":
            if self.embedding_provider is None:
                raise RuntimeError("dense stage requires an embedding provider")  # noqa: TRY003
            raw_vector = self.embedding_provider.embed([query.text], input_type="query")
            embeddings = (
                await raw_vector if inspect.isawaitable(raw_vector) else raw_vector
            )
            vector = embeddings[0]
            query_body["query"] = {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": limit,
                        "filter": {"bool": {"filter": filters}},
                    }
                }
            }
        response = self.client.search(index=self.index_name, body=query_body)
        return [
            ScoredChunk(chunk=Chunk(**hit["_source"]), score=float(hit.get("_score") or 0.0))
            for hit in response.get("hits", {}).get("hits", [])
        ]
