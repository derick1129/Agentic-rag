"""NeonDB (PostgreSQL) dense retrieval stage using pgvector."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, TypeVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam
from sqlalchemy import text as sqlalchemy_text

from adaptive.db.session import set_tenant_context
from adaptive.interfaces import Chunk, RetrievalQuery
from adaptive.retrieval.contracts import EmbeddingProvider, RetrievalStage, ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.orm import Session

T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _dimension_mismatch_error(actual: int, expected: int) -> ValueError:
    return ValueError(f"Query embedding dimension {actual} does not match expected {expected}")


class NeonDBDenseStage(RetrievalStage):
    """Dense retrieval stage that uses NeonDB (PostgreSQL) with pgvector for similarity search."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedding_provider: EmbeddingProvider,
        *,
        embedding_dimensions: int = 1536,
    ) -> None:
        """
        Args:
            session_factory: A callable that returns a new database session.
            embedding_provider: Provider to compute the query embedding.
            embedding_dimensions: The dimensionality of the embeddings.
        """
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.embedding_dimensions = embedding_dimensions

    async def search(self, query: RetrievalQuery, limit: int) -> list[ScoredChunk]:
        session = self.session_factory()
        try:
            set_tenant_context(session, query.context.tenant_id)
            # Compute the query embedding
            query_embedding_list = await _maybe_await(
                self.embedding_provider.embed([query.text], input_type="query")
            )
            query_embedding = query_embedding_list[0]
            # Ensure the embedding is the right dimension
            if len(query_embedding) != self.embedding_dimensions:
                raise _dimension_mismatch_error(len(query_embedding), self.embedding_dimensions)

            # Use the pgvector <=> operator to order candidates by cosine distance,
            # then filter by ACL and compute cosine similarity in Python.
            tenant_id = query.context.tenant_id
            document_ids = query.document_ids

            # Colon-style bindings are used with an explicit pgvector bind type so
            # the query embedding is sent as a real `vector` value.
            sql = """
                SELECT
                    id, document_id, tenant_id, version, ordinal, text,
                    heading_path, page_number, token_count, metadata, acl, embedding
                FROM chunks
                WHERE tenant_id = :tenant_id
            """
            params: dict[str, Any] = {
                "tenant_id": tenant_id,
                "query_embedding": query_embedding,
            }
            if document_ids:
                sql += " AND document_id IN :document_ids"
                params["document_ids"] = list(document_ids)
            # Order by vector distance and fetch extra candidates to account for ACL filtering.
            sql += " ORDER BY embedding <=> :query_embedding"
            sql += " LIMIT :limit"
            params["limit"] = limit * 3

            binds = [bindparam("query_embedding", type_=Vector(self.embedding_dimensions))]
            if document_ids:
                binds.append(bindparam("document_ids", expanding=True))
            statement = sqlalchemy_text(sql).bindparams(*binds)
            result = session.execute(statement, params)
            rows = result.fetchall()

            # Convert rows to Chunk objects and compute cosine similarity.
            candidates: list[tuple[Chunk, float]] = []
            for row in rows:
                (
                    chunk_id,
                    document_id,
                    tenant_id_from_db,
                    version,
                    ordinal,
                    text,
                    heading_path_json,
                    page_number,
                    token_count,
                    metadata_json,
                    acl_json,
                    embedding_json,
                ) = row

                # The user can access the chunk if the chunk's ACL is a subset of the user's ACL.
                chunk_acl = frozenset(acl_json or [])
                if not chunk_acl.issubset(query.context.acl):
                    continue

                chunk = Chunk(
                    id=str(chunk_id),
                    document_id=str(document_id),
                    tenant_id=str(tenant_id_from_db),
                    version=int(version),
                    ordinal=int(ordinal),
                    text=text,
                    heading_path=heading_path_json or [],
                    page_number=page_number,
                    token_count=token_count,
                    metadata=metadata_json or {},
                    acl=chunk_acl,
                )

                # The embedding is a pgvector `vector` column; psycopg returns it
                # as a native list of floats.
                embedding: list[float] = list(embedding_json)
                similarity = _cosine(query_embedding, embedding)
                candidates.append((chunk, similarity))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:limit]

            results: list[ScoredChunk] = []
            for chunk, similarity in top_candidates:
                results.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=similarity,
                    )
                )
            return results
        finally:
            session.close()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
