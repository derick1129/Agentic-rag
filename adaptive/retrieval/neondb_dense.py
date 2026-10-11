"""NeonDB (PostgreSQL) dense retrieval stage using pgvector."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, List, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text as sqlalchemy_text
from sqlalchemy.orm import Session

from adaptive.interfaces import Chunk, RetrievalQuery
from adaptive.retrieval.contracts import EmbeddingProvider, RetrievalStage, ScoredChunk

if TYPE_CHECKING:
    from adaptive.db.models import Chunk as ChunkModel


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class NeonDBDenseStage(RetrievalStage):
    """Dense retrieval stage that uses NeonDB (PostgreSQL) with pgvector for similarity search."""

    def __init__(
        self,
        session_factory: Any,
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

    async def search(self, query: RetrievalQuery, limit: int) -> List[ScoredChunk]:
        session = self.session_factory()
        try:
            from adaptive.db.session import set_tenant_context

            set_tenant_context(session, query.context.tenant_id)
            # Compute the query embedding
            query_embedding_list = await _maybe_await(self.embedding_provider.embed([query.text], input_type="query"))
            query_embedding = query_embedding_list[0]
            # Ensure the embedding is the right dimension
            if len(query_embedding) != self.embedding_dimensions:
                raise ValueError(
                    f"Query embedding dimension {len(query_embedding)} does not match expected {self.embedding_dimensions}"
                )

            # We'll use raw SQL to perform the vector search
            # We assume there is a table named `chunk` with a column `embedding` of type vector
            # and we have a tenant_id column for filtering.
            # We'll also need to filter by ACL and document_ids, but we'll do that in Python after fetching candidates?
            # Alternatively, we can include the tenant_id in the SQL WHERE clause.
            # We'll do:
            #   SELECT * FROM chunk WHERE tenant_id = :tenant_id ORDER BY embedding <=> :query_embedding LIMIT :limit
            # But note: we also need to filter by ACL and document_ids. We'll do that in Python for now.
            # We'll fetch more candidates to account for filtering.
            # We'll get `limit * 3` candidates and then filter.

            # However, we also want to filter by active document versions? We'll leave that to the HybridRetrieverService.

            # We'll get the chunks and their embeddings from the database.
            # We'll use the pgvector operator <=> for cosine distance.

            # We'll get the chunk data and compute the score as 1 - cosine_distance? 
            # Actually, the <=> operator returns the cosine distance (0 is identical, 1 is opposite).
            # We want similarity: 1 - distance.

            # But note: the PostgresCacheStore uses the <=> operator and then computes cosine similarity in Python.
            # We'll do the same: we'll use the <=> operator to order by distance, and then compute the cosine similarity in Python.

            # We'll fetch the chunks and their embeddings, then compute the similarity.

            # We'll get the tenant_id from the query context.
            tenant_id = query.context.tenant_id

            # We'll get the ACL as a list of strings.
            acl_list = list(query.context.acl)

            # We'll get the document_ids if provided.
            document_ids = query.document_ids

            # We'll build the SQL query. Colon-style bindings are used with an
            # explicit pgvector bind type so the query embedding is sent as a
            # real `vector` value (no need for a ::vector cast in the string).
            sql = """
                SELECT id, document_id, tenant_id, version, ordinal, text, heading_path, page_number, token_count, metadata, acl, embedding
                FROM chunks
                WHERE tenant_id = :tenant_id
            """
            params: dict[str, Any] = {"tenant_id": tenant_id, "query_embedding": query_embedding}
            if document_ids:
                sql += " AND document_id IN :document_ids"
                params["document_ids"] = list(document_ids)
            # We'll order by the vector distance and limit to a multiple of `limit` to get candidates for filtering.
            sql += " ORDER BY embedding <=> :query_embedding"
            sql += " LIMIT :limit"
            params["limit"] = limit * 3  # Get more candidates to account for ACL filtering

            # Execute the query
            binds = [bindparam("query_embedding", type_=Vector(self.embedding_dimensions))]
            if document_ids:
                binds.append(bindparam("document_ids", expanding=True))
            statement = sqlalchemy_text(sql).bindparams(*binds)
            result = session.execute(statement, params)
            rows = result.fetchall()

            # Now we have the rows, we need to convert them to Chunk objects and compute the similarity.
            candidates: List[Tuple[Chunk, float]] = []
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

                # Check ACL: we need to see if the chunk's ACL is a subset of the user's ACL?
                # Actually, the user's ACL is the set of permissions they have. The chunk's ACL is the set of permissions required to access it.
                # The user can access the chunk if the chunk's ACL is a subset of the user's ACL.
                chunk_acl = frozenset(acl_json or [])
                if not chunk_acl.issubset(query.context.acl):
                    continue

                # Convert the chunk to an interface Chunk
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
                embedding: List[float] = list(embedding_json)
                # Compute cosine similarity
                similarity = _cosine(query_embedding, embedding)
                candidates.append((chunk, similarity))

            # Sort by similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Take the top `limit`
            top_candidates = candidates[:limit]

            results: List[ScoredChunk] = []
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
