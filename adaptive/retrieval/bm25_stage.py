"""BM25 retrieval stage using the rank-bm25 library."""

from __future__ import annotations

import json
from threading import Lock
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from adaptive.interfaces import Chunk, RetrievalQuery
from adaptive.retrieval.contracts import RetrievalStage, ScoredChunk

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class BM25Stage(RetrievalStage):
    """BM25 retrieval stage that loads chunks from a database and uses rank-bm25 for scoring."""

    def __init__(
        self,
        session_factory: Any,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        """
        Args:
            session_factory: A callable that returns a new database session.
            k1: BM25 parameter.
            b: BM25 parameter.
            epsilon: BM25 parameter.
        """
        self.session_factory = session_factory
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        # Mapping from tenant_id to (bm25_index, chunks, lock)
        self._indices: dict[str, Tuple[BM25Okapi, List[Chunk], Lock]] = {}
        self._lock = Lock()  # Lock for the indices dictionary

    def _get_or_create_index(self, tenant_id: str) -> Tuple[BM25Okapi, List[Chunk]]:
        """Get the BM25 index for a tenant, creating it if necessary."""
        with self._lock:
            if tenant_id in self._indices and self._indices[tenant_id][0] is not None:
                return self._indices[tenant_id][0], self._indices[tenant_id][1]
            if tenant_id not in self._indices:
                self._indices[tenant_id] = (None, [], Lock())
            tenant_lock = self._indices[tenant_id][2]

        # Now we have the tenant's lock, we can load the chunks and build the index
        with tenant_lock:
            # Double-check in case another thread built the index while we were waiting for the lock
            with self._lock:
                if tenant_id in self._indices and self._indices[tenant_id][0] is not None:
                    return self._indices[tenant_id][0], self._indices[tenant_id][1]

            # Load chunks for this tenant from the database
            session = self.session_factory()
            try:
                from adaptive.db.session import set_tenant_context
                # We need to import the Chunk model from the database
                from adaptive.db.models import Chunk as ChunkModel

                set_tenant_context(session, tenant_id)
                chunks_orm = (
                    session.query(ChunkModel)
                    .filter(ChunkModel.tenant_id == tenant_id)
                    .all()
                )
                # Convert ORM chunks to interface Chunk
                chunks: List[Chunk] = []
                for chunk_orm in chunks_orm:
                    chunks.append(
                        Chunk(
                            id=str(chunk_orm.id),
                            document_id=str(chunk_orm.document_id),
                            tenant_id=str(chunk_orm.tenant_id),
                            version=int(chunk_orm.version),
                            ordinal=int(chunk_orm.ordinal),
                            text=chunk_orm.text,
                            heading_path=chunk_orm.heading_path or [],
                            page_number=chunk_orm.page_number,
                            token_count=chunk_orm.token_count,
                            metadata=chunk_orm.metadata_json or {},
                            acl=frozenset(chunk_orm.acl or []),
                        )
                    )
                if not chunks:
                    with self._lock:
                        self._indices.pop(tenant_id, None)
                    return None, []
                # Tokenize the chunks for BM25
                tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
                bm25_index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)
                # Store the index and chunks
                self._indices[tenant_id] = (bm25_index, chunks, tenant_lock)
                return bm25_index, chunks
            finally:
                session.close()

    def invalidate(self, tenant_id: str | None = None) -> None:
        """Invalidate the cached BM25 index for a tenant, or all tenants if None."""
        with self._lock:
            if tenant_id is None:
                self._indices.clear()
            elif tenant_id in self._indices:
                del self._indices[tenant_id]

    async def search(self, query: RetrievalQuery, limit: int) -> List[ScoredChunk]:
        tenant_id = query.context.tenant_id
        bm25_index, chunks = self._get_or_create_index(tenant_id)
        if bm25_index is None or not chunks:
            return []

        # Tokenize the query
        tokenized_query = query.text.lower().split()
        # Get scores for all chunks
        scores = bm25_index.get_scores(tokenized_query)
        # Get the top `limit` chunks by score
        # We'll get the indices of the top scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        results: List[ScoredChunk] = []
        for idx in top_indices:
            chunk = chunks[idx]
            # The HybridRetrieverService will do further filtering by ACL and document_ids
            # But we can also do some filtering here if we want to reduce the candidates.
            # We'll leave it to the service for now.
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(scores[idx]),
                )
            )
        return results
