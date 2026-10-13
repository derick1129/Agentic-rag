"""Redis-backed semantic cache with RediSearch for vector similarity search."""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from adaptive.cache.contracts import CacheEntry, CacheHit, CacheLookup, CacheStore

if TYPE_CHECKING:
    from redis.client import Redis

logger = logging.getLogger(__name__)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _invalid_similarity_threshold_error() -> ValueError:
    return ValueError("similarity_threshold must be between 0 and 1")


def _missing_tenant_error() -> ValueError:
    return ValueError("cache entry requires a tenant")


def _doc_to_entry_dict(doc: object) -> dict[str, Any]:
    return {
        "id": getattr(doc, "id", ""),
        "query_embedding": (
            [float(v) for v in doc.query_embedding.split(",")]
            if hasattr(doc, "query_embedding")
            else []
        ),
        "tenant_id": doc.tenant_id if hasattr(doc, "tenant_id") else None,
        "acl": (frozenset(doc.acl.split(",")) if hasattr(doc, "acl") and doc.acl else frozenset()),
        "model_profile": doc.model_profile if hasattr(doc, "model_profile") else None,
        "response_mode": doc.response_mode if hasattr(doc, "response_mode") else None,
        "answer_text": doc.answer_text if hasattr(doc, "answer_text") else "",
        "citations": (
            json.loads(doc.citations) if hasattr(doc, "citations") and doc.citations else []
        ),
        "referenced_document_versions": (
            json.loads(doc.referenced_document_versions)
            if hasattr(doc, "referenced_document_versions") and doc.referenced_document_versions
            else {}
        ),
        "cost_usd": float(doc.cost_usd) if hasattr(doc, "cost_usd") else 0.0,
        "latency_ms": int(doc.latency_ms) if hasattr(doc, "latency_ms") else 0,
        "created_at": (
            datetime.fromtimestamp(int(doc.created_at), tz=UTC)
            if hasattr(doc, "created_at")
            else datetime.now(UTC)
        ),
        "expires_at": (
            datetime.fromtimestamp(int(doc.expires_at), tz=UTC)
            if hasattr(doc, "expires_at")
            else datetime.max.replace(tzinfo=UTC)
        ),
        "hit_count": int(doc.hit_count) if hasattr(doc, "hit_count") else 0,
    }


def _matches_lookup(entry_dict: dict[str, Any], request: CacheLookup) -> bool:
    return bool(
        entry_dict["tenant_id"] == request.context.tenant_id
        and frozenset(request.context.acl) == entry_dict["acl"]
        and entry_dict["model_profile"] == request.model_profile
        and entry_dict["response_mode"] == request.response_mode
    )


class RedisCacheStore(CacheStore):
    """Store cache entries in Redis with RediSearch for vector similarity search.

    The cache is implemented as a Redis hash for each entry, with a RediSearch index
    on the query_embedding field to enable efficient similarity search.
    """

    def __init__(
        self,
        redis_client: Redis,
        *,
        index_name: str = "semantic_cache_index",
        prefix: str = "cache:",
        similarity_threshold: float = 0.95,
        vector_dimensions: int = 1536,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise _invalid_similarity_threshold_error()
        self.redis = redis_client
        self.index_name = index_name
        self.prefix = prefix
        self.similarity_threshold = similarity_threshold
        self.vector_dimensions = vector_dimensions
        self.last_status = "miss"
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index if it does not exist."""
        try:
            # Try to describe the index to see if it exists
            self.redis.ft(self.index_name).describe()
        except Exception:
            # Index does not exist, create it
            schema = (
                # We don't store query_text, but we might for debugging
                TextField("query_text", no_stem=True),
                TagField("tenant_id"),
                TagField("acl", separator=","),  # Store ACL as a comma-separated string
                TagField("model_profile"),
                TagField("response_mode"),
                NumericField("cost_usd"),
                NumericField("latency_ms"),
                NumericField("hit_count"),
                VectorField(
                    "query_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimensions,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                NumericField("created_at"),
                NumericField("expires_at"),
                TextField("referenced_document_versions"),  # JSON string
                TextField("answer_text"),
                TextField("citations"),  # JSON string
            )
            definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            self.redis.ft(self.index_name).create_index(fields=schema, definition=definition)

    async def lookup(self, request: CacheLookup) -> CacheHit | None:
        self.last_status = "miss"
        now = datetime.now(UTC)

        # Prepare the query vector and convert to list of floats for Redis
        vector = request.query_embedding
        vector_bytes = [float(v) for v in vector]

        # Build the RediSearch query with KNN search to fetch candidates before filtering
        k = 100
        q = (
            Query(f"*=>[KNN {k} @query_embedding $vec AS score]")
            .sort_by("score")
            .return_fields(
                "id",
                "score",
                "query_embedding",
                "tenant_id",
                "acl",
                "model_profile",
                "response_mode",
                "answer_text",
                "citations",
                "referenced_document_versions",
                "cost_usd",
                "latency_ms",
                "created_at",
                "expires_at",
                "hit_count",
            )
            .dialect(2)
        )
        # In RediSearch, the score for a vector field with COSINE metric is cosine similarity.
        params = {"vec": vector_bytes}

        try:
            results = self.redis.ft(self.index_name).search(q, params=params)
        except Exception as e:
            logger.warning("Redis search failed: %s", e)
            self.last_status = "error"
            return None

        candidates: list[tuple[dict[str, Any], float]] = []
        saw_stale = False
        for doc in results.docs:
            entry_dict = _doc_to_entry_dict(doc)
            if not _matches_lookup(entry_dict, request):
                continue
            if entry_dict["expires_at"] <= now:
                saw_stale = True
                continue
            # Referenced document version checks rely on explicit invalidation when
            # document versions change, since this store does not hold a DB session.
            similarity = (
                float(doc.score)
                if hasattr(doc, "score")
                else _cosine(entry_dict["query_embedding"], request.query_embedding)
            )
            if similarity >= self.similarity_threshold:
                candidates.append((entry_dict, similarity))

        if not candidates:
            self.last_status = "stale" if saw_stale else "miss"
            return None

        # Find the candidate with the highest similarity
        best_entry, best_similarity = max(candidates, key=lambda x: x[1])

        # Update hit count in Redis
        cache_key = best_entry["id"]
        try:
            self.redis.hincrby(cache_key, "hit_count", 1)
        except Exception:
            logger.debug("Failed to increment hit_count for cache key %s", cache_key)

        entry = CacheEntry(
            query_embedding=best_entry["query_embedding"],
            tenant_id=best_entry["tenant_id"],
            acl=best_entry["acl"],
            answer_text=best_entry["answer_text"],
            citations=best_entry["citations"],
            model_profile=best_entry["model_profile"],
            response_mode=best_entry["response_mode"],
            referenced_document_versions=best_entry["referenced_document_versions"],
            cost_usd=best_entry["cost_usd"],
            latency_ms=best_entry["latency_ms"],
            created_at=best_entry["created_at"],
            expires_at=best_entry["expires_at"],
            hit_count=best_entry["hit_count"] + 1,
        )
        return CacheHit(entry=entry, similarity=best_similarity)

    async def put(self, entry: CacheEntry) -> None:
        if not entry.tenant_id:
            raise _missing_tenant_error()
        cache_key = f"{self.prefix}{uuid.uuid4()}"
        data = {
            "query_text": "",
            "tenant_id": entry.tenant_id,
            "acl": ",".join(sorted(entry.acl)),
            "model_profile": entry.model_profile,
            "response_mode": entry.response_mode,
            "answer_text": entry.answer_text,
            "citations": json.dumps(entry.citations),
            "referenced_document_versions": json.dumps(entry.referenced_document_versions),
            "cost_usd": entry.cost_usd,
            "latency_ms": entry.latency_ms,
            "created_at": int(entry.created_at.timestamp()),
            "expires_at": int(entry.expires_at.timestamp()),
            "hit_count": entry.hit_count,
            "query_embedding": ",".join(str(f) for f in entry.query_embedding),
        }
        self.redis.hset(cache_key, mapping=data)
        ttl = max(int((entry.expires_at - datetime.now(UTC)).total_seconds()), 1)
        self.redis.expire(cache_key, ttl)
        # The RediSearch index updates automatically via the configured key prefix.

    async def invalidate_document(self, document_id: str, version: int) -> int:
        # Referenced document versions are stored as a JSON string in each cache entry.
        # Full document-version invalidation requires an inverse index of document
        # versions to cache keys; for now this method records the invalidation status.
        self.last_status = "invalidated"
        return 0
