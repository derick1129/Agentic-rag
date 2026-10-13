"""Redis-backed semantic cache with RediSearch for vector similarity search."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar, cast

from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from adaptive.cache.contracts import CacheEntry, CacheHit, CacheLookup, CacheStore
from adaptive.cache.references import has_complete_references

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from redis.client import Redis

T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return cast("T", await value)
    return value


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


def _prepare_entry_data(entry: CacheEntry) -> dict[str, str]:
    return {
        "query_text": "",
        "tenant_id": entry.tenant_id,
        "acl": ",".join(sorted(entry.acl)),
        "model_profile": entry.model_profile,
        "response_mode": entry.response_mode,
        "answer_text": entry.answer_text,
        "citations": json.dumps(entry.citations),
        "referenced_document_versions": json.dumps(entry.referenced_document_versions),
        "cost_usd": str(entry.cost_usd),
        "latency_ms": str(entry.latency_ms),
        "created_at": str(int(entry.created_at.timestamp())),
        "expires_at": str(int(entry.expires_at.timestamp())),
        "hit_count": str(entry.hit_count),
        "query_embedding": ",".join(str(f) for f in entry.query_embedding),
    }


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
            msg = "similarity_threshold must be between 0 and 1"
            raise ValueError(msg)
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
            self.redis.ft(self.index_name).describe()
        except Exception:
            schema = (
                TextField("query_text", no_stem=True),
                TagField("tenant_id"),
                TagField("acl", separator=","),
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
                TextField("referenced_document_versions"),
                TextField("answer_text"),
                TextField("citations"),
            )
            definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            with contextlib.suppress(Exception):
                self.redis.ft(self.index_name).create_index(
                    fields=schema,
                    definition=definition,
                )

    @staticmethod
    def _parse_doc(doc: object) -> dict[str, object]:
        def _get(field: str, default: object = None) -> object:
            val = getattr(doc, field, default)
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return val

        raw_embedding = _get("query_embedding", "")
        if isinstance(raw_embedding, str) and raw_embedding:
            embedding = [float(v) for v in raw_embedding.split(",")]
        else:
            embedding = []

        raw_acl = _get("acl", "")
        acl = frozenset(raw_acl.split(",")) if isinstance(raw_acl, str) and raw_acl else frozenset()

        raw_citations = _get("citations", "[]")
        citations: list[object] = []
        if isinstance(raw_citations, str) and raw_citations:
            with contextlib.suppress(Exception):
                parsed = json.loads(raw_citations)
                if isinstance(parsed, list):
                    citations = parsed

        raw_refs = _get("referenced_document_versions", "{}")
        refs: dict[str, int] = {}
        if isinstance(raw_refs, str) and raw_refs:
            with contextlib.suppress(Exception):
                parsed = json.loads(raw_refs)
                if isinstance(parsed, dict):
                    refs = {str(k): int(v) for k, v in parsed.items()}

        raw_created = _get("created_at")
        created_at = (
            datetime.fromtimestamp(int(str(raw_created)), tz=UTC)
            if raw_created is not None
            else datetime.now(UTC)
        )

        raw_expires = _get("expires_at")
        expires_at = (
            datetime.fromtimestamp(int(str(raw_expires)), tz=UTC)
            if raw_expires is not None
            else datetime.max.replace(tzinfo=UTC)
        )

        return {
            "id": getattr(doc, "id", ""),
            "query_embedding": embedding,
            "tenant_id": _get("tenant_id"),
            "acl": acl,
            "model_profile": _get("model_profile"),
            "response_mode": _get("response_mode"),
            "answer_text": _get("answer_text", ""),
            "citations": citations,
            "referenced_document_versions": refs,
            "cost_usd": float(str(_get("cost_usd", 0.0) or 0.0)),
            "latency_ms": int(str(_get("latency_ms", 0) or 0)),
            "created_at": created_at,
            "expires_at": expires_at,
            "hit_count": int(str(_get("hit_count", 0) or 0)),
        }

    def _reverse_set_keys(self, entry: CacheEntry) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for doc_id, ver in entry.referenced_document_versions.items():
            ver_key = f"{self.prefix}ref:{entry.tenant_id}:{doc_id}:v:{int(ver)}"
            doc_key = f"{self.prefix}ref:{entry.tenant_id}:{doc_id}:all"
            keys.append((ver_key, doc_key))
        return keys

    async def _resolve_set_ttls(
        self, reverse_keys: list[tuple[str, str]], entry_ttl: int
    ) -> dict[str, int]:
        set_ttls: dict[str, int] = {}
        for ver_key, doc_key in reverse_keys:
            for k in (ver_key, doc_key):
                if k not in set_ttls:
                    existing_ttl = await _maybe_await(self.redis.ttl(k))
                    if isinstance(existing_ttl, int) and existing_ttl > 0:
                        set_ttls[k] = max(existing_ttl, entry_ttl)
                    else:
                        set_ttls[k] = entry_ttl
        return set_ttls

    async def _delete_tenant_keys(self, keys: list[str], prefix: str) -> int:
        deleted = 0
        for key in keys:
            if key.startswith(prefix):
                res = await _maybe_await(self.redis.delete(key))
                if res:
                    deleted += int(str(res))
        return deleted

    async def _invalidate_version(self, tenant_id: str, document_id: str, version: int) -> int:
        ver_set_key = f"{self.prefix}ref:{tenant_id}:{document_id}:v:{int(version)}"
        doc_set_key = f"{self.prefix}ref:{tenant_id}:{document_id}:all"
        raw_members = await _maybe_await(self.redis.smembers(ver_set_key))
        members = raw_members if isinstance(raw_members, set | list) else set()
        keys = [m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members]
        deleted_count = await self._delete_tenant_keys(keys, f"{self.prefix}{tenant_id}:")
        await _maybe_await(self.redis.delete(ver_set_key))
        if keys:
            await _maybe_await(self.redis.srem(doc_set_key, *keys))
        return deleted_count

    async def _invalidate_all(self, tenant_id: str, document_id: str) -> int:
        doc_set_key = f"{self.prefix}ref:{tenant_id}:{document_id}:all"
        raw_members = await _maybe_await(self.redis.smembers(doc_set_key))
        members = raw_members if isinstance(raw_members, set | list) else set()
        keys = [m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members]
        deleted_count = await self._delete_tenant_keys(keys, f"{self.prefix}{tenant_id}:")
        await _maybe_await(self.redis.delete(doc_set_key))
        pattern = f"{self.prefix}ref:{tenant_id}:{document_id}:v:*"
        with contextlib.suppress(Exception):
            raw_ver_keys = await _maybe_await(self.redis.keys(pattern))
            ver_keys = raw_ver_keys if isinstance(raw_ver_keys, list | set) else []
            for vk in ver_keys:
                vk_str = vk.decode("utf-8") if isinstance(vk, bytes) else str(vk)
                await _maybe_await(self.redis.delete(vk_str))
        return deleted_count

    async def lookup(self, request: CacheLookup) -> CacheHit | None:
        self.last_status = "miss"
        now = datetime.now(UTC)

        vector_bytes = [float(v) for v in request.query_embedding]
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
        params = {"vec": vector_bytes}

        try:
            search_cmd = self.redis.ft(self.index_name).search(q, params=params)
            results = await _maybe_await(search_cmd)
        except Exception:
            self.last_status = "error"
            return None

        candidates: list[tuple[dict[str, object], float]] = []
        saw_stale = False
        docs = getattr(results, "docs", [])
        for doc in docs:
            entry_dict = self._parse_doc(doc)
            if entry_dict["tenant_id"] != request.context.tenant_id:
                continue
            if frozenset(request.context.acl) != entry_dict["acl"]:
                continue
            if entry_dict["model_profile"] != request.model_profile:
                continue
            if entry_dict["response_mode"] != request.response_mode:
                continue
            expires_at = entry_dict["expires_at"]
            if isinstance(expires_at, datetime) and expires_at <= now:
                saw_stale = True
                continue

            query_emb = entry_dict["query_embedding"]
            emb_list = query_emb if isinstance(query_emb, list) else []
            similarity = (
                float(doc.score)
                if hasattr(doc, "score")
                else _cosine(emb_list, request.query_embedding)
            )
            if similarity >= self.similarity_threshold:
                candidates.append((entry_dict, similarity))

        if not candidates:
            self.last_status = "stale" if saw_stale else "miss"
            return None

        best_entry, best_similarity = max(candidates, key=lambda x: x[1])
        cache_key = str(best_entry["id"])
        with contextlib.suppress(Exception):
            await _maybe_await(self.redis.hincrby(cache_key, "hit_count", 1))

        citations = best_entry["citations"]
        refs = best_entry["referenced_document_versions"]
        entry = CacheEntry(
            query_embedding=(
                best_entry["query_embedding"]
                if isinstance(best_entry["query_embedding"], list)
                else []
            ),
            tenant_id=str(best_entry["tenant_id"]),
            acl=(best_entry["acl"] if isinstance(best_entry["acl"], frozenset) else frozenset()),
            answer_text=str(best_entry["answer_text"]),
            citations=citations if isinstance(citations, list) else [],
            model_profile=str(best_entry["model_profile"]),
            response_mode=str(best_entry["response_mode"]),
            referenced_document_versions=refs if isinstance(refs, dict) else {},
            cost_usd=float(str(best_entry["cost_usd"])),
            latency_ms=int(str(best_entry["latency_ms"])),
            created_at=(
                best_entry["created_at"]
                if isinstance(best_entry["created_at"], datetime)
                else datetime.now(UTC)
            ),
            expires_at=(
                best_entry["expires_at"]
                if isinstance(best_entry["expires_at"], datetime)
                else datetime.max.replace(tzinfo=UTC)
            ),
            hit_count=int(str(best_entry["hit_count"])) + 1,
        )
        return CacheHit(entry=entry, similarity=best_similarity)

    async def put(self, entry: CacheEntry) -> None:
        if not entry.tenant_id or not entry.tenant_id.strip():
            msg = "cache entry requires a tenant"
            raise ValueError(msg)
        if entry.citations and not has_complete_references(entry):
            msg = "cache entry requires complete referenced document versions"
            raise ValueError(msg)

        now = datetime.now(UTC)
        ttl = max(int((entry.expires_at - now).total_seconds()), 1)
        cache_key = f"{self.prefix}{entry.tenant_id}:{uuid.uuid4()}"
        data = _prepare_entry_data(entry)
        reverse_keys = self._reverse_set_keys(entry)
        set_ttls = await self._resolve_set_ttls(reverse_keys, ttl)

        pipeline = self.redis.pipeline(transaction=True)
        try:
            pipeline.hset(cache_key, mapping=data)
            pipeline.expire(cache_key, ttl)

            for ver_key, doc_key in reverse_keys:
                pipeline.sadd(ver_key, cache_key)
                pipeline.expire(ver_key, set_ttls[ver_key])
                pipeline.sadd(doc_key, cache_key)
                pipeline.expire(doc_key, set_ttls[doc_key])

            await _maybe_await(pipeline.execute())
        except Exception:
            with contextlib.suppress(Exception):
                await _maybe_await(self.redis.delete(cache_key))
            for ver_key, doc_key in reverse_keys:
                with contextlib.suppress(Exception):
                    await _maybe_await(self.redis.srem(ver_key, cache_key))
                with contextlib.suppress(Exception):
                    await _maybe_await(self.redis.srem(doc_key, cache_key))
            raise

    async def invalidate_document(
        self, tenant_id: str, document_id: str, version: int | None = None
    ) -> int:
        if not tenant_id or not tenant_id.strip():
            msg = "tenant_id is required"
            raise ValueError(msg)
        if not document_id or not document_id.strip():
            msg = "document_id is required"
            raise ValueError(msg)

        if version is not None:
            deleted_count = await self._invalidate_version(tenant_id, document_id, version)
        else:
            deleted_count = await self._invalidate_all(tenant_id, document_id)

        self.last_status = "invalidated" if deleted_count else "miss"
        return deleted_count
