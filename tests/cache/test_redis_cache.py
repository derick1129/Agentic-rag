"""Tests for tenant-scoped Redis reverse indexes and invalidation."""

from __future__ import annotations

import fnmatch
import math
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from adaptive.cache.redis import RedisCacheStore
from adaptive.interfaces import CacheEntry, CacheLookup, RequestContext, RouteDepth, RouteTool

EXPECTED_COUNT_TWO = 2


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class FakeRediSearch:
    def __init__(self, client: FakeRedisClient, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def describe(self) -> dict[str, object]:
        if self.index_name not in self.client.indexes:
            msg = f"Index {self.index_name} does not exist"
            raise RuntimeError(msg)
        return {"index_name": self.index_name}

    def create_index(self, fields: object, definition: object) -> None:
        self.client.indexes[self.index_name] = {"fields": fields, "definition": definition}

    def search(self, query: object, params: dict[str, object] | None = None) -> object:
        target_vec = (params or {}).get("vec", [])
        if not isinstance(target_vec, list):
            target_vec = []

        now = time.time()
        candidates: list[tuple[float, object]] = []
        for key, val in list(self.client.hashes.items()):
            if key in self.client.expires and self.client.expires[key] <= now:
                continue
            raw_emb = val.get("query_embedding", "")
            if isinstance(raw_emb, str) and raw_emb:
                emb = [float(x) for x in raw_emb.split(",")]
                score = _cosine(emb, target_vec)
            else:
                score = 0.0
            doc = type("Doc", (), {"id": key, "score": score, **val})()
            candidates.append((score, doc))

        candidates.sort(key=lambda x: x[0], reverse=True)
        docs = [c[1] for c in candidates]
        return type("SearchResult", (), {"docs": docs, "total": len(docs)})()


class FakeRedisPipeline:
    def __init__(self, client: FakeRedisClient, transaction: bool = True) -> None:
        self.client = client
        self.transaction = transaction
        self.commands: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def hset(self, key: str, mapping: dict[str, str]) -> FakeRedisPipeline:
        self.commands.append((self.client.hset, (key,), {"mapping": mapping}))
        return self

    def expire(self, key: str, ttl: int) -> FakeRedisPipeline:
        self.commands.append((self.client.expire, (key, ttl), {}))
        return self

    def sadd(self, key: str, *members: str) -> FakeRedisPipeline:
        self.commands.append((self.client.sadd, (key, *members), {}))
        return self

    def srem(self, key: str, *members: str) -> FakeRedisPipeline:
        self.commands.append((self.client.srem, (key, *members), {}))
        return self

    def execute(self) -> list[object]:
        results: list[object] = []
        for fn, args, kwargs in self.commands:
            results.append(fn(*args, **kwargs))
        return results


class FakeRedisClient:
    """Faithful in-memory Redis Stack test double."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.expires: dict[str, float] = {}
        self.indexes: dict[str, object] = {}

    def ft(self, index_name: str = "semantic_cache_index") -> FakeRediSearch:
        return FakeRediSearch(self, index_name)

    def pipeline(self, transaction: bool = True) -> FakeRedisPipeline:
        return FakeRedisPipeline(self, transaction=transaction)

    def ttl(self, key: str) -> int:
        now = time.time()
        if key not in self.hashes and key not in self.sets:
            return -2
        if key not in self.expires:
            return -1
        rem = int(self.expires[key] - now)
        if rem <= 0:
            self._purge(key)
            return -2
        return rem

    def _purge(self, key: str) -> None:
        self.hashes.pop(key, None)
        self.sets.pop(key, None)
        self.expires.pop(key, None)

    def _check_expired(self, key: str) -> None:
        if key in self.expires and self.expires[key] <= time.time():
            self._purge(key)

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._check_expired(key)
        if key not in self.hashes:
            self.hashes[key] = {}
        self.hashes[key].update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        self._check_expired(key)
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        self._check_expired(key)
        curr = int(self.hashes.get(key, {}).get(field, "0"))
        new_val = curr + amount
        if key in self.hashes:
            self.hashes[key][field] = str(new_val)
        return new_val

    def expire(self, key: str, ttl: int) -> int:
        if key not in self.hashes and key not in self.sets:
            return 0
        self.expires[key] = time.time() + ttl
        return 1

    def sadd(self, key: str, *members: str) -> int:
        self._check_expired(key)
        if key not in self.sets:
            self.sets[key] = set()
        prev_len = len(self.sets[key])
        self.sets[key].update(members)
        return len(self.sets[key]) - prev_len

    def smembers(self, key: str) -> set[str]:
        self._check_expired(key)
        return set(self.sets.get(key, set()))

    def srem(self, key: str, *members: str) -> int:
        self._check_expired(key)
        if key not in self.sets:
            return 0
        removed = 0
        for m in members:
            if m in self.sets[key]:
                self.sets[key].remove(m)
                removed += 1
        return removed

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            self._check_expired(k)
            had = False
            if k in self.hashes:
                del self.hashes[k]
                had = True
            if k in self.sets:
                del self.sets[k]
                had = True
            self.expires.pop(k, None)
            if had:
                count += 1
        return count

    def keys(self, pattern: str) -> list[str]:
        now = time.time()
        res: list[str] = []
        all_keys = set(self.hashes.keys()) | set(self.sets.keys())
        for k in all_keys:
            if k in self.expires and self.expires[k] <= now:
                self._purge(k)
                continue
            if fnmatch.fnmatch(k, pattern):
                res.append(k)
        return res


def _make_lookup(
    *,
    tenant: str = "acme",
    acl: frozenset[str] = frozenset({"support"}),
    embedding: list[float] | None = None,
) -> CacheLookup:
    return CacheLookup(
        query_text="what is the return policy?",
        query_embedding=embedding or [1.0, 0.0],
        context=RequestContext(tenant_id=tenant, subject_id="user-1", acl=acl),
        route_depth=RouteDepth.SINGLE_HOP,
        route_tool=RouteTool.VECTOR,
        model_profile="default",
        response_mode="direct",
    )


def _make_entry(
    *,
    tenant: str = "acme",
    references: dict[str, int] | None = None,
    embedding: list[float] | None = None,
    answer_text: str = "30 days return window",
    citations: list[dict[str, object]] | None = None,
) -> CacheEntry:
    refs = {"doc-1": 1} if references is None else references
    cits = (
        [{"document_id": doc_id, "version": ver} for doc_id, ver in refs.items()]
        if citations is None
        else citations
    )
    return CacheEntry(
        query_embedding=embedding or [1.0, 0.0],
        tenant_id=tenant,
        acl=frozenset({"support"}),
        answer_text=answer_text,
        citations=cits,
        model_profile="default",
        response_mode="direct",
        referenced_document_versions=refs,
        cost_usd=0.001,
        latency_ms=15,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_atomic_entry_and_index_storage() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry = _make_entry(
        tenant="acme",
        references={"doc-1": 1, "doc-2": 2},
    )
    await store.put(entry)

    # Entry hash exists with tenant prefix in key
    cache_keys = client.keys("cache:acme:*")
    assert len(cache_keys) == 1
    cache_key = cache_keys[0]

    # Reverse sets exist and contain the cache key
    v1_set = client.smembers("cache:ref:acme:doc-1:v:1")
    all1_set = client.smembers("cache:ref:acme:doc-1:all")
    v2_set = client.smembers("cache:ref:acme:doc-2:v:2")
    all2_set = client.smembers("cache:ref:acme:doc-2:all")

    assert cache_key in v1_set
    assert cache_key in all1_set
    assert cache_key in v2_set
    assert cache_key in all2_set

    # TTL is set on cache entry and reverse sets
    assert client.ttl(cache_key) > 0
    assert client.ttl("cache:ref:acme:doc-1:v:1") > 0
    assert client.ttl("cache:ref:acme:doc-1:all") > 0


@pytest.mark.asyncio
async def test_cleanup_on_invalidation() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    await store.put(_make_entry(tenant="acme", references={"doc-1": 1}))
    cache_keys = client.keys("cache:acme:*")
    assert len(cache_keys) == 1
    cache_key = cache_keys[0]

    count = await store.invalidate_document("acme", "doc-1", 1)
    assert count == 1

    # Live cache key deleted
    assert client.hgetall(cache_key) == {}
    # Reverse set cleaned up
    assert client.smembers("cache:ref:acme:doc-1:v:1") == set()
    assert client.smembers("cache:ref:acme:doc-1:all") == set()


@pytest.mark.asyncio
async def test_exact_version_invalidation() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry_v1 = _make_entry(
        tenant="acme",
        embedding=[1.0, 0.0],
        references={"doc-1": 1},
        answer_text="Version 1 policy",
    )
    entry_v2 = _make_entry(
        tenant="acme",
        embedding=[0.0, 1.0],
        references={"doc-1": 2},
        answer_text="Version 2 policy",
    )

    await store.put(entry_v1)
    await store.put(entry_v2)

    # Invalidate ONLY version 1
    count = await store.invalidate_document("acme", "doc-1", 1)
    assert count == 1

    # Version 1 misses
    hit_v1 = await store.lookup(_make_lookup(tenant="acme", embedding=[1.0, 0.0]))
    assert hit_v1 is None

    # Version 2 remains intact and servable
    hit_v2 = await store.lookup(_make_lookup(tenant="acme", embedding=[0.0, 1.0]))
    assert hit_v2 is not None
    assert hit_v2.entry.answer_text == "Version 2 policy"


@pytest.mark.asyncio
async def test_whole_document_invalidation() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry_v1 = _make_entry(
        tenant="acme",
        embedding=[1.0, 0.0],
        references={"doc-1": 1},
        answer_text="Version 1",
    )
    entry_v2 = _make_entry(
        tenant="acme",
        embedding=[0.0, 1.0],
        references={"doc-1": 2},
        answer_text="Version 2",
    )

    await store.put(entry_v1)
    await store.put(entry_v2)

    # Invalidate document completely (version=None)
    count = await store.invalidate_document("acme", "doc-1", None)
    assert count == EXPECTED_COUNT_TWO

    assert await store.lookup(_make_lookup(tenant="acme", embedding=[1.0, 0.0])) is None
    assert await store.lookup(_make_lookup(tenant="acme", embedding=[0.0, 1.0])) is None
    assert client.smembers("cache:ref:acme:doc-1:all") == set()
    assert client.smembers("cache:ref:acme:doc-1:v:1") == set()
    assert client.smembers("cache:ref:acme:doc-1:v:2") == set()


@pytest.mark.asyncio
async def test_tenant_isolation() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry_acme = _make_entry(
        tenant="acme",
        embedding=[1.0, 0.0],
        references={"doc-1": 1},
        answer_text="Acme policy",
    )
    entry_globex = _make_entry(
        tenant="globex",
        embedding=[1.0, 0.0],
        references={"doc-1": 1},
        answer_text="Globex policy",
    )

    await store.put(entry_acme)
    await store.put(entry_globex)

    # Invalidate for Acme only
    count = await store.invalidate_document("acme", "doc-1", 1)
    assert count == 1

    # Acme entry is gone
    assert await store.lookup(_make_lookup(tenant="acme", embedding=[1.0, 0.0])) is None

    # Globex entry is untouched and hits
    hit_globex = await store.lookup(_make_lookup(tenant="globex", embedding=[1.0, 0.0]))
    assert hit_globex is not None
    assert hit_globex.entry.answer_text == "Globex policy"

    # Reverse sets for Globex remain intact
    assert len(client.smembers("cache:ref:globex:doc-1:v:1")) == 1


@pytest.mark.asyncio
async def test_exact_invalidation_count() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    # Two entries referencing doc-1:1
    await store.put(_make_entry(tenant="acme", embedding=[1.0, 0.0], references={"doc-1": 1}))
    await store.put(_make_entry(tenant="acme", embedding=[0.0, 1.0], references={"doc-1": 1}))
    # One entry referencing doc-2:1
    await store.put(_make_entry(tenant="acme", embedding=[0.7, 0.7], references={"doc-2": 1}))

    count = await store.invalidate_document("acme", "doc-1", 1)
    assert count == EXPECTED_COUNT_TWO

    # Second invalidation should return 0
    count_second = await store.invalidate_document("acme", "doc-1", 1)
    assert count_second == 0


@pytest.mark.asyncio
async def test_stale_reverse_members_tolerated() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    await store.put(_make_entry(tenant="acme", references={"doc-1": 1}))
    cache_keys = client.keys("cache:acme:*")
    assert len(cache_keys) == 1
    live_key = cache_keys[0]

    # Artificially delete the live cache key (simulating Redis TTL expiry)
    client.delete(live_key)

    # Reverse set still contains live_key member
    assert live_key in client.smembers("cache:ref:acme:doc-1:v:1")

    # Invalidation should handle the stale member cleanly without counting it as deleted
    count = await store.invalidate_document("acme", "doc-1", 1)
    assert count == 0
    assert store.last_status == "miss"
    # Reverse set was cleaned up
    assert client.smembers("cache:ref:acme:doc-1:v:1") == set()


@pytest.mark.asyncio
async def test_index_write_failure_leaves_no_servable_entry() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry = _make_entry(tenant="acme", references={"doc-1": 1})

    # Cause pipeline.execute to fail
    with (
        patch.object(FakeRedisPipeline, "execute", side_effect=RuntimeError("connection error")),
        pytest.raises(RuntimeError, match="connection error"),
    ):
        await store.put(entry)

    # Assert no cache hash key was left behind in Redis
    assert client.keys("cache:acme:*") == []

    # Lookup produces a miss
    hit = await store.lookup(_make_lookup(tenant="acme"))
    assert hit is None


@pytest.mark.asyncio
async def test_put_requires_non_empty_tenant() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry = _make_entry(tenant="")
    with pytest.raises(ValueError, match="cache entry requires a tenant"):
        await store.put(entry)


@pytest.mark.asyncio
async def test_put_requires_complete_references_if_citations_present() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    entry = _make_entry(
        tenant="acme",
        citations=[{"document_id": "doc-1", "version": 1}],
        references={},
    )
    msg = "cache entry requires complete referenced document versions"
    with pytest.raises(ValueError, match=msg):
        await store.put(entry)


@pytest.mark.asyncio
async def test_invalidate_requires_tenant_and_doc_id() -> None:
    client = FakeRedisClient()
    store = RedisCacheStore(client, prefix="cache:")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tenant_id is required"):
        await store.invalidate_document("", "doc-1", 1)

    with pytest.raises(ValueError, match="document_id is required"):
        await store.invalidate_document("acme", "", 1)
