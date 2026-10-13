"""Integration tests for semantic cache safety, tenant isolation, and reference checking."""

from __future__ import annotations

import fnmatch
import os
from typing import Any

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/testdb"
)
os.environ.setdefault("OPENSEARCH_URL", "http://localhost:9200")

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import adaptive.api.main as api_main
from adaptive.api.auth import authenticate_request
from adaptive.api.main import create_app
from adaptive.cache.redis import RedisCacheStore
from adaptive.cache.references import PostgresReferenceChecker
from adaptive.cache.service import SemanticCacheService
from adaptive.db.models import Base, Document, DocumentVersion
from adaptive.interfaces import (
    Chunk,
    RequestContext,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResult,
)


class FakeRediSearch:
    def __init__(self, client: FakeRedisClient, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def describe(self) -> dict[str, object]:
        return {"index_name": self.index_name}

    def create_index(self, fields: object, definition: object) -> None:
        pass

    def search(self, query: object, params: dict[str, object] | None = None) -> object:
        docs = []
        for key, val in self.client.hashes.items():
            doc = type("Doc", (), {"id": key, "score": 1.0, **val})()
            docs.append(doc)
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
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}

    def ft(self, index_name: str = "semantic_cache_index") -> FakeRediSearch:
        return FakeRediSearch(self, index_name)

    def pipeline(self, transaction: bool = True) -> FakeRedisPipeline:
        return FakeRedisPipeline(self, transaction=transaction)

    def ttl(self, key: str) -> int:
        return 3600 if key in self.hashes or key in self.sets else -2

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        if key not in self.hashes:
            self.hashes[key] = {}
        self.hashes[key].update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        curr = int(self.hashes.get(key, {}).get(field, "0"))
        new_val = curr + amount
        if key in self.hashes:
            self.hashes[key][field] = str(new_val)
        return new_val

    def expire(self, key: str, ttl: int) -> int:
        return 1

    def sadd(self, key: str, *members: str) -> int:
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].update(members)
        return len(members)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def srem(self, key: str, *members: str) -> int:
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
            if k in self.hashes:
                del self.hashes[k]
                count += 1
            if k in self.sets:
                del self.sets[k]
        return count

    def keys(self, pattern: str) -> list[str]:
        all_keys = list(self.hashes.keys()) + list(self.sets.keys())
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]


def _setup_db() -> tuple[Any, Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        Document(
            id="doc-1",
            tenant_id="acme",
            filename="policy.md",
            content_hash="h1",
            version=1,
            canonical_text="Return within 30 days.",
            acl=["support"],
            status="active",
        )
    )
    session.add(
        DocumentVersion(
            document_id="doc-1",
            tenant_id="acme",
            version=1,
            content_hash="h1",
            canonical_text="Return within 30 days.",
            acl=["support"],
            status="active",
        )
    )
    session.commit()
    return engine, session


class FakeRetriever:
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            tenant_id="acme",
            version=1,
            ordinal=0,
            text="Return within 30 days.",
            score=0.98,
            metadata={"version": 1},
        )
        diag = RetrievalDiagnostics(
            bm25_count=1,
            dense_count=1,
            fusion="rrf",
            final_count=1,
            latency_ms=10,
        )
        return RetrievalResult(
            query=query,
            chunks=[chunk],
            diagnostics=diag,
            total_candidates=1,
            stage_latencies_ms={},
        )


class FakeEmbeddingProvider:
    async def embed(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeGenerationProvider:
    async def generate(
        self, prompt: str, max_tokens: int, temperature: float = 0.0
    ) -> str:
        return "Returns are accepted within 30 days [1]."


@pytest.fixture
def cache_app_fixture(monkeypatch: pytest.MonkeyPatch):
    engine, session = _setup_db()
    redis_client = FakeRedisClient()
    store = RedisCacheStore(redis_client, prefix="cache:")  # type: ignore[arg-type]
    checker = PostgresReferenceChecker(
        session_factory=lambda: Session(engine)
    )
    cache_service = SemanticCacheService(
        store=store, enabled=True, reference_checker=checker
    )

    @asynccontextmanager
    async def fake_lifespan(app: FastAPI):
        yield

    monkeypatch.setattr(api_main, "lifespan", fake_lifespan)
    monkeypatch.setattr(api_main, "_cache_service", cache_service)
    monkeypatch.setattr(api_main, "_cache_store", store)
    monkeypatch.setattr(api_main, "_retrieval_service", FakeRetriever())
    monkeypatch.setattr(api_main, "_embedding_provider", FakeEmbeddingProvider())
    monkeypatch.setattr(api_main, "_generation_provider", FakeGenerationProvider())

    app = create_app()

    yield app, session, cache_service, store

    session.close()


def test_query_cache_hit_and_post_update_miss(cache_app_fixture: tuple[Any, ...]) -> None:
    app, session, _cache_service, _store = cache_app_fixture
    context = RequestContext(
        tenant_id="acme", subject_id="user-1", acl=frozenset({"support"})
    )
    app.dependency_overrides[authenticate_request] = lambda: context

    client = TestClient(app)
    query_payload = {"query": "what is the return window?", "generate": True}

    # 1. First query is a miss, synthesizes answer and caches it
    res1 = client.post("/v1/queries", json=query_payload)
    assert res1.status_code == status.HTTP_200_OK
    data1 = res1.json()
    assert data1["cache_status"] == "miss"
    assert "Returns are accepted" in data1["answer"]

    # 2. Second query is an immediate cache hit
    res2 = client.post("/v1/queries", json=query_payload)
    assert res2.status_code == status.HTTP_200_OK
    data2 = res2.json()
    assert data2["cache_status"] == "hit"
    assert data2["answer"] == data1["answer"]

    # 3. Simulate authoritative database transition: document version replaced
    session.query(DocumentVersion).filter_by(document_id="doc-1", version=1).update(
        {"status": "replaced"}
    )
    session.commit()

    # 4. Third query without eager invalidation fails closed (reference check produces miss)
    res3 = client.post("/v1/queries", json=query_payload)
    assert res3.status_code == status.HTTP_200_OK
    data3 = res3.json()
    assert data3["cache_status"] == "miss"


def test_query_cache_tenant_and_acl_isolation(cache_app_fixture: tuple[Any, ...]) -> None:
    app, _session, _cache_service, _store = cache_app_fixture
    acme_context = RequestContext(
        tenant_id="acme", subject_id="user-1", acl=frozenset({"support"})
    )
    app.dependency_overrides[authenticate_request] = lambda: acme_context

    client = TestClient(app)
    query_payload = {"query": "return window?", "generate": True}

    # Populate cache for acme
    res1 = client.post("/v1/queries", json=query_payload)
    assert res1.status_code == status.HTTP_200_OK
    res_hit = client.post("/v1/queries", json=query_payload)
    assert res_hit.json()["cache_status"] == "hit"

    # Query with globex tenant must NOT hit acme cache
    globex_context = RequestContext(
        tenant_id="globex", subject_id="user-2", acl=frozenset({"support"})
    )
    app.dependency_overrides[authenticate_request] = lambda: globex_context
    res_globex = client.post("/v1/queries", json=query_payload)
    assert res_globex.status_code == status.HTTP_200_OK
    assert res_globex.json()["cache_status"] == "miss"

    # Query with different ACL must NOT hit acme cache
    other_acl_context = RequestContext(
        tenant_id="acme", subject_id="user-1", acl=frozenset({"finance"})
    )
    app.dependency_overrides[authenticate_request] = lambda: other_acl_context
    res_acl = client.post("/v1/queries", json=query_payload)
    assert res_acl.status_code == status.HTTP_200_OK
    assert res_acl.json()["cache_status"] == "miss"


@pytest.mark.asyncio
async def test_query_cache_eager_invalidation(cache_app_fixture: tuple[Any, ...]) -> None:
    app, _session, cache_service, _store = cache_app_fixture
    context = RequestContext(
        tenant_id="acme", subject_id="user-1", acl=frozenset({"support"})
    )
    app.dependency_overrides[authenticate_request] = lambda: context

    client = TestClient(app)
    query_payload = {"query": "return window?", "generate": True}

    # 1. Populate cache and verify hit
    client.post("/v1/queries", json=query_payload)
    assert client.post("/v1/queries", json=query_payload).json()["cache_status"] == "hit"

    # 2. Invalidate document version
    count = await cache_service.invalidate_document("acme", "doc-1", 1)
    assert count == 1

    # 3. Next query misses
    res_post_inval = client.post("/v1/queries", json=query_payload)
    assert res_post_inval.json()["cache_status"] == "miss"
