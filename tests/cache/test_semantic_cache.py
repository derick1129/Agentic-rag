from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from adaptive.cache.postgres import PostgresCacheStore
from adaptive.cache.service import CacheMetrics, SemanticCacheService
from adaptive.db.models import Base, Document, DocumentVersion
from adaptive.interfaces import (
    CacheEntry,
    CacheHit,
    CacheLookup,
    RequestContext,
    RouteDepth,
    RouteTool,
)

SIMILARITY_THRESHOLD = 0.95


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _lookup(
    *,
    tenant: str = "acme",
    acl: frozenset[str] = frozenset({"support"}),
    embedding: list[float] | None = None,
    model: str = "gpt-test",
    mode: str = "answer",
) -> CacheLookup:
    return CacheLookup(
        query_text="refund policy",
        query_embedding=embedding or [1.0, 0.0],
        context=RequestContext(tenant_id=tenant, subject_id="user-1", acl=acl),
        route_depth=RouteDepth.SINGLE_HOP,
        route_tool=RouteTool.VECTOR,
        model_profile=model,
        response_mode=mode,
    )


def _entry(  # noqa: PLR0913
    *,
    tenant: str = "acme",
    acl: frozenset[str] = frozenset({"support"}),
    embedding: list[float] | None = None,
    expires_at: datetime | None = None,
    model: str = "gpt-test",
    mode: str = "answer",
    citations: list[dict[str, object]] | None = None,
    references: dict[str, int] | None = None,
) -> CacheEntry:
    return CacheEntry(
        query_embedding=embedding or [1.0, 0.0],
        tenant_id=tenant,
        acl=acl,
        answer_text="approved",
        citations=([{"document_id": "doc-1", "version": 1}] if citations is None else citations),
        model_profile=model,
        response_mode=mode,
        referenced_document_versions={"doc-1": 1} if references is None else references,
        cost_usd=0.01,
        latency_ms=25,
        expires_at=expires_at or datetime.now(UTC) + timedelta(hours=1),
    )


async def _put(store: PostgresCacheStore, entry: CacheEntry) -> None:
    await store.put(entry)


def _active_document(session: Session, document_id: str = "doc-1", version: int = 1) -> None:
    if session.get(Document, document_id) is None:
        session.add(
            Document(
                id=document_id,
                tenant_id="acme",
                filename="policy.md",
                content_hash=f"hash-{version}",
                version=version,
                canonical_text="refund policy",
                acl=["support"],
                status="active",
            )
        )
    existing_version = (
        session.query(DocumentVersion)
        .filter_by(document_id=document_id, version=version)
        .one_or_none()
    )
    if existing_version is None:
        session.add(
            DocumentVersion(
                document_id=document_id,
                tenant_id="acme",
                version=version,
                content_hash=f"hash-{version}",
                canonical_text="refund policy",
                acl=["support"],
                status="active",
            )
        )
    session.flush()


@pytest.mark.asyncio
async def test_cache_hit_requires_same_tenant_and_acl(session: Session) -> None:
    _active_document(session)
    store = PostgresCacheStore(session, similarity_threshold=0.9)
    await _put(store, _entry())

    assert await store.lookup(_lookup(tenant="globex")) is None
    assert await store.lookup(_lookup(acl=frozenset({"finance"}))) is None
    assert (await store.lookup(_lookup())).entry.answer_text == "approved"


@pytest.mark.asyncio
async def test_cache_lookup_requires_cosine_similarity_threshold(session: Session) -> None:
    _active_document(session)
    store = PostgresCacheStore(session, similarity_threshold=SIMILARITY_THRESHOLD)
    await _put(store, _entry())

    assert await store.lookup(_lookup(embedding=[0.7, 0.714])) is None
    hit = await store.lookup(_lookup(embedding=[0.999, 0.01]))
    assert hit is not None
    assert hit.similarity > SIMILARITY_THRESHOLD - 0.01


@pytest.mark.asyncio
async def test_cache_lookup_rejects_expired_entries(session: Session) -> None:
    _active_document(session)
    store = PostgresCacheStore(session)
    await _put(store, _entry(expires_at=datetime.now(UTC) - timedelta(seconds=1)))

    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_cache_lookup_requires_model_and_response_mode_compatibility(
    session: Session,
) -> None:
    _active_document(session)
    store = PostgresCacheStore(session)
    await _put(store, _entry())

    assert await store.lookup(_lookup(model="other-model")) is None
    assert await store.lookup(_lookup(mode="json")) is None
    assert await store.lookup(_lookup()) is not None


@pytest.mark.asyncio
async def test_cache_lookup_validates_referenced_document_versions(session: Session) -> None:
    _active_document(session)
    store = PostgresCacheStore(session)
    await _put(store, _entry(references={"doc-1": 1}))
    assert await store.lookup(_lookup()) is not None

    session.query(DocumentVersion).filter_by(document_id="doc-1", version=1).update(
        {"status": "replaced"}
    )
    session.flush()
    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_document_version_invalidation_expires_matching_entries(session: Session) -> None:
    store = PostgresCacheStore(session)
    await _put(store, _entry(references={"doc-1": 1}))
    await _put(store, _entry(references={"doc-2": 1}))

    assert await store.invalidate_document("doc-1", 1) == 1
    assert await store.lookup(_lookup()) is None
    assert store.last_status == "stale"


@pytest.mark.asyncio
async def test_cache_service_emits_safe_status_metrics(session: Session) -> None:
    _active_document(session)
    metrics = CacheMetrics()
    events: list[dict[str, object]] = []
    service = SemanticCacheService(
        PostgresCacheStore(session), enabled=True, metrics=metrics, event_sink=events.append
    )
    await service.put(_entry())
    assert await service.lookup(_lookup()) is not None
    assert await service.lookup(_lookup(model="other-model")) is None

    assert metrics.counts["hit"] == 1
    assert metrics.counts["miss"] == 1
    assert {event["status"] for event in events} == {"hit", "miss"}
    assert all("refund policy" not in str(event) for event in events)


@pytest.mark.asyncio
async def test_disabled_cache_reports_bypass_without_query_payload(session: Session) -> None:
    metrics = CacheMetrics()
    events: list[dict[str, object]] = []
    service = SemanticCacheService(
        PostgresCacheStore(session), enabled=False, metrics=metrics, event_sink=events.append
    )

    assert await service.lookup(_lookup()) is None
    assert metrics.counts["bypass"] == 1
    assert events[0]["status"] == "bypass"
    assert "refund policy" not in str(events[0])


class _StaticCandidateStore:
    """In-memory store double that returns stored entries without DB validation."""

    def __init__(self, candidate: CacheEntry | None = None) -> None:
        self.entries: list[CacheEntry] = [candidate] if candidate is not None else []
        self.last_status = "miss"

    async def lookup(self, request: CacheLookup) -> CacheHit | None:
        if not self.entries:
            self.last_status = "miss"
            return None
        return CacheHit(entry=self.entries[-1], similarity=0.99)

    async def put(self, entry: CacheEntry) -> None:
        self.entries.append(entry)

    async def invalidate_document(
        self, tenant_id: str, document_id: str, version: int | None = None
    ) -> int:
        return 0


class _FakeReferenceChecker:
    def __init__(
        self,
        *,
        active: dict[tuple[str, str, int], frozenset[str]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.active = active or {}
        self.error = error
        self.calls: list[tuple[str, frozenset[str], dict[str, int]]] = []

    async def are_active(
        self,
        tenant_id: str,
        acl: frozenset[str],
        references: dict[str, int],
    ) -> bool:
        self.calls.append((tenant_id, acl, dict(references)))
        if self.error is not None:
            raise self.error
        if not references:
            return False
        for doc_id, version in references.items():
            doc_acl = self.active.get((tenant_id, doc_id, version))
            if doc_acl is None or not doc_acl.issubset(acl):
                return False
        return True


@pytest.mark.asyncio
async def test_service_validates_active_references_and_misses_when_replaced_or_deleted() -> None:
    checker = _FakeReferenceChecker(active={("acme", "doc-1", 1): frozenset({"support"})})
    store = _StaticCandidateStore()
    metrics = CacheMetrics()
    service = SemanticCacheService(
        store,
        enabled=True,
        reference_checker=checker,
        metrics=metrics,
    )
    await service.put(_entry(references={"doc-1": 1}))

    hit = await service.lookup(_lookup())
    assert hit is not None
    assert hit.entry.answer_text == "approved"
    assert metrics.counts["hit"] == 1

    # Simulate replaced/deleted version in authoritative store without eager cache invalidation
    checker.active.clear()
    assert await service.lookup(_lookup()) is None
    assert metrics.counts["stale"] == 1


@pytest.mark.asyncio
async def test_service_fails_closed_on_acl_change_tenant_mismatch_and_checker_error() -> None:
    # Document ACL broadened to require {"support", "legal"}, which is not a subset of {"support"}
    checker = _FakeReferenceChecker(active={("acme", "doc-1", 1): frozenset({"support", "legal"})})
    store = _StaticCandidateStore(_entry(references={"doc-1": 1}))
    metrics = CacheMetrics()
    service = SemanticCacheService(
        store,
        enabled=True,
        reference_checker=checker,
        metrics=metrics,
    )
    assert await service.lookup(_lookup(acl=frozenset({"support"}))) is None
    assert metrics.counts["stale"] == 1

    # Preserve exact ACL matching between entry and request even if checker would allow subset
    checker.active = {("acme", "doc-1", 1): frozenset({"support"})}
    assert await service.lookup(_lookup(acl=frozenset({"support", "extra"}))) is None

    # Tenant mismatch from underlying store must fail closed before returning
    cross_tenant_store = _StaticCandidateStore(
        _entry(tenant="other-tenant", references={"doc-1": 1})
    )
    cross_tenant_service = SemanticCacheService(
        cross_tenant_store,
        enabled=True,
        reference_checker=checker,
    )
    assert await cross_tenant_service.lookup(_lookup(tenant="acme")) is None

    # Checker exception must fail closed as a miss
    failing_checker = _FakeReferenceChecker(error=RuntimeError("db unavailable"))
    failing_metrics = CacheMetrics()
    failing_service = SemanticCacheService(
        _StaticCandidateStore(_entry(references={"doc-1": 1})),
        enabled=True,
        reference_checker=failing_checker,
        metrics=failing_metrics,
    )
    assert await failing_service.lookup(_lookup()) is None
    assert failing_metrics.counts["miss"] == 1


@pytest.mark.asyncio
async def test_service_rejects_entries_with_incomplete_citation_references() -> None:
    checker = _FakeReferenceChecker(
        active={
            ("acme", "doc-1", 1): frozenset({"support"}),
            ("acme", "doc-2", 1): frozenset({"support"}),
        }
    )
    store = _StaticCandidateStore()
    service = SemanticCacheService(store, enabled=True, reference_checker=checker)

    # Entry with citations but empty referenced_document_versions must not be cached
    await service.put(
        _entry(
            citations=[{"document_id": "doc-1", "version": 1}],
            references={},
        )
    )
    assert store.entries == []

    # Entry with multiple citations where one cited document is missing from references
    await service.put(
        _entry(
            citations=[
                {"document_id": "doc-1", "version": 1},
                {"document_id": "doc-2", "version": 1},
            ],
            references={"doc-1": 1},
        )
    )
    assert store.entries == []

    # Even if an incomplete entry is already in the store, lookup must fail closed
    store.entries.append(
        _entry(
            citations=[{"document_id": "doc-1", "version": 1}],
            references={},
        )
    )
    assert await service.lookup(_lookup()) is None
