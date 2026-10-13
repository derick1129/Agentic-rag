"""Cache orchestration and safe observability hooks."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from adaptive.cache.references import has_complete_references
from adaptive.observability.attributes import query_hash, scope_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from adaptive.cache.contracts import (
        CacheEntry,
        CacheHit,
        CacheLookup,
        CacheStore,
        ReferenceChecker,
    )


class CacheMetrics:
    """Deterministic counters suitable for tests or an application metrics adapter."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def record(self, status: str) -> None:
        self._counts[status] += 1


class SemanticCacheService:
    """Apply the cache feature flag and emit redacted cache lifecycle events."""

    def __init__(
        self,
        store: CacheStore,
        *,
        enabled: bool = True,
        reference_checker: ReferenceChecker | None = None,
        metrics: CacheMetrics | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.reference_checker = reference_checker
        self.metrics = metrics or CacheMetrics()
        self.event_sink = event_sink

    def _record(self, status: str, request: CacheLookup | None = None, **extra: object) -> None:
        self.metrics.record(status)
        event: dict[str, Any] = {"status": status, "cache.status": status, **extra}
        if request is not None:
            event.update(
                {
                    "query_hash": query_hash(request.query_text),
                    "query.hash": query_hash(request.query_text),
                    "scope_id": scope_id(request.context.tenant_id, request.context.acl),
                    "tenant.scope_id": scope_id(request.context.tenant_id, request.context.acl),
                }
            )
        if self.event_sink:
            self.event_sink(event)

    async def _validate_hit(self, hit: CacheHit, request: CacheLookup) -> str | None:
        entry = hit.entry
        if (
            entry.tenant_id != request.context.tenant_id
            or frozenset(entry.acl) != frozenset(request.context.acl)
            or not has_complete_references(entry)
        ):
            return "miss"
        if self.reference_checker is None:
            return None
        try:
            is_active = await self.reference_checker.are_active(
                request.context.tenant_id,
                frozenset(request.context.acl),
                dict(entry.referenced_document_versions),
            )
        except Exception:
            return "miss"
        return None if is_active else "stale"

    async def lookup(self, request: CacheLookup) -> CacheHit | None:
        if not self.enabled:
            self._record("bypass", request)
            return None
        if not request.context.tenant_id or not request.context.tenant_id.strip():
            self._record("miss", request)
            return None
        hit = await self.store.lookup(request)
        if hit is None:
            status = "stale" if getattr(self.store, "last_status", "miss") == "stale" else "miss"
            self._record(status, request)
            return None
        failure_status = await self._validate_hit(hit, request)
        if failure_status is not None:
            self._record(failure_status, request)
            return None
        self._record("hit", request, similarity=hit.similarity)
        return hit

    async def put(self, entry: CacheEntry) -> None:
        if self.enabled and has_complete_references(entry):
            await self.store.put(entry)

    async def invalidate_document(
        self, tenant_id: str, document_id: str | None = None, version: int | None = None
    ) -> int:
        if isinstance(document_id, int) and version is None:
            doc_id = tenant_id
            ver: int | None = document_id
            t_id = ""
        else:
            doc_id = document_id or ""
            ver = version
            t_id = tenant_id
        count = await self.store.invalidate_document(t_id, doc_id, ver)
        self._record(
            "invalidated",
            document_id_hash=query_hash(doc_id),
            count=count,
            scope_id=scope_id(t_id) if t_id else "",
        )
        return count
