"""PostgreSQL-backed semantic cache with a SQLite-compatible test path."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from adaptive.cache.contracts import CacheEntry, CacheHit, CacheLookup, CacheStore
from adaptive.db.models import DocumentVersion
from adaptive.db.models import SemanticCacheEntry as CacheModel

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


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


class PostgresCacheStore(CacheStore):
    """Store cache entries in the control-plane database.

    PostgreSQL persists embeddings as ``vector(1536)`` through the ORM type. The
    Python ranking path is intentionally retained for deterministic SQLite tests
    and for deployments where a small candidate set is acceptable.
    """

    def __init__(self, session: Session, *, similarity_threshold: float = 0.95) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")  # noqa: TRY003
        self.session = session
        self.similarity_threshold = similarity_threshold
        self.last_status = "miss"

    async def lookup(self, request: CacheLookup) -> CacheHit | None:
        self.last_status = "miss"
        now = datetime.now(UTC)
        query = select(CacheModel).where(CacheModel.tenant_id == request.context.tenant_id)
        use_pgvector = (
            self.session.bind is not None and self.session.bind.dialect.name == "postgresql"
        )
        if use_pgvector:
            distance = CacheModel.query_embedding.op("<=>")(request.query_embedding)
            query = query.order_by(distance)
        models = self.session.scalars(query).all()
        candidates: list[tuple[CacheModel, float]] = []
        saw_stale = False
        for model in models:
            if frozenset(model.acl) != request.context.acl:
                continue
            if model.expires_at is not None and _utc(model.expires_at) <= now:
                saw_stale = True
                continue
            if (
                model.model_profile != request.model_profile
                or model.response_mode != request.response_mode
            ):
                continue
            if not await self._references_are_active(model, request):
                saw_stale = True
                continue
            similarity = _cosine(model.query_embedding, request.query_embedding)
            if similarity >= self.similarity_threshold:
                candidates.append((model, similarity))
        if not candidates:
            self.last_status = "stale" if saw_stale else "miss"
            return None
        model, similarity = max(candidates, key=lambda candidate: candidate[1])
        model.hit_count += 1
        self.session.flush()
        return CacheHit(entry=self._to_entry(model), similarity=similarity)

    async def put(self, entry: CacheEntry) -> None:
        if not entry.tenant_id:
            raise ValueError("cache entry requires a tenant")  # noqa: TRY003
        self.session.add(
            CacheModel(
                tenant_id=entry.tenant_id,
                acl=sorted(entry.acl),
                query_embedding=entry.query_embedding,
                answer_text=entry.answer_text,
                citations=entry.citations,
                model_profile=entry.model_profile,
                response_mode=entry.response_mode,
                referenced_document_versions=entry.referenced_document_versions,
                expires_at=entry.expires_at,
                hit_count=entry.hit_count,
                cost_usd=entry.cost_usd,
                latency_ms=entry.latency_ms,
                created_at=entry.created_at,
            )
        )
        self.session.flush()

    async def invalidate_document(
        self, tenant_id: str, document_id: str | None = None, version: int | None = None
    ) -> int:
        now = datetime.now(UTC)
        if isinstance(document_id, int) and version is None:
            doc_id = tenant_id
            ver: int | None = document_id
            t_id: str | None = None
        else:
            doc_id = document_id or ""
            ver = version
            t_id = tenant_id

        entries = self.session.scalars(select(CacheModel)).all()
        count = 0
        for entry in entries:
            if t_id is not None and entry.tenant_id != t_id:
                continue
            if not doc_id:
                continue
            refs = entry.referenced_document_versions or {}
            if doc_id not in refs:
                continue
            if ver is not None and refs.get(doc_id) != ver:
                continue
            if entry.expires_at is not None and _utc(entry.expires_at) <= now:
                continue
            entry.expires_at = now
            count += 1
        self.session.flush()
        self.last_status = "invalidated" if count else "miss"
        return count

    async def _references_are_active(self, model: CacheModel, request: CacheLookup) -> bool:
        references = model.referenced_document_versions or {}
        if not references:
            return True
        for document_id, version in references.items():
            active_row = self.session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.version == int(version),
                    DocumentVersion.tenant_id == request.context.tenant_id,
                    DocumentVersion.status == "active",
                )
            )
            if active_row is None:
                return False
            active_acl = set(active_row.acl) if active_row.acl else set()
            if not active_acl.issubset(request.context.acl):
                return False
        return True

    @staticmethod
    def _to_entry(model: CacheModel) -> CacheEntry:
        expires_at = model.expires_at or datetime.max.replace(tzinfo=UTC)
        return CacheEntry(
            query_embedding=list(model.query_embedding),
            tenant_id=model.tenant_id,
            acl=frozenset(model.acl),
            answer_text=model.answer_text,
            citations=list(model.citations),
            model_profile=model.model_profile,
            response_mode=model.response_mode,
            referenced_document_versions={
                str(key): int(value) for key, value in model.referenced_document_versions.items()
            },
            cost_usd=model.cost_usd,
            latency_ms=model.latency_ms,
            created_at=_utc(model.created_at),
            expires_at=_utc(expires_at),
            hit_count=model.hit_count,
        )
