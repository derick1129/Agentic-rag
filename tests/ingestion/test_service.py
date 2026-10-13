from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from adaptive.db.models import Base, DocumentVersion
from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.parsers import parse_bytes
from adaptive.ingestion.service import IngestionService
from adaptive.interfaces import IngestionStatus, RequestContext

VERSION_TWO = 2


@dataclass
class FakeSource:
    data: bytes
    filename: str = "guide.md"
    mime_type: str = "text/markdown"
    reads: int = 0

    async def read(self) -> SourcePayload:
        self.reads += 1
        return SourcePayload(data=self.data, filename=self.filename, mime_type=self.mime_type)


@dataclass
class FakeIndexer:
    indexed: list[list[str]] = field(default_factory=list)

    async def index_chunks(self, chunks: list):  # noqa: ANN401
        self.indexed.append([chunk.id for chunk in chunks])


@pytest.mark.asyncio
async def test_service_deduplicates_same_tenant_content_hash():
    source = FakeSource(b"# Guide\n\nSame content")
    service = IngestionService()
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    first = await service.submit(source, context)
    second = await service.submit(source, context)
    first_done = await service.process(first.id)
    second_done = await service.process(second.id)

    assert first_done.status == IngestionStatus.ACTIVE
    assert second_done.status == IngestionStatus.ACTIVE
    assert second_done.document_id == first_done.document_id
    assert service.active_document_count(context) == 1


@pytest.mark.asyncio
async def test_service_replaces_previous_version_and_is_retry_safe():
    service = IngestionService(indexer=FakeIndexer())
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    first = await service.submit(FakeSource(b"# Guide\n\nVersion one"), context)
    first_done = await service.process(first.id)
    replacement = await service.submit(FakeSource(b"# Guide\n\nVersion two"), context)
    replacement_done = await service.process(replacement.id)
    retried = await service.process(replacement.id)

    assert first_done.status == IngestionStatus.ACTIVE
    assert replacement_done.status == IngestionStatus.ACTIVE
    assert replacement_done.document_id == first_done.document_id
    assert retried.status == IngestionStatus.ACTIVE
    assert service.active_version(first_done.document_id) == VERSION_TWO
    assert service.active_chunk_count(first_done.document_id) == 1


@pytest.mark.asyncio
async def test_service_records_failed_stage_and_can_retry():
    calls = 0

    def parser(payload: bytes, mime_type: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("bad input")  # noqa: TRY003
        return parse_bytes(payload, mime_type)

    service = IngestionService(parser=parser)
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    job = await service.submit(FakeSource(b"not valid"), context)
    failed = await service.process(job.id)

    assert failed.status == IngestionStatus.FAILED
    assert failed.current_stage == "parsing"
    assert failed.error_code == "parse_error"
    assert failed.attempts == 1

    retried = await service.process(job.id)

    assert retried.status == IngestionStatus.ACTIVE
    assert retried.attempts == 1


@pytest.mark.asyncio
async def test_service_persists_and_activates_replacement_versions():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # SQLite ignores the PostgreSQL-only partial-index predicate from the model.
        session.execute(text("DROP INDEX ix_document_versions_active"))
        session.execute(
            text(
                "CREATE UNIQUE INDEX ix_document_versions_active ON document_versions "
                "(document_id) WHERE status = 'active'"
            )
        )
        session.commit()
        service = IngestionService(session=session)
        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        first = await service.submit(FakeSource(b"# Guide\n\nVersion one"), context)
        first_done = await service.process(first.id)
        replacement = await service.submit(FakeSource(b"# Guide\n\nVersion two"), context)
        replacement_done = await service.process(replacement.id)

        versions = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == first_done.document_id)
        ).all()
        assert replacement_done.status == IngestionStatus.ACTIVE
        assert [version.status for version in versions] == ["replaced", "active"]


class FakeCacheService:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls: list[tuple[str, str, int | None]] = []
        self.should_fail = should_fail

    async def invalidate_document(
        self, tenant_id: str, document_id: str, version: int | None = None
    ) -> int:
        if self.should_fail:
            msg = "Redis connection lost"
            raise RuntimeError(msg)
        self.calls.append((tenant_id, document_id, version))
        return 1


@pytest.mark.asyncio
async def test_ingestion_invalidates_cache_on_version_replacement():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(text("DROP INDEX ix_document_versions_active"))
        session.execute(
            text(
                "CREATE UNIQUE INDEX ix_document_versions_active ON document_versions "
                "(document_id) WHERE status = 'active'"
            )
        )
        session.commit()
        cache = FakeCacheService()
        service = IngestionService(session=session, cache_service=cache)
        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        first = await service.submit(FakeSource(b"# Guide\n\nVersion one"), context)
        first_done = await service.process(first.id)
        replacement = await service.submit(FakeSource(b"# Guide\n\nVersion two"), context)
        replacement_done = await service.process(replacement.id)

        assert replacement_done.status == IngestionStatus.ACTIVE
        assert (context.tenant_id, first_done.document_id, 1) in cache.calls


@pytest.mark.asyncio
async def test_ingestion_invalidates_cache_on_document_deletion():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(text("DROP INDEX ix_document_versions_active"))
        session.execute(
            text(
                "CREATE UNIQUE INDEX ix_document_versions_active ON document_versions "
                "(document_id) WHERE status = 'active'"
            )
        )
        session.commit()
        cache = FakeCacheService()
        service = IngestionService(session=session, cache_service=cache)
        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        first = await service.submit(FakeSource(b"# Guide\n\nVersion one"), context)
        first_done = await service.process(first.id)

        await service.delete_document(first_done.document_id, context)
        assert (context.tenant_id, first_done.document_id, None) in cache.calls


@pytest.mark.asyncio
async def test_ingestion_invalidation_failure_does_not_fail_commit_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(text("DROP INDEX ix_document_versions_active"))
        session.execute(
            text(
                "CREATE UNIQUE INDEX ix_document_versions_active ON document_versions "
                "(document_id) WHERE status = 'active'"
            )
        )
        session.commit()
        cache = FakeCacheService(should_fail=True)
        service = IngestionService(session=session, cache_service=cache)
        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        first = await service.submit(FakeSource(b"# Guide\n\nVersion one"), context)
        await service.process(first.id)
        replacement = await service.submit(FakeSource(b"# Guide\n\nVersion two"), context)

        with caplog.at_level("WARNING"):
            replacement_done = await service.process(replacement.id)

        assert replacement_done.status == IngestionStatus.ACTIVE
        assert any("Cache invalidation failed" in record.message for record in caplog.records)
        assert all(context.tenant_id not in record.message for record in caplog.records)
