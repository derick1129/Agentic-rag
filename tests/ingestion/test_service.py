from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from adaptive.db.models import Base, DocumentVersion
from adaptive.db.models import Chunk as ChunkModel
from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.parsers import parse_bytes
from adaptive.ingestion.service import IngestionService
from adaptive.ingestion.storage import FilesystemSourceStorage
from adaptive.interfaces import (
    IdempotencyConflictError,
    IngestionStatus,
    RequestContext,
)

VERSION_TWO = 2
ATTEMPT_TWO = 2
MAX_ATTEMPTS = 5


class TransientTimeoutError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("transient timeout")


class FlakyServiceUnavailableError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("flaky service unavailable")


class CrashPersistenceError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("crash after database persistence")


class InvalidMalformedSyntaxError(ValueError):
    def __init__(self) -> None:
        super().__init__("invalid malformed input syntax")


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
async def test_service_deduplicates_same_tenant_content_hash(tmp_path: Path):
    storage = FilesystemSourceStorage(tmp_path)
    source = FakeSource(b"# Guide\n\nSame content")
    service = IngestionService(storage=storage)
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
async def test_service_replaces_previous_version_and_is_retry_safe(tmp_path: Path):
    storage = FilesystemSourceStorage(tmp_path)
    service = IngestionService(storage=storage, indexer=FakeIndexer())
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
async def test_service_records_transient_failure_and_can_retry(tmp_path: Path):
    calls = 0

    def parser(payload: bytes, mime_type: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientTimeoutError

        return parse_bytes(payload, mime_type)

    storage = FilesystemSourceStorage(tmp_path)
    service = IngestionService(storage=storage, parser=parser)
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    job = await service.submit(FakeSource(b"# Transient test"), context)
    failed = await service.process(job.id)

    assert failed.status == IngestionStatus.RETRY_WAIT
    assert failed.current_stage == "parsing"
    assert failed.error_code == "parse_error"
    assert failed.attempts == 1
    assert failed.available_at > datetime.now(UTC)
    assert (tmp_path / job.source_key).is_file()

    retried = await service.process(job.id)

    assert retried.status == IngestionStatus.ACTIVE
    assert retried.attempts == ATTEMPT_TWO
    assert not (tmp_path / job.source_key).exists()


@pytest.mark.asyncio
async def test_service_permanent_failure_dead_letters_and_deletes_source(tmp_path: Path):
    def parser(payload: bytes, mime_type: str):
        raise InvalidMalformedSyntaxError

    storage = FilesystemSourceStorage(tmp_path)
    service = IngestionService(storage=storage, parser=parser)
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    job = await service.submit(FakeSource(b"malformed data"), context)
    assert (tmp_path / job.source_key).is_file()

    failed = await service.process(job.id)

    assert failed.status == IngestionStatus.DEAD_LETTER
    assert failed.error_code == "parse_error"
    assert not (tmp_path / job.source_key).exists()


@pytest.mark.asyncio
async def test_service_transient_failure_exhausts_to_dead_letter(tmp_path: Path):
    def parser(payload: bytes, mime_type: str):
        raise FlakyServiceUnavailableError

    storage = FilesystemSourceStorage(tmp_path)
    service = IngestionService(storage=storage, parser=parser, max_attempts=MAX_ATTEMPTS)
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    job = await service.submit(FakeSource(b"content"), context)

    # Attempts 1 through 4 remain retry_wait with source kept
    for attempt in range(1, 5):
        res = await service.process(job.id)
        assert res.status == IngestionStatus.RETRY_WAIT
        assert res.attempts == attempt
        assert (tmp_path / job.source_key).is_file()

    # Attempt 5 exhausts retries -> dead_letter and source deleted
    res5 = await service.process(job.id)
    assert res5.status == IngestionStatus.DEAD_LETTER
    assert res5.attempts == MAX_ATTEMPTS
    assert not (tmp_path / job.source_key).exists()


@pytest.mark.asyncio
async def test_service_restart_simulation(tmp_path: Path):
    storage = FilesystemSourceStorage(tmp_path)
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

        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        # Service 1 submits the job
        service1 = IngestionService(storage=storage, session=session)
        submitted = await service1.submit(FakeSource(b"# Restartable\n\nDurable"), context)
        assert submitted.status == IngestionStatus.QUEUED
        assert (tmp_path / submitted.source_key).is_file()

        # Recreate service 2 with fresh instance (simulating restart)
        service2 = IngestionService(storage=storage, session=session)
        processed = await service2.process(submitted.id)

        assert processed.status == IngestionStatus.ACTIVE
        assert processed.document_id is not None
        assert not (tmp_path / submitted.source_key).exists()


@pytest.mark.asyncio
async def test_service_crash_replay_idempotency(tmp_path: Path):
    storage = FilesystemSourceStorage(tmp_path)
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

        context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

        # Simulate indexer crashing on attempt 1
        calls = 0

        class CrashingIndexer:
            async def upsert_chunks(self, chunks, embeddings=None):  # noqa: ANN001, ANN201
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise CrashPersistenceError

        service = IngestionService(storage=storage, session=session, indexer=CrashingIndexer())
        job = await service.submit(FakeSource(b"# Crash Replay\n\nContent body"), context)

        # Attempt 1 crashes at indexing stage -> retry_wait
        failed = await service.process(job.id)
        assert failed.status == IngestionStatus.RETRY_WAIT

        # Attempt 2 succeeds
        recovered = await service.process(job.id)
        assert recovered.status == IngestionStatus.ACTIVE

        # Verify no duplicate chunks in database
        chunk_count = (
            session.query(ChunkModel)
            .filter(ChunkModel.document_id == recovered.document_id)
            .count()
        )
        assert chunk_count == 1

        # Re-running an already ACTIVE job is an idempotent no-op
        replay = await service.process(job.id)
        assert replay.status == IngestionStatus.ACTIVE


@pytest.mark.asyncio
async def test_service_submit_idempotency_conflict_and_replay(tmp_path: Path):
    storage = FilesystemSourceStorage(tmp_path)
    service = IngestionService(storage=storage)
    context = RequestContext(tenant_id="tenant-1", subject_id="user-1")

    # Replaying same key and same payload returns existing job
    first = await service.submit(
        FakeSource(b"hello world", filename="hello.txt"),
        context,
        idempotency_key="idemp-key-1",
    )
    second = await service.submit(
        FakeSource(b"hello world", filename="hello.txt"),
        context,
        idempotency_key="idemp-key-1",
    )
    assert first.id == second.id

    # Reusing same key with changed payload raises IdempotencyConflictError
    with pytest.raises(IdempotencyConflictError):
        await service.submit(
            FakeSource(b"different content", filename="hello.txt"),
            context,
            idempotency_key="idemp-key-1",
        )

    # Omitted key derives deterministic key from tenant + filename + sha256
    auto1 = await service.submit(FakeSource(b"auto content", filename="auto.txt"), context)
    auto2 = await service.submit(FakeSource(b"auto content", filename="auto.txt"), context)
    assert auto1.id == auto2.id
    assert auto1.idempotency_key is not None


@pytest.mark.asyncio
async def test_service_persists_and_activates_replacement_versions():
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
