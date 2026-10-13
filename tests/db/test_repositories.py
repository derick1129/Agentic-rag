from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adaptive.db.models import Base
from adaptive.db.repositories import (
    ChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
    EvaluationAnnotationRepository,
    IngestionJobRepository,
    QueryRunRepository,
    SemanticCacheEntryRepository,
    TenantRepository,
)
from adaptive.interfaces import (
    ChunkDraft,
    DocumentCreateParams,
    IngestionJob,
    IngestionStatus,
    RequestContext,
)

EXPECTED_RETRY_ATTEMPTS = 2
MAX_ERROR_MESSAGE_LENGTH = 500


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def test_document_repository_cannot_read_another_tenant(session: Session):
    repo = DocumentRepository(session)
    repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    repo.create(
        DocumentCreateParams(
            tenant_id="globex",
            filename="b.md",
            content_hash="hash-b",
            title=None,
            canonical_text="B",
            metadata={},
            acl=frozenset(),
        )
    )

    documents = repo.list_active(RequestContext(tenant_id="acme", subject_id="user-1"))

    assert [document.tenant_id for document in documents] == ["acme"]
    assert (
        repo.get(documents[0].id, RequestContext(tenant_id="globex", subject_id="user-2")) is None
    )


def test_active_version_is_unique_per_document(session: Session):
    repo = DocumentRepository(session)
    document = repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )

    repo.activate_version(document.id, version=1)
    with pytest.raises(IntegrityError):
        repo.activate_version(document.id, version=1)


def test_document_version_repository_is_tenant_scoped(session: Session):
    document = DocumentRepository(session).create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    versions = DocumentVersionRepository(session)
    context = RequestContext(tenant_id="acme", subject_id="user-1")
    other = RequestContext(tenant_id="globex", subject_id="user-2")

    assert versions.get(document.id, 1, context).document_id == document.id
    assert versions.get(document.id, 1, other) is None


def test_all_repositories_apply_request_context_tenant_filter(session: Session):
    context = RequestContext(tenant_id="acme", subject_id="user-1")
    other = RequestContext(tenant_id="globex", subject_id="user-2")
    document_repo = DocumentRepository(session)
    document = document_repo.create(
        DocumentCreateParams(
            tenant_id="acme",
            filename="a.md",
            content_hash="hash-a",
            title=None,
            canonical_text="A",
            metadata={},
            acl=frozenset(),
        )
    )
    ChunkRepository(session).bulk_create(
        [
            ChunkDraft(
                id="chunk-a",
                document_id=document.id,
                tenant_id="acme",
                version=1,
                ordinal=0,
                text="A",
                heading_path=[],
                page_number=None,
                token_count=1,
                metadata={},
                acl=frozenset(),
            )
        ]
    )
    IngestionJobRepository(session).create(
        IngestionJob(
            id="job-a",
            tenant_id="acme",
            document_id=document.id,
            status=IngestionStatus.RECEIVED,
            current_stage="received",
        )
    )
    assert ChunkRepository(session).get_by_document(document.id, 1, context) != []
    assert ChunkRepository(session).get_by_document(document.id, 1, other) == []
    assert IngestionJobRepository(session).get("job-a", context) is not None
    assert IngestionJobRepository(session).get("job-a", other) is None


def test_tenant_repository_and_other_control_plane_repositories_round_trip(session: Session):
    tenant = TenantRepository(session).create(slug="acme", name="Acme")
    context = RequestContext(tenant_id=tenant.id, subject_id="user-1")
    query = QueryRunRepository(session).create(
        context, query_text="hello", status="completed", response={"text": "hi"}
    )
    cache = SemanticCacheEntryRepository(session).create(
        context,
        query_embedding=[1.0, 0.0],
        answer_text="hi",
        citations=[],
        model_profile="fake",
        response_mode="answer",
        referenced_document_versions={},
        expires_at=None,
    )
    annotation = EvaluationAnnotationRepository(session).create(
        context,
        query_run_id=query.id,
        evaluator="deterministic",
        score=1.0,
        explanation="exact",
        dataset_item_id="item-1",
        label_source="reference",
    )

    assert TenantRepository(session).get(tenant.id).slug == "acme"
    assert QueryRunRepository(session).get(query.id, context).id == query.id
    assert SemanticCacheEntryRepository(session).get(cache.id, context).answer_text == "hi"
    assert EvaluationAnnotationRepository(session).get(annotation.id, context).score == 1.0


def test_ingestion_job_repository_persists_durable_metadata_and_request_scope(
    session: Session,
) -> None:
    repo = IngestionJobRepository(session)
    due = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    lease_until = datetime(2026, 10, 13, 12, 5, tzinfo=UTC)
    expected_lease_id = "lease-1"
    created = IngestionJob(
        id="job-durable-1",
        tenant_id="acme",
        document_id=None,
        status=IngestionStatus.QUEUED,
        current_stage=IngestionStatus.QUEUED.value,
        attempts=0,
        source_key="job-durable-1.bin",
        filename="handbook.md",
        mime_type="text/markdown",
        source_uri="https://example.com/handbook.md",
        metadata={"department": "security", "priority": 2},
        subject_id="user-42",
        acl=frozenset({"group:sec", "group:eng"}),
        idempotency_key="idem-key-1",
        payload_hash="sha256-1234",
        available_at=due,
        worker_id="worker-a",
        lease_token=expected_lease_id,
        lease_expires_at=lease_until,
    )

    persisted = repo.create(created)
    session.expunge_all()

    loaded = repo.get("job-durable-1", RequestContext(tenant_id="acme", subject_id="user-42"))
    assert loaded is not None
    assert loaded.id == persisted.id
    assert loaded.status == IngestionStatus.QUEUED
    assert loaded.source_key == "job-durable-1.bin"
    assert loaded.filename == "handbook.md"
    assert loaded.mime_type == "text/markdown"
    assert loaded.source_uri == "https://example.com/handbook.md"
    assert loaded.metadata == {"department": "security", "priority": 2}
    assert loaded.subject_id == "user-42"
    assert loaded.acl == frozenset({"group:sec", "group:eng"})
    assert loaded.idempotency_key == "idem-key-1"
    assert loaded.payload_hash == "sha256-1234"
    assert loaded.available_at == due
    assert loaded.worker_id == "worker-a"
    assert loaded.lease_token == expected_lease_id
    assert loaded.lease_expires_at == lease_until

    by_key = repo.get_by_idempotency_key("acme", "idem-key-1")
    assert by_key is not None
    assert by_key.id == "job-durable-1"
    assert repo.get_by_idempotency_key("globex", "idem-key-1") is None

    updated = repo.update(
        loaded.model_copy(
            update={
                "status": IngestionStatus.RETRY_WAIT,
                "current_stage": IngestionStatus.EMBEDDING.value,
                "attempts": EXPECTED_RETRY_ATTEMPTS,
                "error_code": "embedding_error",
                "error_message": "transient timeout",
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
            }
        )
    )
    assert updated.status == IngestionStatus.RETRY_WAIT
    assert updated.attempts == EXPECTED_RETRY_ATTEMPTS
    assert updated.worker_id is None
    assert updated.lease_token is None
    assert updated.lease_expires_at is None


def test_ingestion_job_idempotency_key_unique_per_tenant(session: Session) -> None:
    repo = IngestionJobRepository(session)
    repo.create(
        IngestionJob(
            id="job-acme-1",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            idempotency_key="shared-key",
            payload_hash="hash-1",
        )
    )
    # Different tenant can use the same idempotency key
    globex_job = repo.create(
        IngestionJob(
            id="job-globex-1",
            tenant_id="globex",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            idempotency_key="shared-key",
            payload_hash="hash-1",
        )
    )
    assert globex_job.tenant_id == "globex"

    # Same tenant with duplicate idempotency_key violates uniqueness
    with pytest.raises(IntegrityError):
        repo.create(
            IngestionJob(
                id="job-acme-2",
                tenant_id="acme",
                document_id=None,
                status=IngestionStatus.QUEUED,
                current_stage="queued",
                idempotency_key="shared-key",
                payload_hash="hash-1",
            )
        )


def test_durable_ingestion_jobs_migration_upgrade_and_downgrade() -> None:
    migration_path = (
        Path(__file__).parents[2] / "alembic" / "versions" / "0004_durable_ingestion_jobs.py"
    )
    assert migration_path.exists()

    spec = importlib.util.spec_from_file_location("migration_0004", migration_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.revision == "0004_durable_ingestion_jobs"
    assert mod.down_revision == "0003_add_tenant_scoped_rls"

    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE ingestion_jobs (
                    id VARCHAR(36) PRIMARY KEY,
                    tenant_id VARCHAR(36) NOT NULL,
                    document_id VARCHAR(36),
                    status VARCHAR(32) NOT NULL,
                    current_stage VARCHAR(32) NOT NULL,
                    attempts INTEGER NOT NULL,
                    error_code VARCHAR(80),
                    error_message TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        long_noisy_msg = "  failed \n\t with   whitespace   " + ("x" * 600)
        conn.execute(
            text(
                """
                INSERT INTO ingestion_jobs (
                    id, tenant_id, document_id, status,
                    current_stage, attempts, error_code, error_message
                )
                VALUES
                    ('job-failed-null', 'acme', NULL, 'failed', 'parsing', 1, NULL, :long_msg),
                    (
                        'job-failed-code', 'acme', NULL, 'failed',
                        'embedding', 2, 'embedding_error', 'bad   gateway\n error'
                    ),
                    ('job-received-no-src', 'acme', NULL, 'received', 'received', 0, NULL, NULL),
                    ('job-active', 'acme', 'doc-1', 'active', 'active', 1, NULL, NULL)
                """
            ),
            {"long_msg": long_noisy_msg},
        )

        ctx = MigrationContext.configure(conn)
        mod.op = Operations(ctx)  # type: ignore[attr-defined]
        mod.upgrade()

        rows = {
            row["id"]: row
            for row in conn.execute(
                text(
                    "SELECT id, status, error_code, error_message, subject_id, source_key "
                    "FROM ingestion_jobs"
                )
            )
            .mappings()
            .all()
        }

        assert rows["job-failed-null"]["status"] == "dead_letter"
        assert rows["job-failed-null"]["error_code"] == "legacy_failed"
        assert len(rows["job-failed-null"]["error_message"]) == MAX_ERROR_MESSAGE_LENGTH
        assert "\n" not in rows["job-failed-null"]["error_message"]
        assert "  " not in rows["job-failed-null"]["error_message"]
        assert rows["job-failed-null"]["subject_id"] == "system"

        assert rows["job-failed-code"]["status"] == "dead_letter"
        assert rows["job-failed-code"]["error_code"] == "embedding_error"
        assert rows["job-failed-code"]["error_message"] == "bad gateway error"

        assert rows["job-received-no-src"]["status"] == "dead_letter"
        assert rows["job-active"]["status"] == "active"

        inspector = inspect(conn)
        col_names = {col["name"] for col in inspector.get_columns("ingestion_jobs")}
        assert {
            "source_key",
            "filename",
            "mime_type",
            "source_uri",
            "metadata",
            "subject_id",
            "acl",
            "idempotency_key",
            "payload_hash",
            "available_at",
            "worker_id",
            "lease_token",
            "lease_expires_at",
        }.issubset(col_names)
        index_names = {idx["name"] for idx in inspector.get_indexes("ingestion_jobs")}
        assert "ix_ingestion_jobs_queue_claim" in index_names

        mod.downgrade()

        downgraded_cols = {col["name"] for col in inspect(conn).get_columns("ingestion_jobs")}
        assert "source_key" not in downgraded_cols
        assert "lease_token" not in downgraded_cols

        downgraded_rows = {
            row["id"]: row["status"]
            for row in conn.execute(text("SELECT id, status FROM ingestion_jobs")).mappings().all()
        }
        assert downgraded_rows["job-failed-null"] == "failed"
        assert downgraded_rows["job-failed-code"] == "failed"
        assert downgraded_rows["job-active"] == "active"

