from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
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
    LeaseLostError,
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


def test_claim_next_orders_oldest_eligible_and_excludes_future_retries(
    session: Session,
) -> None:
    repo = IngestionJobRepository(session)
    now = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)

    repo.create(
        IngestionJob(
            id="job-future",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.RETRY_WAIT,
            current_stage="embedding",
            attempts=1,
            available_at=now + timedelta(seconds=60),
            created_at=now - timedelta(seconds=100),
        )
    )
    repo.create(
        IngestionJob(
            id="job-later",
            tenant_id="globex",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=now - timedelta(seconds=10),
            created_at=now - timedelta(seconds=10),
        )
    )
    repo.create(
        IngestionJob(
            id="job-oldest",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=now - timedelta(seconds=30),
            created_at=now - timedelta(seconds=30),
        )
    )
    repo.create(
        IngestionJob(
            id="job-terminal",
            tenant_id="acme",
            document_id="doc-1",
            status=IngestionStatus.ACTIVE,
            current_stage="active",
            available_at=now - timedelta(seconds=120),
        )
    )

    first = repo.claim_next("worker-1", now, lease_seconds=300)
    assert first is not None
    assert first.job.id == "job-oldest"
    assert first.job.status == IngestionStatus.RUNNING
    assert first.job.attempts == 1
    assert first.worker_id == "worker-1"
    assert first.lease_expires_at == now + timedelta(seconds=300)
    assert first.job.lease_token == first.lease_token

    second = repo.claim_next("worker-2", now, lease_seconds=300)
    assert second is not None
    assert second.job.id == "job-later"
    assert second.job.status == IngestionStatus.RUNNING
    assert second.job.attempts == 1
    assert second.worker_id == "worker-2"
    assert second.lease_token != first.lease_token

    assert repo.claim_next("worker-3", now, lease_seconds=300) is None

    due_later = now + timedelta(seconds=60)
    third = repo.claim_next("worker-3", due_later, lease_seconds=300)
    assert third is not None
    assert third.job.id == "job-future"
    assert third.job.attempts == EXPECTED_RETRY_ATTEMPTS


def test_claim_next_reclaims_expired_lease_and_heartbeat_extends_ownership(
    session: Session,
) -> None:
    repo = IngestionJobRepository(session)
    t0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    repo.create(
        IngestionJob(
            id="job-lease-1",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=t0,
        )
    )

    claimed = repo.claim_next("worker-1", t0, lease_seconds=60)
    assert claimed is not None
    assert claimed.job.attempts == 1

    # Extending heartbeat before expiry succeeds without incrementing attempts
    extended = repo.heartbeat(
        "job-lease-1",
        worker_id="worker-1",
        lease_token=claimed.lease_token,
        now=t0 + timedelta(seconds=20),
        lease_seconds=60,
    )
    assert extended.lease_expires_at == t0 + timedelta(seconds=80)
    assert extended.job.attempts == 1

    # Another worker cannot claim before the extended expiry
    assert repo.claim_next("worker-2", t0 + timedelta(seconds=79), lease_seconds=60) is None

    # Wrong worker or wrong token cannot heartbeat
    with pytest.raises(LeaseLostError):
        repo.heartbeat(
            "job-lease-1",
            worker_id="worker-2",
            lease_token=claimed.lease_token,
            now=t0 + timedelta(seconds=30),
            lease_seconds=60,
        )
    with pytest.raises(LeaseLostError):
        repo.heartbeat(
            "job-lease-1",
            worker_id="worker-1",
            lease_token="wrong-lease-id",
            now=t0 + timedelta(seconds=30),
            lease_seconds=60,
        )

    # Once lease expires at t0 + 80s, heartbeat fails and another worker can reclaim
    with pytest.raises(LeaseLostError):
        repo.heartbeat(
            "job-lease-1",
            worker_id="worker-1",
            lease_token=claimed.lease_token,
            now=t0 + timedelta(seconds=80),
            lease_seconds=60,
        )

    reclaimed = repo.claim_next("worker-2", t0 + timedelta(seconds=80), lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed.job.id == "job-lease-1"
    assert reclaimed.worker_id == "worker-2"
    assert reclaimed.lease_token != claimed.lease_token
    assert reclaimed.job.attempts == EXPECTED_RETRY_ATTEMPTS


def test_guarded_transitions_reject_stale_lease_without_modifying_row(
    session: Session,
) -> None:
    repo = IngestionJobRepository(session)
    t0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    repo.create(
        IngestionJob(
            id="job-guard-1",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=t0,
        )
    )

    first_claim = repo.claim_next("worker-1", t0, lease_seconds=30)
    assert first_claim is not None

    progressed = repo.update_stage(
        "job-guard-1",
        worker_id="worker-1",
        lease_token=first_claim.lease_token,
        stage=IngestionStatus.PARSING.value,
    )
    assert progressed.current_stage == IngestionStatus.PARSING.value

    # Lease expires and worker-2 reclaims the job
    second_claim = repo.claim_next("worker-2", t0 + timedelta(seconds=30), lease_seconds=60)
    assert second_claim is not None
    assert second_claim.worker_id == "worker-2"

    # Stale worker-1 cannot update stage, schedule retry, complete, or dead-letter
    with pytest.raises(LeaseLostError):
        repo.update_stage(
            "job-guard-1",
            worker_id="worker-1",
            lease_token=first_claim.lease_token,
            stage=IngestionStatus.CHUNKING.value,
        )
    with pytest.raises(LeaseLostError):
        repo.schedule_retry(
            "job-guard-1",
            worker_id="worker-1",
            lease_token=first_claim.lease_token,
            available_at=t0 + timedelta(seconds=90),
            error_code="embedding_error",
            error_message="stale retry",
        )
    with pytest.raises(LeaseLostError):
        repo.complete(
            "job-guard-1",
            worker_id="worker-1",
            lease_token=first_claim.lease_token,
            document_id="doc-stale",
        )
    with pytest.raises(LeaseLostError):
        repo.dead_letter(
            "job-guard-1",
            worker_id="worker-1",
            lease_token=first_claim.lease_token,
            error_code="parse_error",
            error_message="stale dead letter",
        )

    # Verify row still belongs to worker-2 and is RUNNING
    current = repo.get("job-guard-1")
    assert current is not None
    assert current.status == IngestionStatus.RUNNING
    assert current.worker_id == "worker-2"
    assert current.lease_token == second_claim.lease_token
    assert current.document_id is None


def test_guarded_transitions_schedule_retry_complete_and_dead_letter(
    session: Session,
) -> None:
    repo = IngestionJobRepository(session)
    t0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    repo.create(
        IngestionJob(
            id="job-guard-2",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=t0,
        )
    )
    claim_1 = repo.claim_next("worker-2", t0, lease_seconds=60)
    assert claim_1 is not None

    retry_at = t0 + timedelta(seconds=90)
    retried = repo.schedule_retry(
        "job-guard-2",
        worker_id="worker-2",
        lease_token=claim_1.lease_token,
        available_at=retry_at,
        error_code="embedding_error",
        error_message="  transient \n provider   timeout  ",
    )
    assert retried.status == IngestionStatus.RETRY_WAIT
    assert retried.available_at == retry_at
    assert retried.error_code == "embedding_error"
    assert retried.error_message == "transient provider timeout"
    assert retried.worker_id is None
    assert retried.lease_token is None
    assert retried.lease_expires_at is None
    assert retried.attempts == 1

    # Re-claim when retry is due, then complete
    claim_2 = repo.claim_next("worker-3", retry_at, lease_seconds=60)
    assert claim_2 is not None
    assert claim_2.job.attempts == EXPECTED_RETRY_ATTEMPTS

    completed = repo.complete(
        "job-guard-2",
        worker_id="worker-3",
        lease_token=claim_2.lease_token,
        document_id="doc-final",
    )
    assert completed.status == IngestionStatus.ACTIVE
    assert completed.current_stage == IngestionStatus.ACTIVE.value
    assert completed.document_id == "doc-final"
    assert completed.error_code is None
    assert completed.error_message is None
    assert completed.worker_id is None
    assert completed.lease_token is None
    assert completed.lease_expires_at is None

    # Create another job and dead-letter it
    repo.create(
        IngestionJob(
            id="job-guard-3",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage="queued",
            available_at=t0,
        )
    )
    dl_claim = repo.claim_next("worker-3", t0, lease_seconds=60)
    assert dl_claim is not None
    dead = repo.dead_letter(
        "job-guard-3",
        worker_id="worker-3",
        lease_token=dl_claim.lease_token,
        error_code="parse_error",
        error_message="  corrupt \t file  " + ("z" * 600),
    )
    assert dead.status == IngestionStatus.DEAD_LETTER
    assert dead.error_code == "parse_error"
    assert dead.error_message is not None
    assert len(dead.error_message) == MAX_ERROR_MESSAGE_LENGTH
    assert dead.worker_id is None
    assert dead.lease_token is None
    assert dead.lease_expires_at is None

