from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://localhost/test")
os.environ.setdefault("OPENSEARCH_URL", "http://localhost:9200")

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import adaptive.api.main as api_main
from adaptive.db.models import Base
from adaptive.ingestion.service import IngestionService
from adaptive.ingestion.storage import FilesystemSourceStorage
from adaptive.interfaces import IngestionStatus, RequestContext

if TYPE_CHECKING:
    from pathlib import Path


@asynccontextmanager
async def _noop_lifespan(_app: object):
    yield


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/test")
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost:9200")


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def source_storage(tmp_path: Path) -> FilesystemSourceStorage:
    return FilesystemSourceStorage(tmp_path / "uploads")


@pytest.fixture
def ingestion_service(
    db_session: Session, source_storage: FilesystemSourceStorage
) -> IngestionService:
    return IngestionService(
        session=db_session,
        storage=source_storage,
        embedding_provider=None,
        indexer=None,
    )


@pytest.fixture
def test_app(ingestion_service: IngestionService) -> TestClient:
    app = api_main.create_app()
    app.router.lifespan_context = _noop_lifespan

    default_context = RequestContext(
        tenant_id="tenant-alpha",
        subject_id="user-1",
        acl=frozenset({"read", "write"}),
    )

    app.dependency_overrides[api_main.get_current_context] = (
        lambda: default_context
    )
    app.dependency_overrides[api_main.get_ingestion_service] = (
        lambda: ingestion_service
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


def test_post_ingestion_returns_queued_immediately(
    test_app: TestClient, ingestion_service: IngestionService
) -> None:
    response = test_app.post(
        "/v1/ingestion",
        json={"filename": "notes.txt", "content": "Sample content"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "queued"
    job_id = data["job_id"]
    assert job_id

    job = ingestion_service.jobs.get(job_id)
    assert job is not None
    assert job.status == IngestionStatus.QUEUED
    assert job.current_stage == "queued"
    assert job.source_key is not None
    with ingestion_service.storage.open(job.source_key) as f:
        assert f.read() == b"Sample content"


def test_post_ingestion_upload_returns_queued_immediately(
    test_app: TestClient, ingestion_service: IngestionService
) -> None:
    files = {"file": ("report.pdf", b"PDF bytes content", "application/pdf")}
    data = {"metadata": '{"category": "finance"}'}
    response = test_app.post("/v1/ingestion/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    res_data = response.json()
    assert res_data["status"] == "queued"
    job_id = res_data["job_id"]

    job = ingestion_service.jobs.get(job_id)
    assert job is not None
    assert job.status == IngestionStatus.QUEUED
    assert job.source_key is not None
    with ingestion_service.storage.open(job.source_key) as f:
        assert f.read() == b"PDF bytes content"


def test_post_ingestion_idempotency_key_replay(test_app: TestClient) -> None:
    headers = {"Idempotency-Key": "client-key-001"}
    payload = {"filename": "doc.txt", "content": "Alpha content"}

    res1 = test_app.post("/v1/ingestion", json=payload, headers=headers)
    assert res1.status_code == status.HTTP_200_OK
    job_1 = res1.json()

    res2 = test_app.post("/v1/ingestion", json=payload, headers=headers)
    assert res2.status_code == status.HTTP_200_OK
    job_2 = res2.json()

    assert job_1["job_id"] == job_2["job_id"]


def test_post_ingestion_idempotency_key_conflict(test_app: TestClient) -> None:
    headers = {"Idempotency-Key": "client-key-conflict"}
    payload1 = {"filename": "doc.txt", "content": "Original content"}
    payload2 = {"filename": "doc.txt", "content": "Changed content"}

    res1 = test_app.post("/v1/ingestion", json=payload1, headers=headers)
    assert res1.status_code == status.HTTP_200_OK

    res2 = test_app.post("/v1/ingestion", json=payload2, headers=headers)
    assert res2.status_code == status.HTTP_409_CONFLICT
    assert "already used with different payload" in res2.json()["detail"]


def test_post_ingestion_upload_idempotency_key_replay_and_conflict(
    test_app: TestClient,
) -> None:
    headers = {"Idempotency-Key": "upload-key-replay"}
    files1 = {"file": ("data.csv", b"col1,col2\n1,2", "text/csv")}

    res1 = test_app.post("/v1/ingestion/upload", files=files1, headers=headers)
    assert res1.status_code == status.HTTP_200_OK
    job1_id = res1.json()["job_id"]

    files2 = {"file": ("data.csv", b"col1,col2\n1,2", "text/csv")}
    res2 = test_app.post("/v1/ingestion/upload", files=files2, headers=headers)
    assert res2.status_code == status.HTTP_200_OK
    assert res2.json()["job_id"] == job1_id

    files3 = {"file": ("data.csv", b"different,content\n3,4", "text/csv")}
    res3 = test_app.post("/v1/ingestion/upload", files=files3, headers=headers)
    assert res3.status_code == status.HTTP_409_CONFLICT


def test_omitted_idempotency_key_derived_deterministically(
    test_app: TestClient,
) -> None:
    payload = {"filename": "omitted.txt", "content": "Deterministic hashing content"}

    res1 = test_app.post("/v1/ingestion", json=payload)
    assert res1.status_code == status.HTTP_200_OK

    res2 = test_app.post("/v1/ingestion", json=payload)
    assert res2.status_code == status.HTTP_200_OK
    assert res1.json()["job_id"] == res2.json()["job_id"]


def test_tenant_isolation_submission_and_get(
    test_app: TestClient, db_session: Session
) -> None:
    context_alpha = RequestContext(
        tenant_id="tenant-alpha",
        subject_id="user-1",
        acl=frozenset(),
    )
    context_beta = RequestContext(
        tenant_id="tenant-beta",
        subject_id="user-2",
        acl=frozenset(),
    )

    current_context = context_alpha
    test_app.app.dependency_overrides[api_main.get_current_context] = (
        lambda: current_context
    )

    headers = {"Idempotency-Key": "shared-key"}
    payload = {"filename": "shared.txt", "content": "Shared content across tenants"}

    res_alpha = test_app.post("/v1/ingestion", json=payload, headers=headers)
    assert res_alpha.status_code == status.HTTP_200_OK
    job_alpha_id = res_alpha.json()["job_id"]

    current_context = context_beta
    res_beta = test_app.post("/v1/ingestion", json=payload, headers=headers)
    assert res_beta.status_code == status.HTTP_200_OK
    job_beta_id = res_beta.json()["job_id"]

    assert job_alpha_id != job_beta_id

    current_context = context_alpha
    get_alpha = test_app.get(f"/v1/ingestion/{job_alpha_id}")
    assert get_alpha.status_code == status.HTTP_200_OK
    assert get_alpha.json()["job_id"] == job_alpha_id

    get_alpha_from_beta = test_app.get(f"/v1/ingestion/{job_beta_id}")
    assert get_alpha_from_beta.status_code == status.HTTP_404_NOT_FOUND

    current_context = context_beta
    get_beta = test_app.get(f"/v1/ingestion/{job_beta_id}")
    assert get_beta.status_code == status.HTTP_200_OK
    assert get_beta.json()["job_id"] == job_beta_id


def test_get_ingestion_not_found(test_app: TestClient) -> None:
    res = test_app.get("/v1/ingestion/nonexistent-job-id")
    assert res.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in res.json()["detail"]


def test_upload_empty_file_returns_bad_request(test_app: TestClient) -> None:
    files = {"file": ("empty.txt", b"", "text/plain")}
    res = test_app.post("/v1/ingestion/upload", files=files)
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty" in res.json()["detail"]


def test_upload_invalid_metadata_returns_bad_request(test_app: TestClient) -> None:
    files = {"file": ("test.txt", b"content", "text/plain")}
    data = {"metadata": "not-a-json"}
    res = test_app.post("/v1/ingestion/upload", files=files, data=data)
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "metadata must be a valid JSON object" in res.json()["detail"]


def test_submission_failure_cleans_orphan_source(
    test_app: TestClient, ingestion_service: IngestionService
) -> None:
    mock_repo = MagicMock()
    mock_repo.get_by_idempotency_key.return_value = None
    mock_repo.create.side_effect = RuntimeError("Database insert failure")
    ingestion_service.jobs = mock_repo

    payload = {"filename": "fail.txt", "content": "Doomed content"}
    res = test_app.post("/v1/ingestion", json=payload)
    assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    storage_root = ingestion_service.storage.base_dir
    remaining_files = list(storage_root.rglob("*.*"))
    assert len(remaining_files) == 0
