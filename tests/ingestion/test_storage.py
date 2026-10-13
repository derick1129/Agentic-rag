from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from adaptive.ingestion.models import ClaimedIngestionJob
from adaptive.ingestion.storage import FilesystemSourceStorage, SourceStorage
from adaptive.interfaces import IngestionJob, IngestionStatus

if TYPE_CHECKING:
    from pathlib import Path


def test_store_open_and_delete_roundtrip(tmp_path: Path) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=1024)
    assert isinstance(storage, SourceStorage)

    payload = b"durable ingestion payload\nline two"
    source_key = storage.store("job_123-abc", payload)

    assert "/" not in source_key
    assert "\\" not in source_key
    assert not source_key.startswith(".tmp-")
    assert source_key.startswith("job_123-abc-")
    assert source_key.endswith(".blob")
    assert (tmp_path / source_key).is_file()

    with storage.open(source_key) as stream:
        assert stream.read() == payload

    storage.delete(source_key)
    assert not (tmp_path / source_key).exists()
    # Idempotent delete of missing key
    storage.delete(source_key)


def test_store_generates_opaque_unique_keys(tmp_path: Path) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=1024)

    key_one = storage.store("job-1", b"alpha")
    key_two = storage.store("job-1", b"beta")

    assert key_one != key_two
    assert "report.pdf" not in key_one
    with storage.open(key_one) as f1, storage.open(key_two) as f2:
        assert f1.read() == b"alpha"
        assert f2.read() == b"beta"


def test_store_enforces_max_bytes_and_validates_job_id(tmp_path: Path) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=8)

    with pytest.raises(ValueError, match="exceeds maximum"):
        storage.store("job-1", b"012345678")

    for bad_job_id in ["", "../escape", "nested/job", "back\\slash", "null\x00id", ".dot"]:
        with pytest.raises(ValueError, match="job_id"):
            storage.store(bad_job_id, b"ok")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "..",
        "../secret.txt",
        "/etc/passwd",
        "nested/file.blob",
        "nested\\file.blob",
        ".tmp-ingest-12345",
        "bad\x00key.blob",
    ],
)
def test_open_and_delete_reject_unsafe_keys(tmp_path: Path, bad_key: str) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=1024)

    with pytest.raises(ValueError, match="source_key"):
        storage.open(bad_key)

    with pytest.raises(ValueError, match="source_key"):
        storage.delete(bad_key)


def test_open_and_delete_reject_symlink_escape(tmp_path: Path) -> None:
    base_dir = tmp_path / "ingest"
    outside_file = tmp_path / "outside.txt"
    outside_file.write_bytes(b"secret")

    storage = FilesystemSourceStorage(base_dir, max_bytes=1024)
    symlink_path = base_dir / "job-1-escape.blob"
    symlink_path.symlink_to(outside_file)

    with pytest.raises(ValueError, match="source_key"):
        storage.open("job-1-escape.blob")

    with pytest.raises(ValueError, match="source_key"):
        storage.delete("job-1-escape.blob")

    assert outside_file.exists()


def test_store_cleans_up_temp_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=1024)

    def fail_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated rename failure")  # noqa: TRY003

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        storage.store("job-atomic", b"payload")

    leftovers = list(tmp_path.iterdir())
    assert leftovers == []


def test_store_cleans_up_temp_file_when_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FilesystemSourceStorage(tmp_path, max_bytes=1024)

    def fail_fsync(fd: int) -> None:
        raise OSError("simulated fsync failure")  # noqa: TRY003

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        storage.store("job-fsync", b"payload")

    assert list(tmp_path.iterdir()) == []


def test_durable_ingestion_job_and_claimed_job_models() -> None:
    now = datetime.now(UTC)
    lease_expires_at = now + timedelta(minutes=5)
    job = IngestionJob(
        id="job-1",
        tenant_id="acme",
        status=IngestionStatus.QUEUED,
        current_stage="queued",
        source_key="job-1-deadbeef.blob",
        filename="handbook.md",
        mime_type="text/markdown",
        metadata={"department": "support"},
        subject_id="user-42",
        acl=frozenset({"support", "internal"}),
        idempotency_key="idem-123",
        payload_hash="sha256:abc",
        available_at=now,
        worker_id="worker-1",
        lease_token="lease-xyz",
        lease_expires_at=lease_expires_at,
    )

    assert job.document_id is None
    assert job.status is IngestionStatus.QUEUED
    assert IngestionStatus.RUNNING == "running"
    assert IngestionStatus.RETRY_WAIT == "retry_wait"
    assert IngestionStatus.DEAD_LETTER == "dead_letter"

    claimed = ClaimedIngestionJob(
        job=job,
        worker_id="worker-1",
        lease_token="lease-xyz",
        lease_expires_at=lease_expires_at,
    )
    assert claimed.job.source_key == "job-1-deadbeef.blob"
    assert claimed.job.acl == frozenset({"support", "internal"})
