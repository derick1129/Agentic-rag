"""Unit tests for the durable ingestion background worker."""

# ruff: noqa: ANN401, ARG005, E501, PLR0913, PLR2004, S105, TRY003

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from adaptive.config import Settings
from adaptive.ingestion.worker import (
    ClaimedIngestionJob,
    IngestionWorker,
    LeaseLostError,
    compute_retry_delay_seconds,
)
from adaptive.interfaces import IngestionJob, IngestionStatus


class _PermanentFormatError(Exception):
    permanent = True
    error_code = "unsupported_format"


@dataclass
class _FakeSession:
    name: str
    events: list[str] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False
    closed: bool = False

    def commit(self) -> None:
        self.committed = True
        self.events.append(f"{self.name}:commit")

    def rollback(self) -> None:
        self.rolled_back = True
        self.events.append(f"{self.name}:rollback")

    def close(self) -> None:
        self.closed = True
        self.events.append(f"{self.name}:close")


class _FakeStorage:
    def __init__(self, events: list[str] | None = None) -> None:
        self.deleted_keys: list[str] = []
        self.events = events if events is not None else []

    def delete(self, source_key: str) -> None:
        self.deleted_keys.append(source_key)
        self.events.append(f"storage:delete:{source_key}")


def _make_job(
    *,
    job_id: str = "job-1",
    tenant_id: str = "acme",
    attempts: int = 1,
    source_key: str | None = "acme/job-1.bin",
    stage: str = "parsing",
) -> IngestionJob:
    job = IngestionJob(
        id=job_id,
        tenant_id=tenant_id,
        document_id=None,
        status=IngestionStatus.PARSING,
        current_stage=stage,
        attempts=attempts,
    )
    object.__setattr__(job, "source_key", source_key)
    return job


class TestWorkerConfigAndBackoff:
    """Verify Spec Requirement 6 & 10 configuration bounds and backoff schedule."""

    def test_settings_defaults_match_spec(self) -> None:
        settings = Settings(DATABASE_URL="postgresql://db")
        assert settings.ingestion_max_attempts == 5
        assert settings.ingestion_retry_base_seconds == 2
        assert settings.ingestion_retry_max_seconds == 300
        assert settings.ingestion_job_lease_seconds == 300
        assert settings.ingestion_poll_interval_seconds == 1.0
        assert settings.ingestion_heartbeat_interval_seconds is None

    @pytest.mark.parametrize(
        ("field_name", "invalid_value"),
        [
            ("INGESTION_MAX_ATTEMPTS", 0),
            ("INGESTION_MAX_ATTEMPTS", -1),
            ("INGESTION_RETRY_BASE_SECONDS", 0),
            ("INGESTION_RETRY_MAX_SECONDS", 0),
            ("INGESTION_JOB_LEASE_SECONDS", 0),
            ("INGESTION_POLL_INTERVAL_SECONDS", 0.0),
            ("INGESTION_POLL_INTERVAL_SECONDS", -0.5),
            ("INGESTION_HEARTBEAT_INTERVAL_SECONDS", 0.0),
        ],
    )
    def test_settings_reject_non_positive_values(
        self, field_name: str, invalid_value: float
    ) -> None:
        kwargs: dict[str, Any] = {
            "DATABASE_URL": "postgresql://db",
            field_name: invalid_value,
        }
        with pytest.raises(ValidationError):
            Settings(**kwargs)

    def test_settings_reject_retry_max_smaller_than_base(self) -> None:
        with pytest.raises(ValidationError, match="ingestion_retry_max_seconds"):
            Settings(
                DATABASE_URL="postgresql://db",
                INGESTION_RETRY_BASE_SECONDS=10,
                INGESTION_RETRY_MAX_SECONDS=5,
            )

    @pytest.mark.parametrize(
        ("attempts", "expected_seconds"),
        [
            (1, 2),
            (2, 4),
            (3, 8),
            (4, 16),
            (5, 32),
            (6, 64),
            (7, 128),
            (8, 256),
            (9, 300),
            (12, 300),
        ],
    )
    def test_exponential_backoff_matches_spec_formula(
        self, attempts: int, expected_seconds: int
    ) -> None:
        assert compute_retry_delay_seconds(attempts) == expected_seconds


class TestWorkerExecution:
    """Verify claim/process session isolation, heartbeats, lease loss, and retries."""

    async def test_run_once_returns_false_when_queue_is_empty(self) -> None:
        sessions: list[_FakeSession] = []

        def session_factory() -> _FakeSession:
            session = _FakeSession(name=f"session-{len(sessions) + 1}")
            sessions.append(session)
            return session

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, now: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return None

        worker = IngestionWorker(
            session_factory=session_factory,
            job_repository_factory=Repo,
            service_factory=lambda session: pytest.fail("service should not be created when idle"),
        )

        assert await worker.run_once() is False
        assert len(sessions) == 1
        assert sessions[0].closed is True

    async def test_run_once_claims_and_processes_with_fresh_sessions_and_cleans_source_after_commit(
        self,
    ) -> None:
        events: list[str] = []
        sessions: list[_FakeSession] = []
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-42", source_key="acme/job-42.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-a",
            lease_token="lease-token-1",
            lease_expires_at=now + timedelta(seconds=300),
        )
        storage = _FakeStorage(events=events)
        completed_calls: list[dict[str, Any]] = []

        def session_factory() -> _FakeSession:
            session = _FakeSession(name=f"s{len(sessions) + 1}", events=events)
            sessions.append(session)
            return session

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                events.append(f"{self.session.name}:claim_next")
                return claimed

            def complete(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                document_id: str,
            ) -> IngestionJob:
                events.append(f"{self.session.name}:complete")
                completed_calls.append(
                    {
                        "session": self.session.name,
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "lease_token": lease_token,
                        "document_id": document_id,
                    }
                )
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": document_id}
                )

        class Service:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                events.append(f"{self.session.name}:process:{job_id}")
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-99"}
                )

        worker = IngestionWorker(
            session_factory=session_factory,
            job_repository_factory=Repo,
            service_factory=Service,
            storage=storage,
            worker_id="worker-a",
            clock=lambda: now,
        )

        processed = await worker.run_once()
        assert processed is True
        assert len(sessions) == 2
        assert sessions[0] is not sessions[1]
        assert sessions[0].committed is True
        assert sessions[0].closed is True
        assert sessions[1].committed is True
        assert sessions[1].closed is True
        assert completed_calls == [
            {
                "session": "s2",
                "job_id": "job-42",
                "worker_id": "worker-a",
                "lease_token": "lease-token-1",
                "document_id": "doc-99",
            }
        ]
        # Verify strict ordering: claim commit -> process -> complete -> process commit -> storage delete
        assert events.index("s1:claim_next") < events.index("s1:commit")
        assert events.index("s1:close") < events.index("s2:process:job-42")
        assert events.index("s2:complete") < events.index("s2:commit")
        assert events.index("s2:commit") < events.index("storage:delete:acme/job-42.bin")

    async def test_heartbeat_extends_lease_during_long_running_execution(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-hb")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-hb",
            lease_token="token-hb",
            lease_expires_at=now + timedelta(seconds=300),
        )
        heartbeats: list[tuple[str, str, str, int]] = []

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def heartbeat(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                now: datetime,
                lease_seconds: int,
            ) -> ClaimedIngestionJob:
                heartbeats.append((job_id, worker_id, lease_token, lease_seconds))
                return ClaimedIngestionJob(
                    job=job,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                )

            def complete(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                document_id: str,
            ) -> IngestionJob:
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": document_id}
                )

        class SlowService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                await asyncio.sleep(0.06)
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-hb"}
                )

        idx = 0

        def session_factory() -> _FakeSession:
            nonlocal idx
            idx += 1
            return _FakeSession(name=f"hb-s{idx}")

        worker = IngestionWorker(
            session_factory=session_factory,
            job_repository_factory=Repo,
            service_factory=SlowService,
            worker_id="worker-hb",
            lease_seconds=300,
            heartbeat_interval_seconds=0.015,
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert len(heartbeats) >= 2
        assert all(hb == ("job-hb", "worker-hb", "token-hb", 300) for hb in heartbeats)

    async def test_lease_loss_during_heartbeat_stops_writes_and_keeps_source(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-lost", source_key="acme/job-lost.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-1",
            lease_token="stale-token",
            lease_expires_at=now + timedelta(seconds=300),
        )
        storage = _FakeStorage()
        transitions: list[str] = []
        sessions: list[_FakeSession] = []

        def session_factory() -> _FakeSession:
            session = _FakeSession(name=f"s{len(sessions) + 1}")
            sessions.append(session)
            return session

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def heartbeat(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                now: datetime,
                lease_seconds: int,
            ) -> ClaimedIngestionJob:
                raise LeaseLostError("lease owned by another worker")

            def complete(self, *args: Any, **kwargs: Any) -> IngestionJob:
                transitions.append("complete")
                raise AssertionError("complete must not be called after lease loss")

            def schedule_retry(self, *args: Any, **kwargs: Any) -> IngestionJob:
                transitions.append("schedule_retry")
                raise AssertionError("schedule_retry must not be called after lease loss")

            def dead_letter(self, *args: Any, **kwargs: Any) -> IngestionJob:
                transitions.append("dead_letter")
                raise AssertionError("dead_letter must not be called after lease loss")

        class LongService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                await asyncio.sleep(0.2)
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-never"}
                )

        worker = IngestionWorker(
            session_factory=session_factory,
            job_repository_factory=Repo,
            service_factory=LongService,
            storage=storage,
            worker_id="worker-1",
            heartbeat_interval_seconds=0.01,
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert transitions == []
        assert storage.deleted_keys == []
        # s2 is the processing session (or s2/s3 depending on heartbeat opening): find processing session
        processing_session = sessions[1]
        assert processing_session.committed is False
        assert processing_session.rolled_back is True
        assert processing_session.closed is True

    async def test_transient_failure_schedules_retry_with_backoff_and_keeps_source(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-retry", attempts=3, source_key="acme/job-retry.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-r",
            lease_token="token-r",
            lease_expires_at=now + timedelta(seconds=300),
        )
        storage = _FakeStorage()
        retry_calls: list[dict[str, Any]] = []

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def schedule_retry(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                available_at: datetime,
                error_code: str,
                error_message: str,
            ) -> IngestionJob:
                retry_calls.append(
                    {
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "lease_token": lease_token,
                        "available_at": available_at,
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                return job

        class FailingService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                raise ConnectionError("embedding upstream reset")

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="s"),
            job_repository_factory=Repo,
            service_factory=FailingService,
            storage=storage,
            worker_id="worker-r",
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert len(retry_calls) == 1
        # attempts=3 -> delay = min(2 * 2^(3-1), 300) = 8 seconds
        assert retry_calls[0]["available_at"] == now + timedelta(seconds=8)
        assert retry_calls[0]["worker_id"] == "worker-r"
        assert retry_calls[0]["lease_token"] == "token-r"
        assert storage.deleted_keys == []

    async def test_exhausted_transient_failure_transitions_to_dead_letter_and_deletes_source(
        self,
    ) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-dlq", attempts=5, source_key="acme/job-dlq.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-d",
            lease_token="token-d",
            lease_expires_at=now + timedelta(seconds=300),
        )
        events: list[str] = []
        storage = _FakeStorage(events=events)
        dead_letters: list[dict[str, Any]] = []

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def dead_letter(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                error_code: str,
                error_message: str,
            ) -> IngestionJob:
                events.append("repo:dead_letter")
                dead_letters.append(
                    {
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "lease_token": lease_token,
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                return job

        class FailingService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                raise TimeoutError("embedding provider timed out")

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="s", events=events),
            job_repository_factory=Repo,
            service_factory=FailingService,
            storage=storage,
            worker_id="worker-d",
            max_attempts=5,
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert len(dead_letters) == 1
        assert dead_letters[0]["job_id"] == "job-dlq"
        assert storage.deleted_keys == ["acme/job-dlq.bin"]
        assert events.index("repo:dead_letter") < events.index("storage:delete:acme/job-dlq.bin")

    async def test_permanent_failure_dead_letters_immediately_on_first_attempt(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-perm", attempts=1, source_key="acme/job-perm.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-p",
            lease_token="token-p",
            lease_expires_at=now + timedelta(seconds=300),
        )
        storage = _FakeStorage()
        dead_letters: list[dict[str, Any]] = []

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def dead_letter(
                self,
                job_id: str,
                *,
                worker_id: str,
                lease_token: str,
                error_code: str,
                error_message: str,
            ) -> IngestionJob:
                dead_letters.append(
                    {
                        "job_id": job_id,
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                return job

        class InvalidFormatService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                raise _PermanentFormatError("unsupported binary format")

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="s"),
            job_repository_factory=Repo,
            service_factory=InvalidFormatService,
            storage=storage,
            worker_id="worker-p",
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert len(dead_letters) == 1
        assert dead_letters[0]["error_code"] == "unsupported_format"
        assert storage.deleted_keys == ["acme/job-perm.bin"]

    async def test_shutdown_stops_polling_and_cancels_heartbeats_cleanly(self) -> None:
        poll_count = 0
        stop_event = asyncio.Event()

        class IdleRepo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                nonlocal poll_count
                poll_count += 1
                if poll_count >= 2:
                    stop_event.set()
                return None

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="idle"),
            job_repository_factory=IdleRepo,
            service_factory=lambda session: None,
            poll_interval_seconds=0.05,
        )

        await asyncio.wait_for(worker.run_forever(stop_event=stop_event), timeout=1.0)
        count_at_shutdown = poll_count
        await asyncio.sleep(0.08)
        assert poll_count == count_at_shutdown

    async def test_worker_logs_omit_sensitive_payload_and_credentials(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-safe-log", attempts=1, source_key="acme/job-safe-log.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-log",
            lease_token="token-log",
            lease_expires_at=now + timedelta(seconds=300),
        )

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def dead_letter(self, *args: Any, **kwargs: Any) -> IngestionJob:
                return job

        class LeakyService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                raise ValueError(
                    "parse failed on secret-extracted-document-text Bearer eyJhbGciOiJIUzI1NiJ9.secret.sig"
                )

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="s"),
            job_repository_factory=Repo,
            service_factory=LeakyService,
            worker_id="worker-log",
            clock=lambda: now,
        )

        with caplog.at_level(logging.INFO, logger="adaptive.ingestion.worker"):
            await worker.run_once()

        logged_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "job-safe-log" in logged_text
        assert "secret-extracted-document-text" not in logged_text
        assert "eyJhbGciOiJIUzI1NiJ9" not in logged_text

    async def test_lease_loss_on_complete_rolls_back_and_preserves_source(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-complete-lost", source_key="acme/job-complete-lost.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-c",
            lease_token="token-c",
            lease_expires_at=now + timedelta(seconds=300),
        )
        storage = _FakeStorage()
        sessions: list[_FakeSession] = []

        def session_factory() -> _FakeSession:
            session = _FakeSession(name=f"s{len(sessions) + 1}")
            sessions.append(session)
            return session

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def complete(self, *args: Any, **kwargs: Any) -> IngestionJob:
                raise LeaseLostError("lease token no longer valid")

        class Service:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process(self, job_id: str) -> IngestionJob:
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-1"}
                )

        worker = IngestionWorker(
            session_factory=session_factory,
            job_repository_factory=Repo,
            service_factory=Service,
            storage=storage,
            worker_id="worker-c",
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert sessions[1].committed is False
        assert sessions[1].rolled_back is True
        assert storage.deleted_keys == []

    async def test_process_claimed_method_is_preferred_when_available(self) -> None:
        now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
        job = _make_job(job_id="job-claimed-api", source_key="acme/job-claimed-api.bin")
        claimed = ClaimedIngestionJob(
            job=job,
            worker_id="worker-claimed",
            lease_token="token-claimed",
            lease_expires_at=now + timedelta(seconds=300),
        )
        seen_claimed: list[ClaimedIngestionJob] = []

        class Repo:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            def claim_next(
                self, worker_id: str, claimed_at: datetime, lease_seconds: int
            ) -> ClaimedIngestionJob | None:
                return claimed

            def complete(self, *args: Any, **kwargs: Any) -> IngestionJob:
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-claimed"}
                )

        class ClaimedAwareService:
            def __init__(self, session: _FakeSession) -> None:
                self.session = session

            async def process_claimed(self, item: ClaimedIngestionJob) -> IngestionJob:
                seen_claimed.append(item)
                return job.model_copy(
                    update={"status": IngestionStatus.ACTIVE, "document_id": "doc-claimed"}
                )

        worker = IngestionWorker(
            session_factory=lambda: _FakeSession(name="s"),
            job_repository_factory=Repo,
            service_factory=ClaimedAwareService,
            worker_id="worker-claimed",
            clock=lambda: now,
        )

        assert await worker.run_once() is True
        assert seen_claimed == [claimed]
