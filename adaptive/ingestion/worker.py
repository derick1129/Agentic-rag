"""Leased background worker for durable PostgreSQL-backed ingestion jobs."""

# Duck-typed repository, session, storage, and service factories allow unit tests
# and SQLAlchemy runtime sessions to share one worker implementation.
# ruff: noqa: ANN401, C901, PLC0415, PLR0911, PLR0912, PLR0913, PLR0915, TRY003, TRY300, TRY301, TRY400

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import inspect
import logging
import re
import signal
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import adaptive.interfaces as _interfaces

if TYPE_CHECKING:
    from collections.abc import Callable

    from adaptive.config import Settings
    from adaptive.interfaces import IngestionJob

logger = logging.getLogger("adaptive.ingestion.worker")

PERMANENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "parse_error",
        "validation_error",
        "unsupported_format",
        "empty_payload",
        "invalid_source",
        "corrupt_document",
    }
)

_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+\S+")


@dataclass(frozen=True)
class ClaimedIngestionJob:
    """Lease ownership wrapper for a claimed ingestion job."""

    job: IngestionJob
    worker_id: str
    lease_token: str
    lease_expires_at: datetime


class LeaseLostError(LookupError):
    """Raised when a worker no longer owns the active job lease."""


if not TYPE_CHECKING:
    ClaimedIngestionJob = getattr(_interfaces, "ClaimedIngestionJob", ClaimedIngestionJob)
    LeaseLostError = getattr(_interfaces, "LeaseLostError", LeaseLostError)


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _status_str(status: Any) -> str:
    return str(status.value) if hasattr(status, "value") else str(status)


def _tenant_scope_hash(tenant_id: str) -> str:
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:12]


def _safe_log_extra(
    job: IngestionJob,
    *,
    status: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "job_id": job.id,
        "tenant_scope": _tenant_scope_hash(job.tenant_id),
        "status": status or _status_str(job.status),
        "stage": job.current_stage,
        "attempts": job.attempts,
    }
    if error_code is not None:
        extra["error_code"] = error_code
    return extra


def sanitize_error_message(message: str) -> str:
    """Collapse whitespace, redact bearer/JWT tokens, and bound diagnostic length."""
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", str(message))
    redacted = _JWT_PATTERN.sub("[REDACTED_JWT]", redacted)
    collapsed = " ".join(redacted.split())[:500]
    return collapsed or "ingestion processing failed"


def compute_retry_delay_seconds(
    attempts: int,
    *,
    base_seconds: int = 2,
    max_seconds: int = 300,
) -> int:
    """Compute bounded exponential backoff `min(base * 2^(attempts - 1), max)`."""
    if base_seconds < 1:
        raise ValueError("base_seconds must be >= 1")
    if max_seconds < base_seconds:
        raise ValueError("max_seconds must be >= base_seconds")
    exponent = max(attempts - 1, 0)
    return min(base_seconds * (1 << exponent), max_seconds)


def classify_failure(exc: Exception, *, stage: str = "parsing") -> tuple[bool, str, str]:
    """Classify an exception into (is_permanent, error_code, sanitized_message)."""
    explicit_permanent = getattr(exc, "permanent", None)
    if explicit_permanent is None:
        explicit_permanent = getattr(exc, "is_permanent", None)
    explicit_retryable = getattr(exc, "retryable", None)
    if explicit_retryable is None:
        explicit_retryable = getattr(exc, "is_retryable", None)

    raw_code = getattr(exc, "error_code", None) or getattr(exc, "code", None)
    code_str = raw_code if isinstance(raw_code, str) and raw_code.strip() else None

    cls_name = type(exc).__name__.lower()
    if explicit_permanent is True or explicit_retryable is False:
        permanent = True
    elif explicit_permanent is False or explicit_retryable is True:
        permanent = False
    elif code_str in PERMANENT_ERROR_CODES or isinstance(
        exc, (ValueError, TypeError, UnicodeError)
    ):
        permanent = True
    else:
        permanent = any(
            token in cls_name
            for token in ("parse", "validation", "unsupported", "format", "corrupt", "permanent")
        )

    if code_str is None:
        if permanent:
            if "unsupported" in cls_name or "format" in cls_name:
                code_str = "unsupported_format"
            elif "validation" in cls_name:
                code_str = "validation_error"
            else:
                code_str = "parse_error"
        else:
            code_str = {
                "parsing": "parse_error",
                "chunking": "chunk_error",
                "embedding": "embedding_error",
                "indexing": "index_error",
            }.get(stage, "ingestion_error")

    return permanent, code_str, sanitize_error_message(str(exc) or type(exc).__name__)


def _is_lease_loss_error(exc: BaseException) -> bool:
    if isinstance(exc, LeaseLostError):
        return True
    if isinstance(exc, LookupError):
        msg = str(exc).lower()
        return "lease" in msg or "token" in msg or "owner" in msg
    return False


@dataclass
class _TransitionTracker:
    transition: str | None = None
    job: Any = None


class _GuardedJobRepoProxy:
    """Forwards repository calls while blocking writes after lease loss."""

    def __init__(
        self,
        repo: Any,
        *,
        lease_lost: asyncio.Event,
        tracker: _TransitionTracker,
    ) -> None:
        self._repo = repo
        self._lease_lost = lease_lost
        self._tracker = tracker

    def _ensure_lease(self) -> None:
        if self._lease_lost.is_set():
            raise LeaseLostError("ingestion job lease lost")

    def update(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_lease()
        return self._repo.update(*args, **kwargs)

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_lease()
        res = await _maybe_await(self._repo.complete(*args, **kwargs))
        self._tracker.transition = "complete"
        self._tracker.job = res
        return res

    async def schedule_retry(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_lease()
        res = await _maybe_await(self._repo.schedule_retry(*args, **kwargs))
        self._tracker.transition = "schedule_retry"
        self._tracker.job = res
        return res

    async def dead_letter(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_lease()
        res = await _maybe_await(self._repo.dead_letter(*args, **kwargs))
        self._tracker.transition = "dead_letter"
        self._tracker.job = res
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repo, name)


def _default_job_repository_factory(session: Any) -> Any:
    repos_mod = importlib.import_module("adaptive.db.repositories")
    return repos_mod.IngestionJobRepository(session)


class IngestionWorker:
    """Polls due ingestion jobs, renews leases via heartbeats, and records guarded outcomes."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        job_repository_factory: Callable[[Any], Any] | None = None,
        service_factory: Callable[[Any], Any],
        storage: Any = None,
        worker_id: str | None = None,
        lease_seconds: int = 300,
        max_attempts: int = 5,
        retry_base_seconds: int = 2,
        retry_max_seconds: int = 300,
        poll_interval_seconds: float = 1.0,
        heartbeat_interval_seconds: float | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if retry_base_seconds < 1:
            raise ValueError("retry_base_seconds must be >= 1")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError("retry_max_seconds must be >= retry_base_seconds")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if heartbeat_interval_seconds is not None and heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")

        self.session_factory = session_factory
        self.job_repository_factory = job_repository_factory or _default_job_repository_factory
        self.service_factory = service_factory
        self.storage = storage
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else max(min(lease_seconds / 3.0, 30.0), 0.05)
        )
        self.clock = clock
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        """Signal the worker to stop polling and exit cleanly."""
        self._stop_event.set()

    async def _claim_job(self) -> ClaimedIngestionJob | None:
        claim_session = self.session_factory()
        try:
            claim_repo = self.job_repository_factory(claim_session)
            claimed: ClaimedIngestionJob | None = await _maybe_await(
                claim_repo.claim_next(self.worker_id, self.clock(), self.lease_seconds)
            )
            if claimed is None:
                return None
            if hasattr(claim_session, "commit"):
                await _maybe_await(claim_session.commit())
            logger.info(
                "Claimed ingestion job %s (attempt %d)",
                claimed.job.id,
                claimed.job.attempts,
                extra=_safe_log_extra(claimed.job, status="running"),
            )
            return claimed
        except Exception:
            if hasattr(claim_session, "rollback"):
                await _maybe_await(claim_session.rollback())
            raise
        finally:
            if hasattr(claim_session, "close"):
                await _maybe_await(claim_session.close())

    async def _invoke_service(self, service: Any, claimed: ClaimedIngestionJob) -> Any:
        if hasattr(service, "process_claimed") and callable(service.process_claimed):
            return await _maybe_await(service.process_claimed(claimed))

        process_fn = service.process
        sig = inspect.signature(process_fn)
        kwargs: dict[str, Any] = {}
        if "worker_id" in sig.parameters:
            kwargs["worker_id"] = claimed.worker_id
        if "lease_token" in sig.parameters:
            kwargs["lease_token"] = claimed.lease_token
        return await _maybe_await(process_fn(claimed.job.id, **kwargs))

    async def _delete_source_after_commit(
        self,
        service: Any,
        claimed: ClaimedIngestionJob,
        final_job: Any = None,
    ) -> None:
        storage = (
            self.storage
            or getattr(service, "storage", None)
            or getattr(service, "source_storage", None)
        )
        if storage is None or not hasattr(storage, "delete"):
            return
        source_key = getattr(final_job, "source_key", None) or getattr(
            claimed.job, "source_key", None
        )
        if not source_key:
            return
        with contextlib.suppress(FileNotFoundError):
            await _maybe_await(storage.delete(source_key))

    async def _reset_dirty_sqlalchemy_session(self, session: Any) -> None:
        if (
            not getattr(session, "is_active", True)
            or getattr(session, "new", None)
            or getattr(session, "dirty", None)
            or getattr(session, "deleted", None)
        ) and hasattr(session, "rollback"):
            await _maybe_await(session.rollback())

    async def run_once(self) -> bool:
        """Claim and process at most one due ingestion job. Returns True if a job was claimed."""
        claimed = await self._claim_job()
        if claimed is None:
            return False

        active_lease = claimed
        lease_lost = asyncio.Event()
        heartbeat_stop = asyncio.Event()
        tracker = _TransitionTracker()

        process_session = self.session_factory()
        raw_repo = self.job_repository_factory(process_session)
        guarded_repo = _GuardedJobRepoProxy(raw_repo, lease_lost=lease_lost, tracker=tracker)
        service = self.service_factory(process_session)
        if service is not None and getattr(service, "jobs", None) is not None:
            service.jobs = _GuardedJobRepoProxy(
                service.jobs, lease_lost=lease_lost, tracker=tracker
            )

        processing_task = asyncio.create_task(self._invoke_service(service, active_lease))

        async def _heartbeat_loop() -> None:
            nonlocal active_lease
            while not heartbeat_stop.is_set() and not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        heartbeat_stop.wait(),
                        timeout=self.heartbeat_interval_seconds,
                    )
                    break
                except TimeoutError:
                    pass
                if heartbeat_stop.is_set() or self._stop_event.is_set():
                    break
                hb_session = self.session_factory()
                try:
                    hb_repo = self.job_repository_factory(hb_session)
                    if not hasattr(hb_repo, "heartbeat"):
                        break
                    refreshed = await _maybe_await(
                        hb_repo.heartbeat(
                            active_lease.job.id,
                            worker_id=active_lease.worker_id,
                            lease_token=active_lease.lease_token,
                            now=self.clock(),
                            lease_seconds=self.lease_seconds,
                        )
                    )
                    if refreshed is None:
                        raise LeaseLostError("ingestion job lease lost during heartbeat")
                    if hasattr(hb_session, "commit"):
                        await _maybe_await(hb_session.commit())
                    active_lease = refreshed
                except LookupError:
                    lease_lost.set()
                    if hasattr(hb_session, "rollback"):
                        await _maybe_await(hb_session.rollback())
                    logger.warning(
                        "Lost lease for ingestion job %s during heartbeat",
                        active_lease.job.id,
                        extra=_safe_log_extra(
                            active_lease.job, status="running", error_code="lease_lost"
                        ),
                    )
                    processing_task.cancel()
                    break
                except Exception:
                    if hasattr(hb_session, "rollback"):
                        await _maybe_await(hb_session.rollback())
                    logger.warning(
                        "Heartbeat failed for ingestion job %s",
                        active_lease.job.id,
                        extra=_safe_log_extra(
                            active_lease.job, status="running", error_code="heartbeat_error"
                        ),
                    )
                finally:
                    if hasattr(hb_session, "close"):
                        await _maybe_await(hb_session.close())

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        try:
            try:
                result = await processing_task
            except asyncio.CancelledError:
                if lease_lost.is_set():
                    if hasattr(process_session, "rollback"):
                        await _maybe_await(process_session.rollback())
                    return True
                if hasattr(process_session, "rollback"):
                    await _maybe_await(process_session.rollback())
                raise
            except Exception as exc:  # noqa: BLE001
                if lease_lost.is_set() or _is_lease_loss_error(exc):
                    lease_lost.set()
                    if hasattr(process_session, "rollback"):
                        await _maybe_await(process_session.rollback())
                    logger.warning(
                        "Ingestion job %s aborted after lease loss",
                        active_lease.job.id,
                        extra=_safe_log_extra(
                            active_lease.job, status="running", error_code="lease_lost"
                        ),
                    )
                    return True

                # Stop heartbeats before recording failure transition
                heartbeat_stop.set()
                await self._reset_dirty_sqlalchemy_session(process_session)

                attempts = max(active_lease.job.attempts, 1)
                permanent, error_code, error_message = classify_failure(
                    exc, stage=active_lease.job.current_stage
                )
                try:
                    if permanent or attempts >= self.max_attempts:
                        if tracker.transition is None:
                            await guarded_repo.dead_letter(
                                active_lease.job.id,
                                worker_id=active_lease.worker_id,
                                lease_token=active_lease.lease_token,
                                error_code=error_code,
                                error_message=error_message,
                            )
                        if hasattr(process_session, "commit"):
                            await _maybe_await(process_session.commit())
                        await self._delete_source_after_commit(service, active_lease, tracker.job)
                        logger.error(
                            "Ingestion job %s dead-lettered (attempt %d, code=%s)",
                            active_lease.job.id,
                            attempts,
                            error_code,
                            extra=_safe_log_extra(
                                active_lease.job,
                                status="dead_letter",
                                error_code=error_code,
                            ),
                        )
                    else:
                        delay = compute_retry_delay_seconds(
                            attempts,
                            base_seconds=self.retry_base_seconds,
                            max_seconds=self.retry_max_seconds,
                        )
                        available_at = self.clock() + timedelta(seconds=delay)
                        if tracker.transition is None:
                            await guarded_repo.schedule_retry(
                                active_lease.job.id,
                                worker_id=active_lease.worker_id,
                                lease_token=active_lease.lease_token,
                                available_at=available_at,
                                error_code=error_code,
                                error_message=error_message,
                            )
                        if hasattr(process_session, "commit"):
                            await _maybe_await(process_session.commit())
                        logger.warning(
                            "Ingestion job %s scheduled for retry in %ds (attempt %d, code=%s)",
                            active_lease.job.id,
                            delay,
                            attempts,
                            error_code,
                            extra=_safe_log_extra(
                                active_lease.job,
                                status="retry_wait",
                                error_code=error_code,
                            ),
                        )
                except LookupError:
                    lease_lost.set()
                    if hasattr(process_session, "rollback"):
                        await _maybe_await(process_session.rollback())
                    logger.warning(
                        "Lost lease for ingestion job %s while persisting failure outcome",
                        active_lease.job.id,
                        extra=_safe_log_extra(
                            active_lease.job, status="running", error_code="lease_lost"
                        ),
                    )
                return True

            # Processing task returned without raising
            heartbeat_stop.set()
            if lease_lost.is_set():
                if hasattr(process_session, "rollback"):
                    await _maybe_await(process_session.rollback())
                return True

            try:
                outcome_kind, final_job = await self._persist_returned_outcome(
                    guarded_repo, tracker, active_lease, result
                )
                if hasattr(process_session, "commit"):
                    await _maybe_await(process_session.commit())
            except LookupError:
                lease_lost.set()
                if hasattr(process_session, "rollback"):
                    await _maybe_await(process_session.rollback())
                logger.warning(
                    "Lost lease for ingestion job %s while persisting outcome",
                    active_lease.job.id,
                    extra=_safe_log_extra(
                        active_lease.job, status="running", error_code="lease_lost"
                    ),
                )
                return True

            if outcome_kind in {"complete", "dead_letter"}:
                await self._delete_source_after_commit(service, active_lease, final_job)

            logger.info(
                "Finished ingestion job %s with outcome %s",
                active_lease.job.id,
                outcome_kind,
                extra=_safe_log_extra(
                    active_lease.job,
                    status="active" if outcome_kind == "complete" else outcome_kind,
                ),
            )
            return True
        finally:
            heartbeat_stop.set()
            if not heartbeat_task.done():
                heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat_task
            if hasattr(process_session, "close"):
                await _maybe_await(process_session.close())

    async def _persist_returned_outcome(
        self,
        guarded_repo: _GuardedJobRepoProxy,
        tracker: _TransitionTracker,
        active_lease: ClaimedIngestionJob,
        result: Any,
    ) -> tuple[str, Any]:
        if tracker.transition is not None:
            return tracker.transition, tracker.job or result

        attempts = max(active_lease.job.attempts, 1)
        result_status = (
            _status_str(result.status).lower() if hasattr(result, "status") else "active"
        )

        if result_status == "dead_letter":
            error_code = getattr(result, "error_code", None) or "parse_error"
            error_message = sanitize_error_message(
                getattr(result, "error_message", None) or "permanent ingestion failure"
            )
            job = await guarded_repo.dead_letter(
                active_lease.job.id,
                worker_id=active_lease.worker_id,
                lease_token=active_lease.lease_token,
                error_code=error_code,
                error_message=error_message,
            )
            return "dead_letter", job

        if result_status == "retry_wait":
            delay = compute_retry_delay_seconds(
                attempts,
                base_seconds=self.retry_base_seconds,
                max_seconds=self.retry_max_seconds,
            )
            available_at = getattr(result, "available_at", None) or (
                self.clock() + timedelta(seconds=delay)
            )
            error_code = getattr(result, "error_code", None) or "ingestion_error"
            error_message = sanitize_error_message(
                getattr(result, "error_message", None) or "transient ingestion failure"
            )
            job = await guarded_repo.schedule_retry(
                active_lease.job.id,
                worker_id=active_lease.worker_id,
                lease_token=active_lease.lease_token,
                available_at=available_at,
                error_code=error_code,
                error_message=error_message,
            )
            return "schedule_retry", job

        if result_status == "failed":
            error_code = getattr(result, "error_code", None) or "ingestion_error"
            error_message = sanitize_error_message(
                getattr(result, "error_message", None) or "ingestion processing failed"
            )
            if error_code in PERMANENT_ERROR_CODES or attempts >= self.max_attempts:
                job = await guarded_repo.dead_letter(
                    active_lease.job.id,
                    worker_id=active_lease.worker_id,
                    lease_token=active_lease.lease_token,
                    error_code=error_code,
                    error_message=error_message,
                )
                return "dead_letter", job
            delay = compute_retry_delay_seconds(
                attempts,
                base_seconds=self.retry_base_seconds,
                max_seconds=self.retry_max_seconds,
            )
            job = await guarded_repo.schedule_retry(
                active_lease.job.id,
                worker_id=active_lease.worker_id,
                lease_token=active_lease.lease_token,
                available_at=self.clock() + timedelta(seconds=delay),
                error_code=error_code,
                error_message=error_message,
            )
            return "schedule_retry", job

        if isinstance(result, str):
            document_id = result
        elif hasattr(result, "document_id"):
            document_id = getattr(result, "document_id", None) or ""
        elif hasattr(result, "id") and not hasattr(result, "status"):
            document_id = getattr(result, "id", None) or ""
        else:
            document_id = getattr(active_lease.job, "document_id", None) or ""

        job = await guarded_repo.complete(
            active_lease.job.id,
            worker_id=active_lease.worker_id,
            lease_token=active_lease.lease_token,
            document_id=document_id,
        )
        return "complete", job

    async def _wait_for_stop(self, stop_event: asyncio.Event | None, timeout: float) -> bool:
        if stop_event is None:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
            except TimeoutError:
                return self._stop_event.is_set()
            else:
                return True

        wait_tasks = [
            asyncio.create_task(self._stop_event.wait()),
            asyncio.create_task(stop_event.wait()),
        ]
        try:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            return bool(done)
        finally:
            for task in wait_tasks:
                if not task.done():
                    task.cancel()

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        """Continuously poll and execute due ingestion jobs until stopped."""
        while not self._stop_event.is_set() and not (stop_event and stop_event.is_set()):
            processed = await self.run_once()
            if self._stop_event.is_set() or (stop_event and stop_event.is_set()):
                break
            if not processed and await self._wait_for_stop(stop_event, self.poll_interval_seconds):
                break


def _build_embedding_provider(settings: Settings) -> Any:
    fallback_mod = importlib.import_module("adaptive.providers.fallback_embedding")
    openai_mod = importlib.import_module("adaptive.providers.openai")
    contracts_mod = importlib.import_module("adaptive.retrieval.contracts")
    hash_cls = contracts_mod.HashEmbeddingProvider
    openai_cls = openai_mod.OpenAIEmbeddingProvider
    fallback_cls = fallback_mod.FallbackEmbeddingProvider

    if settings.embedding_provider in ("local", "fake"):
        return hash_cls(dimensions=settings.embedding_dimensions)

    nims_provider = None
    if settings.nims_api_key:
        nims_provider = openai_cls(
            api_key=settings.nims_api_key,
            api_base=settings.nims_base_url,
            model=settings.nims_embedding_model or "nvidia/llama-nemotron-embed-vl-1b-v2",
            dimensions=settings.embedding_dimensions,
            input_type="passage",
        )

    openai_provider = None
    if settings.embedding_api_key:
        openai_provider = openai_cls(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_api_base,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    if openai_provider and nims_provider:
        return fallback_cls(primary=openai_provider, fallback=nims_provider)
    if nims_provider:
        return nims_provider
    if openai_provider:
        return openai_provider
    return hash_cls(dimensions=settings.embedding_dimensions)


def _build_default_storage(settings: Settings) -> Any:
    try:
        storage_mod = importlib.import_module("adaptive.ingestion.storage")
    except ImportError:
        return None

    for attr_name in ("FilesystemSourceStorage", "LocalSourceStorage", "SourceStorage"):
        storage_cls = getattr(storage_mod, attr_name, None)
        if isinstance(storage_cls, type):
            with contextlib.suppress(TypeError):
                return storage_cls(settings.ingestion_temp_dir)
    return None


def build_worker_from_settings(settings: Settings | None = None) -> IngestionWorker:
    """Construct a production IngestionWorker wired to configured repositories and storage."""
    from adaptive.config import get_settings
    from adaptive.db.session import SessionLocal
    from adaptive.interfaces import ChunkPolicy

    service_mod = importlib.import_module("adaptive.ingestion.service")
    service_cls = service_mod.IngestionService

    resolved = settings or get_settings()
    embedding_provider = _build_embedding_provider(resolved)
    storage = _build_default_storage(resolved)
    chunk_policy = ChunkPolicy(
        max_tokens=resolved.ingestion_chunk_max_tokens,
        overlap_tokens=resolved.ingestion_chunk_overlap_tokens,
    )

    def service_factory(session: Any) -> Any:
        sig = inspect.signature(service_cls)
        kwargs: dict[str, Any] = {
            "session": session,
            "chunk_policy": chunk_policy,
            "embedding_provider": embedding_provider,
            "indexer": None,
        }
        if storage is not None:
            if "storage" in sig.parameters:
                kwargs["storage"] = storage
            elif "source_storage" in sig.parameters:
                kwargs["source_storage"] = storage
        return service_cls(**kwargs)

    return IngestionWorker(
        session_factory=SessionLocal,
        service_factory=service_factory,
        storage=storage,
        lease_seconds=resolved.ingestion_job_lease_seconds,
        max_attempts=resolved.ingestion_max_attempts,
        retry_base_seconds=resolved.ingestion_retry_base_seconds,
        retry_max_seconds=resolved.ingestion_retry_max_seconds,
        poll_interval_seconds=resolved.ingestion_poll_interval_seconds,
        heartbeat_interval_seconds=resolved.ingestion_heartbeat_interval_seconds,
    )


async def main(stop_event: asyncio.Event | None = None) -> None:
    """Run the ingestion worker process until SIGINT/SIGTERM or stop_event."""
    worker = build_worker_from_settings()
    loop = asyncio.get_running_loop()
    registered_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, worker.stop)
            registered_signals.append(sig)
    try:
        await worker.run_forever(stop_event=stop_event)
    finally:
        for sig in registered_signals:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
