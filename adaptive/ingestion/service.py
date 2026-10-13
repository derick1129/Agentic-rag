"""Asynchronous ingestion orchestration with idempotent local persistence."""

# Repository and provider seams are intentionally duck-typed so synchronous
# SQLAlchemy repositories and asynchronous test doubles share one service API.
# ruff: noqa: ANN401, C901, PLC0415, PLR0912, PLR0913, SIM115, TRY003

from __future__ import annotations

import contextlib
import hashlib
import inspect
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from adaptive.config import get_settings
from adaptive.ingestion.chunking import chunk_document
from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.normalize import normalize
from adaptive.ingestion.parsers import parse_bytes
from adaptive.ingestion.storage import FilesystemSourceStorage
from adaptive.interfaces import (
    CanonicalDocument,
    ChunkDraft,
    ChunkPolicy,
    ClaimedIngestionJob,
    DocumentCreateParams,
    DocumentStatus,
    IdempotencyConflictError,
    IngestionJob,
    IngestionStatus,
    RequestContext,
    SourceStorage,
)

PERMANENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "parse_error",
        "validation_error",
        "unsupported_format",
        "unsupported_mime",
        "invalid_payload",
        "empty_payload",
        "invalid_source",
        "corrupt_document",
    }
)


def classify_failure(exc: Exception, *, stage: str = "parsing") -> tuple[bool, str, str]:
    """Classify failure as (permanent, error_code, error_message)."""
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
    elif (code_str is not None and code_str in PERMANENT_ERROR_CODES) or isinstance(
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

    safe_message = " ".join(str(exc).split())[:500] or "ingestion processing failed"
    return permanent, code_str, safe_message



async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@dataclass
class _MemoryDocument:
    id: str
    tenant_id: str
    content_hash: str
    version: int
    canonical_text: str
    status: str = "active"
    chunks: list[ChunkDraft] = field(default_factory=list)


class _MemoryIngestionJobRepository:
    """In-memory fallback repository for testing when no database session is supplied."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}

    def create(self, job: IngestionJob) -> IngestionJob:
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str, context: RequestContext | None = None) -> IngestionJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if context is not None and job.tenant_id != context.tenant_id:
            return None
        return job

    def get_by_idempotency_key(
        self, tenant_id: str, idempotency_key: str
    ) -> IngestionJob | None:
        for job in self._jobs.values():
            if job.tenant_id == tenant_id and job.idempotency_key == idempotency_key:
                return job
        return None

    def update(self, job: IngestionJob) -> IngestionJob:
        if job.id not in self._jobs:
            raise LookupError("ingestion job not found")
        self._jobs[job.id] = job
        return job

    def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: int
    ) -> ClaimedIngestionJob | None:
        for job in sorted(
            self._jobs.values(), key=lambda j: (j.available_at, j.created_at, j.id)
        ):
            is_due_queued = (
                job.status
                in (
                    IngestionStatus.QUEUED,
                    IngestionStatus.RETRY_WAIT,
                    IngestionStatus.RECEIVED,
                )
                and job.available_at <= now
            )
            is_expired_running = (
                job.status == IngestionStatus.RUNNING
                and job.lease_expires_at is not None
                and job.lease_expires_at <= now
            )
            if is_due_queued or is_expired_running:
                lease_token = str(uuid.uuid4())
                lease_expires = now + timedelta(seconds=lease_seconds)
                updated = job.model_copy(
                    update={
                        "status": IngestionStatus.RUNNING,
                        "worker_id": worker_id,
                        "lease_token": lease_token,
                        "lease_expires_at": lease_expires,
                        "attempts": job.attempts + 1,
                        "updated_at": now,
                    }
                )
                self._jobs[job.id] = updated
                return ClaimedIngestionJob(
                    job=updated,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    lease_expires_at=lease_expires,
                )
        return None

    def heartbeat(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        now: datetime,
        lease_seconds: int,
    ) -> ClaimedIngestionJob:
        job = self._jobs.get(job_id)
        if (
            job is None
            or job.status != IngestionStatus.RUNNING
            or job.worker_id != worker_id
            or job.lease_token != lease_token
            or (job.lease_expires_at is not None and job.lease_expires_at <= now)
        ):
            raise LookupError("ingestion job lease lost")
        lease_expires = now + timedelta(seconds=lease_seconds)
        updated = job.model_copy(update={"lease_expires_at": lease_expires, "updated_at": now})
        self._jobs[job.id] = updated
        return ClaimedIngestionJob(
            job=updated,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_expires_at=lease_expires,
        )

    def update_stage(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        stage: str,
    ) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise LookupError("ingestion job not found")
        if (worker_id is not None or lease_token is not None) and (
            job.status != IngestionStatus.RUNNING
            or job.worker_id != worker_id
            or job.lease_token != lease_token
        ):
            raise LookupError("ingestion job lease lost")

        updated = job.model_copy(update={"current_stage": stage, "updated_at": datetime.now(UTC)})
        self._jobs[job.id] = updated
        return updated

    def schedule_retry(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        available_at: datetime,
        error_code: str,
        error_message: str,
    ) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise LookupError("ingestion job not found")
        if (worker_id is not None or lease_token is not None) and (
            job.status != IngestionStatus.RUNNING
            or job.worker_id != worker_id
            or job.lease_token != lease_token
        ):
            raise LookupError("ingestion job lease lost")

        updated = job.model_copy(
            update={
                "status": IngestionStatus.RETRY_WAIT,
                "current_stage": IngestionStatus.RETRY_WAIT.value,
                "available_at": available_at,
                "error_code": error_code,
                "error_message": " ".join(error_message.split())[:500],
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        self._jobs[job.id] = updated
        return updated

    def complete(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        document_id: str,
    ) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise LookupError("ingestion job not found")
        if (worker_id is not None or lease_token is not None) and (
            job.status != IngestionStatus.RUNNING
            or job.worker_id != worker_id
            or job.lease_token != lease_token
        ):
            raise LookupError("ingestion job lease lost")

        updated = job.model_copy(
            update={
                "status": IngestionStatus.ACTIVE,
                "current_stage": IngestionStatus.ACTIVE.value,
                "document_id": document_id,
                "error_code": None,
                "error_message": None,
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        self._jobs[job.id] = updated
        return updated

    def dead_letter(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        error_code: str,
        error_message: str,
    ) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise LookupError("ingestion job not found")
        if (worker_id is not None or lease_token is not None) and (
            job.status != IngestionStatus.RUNNING
            or job.worker_id != worker_id
            or job.lease_token != lease_token
        ):
            raise LookupError("ingestion job lease lost")

        updated = job.model_copy(
            update={
                "status": IngestionStatus.DEAD_LETTER,
                "current_stage": IngestionStatus.DEAD_LETTER.value,
                "error_code": error_code,
                "error_message": " ".join(error_message.split())[:500],
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        self._jobs[job.id] = updated
        return updated


class IngestionService:
    def __init__(
        self,
        *,
        storage: SourceStorage | None = None,
        job_repository: Any = None,
        document_repository: Any = None,
        version_repository: Any = None,
        chunk_repository: Any = None,
        session: Any = None,
        parser: Callable[..., Any] = parse_bytes,
        chunk_policy: ChunkPolicy | None = None,
        embedding_provider: Any = None,
        indexer: Any = None,
        max_attempts: int = 5,
    ) -> None:
        if session is not None:
            from adaptive.db.repositories import (
                ChunkRepository,
                DocumentRepository,
                DocumentVersionRepository,
                IngestionJobRepository,
            )

            document_repository = document_repository or DocumentRepository(session)
            version_repository = version_repository or DocumentVersionRepository(session)
            chunk_repository = chunk_repository or ChunkRepository(session)
            job_repository = job_repository or IngestionJobRepository(session)

        if storage is not None:
            self.storage = storage
        else:
            try:
                temp_dir = get_settings().ingestion_temp_dir
            except Exception:
                temp_dir = str(Path(tempfile.gettempdir()) / "adaptive_rag_ingestion")
            self.storage = FilesystemSourceStorage(temp_dir)


        self.jobs = (
            job_repository if job_repository is not None else _MemoryIngestionJobRepository()
        )

        self.session = session
        self.documents = document_repository
        self.versions = version_repository
        self.chunks = chunk_repository
        self.parser = parser
        self.chunk_policy = chunk_policy or ChunkPolicy()
        self.embedding_provider = embedding_provider
        self.indexer = indexer
        self.max_attempts = max_attempts
        self._memory_documents: dict[str, _MemoryDocument] = {}

    async def submit(
        self,
        source: Any,
        context: RequestContext,
        *,
        idempotency_key: str | None = None,
    ) -> IngestionJob:
        if hasattr(source, "read"):
            payload = await _maybe_await(source.read())
            if isinstance(payload, bytes):
                payload = SourcePayload(
                    data=payload,
                    filename=getattr(source, "filename", "upload"),
                    mime_type=getattr(source, "mime_type", "text/plain"),
                )
        else:
            payload = source

        if not isinstance(payload, SourcePayload):
            if isinstance(payload, bytes):
                payload = SourcePayload(data=payload, filename="upload", mime_type="text/plain")
            else:
                raise TypeError("source must provide SourcePayload or bytes")
        elif isinstance(payload.data, str):
            payload = payload.model_copy(update={"data": payload.data.encode("utf-8")})

        payload_hash = hashlib.sha256(payload.data).hexdigest()
        raw_filename = payload.filename or getattr(source, "filename", None) or "upload"
        normalized_filename = Path(raw_filename).name.strip() or "upload"

        if idempotency_key is not None:
            existing = await _maybe_await(
                self.jobs.get_by_idempotency_key(context.tenant_id, idempotency_key)
            )
            if existing is not None:
                if (
                    existing.payload_hash == payload_hash
                    and (existing.filename or "upload") == normalized_filename
                ):
                    return existing
                raise IdempotencyConflictError(
                    f"Idempotency key '{idempotency_key}' already used with different payload"
                )
        else:
            derived_key = hashlib.sha256(
                f"{context.tenant_id}:{normalized_filename}:{payload_hash}".encode()
            ).hexdigest()
            existing = await _maybe_await(
                self.jobs.get_by_idempotency_key(context.tenant_id, derived_key)
            )
            if existing is not None:
                return existing
            idempotency_key = derived_key

        job_id = str(uuid.uuid4())
        source_key = self.storage.store(job_id, payload.data)

        job = IngestionJob(
            id=job_id,
            tenant_id=context.tenant_id,
            document_id=None,
            status=IngestionStatus.QUEUED,
            current_stage=IngestionStatus.QUEUED.value,
            attempts=0,
            source_key=source_key,
            filename=normalized_filename,
            mime_type=payload.mime_type,
            source_uri=payload.source_uri,
            metadata=dict(payload.metadata or {}),
            subject_id=context.subject_id or "system",
            acl=frozenset(context.acl) if context.acl else frozenset(),
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
        )

        try:
            persisted = await _maybe_await(self.jobs.create(job))
            if self.session is not None and hasattr(self.session, "commit"):
                self.session.commit()
        except Exception:
            self.storage.delete(source_key)
            if self.session is not None and hasattr(self.session, "rollback"):
                self.session.rollback()
            raise
        else:
            return persisted or job


    async def get_job(
        self, job_id: str, context: RequestContext | None = None
    ) -> IngestionJob | None:
        if self.jobs is not None:
            if "context" in inspect.signature(self.jobs.get).parameters:
                return await _maybe_await(self.jobs.get(job_id, context=context))
            job = await _maybe_await(self.jobs.get(job_id))
            if job is not None and context is not None and job.tenant_id != context.tenant_id:
                return None
            return job
        return None

    async def process(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
    ) -> IngestionJob:
        job = await _maybe_await(self.jobs.get(job_id))
        if job is None:
            raise LookupError("ingestion job not found")
        if job.status == IngestionStatus.ACTIVE:
            return job

        if self.session is not None:
            from adaptive.db.session import set_tenant_context

            set_tenant_context(self.session, job.tenant_id)

        context = RequestContext(
            tenant_id=job.tenant_id,
            subject_id=job.subject_id or "system",
            acl=frozenset(job.acl) if job.acl else frozenset(),
        )

        if not job.source_key:
            raise FileNotFoundError("job source_key is missing")

        with self.storage.open(job.source_key) as f:
            data = f.read()

        payload = SourcePayload(
            data=data,
            filename=job.filename or "source",
            mime_type=job.mime_type or "text/plain",
            source_uri=job.source_uri,
            metadata=dict(job.metadata or {}),
        )

        attempts = job.attempts if (worker_id and lease_token) else job.attempts + 1
        job = job.model_copy(update={"attempts": attempts})

        stage = "parsing"
        try:
            job = await self._transition_stage(
                job, IngestionStatus.PARSING, worker_id=worker_id, lease_token=lease_token
            )
            stage = "parsing"
            parsed = self.parser(payload.data, payload.mime_type)
            if payload.metadata:
                parsed = parsed.model_copy(
                    update={"metadata": {**parsed.metadata, **payload.metadata}}
                )

            job = await self._transition_stage(
                job, IngestionStatus.CHUNKING, worker_id=worker_id, lease_token=lease_token
            )
            stage = "chunking"
            document, chunks = await self._chunk(job, parsed, context)

            job = await self._transition_stage(
                job, IngestionStatus.EMBEDDING, worker_id=worker_id, lease_token=lease_token
            )
            stage = "embedding"
            embeddings = await self._embed(chunks)

            job = await self._transition_stage(
                job, IngestionStatus.INDEXING, worker_id=worker_id, lease_token=lease_token
            )
            stage = "indexing"
            document = await self._persist(job, document, chunks, embeddings, context)

            document_id = document.metadata.get("document_id")

            job = await self._transition_complete(
                job,
                document_id=document_id,
                worker_id=worker_id,
                lease_token=lease_token,
                attempts=attempts,
            )
            if self.session is not None and hasattr(self.session, "commit"):
                self.session.commit()

            if job.source_key:
                self.storage.delete(job.source_key)
        except Exception as exc:
            return await self._handle_process_failure(
                job, exc, stage, attempts, worker_id=worker_id, lease_token=lease_token
            )
        else:
            return job

    async def _handle_process_failure(
        self,
        job: IngestionJob,
        exc: Exception,
        stage: str,
        attempts: int,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
    ) -> IngestionJob:

        permanent, code, error_message = classify_failure(exc, stage=stage)

        if permanent or attempts >= self.max_attempts:
            job = await self._transition_dead_letter(
                job,
                stage=stage,
                error_code=code,
                error_message=error_message,
                worker_id=worker_id,
                lease_token=lease_token,
                attempts=attempts,
            )

            if self.session is not None and hasattr(self.session, "commit"):
                self.session.commit()
            if job.source_key:
                self.storage.delete(job.source_key)
            return job

        delay = min(2 * (2 ** (attempts - 1)), 300)
        available_at = datetime.now(UTC) + timedelta(seconds=delay)
        job = await self._transition_retry_wait(
            job,
            stage=stage,
            available_at=available_at,
            error_code=code,
            error_message=error_message,
            worker_id=worker_id,
            lease_token=lease_token,
            attempts=attempts,
        )
        if self.session is not None and hasattr(self.session, "commit"):
            self.session.commit()
        return job



    async def _transition_stage(
        self,
        job: IngestionJob,
        status: IngestionStatus,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
    ) -> IngestionJob:
        if worker_id is not None and lease_token is not None and hasattr(self.jobs, "update_stage"):
            return await _maybe_await(
                self.jobs.update_stage(
                    job.id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    stage=status.value,
                )
            )
        updated = job.model_copy(update={"status": status, "current_stage": status.value})
        if hasattr(self.jobs, "update"):
            with contextlib.suppress(Exception):
                return await _maybe_await(self.jobs.update(updated))
        return updated

    async def _transition_complete(
        self,
        job: IngestionJob,
        document_id: str | None,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        attempts: int,
    ) -> IngestionJob:
        if worker_id is not None and lease_token is not None and hasattr(self.jobs, "complete"):
            return await _maybe_await(
                self.jobs.complete(
                    job.id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    document_id=document_id or "",
                )
            )
        updated = job.model_copy(
            update={
                "status": IngestionStatus.ACTIVE,
                "current_stage": IngestionStatus.ACTIVE.value,
                "document_id": document_id,
                "error_code": None,
                "error_message": None,
                "attempts": attempts,
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        if hasattr(self.jobs, "complete"):
            with contextlib.suppress(Exception):
                return await _maybe_await(
                    self.jobs.complete(
                        job.id,
                        worker_id=worker_id or "",
                        lease_token=lease_token or "",
                        document_id=document_id or "",
                    )
                )
        if hasattr(self.jobs, "update"):
            return await _maybe_await(self.jobs.update(updated))
        return updated

    async def _transition_dead_letter(
        self,
        job: IngestionJob,
        stage: str,
        error_code: str,
        error_message: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        attempts: int,
    ) -> IngestionJob:
        if worker_id is not None and lease_token is not None and hasattr(self.jobs, "dead_letter"):
            return await _maybe_await(
                self.jobs.dead_letter(
                    job.id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    error_code=error_code,
                    error_message=error_message,
                )
            )
        updated = job.model_copy(
            update={
                "status": IngestionStatus.DEAD_LETTER,
                "current_stage": stage,
                "error_code": error_code,
                "error_message": error_message,
                "attempts": attempts,
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        if hasattr(self.jobs, "dead_letter"):
            with contextlib.suppress(Exception):
                return await _maybe_await(
                    self.jobs.dead_letter(
                        job.id,
                        worker_id=worker_id or "",
                        lease_token=lease_token or "",
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
        if hasattr(self.jobs, "update"):
            return await _maybe_await(self.jobs.update(updated))
        return updated

    async def _transition_retry_wait(
        self,
        job: IngestionJob,
        stage: str,
        available_at: datetime,
        error_code: str,
        error_message: str,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        attempts: int,
    ) -> IngestionJob:
        has_retry = hasattr(self.jobs, "schedule_retry")
        if worker_id is not None and lease_token is not None and has_retry:
            return await _maybe_await(
                self.jobs.schedule_retry(
                    job.id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    available_at=available_at,
                    error_code=error_code,
                    error_message=error_message,
                )
            )
        updated = job.model_copy(
            update={
                "status": IngestionStatus.RETRY_WAIT,
                "current_stage": stage,
                "available_at": available_at,
                "error_code": error_code,
                "error_message": error_message,
                "attempts": attempts,
                "worker_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "updated_at": datetime.now(UTC),
            }
        )
        if hasattr(self.jobs, "schedule_retry"):
            with contextlib.suppress(Exception):
                return await _maybe_await(
                    self.jobs.schedule_retry(
                        job.id,
                        worker_id=worker_id or "",
                        lease_token=lease_token or "",
                        available_at=available_at,
                        error_code=error_code,
                        error_message=error_message,
                    )
                )
        if hasattr(self.jobs, "update"):
            return await _maybe_await(self.jobs.update(updated))
        return updated

    async def _chunk(
        self, job: IngestionJob, parsed: Any, context: RequestContext
    ) -> tuple[CanonicalDocument, list[ChunkDraft]]:
        document = normalize(parsed, acl=context.acl)
        content_hash = document.content_hash
        filename = job.filename or document.metadata.get("filename") or "source"

        existing: Any = None
        if self.documents is not None:
            if hasattr(self.documents, "find_active_by_hash"):
                existing = await _maybe_await(
                    self.documents.find_active_by_hash(job.tenant_id, content_hash)
                )
            if existing is None and hasattr(self.documents, "find_active"):
                existing = await _maybe_await(self.documents.find_active(job.tenant_id, filename))
        else:
            existing = self._find_by_hash(job.tenant_id, content_hash) or self._find_active(
                job.tenant_id
            )

        if existing is not None:
            next_version = (
                existing.version if existing.content_hash == content_hash else existing.version + 1
            )
            document = document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "document_id": existing.id,
                        "tenant_id": job.tenant_id,
                        "version": next_version,
                        "filename": filename,
                    }
                }
            )
        else:
            document = document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "document_id": str(uuid.uuid4()),
                        "tenant_id": job.tenant_id,
                        "version": self._next_version(job.tenant_id),
                        "filename": filename,
                    }
                }
            )
        return document, chunk_document(document, self.chunk_policy)

    async def _embed(self, chunks: list[ChunkDraft]) -> list[list[float]]:
        if self.embedding_provider is not None and chunks:
            return cast(
                "list[list[float]]",
                await _maybe_await(
                    self.embedding_provider.embed([chunk.text for chunk in chunks])
                ),
            )
        return []

    async def _persist_document_repo(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        context: RequestContext,
        replaced_version: int | None,
    ) -> tuple[CanonicalDocument, list[ChunkDraft] | None]:
        params = DocumentCreateParams(
            tenant_id=job.tenant_id,
            filename=document.metadata.get("filename", "source"),
            content_hash=document.content_hash,
            title=document.title,
            canonical_text=document.canonical_text,
            metadata=document.metadata,
            acl=document.acl,
        )
        persisted = (
            await _maybe_await(
                self.documents.get(document.metadata["document_id"], context)
            )
            if hasattr(self.documents, "get")
            else None
        )
        rechunked: list[ChunkDraft] | None = None
        if persisted is None:
            persisted = await _maybe_await(self.documents.create(params))
            if hasattr(self.documents, "activate_version"):
                await _maybe_await(self.documents.activate_version(persisted.id, 1))
            if persisted.id != document.metadata["document_id"]:
                document = document.model_copy(
                    update={"metadata": {**document.metadata, "document_id": persisted.id}}
                )
                rechunked = chunk_document(document, self.chunk_policy)
        elif (
            persisted.content_hash == document.content_hash
            and getattr(persisted, "status", None) in ("active", DocumentStatus.ACTIVE)
        ):
            pass
        elif persisted.content_hash != document.content_hash:
            version = document.metadata["version"]
            if self.versions is None:
                raise RuntimeError("version repository is required for replacement")
            existing_ver = (
                await _maybe_await(self.versions.get(persisted.id, version, context))
                if hasattr(self.versions, "get")
                else None
            )
            if existing_ver is None:
                await _maybe_await(
                    self.versions.create(
                        document_id=persisted.id,
                        tenant_id=job.tenant_id,
                        version=version,
                        content_hash=document.content_hash,
                        title=document.title,
                        canonical_text=document.canonical_text,
                        metadata=document.metadata,
                        acl=document.acl,
                    )
                )
            if getattr(persisted, "version", None) != version:
                await _maybe_await(self.documents.activate_version(persisted.id, version))
            if self.chunks is not None and hasattr(self.chunks, "delete_by_document_version"):
                await _maybe_await(
                    self.chunks.delete_by_document_version(
                        persisted.id, replaced_version or persisted.version, context
                    )
                )
        return document, rechunked

    async def _persist(

        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: list[ChunkDraft],
        embeddings: list[list[float]],
        context: RequestContext,
    ) -> CanonicalDocument:
        existing = self._find_active(job.tenant_id)
        replaced_version = (
            existing.version
            if existing is not None and existing.content_hash != document.content_hash
            else None
        )
        if existing is None:
            record = _MemoryDocument(
                document.metadata["document_id"],
                job.tenant_id,
                document.content_hash,
                document.metadata["version"],
                document.canonical_text,
                chunks=chunks,
            )
            self._memory_documents[record.id] = record
        else:
            next_version = (
                existing.version
                if existing.content_hash == document.content_hash
                else existing.version + 1
            )
            if existing.content_hash != document.content_hash:
                existing.status = "replaced"
            record = _MemoryDocument(
                existing.id,
                job.tenant_id,
                document.content_hash,
                next_version,
                document.canonical_text,
                chunks=chunks,
            )
            self._memory_documents[record.id] = record

        if self.documents is not None:
            document, rechunked = await self._persist_document_repo(
                job, document, context, replaced_version
            )
            if rechunked is not None:
                chunks = rechunked

        # Idempotent chunk persistence: delete existing chunks for (doc_id, version) before insert


        # Idempotent chunk persistence: delete existing chunks for (doc_id, version) before insert
        if self.chunks is not None:
            doc_id = document.metadata["document_id"]
            ver = document.metadata["version"]
            if hasattr(self.chunks, "delete_by_document_version"):
                await _maybe_await(self.chunks.delete_by_document_version(doc_id, ver, context))
            await _maybe_await(self.chunks.bulk_create(chunks, embeddings=embeddings or None))

        if self.indexer is not None:
            if hasattr(self.indexer, "upsert_chunks"):
                await _maybe_await(
                    self.indexer.upsert_chunks(chunks, embeddings=embeddings or None)
                )
            else:
                await _maybe_await(self.indexer.index_chunks(chunks))
            if replaced_version is not None and hasattr(self.indexer, "delete_document_version"):
                await _maybe_await(
                    self.indexer.delete_document_version(
                        document.metadata["document_id"], replaced_version
                    )
                )

        if record.id != document.metadata["document_id"]:
            self._memory_documents.pop(record.id, None)
            record.id = document.metadata["document_id"]
            record.chunks = chunks
            self._memory_documents[record.id] = record

        return document

    def _find_by_hash(self, tenant_id: str, content_hash: str) -> _MemoryDocument | None:
        return next(
            (
                doc
                for doc in self._memory_documents.values()
                if doc.tenant_id == tenant_id
                and doc.content_hash == content_hash
                and doc.status == "active"
            ),
            None,
        )

    def _find_active(self, tenant_id: str) -> _MemoryDocument | None:
        return next(
            (
                doc
                for doc in self._memory_documents.values()
                if doc.tenant_id == tenant_id and doc.status == "active"
            ),
            None,
        )

    def _next_version(self, tenant_id: str) -> int:
        versions = [
            doc.version for doc in self._memory_documents.values() if doc.tenant_id == tenant_id
        ]
        return max(versions, default=0) + 1

    def active_document_count(self, context: RequestContext) -> int:
        return sum(
            doc.tenant_id == context.tenant_id and doc.status == "active"
            for doc in self._memory_documents.values()
        )

    def active_version(self, document_id: str) -> int:
        return self._memory_documents[document_id].version

    def active_chunk_count(self, document_id: str) -> int:
        return len(self._memory_documents[document_id].chunks)
