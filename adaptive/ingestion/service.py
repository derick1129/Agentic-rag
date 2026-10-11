"""Asynchronous ingestion orchestration with idempotent local persistence."""

# Repository and provider seams are intentionally duck-typed so synchronous
# SQLAlchemy repositories and asynchronous test doubles share one service API.
# ruff: noqa: ANN401, C901, PLC0415, PLR0912, PLR0913, SIM115, TRY003

from __future__ import annotations

import inspect
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from adaptive.ingestion.chunking import chunk_document
from adaptive.ingestion.jobs import advance, failed
from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.normalize import normalize
from adaptive.ingestion.parsers import parse_bytes
from adaptive.interfaces import (
    CanonicalDocument,
    ChunkDraft,
    ChunkPolicy,
    DocumentCreateParams,
    IngestionJob,
    IngestionStatus,
    RequestContext,
)


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


class IngestionService:
    def __init__(
        self,
        *,
        document_repository: Any = None,
        version_repository: Any = None,
        chunk_repository: Any = None,
        job_repository: Any = None,
        session: Any = None,
        parser: Callable[..., Any] = parse_bytes,
        chunk_policy: ChunkPolicy | None = None,
        embedding_provider: Any = None,
        indexer: Any = None,
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
        self.session = session
        self.documents = document_repository
        self.versions = version_repository
        self.chunks = chunk_repository
        self.jobs = job_repository
        self.parser = parser
        self.chunk_policy = chunk_policy or ChunkPolicy()
        self.embedding_provider = embedding_provider
        self.indexer = indexer
        self._jobs: dict[str, IngestionJob] = {}
        self._sources: dict[str, Path] = {}
        self._payloads: dict[str, SourcePayload] = {}
        self._contexts: dict[str, RequestContext] = {}
        self._memory_documents: dict[str, _MemoryDocument] = {}

    async def submit(self, source: Any, context: RequestContext) -> IngestionJob:
        if hasattr(source, "read"):
            payload = await _maybe_await(source.read())
            if isinstance(payload, bytes):
                payload = SourcePayload(
                    data=payload, filename=source.filename, mime_type=source.mime_type
                )
        else:
            payload = source
        temp = tempfile.NamedTemporaryFile(prefix="adaptive-ingest-", delete=False)
        try:
            temp.write(payload.data)
        finally:
            temp.close()
        job = IngestionJob(
            id=str(uuid.uuid4()),
            tenant_id=context.tenant_id,
            document_id=None,
            status=IngestionStatus.RECEIVED,
            current_stage=IngestionStatus.RECEIVED.value,
        )
        self._sources[job.id] = Path(temp.name)
        self._payloads[job.id] = payload
        self._contexts[job.id] = context
        self._jobs[job.id] = job
        if self.jobs is not None:
            await _maybe_await(self.jobs.create(job))
        return job

    async def get_job(self, job_id: str) -> IngestionJob | None:
        job = self._jobs.get(job_id)
        if job is None and self.jobs is not None:
            job = await _maybe_await(self.jobs.get(job_id))
        return job

    async def process(self, job_id: str) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None and self.jobs is not None:
            job = await _maybe_await(self.jobs.get(job_id))
        if job is None:
            raise LookupError("ingestion job not found")  # noqa: TRY003
        if job.status == IngestionStatus.ACTIVE:
            return job
        path = self._sources.get(job.id)
        try:
            if self.session is not None:
                from adaptive.db.session import set_tenant_context

                set_tenant_context(self.session, job.tenant_id)
            retrying = job.status == IngestionStatus.FAILED
            if retrying:
                # Re-run the deterministic pipeline from parsing; attempts and diagnostics persist.
                job = job.model_copy(update={"status": IngestionStatus.RECEIVED})
            if path is None and job.id in self._payloads:
                temp = tempfile.NamedTemporaryFile(prefix="adaptive-ingest-", delete=False)
                temp.write(self._payloads[job.id].data)
                temp.close()
                path = Path(temp.name)
                self._sources[job.id] = path
            if path is None:
                raise FileNotFoundError("temporary source is unavailable")  # noqa: TRY301
            payload = self._payloads[job.id].model_copy(update={"data": path.read_bytes()})
            job = advance(job)
            job = await self._parse(job, payload)
            job = advance(job)
            document, chunks = await self._chunk(job)
            job = advance(job)
            embeddings = await self._embed(chunks)
            job = advance(job)
            document = await self._persist(job, document, chunks, embeddings)
            job = advance(job)
            job = job.model_copy(update={"document_id": document.metadata.get("document_id")})
        except Exception as exc:  # noqa: BLE001
            stage = job.current_stage
            code = {
                "parsing": "parse_error",
                "chunking": "chunk_error",
                "embedding": "embedding_error",
                "indexing": "index_error",
            }.get(stage, "ingestion_error")
            job = failed(job, stage, code, str(exc))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
                self._sources.pop(job.id, None)
            if job.status == IngestionStatus.ACTIVE:
                self._payloads.pop(job.id, None)
                self._contexts.pop(job.id, None)
        self._jobs[job.id] = job
        if self.jobs is not None:
            await _maybe_await(self.jobs.update(job))
        if self.session is not None:
            self.session.commit()
        return job

    async def _parse(self, job: IngestionJob, payload: SourcePayload) -> IngestionJob:
        parsed = self.parser(payload.data, payload.mime_type)
        if payload.metadata:
            parsed = parsed.model_copy(update={"metadata": {**parsed.metadata, **payload.metadata}})
        self._parsed = parsed
        return job

    async def _chunk(self, job: IngestionJob) -> tuple[CanonicalDocument, list[ChunkDraft]]:
        document = normalize(self._parsed, acl=self._contexts[job.id].acl)
        content_hash = document.content_hash
        existing = self._find_by_hash(job.tenant_id, content_hash) or self._find_active(
            job.tenant_id
        )
        if existing is None and self.documents is not None:
            filename = document.metadata.get("filename")
            existing = (
                await _maybe_await(self.documents.find_active(job.tenant_id, filename))
                if hasattr(self.documents, "find_active")
                else None
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
                    }
                }
            )
        return document, chunk_document(document, self.chunk_policy)

    async def _embed(self, chunks: list[ChunkDraft]) -> list[list[float]]:
        if self.embedding_provider is not None and chunks:
            return cast(
                "list[list[float]]",
                await _maybe_await(self.embedding_provider.embed([chunk.text for chunk in chunks])),
            )
        return []

    async def _persist(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: list[ChunkDraft],
        embeddings: list[list[float]],
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
            params = DocumentCreateParams(
                tenant_id=job.tenant_id,
                filename="source",
                content_hash=document.content_hash,
                title=document.title,
                canonical_text=document.canonical_text,
                metadata=document.metadata,
                acl=document.acl,
            )
            persisted = (
                await _maybe_await(
                    self.documents.get(document.metadata["document_id"], self._contexts[job.id])
                )
                if hasattr(self.documents, "get")
                else None
            )
            if persisted is None:
                persisted = await _maybe_await(self.documents.create(params))
                if hasattr(self.documents, "activate_version"):
                    await _maybe_await(self.documents.activate_version(persisted.id, 1))
                if persisted.id != document.metadata["document_id"]:
                    document = document.model_copy(
                        update={"metadata": {**document.metadata, "document_id": persisted.id}}
                    )
                    chunks = chunk_document(document, self.chunk_policy)
            elif persisted.content_hash != document.content_hash:
                version = document.metadata["version"]
                if self.versions is None:
                    raise RuntimeError("version repository is required for replacement")  # noqa: TRY003
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
                await _maybe_await(self.documents.activate_version(persisted.id, version))
                if self.chunks is not None:
                    await _maybe_await(
                        self.chunks.delete_by_document_version(
                            persisted.id, persisted.version, self._contexts[job.id]
                        )
                    )
        if self.chunks is not None:
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
