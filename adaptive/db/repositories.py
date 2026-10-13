"""Tenant-scoped repositories for the PostgreSQL control plane."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from sqlalchemy.orm import InstrumentedAttribute, Session
    from sqlalchemy.sql.elements import ColumnElement

from adaptive.db.models import (
    Chunk as ChunkModel,
)
from adaptive.db.models import (
    Document as DocumentModel,
)
from adaptive.db.models import (
    DocumentVersion,
    new_id,
)
from adaptive.db.models import (
    EvaluationAnnotation as EvaluationModel,
)
from adaptive.db.models import (
    IngestionJob as JobModel,
)
from adaptive.db.models import (
    QueryRun as QueryModel,
)
from adaptive.db.models import (
    SemanticCacheEntry as CacheModel,
)
from adaptive.db.models import (
    Tenant as TenantModel,
)
from adaptive.interfaces import (
    ChunkDraft,
    ClaimedIngestionJob,
    Document,
    DocumentCreateParams,
    DocumentStatus,
    IngestionJob,
    IngestionStatus,
    RequestContext,
)


class LeaseLostError(LookupError):
    """Raised when an ingestion job lease is no longer owned by the worker."""


def _scope(
    context: RequestContext,
    tenant_column: ColumnElement[str] | InstrumentedAttribute[str],
) -> ColumnElement[bool]:
    return tenant_column == context.tenant_id


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class TenantRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, slug: str, name: str, metadata: dict[str, Any] | None = None) -> TenantModel:
        tenant = TenantModel(id=new_id(), slug=slug, name=name, metadata_json=metadata or {})
        self.session.add(tenant)
        self.session.flush()
        return tenant

    def get(self, tenant_id: str) -> TenantModel | None:
        return self.session.get(TenantModel, tenant_id)


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, params: DocumentCreateParams) -> Document:
        model = DocumentModel(
            id=new_id(),
            tenant_id=params.tenant_id,
            filename=params.filename,
            content_hash=params.content_hash,
            title=params.title,
            canonical_text=params.canonical_text,
            metadata_json=params.metadata,
            acl=sorted(params.acl),
            status="active",
            version=1,
        )
        self.session.add(model)
        self.session.add(
            DocumentVersion(
                document_id=model.id,
                tenant_id=model.tenant_id,
                version=1,
                content_hash=model.content_hash,
                title=model.title,
                canonical_text=model.canonical_text,
                metadata_json=model.metadata_json,
                acl=model.acl,
                status="pending",
            )
        )
        self.session.flush()
        return self._to_contract(model)

    def get(self, document_id: str, context: RequestContext) -> Document | None:
        model = self.session.scalar(
            select(DocumentModel).where(
                DocumentModel.id == document_id, _scope(context, DocumentModel.tenant_id)
            )
        )
        return (
            self._to_contract(model) if model and context.can_access(frozenset(model.acl)) else None
        )

    def find_active(self, tenant_id: str, filename: str | None = None) -> Document | None:
        query = select(DocumentModel).where(
            DocumentModel.tenant_id == tenant_id, DocumentModel.status == "active"
        )
        if filename:
            query = query.where(DocumentModel.filename == filename)
        model = self.session.scalar(query.order_by(DocumentModel.created_at))
        return self._to_contract(model) if model else None

    def find_active_by_hash(self, tenant_id: str, content_hash: str) -> Document | None:
        model = self.session.scalar(
            select(DocumentModel).where(
                DocumentModel.tenant_id == tenant_id,
                DocumentModel.content_hash == content_hash,
                DocumentModel.status == "active",
            )
        )
        return self._to_contract(model) if model else None

    def list_active(self, context: RequestContext, limit: int = 100) -> list[Document]:
        models = self.session.scalars(
            select(DocumentModel)
            .where(DocumentModel.status == "active", _scope(context, DocumentModel.tenant_id))
            .order_by(DocumentModel.created_at)
            .limit(limit)
        ).all()
        return [
            self._to_contract(model) for model in models if context.can_access(frozenset(model.acl))
        ]

    def activate_version(self, document_id: str, version: int) -> Document:
        model = self.session.get(DocumentModel, document_id)
        record = self.session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id, DocumentVersion.version == version
            )
        )
        if not model or not record:
            raise LookupError("document version not found")  # noqa: TRY003
        if record.status == "active":
            raise IntegrityError(  # noqa: TRY003
                "active document version already exists", {}, Exception("duplicate active version")
            )
        active = self.session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id, DocumentVersion.status == "active"
            )
        )
        if active:
            active.status = "replaced"
            # Flush the deactivation first so engines with immediate unique
            # constraints cannot observe two active versions in one flush.
            self.session.flush()
        record.status = "active"
        model.version = version
        model.content_hash = record.content_hash
        model.title = record.title
        model.canonical_text = record.canonical_text
        model.metadata_json = record.metadata_json
        model.acl = record.acl
        model.status = "active"
        self.session.flush()
        return self._to_contract(model)

    def deactivate(self, document_id: str, context: RequestContext | None = None) -> None:
        query = select(DocumentModel).where(DocumentModel.id == document_id)
        if context:
            query = query.where(_scope(context, DocumentModel.tenant_id))
        model = self.session.scalar(query)
        if model:
            model.status = "deleted"
            self.session.query(DocumentVersion).filter(
                DocumentVersion.document_id == document_id
            ).update({"status": "deleted"})
            self.session.query(ChunkModel).filter(ChunkModel.document_id == document_id).delete()
            self.session.flush()

    @staticmethod
    def _to_contract(model: DocumentModel) -> Document:
        return Document(
            id=model.id,
            tenant_id=model.tenant_id,
            source_type=model.source_type,
            source_uri=model.source_uri,
            filename=model.filename,
            content_hash=model.content_hash,
            version=model.version,
            title=model.title,
            canonical_text=model.canonical_text,
            metadata=model.metadata_json,
            acl=frozenset(model.acl),
            status=DocumentStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class DocumentVersionRepository:
    """Access version records only within the caller's tenant scope."""

    def __init__(self, session: Session):
        self.session = session

    def get(
        self, document_id: str, version: int, context: RequestContext
    ) -> DocumentVersion | None:
        return self.session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.version == version,
                _scope(context, DocumentVersion.tenant_id),
            )
        )

    def list(self, document_id: str, context: RequestContext) -> list[DocumentVersion]:
        return list(
            self.session.scalars(
                select(DocumentVersion)
                .where(
                    DocumentVersion.document_id == document_id,
                    _scope(context, DocumentVersion.tenant_id),
                )
                .order_by(DocumentVersion.version)
            ).all()
        )

    def create(  # noqa: PLR0913
        self,
        *,
        document_id: str,
        tenant_id: str,
        version: int,
        content_hash: str,
        title: str | None,
        canonical_text: str,
        metadata: dict[str, Any],
        acl: frozenset[str],
    ) -> DocumentVersion:
        record = DocumentVersion(
            document_id=document_id,
            tenant_id=tenant_id,
            version=version,
            content_hash=content_hash,
            title=title,
            canonical_text=canonical_text,
            metadata_json=metadata,
            acl=sorted(acl),
            status="pending",
        )
        self.session.add(record)
        self.session.flush()
        return record


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def bulk_create(
        self, chunks: list[ChunkDraft], embeddings: list[list[float]] | None = None
    ) -> list[ChunkModel]:
        models = []
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            models.append(
                ChunkModel(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    tenant_id=chunk.tenant_id,
                    version=chunk.version,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    heading_path=chunk.heading_path,
                    page_number=chunk.page_number,
                    token_count=chunk.token_count,
                    metadata_json=chunk.metadata,
                    acl=sorted(chunk.acl),
                    embedding_status="pending",
                    index_status="pending",
                    embedding=embedding,
                )
            )
        self.session.add_all(models)
        self.session.flush()
        return models

    def get_by_document(
        self, document_id: str, version: int, context: RequestContext | None = None
    ) -> list[ChunkModel]:
        query = select(ChunkModel).where(
            ChunkModel.document_id == document_id, ChunkModel.version == version
        )
        if context:
            query = query.where(_scope(context, ChunkModel.tenant_id))
        return list(self.session.scalars(query.order_by(ChunkModel.ordinal)).all())

    def delete_by_document_version(
        self, document_id: str, version: int, context: RequestContext | None = None
    ) -> int:
        query = delete(ChunkModel).where(
            ChunkModel.document_id == document_id, ChunkModel.version == version
        )
        if context:
            query = query.where(_scope(context, ChunkModel.tenant_id))
        result = self.session.execute(query)
        self.session.flush()
        return int(getattr(result, "rowcount", 0) or 0)


class IngestionJobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, job: IngestionJob) -> IngestionJob:
        model = JobModel(
            id=job.id,
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            status=job.status.value,
            current_stage=job.current_stage,
            attempts=job.attempts,
            error_code=job.error_code,
            error_message=job.error_message,
            source_key=job.source_key,
            filename=job.filename,
            mime_type=job.mime_type,
            source_uri=job.source_uri,
            metadata_json=dict(job.metadata),
            subject_id=job.subject_id,
            acl=sorted(job.acl),
            idempotency_key=job.idempotency_key,
            payload_hash=job.payload_hash,
            available_at=_utc(job.available_at),
            worker_id=job.worker_id,
            lease_token=job.lease_token,
            lease_expires_at=_utc(job.lease_expires_at) if job.lease_expires_at else None,
            created_at=_utc(job.created_at),
            updated_at=_utc(job.updated_at),
        )
        self.session.add(model)
        self.session.flush()
        return self._contract(model)

    def get(self, job_id: str, context: RequestContext | None = None) -> IngestionJob | None:
        query = select(JobModel).where(JobModel.id == job_id)
        if context:
            query = query.where(_scope(context, JobModel.tenant_id))
        model = self.session.scalar(query)
        return self._contract(model) if model else None

    def get_by_idempotency_key(
        self, tenant_id: str, idempotency_key: str
    ) -> IngestionJob | None:
        model = self.session.scalar(
            select(JobModel).where(
                JobModel.tenant_id == tenant_id,
                JobModel.idempotency_key == idempotency_key,
            )
        )
        return self._contract(model) if model else None

    def update(self, job: IngestionJob) -> IngestionJob:
        model = self.session.get(JobModel, job.id)
        if not model or model.tenant_id != job.tenant_id:
            raise LookupError("ingestion job not found")  # noqa: TRY003
        model.document_id = job.document_id
        model.status = job.status.value
        model.current_stage = job.current_stage
        model.attempts = job.attempts
        model.error_code = job.error_code
        model.error_message = job.error_message
        model.source_key = job.source_key
        model.filename = job.filename
        model.mime_type = job.mime_type
        model.source_uri = job.source_uri
        model.metadata_json = dict(job.metadata)
        model.subject_id = job.subject_id
        model.acl = sorted(job.acl)
        model.idempotency_key = job.idempotency_key
        model.payload_hash = job.payload_hash
        model.available_at = _utc(job.available_at)
        model.worker_id = job.worker_id
        model.lease_token = job.lease_token
        model.lease_expires_at = _utc(job.lease_expires_at) if job.lease_expires_at else None
        self.session.flush()
        return self._contract(model)

    def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: int
    ) -> ClaimedIngestionJob | None:
        now_utc = _utc(now)
        lease_expires_at = now_utc + timedelta(seconds=lease_seconds)
        query = (
            select(JobModel)
            .where(
                or_(
                    and_(
                        JobModel.status.in_(
                            [
                                IngestionStatus.QUEUED.value,
                                IngestionStatus.RETRY_WAIT.value,
                                IngestionStatus.RECEIVED.value,
                            ]
                        ),
                        JobModel.available_at <= now_utc,
                    ),
                    and_(
                        JobModel.status == IngestionStatus.RUNNING.value,
                        JobModel.lease_expires_at.is_not(None),
                        JobModel.lease_expires_at <= now_utc,
                    ),
                )
            )
            .order_by(JobModel.available_at.asc(), JobModel.created_at.asc(), JobModel.id.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        model = self.session.scalar(query)
        if model is None:
            return None
        lease_token = new_id()
        model.status = IngestionStatus.RUNNING.value
        model.worker_id = worker_id
        model.lease_token = lease_token
        model.lease_expires_at = lease_expires_at
        model.attempts = (model.attempts or 0) + 1
        model.updated_at = now_utc
        self.session.flush()
        return ClaimedIngestionJob(
            job=self._contract(model),
            worker_id=worker_id,
            lease_token=lease_token,
            lease_expires_at=lease_expires_at,
        )

    def heartbeat(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        now: datetime,
        lease_seconds: int,
    ) -> ClaimedIngestionJob:
        now_utc = _utc(now)
        model = self._require_owned_running(
            job_id, worker_id=worker_id, lease_token=lease_token
        )
        if model.lease_expires_at is not None and _utc(model.lease_expires_at) <= now_utc:
            raise LeaseLostError("ingestion job lease lost")  # noqa: TRY003
        lease_expires_at = now_utc + timedelta(seconds=lease_seconds)
        model.lease_expires_at = lease_expires_at
        model.updated_at = now_utc
        self.session.flush()
        return ClaimedIngestionJob(
            job=self._contract(model),
            worker_id=worker_id,
            lease_token=lease_token,
            lease_expires_at=lease_expires_at,
        )

    def update_stage(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        stage: str,
    ) -> IngestionJob:
        model = self._require_owned_running(
            job_id, worker_id=worker_id, lease_token=lease_token
        )
        model.current_stage = stage
        model.updated_at = datetime.now(UTC)
        self.session.flush()
        return self._contract(model)

    def schedule_retry(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        available_at: datetime,
        error_code: str,
        error_message: str,
    ) -> IngestionJob:
        model = self._require_owned_running(
            job_id, worker_id=worker_id, lease_token=lease_token
        )
        model.status = IngestionStatus.RETRY_WAIT.value
        model.available_at = _utc(available_at)
        model.error_code = error_code
        model.error_message = " ".join(error_message.split())[:500]
        model.worker_id = None
        model.lease_token = None
        model.lease_expires_at = None
        model.updated_at = datetime.now(UTC)
        self.session.flush()
        return self._contract(model)

    def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        document_id: str,
    ) -> IngestionJob:
        model = self._require_owned_running(
            job_id, worker_id=worker_id, lease_token=lease_token
        )
        model.status = IngestionStatus.ACTIVE.value
        model.current_stage = IngestionStatus.ACTIVE.value
        model.document_id = document_id
        model.error_code = None
        model.error_message = None
        model.worker_id = None
        model.lease_token = None
        model.lease_expires_at = None
        model.updated_at = datetime.now(UTC)
        self.session.flush()
        return self._contract(model)

    def dead_letter(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        error_code: str,
        error_message: str,
    ) -> IngestionJob:
        model = self._require_owned_running(
            job_id, worker_id=worker_id, lease_token=lease_token
        )
        model.status = IngestionStatus.DEAD_LETTER.value
        model.error_code = error_code
        model.error_message = " ".join(error_message.split())[:500]
        model.worker_id = None
        model.lease_token = None
        model.lease_expires_at = None
        model.updated_at = datetime.now(UTC)
        self.session.flush()
        return self._contract(model)

    def _require_owned_running(
        self, job_id: str, *, worker_id: str, lease_token: str
    ) -> JobModel:
        model = self.session.get(JobModel, job_id)
        if (
            model is None
            or model.status != IngestionStatus.RUNNING.value
            or model.worker_id != worker_id
            or model.lease_token != lease_token
        ):
            raise LeaseLostError("ingestion job lease lost")  # noqa: TRY003
        return model

    @staticmethod
    def _contract(model: JobModel) -> IngestionJob:
        return IngestionJob(
            id=model.id,
            tenant_id=model.tenant_id,
            document_id=model.document_id,
            status=IngestionStatus(model.status),
            current_stage=model.current_stage,
            attempts=model.attempts,
            error_code=model.error_code,
            error_message=model.error_message,
            source_key=model.source_key,
            filename=model.filename,
            mime_type=model.mime_type,
            source_uri=model.source_uri,
            metadata=dict(model.metadata_json or {}),
            subject_id=model.subject_id,
            acl=frozenset(model.acl or []),
            idempotency_key=model.idempotency_key,
            payload_hash=model.payload_hash,
            available_at=_utc(model.available_at),
            worker_id=model.worker_id,
            lease_token=model.lease_token,
            lease_expires_at=_utc(model.lease_expires_at) if model.lease_expires_at else None,
            created_at=_utc(model.created_at),
            updated_at=_utc(model.updated_at),
        )


class QueryRunRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(  # noqa: PLR0913
        self, context: RequestContext, query_text: str, status: str, response: dict[str, Any]
    ) -> QueryModel:
        model = QueryModel(
            id=new_id(),
            tenant_id=context.tenant_id,
            subject_id=context.subject_id,
            query_text=query_text,
            status=status,
            response=response,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def get(self, run_id: str, context: RequestContext) -> QueryModel | None:
        return self.session.scalar(
            select(QueryModel).where(QueryModel.id == run_id, _scope(context, QueryModel.tenant_id))
        )


class SemanticCacheEntryRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(  # noqa: PLR0913
        self,
        context: RequestContext,
        *,
        query_embedding: list[float],
        answer_text: str,
        citations: list[dict],
        model_profile: str,
        response_mode: str,
        referenced_document_versions: dict[str, int],
        expires_at: datetime | None,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
    ) -> CacheModel:
        model = CacheModel(
            id=new_id(),
            tenant_id=context.tenant_id,
            acl=sorted(context.acl),
            query_embedding=query_embedding,
            answer_text=answer_text,
            citations=citations,
            model_profile=model_profile,
            response_mode=response_mode,
            referenced_document_versions=referenced_document_versions,
            expires_at=expires_at,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def get(self, entry_id: str, context: RequestContext) -> CacheModel | None:
        model = self.session.scalar(
            select(CacheModel).where(
                CacheModel.id == entry_id, _scope(context, CacheModel.tenant_id)
            )
        )
        return model if model and frozenset(model.acl) == context.acl else None

    def invalidate_document(
        self, document_id: str, version: int, context: RequestContext | None = None
    ) -> int:
        entries = self.session.scalars(select(CacheModel)).all()
        count = 0
        for entry in entries:
            if context and entry.tenant_id != context.tenant_id:
                continue
            if entry.referenced_document_versions.get(document_id) == version:
                entry.expires_at = datetime.now(UTC)
                count += 1
        self.session.flush()
        return count


class EvaluationAnnotationRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(  # noqa: PLR0913
        self,
        context: RequestContext,
        *,
        query_run_id: str | None,
        evaluator: str,
        score: float,
        explanation: str,
        dataset_item_id: str,
        label_source: str,
    ) -> EvaluationModel:
        model = EvaluationModel(
            id=new_id(),
            tenant_id=context.tenant_id,
            query_run_id=query_run_id,
            evaluator=evaluator,
            score=score,
            explanation=explanation,
            dataset_item_id=dataset_item_id,
            label_source=label_source,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def get(self, annotation_id: str, context: RequestContext) -> EvaluationModel | None:
        return self.session.scalar(
            select(EvaluationModel).where(
                EvaluationModel.id == annotation_id, _scope(context, EvaluationModel.tenant_id)
            )
        )
