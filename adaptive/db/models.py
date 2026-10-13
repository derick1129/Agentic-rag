"""SQLAlchemy control-plane models.

The ORM deliberately uses portable JSON for ACL and metadata fields so the
repository can be tested with SQLite. The Alembic migration uses PostgreSQL
UUID and vector types for production deployments.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect
    from sqlalchemy.types import TypeEngine


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class EmbeddingVector(TypeDecorator[list[float]]):
    """Use pgvector in PostgreSQL while keeping SQLite tests portable."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[list[float]]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(1536))
        return dialect.type_descriptor(JSON())


class Tenant(TimestampMixin, Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(40), default="upload", nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    canonical_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False, index=True)

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(TimestampMixin, Base):
    __tablename__ = "document_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    canonical_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    document: Mapped[Document] = relationship(back_populates="versions")
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_version_number"),
        Index(
            "ix_document_versions_active",
            "document_id",
            unique=True,
            postgresql_where=(status == "active"),
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    index_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    __table_args__ = (
        Index("ix_chunks_tenant_document_version", "tenant_id", "document_id", "version"),
    )


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    source_key: Mapped[str | None] = mapped_column(String(512))
    filename: Mapped[str | None] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    source_uri: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )
    subject_id: Mapped[str] = mapped_column(String(255), default="system", nullable=False)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    payload_hash: Mapped[str | None] = mapped_column(String(128))
    available_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(255))
    lease_token: Mapped[str | None] = mapped_column(String(255))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_ingestion_jobs_tenant_idempotency"
        ),
        Index(
            "ix_ingestion_jobs_queue_claim",
            "status",
            "available_at",
            "lease_expires_at",
            "created_at",
        ),
    )


class QueryRun(TimestampMixin, Base):
    __tablename__ = "query_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    response: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class SemanticCacheEntry(TimestampMixin, Base):
    __tablename__ = "semantic_cache_entries"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    acl: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    query_embedding: Mapped[list[float]] = mapped_column(EmbeddingVector(), nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    model_profile: Mapped[str] = mapped_column(String(255), nullable=False)
    response_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    referenced_document_versions: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    __table_args__ = (
        Index("ix_cache_tenant_expiry", "tenant_id", "expires_at"),
        Index(
            "ix_cache_referenced_document_versions",
            "referenced_document_versions",
            postgresql_using="gin",
        ),
    )


class EvaluationAnnotation(TimestampMixin, Base):
    __tablename__ = "evaluation_annotations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    query_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("query_runs.id", ondelete="CASCADE")
    )
    evaluator: Mapped[str] = mapped_column(String(255), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    dataset_item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    label_source: Mapped[str] = mapped_column(String(32), nullable=False)
