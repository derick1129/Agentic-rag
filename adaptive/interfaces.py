"""Core protocols and data models for Adaptive Agentic RAG V1.

This module defines the contracts that all implementations must satisfy.
Tests are written against these interfaces before implementations exist.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, BinaryIO, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

# =============================================================================
# Enums and Literals
# =============================================================================


class IngestionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    RECEIVED = "received"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    ACTIVE = "active"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"
    REPLACED = "replaced"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    STALE = "stale"
    INVALIDATED = "invalidated"


class RouteDepth(StrEnum):
    PARAMETRIC = "parametric"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class RouteTool(StrEnum):
    PARAMETRIC = "parametric"
    VECTOR = "vector"
    SQL = "sql"
    WEB = "web"


class RouteSource(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"
    POLICY = "policy"
    ESCALATED = "escalated"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    QUALIFY = "qualify"
    REFUSE = "refuse"


class TerminationReason(StrEnum):
    ANSWER_COMPLETE = "answer_complete"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    TIMEOUT = "timeout"
    REPEATED_ACTION = "repeated_action"
    TOOL_FAILURE = "tool_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# =============================================================================
# Core Data Models
# =============================================================================


class RequestContext(BaseModel):
    """Server-derived request context. Never from user input."""

    tenant_id: str
    subject_id: str
    acl: frozenset[str] = Field(default_factory=frozenset)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {"frozen": True}

    def can_access(self, required_acl: frozenset[str]) -> bool:
        return required_acl.issubset(self.acl)


class Budget(BaseModel):
    """Per-query resource budgets."""

    max_tokens: int = 8000
    max_cost_usd: float = 0.50
    max_steps: int = 10
    max_wall_time_seconds: int = 120

    tokens_used: int = 0
    cost_usd: float = 0.0
    steps_taken: int = 0
    start_time: float = Field(default_factory=time.time)

    def charge(self, tokens: int = 0, cost_usd: float = 0.0, steps: int = 0) -> None:
        self.tokens_used += tokens
        self.cost_usd += cost_usd
        self.steps_taken += steps

    def exceeded(self) -> bool:
        wall_time = time.time() - self.start_time
        return (
            self.tokens_used >= self.max_tokens
            or self.cost_usd >= self.max_cost_usd
            or self.steps_taken >= self.max_steps
            or wall_time >= self.max_wall_time_seconds
        )

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost_usd - self.cost_usd)


class RouteDecision(BaseModel):
    """Router output - typed decision with confidence and rationale."""

    depth: RouteDepth
    tool: RouteTool
    confidence: float = Field(ge=0.0, le=1.0)
    source: RouteSource
    rationale: str | None = None

    model_config = {"frozen": True}


class ToolContext(BaseModel):
    """Context passed to tool execution."""

    request_context: RequestContext
    budget: Budget
    query: str
    route_decision: RouteDecision
    trace_id: str
    step_number: int = 0

    model_config = {"frozen": True}


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool: RouteTool
    status: ToolStatus
    text: str = ""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    error_code: str | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Chunk(BaseModel):
    """Retrieved chunk with metadata."""

    id: str
    document_id: str
    tenant_id: str
    version: int
    ordinal: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: frozenset[str] = Field(default_factory=frozenset)
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """Query for hybrid retrieval."""

    text: str
    context: RequestContext
    limit: int = 20
    document_ids: list[str] | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {"frozen": True}


class RetrievalDiagnostics(BaseModel):
    """Diagnostics from retrieval stages."""

    bm25_count: int = 0
    dense_count: int = 0
    fusion: str = "rrf"
    reranked: bool = False
    final_count: int = 0
    latency_ms: int = 0


class RetrievalResult(BaseModel):
    """Result from hybrid retrieval."""

    chunks: list[Chunk]
    diagnostics: RetrievalDiagnostics
    query: RetrievalQuery

    model_config = {"frozen": True}


class CacheLookup(BaseModel):
    """Cache lookup request."""

    query_text: str
    query_embedding: list[float]
    context: RequestContext
    route_depth: RouteDepth
    route_tool: RouteTool
    model_profile: str
    response_mode: str

    model_config = {"frozen": True}


class CacheEntry(BaseModel):
    """Cache entry for semantic caching."""

    query_embedding: list[float]
    tenant_id: str
    acl: frozenset[str]
    answer_text: str
    citations: list[dict[str, Any]]
    model_profile: str
    response_mode: str
    referenced_document_versions: dict[str, int]  # document_id -> version
    cost_usd: float
    latency_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    hit_count: int = 0

    model_config = {"frozen": True}


class CacheHit(BaseModel):
    """Cache hit result."""

    entry: CacheEntry
    similarity: float

    model_config = {"frozen": True}


class Answer(BaseModel):
    """Final answer with metadata."""

    text: str
    citations: list[dict[str, Any]]
    groundedness_status: Literal["grounded", "partial", "ungrounded", "refused"]
    unsupported_claims: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    route_decision: RouteDecision
    actual_tools_used: list[RouteTool] = Field(default_factory=list)
    cache_status: CacheStatus
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    trace_id: str

    model_config = {"frozen": True}


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    action: GuardrailAction
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    span_attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class AgentState(BaseModel):
    """State for multi-hop agent loop."""

    query: str
    context: ToolContext
    route_decision: RouteDecision
    observations: list[ToolResult] = Field(default_factory=list)
    current_step: int = 0
    termination_reason: TerminationReason | None = None
    answer: Answer | None = None
    evidence_sufficient: bool = False

    model_config = {"frozen": True}


# =============================================================================
# Protocol definitions
# =============================================================================

T = TypeVar("T")


@runtime_checkable
class Router(Protocol):
    """Router protocol - produces RouteDecision from query context."""

    async def route(self, query: str, context: RequestContext, budget: Budget) -> RouteDecision: ...


@runtime_checkable
class Tool(Protocol):
    """Tool protocol - executes a single tool call."""

    @property
    def tool_type(self) -> RouteTool: ...

    async def run(self, context: ToolContext) -> ToolResult: ...


@runtime_checkable
class HybridRetriever(Protocol):
    """Hybrid retrieval protocol - BM25 + dense with fusion."""

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...

    async def index_chunks(self, chunks: list[Chunk]) -> dict[str, Any]: ...

    async def delete_document_version(self, document_id: str, version: int) -> None: ...


@runtime_checkable
class CacheStore(Protocol):
    """Semantic cache protocol."""

    async def lookup(self, request: CacheLookup) -> CacheHit | None: ...

    async def put(self, entry: CacheEntry) -> None: ...

    async def invalidate_document(self, document_id: str, version: int) -> int: ...


@runtime_checkable
class Generator(Protocol):
    """Generation protocol - LLM text generation."""

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str: ...

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], max_tokens: int
    ) -> BaseModel: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding provider protocol."""

    async def embed(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


@runtime_checkable
class Guardrail(Protocol):
    """Guardrail protocol - checks at trust boundaries."""

    async def check(self, context: RequestContext, data: object) -> GuardrailResult: ...


@runtime_checkable
class DocumentSource(Protocol):
    """Document source protocol - for ingestion."""

    async def read(self) -> bytes: ...

    @property
    def mime_type(self) -> str: ...

    @property
    def filename(self) -> str: ...

    @property
    def size(self) -> int: ...


@runtime_checkable
class SourceStorage(Protocol):
    """Durable source payload storage protocol."""

    def store(self, job_id: str, payload: bytes) -> str: ...

    def open(self, source_key: str) -> BinaryIO: ...

    def delete(self, source_key: str) -> None: ...


@runtime_checkable
class Parser(Protocol):
    """Parser protocol - converts raw bytes to parsed document."""

    def supports(self, mime_type: str) -> bool: ...

    def parse(self, payload: bytes, mime_type: str) -> ParsedDocument: ...


class ParsedDocument(BaseModel):
    """Parsed document output."""

    title: str | None
    text: str
    pages: list[dict[str, Any]] = Field(default_factory=list)
    headings: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    """Normalized, canonical document ready for chunking."""

    content_hash: str
    title: str | None
    canonical_text: str
    metadata: dict[str, Any]
    acl: frozenset[str]
    pages: list[dict[str, Any]]
    headings: list[dict[str, Any]]
    tables: list[dict[str, Any]]


class ChunkDraft(BaseModel):
    """Draft chunk before persistence."""

    id: str
    document_id: str
    tenant_id: str
    version: int
    ordinal: int
    text: str
    heading_path: list[str]
    page_number: int | None
    token_count: int
    metadata: dict[str, Any]
    acl: frozenset[str]


class ChunkPolicy(BaseModel):
    """Chunking policy configuration."""

    max_tokens: int = 512
    overlap_tokens: int = 50
    respect_headings: bool = True
    respect_pages: bool = True

    @field_validator("overlap_tokens")
    @classmethod
    def overlap_less_than_max(cls, v: int, info: ValidationInfo) -> int:
        max_tokens = info.data.get("max_tokens", 512)
        if v >= max_tokens:
            raise PydanticCustomError("value_error", "overlap_tokens must be less than max_tokens")
        return v


class IngestionJob(BaseModel):
    """Ingestion job tracking."""

    id: str
    tenant_id: str
    document_id: str | None = None
    status: IngestionStatus
    current_stage: str
    attempts: int = 0
    error_code: str | None = None
    error_message: str | None = None
    source_key: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    subject_id: str = "system"
    acl: frozenset[str] = Field(default_factory=frozenset)
    idempotency_key: str | None = None
    payload_hash: str | None = None
    available_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    worker_id: str | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ClaimedIngestionJob(BaseModel):
    """Durable ingestion job claimed under a worker lease."""

    job: IngestionJob
    worker_id: str
    lease_token: str
    lease_expires_at: datetime

    model_config = {"frozen": True}


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused with a different payload."""



# =============================================================================
# Repository Protocols
# =============================================================================


class DocumentCreateParams(BaseModel):
    """Parameters for document creation."""

    tenant_id: str
    filename: str
    content_hash: str
    title: str | None
    canonical_text: str
    metadata: dict[str, Any]
    acl: frozenset[str]


@runtime_checkable
class DocumentRepository(Protocol):
    async def create(self, params: DocumentCreateParams) -> Document: ...

    async def get(self, document_id: str, context: RequestContext) -> Document | None: ...

    async def list_active(self, context: RequestContext, limit: int = 100) -> list[Document]: ...

    async def activate_version(self, document_id: str, version: int) -> Document: ...

    async def deactivate(self, document_id: str) -> None: ...


@runtime_checkable
class ChunkRepository(Protocol):
    async def bulk_create(self, chunks: list[ChunkDraft]) -> list[ChunkRecord]: ...

    async def get_by_document(self, document_id: str, version: int) -> list[ChunkRecord]: ...

    async def delete_by_document_version(self, document_id: str, version: int) -> int: ...


@runtime_checkable
class IngestionJobRepository(Protocol):
    async def create(self, job: IngestionJob) -> IngestionJob: ...

    async def get(self, job_id: str) -> IngestionJob | None: ...

    async def update(self, job: IngestionJob) -> IngestionJob: ...

    async def get_by_idempotency_key(
        self, tenant_id: str, idempotency_key: str
    ) -> IngestionJob | None: ...

    async def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: int
    ) -> ClaimedIngestionJob | None: ...

    async def heartbeat(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        now: datetime,
        lease_seconds: int,
    ) -> ClaimedIngestionJob: ...

    async def schedule_retry(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        available_at: datetime,
        error_code: str,
        error_message: str,
    ) -> IngestionJob: ...

    async def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        document_id: str,
    ) -> IngestionJob: ...

    async def dead_letter(  # noqa: PLR0913
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        error_code: str,
        error_message: str,
    ) -> IngestionJob: ...

class Document(BaseModel):
    id: str
    tenant_id: str
    source_type: str
    source_uri: str | None
    filename: str
    content_hash: str
    version: int
    title: str | None
    canonical_text: str
    metadata: dict[str, Any]
    acl: frozenset[str]
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class ChunkRecord(BaseModel):
    id: str
    document_id: str
    tenant_id: str
    version: int
    ordinal: int
    text: str
    heading_path: list[str]
    page_number: int | None
    token_count: int
    metadata: dict[str, Any]
    acl: frozenset[str]
    embedding_status: str
    index_status: str
    created_at: datetime


# Re-export commonly used types
__all__ = [
    # Enums
    "IngestionStatus",
    "DocumentStatus",
    "CacheStatus",
    "RouteDepth",
    "RouteTool",
    "RouteSource",
    "ToolStatus",
    "GuardrailAction",
    "TerminationReason",
    # Core Models
    "RequestContext",
    "Budget",
    "RouteDecision",
    "ToolContext",
    "ToolResult",
    "Chunk",
    "RetrievalQuery",
    "RetrievalDiagnostics",
    "RetrievalResult",
    "CacheLookup",
    "CacheEntry",
    "CacheHit",
    "Answer",
    "GuardrailResult",
    "AgentState",
    # Protocols
    "Router",
    "Tool",
    "HybridRetriever",
    "CacheStore",
    "Generator",
    "EmbeddingProvider",
    "Guardrail",
    "DocumentSource",
    "SourceStorage",
    "Parser",
    "DocumentRepository",
    "ChunkRepository",
    "IngestionJobRepository",
    # Data Classes
    "ParsedDocument",
    "CanonicalDocument",
    "ChunkDraft",
    "ChunkPolicy",
    "IngestionJob",
    "ClaimedIngestionJob",
    "Document",
    "ChunkRecord",
    "DocumentCreateParams",
]
