"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    """Request model for document ingestion."""

    filename: str = Field(..., description="Name of the file being ingested")
    content: str = Field(..., description="Content of the document to ingest")
    source_type: str = Field(default="upload", description="Source type of the document")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""

    job_id: str = Field(..., description="Unique identifier for the ingestion job")
    status: str = Field(..., description="Current status of the ingestion job")
    message: str = Field(..., description="Human-readable status message")


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""

    query: str = Field(..., description="The query text to search for")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results to return")
    document_ids: list[str] | None = Field(
        default=None, description="Optional list of document IDs to restrict search to"
    )
    generate: bool = Field(
        default=True, description="Whether to generate an answer from the retrieved context"
    )


class QueryResponse(BaseModel):
    """Response model for query results."""

    results: list[dict[str, Any]] = Field(..., description="List of retrieved chunks with scores")
    trace_id: str = Field(..., description="Trace ID for observability")
    query: str = Field(..., description="The original query text")
    answer: str | None = Field(default=None, description="Generated answer text")
    citations: list[dict[str, Any]] | None = Field(
        default=None, description="Citations used in answer"
    )
    cache_status: str | None = Field(
        default="bypass", description="Cache status: hit, miss, or bypass"
    )
    groundedness_status: str | None = Field(default=None, description="Post-generation NLI status")
    diagnostics: dict[str, Any] | None = Field(
        default=None, description="Retrieval and guardrail diagnostics"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status of the service")
    version: str = Field(default="1.0.0", description="Service version")
    timestamp: str = Field(..., description="Current timestamp in ISO format")
