"""Models used by the local document ingestion boundary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from adaptive.interfaces import CanonicalDocument, ClaimedIngestionJob, ParsedDocument


class SourcePayload(BaseModel):
    """Bytes read from a validated local source."""

    data: bytes
    filename: str
    mime_type: str
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["CanonicalDocument", "ClaimedIngestionJob", "ParsedDocument", "SourcePayload"]
