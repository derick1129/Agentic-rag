"""Direct vector/hybrid retrieval tool."""

from __future__ import annotations

import time
from typing import Protocol

from adaptive.interfaces import (
    RetrievalQuery,
    RetrievalResult,
    RouteTool,
    ToolContext,
    ToolResult,
    ToolStatus,
)


class _VectorRetriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...


class VectorTool:
    tool_type = RouteTool.VECTOR

    def __init__(self, retriever: _VectorRetriever) -> None:
        self.retriever = retriever

    async def run(self, context: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            result = await self.retriever.retrieve(
                RetrievalQuery(
                    text=context.query,
                    context=context.request_context,
                    trace_id=context.trace_id,
                )
            )
            citations = [
                {"chunk_id": chunk.id, "document_id": chunk.document_id, "version": chunk.version}
                for chunk in result.chunks
            ]
            text = "\n\n".join(chunk.text for chunk in result.chunks)
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.SUCCESS,
                text=text,
                citations=citations,
                latency_ms=int((time.monotonic() - started) * 1000),
                diagnostics=result.diagnostics.model_dump(),
            )
        except Exception:
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.ERROR,
                error_code="VECTOR_FAILED",
                error_message="vector retrieval failed",
            )
