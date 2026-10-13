"""Structured-output LLM route selection."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, Field

from adaptive.interfaces import (
    Budget,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
)
from adaptive.routers.contracts import QueryContext

if TYPE_CHECKING:
    from collections.abc import Awaitable

T = TypeVar("T")


class _StructuredGenerator(Protocol):
    def generate_structured(
        self, prompt: str, schema: type[BaseModel], max_tokens: int
    ) -> Awaitable[object] | object: ...


class RouteModelOutput(BaseModel):
    depth: RouteDepth
    tool: RouteTool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


async def _maybe(value: Awaitable[T] | T) -> T:
    return await value if inspect.isawaitable(value) else value


def _context(
    query_context: QueryContext | str,
    context: RequestContext | None = None,
    budget: Budget | None = None,
) -> QueryContext:
    if isinstance(query_context, QueryContext):
        return query_context
    return QueryContext(
        query=query_context,
        request_context=context or RequestContext(tenant_id="default", subject_id="anonymous"),
        budget=budget or Budget(),
    )


class LLMRouter:
    """Convert only validated structured model output into a RouteDecision."""

    def __init__(self, generator: _StructuredGenerator, *, max_tokens: int = 128) -> None:
        self.generator = generator
        self.max_tokens = max_tokens

    async def route(
        self,
        query_context: QueryContext | str,
        context: RequestContext | None = None,
        budget: Budget | None = None,
    ) -> RouteDecision:
        query_context = _context(query_context, context, budget)
        prompt = (
            "Classify the query into exactly one depth and tool. Return structured output only. "
            f"Query: {query_context.query}"
        )
        try:
            output = await _maybe(
                self.generator.generate_structured(prompt, RouteModelOutput, self.max_tokens)
            )
            if not isinstance(output, RouteModelOutput):
                output = RouteModelOutput.model_validate(output)
            return RouteDecision(source=RouteSource.LLM, **output.model_dump())
        except Exception:
            return RouteDecision(
                depth=RouteDepth.MULTI_HOP,
                tool=RouteTool.VECTOR,
                confidence=0.0,
                source=RouteSource.ESCALATED,
                rationale="invalid or unavailable structured route; escalated safely",
            )


StructuredLLMRouter = LLMRouter
