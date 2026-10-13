"""Embedding nearest-seed route selection."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Protocol, TypeVar

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


class _Embedder(Protocol):
    def embed(
        self, texts: list[str], *, input_type: str | None = None
    ) -> Awaitable[list[list[float]]] | list[list[float]]: ...


async def _maybe(value: Awaitable[T] | T) -> T:
    return await value if inspect.isawaitable(value) else value


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class EmbeddingRouter:
    """Choose the route represented by the closest labeled embedding seed."""

    def __init__(
        self,
        embedder: _Embedder,
        seeds: dict[str, tuple[RouteDepth, RouteTool, list[float]]],
    ) -> None:
        self.embedder = embedder
        self.seeds = seeds

    async def route(
        self,
        query_context: QueryContext | str,
        context: RequestContext | None = None,
        budget: Budget | None = None,
    ) -> RouteDecision:
        if not isinstance(query_context, QueryContext):
            query_context = QueryContext(
                query=query_context,
                request_context=context
                or RequestContext(tenant_id="default", subject_id="anonymous"),
                budget=budget or Budget(),
            )
        vectors = await _maybe(self.embedder.embed([query_context.query], input_type="query"))
        query_vector = vectors[0]
        label, (depth, tool, seed) = max(
            self.seeds.items(), key=lambda item: _cosine(query_vector, item[1][2])
        )
        confidence = max(0.0, min(1.0, (_cosine(query_vector, seed) + 1.0) / 2.0))
        return RouteDecision(
            depth=depth,
            tool=tool,
            confidence=confidence,
            source=RouteSource.EMBEDDING,
            rationale=f"nearest route seed: {label}",
        )


EmbeddingSeedRouter = EmbeddingRouter
