"""Shared inputs and policy for route selection."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive.interfaces import Budget, RequestContext, RouteDepth, RouteTool


class QueryContext(BaseModel):
    """Immutable query data passed to every router implementation."""

    query: str = Field(min_length=1, max_length=4000)
    request_context: RequestContext
    budget: Budget


class RoutingPolicy(BaseModel):
    """Server-controlled routing constraints; model output cannot override these."""

    forced_depth: RouteDepth | None = None
    forced_tool: RouteTool | None = None
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    escalation_depth: RouteDepth = RouteDepth.MULTI_HOP
    model_profile: str = "default"
    response_mode: str = "answer"

    def forced_decision(self) -> tuple[RouteDepth, RouteTool] | None:
        if self.forced_depth is None and self.forced_tool is None:
            return None
        depth = self.forced_depth or RouteDepth.SINGLE_HOP
        tool = self.forced_tool or RouteTool.VECTOR
        if depth is RouteDepth.PARAMETRIC:
            tool = RouteTool.PARAMETRIC
        return depth, tool
