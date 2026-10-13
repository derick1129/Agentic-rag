"""Execution nodes for direct and multi-hop routes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from adaptive.graph.agent_subgraph import AgentDeps, run_agent
from adaptive.interfaces import (
    AgentState,
    Answer,
    Budget,
    CacheStatus,
    RequestContext,
    RouteDecision,
    RouteTool,
    Tool,
    ToolContext,
    ToolResult,
    ToolStatus,
)
from adaptive.tools.parametric_tool import ParametricTool
from adaptive.tools.vector_tool import VectorTool

if TYPE_CHECKING:
    from adaptive.graph.build import GraphDeps

_MISSING_MULTI_HOP_ANSWER = "multi-hop graph ended without an answer"


async def execute_direct(
    query: str,
    tool: Tool,
    route: RouteDecision,
    context: RequestContext,
    budget: Budget,
    **options: list[ToolResult] | None,
) -> Answer:
    evidence = options.get("evidence")
    tool_context = ToolContext(
        request_context=context,
        budget=budget,
        query=query,
        route_decision=route,
        trace_id=context.trace_id,
    )
    result: ToolResult = await tool.run(tool_context)
    if evidence is not None:
        evidence.append(result)
    success = result.status is ToolStatus.SUCCESS
    text = result.text or (json.dumps(result.rows, default=str) if result.rows else "")
    return Answer(
        text=text if success else "I couldn't obtain a reliable answer.",
        citations=result.citations if success else [],
        groundedness_status="grounded" if success else "refused",
        refusal_reason=None if success else (result.error_code or "tool_failure"),
        route_decision=route,
        actual_tools_used=[route.tool],
        cache_status=CacheStatus.MISS,
        usage=result.usage,
        latency_ms=result.latency_ms,
        trace_id=context.trace_id,
    )


async def execute_multi_hop(
    query: str,
    route: RouteDecision,
    context: RequestContext,
    budget: Budget,
    deps: GraphDeps,
) -> Answer:
    state = AgentState(
        query=query,
        context=ToolContext(
            request_context=context,
            budget=budget,
            query=query,
            route_decision=route,
            trace_id=context.trace_id,
        ),
        route_decision=route,
    )
    result = await run_agent(
        state,
        AgentDeps(
            tools=deps.tools,
            generator=deps.generator,
            policy=deps.agent_policy,
            evidence_checker=deps.evidence_checker,
            synthesizer=deps.synthesizer,
            guardrail_runner=deps.guardrail_runner,
        ),
    )
    if result.answer is None:
        raise RuntimeError(_MISSING_MULTI_HOP_ANSWER)
    return result.answer


def default_tool(tool: RouteTool, deps: GraphDeps) -> Tool | None:
    if tool in deps.tools:
        selected: Tool = deps.tools[tool]
        return selected
    if tool is RouteTool.PARAMETRIC:
        return ParametricTool(deps.generator)
    if tool is RouteTool.VECTOR:
        return VectorTool(deps.retriever)
    return None
