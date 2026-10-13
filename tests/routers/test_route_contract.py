import pytest
from pydantic import BaseModel

from adaptive.interfaces import (
    Budget,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
)
from adaptive.routers.contracts import QueryContext, RoutingPolicy
from adaptive.routers.embedding_router import EmbeddingRouter
from adaptive.routers.escalation import apply_escalation
from adaptive.routers.llm_router import LLMRouter, RouteModelOutput


def context(query: str = "question") -> QueryContext:
    return QueryContext(
        query=query,
        request_context=RequestContext(tenant_id="acme", subject_id="u1"),
        budget=Budget(max_steps=4, max_tokens=1000, max_cost_usd=10),
    )


class FixedGenerator:
    def __init__(self, value: object) -> None:
        self.value = value

    async def generate_structured(self, prompt: str, schema: type[BaseModel], max_tokens: int):
        if isinstance(self.value, BaseModel):
            return self.value
        return self.value


class FixedEmbedder:
    async def embed(self, texts: list[str], *, input_type: str | None = None) -> list[list[float]]:
        return await _embedding(texts, input_type=input_type)


@pytest.mark.asyncio
async def test_llm_router_returns_typed_decision():
    generator = FixedGenerator(
        RouteModelOutput(depth="single_hop", tool="sql", confidence=0.91, rationale="structured")
    )
    decision = await LLMRouter(generator).route(context())
    assert decision == RouteDecision(
        depth=RouteDepth.SINGLE_HOP,
        tool=RouteTool.SQL,
        confidence=0.91,
        source=RouteSource.LLM,
        rationale="structured",
    )


@pytest.mark.asyncio
async def test_invalid_llm_output_escalates_to_safe_multi_hop():
    decision = await LLMRouter(FixedGenerator(object())).route(context())
    assert decision.depth is RouteDepth.MULTI_HOP
    assert decision.source is RouteSource.ESCALATED
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_embedding_router_uses_nearest_seed():
    router = EmbeddingRouter(
        embedder=FixedEmbedder(),
        seeds={
            "math": (RouteDepth.PARAMETRIC, RouteTool.PARAMETRIC, [1.0, 0.0]),
            "docs": (RouteDepth.SINGLE_HOP, RouteTool.VECTOR, [0.0, 1.0]),
        },
    )
    decision = await router.route(context("docs"))
    assert decision.depth is RouteDepth.SINGLE_HOP
    assert decision.tool is RouteTool.VECTOR
    assert decision.source is RouteSource.EMBEDDING
    assert decision.confidence == 1.0


async def _embedding(texts: list[str], *, input_type: str | None = None) -> list[list[float]]:
    assert input_type in (None, "query")
    return [[0.0, 1.0] if text == "docs" else [1.0, 0.0] for text in texts]


def test_low_confidence_escalation_preserves_scope_and_budget():
    original = RouteDecision(
        depth=RouteDepth.SINGLE_HOP,
        tool=RouteTool.VECTOR,
        confidence=0.2,
        source=RouteSource.LLM,
    )
    max_steps = 3
    budget = Budget(max_steps=max_steps, max_tokens=10, max_cost_usd=1)
    escalated = apply_escalation(original, RoutingPolicy(min_confidence=0.7), budget)
    assert escalated.depth is RouteDepth.MULTI_HOP
    assert escalated.tool is RouteTool.VECTOR
    assert escalated.source is RouteSource.ESCALATED
    assert budget.max_steps == max_steps


def test_policy_forced_route_is_typed():
    policy = RoutingPolicy(forced_tool=RouteTool.PARAMETRIC, forced_depth=RouteDepth.PARAMETRIC)
    assert policy.forced_depth is RouteDepth.PARAMETRIC
