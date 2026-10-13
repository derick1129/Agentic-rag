"""Composable policy, primary-router, and confidence-escalation router."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from adaptive.routers.contracts import QueryContext, RoutingPolicy
from adaptive.routers.escalation import apply_escalation
from adaptive.routers.policy_router import PolicyRouter

if TYPE_CHECKING:
    from adaptive.interfaces import RouteDecision


class _PrimaryRouter(Protocol):
    async def route(self, query_context: QueryContext) -> RouteDecision: ...


class AdaptiveRouter:
    def __init__(self, primary: _PrimaryRouter, policy: RoutingPolicy | None = None) -> None:
        self.primary = primary
        self.policy = policy or RoutingPolicy()
        self.policy_router = PolicyRouter(self.policy)

    async def route(self, query_context: QueryContext) -> RouteDecision:
        forced = await self.policy_router.route(query_context)
        if forced is not None:
            return forced
        return apply_escalation(
            await self.primary.route(query_context), self.policy, query_context.budget
        )
