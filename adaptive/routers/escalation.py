"""Confidence and policy enforcement for route decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive.interfaces import RouteDecision, RouteDepth, RouteSource, RouteTool

if TYPE_CHECKING:
    from adaptive.interfaces import Budget
    from adaptive.routers.contracts import RoutingPolicy


def normalize_decision(decision: RouteDecision) -> RouteDecision:
    """Enforce the invariant that parametric depth never retrieves."""
    if decision.depth is RouteDepth.PARAMETRIC and decision.tool is not RouteTool.PARAMETRIC:
        return decision.model_copy(update={"tool": RouteTool.PARAMETRIC})
    return decision


def apply_escalation(
    decision: RouteDecision, policy: RoutingPolicy, budget: Budget
) -> RouteDecision:
    """Escalate uncertain routes without mutating request scope or budgets."""
    decision = normalize_decision(decision)
    if decision.confidence >= policy.min_confidence:
        return decision
    if budget.exceeded() or budget.remaining_tokens() <= 0:
        return decision
    return decision.model_copy(
        update={
            "depth": policy.escalation_depth,
            "source": RouteSource.ESCALATED,
            "rationale": (
                f"confidence {decision.confidence:.3f} below policy threshold "
                f"{policy.min_confidence:.3f}; escalated"
            ),
        }
    )
