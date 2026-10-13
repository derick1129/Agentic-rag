"""Direct no-retrieval parametric answer tool."""

from __future__ import annotations

import time
from typing import Protocol

from adaptive.interfaces import RouteTool, ToolContext, ToolResult, ToolStatus


class _ParametricGenerator(Protocol):
    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str: ...


class ParametricTool:
    tool_type = RouteTool.PARAMETRIC

    def __init__(self, generator: _ParametricGenerator) -> None:
        self.generator = generator

    async def run(self, context: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            text = await self.generator.generate(
                context.query, context.budget.remaining_tokens(), temperature=0
            )
            usage = getattr(self.generator, "last_usage", None) or {}
            context.budget.charge(
                tokens=int(usage.get("tokens", usage.get("total_tokens", 0))),
                cost_usd=float(usage.get("cost_usd", 0.0)),
            )
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.SUCCESS,
                text=str(text),
                usage=usage,
                latency_ms=int((time.monotonic() - started) * 1000),
                diagnostics={"retrieval": False},
            )
        except Exception:
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.ERROR,
                error_code="PARAMETRIC_FAILED",
                error_message="parametric generation failed",
            )
