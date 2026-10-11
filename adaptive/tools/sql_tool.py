"""Read-only SQL tool with server-enforced tenant scope."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, text

from adaptive.db.session import read_only_engine
from adaptive.interfaces import RouteTool, ToolContext, ToolResult, ToolStatus
from adaptive.tools.sql_validate import SQLValidationError, validate_sql

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class SqlTool:
    tool_type = RouteTool.SQL

    def __init__(  # noqa: PLR0913
        self,
        executor: Any = None,  # noqa: ANN401
        *,
        engine: Engine | None = None,
        allowed_schemas: set[str] | frozenset[str] = frozenset({"public"}),
        max_rows: int = 1000,
        statement_timeout_ms: int = 5000,
        query_builder: Callable[[str], str | Awaitable[str]] | None = None,
    ) -> None:
        if executor is not None:
            self.executor = executor
        elif engine is not None:
            self.executor = engine
        else:
            self.executor = read_only_engine
        self.allowed_schemas = allowed_schemas
        self.max_rows = max_rows
        self.statement_timeout_ms = statement_timeout_ms
        self.query_builder = query_builder

    async def _maybe(self, value: Any) -> Any:  # noqa: ANN401
        return await value if inspect.isawaitable(value) else value

    async def run(self, context: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            sql = (
                context.query
                if self.query_builder is None
                else await self._maybe(self.query_builder(context.query))
            )
            validated = validate_sql(sql, self.allowed_schemas, self.max_rows).with_tenant(
                context.request_context.tenant_id
            )
            if hasattr(self.executor, "set_read_only"):
                await self._maybe(self.executor.set_read_only(True))
            if hasattr(self.executor, "set_statement_timeout"):
                await self._maybe(self.executor.set_statement_timeout(self.statement_timeout_ms))
            if isinstance(self.executor, Engine):
                with self.executor.connect() as conn:
                    if conn.dialect.name == "postgresql":
                        conn.execute(
                            text(f"SET LOCAL statement_timeout = {int(self.statement_timeout_ms)}")
                        )
                    res = conn.execute(text(validated.sql), validated.params)
                    rows = [dict(m) for m in res.mappings().all()]
            elif hasattr(self.executor, "execute"):
                rows = await self._maybe(
                    self.executor.execute(
                        validated.sql,
                        validated.params,
                        timeout_ms=self.statement_timeout_ms,
                        read_only=True,
                    )
                )
            else:
                rows = await self._maybe(
                    self.executor(validated.sql, validated.params, self.statement_timeout_ms)
                )
            rows = list(rows or [])
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.SUCCESS,
                rows=rows[: self.max_rows],
                citations=[{"source": "sql", "tenant_id": context.request_context.tenant_id}],
                usage={"row_count": len(rows[: self.max_rows])},
                latency_ms=int((time.monotonic() - started) * 1000),
                diagnostics={"sql": validated.sql, "read_only": True},
            )
        except SQLValidationError as exc:
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.REJECTED,
                error_code="SQL_VALIDATION_FAILED",
                error_message=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except TimeoutError:
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.TIMEOUT,
                error_code="SQL_TIMEOUT",
                error_message="SQL statement timed out",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception:
            return ToolResult(
                tool=self.tool_type,
                status=ToolStatus.ERROR,
                error_code="SQL_EXECUTION_FAILED",
                error_message="SQL execution failed",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
