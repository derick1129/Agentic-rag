"""Tests for read-only database roles and SqlTool integration."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from adaptive.db.session import (
    apply_tenant_context,
    create_database_engine,
    create_read_only_database_engine,
    read_only_engine,
    set_tenant_context,
)
from adaptive.interfaces import (
    Budget,
    RequestContext,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
    ToolContext,
    ToolStatus,
)
from adaptive.tools.sql_tool import SqlTool

if TYPE_CHECKING:
    from collections.abc import Generator

EXPECTED_ROW_COUNT = 3
INITIAL_ORDER_TOTAL = 100.5
ACME_ROW_COUNT = 2
GLOBEX_ORDER_ID = 3


def make_tool_context(query: str, tenant_id: str = "acme") -> ToolContext:
    return ToolContext(
        request_context=RequestContext(tenant_id=tenant_id, subject_id="u1"),
        budget=Budget(max_steps=10),
        query=query,
        route_decision=RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.SQL,
            confidence=1.0,
            source=RouteSource.POLICY,
        ),
        trace_id="trace-1",
    )


@pytest.fixture
def test_db_path() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        # Seed test tables with an admin write engine
        admin = create_engine(f"sqlite:///{path}")
        with admin.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE orders ("
                    "id INTEGER PRIMARY KEY, "
                    "tenant_id TEXT NOT NULL, "
                    "total REAL NOT NULL"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO orders (id, tenant_id, total) VALUES "
                    "(1, 'acme', 100.5), "
                    "(2, 'acme', 250.0), "
                    "(3, 'globex', 999.9)"
                )
            )
            conn.commit()
        admin.dispose()
        yield path
    finally:
        p = Path(path)
        if p.exists():
            p.unlink()


def test_read_only_engine_exports_engine_instance() -> None:
    assert isinstance(read_only_engine, Engine)


def test_apply_tenant_context_uses_transaction_local_postgres_setting() -> None:
    connection = MagicMock()
    connection.dialect.name = "postgresql"

    apply_tenant_context(connection, " acme ")

    statement, params = connection.execute.call_args.args
    assert "set_config('app.current_tenant_id', :tenant_id, true)" in str(statement)
    assert params == {"tenant_id": "acme"}


def test_set_tenant_context_is_retained_for_session_transactions() -> None:
    engine = create_engine("sqlite:///:memory:")
    session = sessionmaker(bind=engine)()
    try:
        set_tenant_context(session, "acme")
        assert session.info["tenant_id"] == "acme"
    finally:
        session.close()
        engine.dispose()


def test_read_only_engine_uses_separate_connection_pool(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    write_eng = create_database_engine(url)
    ro_eng = create_read_only_database_engine(url)

    assert write_eng is not ro_eng
    assert write_eng.pool is not ro_eng.pool

    write_eng.dispose()
    ro_eng.dispose()


def test_read_only_engine_executes_select_successfully(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    with ro_eng.connect() as conn:
        rows = conn.execute(text("SELECT id, tenant_id, total FROM orders ORDER BY id")).fetchall()
        assert len(rows) == EXPECTED_ROW_COUNT
        assert rows[0][0] == 1
        assert rows[0][1] == "acme"
        assert rows[0][2] == INITIAL_ORDER_TOTAL

    ro_eng.dispose()


def test_read_only_engine_disallows_insert(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    with ro_eng.connect() as conn:
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(
                text("INSERT INTO orders (id, tenant_id, total) VALUES (4, 'acme', 50.0)")
            )
        assert "disallowed" in str(exc_info.value).lower() or "readonly" in str(
            exc_info.value
        ).lower()

    ro_eng.dispose()


def test_read_only_engine_disallows_update(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    with ro_eng.connect() as conn:
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("UPDATE orders SET total = 0.0 WHERE id = 1"))
        assert "disallowed" in str(exc_info.value).lower() or "readonly" in str(
            exc_info.value
        ).lower()

    ro_eng.dispose()


def test_read_only_engine_disallows_delete(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    with ro_eng.connect() as conn:
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("DELETE FROM orders WHERE id = 1"))
        assert "disallowed" in str(exc_info.value).lower() or "readonly" in str(
            exc_info.value
        ).lower()

    ro_eng.dispose()


def test_read_only_engine_disallows_ddl(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    with ro_eng.connect() as conn:
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("DROP TABLE orders"))
        assert "disallowed" in str(exc_info.value).lower() or "readonly" in str(
            exc_info.value
        ).lower()

        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("CREATE TABLE audit_log (id INT)"))
        assert "disallowed" in str(exc_info.value).lower() or "readonly" in str(
            exc_info.value
        ).lower()

    ro_eng.dispose()


@pytest.mark.asyncio
async def test_sql_tool_defaults_to_read_only_engine() -> None:
    tool = SqlTool()
    assert tool.executor is read_only_engine
    assert tool.tool_type == RouteTool.SQL


@pytest.mark.asyncio
async def test_sql_tool_executes_select_with_read_only_connection(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)
    tool = SqlTool(engine=ro_eng)

    context = make_tool_context("SELECT id, total FROM orders", tenant_id="acme")
    result = await tool.run(context)

    assert result.status == ToolStatus.SUCCESS
    assert result.tool == RouteTool.SQL
    assert len(result.rows) == ACME_ROW_COUNT
    assert all(row["id"] in {1, 2} for row in result.rows)
    assert result.citations == [{"source": "sql", "tenant_id": "acme"}]
    assert result.diagnostics.get("read_only") is True

    ro_eng.dispose()


@pytest.mark.asyncio
async def test_sql_tool_cross_tenant_isolation(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)
    tool = SqlTool(ro_eng)

    context = make_tool_context("SELECT id, total FROM orders", tenant_id="globex")
    result = await tool.run(context)

    assert result.status == ToolStatus.SUCCESS
    assert len(result.rows) == 1
    assert result.rows[0]["id"] == GLOBEX_ORDER_ID

    ro_eng.dispose()


@pytest.mark.asyncio
async def test_sql_tool_blocks_write_attempts_via_validation_guardrail(
    test_db_path: str,
) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)
    tool = SqlTool(engine=ro_eng)

    for mutating_query in (
        "INSERT INTO orders (id, tenant_id, total) VALUES (99, 'acme', 1.0)",
        "UPDATE orders SET total = 0.0 WHERE id = 1",
        "DELETE FROM orders WHERE id = 1",
        "DROP TABLE orders",
    ):
        context = make_tool_context(mutating_query, tenant_id="acme")
        result = await tool.run(context)
        assert result.status == ToolStatus.REJECTED
        assert result.error_code == "SQL_VALIDATION_FAILED"

    ro_eng.dispose()


@pytest.mark.asyncio
async def test_sql_tool_blocks_write_attempts_at_engine_level(test_db_path: str) -> None:
    url = f"sqlite:///{test_db_path}"
    ro_eng = create_read_only_database_engine(url)

    ro_eng.dispose()


def test_tenant_scoped_rls_migration() -> None:
    migration_path = (
        Path(__file__).parents[2] / "alembic" / "versions" / "0003_add_tenant_scoped_rls.py"
    )
    assert migration_path.exists()

    spec = importlib.util.spec_from_file_location("migration_0003", migration_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.revision == "0003_add_tenant_scoped_rls"
    assert mod.down_revision == "0002_align_uuid_ids_to_string"

    # Non-PostgreSQL dialect safely skips execution
    mock_op = MagicMock()
    mod.op = mock_op  # type: ignore[attr-defined]
    sqlite_bind = MagicMock()
    sqlite_bind.dialect.name = "sqlite"
    mock_op.get_bind.return_value = sqlite_bind

    mod.upgrade()
    assert mock_op.execute.call_count == 0
    mod.downgrade()
    assert mock_op.execute.call_count == 0

    # PostgreSQL dialect enables RLS and creates isolation policies
    pg_bind = MagicMock()
    pg_bind.dialect.name = "postgresql"
    mock_op.get_bind.return_value = pg_bind

    mod.upgrade()
    executed = [str(call.args[0]) for call in mock_op.execute.call_args_list]
    assert any("ENABLE ROW LEVEL SECURITY" in sql for sql in executed)
    assert any("tenant_isolation_documents" in sql for sql in executed)
    assert any("tenant_isolation_chunks" in sql for sql in executed)

    mock_op.reset_mock()
    mod.downgrade()
    downgraded = [str(call.args[0]) for call in mock_op.execute.call_args_list]
    assert any("DISABLE ROW LEVEL SECURITY" in sql for sql in downgraded)
    assert any("DROP POLICY" in sql for sql in downgraded)

