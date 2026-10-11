"""SQLAlchemy engine and session helpers for the control plane."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

from adaptive.config import get_settings

logger = logging.getLogger(__name__)


def apply_tenant_context(connection: object, tenant_id: str) -> None:
    """Set the transaction-local tenant used by PostgreSQL RLS policies."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if getattr(getattr(connection, "dialect", None), "name", None) != "postgresql":
        return
    connection.execute(  # type: ignore[attr-defined]
        text("SELECT set_config('app.current_tenant_id', :tenant_id, true)"),
        {"tenant_id": tenant_id.strip()},
    )


@event.listens_for(Session, "after_begin")
def _apply_session_tenant_context(session: Session, transaction: object, connection: object) -> None:
    """Restore the session's tenant after each transaction begins."""
    tenant_id = session.info.get("tenant_id")
    if tenant_id:
        apply_tenant_context(connection, tenant_id)


def set_tenant_context(session: Session, tenant_id: str) -> None:
    """Bind a SQLAlchemy session to a tenant for its current and later transactions."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    session.info["tenant_id"] = tenant_id.strip()
    if session.in_transaction():
        apply_tenant_context(session.connection(), tenant_id)

_DISALLOWED_PREFIXES = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    }
)


def _resolve_database_url(database_url: str | None = None, *, read_only: bool = False) -> str:
    if database_url:
        return database_url
    if read_only:
        ro_url = os.getenv("DATABASE_READ_ONLY_URL")
        if ro_url:
            return ro_url
    try:
        settings = get_settings()
    except Exception:
        return "sqlite+pysqlite:///:memory:"
    else:
        ro_setting = getattr(settings, "database_read_only_url", None)
        if read_only and isinstance(ro_setting, str) and ro_setting:
            return ro_setting
        return str(settings.database_url)


def _enforce_read_only_connect(  # noqa: ANN401
    dialect_name: str,
    dbapi_connection: object,
    connection_record: object,
) -> None:
    cursor = getattr(dbapi_connection, "cursor", None)
    if cursor is None:
        return
    cur = cursor()
    try:
        if dialect_name == "sqlite":
            cur.execute("PRAGMA query_only = ON")
        elif dialect_name == "postgresql":
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    except Exception:
        logger.debug("Could not set read-only session mode", exc_info=True)
    finally:
        cur.close()


def _check_disallowed_dml_ddl(statement: str, parameters: object) -> None:
    raw = statement.strip()
    while raw.startswith(("--", "/*")):
        if raw.startswith("--"):
            raw = raw.split("\n", 1)[1].strip() if "\n" in raw else ""
        elif raw.startswith("/*"):
            raw = raw.split("*/", 1)[1].strip() if "*/" in raw else ""
    first_word = raw.split(None, 1)[0].upper() if raw else ""
    if first_word in _DISALLOWED_PREFIXES:
        raise OperationalError(
            statement,
            parameters,  # type: ignore[arg-type]
            Exception(f"Write operation '{first_word}' is disallowed on read-only database engine"),
        )


def _on_before_cursor_execute(  # noqa: PLR0913, PLR0917, ANN401
    conn: object,
    cursor: object,
    statement: str,
    parameters: object,
    context: object,
    executemany: bool,
) -> None:
    _check_disallowed_dml_ddl(statement, parameters)


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create an engine without opening a connection until it is used."""
    url = _resolve_database_url(database_url, read_only=False)
    options: dict[str, object] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


def create_read_only_database_engine(database_url: str | None = None) -> Engine:
    """Create a read-only database engine with an independent pool and write protections."""
    url = _resolve_database_url(database_url, read_only=True)
    options: dict[str, object] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}

    ro_engine = create_engine(url, **options)

    @event.listens_for(ro_engine, "connect")
    def _on_connect(dbapi_connection: object, connection_record: object) -> None:
        _enforce_read_only_connect(ro_engine.dialect.name, dbapi_connection, connection_record)

    event.listen(ro_engine, "before_cursor_execute", _on_before_cursor_execute)

    return ro_engine


engine = create_database_engine()
read_only_engine = create_read_only_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """Yield a request-scoped session and always close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


