"""AST-level validation and scoping for read-only SQL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SQLValidationError(ValueError):
    """Raised when a generated SQL statement is unsafe or outside policy."""


def _raise_validation_error(reason: str, cause: Exception | None = None) -> NoReturn:
    if cause is not None:
        raise SQLValidationError(reason) from cause
    raise SQLValidationError(reason)


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str
    ast: exp.Expression
    max_rows: int
    params: dict[str, Any]

    def with_tenant(self, tenant_id: str) -> ValidatedSQL:
        if not tenant_id or any(char in tenant_id for char in ("\x00", "\n", "\r")):
            _raise_validation_error("tenant_id is invalid")

        scoped_table_count = 0
        for select in self.ast.find_all(exp.Select):
            # Scope each query block independently. Predicates added to an outer
            # SELECT do not protect tables read by a subquery or CTE.
            cte_names = {
                cte.alias_or_name
                for cte in select.find_all(exp.CTE)
                if cte.find_ancestor(exp.Select) is select
            }
            predicate: exp.Expression | None = None
            for table in select.find_all(exp.Table):
                if table.find_ancestor(exp.Select) is not select or table.name in cte_names:
                    continue
                qualifier = table.alias_or_name or table.name
                current = exp.EQ(
                    this=exp.column("tenant_id", table=qualifier),
                    expression=exp.Literal.string(tenant_id),
                )
                predicate = (
                    current if predicate is None else exp.And(this=predicate, expression=current)
                )
                scoped_table_count += 1

            if predicate is not None:
                existing = select.args.get("where")
                select.set(
                    "where",
                    exp.Where(
                        this=predicate
                        if existing is None
                        else exp.And(this=existing.this, expression=predicate)
                    ),
                )

        if scoped_table_count == 0:
            _raise_validation_error("query must reference a tenant-scoped table")
        return ValidatedSQL(
            sql=self.ast.sql(dialect="postgres"),
            ast=self.ast,
            max_rows=self.max_rows,
            params={"__tenant_id": tenant_id},
        )


_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Alter,
    exp.Drop,
    exp.Merge,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)


def _parse_single_select(sql: str) -> exp.Select:
    if not sql or not sql.strip():
        _raise_validation_error("SQL is empty")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except ParseError as exc:
        _raise_validation_error("SQL could not be parsed", exc)
    if len(statements) != 1:
        _raise_validation_error("exactly one SQL statement is required")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        _raise_validation_error("only SELECT statements are allowed")
    if any(statement.find(node_type) is not None for node_type in _FORBIDDEN):
        _raise_validation_error("read-only SELECT required")
    if any(isinstance(node, (exp.Into, exp.Lock)) for node in statement.walk()):
        _raise_validation_error("locking and INTO clauses are not allowed")
    return statement


def _validate_table_schemas(
    statement: exp.Select,
    allowed_schemas: set[str] | frozenset[str],
) -> None:
    allowed = set(allowed_schemas)
    for table in statement.find_all(exp.Table):
        if table.catalog or (table.db and table.db not in allowed):
            _raise_validation_error("table schema is not allowlisted")
        if not table.name:
            _raise_validation_error("table name is required")


def _apply_row_limit(statement: exp.Select, max_rows: int) -> None:
    if max_rows < 1:
        _raise_validation_error("max_rows must be positive")
    limit = statement.args.get("limit")
    if limit is not None:
        expression = limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int:
            _raise_validation_error("LIMIT must be a positive integer")
        if int(expression.this) > max_rows:
            _raise_validation_error("LIMIT exceeds configured maximum")
    else:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))


def validate_sql(
    sql: str,
    allowed_schemas: set[str] | frozenset[str] = frozenset({"public"}),
    max_rows: int = 1000,
) -> ValidatedSQL:
    """Parse one bounded SELECT and reject all write/control statements."""
    if not sql or not sql.strip():
        _raise_validation_error("SQL is empty")
    if max_rows < 1:
        _raise_validation_error("max_rows must be positive")
    statement = _parse_single_select(sql)
    _validate_table_schemas(statement, allowed_schemas)
    _apply_row_limit(statement, max_rows)
    return ValidatedSQL(
        sql=statement.sql(dialect="postgres"),
        ast=statement,
        max_rows=max_rows,
        params={},
    )


def validate_and_scope_sql(
    sql: str,
    tenant_id: str,
    allowed_schemas: set[str] | frozenset[str] = frozenset({"public"}),
    max_rows: int = 1000,
) -> ValidatedSQL:
    """Convenience boundary used by SQL tools and callers outside the tool class."""
    return validate_sql(sql, allowed_schemas, max_rows).with_tenant(tenant_id)
