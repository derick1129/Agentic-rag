"""Add durable queue metadata, idempotency, and lease fields to ingestion_jobs."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_durable_ingestion_jobs"
down_revision = "0003_add_tenant_scoped_rls"
branch_labels = None
depends_on = None

_ADDED_COLUMNS = (
    "lease_expires_at",
    "lease_token",
    "worker_id",
    "available_at",
    "payload_hash",
    "idempotency_key",
    "acl",
    "subject_id",
    "metadata",
    "source_uri",
    "mime_type",
    "filename",
    "source_key",
)


def _sanitize_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    return " ".join(str(message).split())[:500]


def upgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.add_column(sa.Column("source_key", sa.String(512), nullable=True))
        batch_op.add_column(sa.Column("filename", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("mime_type", sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("source_uri", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'"))
        )
        batch_op.add_column(
            sa.Column(
                "subject_id",
                sa.String(255),
                nullable=False,
                server_default=sa.text("'system'"),
            )
        )
        batch_op.add_column(
            sa.Column("acl", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.add_column(sa.Column("idempotency_key", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("payload_hash", sa.String(128), nullable=True))
        batch_op.add_column(
            sa.Column(
                "available_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch_op.add_column(sa.Column("worker_id", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("lease_token", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("lease_expires_at", sa.DateTime(), nullable=True))
        batch_op.create_unique_constraint(
            "uq_ingestion_jobs_tenant_idempotency",
            ["tenant_id", "idempotency_key"],
        )
        batch_op.create_index(
            "ix_ingestion_jobs_queue_claim",
            ["status", "available_at", "lease_expires_at", "created_at"],
        )

    bind = op.get_bind()
    jobs = sa.table(
        "ingestion_jobs",
        sa.column("id", sa.String),
        sa.column("status", sa.String),
        sa.column("source_key", sa.String),
        sa.column("error_code", sa.String),
        sa.column("error_message", sa.Text),
    )

    rows = bind.execute(
        sa.select(
            jobs.c.id,
            jobs.c.status,
            jobs.c.source_key,
            jobs.c.error_code,
            jobs.c.error_message,
        ).where(jobs.c.status.in_(["failed", "received"]))
    ).mappings()

    for row in list(rows):
        status = row["status"]
        source_key = row["source_key"]
        if status == "failed" and source_key is None:
            bind.execute(
                sa.update(jobs)
                .where(jobs.c.id == row["id"])
                .values(
                    status="dead_letter",
                    error_code=row["error_code"] or "legacy_failed",
                    error_message=_sanitize_error_message(row["error_message"]),
                )
            )
        elif status == "received":
            if source_key is not None:
                bind.execute(sa.update(jobs).where(jobs.c.id == row["id"]).values(status="queued"))
            else:
                bind.execute(
                    sa.update(jobs)
                    .where(jobs.c.id == row["id"])
                    .values(
                        status="dead_letter",
                        error_code=row["error_code"] or "legacy_failed",
                        error_message=_sanitize_error_message(row["error_message"]),
                    )
                )


def downgrade() -> None:
    bind = op.get_bind()
    jobs = sa.table(
        "ingestion_jobs",
        sa.column("id", sa.String),
        sa.column("status", sa.String),
    )
    bind.execute(sa.update(jobs).where(jobs.c.status == "dead_letter").values(status="failed"))
    bind.execute(sa.update(jobs).where(jobs.c.status == "queued").values(status="received"))

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_index("ix_ingestion_jobs_queue_claim")
        batch_op.drop_constraint("uq_ingestion_jobs_tenant_idempotency", type_="unique")
        for col_name in _ADDED_COLUMNS:
            batch_op.drop_column(col_name)
