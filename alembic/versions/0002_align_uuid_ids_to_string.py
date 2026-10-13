"""Align UUID id/FK columns to String to match the ORM.

The initial migration (0001) used native PostgreSQL UUID for all primary and
foreign key columns. The SQLAlchemy ORM, however, models every id and FK as a
String(36) (and generates string UUIDs), and the default request context uses a
literal ``"default"`` tenant id. On SQLite (used by tests) this is invisible,
but on PostgreSQL writes fail with ``DatatypeMismatch``.

This migration converts the id/FK columns to ``varchar(36)`` and seeds a
``default`` tenant so the string-based ORM can persist against the real DB.
"""

import sqlalchemy as sa

from alembic import op

revision = "0002_align_uuid_ids_to_string"
down_revision = "0001_control_plane"
branch_labels = None
depends_on = None

# (table, column) pairs that are native UUID in 0001 but String(36) in the ORM.
_ID_STRING_COLUMNS = [
    ("tenants", "id"),
    ("documents", "id"),
    ("documents", "tenant_id"),
    ("document_versions", "id"),
    ("document_versions", "document_id"),
    ("document_versions", "tenant_id"),
    ("chunks", "document_id"),
    ("chunks", "tenant_id"),
    ("ingestion_jobs", "id"),
    ("ingestion_jobs", "tenant_id"),
    ("ingestion_jobs", "document_id"),
    ("query_runs", "id"),
    ("query_runs", "tenant_id"),
    ("semantic_cache_entries", "id"),
    ("semantic_cache_entries", "tenant_id"),
    ("evaluation_annotations", "id"),
    ("evaluation_annotations", "tenant_id"),
    ("evaluation_annotations", "query_run_id"),
]

# Foreign key constraints that must be dropped (referencing columns that are
# being re-typed) and re-created with these names.
_FK_CONSTRAINTS = [
    ("documents", "documents_tenant_id_fkey"),
    ("document_versions", "document_versions_document_id_fkey"),
    ("document_versions", "document_versions_tenant_id_fkey"),
    ("chunks", "chunks_document_id_fkey"),
    ("chunks", "chunks_tenant_id_fkey"),
    ("ingestion_jobs", "ingestion_jobs_tenant_id_fkey"),
    ("ingestion_jobs", "ingestion_jobs_document_id_fkey"),
    ("query_runs", "query_runs_tenant_id_fkey"),
    ("semantic_cache_entries", "semantic_cache_entries_tenant_id_fkey"),
    ("evaluation_annotations", "evaluation_annotations_tenant_id_fkey"),
    ("evaluation_annotations", "evaluation_annotations_query_run_id_fkey"),
]


def upgrade() -> None:
    # Drop foreign key constraints first so column re-typing is allowed.
    for table, constraint in _FK_CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")

    # Re-type UUID columns to varchar(36). Existing UUID values are preserved
    # as their canonical string form.
    for table, column in _ID_STRING_COLUMNS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE varchar(36) USING {column}::varchar(36)"
        )

    # Re-create the foreign key constraints now that both sides are varchar.
    op.create_foreign_key(
        "documents_tenant_id_fkey",
        "documents",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "document_versions_document_id_fkey",
        "document_versions",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "document_versions_tenant_id_fkey",
        "document_versions",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "chunks_document_id_fkey",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "chunks_tenant_id_fkey",
        "chunks",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "ingestion_jobs_tenant_id_fkey",
        "ingestion_jobs",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "ingestion_jobs_document_id_fkey",
        "ingestion_jobs",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "query_runs_tenant_id_fkey",
        "query_runs",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "semantic_cache_entries_tenant_id_fkey",
        "semantic_cache_entries",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "evaluation_annotations_tenant_id_fkey",
        "evaluation_annotations",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "evaluation_annotations_query_run_id_fkey",
        "evaluation_annotations",
        "query_runs",
        ["query_run_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Seed the default tenant referenced by the default request context.
    op.bulk_insert(
        sa.table(
            "tenants",
            sa.column("id", sa.String),
            sa.column("slug", sa.String),
            sa.column("name", sa.String),
            sa.column("metadata", sa.JSON),
            sa.column("created_at", sa.DateTime),
            sa.column("updated_at", sa.DateTime),
        ),
        [
            {
                "id": "default",
                "slug": "default",
                "name": "Default Tenant",
                "metadata": {},
            }
        ],
    )


def downgrade() -> None:
    # Remove the seeded default tenant (only if it still references nothing).
    op.execute("DELETE FROM tenants WHERE id = 'default'")

    for table, constraint in _FK_CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")

    for table, column in _ID_STRING_COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE uuid USING {column}::uuid")

    op.create_foreign_key(
        "documents_tenant_id_fkey",
        "documents",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "document_versions_document_id_fkey",
        "document_versions",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "document_versions_tenant_id_fkey",
        "document_versions",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "chunks_document_id_fkey",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "chunks_tenant_id_fkey",
        "chunks",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "ingestion_jobs_tenant_id_fkey",
        "ingestion_jobs",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "ingestion_jobs_document_id_fkey",
        "ingestion_jobs",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "query_runs_tenant_id_fkey",
        "query_runs",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "semantic_cache_entries_tenant_id_fkey",
        "semantic_cache_entries",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "evaluation_annotations_tenant_id_fkey",
        "evaluation_annotations",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "evaluation_annotations_query_run_id_fkey",
        "evaluation_annotations",
        "query_runs",
        ["query_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
