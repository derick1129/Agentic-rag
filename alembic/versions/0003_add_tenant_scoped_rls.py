"""Add tenant-scoped row-level security (RLS) policies on documents and chunks.

Enables PostgreSQL row-level security and creates isolation policies on the
``documents`` and ``chunks`` tables. The policies enforce that operations
only access rows where ``tenant_id`` matches the session's authenticated tenant
identity (configured via ``app.current_tenant_id``).
"""

from alembic import op

revision = "0003_add_tenant_scoped_rls"
down_revision = "0002_align_uuid_ids_to_string"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Enable and force RLS on documents table
    op.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE documents FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation_documents ON documents
        FOR ALL
        USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), ''))
        WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), ''))
        """
    )

    # Enable and force RLS on chunks table
    op.execute("ALTER TABLE chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE chunks FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation_chunks ON chunks
        FOR ALL
        USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), ''))
        WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), ''))
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Drop policy and disable RLS on chunks table
    op.execute("DROP POLICY IF EXISTS tenant_isolation_chunks ON chunks")
    op.execute("ALTER TABLE chunks NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE chunks DISABLE ROW LEVEL SECURITY")

    # Drop policy and disable RLS on documents table
    op.execute("DROP POLICY IF EXISTS tenant_isolation_documents ON documents")
    op.execute("ALTER TABLE documents NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE documents DISABLE ROW LEVEL SECURITY")
