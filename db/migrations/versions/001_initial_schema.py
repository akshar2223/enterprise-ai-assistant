"""Initial schema: enterprise_embeddings, audit_logs, analytics_data.

Revision ID: 001
Revises:     (none — first migration)
Create Date: 2026-10-05

This migration reflects the schema defined in db/init.sql.
Use alembic for ALL future schema changes — never edit init.sql for changes.

Run: alembic upgrade head
"""
from __future__ import annotations

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── pgvector extension ────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # ── enterprise_embeddings ─────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS enterprise_embeddings (
            chunk_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id       VARCHAR(255)    NOT NULL,
            tenant_id    VARCHAR(64)     NOT NULL,
            content      TEXT            NOT NULL,
            metadata     JSONB           DEFAULT '{}'::jsonb,
            embedding    vector(1536),
            content_tsv  TSVECTOR        GENERATED ALWAYS AS (
                             to_tsvector('english', content)
                         ) STORED,
            created_at   TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
            ON enterprise_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON enterprise_embeddings (tenant_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_fts ON enterprise_embeddings USING gin (content_tsv)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_tenant_doc ON enterprise_embeddings (tenant_id, doc_id)")

    # ── audit_logs ────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id    VARCHAR(128)    NOT NULL,
            tenant_id          VARCHAR(64)     NOT NULL,
            user_id            VARCHAR(128),
            event_type         VARCHAR(64)     NOT NULL,
            agent_name         VARCHAR(64),
            payload            JSONB           DEFAULT '{}'::jsonb,
            latency_ms         INTEGER,
            faithfulness_score NUMERIC(4,3),
            created_at         TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("CREATE OR REPLACE RULE audit_logs_no_update AS ON UPDATE TO audit_logs DO INSTEAD NOTHING")
    op.execute("CREATE OR REPLACE RULE audit_logs_no_delete AS ON DELETE TO audit_logs DO INSTEAD NOTHING")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_conversation ON audit_logs (conversation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_logs (tenant_id, created_at DESC)")

    # ── analytics_data ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS analytics_data (
            record_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id    VARCHAR(64)     NOT NULL,
            region       VARCHAR(64),
            department   VARCHAR(128),
            metric_name  VARCHAR(128),
            metric_value NUMERIC(18, 4),
            period       VARCHAR(20),
            tags         JSONB           DEFAULT '{}'::jsonb,
            source_file  VARCHAR(255),
            created_at   TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_analytics_tenant ON analytics_data (tenant_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_analytics_period ON analytics_data (tenant_id, period)")


def downgrade() -> None:
    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS analytics_data CASCADE")
    op.execute("DROP TABLE IF EXISTS audit_logs CASCADE")
    op.execute("DROP TABLE IF EXISTS enterprise_embeddings CASCADE")
    # Note: extensions are intentionally not dropped (may be used by other schemas)
