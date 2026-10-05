-- ============================================================================
-- Enterprise AI Knowledge & Decision Assistant — Database Initialization
-- PostgreSQL 16 + pgvector
-- ============================================================================

-- Enable pgvector extension (ships in the docker image)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Table: enterprise_embeddings ─────────────────────────────────────────────
-- Stores chunked document content with dense vector embeddings.
-- tenant_id enforces row-level multi-tenancy isolation.
-- content_tsv is a tsvector column for BM25-equivalent full-text search,
--   enabling hybrid retrieval (dense + sparse) via Reciprocal Rank Fusion.

CREATE TABLE IF NOT EXISTS enterprise_embeddings (
    chunk_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       VARCHAR(255)    NOT NULL,
    tenant_id    VARCHAR(64)     NOT NULL,
    content      TEXT            NOT NULL,
    metadata     JSONB           DEFAULT '{}'::jsonb,
    embedding    vector(1536),                          -- text-embedding-3-large dim
    content_tsv  TSVECTOR        GENERATED ALWAYS AS (
                     to_tsvector('english', content)
                 ) STORED,                              -- BM25 sparse retrieval column
    created_at   TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast approximate nearest-neighbor search (cosine distance)
-- m=16, ef_construction=64 is a solid balance of recall vs build speed.
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON enterprise_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- B-tree index for tenant isolation filter (always applied in queries)
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant
    ON enterprise_embeddings (tenant_id);

-- GIN index for full-text search (used in BM25/RRF hybrid retrieval)
CREATE INDEX IF NOT EXISTS idx_embeddings_fts
    ON enterprise_embeddings
    USING gin (content_tsv);

-- Composite index for combined tenant + document lookups
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant_doc
    ON enterprise_embeddings (tenant_id, doc_id);


-- ── Table: audit_logs ────────────────────────────────────────────────────────
-- Immutable append-only log of every agent action.
-- Satisfies NFR-2 (Observability & Auditability) from requirements.md.

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  VARCHAR(128)    NOT NULL,
    tenant_id        VARCHAR(64)     NOT NULL,
    user_id          VARCHAR(128),
    event_type       VARCHAR(64)     NOT NULL,    -- e.g. VECTOR_SEARCH, SQL_QUERY, VALIDATION
    agent_name       VARCHAR(64),
    payload          JSONB           DEFAULT '{}'::jsonb,
    latency_ms       INTEGER,
    faithfulness_score NUMERIC(4,3),
    created_at       TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
);

-- Append-only enforcement: prevent UPDATE/DELETE on audit_logs via rule
CREATE OR REPLACE RULE audit_logs_no_update AS
    ON UPDATE TO audit_logs DO INSTEAD NOTHING;

CREATE OR REPLACE RULE audit_logs_no_delete AS
    ON DELETE TO audit_logs DO INSTEAD NOTHING;

CREATE INDEX IF NOT EXISTS idx_audit_conversation
    ON audit_logs (conversation_id);

CREATE INDEX IF NOT EXISTS idx_audit_tenant
    ON audit_logs (tenant_id, created_at DESC);


-- ── Table: analytics_data ────────────────────────────────────────────────────
-- Normalized enterprise tabular data ingested via PySpark ETL.
-- Used by SQL Analytics Agent for Text-to-SQL queries.

CREATE TABLE IF NOT EXISTS analytics_data (
    record_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    VARCHAR(64)     NOT NULL,
    region       VARCHAR(64),
    department   VARCHAR(128),
    metric_name  VARCHAR(128),
    metric_value NUMERIC(18, 4),
    period       VARCHAR(20),                    -- e.g. "2026-Q3"
    tags         JSONB           DEFAULT '{}'::jsonb,
    source_file  VARCHAR(255),
    created_at   TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analytics_tenant
    ON analytics_data (tenant_id);

CREATE INDEX IF NOT EXISTS idx_analytics_period
    ON analytics_data (tenant_id, period);
