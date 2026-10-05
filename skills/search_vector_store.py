"""
Hybrid Retrieval Skill: pgvector Dense + PostgreSQL FTS Sparse + RRF Fusion.

Architecture decision (documented in requirements.md trade-off matrix):
  Dense:   pgvector cosine similarity — captures semantic meaning
  Sparse:  PostgreSQL tsvector / ts_rank_cd — captures exact keyword matches
  Fusion:  Reciprocal Rank Fusion (k=60) — combines both rank lists without
           needing to normalize score distributions from different spaces

Why RRF over weighted score averaging?
  Score distributions are fundamentally different between cosine similarity
  (bounded [-1,1]) and ts_rank_cd (unbounded float). Rank-based fusion is
  distribution-agnostic and empirically outperforms score fusion on BEIR
  benchmark (Thakur et al., 2021).

RRF formula:
  rrf_score(d) = Σ_r  1 / (k + rank_r(d))   where k=60 (standard constant)

Why k=60?
  Proposed in the original RRF paper (Cormack et al., 2009). Reduces the
  outsized influence of rank-1 results and was shown to maximize nDCG@10.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import asyncpg
from pgvector.asyncpg import register_vector

from config import get_settings

logger = logging.getLogger(__name__)

# RRF constant — do not change without benchmarking
_RRF_K = 60

# Candidate pool per retrieval arm before fusion
# 3× top_k gives RRF enough candidates to work with
_CANDIDATE_MULTIPLIER = 3


# ── Hybrid Search (Async — used by FastAPI agents) ────────────────────────────

async def search_vector_store(
    query_embedding: List[float],
    query_text: str,
    tenant_id: str,
    top_k: int | None = None,
    pool: asyncpg.Pool | None = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval with Reciprocal Rank Fusion.

    Args:
        query_embedding: 1536-dim float vector from text-embedding-3-large.
        query_text:      Raw user query string (used for BM25 sparse search).
        tenant_id:       Tenant isolation filter — ALWAYS applied.
        top_k:           Number of chunks to return (default from settings).
        pool:            Optional asyncpg pool (pass from FastAPI app state
                         to reuse connections; creates a one-off conn if None).

    Returns:
        List of dicts with keys: chunk_id, doc_id, content, metadata,
        rrf_score, dense_score, sparse_score.
        Sorted by rrf_score descending.
    """
    settings = get_settings()
    k = top_k or settings.vector_search_top_k
    candidate_limit = k * _CANDIDATE_MULTIPLIER

    ts_query = _build_tsquery(query_text)
    sql = _build_rrf_sql()

    async def _execute(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        rows = await conn.fetch(
            sql,
            query_embedding,   # $1 — vector for dense search
            tenant_id,         # $2 — always applied
            candidate_limit,   # $3 — candidate pool size
            ts_query,          # $4 — plainto_tsquery input
            _RRF_K,            # $5 — RRF k constant
            k,                 # $6 — final top_k result limit
        )
        return [dict(row) for row in rows]

    if pool:
        async with pool.acquire() as conn:
            await register_vector(conn)
            return await _execute(conn)
    else:
        # Fallback: create a one-off connection (use pool in production)
        settings = get_settings()
        conn = await asyncpg.connect(settings.database_url)
        await register_vector(conn)
        try:
            return await _execute(conn)
        finally:
            await conn.close()


# ── Sync wrapper (used by PySpark ETL / CLI tools) ───────────────────────────

def search_vector_store_sync(
    query_embedding: List[float],
    query_text: str,
    tenant_id: str,
    db_connection_str: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Synchronous version for non-async contexts (scripts, CLI, Spark).
    Uses psycopg2 with a direct cosine search (no RRF) for simplicity.
    Full hybrid search is async-only via search_vector_store().
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    sql = """
        SELECT doc_id, chunk_id::text, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM enterprise_embeddings
        WHERE tenant_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with psycopg2.connect(db_connection_str) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # pgvector psycopg2 needs the vector as a list literal string
            vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            cur.execute(sql, (vec_str, tenant_id, vec_str, top_k))
            return [dict(row) for row in cur.fetchall()]


# ── SQL Builder ───────────────────────────────────────────────────────────────

def _build_rrf_sql() -> str:
    """
    Constructs the RRF fusion SQL.

    Structure:
      1. dense_results CTE  — top N by cosine similarity (pgvector)
      2. sparse_results CTE — top N by ts_rank_cd (PostgreSQL FTS)
      3. rrf_fusion CTE     — FULL OUTER JOIN + RRF score computation
      4. Final SELECT       — ORDER BY rrf_score DESC LIMIT top_k
    
    COALESCE in rrf_fusion handles chunks that appear in only one arm:
      - Dense-only chunk: sparse_rank contribution is 0
      - Sparse-only chunk: dense_rank contribution is 0
    
    plainto_tsquery vs to_tsquery:
      plainto_tsquery handles arbitrary user text (spaces, stop words)
      without requiring tsquery syntax. Safer than to_tsquery for user input.
    """
    return """
    WITH dense_results AS (
        SELECT
            chunk_id,
            doc_id,
            content,
            metadata,
            1 - (embedding <=> $1::vector)          AS dense_score,
            ROW_NUMBER() OVER (
                ORDER BY embedding <=> $1::vector
            )                                        AS dense_rank
        FROM enterprise_embeddings
        WHERE tenant_id = $2
        LIMIT $3
    ),
    sparse_results AS (
        SELECT
            chunk_id,
            doc_id,
            content,
            metadata,
            ts_rank_cd(content_tsv, plainto_tsquery('english', $4))  AS sparse_score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $4)) DESC
            )                                                          AS sparse_rank
        FROM enterprise_embeddings
        WHERE tenant_id   = $2
          AND content_tsv @@ plainto_tsquery('english', $4)
        LIMIT $3
    ),
    rrf_fusion AS (
        SELECT
            COALESCE(d.chunk_id,  s.chunk_id)   AS chunk_id,
            COALESCE(d.doc_id,    s.doc_id)     AS doc_id,
            COALESCE(d.content,   s.content)    AS content,
            COALESCE(d.metadata,  s.metadata)   AS metadata,
            COALESCE(1.0 / ($5::float + d.dense_rank),  0.0) +
            COALESCE(1.0 / ($5::float + s.sparse_rank), 0.0) AS rrf_score,
            d.dense_score,
            s.sparse_score
        FROM  dense_results  d
        FULL OUTER JOIN sparse_results s USING (chunk_id)
    )
    SELECT *
    FROM   rrf_fusion
    ORDER  BY rrf_score DESC
    LIMIT  $6;
    """


def _build_tsquery(text: str) -> str:
    """
    Sanitise free-text for use as a plainto_tsquery parameter.

    Although the string is passed as a SQL bind parameter (never interpolated
    directly), we strip SQL-special and tsquery-special characters as
    defence-in-depth and to prevent unexpected tsquery parse errors.

    Stripped characters:
      SQL-special:    '  ;  --  /*  */
      tsquery-syntax: &  |  !  :  (  )
      Non-printable:  control chars (< 0x20)
    """
    if not text:
        return "enterprise"

    # Strip non-printable / control characters
    cleaned = re.sub(r"[^\x20-\x7E]", " ", text)

    # Strip SQL-injection chars and tsquery syntax tokens (defence-in-depth)
    # Note: plainto_tsquery is parameterised so SQL injection is already blocked,
    # but removing these prevents tsquery parse errors and operator smuggling.
    cleaned = re.sub(r"[';:&|!()\"/\\]", " ", cleaned)
    # Remove SQL comment sequences
    cleaned = cleaned.replace("--", " ").replace("/*", " ").replace("*/", " ")

    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Cap at 500 chars — plainto_tsquery tokenises the whole string
    return cleaned[:500] if cleaned else "enterprise"
