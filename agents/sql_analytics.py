"""
SQL Analytics Agent.

Responsibilities (from sql_analytics.agent.md):
  1. Inspect database schema to ground the SQL generation
  2. Translate user natural language into a PostgreSQL SELECT query (LLM)
  3. Validate the generated SQL via AST check (validate_sql_ast)
  4. Execute the read-only query against enterprise analytics tables
  5. Return structured JSON rows for the Synthesizer

Security architecture:
  validate_sql_ast → prohibits INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/COMMAND
  run_read_only_sql → wraps execute behind the validator
  asyncio.to_thread → runs synchronous psycopg2 in thread pool (non-blocking)
  Tenant isolation → SQL WHERE clause ALWAYS includes tenant_id filter

Why fetch schema at runtime?
  Hardcoding DDL in the prompt leads to drift as schemas evolve.
  Fetching from information_schema ensures the LLM always sees the current
  column definitions. In production, consider a schema cache with TTL.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

from agents.state import AgentState
from core.llm_client import LLMClient
from config import get_settings
from skills.run_read_only_sql import run_read_only_sql

logger = logging.getLogger(__name__)

# ── System Prompt (mirrors sql_analytics.instructions.md) ─────────────────────

_SYSTEM_PROMPT = """\
You are an expert PostgreSQL Data Engineer and SQL Specialist.

Your task: translate the user's natural language request into a safe, read-only
PostgreSQL SELECT query using the provided schema context.

STRICT RULES:
1. Generate ONLY a SELECT statement — NEVER INSERT, UPDATE, DELETE, DROP, or ALTER.
2. Always include explicit column names — NEVER use SELECT *.
3. Add LIMIT 100 unless the user explicitly requests a different row count.
4. Always include WHERE tenant_id = '<tenant_id_placeholder>' to enforce isolation.
   Replace <tenant_id_placeholder> with the actual tenant_id from context.
5. Use explicit table aliases for multi-table queries.
6. Prefer ILIKE over LIKE for case-insensitive text matching.

OUTPUT — respond with ONLY valid JSON, no other text:
{
  "sql_query": "<valid PostgreSQL SELECT statement>",
  "explanation": "<one sentence describing what the query does>"
}
"""


# ── Node Function ─────────────────────────────────────────────────────────────

async def sql_analytics_node(state: AgentState) -> dict:
    """
    LangGraph node: generate SQL → validate AST → execute → return rows.
    """
    start_ms = time.time()
    settings = get_settings()
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "default")

    logger.info("[SQLAnalytics] Starting text-to-SQL | tenant=%s", tenant_id)

    try:
        # ── Step 1: Fetch live schema from information_schema ─────────────────
        schema_ddl = await asyncio.to_thread(
            _fetch_schema_context, settings.database_url, tenant_id
        )

        # ── Step 2: Generate SQL via LLM ──────────────────────────────────────
        llm = LLMClient(temperature=0.0)
        user_message = (
            f"Database Schema:\n{schema_ddl}\n\n"
            f"Tenant ID: {tenant_id}\n\n"
            f"User Request: {query}"
        )

        result = await llm.chat_json(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
        )

        sql_query: str = result.get("sql_query", "")
        explanation: str = result.get("explanation", "")

        if not sql_query:
            raise ValueError("LLM returned empty sql_query field")

        logger.info("[SQLAnalytics] Generated SQL: %s…", sql_query[:120])

        # ── Step 3: Execute (validate_sql_ast is called inside run_read_only_sql)
        sql_results: List[Dict[str, Any]] = await asyncio.to_thread(
            run_read_only_sql,
            sql_query,
            settings.database_url,
        )

        latency = int((time.time() - start_ms) * 1000)
        metrics = state.get("execution_metrics") or {}
        metrics["sql_analytics_ms"] = latency
        metrics["sql_queries_run"] = metrics.get("sql_queries_run", 0) + 1
        metrics["sql_rows_returned"] = len(sql_results)

        logger.info(
            "[SQLAnalytics] ✅ %d rows returned | latency=%dms",
            len(sql_results),
            latency,
        )

        return {
            "sql_query": sql_query,
            "sql_explanation": explanation,
            "sql_results": sql_results,
            "execution_metrics": metrics,
        }

    except ValueError as exc:
        # Security violation or empty query — surface as application error
        logger.error("[SQLAnalytics] ❌ Validation error: %s", exc)
        metrics = state.get("execution_metrics") or {}
        metrics["sql_error"] = str(exc)
        return {
            "sql_results": [],
            "sql_query": "",
            "execution_metrics": metrics,
            "error": f"SQL generation or validation failed: {exc}",
        }

    except Exception as exc:
        logger.error("[SQLAnalytics] ❌ Unexpected error: %s", exc, exc_info=True)
        metrics = state.get("execution_metrics") or {}
        metrics["sql_error"] = str(exc)
        return {
            "sql_results": [],
            "sql_query": "",
            "execution_metrics": metrics,
            "error": f"SQL execution failed: {exc}",
        }


# ── Schema Helper ─────────────────────────────────────────────────────────────

def _fetch_schema_context(db_url: str, tenant_id: str) -> str:
    """
    Fetch live column metadata from information_schema for analytics tables.
    Returns a compact DDL-like string for injection into the LLM prompt.

    Runs synchronously — always call via asyncio.to_thread from async context.
    """
    target_tables = ("analytics_data", "enterprise_embeddings")

    query = """
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = ANY(%s)
        ORDER BY table_name, ordinal_position;
    """

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (list(target_tables),))
                rows = cur.fetchall()

        # Group by table and format as compact DDL
        tables: dict = {}
        for row in rows:
            tbl = row["table_name"]
            if tbl not in tables:
                tables[tbl] = []
            nullable = "" if row["is_nullable"] == "NO" else " NULL"
            default = f" DEFAULT {row['column_default']}" if row["column_default"] else ""
            tables[tbl].append(
                f"  {row['column_name']} {row['data_type'].upper()}{nullable}{default}"
            )

        ddl_parts = []
        for tbl, cols in tables.items():
            ddl_parts.append(f"TABLE {tbl} (\n" + ",\n".join(cols) + "\n);")

        # Append tenant hint for the LLM
        ddl_parts.append(
            f"\n-- Always filter: WHERE tenant_id = '{tenant_id}'"
        )

        return "\n\n".join(ddl_parts) if ddl_parts else "No schema available."

    except Exception as exc:
        logger.warning(
            "[SQLAnalytics] Could not fetch schema from DB: %s — using fallback", exc
        )
        # Fallback static schema — keeps SQL generation working even if DB is cold
        return _FALLBACK_SCHEMA.format(tenant_id=tenant_id)


# Fallback schema used when the database is unreachable (cold start / tests)
_FALLBACK_SCHEMA = """\
TABLE analytics_data (
  record_id UUID NOT NULL,
  tenant_id VARCHAR(64) NOT NULL,
  region VARCHAR(64) NULL,
  department VARCHAR(128) NULL,
  metric_name VARCHAR(128) NULL,
  metric_value NUMERIC(18,4) NULL,
  period VARCHAR(20) NULL,
  tags JSONB NULL,
  source_file VARCHAR(255) NULL,
  created_at TIMESTAMPTZ NULL DEFAULT CURRENT_TIMESTAMP
);

-- Always filter: WHERE tenant_id = '{tenant_id}'
"""
