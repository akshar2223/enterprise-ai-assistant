"""
Async Audit Log Writer.

Every agent pipeline execution writes one audit record capturing:
  - Who ran it (tenant_id, user_id)
  - What was asked (conversation_id, event_type)
  - How it went (faithfulness_score, latency_ms)
  - What happened internally (payload JSONB — routing, SQL, citations)

The audit_logs table is append-only (enforced by PostgreSQL RULE in init.sql).
No UPDATE or DELETE is permitted at the database level.

Why async?
  Audit writes happen AFTER the response is sent to the client (fire-and-forget
  via asyncio.create_task). They must not block response latency.

Why JSONB payload?
  The schema of each event_type differs (SQL queries look different from
  vector search hits). JSONB gives schema flexibility without migrations
  for every new event type. Querying specific fields remains fast via
  PostgreSQL JSONB operators and GIN indexes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

from agents.state import AgentState
from core.database import get_pool

logger = logging.getLogger(__name__)


# ── Event Type Constants ──────────────────────────────────────────────────────

class AuditEvent:
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    VECTOR_SEARCH     = "VECTOR_SEARCH"
    SQL_QUERY         = "SQL_QUERY"
    VALIDATION        = "VALIDATION"
    AUTH_FAILURE      = "AUTH_FAILURE"
    PII_REDACTION     = "PII_REDACTION"


# ── Primary Write Function ────────────────────────────────────────────────────

async def write_pipeline_audit(
    final_state: AgentState,
    total_latency_ms: int,
) -> None:
    """
    Write a single PIPELINE_COMPLETE audit record from the final agent state.

    Called as a fire-and-forget task in the API endpoint:
        asyncio.create_task(write_pipeline_audit(state, latency))

    Never raises — audit failure must not surface to the client.
    """
    try:
        pool = await get_pool()
        metrics = final_state.get("execution_metrics", {})

        payload: Dict[str, Any] = {
            "intent": final_state.get("intent"),
            "routing_reasoning": final_state.get("routing_reasoning"),
            "sql_query": final_state.get("sql_query"),
            "sql_rows": len(final_state.get("sql_results", [])),
            "retrieved_chunks": metrics.get("retrieved_chunks", 0),
            "citations": final_state.get("citations", [])[:10],  # cap JSONB size
            "validation_status": final_state.get("validation_status"),
            "unsupported_claims": final_state.get("unsupported_claims", []),
            "execution_metrics": metrics,
        }

        await _insert_audit_row(
            pool=pool,
            conversation_id=final_state.get("conversation_id", "unknown"),
            tenant_id=final_state.get("tenant_id", "unknown"),
            user_id=final_state.get("user_id"),
            event_type=AuditEvent.PIPELINE_COMPLETE,
            agent_name="orchestrator",
            payload=payload,
            latency_ms=total_latency_ms,
            faithfulness_score=final_state.get("faithfulness_score"),
        )

        logger.debug(
            "[Audit] Wrote PIPELINE_COMPLETE | conversation=%s | latency=%dms",
            final_state.get("conversation_id"),
            total_latency_ms,
        )

    except Exception as exc:
        # Audit failure is logged but NEVER re-raised — must not affect client response
        logger.error("[Audit] ❌ Failed to write audit log: %s", exc, exc_info=True)


async def write_auth_failure_audit(
    tenant_id: str,
    user_id: str,
    reason: str,
    request_ip: Optional[str] = None,
) -> None:
    """Write an AUTH_FAILURE audit event for security monitoring."""
    try:
        pool = await get_pool()
        await _insert_audit_row(
            pool=pool,
            conversation_id="n/a",
            tenant_id=tenant_id or "unknown",
            user_id=user_id,
            event_type=AuditEvent.AUTH_FAILURE,
            agent_name="api_gateway",
            payload={"reason": reason, "source_ip": request_ip},
            latency_ms=0,
            faithfulness_score=None,
        )
    except Exception as exc:
        logger.error("[Audit] Failed to write auth failure log: %s", exc)


# ── Internal DB Writer ────────────────────────────────────────────────────────

async def _insert_audit_row(
    pool: asyncpg.Pool,
    conversation_id: str,
    tenant_id: str,
    user_id: Optional[str],
    event_type: str,
    agent_name: str,
    payload: Dict[str, Any],
    latency_ms: int,
    faithfulness_score: Optional[float],
) -> None:
    """
    Insert one row into audit_logs.
    Uses pool.execute (not fetch) — we don't need a return value.
    """
    sql = """
        INSERT INTO audit_logs
            (conversation_id, tenant_id, user_id, event_type,
             agent_name, payload, latency_ms, faithfulness_score)
        VALUES
            ($1, $2, $3, $4, $5, $6::jsonb, $7, $8);
    """
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            conversation_id,
            tenant_id,
            user_id,
            event_type,
            agent_name,
            json.dumps(payload, default=str),
            latency_ms,
            faithfulness_score,
        )
