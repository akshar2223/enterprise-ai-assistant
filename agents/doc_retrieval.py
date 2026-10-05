"""
Document Retrieval Agent.

Responsibilities (from doc_retrieval.agent.md):
  1. Convert user query into a dense embedding (text-embedding-3-large)
  2. Execute hybrid retrieval: pgvector dense + tsvector sparse + RRF fusion
  3. Filter results by tenant_id (enforced in SQL, not application code)
  4. Return ranked document chunks for the Synthesizer

Why embed in the agent rather than pre-compute?
  Query embeddings are ephemeral — they're generated per request and never
  stored. Document embeddings are stored during ETL ingestion. This keeps
  the retrieval path stateless and horizontally scalable.

Tenant isolation guarantee:
  The SQL in search_vector_store ALWAYS includes WHERE tenant_id = $2.
  This is enforced at the data layer, not just the API layer, satisfying
  the RBAC requirement in NFR-1.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from agents.state import AgentState
from core.database import get_pool
from core.embeddings import EmbeddingClient
from skills.search_vector_store import search_vector_store

logger = logging.getLogger(__name__)


async def doc_retrieval_node(state: AgentState) -> dict:
    """
    LangGraph node: embed query → hybrid search → return ranked chunks.
    """
    start_ms = time.time()
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "default")

    logger.info(
        "[DocRetrieval] Starting hybrid search | tenant=%s", tenant_id
    )

    try:
        # ── Step 1: Generate query embedding ──────────────────────────────────
        embed_client = EmbeddingClient()
        query_embedding: List[float] = await embed_client.embed(query)

        # ── Step 2: Hybrid search (dense + sparse + RRF) ──────────────────────
        pool = await get_pool()
        chunks: List[Dict[str, Any]] = await search_vector_store(
            query_embedding=query_embedding,
            query_text=query,
            tenant_id=tenant_id,
            pool=pool,
        )

        # ── Step 3: Serialise chunk_id UUIDs to str for JSON compatibility ────
        # asyncpg returns UUID objects; downstream JSON serialization needs str
        for chunk in chunks:
            if "chunk_id" in chunk and not isinstance(chunk["chunk_id"], str):
                chunk["chunk_id"] = str(chunk["chunk_id"])
            # Convert any float32 numpy types to native Python float
            for score_key in ("rrf_score", "dense_score", "sparse_score"):
                if score_key in chunk and chunk[score_key] is not None:
                    chunk[score_key] = float(chunk[score_key])

        latency = int((time.time() - start_ms) * 1000)
        metrics = state.get("execution_metrics") or {}
        metrics["doc_retrieval_ms"] = latency
        metrics["retrieved_chunks"] = len(chunks)

        logger.info(
            "[DocRetrieval] ✅ Retrieved %d chunks | latency=%dms",
            len(chunks),
            latency,
        )

        return {
            "doc_chunks": chunks,
            "execution_metrics": metrics,
        }

    except Exception as exc:
        logger.error("[DocRetrieval] ❌ Failed: %s", exc, exc_info=True)
        metrics = state.get("execution_metrics") or {}
        metrics["doc_retrieval_error"] = str(exc)
        return {
            "doc_chunks": [],
            "execution_metrics": metrics,
            "error": f"Document retrieval failed: {exc}",
        }
