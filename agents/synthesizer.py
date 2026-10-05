"""
Response Synthesizer Agent.

Responsibilities (from response_synthesizer.agent.md):
  1. Merge structured SQL results with unstructured document chunks
  2. Generate a concise executive summary grounded ONLY in provided context
  3. Produce inline citations referencing source doc_ids
  4. Format citations list for the API response contract

Temperature=0.2 (not 0.0):
  Synthesis benefits from mild creativity for fluent language, but we
  cap at 0.2 to prevent the model from hallucinating beyond the context.
  Temperature=0.0 produces stilted, list-like summaries.

Context construction strategy:
  SQL results → JSON block (numerical precision preserved)
  Doc chunks  → sorted by rrf_score DESC (best evidence first)
  Total context size → capped at ~6000 tokens to fit in GPT-4o's 128k
  context while leaving room for the generated response.

Why cap context?
  Sending all retrieved chunks without a cap risks hitting token limits
  on lower-context models and increases cost. The top-5 chunks (default
  top_k) at 512 tokens each = 2560 tokens — well within limits.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from agents.state import AgentState
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── System Prompt (mirrors rag_synthesis.prompt.md) ───────────────────────────

_SYSTEM_PROMPT = """\
You are an Enterprise Executive AI Assistant.

Your task: synthesize a concise, professional executive briefing from the
provided structured data (SQL query results) and unstructured document context.

STRICT RULES:
1. Use ONLY the information present in the provided context — do not add
   external knowledge or make inferences beyond what is stated.
2. For every factual claim or numerical figure, include an inline citation
   using the format [doc_id] or [SQL] for SQL-derived data.
3. Keep the response under 400 words unless the complexity genuinely requires more.
4. Structure: lead with the direct answer, then supporting evidence, then
   any relevant caveats from the source documents.
5. If the context is insufficient to answer the query, state that explicitly
   rather than speculating.
"""

# ── Node Function ─────────────────────────────────────────────────────────────

async def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph node: build context → call LLM → extract citations.
    """
    start_ms = time.time()
    query = state.get("query", "")
    # Use `or []` not `get(k, [])` — LangGraph 0.0.39 initialises all TypedDict
    # keys to None, so .get(k, default) returns None when the key is present.
    doc_chunks: List[Dict[str, Any]] = state.get("doc_chunks") or []
    sql_results: List[Dict[str, Any]] = state.get("sql_results") or []
    sql_query: str = state.get("sql_query") or ""

    logger.info(
        "[Synthesizer] Fusing context | doc_chunks=%d | sql_rows=%d",
        len(doc_chunks),
        len(sql_results),
    )

    # ── Build context blocks ──────────────────────────────────────────────────
    sql_block = _build_sql_block(sql_results, sql_query)
    doc_block = _build_doc_block(doc_chunks)

    user_message = (
        f"User Query: {query}\n\n"
        f"{sql_block}\n\n"
        f"{doc_block}\n\n"
        "Generate a concise executive briefing with inline citations."
    )

    llm = LLMClient(temperature=0.2)

    try:
        draft_response = await llm.chat(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
        )
    except Exception as exc:
        logger.error("[Synthesizer] ❌ LLM call failed: %s", exc)
        draft_response = (
            "Unable to generate response due to a synthesis error. "
            "Please retry or contact support."
        )

    # ── Build citation list ───────────────────────────────────────────────────
    citations = _build_citations(doc_chunks or [])

    latency = int((time.time() - start_ms) * 1000)
    metrics = state.get("execution_metrics") or {}
    metrics["synthesizer_ms"] = latency

    logger.info(
        "[Synthesizer] ✅ Draft generated | citations=%d | latency=%dms",
        len(citations),
        latency,
    )

    return {
        "draft_response": draft_response,
        "citations": citations,
        "execution_metrics": metrics,
    }


# ── Context Builders ──────────────────────────────────────────────────────────

def _build_sql_block(sql_results: List[Dict], sql_query: str) -> str:
    """Format SQL results as a labelled JSON block."""
    if not sql_results:
        return "Structured Data Context (SQL Results): No data retrieved."

    # Cap at 50 rows to avoid bloating the prompt
    display_rows = sql_results[:50]
    truncated = len(sql_results) > 50

    block = (
        "Structured Data Context (SQL Results):\n"
        f"```json\n{json.dumps(display_rows, indent=2, default=str)}\n```"
    )
    if truncated:
        block += f"\n[Note: Showing first 50 of {len(sql_results)} rows]"
    if sql_query:
        block += f"\nSource SQL: `{sql_query[:300]}{'…' if len(sql_query) > 300 else ''}`"
    return block


def _build_doc_block(doc_chunks: List[Dict]) -> str:
    """Format document chunks sorted by relevance score."""
    if not doc_chunks:
        return "Unstructured Document Context: No documents retrieved."

    # Sort by rrf_score descending — best evidence first
    sorted_chunks = sorted(
        doc_chunks,
        key=lambda c: c.get("rrf_score", c.get("dense_score", 0.0)),
        reverse=True,
    )

    lines = ["Unstructured Document Context:"]
    for chunk in sorted_chunks:
        lines.append(
            f"---\n"
            f"Document ID: {chunk.get('doc_id', 'unknown')}\n"
            f"Relevance Score: {chunk.get('rrf_score', chunk.get('dense_score', 0.0)):.4f}\n"
            f"Content: {chunk.get('content', '')[:800]}"  # cap chunk display at 800 chars
        )
    return "\n".join(lines)


def _build_citations(doc_chunks: List[Dict]) -> List[Dict[str, Any]]:
    """Build the citations list matching the API response contract."""
    citations = []
    for chunk in doc_chunks:
        # Use `or` fallback — dict.get(key, default) returns None when the key
        # exists with a None value (e.g. LangGraph initialised keys). float(None)
        # raises TypeError, so we chain `or 0.0` to guarantee a numeric value.
        raw_score = chunk.get("rrf_score") or chunk.get("dense_score") or 0.0
        citations.append({
            "source_id": chunk.get("doc_id") or "unknown",
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "relevance_score": round(float(raw_score), 4),
        })
    return citations
