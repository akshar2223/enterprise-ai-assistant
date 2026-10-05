"""
Supervisor Router Agent.

Responsibilities (from supervisor_router.agent.md):
  1. Classify user intent into DOC_QUERY | DATA_QUERY | HYBRID_QUERY
  2. Extract operational parameters (tenant_id already in state)
  3. Generate execution plan for downstream agents

Why temperature=0.0?
  Routing decisions must be deterministic. Non-zero temperature introduces
  random variance into the DAG path selection, which would make the system
  non-reproducible and untestable.

Why JSON output?
  Structured routing output eliminates fragile string parsing. LLMClient
  uses OpenAI's json_object response_format so the model is guaranteed
  to return valid JSON.
"""
from __future__ import annotations

import logging
import time

from agents.state import AgentState
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── System Prompt (mirrors supervisor_router.instructions.md) ─────────────────

_SYSTEM_PROMPT = """\
You are the Enterprise AI Supervisor and Request Router.

Your sole task is to classify incoming user queries and generate an execution plan
for the multi-agent pipeline.

CLASSIFICATION RULES:
- DOC_QUERY:   The request requires knowledge from unstructured documents
               (policies, reports, procedures, meeting notes, technical docs).
- DATA_QUERY:  The request requires structured / tabular data analytics
               (sales figures, KPIs, database records, time-series metrics).
- HYBRID_QUERY: The request requires BOTH document knowledge AND structured data
                (e.g., "Compare our Q3 EMEA sales against our strategic risk policy").

EXECUTION PLAN RULES:
- DOC_QUERY   → ["doc_retrieval_agent"]
- DATA_QUERY  → ["sql_analytics_agent"]
- HYBRID_QUERY → ["doc_retrieval_agent", "sql_analytics_agent"]

OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
{
  "intent": "DOC_QUERY" | "DATA_QUERY" | "HYBRID_QUERY",
  "reasoning": "<one sentence rationale>",
  "execution_plan": ["agent_name_1"]
}
"""


# ── Node Function ─────────────────────────────────────────────────────────────

async def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node: classify intent and set routing state.

    Returns a partial AgentState update dict (only keys this node sets).
    """
    start_ms = time.time()
    query = state.get("query", "")

    logger.info(
        "[Supervisor] Routing query | tenant=%s | query_preview=%s…",
        state.get("tenant_id"),
        query[:80],
    )

    llm = LLMClient(temperature=0.0)

    try:
        result = await llm.chat_json(
            system_prompt=_SYSTEM_PROMPT,
            user_message=query,
        )
    except Exception as exc:
        logger.error("[Supervisor] LLM call failed: %s", exc)
        # Safe default: fall back to DOC_QUERY rather than crashing the pipeline
        result = {
            "intent": "DOC_QUERY",
            "reasoning": f"Fallback routing due to LLM error: {exc}",
            "execution_plan": ["doc_retrieval_agent"],
        }

    intent = result.get("intent", "DOC_QUERY")
    reasoning = result.get("reasoning", "")
    execution_plan = result.get("execution_plan", ["doc_retrieval_agent"])

    # Validate intent is one of the known values
    valid_intents = {"DOC_QUERY", "DATA_QUERY", "HYBRID_QUERY"}
    if intent not in valid_intents:
        logger.warning(
            "[Supervisor] Unknown intent '%s' — defaulting to DOC_QUERY", intent
        )
        intent = "DOC_QUERY"
        execution_plan = ["doc_retrieval_agent"]

    latency = int((time.time() - start_ms) * 1000)
    metrics = state.get("execution_metrics") or {}
    metrics["supervisor_ms"] = latency

    logger.info(
        "[Supervisor] ✅ intent=%s | latency=%dms | plan=%s",
        intent,
        latency,
        execution_plan,
    )

    return {
        "intent": intent,
        "routing_reasoning": reasoning,
        "execution_plan": execution_plan,
        "execution_metrics": metrics,
    }
