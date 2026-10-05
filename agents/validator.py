"""
Validation & Hallucination Guardrail Agent.

Responsibilities (from validation_guardrail.agent.md):
  1. Extract all discrete factual claims from the draft response
  2. Verify each claim against raw source context (doc chunks + SQL rows)
  3. Compute a faithfulness_score ∈ [0.0, 1.0]
  4. Output PASS (score ≥ threshold) or REJECT with remediation detail

Why a dedicated validation agent?
  LLMs hallucinate. Even with a well-grounded synthesizer, low-probability
  token sequences can introduce ungrounded claims. An independent evaluator
  using a separate LLM call acts as a second-opinion gate, catching errors
  the synthesizer introduces. This is the "LLM-as-judge" pattern validated
  by the RAGAS and G-Eval evaluation frameworks.

Why use the smaller llama3-8b model in the spec?
  Validation is an entailment task (binary: grounded or not), not generation.
  Smaller models perform comparably to large ones on NLI tasks while being
  faster and cheaper. In this implementation we use GPT-4o-mini equivalent
  via the same LLM client — swap model in .env once a local model is available.

strict_validation behaviour:
  If state["strict_validation"] is True (default):  REJECT blocks final_response
  If state["strict_validation"] is False:           REJECT is logged but
    final_response is still set (soft gate for demo/dev environments)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from agents.state import AgentState
from core.llm_client import LLMClient
from config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompt (mirrors faithfulness_check.prompt.md) ─────────────────────

_SYSTEM_PROMPT = """\
You are an Enterprise Compliance and Fact Validation Specialist.

Your task: evaluate whether EVERY factual claim in the Target Answer is
directly supported by the Ground Truth Source Context (SQL results + document chunks).

EVALUATION PROCESS:
1. Read the Target Answer and extract all discrete factual statements,
   named entities, and numerical figures.
2. For each claim, find corroborating evidence in the Source Context.
3. Classify each claim as SUPPORTED or UNSUPPORTED.
4. Compute faithfulness_score = supported_claims / total_claims.

STRICT RULES:
- A claim is SUPPORTED only if it is explicitly stated in the Source Context.
  Logical inferences or common sense conclusions do NOT count.
- Numerical figures must match exactly (±0 tolerance) unless the claim
  uses explicit approximation language ("approximately", "around").
- If no claims exist (empty response), return faithfulness_score=1.0, is_grounded=true.

OUTPUT — respond with ONLY valid JSON, no other text:
{
  "faithfulness_score": <float 0.0-1.0>,
  "is_grounded": <true|false>,
  "unsupported_claims": ["claim text 1", "claim text 2"]
}
"""

# ── Node Function ─────────────────────────────────────────────────────────────

async def validator_node(state: AgentState) -> dict:
    """
    LangGraph node: evaluate faithfulness → PASS or REJECT draft response.
    """
    start_ms = time.time()
    settings = get_settings()

    # Use `or` fallback — LangGraph 0.0.39 sets all unset TypedDict keys to None
    draft_response: str = state.get("draft_response") or ""
    doc_chunks: List[Dict[str, Any]] = state.get("doc_chunks") or []
    sql_results: List[Dict[str, Any]] = state.get("sql_results") or []
    strict: bool = state.get("strict_validation") if state.get("strict_validation") is not None else True

    logger.info("[Validator] Evaluating faithfulness | strict=%s", strict)

    # ── Build source context for the evaluator ────────────────────────────────
    source_context = _build_source_context(doc_chunks, sql_results)

    user_message = (
        f"Target Answer to Validate:\n{draft_response}\n\n"
        f"Ground Truth Source Context:\n{source_context}"
    )

    llm = LLMClient(temperature=0.0)

    try:
        result = await llm.chat_json(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
        )

        faithfulness_score = float(result.get("faithfulness_score", 0.0))
        is_grounded = bool(result.get("is_grounded", False))
        unsupported_claims: List[str] = result.get("unsupported_claims", [])

    except Exception as exc:
        logger.error("[Validator] ❌ Validation LLM failed: %s", exc)
        # Fail-safe: if we can't validate, conservatively REJECT in strict mode
        faithfulness_score = 0.0
        is_grounded = False
        unsupported_claims = [f"Validation failed with error: {exc}"]

    # ── Apply threshold gate ───────────────────────────────────────────────────
    threshold = settings.faithfulness_threshold  # 0.96 per requirements.md
    passes = faithfulness_score >= threshold and is_grounded

    if passes:
        validation_status = "PASS"
        final_response = draft_response
        logger.info(
            "[Validator] ✅ PASS | score=%.3f | threshold=%.2f",
            faithfulness_score,
            threshold,
        )
    else:
        validation_status = "REJECT"
        # In strict mode: block the response. In soft mode: pass it with warning.
        final_response = None if strict else draft_response
        logger.warning(
            "[Validator] ⚠️ REJECT | score=%.3f | threshold=%.2f | strict=%s"
            " | unsupported_claims=%s",
            faithfulness_score,
            threshold,
            strict,
            unsupported_claims,
        )

    latency = int((time.time() - start_ms) * 1000)
    metrics = state.get("execution_metrics") or {}
    metrics["validator_ms"] = latency
    metrics["faithfulness_score"] = faithfulness_score

    return {
        "validation_status": validation_status,
        "faithfulness_score": faithfulness_score,
        "unsupported_claims": unsupported_claims,
        "final_response": final_response,
        "execution_metrics": metrics,
    }


# ── Context Builder ───────────────────────────────────────────────────────────

def _build_source_context(
    doc_chunks: List[Dict[str, Any]],
    sql_results: List[Dict[str, Any]],
) -> str:
    """
    Compact source context for the faithfulness evaluator.
    Cap at reasonable token size — validation is a classification task,
    not generation, so the evaluator doesn't need every chunk verbatim.
    """
    parts: List[str] = []

    if sql_results:
        # Cap at 20 rows for the evaluator — enough to check numerical claims
        display = sql_results[:20]
        parts.append(
            "SQL Query Results:\n"
            + json.dumps(display, indent=2, default=str)
        )

    if doc_chunks:
        parts.append("Document Chunks:")
        for chunk in doc_chunks[:5]:  # Top-5 most relevant chunks
            parts.append(
                f"[{chunk.get('doc_id', 'unknown')}] "
                f"{chunk.get('content', '')[:600]}"  # 600 chars per chunk
            )

    return "\n\n".join(parts) if parts else "No source context available."
