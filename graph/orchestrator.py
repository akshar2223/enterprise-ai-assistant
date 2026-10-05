"""
LangGraph Multi-Agent Orchestrator.

Implements the Stateful DAG topology from agents/agents.md:

        [START]
           │
      [supervisor]          ← intent classification
           │
    ┌──────┴──────┐
    │             │
[doc_retrieval] [sql_analytics]   ← parallel capable (sequential v1)
    │             │
    └──────┬──────┘
           │
      [synthesizer]         ← context fusion
           │
       [validator]          ← faithfulness gate
           │
         [END]

Routing logic:
  DOC_QUERY    → doc_retrieval → synthesizer → validator
  DATA_QUERY   → sql_analytics → synthesizer → validator
  HYBRID_QUERY → doc_retrieval → sql_analytics → synthesizer → validator

Why sequential for HYBRID_QUERY in v1?
  LangGraph 0.0.39 supports parallel edges via Send API, which requires
  a reducer on the state list fields. For v1, sequential is correct,
  auditable, and avoids race conditions on shared state keys.
  Upgrade path: use langgraph>=0.1.0 with Send + operator.add reducer
  to run doc_retrieval and sql_analytics concurrently.

Usage:
  from graph.orchestrator import build_agent_graph

  graph = build_agent_graph()                  # compile once at startup
  result = await graph.ainvoke(initial_state)  # call per request
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.doc_retrieval import doc_retrieval_node
from agents.sql_analytics import sql_analytics_node
from agents.synthesizer import synthesizer_node
from agents.validator import validator_node

logger = logging.getLogger(__name__)

# ── Node names (single source of truth to avoid typos in edge wiring) ─────────
_SUPERVISOR = "supervisor"
_DOC_RETRIEVAL = "doc_retrieval"
_SQL_ANALYTICS = "sql_analytics"
_SYNTHESIZER = "synthesizer"
_VALIDATOR = "validator"


# ── Routing Functions (conditional edge predicates) ───────────────────────────

def _route_from_supervisor(
    state: AgentState,
) -> Literal["doc_retrieval", "sql_analytics"]:
    """
    After supervisor: decide first downstream node.

    DOC_QUERY   → doc_retrieval
    HYBRID_QUERY → doc_retrieval (sql will run after doc in the HYBRID path)
    DATA_QUERY  → sql_analytics (skip doc retrieval)
    """
    intent = state.get("intent", "DOC_QUERY")
    if state.get("error"):
        # If supervisor errored, attempt doc retrieval as safe fallback
        logger.warning("[Orchestrator] Supervisor error — routing to doc_retrieval")
        return _DOC_RETRIEVAL

    if intent == "DATA_QUERY":
        return _SQL_ANALYTICS
    return _DOC_RETRIEVAL  # DOC_QUERY and HYBRID_QUERY both start with doc


def _route_after_doc_retrieval(
    state: AgentState,
) -> Literal["sql_analytics", "synthesizer"]:
    """
    After doc retrieval: for HYBRID_QUERY also run sql_analytics;
    for DOC_QUERY skip straight to synthesizer.
    """
    if state.get("intent") == "HYBRID_QUERY":
        return _SQL_ANALYTICS
    return _SYNTHESIZER


def _route_after_validation(
    state: AgentState,
) -> Literal["__end__"]:
    """
    After validation: always terminate.
    A REJECT is surfaced via validation_status in the API response;
    we don't loop back to synthesizer in v1 (would risk infinite retries).

    v2 upgrade: add a retry counter to state and loop back to synthesizer
    with remediation instructions from unsupported_claims if retries < 2.
    """
    if state.get("validation_status") == "REJECT":
        logger.warning(
            "[Orchestrator] Response REJECTED by validator "
            "(score=%.3f, claims=%s)",
            state.get("faithfulness_score", 0.0),
            state.get("unsupported_claims", [])[:3],
        )
    return END


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Compile and return the LangGraph StateGraph.

    Call once at application startup and reuse the compiled graph
    across all requests — compilation is expensive (graph validation),
    invocation is cheap.

    Returns:
        A compiled LangGraph CompiledGraph that supports .ainvoke() and .invoke().
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node(_SUPERVISOR, supervisor_node)
    graph.add_node(_DOC_RETRIEVAL, doc_retrieval_node)
    graph.add_node(_SQL_ANALYTICS, sql_analytics_node)
    graph.add_node(_SYNTHESIZER, synthesizer_node)
    graph.add_node(_VALIDATOR, validator_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point(_SUPERVISOR)

    # ── Conditional edge: supervisor → doc_retrieval | sql_analytics ──────────
    graph.add_conditional_edges(
        _SUPERVISOR,
        _route_from_supervisor,
        {
            _DOC_RETRIEVAL: _DOC_RETRIEVAL,
            _SQL_ANALYTICS: _SQL_ANALYTICS,
        },
    )

    # ── Conditional edge: doc_retrieval → sql_analytics | synthesizer ─────────
    graph.add_conditional_edges(
        _DOC_RETRIEVAL,
        _route_after_doc_retrieval,
        {
            _SQL_ANALYTICS: _SQL_ANALYTICS,
            _SYNTHESIZER: _SYNTHESIZER,
        },
    )

    # ── Unconditional edges ───────────────────────────────────────────────────
    graph.add_edge(_SQL_ANALYTICS, _SYNTHESIZER)
    graph.add_edge(_SYNTHESIZER, _VALIDATOR)

    # ── Terminal conditional edge: validator → END ────────────────────────────
    graph.add_conditional_edges(
        _VALIDATOR,
        _route_after_validation,
        {END: END},
    )

    compiled = graph.compile()
    logger.info(
        "[Orchestrator] ✅ Agent graph compiled | nodes=%s",
        [_SUPERVISOR, _DOC_RETRIEVAL, _SQL_ANALYTICS, _SYNTHESIZER, _VALIDATOR],
    )
    return compiled
