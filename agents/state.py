"""
LangGraph Shared Agent State Schema.

Every node in the DAG reads from and writes partial updates to this TypedDict.
LangGraph merges updates per key — nodes return ONLY the keys they change.

Design rules:
  - All fields are Optional (total=False) so nodes can return sparse updates
  - Lists are replaced wholesale by node updates (not accumulated)
    → Use Annotated[List, operator.add] reducer only if you need accumulation
  - execution_metrics is the single dict accumulating perf data across nodes

State flow (from orchestrator.py):
  [INPUT]
  query, tenant_id, user_id, conversation_id, strict_validation
      ↓
  [supervisor]   → intent, execution_plan, routing_reasoning
      ↓
  [doc_retrieval] → doc_chunks, updated execution_metrics
  [sql_analytics] → sql_query, sql_results, updated execution_metrics
      ↓
  [synthesizer]  → draft_response, citations
      ↓
  [validator]    → validation_status, faithfulness_score, final_response
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                         # Original user query (never mutated)
    tenant_id: str                     # Multi-tenancy isolation key
    user_id: str                       # For audit trail
    conversation_id: str               # UUID linking all log entries
    strict_validation: bool            # If True, REJECT blocks final response

    # ── Supervisor Routing ────────────────────────────────────────────────────
    intent: str                        # DOC_QUERY | DATA_QUERY | HYBRID_QUERY
    execution_plan: List[str]          # Ordered list of agent names
    routing_reasoning: str             # Supervisor's rationale (for logs)

    # ── Document Retrieval ────────────────────────────────────────────────────
    doc_chunks: List[Dict[str, Any]]   # [{chunk_id, doc_id, content, metadata,
                                       #   rrf_score, dense_score, sparse_score}]

    # ── SQL Analytics ─────────────────────────────────────────────────────────
    sql_query: str                     # Generated & AST-validated SQL
    sql_explanation: str               # LLM's description of what the SQL does
    sql_results: List[Dict[str, Any]]  # Raw rows from PostgreSQL

    # ── Response Synthesis ────────────────────────────────────────────────────
    draft_response: str                # Pre-validation executive summary
    citations: List[Dict[str, Any]]    # [{source_id, chunk_id, relevance_score}]

    # ── Validation Gate ───────────────────────────────────────────────────────
    validation_status: str             # PASS | REJECT
    faithfulness_score: float          # 0.0–1.0 (threshold: 0.96 per NFR)
    unsupported_claims: List[str]      # Claims flagged as ungrounded
    final_response: Optional[str]      # None if validation rejected

    # ── Cross-cutting ─────────────────────────────────────────────────────────
    error: Optional[str]               # Set by any node on unrecoverable error
    execution_metrics: Dict[str, Any]  # Accumulated perf data across all nodes
