"""
Integration Tests — LangGraph Multi-Agent Pipeline.

Strategy: Mock every external I/O call (LLM, DB, embeddings) so the graph
wiring, routing logic, and state transitions are tested without network access.

What IS tested:
  ✅ Supervisor routing (DOC_QUERY / DATA_QUERY / HYBRID_QUERY)
  ✅ State propagation between nodes
  ✅ Conditional edge logic (_route_from_supervisor, _route_after_doc_retrieval)
  ✅ PASS / REJECT validation gate
  ✅ REJECT + strict_validation=True → final_response is None
  ✅ Error propagation from individual nodes

What is NOT tested here (tested separately):
  ❌ Real OpenAI API responses (cost + non-determinism)
  ❌ Real PostgreSQL queries (requires running DB)
  ❌ Real embedding quality

Patching approach:
  We patch the CLASS (e.g. agents.supervisor.LLMClient) so that when the
  node function does `llm = LLMClient(...)`, it gets the mock instance.
  AsyncMock is used for async methods (.chat, .chat_json, .embed).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graph.orchestrator import build_agent_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_supervisor_mock(intent: str, plan: list[str]) -> AsyncMock:
    """Returns a mock LLMClient whose chat_json returns a routing decision."""
    mock_llm = AsyncMock()
    mock_llm.chat_json = AsyncMock(return_value={
        "intent": intent,
        "reasoning": f"Test routing to {intent}",
        "execution_plan": plan,
    })
    return mock_llm


def _make_synthesizer_mock(response_text: str) -> AsyncMock:
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=response_text)
    return mock_llm


def _make_validator_mock(score: float, grounded: bool) -> AsyncMock:
    mock_llm = AsyncMock()
    mock_llm.chat_json = AsyncMock(return_value={
        "faithfulness_score": score,
        "is_grounded": grounded,
        "unsupported_claims": [] if grounded else ["Claim X is unverified"],
    })
    return mock_llm


SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk-uuid-001",
        "doc_id": "doc_policy_2026_v2",
        "content": "EMEA Q3 sales reached $14.2M (+12% YoY). Risk exposure is within bounds.",
        "metadata": {},
        "rrf_score": 0.892,
        "dense_score": 0.871,
        "sparse_score": 0.634,
    }
]

SAMPLE_SQL_ROWS = [
    {"region": "EMEA", "period": "2026-Q3", "metric_value": 14200000.0}
]

BASE_INPUT = {
    "query": "What were the EMEA Q3 sales and risk policy implications?",
    "tenant_id": "corp_acme_01",
    "user_id": "usr_99823",
    "conversation_id": "conv_test_001",
    "strict_validation": True,
    "execution_metrics": {},
}


# ── Test: DOC_QUERY path ──────────────────────────────────────────────────────

class TestDocQueryPath:

    @pytest.mark.integration
    async def test_doc_query_full_pass(self):
        """DOC_QUERY path: supervisor → doc_retrieval → synthesizer → validator PASS."""
        graph = build_agent_graph()

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DOC_QUERY", ["doc_retrieval_agent"]
            )),
            patch("agents.doc_retrieval.EmbeddingClient") as mock_embed_cls,
            patch("agents.doc_retrieval.search_vector_store", return_value=SAMPLE_CHUNKS),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock(
                "EMEA Q3 sales reached $14.2M [doc_policy_2026_v2]."
            )),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.98, True)),
        ):
            mock_embed = AsyncMock()
            mock_embed.embed = AsyncMock(return_value=[0.01] * 1536)
            mock_embed_cls.return_value = mock_embed

            result = await graph.ainvoke(BASE_INPUT)

        assert result["intent"] == "DOC_QUERY"
        assert result["doc_chunks"] == SAMPLE_CHUNKS
        # DOC path never runs sql_analytics — key absent or None/empty
        assert result.get("sql_results") in (None, [])
        assert result["validation_status"] == "PASS"
        assert result["faithfulness_score"] == 0.98
        assert result["final_response"] is not None
        assert "EMEA" in result["final_response"]

    @pytest.mark.integration
    async def test_doc_query_validator_reject_strict(self):
        """REJECT in strict mode → final_response must be None."""
        graph = build_agent_graph()

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DOC_QUERY", ["doc_retrieval_agent"]
            )),
            patch("agents.doc_retrieval.EmbeddingClient") as mock_embed_cls,
            patch("agents.doc_retrieval.search_vector_store", return_value=SAMPLE_CHUNKS),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock(
                "Sales were $999M — completely fabricated."
            )),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.40, False)),
        ):
            mock_embed = AsyncMock()
            mock_embed.embed = AsyncMock(return_value=[0.01] * 1536)
            mock_embed_cls.return_value = mock_embed

            result = await graph.ainvoke({**BASE_INPUT, "strict_validation": True})

        assert result["validation_status"] == "REJECT"
        assert result["faithfulness_score"] == 0.40
        assert result["final_response"] is None               # strict=True blocks it
        assert len(result["unsupported_claims"]) > 0

    @pytest.mark.integration
    async def test_doc_query_validator_reject_soft_mode(self):
        """REJECT in soft mode (strict=False) → final_response still returned."""
        graph = build_agent_graph()

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DOC_QUERY", ["doc_retrieval_agent"]
            )),
            patch("agents.doc_retrieval.EmbeddingClient") as mock_embed_cls,
            patch("agents.doc_retrieval.search_vector_store", return_value=SAMPLE_CHUNKS),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock(
                "Low confidence answer here."
            )),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.60, False)),
        ):
            mock_embed = AsyncMock()
            mock_embed.embed = AsyncMock(return_value=[0.01] * 1536)
            mock_embed_cls.return_value = mock_embed

            result = await graph.ainvoke({**BASE_INPUT, "strict_validation": False})

        assert result["validation_status"] == "REJECT"
        assert result["final_response"] is not None           # soft mode: still returned


# ── Test: DATA_QUERY path ─────────────────────────────────────────────────────

class TestDataQueryPath:

    @pytest.mark.integration
    async def test_data_query_runs_sql_not_doc_retrieval(self):
        """DATA_QUERY: supervisor → sql_analytics → synthesizer → validator.

        Key patching insight: compiled LangGraph holds FUNCTION REFERENCES,
        not module-level names. Patching `graph.orchestrator.sql_analytics_node`
        does NOT affect the compiled graph. We must patch the INTERNAL components
        each node instantiates at call time (LLMClient, run_read_only_sql, etc).
        """
        graph = build_agent_graph()

        # Track that sql_analytics ran by watching run_read_only_sql being called
        sql_execution_calls = []

        def fake_run_sql(sql_query, db_url):
            sql_execution_calls.append(sql_query)
            return SAMPLE_SQL_ROWS

        _sql_mock = AsyncMock()
        _sql_mock.chat_json = AsyncMock(return_value={
            "sql_query": "SELECT region, metric_value FROM analytics_data WHERE tenant_id = 'corp_acme_01' LIMIT 100",
            "explanation": "Retrieves regional sales data",
        })

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DATA_QUERY", ["sql_analytics_agent"]
            )),
            # Patch SQL node internals (not the node function itself)
            patch("agents.sql_analytics.LLMClient", return_value=_sql_mock),
            patch("agents.sql_analytics._fetch_schema_context", return_value="TABLE analytics_data (tenant_id VARCHAR);"),
            patch("agents.sql_analytics.run_read_only_sql", side_effect=fake_run_sql),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock(
                "EMEA Q3 revenue was $14.2M [SQL]."
            )),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.97, True)),
        ):
            result = await graph.ainvoke(BASE_INPUT)

        assert result["intent"] == "DATA_QUERY"
        assert sql_execution_calls, "run_read_only_sql should have been called"
        assert result.get("doc_chunks") in (None, [])   # doc retrieval not run
        assert result["sql_results"] == SAMPLE_SQL_ROWS
        assert result["validation_status"] == "PASS"


# ── Test: HYBRID_QUERY path ───────────────────────────────────────────────────

class TestHybridQueryPath:

    @pytest.mark.integration
    async def test_hybrid_query_runs_both_agents(self):
        """HYBRID_QUERY: doc_retrieval → sql_analytics → synthesizer → validator.

        Patches internal components of BOTH nodes so we can verify both arms ran.
        """
        graph = build_agent_graph()

        # Track calls to internal components
        embed_calls = []
        sql_execution_calls = []

        async def fake_embed(text):
            embed_calls.append(text)
            return [0.01] * 1536

        def fake_run_sql(sql_query, db_url):
            sql_execution_calls.append(sql_query)
            return SAMPLE_SQL_ROWS

        mock_embed_instance = AsyncMock()
        mock_embed_instance.embed = AsyncMock(side_effect=fake_embed)

        _sql_mock = AsyncMock()
        _sql_mock.chat_json = AsyncMock(return_value={
            "sql_query": "SELECT region FROM analytics_data WHERE tenant_id = 'corp_acme_01' LIMIT 100",
            "explanation": "Regional sales data",
        })

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "HYBRID_QUERY", ["doc_retrieval_agent", "sql_analytics_agent"]
            )),
            # Doc retrieval internals
            patch("agents.doc_retrieval.EmbeddingClient", return_value=mock_embed_instance),
            patch("agents.doc_retrieval.search_vector_store", return_value=SAMPLE_CHUNKS),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            # SQL analytics internals
            patch("agents.sql_analytics.LLMClient", return_value=_sql_mock),
            patch("agents.sql_analytics._fetch_schema_context", return_value="TABLE analytics_data (tenant_id VARCHAR);"),
            patch("agents.sql_analytics.run_read_only_sql", side_effect=fake_run_sql),
            # Downstream
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock(
                "EMEA Q3 sales $14.2M [doc_policy_2026_v2][SQL], within risk bounds."
            )),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.99, True)),
        ):
            result = await graph.ainvoke(BASE_INPUT)

        assert result["intent"] == "HYBRID_QUERY"
        assert embed_calls,         "Doc retrieval (embed) should have been called"
        assert sql_execution_calls, "SQL analytics (run_read_only_sql) should have been called"
        assert result["doc_chunks"] == SAMPLE_CHUNKS
        assert result["sql_results"] == SAMPLE_SQL_ROWS
        assert result["validation_status"] == "PASS"


# ── Test: State propagation ───────────────────────────────────────────────────

class TestStatePropagation:

    @pytest.mark.integration
    async def test_conversation_id_preserved_throughout(self):
        """conversation_id must survive all node transitions unchanged."""
        graph = build_agent_graph()

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DOC_QUERY", ["doc_retrieval_agent"]
            )),
            patch("agents.doc_retrieval.EmbeddingClient") as mock_embed_cls,
            patch("agents.doc_retrieval.search_vector_store", return_value=[]),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock("No data.")),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(1.0, True)),
        ):
            mock_embed = AsyncMock()
            mock_embed.embed = AsyncMock(return_value=[0.0] * 1536)
            mock_embed_cls.return_value = mock_embed

            result = await graph.ainvoke({**BASE_INPUT, "conversation_id": "conv-UNIQUE-XYZ"})

        assert result["conversation_id"] == "conv-UNIQUE-XYZ"

    @pytest.mark.integration
    async def test_execution_metrics_accumulate_across_nodes(self):
        """execution_metrics dict must grow as each node contributes its latency."""
        graph = build_agent_graph()

        with (
            patch("agents.supervisor.LLMClient", return_value=_make_supervisor_mock(
                "DOC_QUERY", ["doc_retrieval_agent"]
            )),
            patch("agents.doc_retrieval.EmbeddingClient") as mock_embed_cls,
            patch("agents.doc_retrieval.search_vector_store", return_value=SAMPLE_CHUNKS),
            patch("agents.doc_retrieval.get_pool", return_value=AsyncMock()),
            patch("agents.synthesizer.LLMClient", return_value=_make_synthesizer_mock("Summary.")),
            patch("agents.validator.LLMClient", return_value=_make_validator_mock(0.97, True)),
        ):
            mock_embed = AsyncMock()
            mock_embed.embed = AsyncMock(return_value=[0.01] * 1536)
            mock_embed_cls.return_value = mock_embed

            result = await graph.ainvoke(BASE_INPUT)

        metrics = result.get("execution_metrics", {})
        assert "supervisor_ms" in metrics,     "Supervisor should record latency"
        assert "doc_retrieval_ms" in metrics,  "Doc retrieval should record latency"
        assert "synthesizer_ms" in metrics,    "Synthesizer should record latency"
        assert "validator_ms" in metrics,      "Validator should record latency"
        assert metrics.get("retrieved_chunks", 0) == len(SAMPLE_CHUNKS)


# ── Test: Supervisor routing functions ────────────────────────────────────────

class TestRoutingLogic:
    """Test the edge routing functions in isolation — pure Python, no LLM."""

    @pytest.mark.unit
    def test_route_doc_query_to_doc_retrieval(self):
        from graph.orchestrator import _route_from_supervisor
        state = {"intent": "DOC_QUERY"}
        assert _route_from_supervisor(state) == "doc_retrieval"

    @pytest.mark.unit
    def test_route_data_query_to_sql_analytics(self):
        from graph.orchestrator import _route_from_supervisor
        state = {"intent": "DATA_QUERY"}
        assert _route_from_supervisor(state) == "sql_analytics"

    @pytest.mark.unit
    def test_route_hybrid_query_starts_with_doc(self):
        from graph.orchestrator import _route_from_supervisor
        state = {"intent": "HYBRID_QUERY"}
        assert _route_from_supervisor(state) == "doc_retrieval"

    @pytest.mark.unit
    def test_route_error_state_falls_back_to_doc(self):
        from graph.orchestrator import _route_from_supervisor
        state = {"intent": "DATA_QUERY", "error": "Supervisor LLM failed"}
        assert _route_from_supervisor(state) == "doc_retrieval"

    @pytest.mark.unit
    def test_after_doc_retrieval_hybrid_goes_to_sql(self):
        from graph.orchestrator import _route_after_doc_retrieval
        state = {"intent": "HYBRID_QUERY"}
        assert _route_after_doc_retrieval(state) == "sql_analytics"

    @pytest.mark.unit
    def test_after_doc_retrieval_doc_query_goes_to_synthesizer(self):
        from graph.orchestrator import _route_after_doc_retrieval
        state = {"intent": "DOC_QUERY"}
        assert _route_after_doc_retrieval(state) == "synthesizer"
