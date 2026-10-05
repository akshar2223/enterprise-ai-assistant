"""
Shared pytest fixtures for the Enterprise AI Assistant test suite.

Fixtures are scoped to avoid redundant setup:
  - session scope:  heavy objects (compiled graph) — built once per test run
  - function scope: stateful mocks — fresh per test (default)
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock  # noqa: F401 — re-exported for test modules

import pytest

# ── Force test environment before any app imports ────────────────────────────
# This ensures config.py reads these values instead of real .env secrets.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:test@localhost:5432/test_db")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-minimum-32-characters!!")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("OPENAI_CHAT_MODEL", "gpt-4o")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


# ── Reusable sample data ──────────────────────────────────────────────────────

@pytest.fixture
def sample_doc_chunks():
    return [
        {
            "chunk_id": "c71a39f0-281a-4f3b-9e12-123456789abc",
            "doc_id": "doc_policy_2026_v2",
            "content": "EMEA Q3 regional sales reached $14.2M, representing a 12% year-over-year increase. Strategic risk assessment shows moderate exposure in eastern markets.",
            "metadata": {"source": "annual_report", "page": 42},
            "rrf_score": 0.892,
            "dense_score": 0.871,
            "sparse_score": 0.634,
        },
        {
            "chunk_id": "d82b50a1-392c-5g4d-af23-234567890bcd",
            "doc_id": "doc_risk_framework_v3",
            "content": "Operational risk threshold for EMEA is defined as 15% variance from quarterly baseline. Current exposure is within acceptable bounds.",
            "metadata": {"source": "risk_framework", "page": 7},
            "rrf_score": 0.741,
            "dense_score": 0.729,
            "sparse_score": 0.512,
        },
    ]


@pytest.fixture
def sample_sql_results():
    return [
        {"region": "EMEA", "period": "2026-Q3", "metric_value": 14200000.0, "metric_name": "sales"},
        {"region": "APAC", "period": "2026-Q3", "metric_value": 9800000.0, "metric_name": "sales"},
    ]


@pytest.fixture
def sample_agent_state(sample_doc_chunks, sample_sql_results):
    return {
        "query": "What were the Q3 EMEA sales figures and how do they compare to our risk policy?",
        "tenant_id": "corp_acme_01",
        "user_id": "usr_99823",
        "conversation_id": "conv_test_001",
        "strict_validation": True,
        "intent": "HYBRID_QUERY",
        "execution_plan": ["doc_retrieval_agent", "sql_analytics_agent"],
        "routing_reasoning": "Query requires both document policy context and sales data.",
        "doc_chunks": sample_doc_chunks,
        "sql_results": sample_sql_results,
        "sql_query": "SELECT region, period, metric_value FROM analytics_data WHERE tenant_id = 'corp_acme_01' LIMIT 100;",
        "sql_explanation": "Retrieves regional sales metrics for all periods.",
        "draft_response": "EMEA Q3 sales reached $14.2M [doc_policy_2026_v2], within risk thresholds [doc_risk_framework_v3].",
        "citations": [
            {"source_id": "doc_policy_2026_v2", "chunk_id": "c71a39f0-281a", "relevance_score": 0.892},
        ],
        "validation_status": "PASS",
        "faithfulness_score": 0.98,
        "unsupported_claims": [],
        "final_response": "EMEA Q3 sales reached $14.2M [doc_policy_2026_v2], within risk thresholds [doc_risk_framework_v3].",
        "execution_metrics": {
            "supervisor_ms": 210,
            "doc_retrieval_ms": 380,
            "sql_analytics_ms": 490,
            "synthesizer_ms": 620,
            "validator_ms": 140,
            "retrieved_chunks": 2,
            "sql_queries_run": 1,
        },
    }


@pytest.fixture
def fake_query_embedding():
    """1536-dim zero vector — valid shape for pgvector tests."""
    return [0.01] * 1536
