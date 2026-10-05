# Enterprise AI Knowledge & Decision Assistant

> A production-grade, multi-agent RAG + SQL analytics platform with hallucination guardrails — built on LangGraph, pgvector, and OpenAI GPT-4o.

[![CI/CD](https://github.com/akshar2223/enterprise-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/akshar2223/enterprise-ai-assistant/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
User Query (POST /api/v1/chat/completions)
    │
    ▼  JWT auth · PII filter · Rate limit
┌─────────────────────┐
│   Supervisor Agent  │  ← GPT-4o, temp=0.0 → DOC_QUERY | DATA_QUERY | HYBRID_QUERY
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌──────────────┐
│   Doc    │ │     SQL      │
│ Retrieval│ │  Analytics   │
│  Agent   │ │    Agent     │
│          │ │              │
│ pgvector │ │ Text→SQL     │
│ BM25 RRF │ │ AST Validate │
│ Hybrid   │ │ Read-only    │
└────┬─────┘ └──────┬───────┘
     └───────┬───────┘
             ▼
    ┌─────────────────┐
    │   Synthesizer   │  ← GPT-4o, temp=0.2 → Executive summary + citations
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   Validator     │  ← GPT-4o, temp=0.0 → Faithfulness ≥ 0.96 → PASS/REJECT
    └────────┬────────┘
             ▼
    Validated Response
```

---

## Features

| Feature | Implementation |
|---|---|
| **Multi-agent DAG** | LangGraph stateful graph — Supervisor → Retrieval → SQL → Synthesizer → Validator |
| **Hybrid RAG** | pgvector dense (cosine) + PostgreSQL tsvector BM25 + Reciprocal Rank Fusion |
| **SQL Analytics** | Text-to-SQL via GPT-4o · AST validation (sqlglot) · read-only enforcement |
| **Hallucination gate** | LLM-as-judge faithfulness evaluation · 0.96 threshold · REJECT blocks response |
| **Multi-tenancy** | Row-level tenant isolation enforced in SQL · tenant_id from verified JWT |
| **PII protection** | 9 regex patterns (SSN, CC, email, phone, IP, IBAN, DOB, passport, API key) |
| **JWT auth** | Self-signed HS256 · secret ≥ 32 chars enforced · expiry validated |
| **Observability** | OpenTelemetry tracing · structlog JSON logging · append-only audit log |
| **ETL pipeline** | PySpark · tiktoken chunking · real OpenAI embeddings · PII masking |
| **Kubernetes** | HPA (2–10 pods) · non-root container · read-only filesystem · health probes |
| **CI/CD** | GitHub Actions · pytest → ruff → Docker build → kubectl rollout |

---

## Project Structure

```
enterprise_ai_assistant/
├── agents/                 # LangGraph agent node implementations
│   ├── state.py            # Shared TypedDict state schema
│   ├── supervisor.py       # Intent classification & DAG routing
│   ├── doc_retrieval.py    # Hybrid vector search agent
│   ├── sql_analytics.py    # Text-to-SQL analytics agent
│   ├── synthesizer.py      # Context fusion & response generation
│   └── validator.py        # Faithfulness guardrail agent
├── core/                   # Shared infrastructure
│   ├── auth.py             # JWT authentication
│   ├── audit.py            # Async append-only audit logger
│   ├── database.py         # asyncpg connection pool
│   ├── embeddings.py       # OpenAI embedding client (batched, retried)
│   ├── llm_client.py       # Abstracted LLM client (OpenAI / vLLM-swappable)
│   └── pii_filter.py       # Input PII sanitizer
├── graph/
│   └── orchestrator.py     # LangGraph DAG compilation & routing logic
├── skills/                 # Callable agent tools
│   ├── search_vector_store.py  # Hybrid RRF search (pgvector + BM25)
│   ├── run_read_only_sql.py    # Validated SQL execution
│   └── validate_sql_ast.py     # sqlglot AST safety checker
├── deployment/
│   ├── main.py             # FastAPI application (full pipeline wired)
│   └── Dockerfile          # Multi-stage, non-root, healthchecked
├── etl/
│   └── spark_ingestion.py  # PySpark document ingestion pipeline
├── db/
│   ├── init.sql            # Schema: embeddings + audit_logs + analytics
│   └── migrations/         # Alembic migration history
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml         # Template only — secrets injected via CI
│   ├── deployment.yaml     # API Deployment (2 replicas, HPA)
│   ├── service.yaml
│   ├── ingress.yaml        # Nginx + TLS (cert-manager)
│   ├── hpa.yaml            # CPU 70% / Memory 75% autoscaling
│   └── postgres-statefulset.yaml
├── tests/
│   ├── test_skills.py      # Unit tests (33) — SQL, PII, JWT, chunking
│   └── test_graph.py       # Integration tests (13) — LangGraph pipeline
├── .github/workflows/
│   └── ci.yml              # 4-job CI/CD pipeline
├── config.py               # Pydantic Settings (env-driven)
├── docker-compose.yml      # Local dev: PostgreSQL + pgvector
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key

### 1. Clone & configure

```bash
git clone https://github.com/akshar2223/enterprise-ai-assistant.git
cd enterprise-ai-assistant

cp .env.example .env
# Edit .env — set OPENAI_API_KEY, POSTGRES_PASSWORD, JWT_SECRET_KEY
```

### 2. Start the database

```bash
docker compose up -d db
# Wait ~10s for PostgreSQL + pgvector to initialise
```

### 3. Install dependencies & run

```bash
make install   # creates venv + pip install
make dev       # starts FastAPI on :8000 with hot-reload
```

### 4. Get a JWT token & make your first query

```bash
# Mint a dev token
make token

# Query the API
TOKEN="<paste token from above>"
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the Q3 EMEA sales figures and how do they compare to our risk policy?",
    "strict_validation": true
  }'
```

### 5. Run tests

```bash
make test          # all 46 tests
make test-unit     # unit tests only (fast, no I/O)
```

---

## Configuration

All config is driven by environment variables. Copy `.env.example` to `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `OPENAI_CHAT_MODEL` | — | `gpt-4o` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | — | `text-embedding-3-large` | Embedding model |
| `DATABASE_URL` | ✅ | — | PostgreSQL connection URL |
| `POSTGRES_PASSWORD` | ✅ | — | DB password |
| `JWT_SECRET_KEY` | ✅ | — | HS256 signing key (≥ 32 chars) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | — | `60` | Token TTL |
| `FAITHFULNESS_THRESHOLD` | — | `0.96` | Minimum score to PASS validation |
| `VECTOR_SEARCH_TOP_K` | — | `5` | Number of chunks to retrieve |
| `SQL_QUERY_MAX_ROWS` | — | `100` | SQL result row cap |
| `APP_ENV` | — | `development` | Set `production` to harden app |

> ⚠️ **Never commit `.env`** — it is gitignored. Use `.env.example` as the template.

---

## API Reference

### `POST /api/v1/chat/completions`

Requires: `Authorization: Bearer <token>`

**Request**
```json
{
  "query": "What were the Q3 EMEA sales and risk policy implications?",
  "strict_validation": true
}
```

**Response (PASS)**
```json
{
  "conversation_id": "conv_883219",
  "response": "EMEA Q3 sales reached $14.2M (+12% YoY) [doc_policy_2026_v2]...",
  "validation_status": "PASS",
  "citations": [
    { "source_id": "doc_policy_2026_v2", "chunk_id": "c71a...", "relevance_score": 0.892 }
  ],
  "execution_metrics": {
    "sql_queries_run": 1,
    "retrieved_chunks": 5,
    "faithfulness_score": 0.98,
    "latency_ms": 1840
  }
}
```

**Response (REJECT — strict mode)**
```
HTTP 422 Unprocessable Entity
{
  "detail": {
    "message": "Response rejected by the hallucination guardrail. The generated answer could not be fully grounded in retrieved sources.",
    "faithfulness_score": 0.71,
    "unsupported_claims": ["Sales figure $20M is not in source context"],
    "conversation_id": "conv_883219"
  }
}
```

### `POST /api/v1/auth/token` *(dev only — disabled in production)*

```json
{ "user_id": "usr_99823", "tenant_id": "corp_acme_01" }
```

### `GET /api/v1/health` · `GET /api/v1/health/db`

Liveness and readiness probes for Kubernetes.

---

## Testing

```
tests/test_skills.py   33 tests  — SQL AST validation, PII filter, JWT auth, chunking
tests/test_graph.py    13 tests  — LangGraph pipeline (all 3 routing paths + state propagation)
─────────────────────────────────────────────────────
Total                  46 tests  — zero external dependencies, ~1.5s
```

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Deployment

### Docker Compose (local / staging)

```bash
docker compose up
```

### Kubernetes (production)

```bash
# Apply namespace + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Inject secrets (replace with real values)
kubectl create secret generic enterprise-ai-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
  --from-literal=DATABASE_URL="$DATABASE_URL" \
  --namespace=enterprise-ai \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy all manifests — explicitly skip secret.yaml (already applied above via --dry-run pipe)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl rollout status deployment/enterprise-ai-api -n enterprise-ai
```

### CI/CD (GitHub Actions)

Push to `main` → tests → lint → Docker build → `kubectl set image` → rollout.

Required GitHub Secrets: `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, `JWT_SECRET_KEY`, `DATABASE_URL`, `KUBECONFIG`

---

## ETL — Document Ingestion

```bash
# Ingest documents from a JSON file
spark-submit etl/spark_ingestion.py \
  /path/to/documents.json \
  jdbc:postgresql://localhost:5432/enterprise_ai \
  corp_acme_01   # tenant_id
```

Input JSON format:
```json
[
  { "doc_id": "doc_policy_2026", "content": "...", "metadata": { "source": "policy" } }
]
```

---

## Security Notes

- **Tenant isolation**: `tenant_id` is always sourced from the verified JWT — never from the request body.
- **SQL safety**: All generated SQL is AST-validated by sqlglot before execution. Only `SELECT`, `UNION`, `INTERSECT`, `EXCEPT` are allowed.
- **PII masking**: User queries are sanitised with 9 regex patterns before reaching any external LLM API.
- **Secrets**: Never stored in code. All secrets are injected via environment variables or Kubernetes secrets.
- **Token endpoint**: `POST /api/v1/auth/token` is automatically disabled when `APP_ENV=production`.
- **Audit log**: Every pipeline execution is logged to an append-only PostgreSQL table (DELETE/UPDATE blocked at DB level).

---

## Roadmap

- [ ] vLLM local inference integration (swap via `OPENAI_BASE_URL`)
- [ ] Microsoft Presidio NER-based PII detection (v2 upgrade)
- [ ] Auth0 / AWS Cognito SSO integration
- [ ] RAGAS evaluation dashboard
- [ ] Streaming response support (`text/event-stream`)
- [ ] Document reranker (cross-encoder reranking after RRF)

---

## License

MIT — see [LICENSE](LICENSE).
