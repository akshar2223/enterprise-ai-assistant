# Enterprise AI Knowledge & Decision Assistant — System Specifications

## 1. End-to-End System Architecture

```
                                    +-----------------------------------+
                                    |        Client Application         |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                    +-----------------+-----------------+
                                    |     FastAPI API Gateway           |
                                    | (JWT, RBAC, Rate Limiting)        |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                    +-----------------+-----------------+
                                    |   Supervisor / Router Agent       |
                                    +--------+----------------+---------+
                                             |                |
                       +---------------------+                +---------------------+
                       |                                                            |
                       v                                                            v
        +--------------+---------------+                            +---------------+--------------+
        |   Doc Retrieval Agent        |                            |     SQL Analytics Agent      |
        +--------------+---------------+                            +---------------+--------------+
                       |                                                            |
        +--------------+---------------+                            +---------------+--------------+
        |  Hybrid Vector Search Engine |                            |    PySpark Batch/Stream ETL  |
        |  (pgvector / BM25 RRF)       |                            |    & PostgreSQL Storage      |
        +--------------+---------------+                            +---------------+--------------+
                       |                                                            |
                       +---------------------+                +---------------------+
                                             |                |
                                             v                v
                                    +--------+----------------+---------+
                                    |  Response Synthesizer Agent       |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                    +-----------------+-----------------+
                                    |   Hallucination & Guardrail Gate  |
                                    |   (RAGAS / AST Validator)         |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                    +-----------------+-----------------+
                                    |  Validated Executive Response     |
                                    +-----------------------------------+
```

---

## 2. Database Schemas (`pgvector`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE enterprise_embeddings (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_enterprise_embeddings_hnsw 
ON enterprise_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_enterprise_embeddings_tenant 
ON enterprise_embeddings (tenant_id);
```

---

## 3. API Contract Specifications

### `POST /api/v1/chat/completions`

**Request:**
```json
{
  "tenant_id": "corp_acme_01",
  "user_id": "usr_99823",
  "query": "What were the Q3 regional sales numbers for EMEA, and how do they compare against our strategic policy risks?",
  "strict_validation": true
}
```

**Response:**
```json
{
  "conversation_id": "conv_883219",
  "response": "EMEA Q3 sales reached $14.2M (+12% YoY)...",
  "citations": [
    {
      "source_id": "doc_policy_2026_v2",
      "chunk_id": "c71a39f0-281a",
      "relevance_score": 0.892
    }
  ],
  "execution_metrics": {
    "sql_queries_run": 1,
    "retrieved_chunks": 4,
    "faithfulness_score": 0.98,
    "latency_ms": 1840
  }
}
```
