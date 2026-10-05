# Enterprise AI Knowledge & Decision Assistant — Requirements Document (PRD/TRD)

## 1. Executive Summary & Business Objectives
The Enterprise AI Knowledge & Decision Assistant is an autonomous, context-aware intelligence platform designed to unify enterprise data silos (unstructured documents, tabular relational data, and analytical data lakes) into an actionable decision engine.

### Key Performance Indicators (KPIs) & Target SLAs
* **Query Latency:** P95 < 2.5s for RAG search; P95 < 4.0s for multi-hop agent orchestration (SQL + RAG + Validation).
* **Accuracy & Groundedness:** Faithfulness score > 0.96 (via RAGAS metric frame); Hallucination rate < 1.5%.
* **Scale Capability:** Ingestion pipeline capacity of 100K+ enterprise records/hour with distributed scale out to 10M+ documents.
* **Cost Efficiency:** Token compression & dynamic model routing reducing API token expenditures by 42%.

---

## 2. Peer Engineering Debate & Trade-off Matrix

| Topic | Staff Peer Argument | Final Decision & Rationale |
| :--- | :--- | :--- |
| **Data Engine (PySpark vs. Polars/Pandas)** | "100K records can easily fit in memory using Polars with lower latency than PySpark overhead." | **PySpark selected.** While 100K is the initial baseline, enterprise requirements mandate seamless scale to 100M+ corporate data lake logs without pipeline re-architecting. |
| **Model Hosting (Self-hosted LLaMA 3 vs. OpenAI)** | "OpenAI API gives faster time-to-market and higher reasoning benchmark scores." | **Hybrid Strategy.** LLaMA 3 70B (via vLLM) on local GPU nodes for PII-sensitive internal SQL/Doc tasks; OpenAI GPT-4o fallback for complex multi-step reasoning. |
| **Vector DB (pgvector vs. Dedicated Qdrant)** | "Dedicated vector databases like Qdrant provide superior HNSW index filtering and QPS." | **PostgreSQL + pgvector.** Unified ACID transactional boundaries for metadata + vectors. Keeps operational complexity low for v1. |

---

## 3. Functional Requirements

### FR-1: Unstructured Document Ingestion & RAG
* **FR-1.1:** Support multi-format ingestion (PDF, DOCX, Markdown, HTML, Plain Text).
* **FR-1.2:** Semantic chunking using dynamic layout analysis with overlap factor (512 tokens with 64 token stride).
* **FR-1.3:** Hybrid retrieval combining dense vector embeddings (`text-embedding-3-large` / `bge-large-en-v1.5`) and sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF).

### FR-2: Tabular & Structured Analytics Pipeline
* **FR-2.1:** Execute PySpark ingestion jobs processing >100K enterprise records, normalizing schema and loading into PostgreSQL analytics schemas.
* **FR-2.2:** Text-to-SQL engine capable of executing read-only parameterized queries against enterprise database tables with automatic AST schema validation and syntax correction.

### FR-3: Multi-Agent Orchestration & Validation
* **FR-3.1:** Central Supervisor Agent dynamically routes user intent between Document Retrieval, SQL Analytics, and Combined Reasoning.
* **FR-3.2:** Independent Validation Agent verifies LLM outputs against source context for hallucination, citation accuracy, and policy compliance prior to rendering response.

---

## 4. Non-Functional Requirements (NFRs)

### NFR-1: Security & Compliance
* **RBAC & ABAC:** Strict row-level and document-level access control filters passed directly into Vector DB and SQL query contexts.
* **Data Privacy:** PII masking filter (Regex + Named Entity Recognition) executed on input prompts and tool inputs before sending to external model APIs.

### NFR-2: Observability & Auditability
* **Traceability:** OpenTelemetry integration tracing every step from agent routing to tool execution and raw context retrieval.
* **Audit Trail:** Immutable append-only transaction logs stored in PostgreSQL for every SQL query generated, vector hit score, and hallucination evaluation result.
