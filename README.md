# Enterprise AI Knowledge & Decision Assistant

This repository contains the complete architectural specification, multi-agent configurations, tool definitions, dynamic prompts, ETL pipelines, and API deployment code for the Enterprise AI Knowledge & Decision Assistant.

## Project Structure

```text
enterprise_ai_assistant/
├── docs/
│   ├── requirements.md
│   └── specifications.md
├── agents/
│   ├── agents.md
│   ├── supervisor_router.agent.md
│   ├── doc_retrieval.agent.md
│   ├── sql_analytics.agent.md
│   ├── response_synthesizer.agent.md
│   └── validation_guardrail.agent.md
├── skills/
│   ├── skills.md
│   ├── search_vector_store.py
│   ├── run_read_only_sql.py
│   └── validate_sql_ast.py
├── instructions/
│   ├── supervisor_router.instructions.md
│   ├── sql_analytics.instructions.md
│   └── validation_guardrail.instructions.md
├── prompts/
│   ├── rag_synthesis.prompt.md
│   ├── text_to_sql.prompt.md
│   └── faithfulness_check.prompt.md
├── etl/
│   └── spark_ingestion.py
├── deployment/
│   ├── Dockerfile
│   └── main.py
└── requirements.txt
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the FastAPI server: `uvicorn deployment.main:app --reload`
3. Execute spark ingestion: `python etl/spark_ingestion.py <input_json> <db_url>`
