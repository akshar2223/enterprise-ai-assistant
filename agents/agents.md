# Multi-Agent Architecture & Topology

## 1. Agent Orchestration Overview
The system implements a Stateful Directed Acyclic Graph (DAG) multi-agent system powered by FastAPI and custom agent runtimes.

```
                  +--------------------------+
                  |     Supervisor Agent     |
                  +------------+-------------+
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
+------------------+                       +------------------+
| Doc Retrieval    |                       | SQL Analytics    |
| Agent            |                       | Agent            |
+--------+---------+                       +--------+---------+
         |                                           |
         +---------------------+---------------------+
                               |
                               v
                  +--------------------------+
                  |  Synthesizer Agent       |
                  +------------+-------------+
                               |
                               v
                  +--------------------------+
                  | Validation Agent         |
                  +--------------------------+
```

## 2. Agent Roles
1. **Supervisor Router Agent (`supervisor_agent`):** Classifies requests and directs routing state transitions.
2. **Document Retrieval Agent (`doc_retrieval_agent`):** Executes hybrid vector and keyword retrieval.
3. **SQL Analytics Agent (`sql_analytics_agent`):** Inspects database schemas and runs AST-validated SQL queries.
4. **Response Synthesizer Agent (`response_synthesizer_agent`):** Combines structured data and unstructured text into a single cohesive response.
5. **Validation Guardrail Agent (`validation_guardrail_agent`):** Evaluates factual accuracy and compliance before response delivery.
