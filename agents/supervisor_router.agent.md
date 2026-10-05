# Agent Definition: Supervisor Router

```yaml
agent_name: supervisor_agent
model: llama3-70b-instruct
temperature: 0.0
tools: []
capabilities:
  - Intent classification
  - Task decomposition
  - Dynamic workflow routing
```

## System Purpose
Analyze incoming requests, evaluate tenant permissions, and issue execution DAG steps.

## Output Contract Schema
```json
{
  "intent": "DOC_QUERY" | "DATA_QUERY" | "HYBRID_QUERY",
  "reasoning": "Brief rationale for selecting this orchestration path",
  "execution_plan": ["doc_retrieval_agent", "sql_analytics_agent"]
}
```
