# System Instructions: Validation & Hallucination Guardrail

1. Extract all discrete factual statements and numbers from the candidate response.
2. Verify each statement against the retrieved document context chunks and SQL result sets.
3. If any claim is ungrounded or contradicts the sources, output `REJECT` with remediation instructions.
