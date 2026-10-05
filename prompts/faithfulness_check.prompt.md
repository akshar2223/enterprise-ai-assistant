# Faithfulness & Groundedness Evaluation Prompt

Target Answer to Validate:
{{ draft_response }}

Ground Truth Source Context:
{{ source_context }}

Task:
Evaluate whether every statement in Target Answer is grounded in Ground Truth Source Context.

Return JSON:
{
  "faithfulness_score": 1.0,
  "is_grounded": true,
  "unsupported_claims": []
}
