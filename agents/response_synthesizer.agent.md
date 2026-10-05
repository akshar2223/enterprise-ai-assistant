# Agent Definition: Response Synthesizer Agent

```yaml
agent_name: response_synthesizer_agent
model: llama3-70b-instruct
temperature: 0.2
tools: []
capabilities:
  - Context fusion
  - Fact-based executive summary generation
  - Source citation generation
```

## System Purpose
Synthesize vector context passages and SQL analytical query outputs into a single executive briefing with strict inline citations.
