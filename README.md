# LLM Output Arbitration System

A multi-agent pipeline that evaluates any LLM-generated output by routing it to
multiple specialized critic models — each checking a different dimension
(factual accuracy, logical consistency, completeness) — then synthesizing their
critiques into a single confidence-scored verdict.

**Status:** In development. Phase 1 complete (three specialized critics with
divergent failure detection). See [NOTES.md](NOTES.md) for design decisions.

## Stack
Python · Pydantic + instructor (typed LLM outputs) · Anthropic + Ollama ·
LangGraph (orchestration) · FastAPI (serving)
