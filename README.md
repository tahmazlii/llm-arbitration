# LLM Output Arbitration System

A multi-agent pipeline that **catches bad LLM outputs instead of generating new ones.** Any LLM-generated answer is routed to three specialized critic models running in parallel — each judging a different dimension — and an adjudicator weighs their findings, resolves their disagreements, and produces a single confidence-scored verdict.

The core idea: different models with different prompts have different blind spots, so their *disagreements* are the valuable signal. A factual-accuracy critic misses logical fallacies; a logic critic misses factual errors. Only the combination — plus an adjudicator to reconcile them — catches what single-model self-evaluation misses.

## Demo

<!-- Record a <60s screen recording (Cmd+Shift+5 on Mac) of pasting an output into
     frontend.html and getting a verdict, then drag the .mov/.mp4 file here.
     GitHub embeds video directly in READMEs. -->
*[demo recording here]*

## Architecture

```
                     ┌─→  Accuracy Critic      ─┐
   Output  ──→  START ─→  Logic Critic          ─┼─→  Collector      ─→  Adjudicator    ─→  Verdict
                     └─→  Completeness Critic   ─┘   (disagreement        (resolves
                          (parallel fan-out)          detection)           conflicts)
```

- **Three critics run in parallel** (LangGraph fan-out). Each returns a typed, structured critique — issues with quoted spans, severities, and a 1–5 score. Different prompts mean different blind spots.
- **Collector / disagreement detector** compares the three critiques: it computes the score spread and finds spans flagged by more than one critic, recording where their severities differ.
- **Adjudicator** receives the output, all three critiques, and the disagreement report, then reasons through each conflict — upholding real issues (with evidence) and dismissing overruled flags (with reasoning) — into a final 1–10 verdict. It runs on the strongest model, since this is the hardest reasoning in the pipeline.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Structured LLM output | Pydantic + [instructor](https://python.useinstructor.com/) |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) (parallel state graph with a reducer-merged critique list) |
| Models | Anthropic Claude (Sonnet for critics, Opus for adjudication) |
| API | FastAPI (async, SQLite-backed audit trail) |
| Frontend | Single-page HTML/JS verdict explorer |
| Deployment | Docker |

## Running it

**Option 1 — Docker (recommended):**

```bash
docker build -t llm-arbitration .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... llm-arbitration
```

**Option 2 — local Python:**

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or put it in a .env file
uvicorn src.arbitration.api:app --reload
```

Then open `http://localhost:8000/docs` for the interactive API, or open `frontend.html` in a browser to use the verdict explorer.

## API

- `POST /arbitrate` — submit a `question` and an `output`; returns a verdict with a unique id.
- `GET /arbitrations/{id}` — retrieve a past verdict from the audit trail.

## Example

Given a deliberately flawed answer about the 1929 stock market crash — with a planted factual error, a logical fallacy, and a missing half-answer — the three critics each catch different problems, and the adjudicator consolidates them:

- **Accuracy** flags that the Federal Reserve was created in 1913, not in response to the crash.
- **Logic** flags "the crash directly caused World War II" as a post hoc fallacy.
- **Completeness** flags the entirely missing account of the actual government response (the New Deal, Glass-Steagall, the SEC).

The adjudicator then confirms the factual and logical errors, **resolves a severity disagreement** between two critics with explicit reasoning, and **dismisses a redundant flag** as double-counting — producing a 2/10 verdict with a one-paragraph summary.

## Project structure

```
src/arbitration/
├── models.py    # Pydantic schemas (Critique, DisagreementReport, Verdict)
├── critics.py   # LLM calls + critic/adjudicator prompts
├── graph.py     # LangGraph orchestration + disagreement detector
└── api.py       # async FastAPI service
```

## Notes & limitations

- Span-overlap detection currently uses substring matching, which can over-group short quotes; semantic matching is a planned improvement.
- The pipeline runs synchronously inside the async API via a threadpool; making it natively async is a future refinement.
- `allow_origins=["*"]` is set for local development and would be restricted to a specific origin in production.

See [NOTES.md](NOTES.md) for the full design decisions and reasoning behind each phase.
