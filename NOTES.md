# Project Notes — LLM Output Arbitration System

## Design decisions (and why)

- **Pydantic + instructor as the foundation.** A "critic" is just an LLM call
  whose output is forced into a typed Pydantic model. Proving one typed critique
  came back from one model was ~60% of the conceptual work.
- **Enums for `severity` and `dimension`, not free strings.** Constrained
  vocabularies so critiques are comparable across critics — the disagreement
  detector (Phase 2) and adjudicator (Phase 3) depend on a shared vocabulary.
- **Field bounds (`ge`/`le`) on score and confidence.** instructor auto-re-prompts
  on out-of-range values — free validation.
- **Field descriptions ARE the rubric.** The model grades against the
  `Field(description=...)` text, so wording there directly drives critic quality.
- **Single `run_critic` function, three role prompts.** The only thing that
  differs between critics is the system prompt. Same schema, same call. This is
  what keeps the multi-provider design clean.
- **Project structure: models.py (schemas only), critics.py (logic + prompts),
  run_demo.py (entry point).** A file's name is a promise about its contents;
  models.py imports no API libraries.

## Key observation (the project's thesis)

Ran all three critics on one output (1929 crash, with a planted factual error,
a planted logic flaw, and a missing half-answer). They diverged exactly as hoped:
- Accuracy critic → caught the Federal Reserve date error, ignored the logic leap.
- Logic critic → caught the WWII non-sequitur and an internal contradiction,
  ignored the factual date error.
- Completeness critic → ignored both specific errors, flagged the missing
  New Deal / Glass-Steagall / SEC content; scored 2 not 1 (partial attempt).

Specialized critics catch *categorically different* failure modes. Single-model
self-evaluation would share blind spots; the combination is comprehensive.

## Design insight (arrived at empirically)

All three flagged the same Fed sentence but for *different reasons* (wrong date /
internal contradiction / thin response). This overlap-with-different-reasoning is
exactly what the disagreement detector and adjudicator must untangle — recognizing
"three lenses on one span" vs. "three separate problems."

## Known limitations / TODO

- Completeness critic needs the *original question* to judge gaps properly;
  `run_critic` currently only passes the output. Extend it to optionally accept
  the question (needed for Phase 3 adjudicator anyway).

## Phase 2 — LangGraph parallel orchestration

### What changed
Converted the three sequential `run_critic` calls into a parallel LangGraph
pipeline. `run_demo.py` no longer loops over prompts — it just invokes the graph
and reads the result. Orchestration now lives in `graph.py`; the entry point no
longer knows how many critics exist or how they're coordinated.

### Design decisions
- **State as a TypedDict (`ArbitrationState`)** with `question`, `output`, and
  `critiques`. Nodes read/write one shared state object rather than passing
  return values down a call chain.
- **Reducer on the `critiques` field** — `Annotated[list[Critique], add]`. The
  three critic nodes write to the same field in parallel; without a reducer the
  concurrent writes would overwrite each other. `operator.add` concatenates, so
  each node must return its critique wrapped in a list (`{"critiques": [c]}`),
  and the state seeds as `[]` so the first `[] + [c]` works.
- **Fan-out / fan-in structure.** `START` → three critic nodes (parallel) →
  collector node (fan-in) → `END`. Each critic has one edge in from START and
  one edge out to the collector. The collector waits for all three before firing.
- **`run_critic` extended with optional `question` param** to resolve the
  Phase 1 completeness-needs-question limitation. Only the completeness node
  passes it; accuracy and logic don't, so they didn't change.
- **Collector is a passthrough for now** (`return {}`). It's the placeholder for
  the disagreement detector (next) and the Phase 3 adjudicator.

### Observation (confirms real parallelism)
Critique return order changed from the Phase 1 sequential order (accuracy, logic,
completeness) to a non-deterministic order (accuracy, completeness, logic). The
reordering is evidence the three nodes ran concurrently and fanned in as each API
call returned — not a sequential loop. This is *why* the reducer is necessary.

### Next
- Disagreement detector in the collector: compare the three critiques — severity
  gaps >2, issues one critic caught that others missed, and overlap on the same
  span for different reasons (e.g. all three flagged the Federal Reserve sentence
  here, but for accuracy / logic / completeness reasons respectively).

## Status
- [x] Phase 1: three specialized critics with divergent failure detection
- [x] Phase 2: LangGraph parallel orchestration (reducer-merged critique list)
- [ ] Phase 2b: disagreement detector in the collector node
- [ ] Phase 3: adjudicator agent