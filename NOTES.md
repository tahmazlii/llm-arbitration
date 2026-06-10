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

## Phase 2b — Disagreement detector

### What it does
The `collector_node` is no longer a passthrough. It reads the three assembled
critiques and produces a typed `DisagreementReport` capturing disagreement at
two levels:
- **Critique level:** `score_spread` (max − min of the three overall scores),
  plus `highest_dimension` / `lowest_dimension` for context.
- **Issue level:** `overlaps` — spans flagged by more than one critic, each
  recording the critics involved and their per-issue severities.

### Schema (in models.py)
Three models, nested like Critique/Issue:
- `CriticFlag` (dimension + severity) — one critic's take on a span.
- `SpanOverlap` (span + list of CriticFlags) — one span multiple critics flagged.
- `DisagreementReport` (score_spread, highest/lowest dimension, list of overlaps).

### How overlaps are detected
1. Flatten all issues across critiques into flags: (dimension, quote, severity).
2. Group flags whose quotes match. Match = exact OR one quote is a substring of
   the other (`quotes_match` helper).
3. Keep only groups with flags from 2+ distinct dimensions (single-critic groups
   aren't disagreements).
4. Convert each surviving group into a SpanOverlap.

Added `disagreements: NotRequired[DisagreementReport]` to ArbitrationState — no
reducer (only the collector writes it, once). NotRequired because it doesn't
exist until the collector runs, so it shouldn't be required at invoke time.

### What it surfaced on the 1929 test (the payoff)
- **WWII span** — flagged by all three, but severities differ (accuracy high,
  logic high, completeness medium): a severity disagreement.
- **Federal Reserve span** — all three at high: the "three lenses on one span"
  case (wrong date / internal contradiction / thin response), now auto-detected.
- **Causes span** — accuracy low vs. completeness high: a genuine substantive
  disagreement the adjudicator will need to resolve. The detector found this
  without any hand-coding for that specific span — the system working as intended.

### Known limitations (for future improvement)
- **Substring matching is crude.** Over-groups on short quotes; a future version
  could use semantic/fuzzy matching, possibly an LLM judge for "same problem?".
- **Intra-critic duplicate flags.** A single critic can contribute multiple flags
  to one overlap if it wrote two issues quoting the same span (logic did this on
  the WWII span — appeared twice). The 2+-distinct-dimensions filter still works
  correctly, but the flags list can contain per-critic duplicates. Could dedupe
  per dimension (e.g. keep highest severity per critic) later.
- **Score ties** in highest/lowest dimension resolve to whichever max/min hits
  first. Fine for now.

## Phase 3 — Adjudicator agent

### What it does
A fourth node that runs *after* the collector (sequential, not parallel — it
needs the disagreements). It receives the original output, the three critiques,
and the DisagreementReport, reasons through the conflicts, and produces a typed
`Verdict`. This is the layer that turns three opinions into one judgment.

### Schema (Verdict, in models.py)
- `quality` (1-10, holistic — note the different scale from critics' 1-5)
- `confidence` (0-1)
- `confirmed_issues` (list of `ConfirmedIssue`: quote, description, the
  dimension(s) that raised it, the adjudicator's final severity, and the reason
  it was upheld)
- `dismissed_flags` (list of `DismissedFlag`: the span, the critic's original
  flag, and the reason for overruling) — the field that makes this adjudication
  rather than averaging
- `summary` (one-paragraph plain-language assessment)

### Implementation
- Dedicated `run_adjudicator(output, critiques, report, question, model)` in
  critics.py — not folded into run_critic, since inputs/output differ enough.
- Critiques and report are serialized into the prompt with `model_dump_json` —
  the adjudicator is an LLM, so its input has to be text; this is the bridge
  from typed objects back to text.
- Runs on `claude-opus-4-7` (not Sonnet like the critics): the strongest model
  goes where the hardest reasoning is — weighing conflicting evidence.
- Wired in graph: collector → adjudicator → END (replaced collector → END).
  Added `verdict: NotRequired[Verdict]` to state (no reducer, written once).

### What it demonstrated on the 1929 test (the payoff)
- **Consolidated overlaps, not averaged them.** Folded the WWII span's accuracy
  + logic + completeness flags into one confirmed issue, noting both the factual
  error and the post hoc fallacy as the same underlying defect (the "three
  lenses on one span" insight, now automatic).
- **Dismissed a flag with reasoning.** Overruled completeness's separate
  "Hoover-era inaction" flag as double-counting an already-confirmed issue. A
  real adjudication decision, not aggregation.
- **Resolved the severity disagreement.** The causes sentence (accuracy=low vs.
  completeness=high) was confirmed at *medium*, with explicit justification:
  legitimate causes named, so omission not error. Reasoned to a judgment and
  defended it — the project's whole thesis in one field.

## Status
- [x] Phase 1: three specialized critics with divergent failure detection
- [x] Phase 2: LangGraph parallel orchestration (reducer-merged critique list)
- [x] Phase 2b: disagreement detector (score spread + span overlaps)
- [x] Phase 3: adjudicator agent producing reasoned verdicts
- [ ] Phase 4: Verdict Explorer UI
- [ ] Phase 5: FastAPI service + analytics
- [ ] Phase 6: portfolio polish (test cases, README narrative, diagram)