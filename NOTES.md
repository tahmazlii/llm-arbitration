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

## Status
- [x] Phase 1: three specialized critics with divergent failure detection
- [ ] Phase 2: LangGraph parallel orchestration (watch the reducer gotcha:
      parallel writes to one list need `Annotated[list, operator.add]`)