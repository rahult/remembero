# Context

Remembero domain glossary. Terms here are the canonical vocabulary; code, docs,
and evals should use them consistently. Decisions that lock a term are recorded
in `docs/adr/`.

## Terms

- **Proof-carrying memory** — the product thesis: stored facts/rules/constraints
  with deterministic Datalog evaluation and an auditable proof/source chain for
  every answer.
- **Agent boundary** — the interface through which an agent (LLM) reads from and
  writes to memory. Two boundaries are compared in the agent-boundary benchmark:
  raw read-only SQL vs Remembero-gated Datalog.
- **Condition** — one arm of the agent-boundary benchmark (`sql` or
  `remembero`). Same seeded database, same model, same few-shot budget; the
  model authors every query itself.
- **Write gate** — the Remembero write path: a proposed write is applied inside
  a savepoint, integrity rules (Datalog) are evaluated, and the write is rolled
  back if any rule derives a violation. Raw SQL has no equivalent step. The
  canonical implementation is the shipped `enforceIntegrityCandidate`
  primitive (strict mode) in `src/knowledge/enforcement.ts`; the agent-boundary
  benchmark routes its gated conditions through it, not a local reimplementation.
- **Trap write** — a deliberately corrupting write the benchmark harness pushes
  through each condition's write path before asking a question. Designed so raw
  SQL applies it silently while the write gate refuses it.
- **Gold query** — the reference `goldSql`/`goldDatalog` pair attached to every
  benchmark question. The test suite executes both and asserts they reproduce
  the question's expected terms, so the benchmark cannot drift into fiction.
- **Prior confound** — the measurement hazard that small models have seen orders
  of magnitude more SQL than Datalog in training, so a SQL-vs-Datalog
  query-authorship comparison measures training-data prior, not boundary
  quality. Must be disclosed and, where possible, leveled by design.
- **Answer-set grading** — grading a model's final natural-language answer by
  exact containment of the gold answer set: every expected value present, no
  extra named entities, no forbidden terms. Replaces loose substring matching
  that passed wrong-superset answers.
- **Reader** — the model that answers a question from retrieved memory (sessions,
  facts, notes). Distinct from the **writer**, the model that turns text into
  facts or claims at write time. One deployment may use different models for each.
- **Computed notes** — a deterministic block, produced by code with no model call,
  appended to the reader's context: temporal expressions in the user's own words
  resolved against the date they were said, distances and gaps between dated
  events, and quantities with units. Code output, never model output.
- **Structured evidence** — claims the writer produced (subject, predicate, object,
  validity interval, quoted source), resolved, dated and superseded by
  deterministic rules before the reader sees them. Model output checked by code;
  never to be confused with computed notes.
- **Arm** — one run of a benchmark under one setting. A **paired run** is two arms
  that differ in exactly one setting with identical retrieval, so a difference in
  score is attributable to that setting.
- **Reader contract** — the fixed shape of what a reader is given: which deterministic
  blocks are present and the byte budget. Distillation, training and evaluation share one
  contract, so a reader is always measured on the prompt it learned from. The contract fixes
  the prompt, not the retrieval depth; a paired run must hold both.
- **Lane** — which side of a memory layer the reader reads from. The **retrieval lane**
  gives the reader raw sessions the memory layer found; the **memories lane** gives it only
  the memory layer's own text. Rows in different lanes are not ranked against each other.
- **Context tier** — how much of a retrieved session the reader sees. Every retrieved
  session gets a short **abstract** built by code; only the highest-ranked few get their
  **full text**. Tiering replaces cutting every session to the same sliver.
- **Judge** — the model that grades a reader's answer against the gold answer.
  Judges differ in leniency, so a score always names its judge and rows in one
  table share one judge.
