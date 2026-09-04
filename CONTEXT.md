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
  back if any rule derives a violation. Raw SQL has no equivalent step.
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
