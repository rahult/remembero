# ADR 0001: Agent-boundary benchmark claims, confound handling, publication, and grading

Date: 2026-09-04
Status: Accepted

## Context

The agent-boundary benchmark (`src/evals/agent-boundary.ts`,
`src/evals/run-agent-boundary.ts`) compares two tool boundaries — model-authored
read-only SQL vs model-authored Datalog behind the Remembero write gate — over
the same seeded SQLite database with the same small model and few-shot budget.

The first run (llama3.2:1b, temperature 0, seed 7, 2 attempts) produced:

- SQL: 6/28 passed, 35 tool errors
- Remembero: 1/28 passed, 48 tool errors
- The single Remembero pass was write-trap `t1`, where the gate refused the
  trap write and the answer stayed correct despite a garbage model query.

The headline inverts a naive reading of the product thesis, the SQL-vs-Datalog
comparison is confounded by training-data prior, and substring grading passed
wrong-superset answers (e.g. `j5`).

## Decision

1. **Split the claim.** The benchmark supports two separate claims, reported
   separately:
   - *Write-gate integrity*: a rule-gated write boundary preserves truth where
     raw SQL silently corrupts. This is the product claim.
   - *Query-authorship capability*: at 1b scale, authoring Datalog is beyond
     the model; authoring SQL is marginal. Reported honestly as a negative
     result, not suppressed.
2. **Disclose and level the prior confound.** The writeup states plainly that
   SQL-vs-Datalog authorship measures training-data prior, and the benchmark
   gains at least one prior-leveling arm (design deferred to a follow-up).
3. **Publish the negative result.** The llama3.2:1b results JSON is committed
   under `docs/research/results/` with a written analysis covering the confound
   and the `t1` gate pass. No selective reporting.
4. **Tighten grading to answer-set match.** Replace substring expect/forbid
   grading with answer-set containment: all gold values present, no extra named
   entities, forbidden terms absent; `yes`/`no` remain special-cased as
   phrasing, not cell values.

## Consequences

- The benchmark can no longer be cited as "models query better through
  Datalog" — only the gate claim is supported at small-model scale.
- Grading changes alter pass/fail for existing outcomes; v1 results keep their
  original grades in the committed JSON and any regrade is labeled as such.
- Terminology locked in `CONTEXT.md`: agent boundary, condition, write gate,
  trap write, gold query, prior confound, answer-set grading.
