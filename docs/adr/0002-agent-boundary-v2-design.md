# ADR 0002: Agent-boundary v2 design — prior leveling, sql-gated arm, versioning, model matrix

Date: 2026-09-04
Status: Accepted

## Context

ADR 0001 split the benchmark's claims, mandated disclosure of the prior
confound, publication of the negative 1b result, and answer-set grading. That
left four design questions: how to level the prior, whether to isolate the gate
from the query language, how to version the benchmark while its design changes,
and which models to run.

## Decision

1. **Prior-leveling arm: cheatsheet + more shots.** The Datalog condition
   receives a compact syntax cheatsheet and a larger few-shot budget than SQL
   (e.g. 8 vs 3). The "same few-shot budget" symmetry from v1 is abandoned as
   false fairness: identical in-context budgets do not equalize a large
   training-data prior. Constrained decoding and structured-intent arms are
   deferred.
2. **Third arm: `sql-gated`.** Model authors SQL; writes pass through the same
   integrity-rule gate as the Remembero condition. This isolates the write-gate
   mechanism from the Datalog query language. If `sql-gated` matches
   `remembero` on write-traps, the supported claim is "the gate, not the
   language" — which survives the prior confound.
3. **Versioning: freeze v1.** The committed v1 results JSON keeps its original
   substring grades, labeled as such. All design changes (grading, arms,
   cheatsheet) land as `agent-boundary-v2`. No regrading of v1.
4. **Model matrix: publish 1b now, matrix on v2.** The llama3.2:1b analysis is
   published immediately against v1. v2 then runs a matrix — llama3.2:3b,
   llama3.2:8b, and one coder model — to test whether the negative
   query-authorship result is scale-dependent.

## Consequences

- `AgentBoundaryCondition` grows a third value (`sql-gated`); summaries report
  three columns.
- Few-shot budgets become condition-specific; the benchmark header comment must
  stop claiming identical budgets and instead document the asymmetry and why.
- v1 evidence is immutable; any future claim drift is visible as v1-vs-v2 diff.
