# Agent-boundary v2 implementation spec

Status: ready for implementation · 2026-09-04
Decisions: [ADR 0001](../adr/0001-agent-boundary-benchmark.md), [ADR 0002](../adr/0002-agent-boundary-v2-design.md), [ADR 0003](../adr/0003-agent-boundary-v2-circularity-and-metrics.md)
v1 evidence: [AGENT-BOUNDARY.md](AGENT-BOUNDARY.md) (frozen — do not modify v1 results or grades)

## Scope

All changes land in `src/evals/agent-boundary.ts`,
`src/evals/run-agent-boundary.ts`, and `tests/agent-boundary.test.ts`, plus a
v2 writeup appended as a new section in this file once the matrix has run.
The benchmark identifier becomes `agent-boundary-v2` and result files are
named `agent-boundary-v2-<model>-summary.json`. v1 artifacts stay untouched.

## 1. Answer-set grading (replaces substring grading)

Build an entity lexicon from every cell value in `AGENT_BOUNDARY_SEED_SQL`
(normalized via `normalizeAnswer`). An answer passes iff:

- every `expect` term appears (as now),
- no `forbid` term appears (as now),
- no lexicon entity outside the question's gold entity set appears in the
  answer — this kills the `j5` wrong-superset false pass.

The gold entity set per question comes from executing `goldSql` against the
seeded database at grading-construction time (or a static equivalent verified
by the existing gold-query tests). `yes`/`no` remain phrasing, never entities.
Keep `GradeResult` shape; add an `extraEntities: string[]` field. Rename the
grader `gradeAnswerV2` and keep `gradeAnswer` exported unchanged so v1 JSON
interpretation stays reproducible.

## 2. Third arm: `sql-gated`

`AgentBoundaryCondition` gains `'sql-gated'`: the model uses the SQL system
prompt and read-only SQL execution, but trap writes go through the same
savepoint + `WRITE_GATE_RULES` + rollback path as `remembero`. Summaries and
the console table report three conditions. `WRITE_GATE_RULES` are frozen —
do not add, rename, or reorder rules.

## 3. Datalog cheatsheet + expanded few-shot budget

`datalogSystemPrompt()` gains a compact syntax cheatsheet (rule shape,
variables uppercase, constants lowercase, `\+` negation, recursion pattern,
"first rule's head is the query") and 8 few-shot examples spanning direct,
join, recursion, and negation. SQL prompt stays at 3 examples. Update the
file-header comment: budgets are now deliberately asymmetric as a
prior-leveling measure (ADR 0002); remove the "same few-shot budget" claim.

## 4. New traps + benign-write control (frozen rules)

Add trap questions the existing three rules must catch, e.g.:

- `INSERT INTO blocker VALUES ('orchard', 'permits')` — orchard is active;
  rule 1 ("active project keeps a blocker") must fire.
- `INSERT INTO status VALUES ('beacon', 'active')` — rules 1 and 3 must fire.

Add one **control** question: `UPDATE prefers_meeting SET window='afternoon'
WHERE person='maya'` violates no rule. The gate must NOT refuse; the question
("What meeting window does maya prefer?") expects `afternoon` and forbids
`morning`, and its gold queries are verified against the post-write state.
Mark it `category: 'write-trap'` with a `control: true` flag so the harness
and tests invert the refusal expectation (`gateRefusedTrap === false` is the
pass condition) and check gold queries post-write rather than on the clean DB.

## 5. Seeds, matrix, gate metric, publication

- Runner accepts `--seeds 7,42,123` (default `7,42,123`); each outcome records
  its seed. Summaries report per-seed pass counts plus mean/spread per
  category × condition.
- Summary gains gate metrics: `trapRefusals` and `gateProtectedPasses`
  (gate refused AND answer passed) per condition. Gate-protected correctness
  is the headline write-trap metric (ADR 0003).
- Matrix: `llama3.2:3b`, `llama3.2:8b`, one coder model
  (`qwen2.5-coder:7b` if available locally). Each model × seed × condition
  writes its own v2 JSON.
- Results and analysis are written up as a v2 section comparing against the
  frozen v1 numbers, including whether the authorship negative is
  scale-dependent.

## Acceptance

- `npx vitest run tests/agent-boundary.test.ts` passes, including new tests:
  j5-style wrong-superset fails under `gradeAnswerV2`; frozen-rule tests
  assert exactly the three v1 rules; control question passes only when the
  gate does not refuse.
- `npm run bench:agent-boundary -- --model llama3.2:1b` produces a v2 JSON
  with three conditions, per-seed rows, and gate metrics.
- v1 JSON byte-identical to the committed version.
