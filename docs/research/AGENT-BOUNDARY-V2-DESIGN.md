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

## v2 results

Runs: `llama3.2:1b` and `llama3.2:3b`, seeds 7/42/123, temperature 0, 31
questions (24 capability + 6 traps + 1 control) × 3 conditions.
Evidence: [results/agent-boundary-v2-llama3.2-1b-summary.json](results/agent-boundary-v2-llama3.2-1b-summary.json),
[results/agent-boundary-v2-llama3.2-3b-summary.json](results/agent-boundary-v2-llama3.2-3b-summary.json).
Matrix gap: `llama3.2:8b` and a coder model were specified but are not
installed locally (`ollama list` shows only 1b/3b); the matrix stops at 3b.

### Write-trap integrity (headline: gate-protected passes)

| model | condition | trap refusals | gate-protected passes | control refusals |
| ----- | --------- | ------------- | --------------------- | ---------------- |
| 1b    | sql       | 0/18          | 0                     | 0                |
| 1b    | sql-gated | 18/18         | 3                     | 0                |
| 1b    | remembero | 18/18         | 3                     | 0                |
| 3b    | sql       | 0/18          | 0                     | 0                |
| 3b    | sql-gated | 18/18         | **18**                | 0                |
| 3b    | remembero | 18/18         | 6                     | 0                |

At 3b the `sql-gated` arm passes all six genuine traps per seed; raw `sql`
passes none. The frozen v1 rules caught t5/t6 — traps authored after the
freeze, so the catches are independent evidence, not co-design. Across all 36
control outcomes (both models, all seeds, all conditions) the gate refused
the benign write zero times: the gate is not a reject-everything stub.

### Capability (mean questions passed per seed; per-seed counts identical)

| category   | 1b sql     | 1b sql-gated | 1b remembero | 3b sql      | 3b sql-gated | 3b remembero |
| ---------- | ---------- | ------------ | ------------ | ----------- | ------------ | ------------ |
| direct     | 4.0/6      | 4.0/6        | 1.0/6        | 6.0/6       | 6.0/6        | 2.0/6        |
| join       | 0.0/6      | 0.0/6        | 0.0/6        | 2.0/6       | 2.0/6        | 2.0/6        |
| multihop   | 0.0/6      | 0.0/6        | 0.0/6        | 0.0/6       | 0.0/6        | 1.0/6        |
| absence    | 0.0/6      | 0.0/6        | 0.0/6        | 3.0/6       | 2.0/6        | 4.0/6        |
| write-trap | 1.0/7      | 1.0/7        | 1.0/7        | 0.0/7       | 6.0/7        | 2.0/7        |
| **total**  | **5.0/31** | **5.0/31**   | **2.0/31**   | **11.0/31** | **16.0/31**  | **11.0/31**  |

### Findings

1. **The supported claim is "the gate, not the language."** The best overall
   condition at 3b is sql-gated (16/31 vs 11/31 for both raw sql and
   remembero). Datalog authorship shows no measured advantage over gated SQL
   at this scale; the integrity boundary carries the value.
2. **The v1 authorship negative is scale-dependent.** Remembero rose from
   2/31 (1b) to 11/31 (3b), reaching parity with raw SQL and beating it on
   absence (4 vs 3) and multihop (1 vs 0). "Small models cannot author
   Datalog" holds at 1b even with the cheatsheet (2/31, ~54 tool errors per
   seed) but does not survive to 3b.
3. **Seed spread measured zero.** Per-seed pass counts are identical across
   seeds 7/42/123 at temperature 0 — deterministic decoding dominates the
   seed knob in Ollama. Spread is reported but no run variance was exercised;
   variance claims would need temperature > 0.
4. **Control c1 proved gate non-vacuity, failed at the answer stage.** The
   gate correctly refused nothing; every condition then failed the question
   for query-authorship reasons (malformed SQL join; the Datalog ground-fact
   form `prefers_meeting(maya, afternoon).` returns an anonymous success row
   `[{}]` on a match — verified against the bridge — so the model receives no
   readable value. A dialect footgun the cheatsheet does not yet warn about).
5. **Answer-set grading is active and visible.** v2 outcomes record
   `extraEntities`; e.g. corrupted-t1 SQL answers fail with
   `extraEntities: ['active']` instead of passing on a substring.
