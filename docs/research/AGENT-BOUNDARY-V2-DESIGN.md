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
gated path as `remembero`. Summaries and
the console table report three conditions. `WRITE_GATE_RULES` are frozen —
do not add, rename, or reorder rules.

Post-publication hardening: the shared gate no longer reimplements its own
savepoint + violation-rule loop. `src/evals/agent-boundary-gate.ts` routes
gated writes through the product's shipped enforcement primitive
(`enforceIntegrityCandidate`, strict mode) over fact clauses materialized from
the seeded tables, with headless constraints derived mechanically from the
frozen `WRITE_GATE_RULES`. A test-enforced equivalence proof asserts the
product path refuses exactly what the frozen violation-headed rules refuse
for every trap and control, so gate evidence exercises the mechanism real
memory writes go through.

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

Runs: `llama3.2:1b`, `llama3.2:3b`, `llama3.1:8b`, and `qwen2.5-coder:7b`,
seeds 7/42/123, temperature 0, 31 questions (24 capability + 6 traps + 1
control) × 3 conditions. All four runs use the identical final harness —
the unified product enforcement gate, the ground-fact query guard, and the
cheatsheet including its bare-fact warning — so the JSONs are directly
comparable. Tag note: `llama3.2:8b` does not exist (the family stops at 3b);
`llama3.1:8b` is the 8B-scale substitute. The 1b/3b JSONs were regenerated
under the final harness, replacing intermediate results preserved in git
history; gate decisions are pinned by the test-enforced equivalence proof.
Evidence: [1b](results/agent-boundary-v2-llama3.2-1b-summary.json),
[3b](results/agent-boundary-v2-llama3.2-3b-summary.json),
[8b](results/agent-boundary-v2-llama3.1-8b-summary.json),
[coder](results/agent-boundary-v2-qwen2.5-coder-7b-summary.json).

### Write-trap integrity (headline: gate-protected passes)

Across the full matrix the gate refused **72/72** trap writes in both gated
conditions while raw `sql` refused **0/72**, and refused the benign control
write **0/36** times — non-vacuity holds everywhere. The frozen v1 rules
caught t5/t6, traps authored after the freeze, at every model. What scales
with the model is converting a refusal into a correct answer:

| model              | sql-gated gate-protected | remembero gate-protected |
| ------------------ | ------------------------ | ------------------------ |
| llama3.2:1b        | 3                        | 0                        |
| llama3.2:3b        | 18                       | 6                        |
| llama3.1:8b        | 12                       | 6                        |
| qwen2.5-coder:7b   | 18                       | 15                       |

The 8b point is below 3b: gate-protected correctness tracks the model's
query-authorship ability per family (llama3.2 1b→3b rises within-family;
llama3.1:8b and the coder are different families), not raw parameter count.

### Capability (mean questions passed per seed; per-seed counts identical)

| model              | sql      | sql-gated   | remembero  |
| ------------------ | -------- | ----------- | ---------- |
| llama3.2:1b        | 5.0/31   | 5.0/31      | 1.0/31     |
| llama3.2:3b        | 11.0/31  | **16.0/31** | 10.0/31    |
| llama3.1:8b        | 19.0/31  | **21.0/31** | 17.0/31    |
| qwen2.5-coder:7b   | 17.0/31  | **23.0/31** | 22.0/31    |

Category notes: absence is Datalog's clearest edge — remembero beats raw sql
at 3b (6 vs 3) and 8b (6 vs 4) and ties at coder scale (5 vs 5), consistent
with `\+` negation being more natural than LEFT JOIN / IS NULL patterns.
Multihop stays hardest for every condition and model (never above 2/6).

### Findings

1. **"The gate, not the language" holds at every measured scale.**
   `sql-gated` is the best condition at all four model points — never below
   raw `sql`, and the overall winner at 3b (16 vs 11), 8b (21 vs 19), and
   coder (23 vs 17). The integrity boundary carries the value, independent of
   query language and model scale.
2. **Datalog authorship is scale- and code-training-dependent, and the prior
   advantage inverts at coder scale.** Remembero rises 1.0 (1b) → 10.0 (3b)
   → 17.0 (8b) → 22.0 (coder 7b), where it beats raw `sql` outright (22 vs
   17) and nearly ties `sql-gated` (22 vs 23). The v1 negative is now a
   1b-only finding: code-trained models transfer to Datalog authorship.
3. **The ground-fact guard removed phantom passes.** Under the final harness
   1b remembero dropped from 2/31 to 1/31 and gate-protected passes from 3 to
   0 — the earlier t1 "pass" came from a garbage ground query returning the
   whole table and the answer model guessing right. The guard rejects those
   queries with actionable guidance, so the 1b floor is now honest.
4. **Control c1 resolves at scale.** The gate never refused it anywhere; its
   answer leg passes from 8b up in all conditions (and at 3b-remembero),
   failing below that for query-authorship reasons.
5. **Seed spread measured zero at every model.** Per-seed counts are
   identical across 7/42/123 at temperature 0; variance claims would need
   temperature > 0.
6. **Answer-set grading is active and visible.** v2 outcomes record
   `extraEntities`; e.g. corrupted-t1 SQL answers fail with
   `extraEntities: ['active']` instead of passing on a substring.
