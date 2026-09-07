# Agent-boundary benchmark: findings

Status: published findings · 2026-09-08
Evidence: four result JSONs under `results/` (llama3.2 1b/3b, llama3.1:8b,
qwen2.5-coder:7b), all produced by one identical final harness.
Method and decisions: [v1 analysis](AGENT-BOUNDARY.md), [v2 spec and full
results](AGENT-BOUNDARY-V2-DESIGN.md), [ADR 0001](../adr/0001-agent-boundary-benchmark.md),
[ADR 0002](../adr/0002-agent-boundary-v2-design.md),
[ADR 0003](../adr/0003-agent-boundary-v2-circularity-and-metrics.md).

## Setup in one paragraph

One seeded SQLite database (projects, people, reporting chains, statuses,
blockers, dependency edges, preferences) served to one local model under three
tool boundaries: **sql** (model-authored read-only SQL, no integrity gate),
**sql-gated** (the same SQL; writes through the Remembero integrity gate), and
**remembero** (model-authored Datalog through the same bridge and gate). 31
gold-verified questions across five categories; write-traps push a corrupting
write through each condition's write path first, plus one benign-write control
the gate must not refuse. The gate runs the shipped `enforceIntegrityCandidate`
primitive (strict mode) over constraints derived from rules frozen before the
traps were authored. Answers are graded by answer-set containment against a
seed-database entity lexicon. Temperature 0, seeds 7/42/123, two
error-feedback attempts.

## The four-model matrix

Mean questions passed per seed (per-seed counts were identical at every model):

| model              | sql      | sql-gated   | remembero  |
| ------------------ | -------- | ----------- | ---------- |
| llama3.2:1b        | 5.0/31   | 5.0/31      | 1.0/31     |
| llama3.2:3b        | 11.0/31  | **16.0/31** | 10.0/31    |
| llama3.1:8b        | 19.0/31  | **21.0/31** | 17.0/31    |
| qwen2.5-coder:7b   | 17.0/31  | **23.0/31** | 22.0/31    |

## Findings

### 1. The gate, not the language — at every measured scale

`sql-gated` is the best condition at all four model points and never falls
below raw `sql`. The integrity boundary carries the value independent of query
language and model scale. Gate integrity itself is uniform: across the matrix
the gate refused **72/72** trap writes in both gated conditions, raw `sql`
resisted **0/72**, and the benign control was refused **0/36** times — the
gate is not a reject-everything stub. The frozen integrity rules caught traps
authored *after* the freeze (t5, t6), so the catches are independent evidence,
not co-design.

### 2. Datalog authorship is scale- and code-training-dependent; the prior advantage inverts at coder scale

Remembero (model-authored Datalog) rises 1.0 (1b) → 10.0 (3b) → 17.0 (8b) →
22.0 (coder 7b). At coder scale it beats raw `sql` outright (22 vs 17) and
nearly ties `sql-gated` (22 vs 23). The v1 negative result — "small models
cannot author Datalog" — is a **1b-only finding**; code training transfers to
Datalog authorship. This is the cleanest confirmation of the v1 prior-confound
analysis: SQL's early lead measured training-data familiarity, not boundary
quality.

### 3. Absence is Datalog's structural edge

On absence questions (missing preferences, missing blockers, managers on no
project) remembero beats raw `sql` at 3b (6 vs 3) and 8b (6 vs 4) and ties at
coder scale (5 vs 5). `\+` negation-as-failure is more natural for models than
LEFT JOIN / IS NULL patterns. Multihop recursion remains hardest for every
condition and model (never above 2/6).

### 4. Tool-boundary fixes measurably clean the evidence

A ground-fact query (`prefers_meeting(maya, afternoon).`) used to return an
anonymous success row `[{}]` — no readable values. The bridge now rejects
ground-fact queries with actionable guidance (`q(W) :- …`). Effect on results:
1b remembero dropped from 2/31 to 1/31 and gate-protected passes from 3 to 0 —
its earlier "pass" was a garbage query returning the whole table plus a lucky
answer. The 1b floor is now honest.

### 5. Gate-protected correctness scales with per-family competence

A refusal only helps if the model can still answer. Gate-protected passes
(sql-gated) went 3 (1b) → 18 (3b) → 12 (llama3.1:8b) → 18 (coder). The 8b dip
below 3b tracks family query competence, not parameter count. The benign
control's answer leg resolves from 8b up in all conditions.

## Limitations

- Temperature 0: seed spread measured zero at every model; run variance is
  unexercised and no confidence claims are made.
- `llama3.2:8b` does not exist as a tag (family stops at 3b); llama3.1:8b is
  the 8B-scale substitute. Families are mixed, so cross-model comparisons are
  qualitative, not a clean scale axis.
- Toy project-management schema; no claim yet that results transfer to a real
  captured-memory store.
- The gate rules are three hand-authored constraints; the benchmark proves the
  mechanism, not the completeness of any particular rule set.
