# Agent-boundary benchmark v1: SQL vs Remembero-gated Datalog at small-model scale

Status: published negative result with integrity-gate finding · 2026-09-04
Evidence: [results/agent-boundary-v1-llama3.2-1b-summary.json](results/agent-boundary-v1-llama3.2-1b-summary.json)
Decisions: [ADR 0001](../adr/0001-agent-boundary-benchmark.md), [ADR 0002](../adr/0002-agent-boundary-v2-design.md), [ADR 0003](../adr/0003-agent-boundary-v2-circularity-and-metrics.md)

## What was measured

The same seeded SQLite database (8 tables: people, projects, reporting chains,
statuses, blockers, dependency edges, meeting preferences, review slots,
promised updates) is served to one small local model under two tool boundaries:

- **sql** — the model authors arbitrary read-only SQL, executed directly.
- **remembero** — the model authors arbitrary Datalog, executed by the same
  Remembero bridge the MCP server and browser labs use. Writes pass through the
  integrity-rule write gate.

28 questions across five categories: `direct`, `join`, `multihop`, `absence`
(4–6 per category), and `write-trap` (4), where the harness pushes a corrupting
write through each condition's write path before asking. Every question carries
gold SQL and gold Datalog verified in `tests/agent-boundary.test.ts`, so the
expected answers cannot drift into fiction.

Run: `llama3.2:1b`, temperature 0, seed 7, up to 2 attempts with error
feedback, 3 few-shot examples per condition. Reproduce with:

```bash
npm run bench:agent-boundary -- --model llama3.2:1b
```

## Results

| category   | sql passed | remembero passed | sql tool errors | remembero tool errors |
| ---------- | ---------- | ---------------- | --------------- | --------------------- |
| direct     | 4/6        | 0/6              | 4               | 7                     |
| join       | 1/6        | 0/6              | 10              | 12                    |
| multihop   | 1/6        | 0/6              | 8               | 12                    |
| absence    | 0/6        | 0/6              | 11              | 11                    |
| write-trap | 0/4        | 1/4              | 2               | 6                     |
| **total**  | **6/28**   | **1/28**         | **35**          | **48**                |

At 1b scale the model cannot author Datalog: it emits Prolog-flavored syntax,
copies `Q:` prompt text into queries, and hallucinates predicates. SQL
authorship is marginal (6/28) but clearly better. Error-feedback retries rarely
rescued either condition.

## The prior confound (read this before citing any number)

Small models have seen orders of magnitude more SQL than Datalog in training.
The SQL-vs-Remembero comparison above therefore measures **training-data
prior, not boundary quality**. Identical few-shot budgets do not equalize that
prior; the v1 "same budget" symmetry was false fairness. v1 numbers must not be
cited as evidence that one boundary is easier for agents in general.

## What v1 does support: the write gate

The single Remembero pass (`t1`) is the informative one. The harness pushed
`UPDATE status SET state='active' WHERE project='atlas'` through both write
paths:

- **sql**: applied silently. The model's (correct) query then returned the
  corrupted state and the answer asserted atlas is active. Truth lost without
  any error.
- **remembero**: the write gate evaluated its integrity rules inside a
  savepoint, derived a violation ("active project keeps a blocker"), rolled the
  write back, and the answer correctly stayed "blocked" — **even though the
  model's own Datalog query was garbage**. The gate, not the model, protected
  the answer.

All four trap writes corrupt the SQL condition's answers (0/4). The gate
refused all four (4/4 refusals), though only `t1` converted to a pass because
query authorship failed on the other three.

### Circularity caveat

The three gate rules were co-designed with the four traps — each rule catches
exactly its trap. v1's gate result is therefore a mechanism demonstration, not
yet independent evidence. v2 pre-registers the frozen rules and adds new traps
they must catch, plus a benign-write control the gate must not refuse
([ADR 0003](../adr/0003-agent-boundary-v2-circularity-and-metrics.md)).

## Grading caveat

v1 grades are substring expect/forbid matching, which passes wrong-superset
answers — `j5` (sql) is graded pass despite the model naming four people where
three were asked and then contradicting itself. Treat v1 pass counts as an
upper bound. v2 replaces this with answer-set grading against a seed-database
entity lexicon. The committed JSON keeps original v1 grades, labeled as such.

## Claims

Supported by v1:

1. A rule-gated write boundary preserves answers where raw SQL silently
   corrupts them, independent of model query quality (4/4 gate refusals vs 0/4
   SQL resistance; mechanism-level, co-design caveat above).
2. At 1b scale, free-form Datalog authorship is beyond the model and SQL
   authorship is marginal — a deployment-relevant negative result for
   local-first agent memory.

Not supported by v1: any claim that agents query better through Datalog than
SQL, or any cross-model generalization.

## v2 plan

Frozen v1; changes land as `agent-boundary-v2`: answer-set grading, a third
`sql-gated` arm isolating the gate from the query language, a Datalog
cheatsheet with an expanded few-shot budget (prior leveling), new traps plus a
benign-write control under frozen rules, three seeds (7, 42, 123) with reported
spread, and a model matrix (llama3.2:3b, llama3.2:8b, one coder model) to test
whether the authorship negative is scale-dependent. Gate-protected correctness
becomes a first-class metric.
