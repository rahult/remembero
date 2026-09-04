# ADR 0003: Agent-boundary v2 — circularity control, grading implementation, statistics, gate metric

Date: 2026-09-04
Status: Accepted

## Context

ADR 0001 and ADR 0002 settled the claim split, prior leveling, the `sql-gated`
arm, versioning, and the model matrix. Four sharper questions remained:

1. The three write-gate rules were co-designed with the four trap writes — each
   rule catches exactly its trap, so the gate result is vulnerable to a
   circularity critique.
2. Answer-set grading needs an "extra named entity" universe to catch
   wrong-superset answers (the `j5` failure mode).
3. v1 ran a single seed; run variance is unmeasured.
4. The `t1` pass was produced by gate protection despite a garbage model query;
   it was unclear whether that counts as a pass or a curiosity.

## Decision

1. **Freeze the gate rules; add traps and a control.** The three v1 gate rules
   are pre-registered by the v1 publication and never change. v2 adds new trap
   writes that the frozen rules must catch, plus one **benign-write control**
   (a write violating no rule, e.g. updating a meeting preference) that the
   gate must *not* refuse and whose effect the answer must reflect. The control
   proves the gate is not a reject-everything stub.
2. **Entity lexicon from the seed database.** Answer-set grading builds its
   entity lexicon from all cell values in `AGENT_BOUNDARY_SEED_SQL`. An answer
   fails if it names a lexicon entity outside the question's gold set.
   `yes`/`no` remain phrasing, not cell values. Fully deterministic; no
   per-question hand-authored forbid lists.
3. **Three seeds, report spread.** v2 runs seeds 7, 42, 123 per model ×
   condition at temperature 0 and reports mean and per-seed spread. Cells stay
   small; seed variance becomes visible without pretending statistical power.
4. **Gate-protected correctness is a first-class metric.** For write-trap
   questions the primary metric is `gateRefusedTrap` plus answer correctness,
   reported separately from query-authorship pass rate. A correct answer with a
   garbage query behind a refusing gate is the integrity claim working, not a
   fluke.

## Consequences

- The gate claim's evidentiary strength now comes from traps authored *after*
  the rules were frozen — the co-design circularity is bounded to v1 and
  disclosed there.
- Grading depends only on the seed SQL and gold queries, both test-enforced.
- v2 summaries carry per-seed rows; headline numbers are means with spread.
- The v1 publication (docs/research/AGENT-BOUNDARY.md) and the results JSON are
  committed as immutable evidence; docs/EVALS.md links the analysis.
