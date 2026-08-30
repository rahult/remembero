# Remembero vs plain SQLite — an honest comparison

This document exists because a fair critique landed on the original
`/labs/chat-memory` design: *the lab behaved the same with or without
Remembero.* The critique was correct. This page records why that happened,
what is genuinely different about Remembero, and how the rebuilt labs and
playground demonstrate the difference structurally instead of cosmetically.

## Why the original lab showed no value

The old lab compared two lanes over one SQLite database, but every variable
that could have separated them was pinned:

- One canned question per scenario — the tool schema `enum`-ed a single
  allowed query string, so neither lane ever authored anything.
- A human had pre-written the perfect SQL for one lane and the perfect
  Datalog for the other. On single-join questions those are equivalent by
  construction; the human had already done all the reasoning.
- Nine rows, no recursion, no writes, no absent data that mattered, no
  contradictions. That is precisely the subspace where SQL and Datalog
  coincide.

The result was a controlled experiment whose control had removed the
treatment. The only visible difference was a proof panel that read as
decoration, because nothing in the questions needed a proof.

## Where SQLite alone is the right answer

Honesty first. If the workload is lookups, filters, joins, aggregates, and
transactions over known shapes — use SQLite. It is faster to write, has
fifty years of tooling, and `WITH RECURSIVE` covers transitive closure when
a developer writes the query by hand. Remembero is *built on* SQLite storage
in its own stack; it does not compete with SQLite as a database. The
comparison that matters is: **what can the tool boundary return to a model
or an agent, beyond rows?**

## The structural differences

Each of these is now a lab scenario or playground preset, and each is
enforced by a test that runs the exact seed and commands over real SQLite
(`tests/chat-memory-lab.test.ts`, `tests/playground-presets.test.ts`).

1. **Answers carry checkable proofs.** `WITH RECURSIVE` and a recursive
   Datalog rule agree on the transitive root blocker — the spec test proves
   they return the same closure. But SQL returns anonymous rows you must
   trust, while `datalog_explain` returns the derivation chain hop by hop,
   with the rule that fired at every step. A model given the proof can only
   restate it; a model given bare rows interprets them.

2. **Contradictory writes can be refused.** `UPDATE status SET
   state='active'` succeeds silently in SQLite while a recorded blocker
   still exists, and every downstream query then answers confidently and
   wrongly. The same store with a violation rule
   (`violation(P, B) :- proposed_status(P, active), blocker(P, B).`)
   derives the exact contradiction, names the culprit facts, and the write
   rolls back. Cross-table invariants in stock SQLite require triggers
   nobody writes; here the constraint is two lines of the same language the
   queries use.

3. **Absence is a proven claim, not a NULL.** A `LEFT JOIN` returns a NULL
   cell whose meaning — unknown? none? not loaded? — is left for the model
   to guess. Stratified negation returns the absence itself
   (`\+ prefers_meeting(jordan, _)`) inside a proof, so "no preference is
   stored" is grounded evidence, not an interpretation.

4. **Missing answers can be diagnosed.** When the schedule query for a
   project returns zero rows, SQL has nothing further to say. Evaluating the
   rule's premises one at a time names exactly which conditions fail — no
   review slot exists, the project is not blocked — and which hold. An
   empty set becomes an explanation.

Beyond the labs, the full stack adds temporal history (what was believed
when), journal checkpoints, trust views, and Memorg-portable bundles — but
the four properties above are the demonstrable core, and they are the ones
an agent harness actually leans on: answers it can check, writes it can
refuse, absences it can state, and failures it can explain.

## What changed in the labs and playground

- `/labs/chat-memory` now runs four scenarios chosen so the lanes
  structurally diverge: the recursive root-cause chain (steelmanned — the
  SQL lane gets `WITH RECURSIVE`, not a handicapped query), the silently
  corrupting write vs the refusing gate, the NULL vs the proven absence,
  and the empty result vs the why-not diagnosis.
- `/playground` gained three "Beyond SQL" presets — proven absence, write
  gate, why-not — runnable and inspectable against the browser-local
  SQLite extension, proofs included.
- Every scenario's seed, SQL, and Datalog are exercised by vitest over
  `node:sqlite` with the real extension semantics, so the demos cannot
  drift into fiction.
