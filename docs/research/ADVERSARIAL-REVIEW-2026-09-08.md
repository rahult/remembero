# Adversarial review: implementation, MCP surface, query language

Date: 2026-09-08. Four independent read-only review passes (MCP surface; query language and
engine; capture, store and recall pipeline; the day's closure, training and harness code),
synthesized and spot-verified. Claims marked **verified** were reproduced by the maintainer
after the review; the rest carry the reviewer's evidence as `file:line`.

The question asked: what would make Remembero powerful, accurate, easy to use and easy to
extend, especially for a small (3–4B) model on both the write and the read side.

## The five findings that matter most

1. **The integrity gate ships off.** `src/env.ts:85` defaults `REMBERO_INTEGRITY_MODE` to
   `off`; `src/init.ts` never sets it; `src/store/store.ts:1074` skips enforcement when no
   mode is configured. **Verified.** The feature the benchmarks identify as carrying most of
   the value is not running for the recommended setup. Fix: default to
   `no_new_violations`, have `init` set `strict`, keep `off` as an explicit opt-out.

2. **The engine answers wrong queries with silence.** Unknown predicate, wrong arity,
   reversed arguments, a doubled `_plus` suffix, and a capitalized constant (silently a
   variable, returning a superset) all yield `[]` or a plausible-looking row set with no
   diagnostic (`src/engine/evaluate.ts:580`, `src/engine/lexer.ts:83-96`,
   `src/mcp/tools.ts:565-604`). For a small model, "empty" and "wrong" are
   indistinguishable, and the yes/no idiom maps empty to "no". Fix: diagnostics on empty
   results (unknown predicate with near-miss names, arity mismatch, "entity never seen in
   argument k"), a hard error for `_plus` with arity ≠ 2 or a doubled suffix, and a guard for
   singleton uppercase variables whose lowercase form is a known atom in that slot.

3. **The "first rule's head is the query" convention is wrong, and the published few-shot
   proves it.** `src/sqlite/extension.ts:322` takes `program[0].head`; the shipped example
   for "final upstream dependency" lists `reach` first and `root` last and returns the whole
   chain instead of the root. **Verified.** The test only asserts the program runs. Fix:
   the query target is the unique sink predicate (a head no body references), with an error
   when ambiguous and an explicit `?- goal.` override. Fix the few-shot and make the test
   compare rows.

4. **Extraction has no vocabulary control, no first-person anchor, and no per-fact
   provenance.** The only defence against `works_at` / `employed_by` / `job` drift is a
   prompt sentence (`src/llm/prompts.ts:94`); "I" becomes `user` in auto-capture and `rahul`
   in `remember` (`tests/autocapture.test.ts:194-199` vs `src/evals/extraction.ts:62`);
   auto-captured facts are stored as accepted with a constant source label
   (`src/llm/pipeline.ts:733-737`); temporal supersession depends on the model emitting
   `retract` (`prompts.ts:95-98`). Fix, in order: closed vocabulary with alias rewriting
   enforced locally; a configured self atom; a required per-fact quote checked as a
   substring of the input; functional-dependency declarations that supersede on key
   collision in the store.

5. **`p_plus` materializes the whole closure.** `src/engine/closure.ts:55-67` synthesizes the
   unanchored rules; an anchored query on a 400-node chain takes 164 ms and 80k tuples where
   a seeded rule takes 2 ms; a 500-node chain hits the 100k fact limit. Fix: when a `p_plus`
   goal has a ground argument, synthesize a seeded unary closure and rewrite the goal.

## MCP surface

- **Cognitive load.** 43 tools registered (README says 36); six core tools carry 9–12
  parameters each, most of them integrity/proof/graph plumbing (`src/mcp/server.ts:762-771`,
  `1280-1295`); five read tools for two jobs (`recall`/`recall_explain`,
  `query`/`explain_query`, `search_knowledge`). Fix: a small-model core of 5–6 tools with ≤4
  parameters, `explain?: boolean` instead of twin tools, advanced fields on the `full` profile
  only.
- **Schema for the model.** `query` tells the model to call `list_memories`, which dumps
  every fact (697 KB at 20k facts) with no argument names. The compact `buildSchemaSummary`
  already exists in `src/llm/prompts.ts:33-68` but MCP callers never see it. Fix: a `schema`
  mode returning predicate, arity, count, samples and argument hints, and an
  `rembero_arg_names(...)` declaration so hints are user-extensible.
- **Errors.** Parse errors are parser-internal (`unexpected character '"'`), integrity
  rejections are proof dumps without a suggested `supersede_facts`/`forget`, and the
  ground-fact rejection suggests a program that itself fails range restriction
  (`extension.ts:579-583` vs `parser.ts:313-316`). Fix: a hint table keyed on the offending
  token, and `suggestedFix` on integrity violations.
- **Slot-filled alternative.** No structured tool for the two failure classes small models
  have (direction, recursion). Fix: `lookup({predicate, subject?, object?, transitive?})`
  compiling to `p`/`p_plus`.
- **Key-gated tools** are registered regardless of key presence; only two of six say so.
  Fix: skip registration without a key, or prefix the title.
- **`p_plus` is unknown to the rest of the product.** `validateQueryPredicates`
  (`src/llm/pipeline.ts:801-804`) rejects it in `recall`; `why_not` reports the closure fact
  as missing; the recall prompt, README dialect section and init snippet never mention it.
- **Extensibility.** Two hard-coded profiles, no tool plugin hook, no bundle import, no rule
  library loader, and two contradictory write policies (`docs/AGENT-HARNESS.md` says propose
  only; README and init say call `remember`). Fix: data-driven profiles, `import_knowledge_bundle`,
  `load_rule_library(name)`, one write-policy doc.

## Query language and engine

- **Two dialects.** MCP `query` rejects the rule-program form the benchmark teaches
  (`parseQuerySpec` at `src/mcp/tools.ts:589`); the bridge accepts rules but its ground-fact
  rule depends on a trailing period. Fix: one normalizer accepting goal-list, `?-` and
  rule-program forms on every surface.
- **Yes/no.** A ground goal should return one boolean row (`{yes: true|false}`) everywhere,
  replacing both the rejection and the `[{}]`-versus-`[]` two-character difference.
- **Stratification is checked per clause, not per namespace, and before closure expansion**
  (`src/engine/parser.ts:546` vs `evaluate.ts:1081`). A stored rule that negates a closure
  poisons every later query in the namespace with an error that names no predicate. Fix:
  expand closures before stratifying, stratify `existing ∪ new` on assert, name the cycle.
- **Dates.** `2026-09-08` lexes as subtraction in comparisons (`lexer.ts:98-111`). Fix: a
  lexer hint to quote it.
- **Two evaluators diverge.** Native and portable disagree on mixed-type comparisons, NULLs,
  `%` comments and limits, and adding `\+` flips routing between them
  (`extension.ts:266-291`). Fix: route `datalogQuery`/`datalogExplain` to portable always;
  keep native only behind `datalogSql`.
- **SQL habits get generic errors.** `not`, `"..."`, `AND`, adjacent atoms. Fix: accept `not`
  and double quotes as synonyms, normalize on serialization, hint on the rest.
- **Direction.** Optional named arguments checked against declared signatures, plus a
  data-driven "never seen in argument k" hint on empty results. Skip inverse predicates and
  tool-only wrappers as the primary fix.
- **Decision engine gaps.** No priorities or defaults/exceptions beyond hand-built `\+`
  ladders, no date type, no computed values, no `avg`/`distinct`/group-by. A `default` /
  `exception` sugar compiling to stratified negation and a `choose(Option) where ...` form
  would make "recommend X unless Y" writable by a small model.

## Capture, store and recall

- Retrieval and reasoning are disjoint: `recall` never uses lexical or semantic candidates to
  choose predicates (`pipeline.ts:172-173`, `1488-1497`); synonym questions fail unless the
  LLM resolves them in one retry. Fix: retrieve-then-reason with alias-aware ranking.
- Auto-capture is context-starved and cap-biased: `userOnly: true` while the prompt promises
  assistant context, and a daily cap of 10 captures the first turns of the day
  (`src/autocapture/hooks.ts:16`, `capture.ts:83,121-123`).
- The 15-case extraction eval is saturated (frontier models score 100 percent) and has no
  negation, coreference, first-person, transcript-mode or realistic-competitor cases.
- Entity identity is a read-time projection that is off by default
  (`src/env.ts:142`); write-side atom normalization does not exist.

## The day's code (closure, training, harness)

Engine: no correctness bug found under closure-over-derived-base, closure under negation,
mixed arities, proof numbering with constraints and aggregates interleaved, and bridge
routing. Defects found and fixed the same day:

- `gradeQueryRows` accepted supersets and any row as "yes"; fixed, all results regraded,
  findings corrected (fine-tune 27 → 26; coder gated-SQL 26 → 24).
- Two vocabulary entries with inverted up/down phrasing (~300 training lines); fixed with a
  per-relation direction test. The round-3 checkpoint was trained on the flawed data.
- Paraphrase constant filter used substring match; fixed to whole-word.

Recorded, not yet fixed: the recipe's "held-out" loss was a slice of the training file
(`benchmarks/tinker/sl_query_dialect.py`, `test_size=100`); held-out worlds share all
vocabulary with training; `--examples` is a global cap that shrank round 3; training and
evaluation prompts differ; the closure prompt contains a yes/no example one constant away
from benchmark question m5; ternary relations are listed in every schema but never queried,
matching the benchmark's remaining ternary misses.

## Recommended order of work

1. Gate on by default; `init` sets strict. One line, highest value.
2. Diagnostics instead of silence: unknown predicate, arity, direction hint, capitalized
   constant, doubled suffix. This is what makes a small model's retry loop converge.
3. Sink-predicate query target and a single query normalizer across MCP, CLI and bridge;
   boolean rows for ground goals; fix the few-shot.
4. Small-model core profile: ≤6 tools, ≤4 parameters, `schema` mode with argument names,
   `lookup` slot-filled tool, actionable parse and integrity errors.
5. Write side: closed vocabulary with aliases, self atom, quote-grounded facts, functional
   dependencies with automatic supersession, auto-capture as tentative with real provenance.
6. Seeded closure for anchored `p_plus` goals; expand closures before stratifying; stratify
   the whole namespace on assert.
7. Retire the native evaluator from the query path.
8. Retrain with the fixed vocabulary, a true held-out evaluator, per-template quotas, and
   ternary templates; then the Qwen3.5-4B run.
