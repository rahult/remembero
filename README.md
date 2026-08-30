# Remembero

[![CI](https://github.com/rahult/remembero/actions/workflows/ci.yml/badge.svg)](https://github.com/rahult/remembero/actions/workflows/ci.yml)

Logic-based memory for LLM chats and agents. Instead of fuzzy vector recall, Remembero stores
memories as **Datalog facts, rules, and explicit integrity constraints** and answers
questions by **logical inference** —
an LLM (Claude Sonnet 5 via OpenRouter by default; set `LLM_MODEL` for the cheaper
`openai/gpt-5.6-luna` economy option) translates natural language in and out, and a
built-in, zero-dependency Datalog engine does the reasoning deterministically.

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Quickstart (Claude Code, ~2 minutes)

```bash
npm install -g remembero
export LLM_API_KEY=sk-or-...   # OpenRouter key; optional — raw query tools work without it
remembero init                 # hooks + core-profile MCP registration + CLAUDE.md snippet
```

`remembero init` installs two Claude Code hooks (ambient capture of durable facts on Stop,
a deterministic memory brief injected at SessionStart), registers the MCP server with the
12-tool `core` profile, and prints a CLAUDE.md snippet to paste. Then, in chat:
*"Remember that my dentist is Dr Chen"* → later, *"Who's my dentist?"*.

Daily commands:

```bash
remembero session-brief          # what the agent sees at session start
remembero review                 # inspect/prune ambient captures (--forget 2,5)
remembero-web                    # browse your real memory at http://127.0.0.1:4318
remembero backup memory.json     # verified whole-store backup (restore with `restore`)
```

Latency expectations: structured queries and briefs are local and sub-millisecond;
natural-language `remember`/`recall` each make 1–2 model calls (seconds, ~$0.001);
opt-in semantic search is the slow path (10–30 s) and never runs inside a turn by default.

Backups, restore, and the two-machine story are covered in
[docs/SYNC-AND-BACKUP.md](docs/SYNC-AND-BACKUP.md). Weekly hygiene: `remembero review`
to prune anything ambient capture got wrong.

The executable [agent database scorecard](docs/AGENT-DATABASE-SCORECARD.md) gates exact
answers, proof citations, stale leakage, engine latency, a real MCP round trip, and the
zero-model/zero-embedding structured-query cost boundary:

```bash
npm run bench:agent-db:check
npm run bench:agent-db:cost
npm run bench:agent-db:install:check
npm run bench:agent-db:million
npm run bench:longmemeval:download
npm run bench:longmemeval
npm run bench:longmemeval:semantic # live embedding cost
npm run bench:longmemeval:answer -- --split dev # live durable formation + QA
npm run bench:memory:external
npm run bench:memory:mem0 # live OpenRouter extraction cost
npm run bench:memory:graphiti # live OpenRouter graph formation cost
```

The pinned external command runs the same hidden-label retrieval protocol against LangGraph
and LlamaIndex without adding their Python packages to Remembero.
The separate Mem0 OSS command requires `OPENROUTER_API_KEY` and records provider-native
formation tokens and charged cost; it is never part of normal CI or prepublish.
The Graphiti OSS command also requires `OPENROUTER_API_KEY`, stays outside CI/prepublish,
and uses native bulk episode formation, local FastEmbed, and a fresh embedded FalkorDBLite
graph per question.
The pinned LongMemEval-S commands provide both a retrieval-only zero-model baseline and a
live durable formation → retrieval → answer evaluation. The locked end-to-end policy scored
75.4% across all 500 questions and 71.5% on untouched held-out data. A role-aware v2
post-hoc run improves overall accuracy to 77.0% while cutting reader tokens 76.7% and
reader-plus-embedding cost to $0.000645/question. Adaptive v3 uses top five only for
multi-session questions, reaching 77.6% overall and 63.9% multi-session accuracy at
$0.000760/question. V4 also uses top five for temporal reasoning, reaching 79.8% overall
and 79.7% temporal accuracy at $0.000786/question. V5 routes multi-session questions
through semantic reranking only when the local top score
is at most 315. It reaches 83.2% overall and 76.7% multi-session accuracy at
$0.000872/question—still 68.2% below v1 runtime cost. Its
[method, evidence split, and limitations](docs/research/LONGMEMEVAL.md) are public.
For fuzzy recommendations, the opt-in [`semantic_search_knowledge`](docs/SEMANTIC-KNOWLEDGE-SEARCH.md)
tool reranks a bounded local shortlist, reports provider cost, and caches document vectors
across process restarts without changing structured query or proof authority. The bounded
`.semantic-embeddings/` directory is derived and can be deleted safely.
Run `remembero semantic-index` or call `prepare_semantic_search` after reviewed writes to
move document embedding out of the next user-facing query.
The measured prepared multi-session path reduces user-turn p95 from 27.3 to 18.3 seconds;
it spends about 19.7% more total embedding cost, so use it when latency matters more than
minimum provider spend.

The separate pinned LongMemEval-V2 adapter runs through the official agentic-memory harness.
Its original fresh text-only enterprise pilot improved Qwen3.5-9B from 0/10 without retrieval
to 3/10 with 121 ms memory-query p95 and zero memory provider calls. The current lexical v2
pilot remains 3/10 at 133 ms p95. The original v1
50-question expansion scored 10/50 at 122 ms p95. See the
[fresh pilot boundary](docs/research/LONGMEMEVAL-V2.md); it is not a leaderboard claim.
The opt-in prepared state-semantic lane reaches 6/10 on the same ten-question pilot at
512 ms memory-query p95, with 46.8 s maintenance and $0.005382 embedding cost for the
shared 100-trajectory haystack; it remains separate from the zero-provider-cost default.
On the frozen 50-question expansion, the same prepared lane scores 18/50 (36.0%) at
503 ms memory-query p95, with $0.005221 embedding cost and no same-question no-retrieval
control.

## Try the real web console

```bash
npm install
npm run web:dev
```

Open `http://127.0.0.1:4318` for a sourced Personal workspace that exercises structured
capture, deterministic guided recall, rule proofs, honest non-answers with related
knowledge, local search, health, and the explicit knowledge graph. It uses a dedicated
`.rembero-web/` sandbox and does not touch existing memory. See
[the web-console contract](docs/WEB-CONSOLE.md).

For an online product tour, open the public
[Remembero site](http://remembero.rahultrikha.com/). Its `/playground` route runs SQLite
WebAssembly with the native extension linked into the same binary, over fictional sample
knowledge entirely in the browser. Install the `remembero` package and run the `remembero`
CLI; the former `rembero` executable remains a compatibility alias. See
[the hosted-playground contract](docs/MARKETING-PLAYGROUND.md).

The local web console also includes a four-document intelligence showcase built from real
public PDFs: English and Spanish IRS W-9 forms, an arXiv research paper, and a UN publication.
The UI displays each rendered source page, publisher URL, original/page hashes, reviewed
claims, deterministic recall, proof, and honest abstention. Run `npm run eval:documents` for
the executable parse/answer/proof/abstention scorecard; see the
[measured document showcase boundary](docs/DOCUMENT-SHOWCASE-EVALUATION.md).
The separate `npm run eval:ocr:live` command exercises the bounded Unlimited-OCR transport
against those real labelled pages; see the current official-provider quota result in the
[live OCR evidence boundary](docs/UNLIMITED-OCR-LIVE-EVALUATION.md).
The same real-PDF corpus can be downloaded from the web console or exported as a
content-addressed, parent-first Memorg memory with `remembero document-memorg`; see the
[Memorg interoperability contract](docs/DOCUMENT-MEMORG-EXPORT.md).
Before publishing, run `npm run ship:check` and review the explicit
[ship-readiness boundary](docs/SHIP-READINESS.md).

```
"Rahul works at Acme. Mira also works at Acme.          works_at(rahul, acme).
 People who work at the same company are colleagues."   works_at(mira, acme).
                                            ──────────▶ colleague(X, Y) :- works_at(X, C),
                                                                          works_at(Y, C), X != Y.

"Who are Rahul's colleagues?"  ──▶  ?- colleague(rahul, X)  ──▶  "Rahul's colleague is Mira."
```

Facts nobody ever stated directly (like `colleague(rahul, mira)`) are *derived*, not stored.

## Install

```bash
npm install -g remembero        # or run ad hoc with: npx -y remembero
```

Configuration is via environment variables (a `.env` file in the working directory also works):

| Variable | Required | Default |
|---|---|---|
| `LLM_API_KEY` | for `remember`, `recall`, or semantic search | — (an [OpenRouter](https://openrouter.ai) key) |
| `LLM_BASE_URL` | no | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | no | `anthropic/claude-sonnet-5` |
| `REMBERO_EMBEDDING_MODEL` | no | `perplexity/pplx-embed-v1-0.6b` |
| `REMBERO_EMBEDDING_BASE_URL` | no | `LLM_BASE_URL` or `https://openrouter.ai/api/v1` |
| `REMBERO_HOME` | no | `~/.rembero` (memories live in `$REMBERO_HOME/memory/`) |
| `REMBERO_LLM_ALLOWED_NAMESPACES` | no | all namespaces (comma-separated allowlist when set; empty blocks all LLM export) |
| `REMBERO_AUTO_CAPTURE_DAILY_CAP` | no | `10` unique attempts per namespace/UTC day |
| `REMBERO_AUTO_CAPTURE_TAIL_BYTES` | no | `24576` bytes (maximum `49152`) |
| `REMBERO_MCP_PROFILE` | no | `full`; `core` registers only the 12 daily-driver MCP tools (also `serve --profile core`) |
| `REMBERO_WEB_DEMO` | no | `false`; the web console shows your real memory by default — `true` (or `--demo`) opens the seeded fictional sandbox |
| `REMBERO_VALID_TIME_MODE` | no | `delete`; set `archive_until` to preserve superseded facts (`remembero init` registrations default to `archive_until`) |
| `REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT` | no | `8` detailed predicates on the first recall pass (range: 1–256) |
| `REMBERO_RECALL_ANSWER_MODE` | no | `natural`; use `deterministic` bindings or compact `evidence` |
| `REMBERO_INTEGRITY_MODE` | no | `off`; use `strict` or migration mode `no_new_violations` for atomic write rejection |
| `REMBERO_INTEGRITY_NAMESPACES` | no | target namespace only; `*` or a comma-separated governed view when enforcement is active |
| `REMBERO_CHECK_MODE` | no | `off`; use `strict` or migration mode `no_regressions` |
| `REMBERO_CHECK_SUITE` | with check mode | regular JSON v1 suite file path |
| `REMBERO_CHECK_NAMESPACES` | no | target namespace only; `*` or a comma-separated governed view |
| `REMBERO_ENTITY_IDENTITY` | no | `off`; use `canonical` for explicit position-scoped alias projection |

The `REMBERO_*` names and `.rembero` directories are stable compatibility contracts;
renaming the package and CLI does not move or hide existing memory.

The raw Datalog tools (`assert`, `claims`, `accept`, `reject`, `supersede`, `query`,
`check`, `assert_facts`, `assert_tentative`, `review_tentative`, `resolve_tentative`,
`supersede_facts`, `propose-memory`, `propose_memory`, `apply-memory`, `apply_memory_proposal`, `what-if`, `what_if`, `apply-rule-change`, `apply_rule_change`, `why-not`, `why_not`, `topology`,
`knowledge_topology`, `diff`, `diff_recorded_knowledge`, `repair`,
`plan_query_repair`, `audit-rules`, `audit_rules`, `health`, `knowledge_health`, `search`, `search_knowledge`,
`browse`, `browse_knowledge_graph`, `connect`, `connect_knowledge_graph`, `bundle`, `export_knowledge_bundle`,
`verify_knowledge_bundle`, `test-knowledge`, `run_knowledge_checks`, `profile`,
`profile_query`, `forget`, `list_memories`)
work with no API key at all—only
natural-language `remember`/`recall` call the LLM.
Opt-in `semantic-search`, `semantic-index`, `semantic_search_knowledge`, and
`prepare_semantic_search` also require the embedding provider key and report their own usage
and cost.

## Use from Claude Code (MCP)

For a framework-neutral model → tool → Remembero → model integration, including typed
validation, proof handling, answer contracts, and proposal-only writes, see
[Add Remembero to an agent harness](docs/AGENT-HARNESS.md).

```bash
claude mcp add remembero --env LLM_API_KEY=sk-or-... -- npx -y remembero serve
```

From a git checkout instead: `claude mcp add remembero -- node /path/to/rembero/dist/cli.js serve`

To make agents use memory *proactively*, add a snippet like this to your `CLAUDE.md`
(or system prompt):

```markdown
## Memory (Remembero)
- At the start of tasks, use `recall` to check for relevant remembered context.
- When I state something durable — a preference, decision, relationship, or fact about
  me or a project — store it with `remember`. Updates ("X is now Y") supersede old facts.
- Never store secrets or transient details. When unsure whether to remember, ask.
```

The `core` profile (`serve --profile core`, used by `remembero init`) registers the
12 daily-driver tools: `remember`, `recall`, `recall_explain`, `list_memories`, `forget`,
`history`, `assert_facts`, `query`, `explain_query`, `supersede_facts`, `check_integrity`,
and `search_knowledge`. The default `full` profile also exposes the knowledge-engineering
surface below.

Tools exposed (full profile): `remember`, `propose_memory`, `apply_memory_proposal`, `recall`, `recall_explain`, `assert_facts`,
`assert_tentative`, `review_tentative`, `resolve_tentative`, `supersede_facts`,
`query`, `explain_query`, `check_integrity`, `conflict_views`, `history`, `forget`,
`what_if`, `apply_rule_change`, `why_not`, `knowledge_topology`, `diff_recorded_knowledge`,
`plan_query_repair`, `audit_rules`, `knowledge_health`, `search_knowledge`,
`semantic_search_knowledge`, `prepare_semantic_search`, `checkpoint_journal`,
`browse_knowledge_graph`, `connect_knowledge_graph`, `export_knowledge_bundle`, `verify_knowledge_bundle`,
`run_knowledge_checks`, `profile_query`, `list_checkpoints`, and `list_memories`.
`remember`/`recall` take natural language; the raw query and integrity tools are direct
and LLM-free.

Raw MCP writes accept an optional caller-stable `opId`. Retrying `assert_facts`,
`supersede_facts`, or `forget` with the same namespace, operation, ID, and normalized request returns the
original result without applying the mutation again. Reusing an ID for a different
request returns a structured `operation_conflict` error.

For inspectable reasoning, `recall_explain` and `explain_query` return the bindings plus
deterministic derivation proofs, durable source statements, and a query-scoped personal
knowledge graph. Facts remain authoritative in the same portable `.dl` files; the graph
is derived and cannot drift into a second source of truth.

Large explanations can be exported as deterministic subgraphs without changing their
rows or proofs. MCP graph selectors choose a result support chain, a node's complete
support closure, or a bounded neighborhood.

Natural-language recall returns an explicit status: `answered`, `no_match`,
`unanswerable`, or `schema_budget_exhausted`. The last status is an honest bounded-result
signal, not a claim that no relevant memory exists.

Add `--related` to a recall command when a non-answer should also return the nearest local
facts, rules, and policies. These suggestions use the exact same snapshot and fixed search
scores; they are discovery evidence, not a replacement answer or proof.

### Optional ambient capture

Manual `remember` remains the default. To opt into ambient capture at the end of Claude
Code turns:

```bash
remembero init-hooks --namespace personal
```

This safely merges one asynchronous Stop hook into your personal Claude settings. It
reads only a bounded trusted transcript tail, removes code/tool noise, rejects secrets,
deduplicates repeated Stop events, applies a per-namespace daily request cap, and accepts
only additive ground facts explicitly grounded in the user's words. It never installs at
package-install time and never performs automatic retractions.

Every capture, empty result, failure, cap, and duplicate is visible locally:

```bash
remembero review --namespace personal
remembero review --namespace personal --forget 2,5
remembero init-hooks --remove
```

The raw transcript is not persisted as per-fact provenance. See the
[auto-capture contract](docs/AUTO-CAPTURE.md) for settings scopes, quotas, review JSON,
and the async-hook lifecycle boundary.

## CLI

```bash
node dist/cli.js remember "Rahul's dentist is Dr Chen"
node dist/cli.js propose-memory 'Mira now works at Initech.' # review, no write
node dist/cli.js propose-memory 'Mira now works at Initech.' --check-suite checks.json
node dist/cli.js apply-memory memory-review.json --op-id reviewed-memory-v2
node dist/cli.js recall   "Who is Rahul's dentist?"
node dist/cli.js recall   "Who owns Atlas?" --schema-predicate-limit 48
node dist/cli.js recall-explain "Who are Rahul's colleagues?"
node dist/cli.js recall "Who works at Acme?" --answer-mode deterministic
node dist/cli.js recall "Who works at Acme?" --answer-mode evidence
node dist/cli.js assert   ':- status(Person, active), status(Person, terminated).'
node dist/cli.js check    --proof-limit 2 --max-violations 100
node dist/cli.js conflicts mira # focused cross-policy conflict evidence
node dist/cli.js conflicts mira --as-of-sequence 17 # exact recorded conflict view
node dist/cli.js assert   'status(mira, terminated).' --integrity-mode strict
node dist/cli.js assert   'status(mira, paused).' --trust tentative
node dist/cli.js claims
node dist/cli.js accept   'status(mira, paused).' --op-id review-17
node dist/cli.js assert   'status(mira, active).' --op-id change-123 # retry-safe
node dist/cli.js supersede 'works_at(mira, initech).' \
  --pattern 'works_at(mira, _)' --at '2026-08-16T16:59:00.000Z' --op-id job-42
node dist/cli.js remember 'Mira is now terminated' --integrity-mode no_new_violations
node dist/cli.js explain  'path(a, X)' --proof-limit 4 # inspect every bounded proof path
node dist/cli.js query    'dentist(rahul, X)'        # raw Datalog, no LLM call
node dist/cli.js query    'employee(X), \+ suspended(X)' # closed-world negation
node dist/cli.js query    'select City where dentist(rahul, D), lives_in(D, City)'
node dist/cli.js query    'age(X, A), age(dana, D), A > D + 5' # numeric arithmetic filter
node dist/cli.js query    'count(*) as Count where works_at(Person, acme)'
node dist/cli.js explain  'colleague(rahul, X)'      # proof + source + graph, no LLM call
node dist/cli.js what-if  'colleague(mira, Who)' \
  --without 'works_at(rahul, _)' --assume 'works_at(rahul, acme).'
node dist/cli.js what-if 'derived(X)' \
  --assume-rule 'derived(X) :- base(X).' --check-suite checks.json
node dist/cli.js apply-rule-change rule-review.json --op-id reviewed-rule-v2
node dist/cli.js why-not  'colleague(mira, rahul)' # missing premises + nearby evidence
node dist/cli.js topology 'colleague' --direction upstream # rule dependency closure
node dist/cli.js diff 17 23 --query 'status(mira, State)' # semantic + consequence diff
node dist/cli.js repair 'eligible(bob)' # minimal verified assumptions/retractions
node dist/cli.js audit-rules 'eligible' --direction upstream # proactive rule health
node dist/cli.js health --check-suite checks.json # complete immutable health snapshot
node dist/cli.js search 'Doctor Chen' --kind fact # local lexical provenance search
node dist/cli.js browse mira --browse-depth 2 # explicit entity neighborhood
node dist/cli.js connect mira rahul --path-depth 4 # shortest explicit relationships
node dist/cli.js connect mira rahul --include-derived # rule conclusions + proofs
node dist/cli.js bundle > knowledge.json && node dist/cli.js verify-bundle knowledge.json
node dist/cli.js export > memories.dl                 # dump every namespace as plain Datalog
node dist/cli.js import personal memories.dl          # load clauses into one namespace
node dist/cli.js backup memory-backup.json            # verified whole-store backup
node dist/cli.js restore memory-backup.json           # restore into a fresh/empty store
node dist/cli.js test-knowledge checks.json # deterministic rule regression suite
node dist/cli.js profile 'relevant(X, Y)' --compare-scan # deterministic work counters
node dist/cli.js explain  'path(a, X)' --graph-result 2 # one result's complete support
node dist/cli.js explain  'path(a, X)' --graph-neighbors 'entity:["a"]' --graph-depth 2
node dist/cli.js query    'works_at(mira, X)' --entity-identity canonical
node dist/cli.js forget   'dentist(rahul, _)'
node dist/cli.js forget   'dentist(rahul, _)' --op-id forget-123 # retry-safe
node dist/cli.js history  'works_at(mira, _)' --json
node dist/cli.js query    'works_at(mira, X)' --as-of-sequence 17 # exact recorded past
node dist/cli.js checkpoint --op-id backup-2026-08-17 # rotate the active journal safely
node dist/cli.js checkpoints                           # inspect verified boundaries
node dist/cli.js list
node dist/cli.js review --namespace personal           # inspect ambient captures
node dist/cli.js review --namespace personal --forget 2 # explicit prune by number
node dist/cli.js init-hooks --namespace personal       # opt in to Claude Stop capture
node dist/cli.js serve                                # MCP server on stdio
```

Natural-language supersession deletes the old fact by default. Opt into valid-time archives with
`REMBERO_VALID_TIME_MODE=archive_until` or `remember --valid-time-mode archive_until`.
An update then keeps the preceding fact as an ordinary
`<predicate>_until(..., '<ISO instant>').` clause. `history` replays the bounded journal
in authoritative append order, while past-tense recall can query and explain those
portable archive facts. See [the temporal history contract](docs/TEMPORAL-HISTORY.md).

At 100+ predicates, recall ranks a deterministic local schema slice, preserves rule
dependencies and temporal companions, and evaluates every accepted query against the
complete selected namespaces. Empty or unanswerable results from a partial slice trigger
one bounded widening pass; if completeness still cannot be established, recall reports
`schema_budget_exhausted` instead of inventing “no memory.” Pruning diagnostics are
returned by the library and MCP surfaces. See
[the recall schema-pruning contract](docs/RECALL-SCHEMA-PRUNING.md).

Explanations keep the first deterministic witness by default. Pass `--proof-limit <n>`
(maximum 16) to `explain` or `recall-explain` to request a complete bounded set of
branch-simple alternative derivations. The same `proofLimit` option is available on the
MCP explain tools. Additional proofs remain query-scoped, retain ordered source evidence,
and appear as distinct proof instances in the graph; no graph sidecar or proof index is
persisted. If more proofs exist than the requested limit, Remembero fails explicitly rather
than presenting incomplete evidence as complete. See
[the alternative-proof contract](docs/ALTERNATIVE-PROOFS.md).

Explicit headless constraints describe forbidden knowledge states without guessing
semantics from predicate names. For example,
`:- status(Person, active), status(Person, terminated).` flags every person with both
current statuses. `check` / MCP `check_integrity` returns complete bounded violations
with proof, source, and graph evidence; the CLI exits `2` when findings exist. Constraints
remain inert during normal query/recall and cannot be generated by natural-language
memory extraction. See [the integrity-constraint contract](docs/INTEGRITY-CONSTRAINTS.md).

`-n <ns>` / `--namespace <ns>` selects the namespace to write to; `--namespaces a,b` or
`--namespaces '*'` selects which namespaces recall, query, check, list, and history read
from.

Namespaces organize one local personal store; they are not access-control or tenant
boundaries. Use separate `REMBERO_HOME` roots and server processes when data must be
isolated. Natural-language operations reject credential-like input before calling an
external LLM. Raw Datalog operations remain local and should never be used to store
secrets. Set `REMBERO_LLM_ALLOWED_NAMESPACES=work,shared` to keep every other namespace
local-only; the policy covers both remembering and recalling, including wildcard reads.

## Storage

Memories live in plain text at `~/.rembero/memory/<namespace>.dl`, one canonical clause per
line — readable, hand-editable, diffable. Duplicate facts, alpha-equivalent rules, and
alpha-equivalent integrity constraints are deduplicated on write. Files are written
atomically. All supported fact/rule/constraint mutations are globally serialized so
optional integrity validation and commit observe the same snapshot. Journaled mutations carry stable
operation IDs; facts captured through `remember` retain their source statement for later
explanation. Cross-process writers are serialized so background capture cannot overwrite
a simultaneous manual mutation. Credential-like source text is redacted before
journaling, and active journal capacity is checked before mutation. Immutable checkpoint
segments reset that active-file capacity while preserving the complete recorded sequence.
Opt-in supersessions atomically
close old facts, add their `_until` archives and replacements, and record exact source
lineage without changing explicit `forget`. See
[the explainable graph contract](docs/EXPLAINABLE-KNOWLEDGE-GRAPH.md).

## Use Remembero as SQLite plus governed memory

For Node applications, `openRememberoDatabase(...)` is the primary database API. It
returns the real `node:sqlite` connection, so ordinary SQL, prepared statements,
transactions, functions, and existing `.db` files keep working. The same object adds
Datalog query/proof methods and a `memory` store whose clauses, provenance journal,
trust state, checkpoints, and recorded history are authoritative in the same SQLite file.

```ts
import { openRememberoDatabase } from 'remembero';

const db = await openRememberoDatabase('app.db');
db.prepare('INSERT INTO works_at(person, company) VALUES (?, ?)').run('mira', 'acme');
console.log(db.datalogQuery('employed(X) :- works_at(X, _).'));

db.memory.assert('default', 'prefers(mira, tea).', {
  opId: 'preference-2026-08-24',
});
db.close();
```

Memory mutations participate in caller-owned SQLite transactions, so application rows
and governed memory commit or roll back together. See the
[SQLite database guide](docs/SQLITE-DATABASE.md) for install steps, the existing-database
path, internal-table boundary, and exact compatibility limits.

### Immutable semantic ledger

`createSemanticLedger(...)` adds a standalone content-addressed version graph to any
`node:sqlite` database. It records immutable objects and versions, typed dependencies,
contracts, evaluation evidence, compatibility vectors, promotion decisions, and append-only
ref history. `captureKnowledgeVersion(...)` optionally links an exact Remembero journal state;
the memory store and ledger remain independently usable. See the
[semantic ledger guide](docs/SEMANTIC-LEDGER.md).

The Remembero review surface builds a complete product version around that generic ledger:
knowledge, document/source lineage, rules, integrity policy, model, runtime, and evaluation
suite are captured as typed members and edges. Use the CLI to inspect the same authority:

```bash
remembero version capture --label remembero@baseline
remembero version list
remembero version diff main <candidate-digest>
remembero version review <candidate-digest>
remembero version promote <candidate-digest> <assessment-digest> \
  --op-id reviewed-candidate \
  --accept-review knowledge-schema \
  --accept-review policy-review
```

The local web console exposes the same flow under **Versions**. Capture and review are
non-mutating; only an explicitly reviewed promotion moves the `main` ref. The MCP server
exposes equivalent `capture_semantic_version`, `diff_semantic_versions`,
`review_semantic_version`, and `promote_semantic_version` tools when started by the CLI.

### Native SQLite extension

Remembero also ships the source for a real loadable SQLite extension. It treats ordinary
SQLite tables (and views) as Datalog predicates: arguments map to columns by position,
and SQLite remains the storage and transaction authority. Ordinary positive rules use
the native extension; the Node adapter deterministically bridges advanced rules to the
same bounded evaluator used by portable `.dl` knowledge. This is a separate
application-facing primitive. The default CLI/MCP memory store continues to use portable
`.dl` files; `openRememberoDatabase(...)` explicitly selects the same-file SQLite-backed
authority for embedded applications.

V0 supports macOS and Linux. Build the extension with a C compiler and the SQLite
development headers. From a source checkout use:

```bash
npm run build:sqlite
```

From an installed npm package use `remembero sqlite-build`. The command compiles the native
library inside the installed package; it does not run automatically during installation,
so Remembero's existing non-SQLite memory features do not acquire a native toolchain
requirement.

Then create a normal database and query it through the CLI (the adapter requires Node.js
22.13 or newer):

```bash
sqlite3 world.db <<'SQL'
CREATE TABLE works_at(person TEXT, company TEXT);
INSERT INTO works_at VALUES ('alice', 'acme'), ('bob', 'acme'), ('carol', 'other');
SQL

npm run build
node dist/cli.js sqlite-query world.db \
  'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.'
node dist/cli.js sqlite-plan world.db \
  'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.'
```

The result is deterministic JSON:

```json
[
  { "X": "alice", "Y": "bob" },
  { "X": "bob", "Y": "alice" }
]
```

The public library adapter exposes the same path:

```ts
import { openDatalogDatabase, sqliteDatalogExecutionMode } from 'remembero';

const db = await openDatalogDatabase('world.db');
const rule = 'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.';
console.log(db.datalogSql(rule));   // inspect the generated SELECT
console.log(db.datalogPlan(rule));  // inspect routing and referenced schema
console.log(db.datalogQuery(rule)); // execute it and parse the JSON rows
console.log(sqliteDatalogExecutionMode(rule)); // "native"
db.close();
```

Recursive programs use multiple rules for one derived predicate. Evaluation is bounded,
semi-naive, and set-based: each round joins at least one recursive body literal against
only the previous round's delta.

```ts
const program = `
  path(X, Y) :- edge(X, Y).
  path(X, Y) :- edge(X, Z), path(Z, Y).
`;

console.log(db.datalogQuery(program));
console.log(db.datalogExplain(program)); // one nested derivation proof per result
```

Inside SQLite, the registered scalar functions are `datalog_sql(rule)`,
`datalog_query(program)`, and `datalog_explain(program)`. `datalog_sql` deliberately
remains a single non-recursive rule compiler; recursive programs execute through the
fixpoint evaluator. Rules support joins through repeated variables, text/number constants,
and `=`, `!=`, `<`, `>`, `<=`, and `>=`.

`DatalogDatabase.datalogQuery`, `DatalogDatabase.datalogExplain`, `sqlite-query`, and
`sqlite-explain` also support raw conjunctions, stratified negation, arithmetic comparison
expressions, scalar aggregates, and programs with multiple derived predicates. The adapter
loads only referenced tables inside a read savepoint, canonicalizes their rows, and runs
the portable evaluator. For a rule program, the first rule head is the result relation and
must contain distinct named variables; later rules may define that relation or its
dependencies. `sqliteDatalogExecutionMode(input)` reports which path will run.
This bridge deliberately uses Datalog value equality rather than SQLite affinity and
accepts only text plus finite safe-range integer/real values; `NULL`, BLOB, non-finite,
and unsafe integer values fail closed.

The stock SQLite scalar functions remain the smaller native surface, so applications that
load only the `.dylib`/`.so` do not receive those adapter capabilities. `datalog_sql` also
rejects advanced syntax because it promises one inspectable SQLite `SELECT`. Integrity
constraints and entity identity declarations remain personal knowledge-store policies,
not database query syntax.

Both paths are bounded. Adapter inputs are limited to 64 KiB, 100,000 referenced base
rows, 10,000 additional facts, 1,000 rounds, 10,000 output rows, proof depth 128, and
16 MiB of input/output. Native programs retain their 16-rule, tuple-check, and proof caps.
Unsafe, malformed, arity-inconsistent, unsupported-value, or cap-exceeding queries fail
closed. Extension loading is disabled again immediately after the library is loaded. See
[SQLite determinism and parity](docs/SQLITE-DETERMINISM.md) for the exact matrix.

## The Datalog dialect

- Facts must be ground: `works_at(rahul, acme).` `birth_year(rahul, 1985).`
- Valid-time archives are ordinary system-managed facts such as
  `works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').`
- Atoms are lowercase (`acme`) or quoted (`'Acme Corp'`); variables uppercase (`X`, `Who`);
  `_` is a wildcard in queries and rule bodies.
- Rules, including recursive ones: `ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).`
- Explicit integrity constraints have no head and describe forbidden states:
  `:- active(Person), suspended(Person).` They use query-style range restriction, never
  derive facts, and are inspected only by `check` / `check_integrity`.
- Entity identity is explicit metadata: `rembero_alias('Mira Patel', mira).` declares
  an alias and `rembero_entity_position(works_at, 2, 0).` opts one zero-based argument
  position into canonical reads. The declarations never rewrite durable facts or history.
- Tentative trust is explicit metadata:
  `rembero_tentative('works_at(mira, acme).').` remains outside accepted reasoning until
  reviewed; use the typed CLI/MCP/library surfaces instead of authoring wrappers manually.
- Stratified closed-world negation: `available(X) :- employee(X), \+ suspended(X).`
  Variables in comparisons and negated literals must be bound by an earlier positive
  goal. Recursive dependency cycles containing negation are rejected.
- Comparisons in rule bodies and queries: `=`, `!=`, `<`, `>`, `<=`, `>=`. Numeric
  operands support deterministic `+`, `-`, `*`, `/`, unary signs, and parentheses with
  standard precedence: `older(X, Y) :- age(X, A), age(Y, B), A > B + 5.` Arithmetic is
  filter-only and cannot create values in facts, rule heads, relation arguments, or
  aggregate inputs. Non-numeric operands, division by zero, and non-finite results fail
  closed.
- Terminal scalar aggregation:
  `count(*) as Count where works_at(Person, acme)`, plus `sum(Value)`, `min(Value)`,
  and `max(Value)`. Aggregation consumes the complete logical solution set and fails
  closed at its dedicated input cap rather than silently reusing the normal row limit.
- Reusable grouped aggregation in rules:
  `company_size(Company, Count) :- count(*) as Count where works_at(Person, Company).`
  Every aggregate dependency is strictly stratified, and aggregate cycles are rejected.
- Every query terminates: arithmetic only filters a finite relation; evaluation remains
  stratified, semi-naive bottom-up over a finite fact universe, with belt-and-braces
  derivation and expression-complexity caps.
- Safety: facts must be ground; every head variable must appear in a positive body literal
  (range restriction). LLM output that violates this is rejected, retried once with the
  error message, then surfaced as an error — nothing unparsed ever reaches the store.

## Troubleshooting

- **`LLM_API_KEY is not set`** — export it, put it in `.env` in the directory you launch
  from, or pass it via `claude mcp add --env`. Only `remember`/`recall` need it.
- **HTTP 401/403 from the LLM** — key is invalid or lacks access to the model; try
  another `LLM_MODEL` you have access to on OpenRouter.
- **`failed to load ….dl`** — a memory file was hand-edited into a state that doesn't
  parse; the error names the file and line. Fix the line (or delete it) and retry.
  Nothing is ever silently dropped.
- **Server shows "disconnected" in Claude Code** — run `npx -y remembero serve` manually;
  anything printed before the JSON handshake (e.g. npm warnings) breaks stdio. Use
  `npx -y` (never a bare `npm run`) so nothing pollutes stdout.

## Development

```bash
npm test          # vitest suite (engine, store, pipeline, tools)
npm run build     # tsc
npm run build:sqlite # compile the native SQLite extension
npm run test:sqlite  # native + Node adapter + CLI end-to-end checks
npm run eval:extract # live labeled comparison of exact personal-knowledge mutations
npm run eval:recall # live labeled comparison of baseline and grounded recall prompts
npm run dev -- …  # run the CLI from source (tsx)
```

The extraction and recall evals report exact-case accuracy plus mutation or binding-row
precision/recall/F1. They can compare OpenRouter models or emit JSON; see
[docs/EVALS.md](docs/EVALS.md).

See [the stratified-negation contract](docs/STRATIFIED-NEGATION.md) for safety,
closed-world, proof, and SQLite-boundary details, and [the scalar aggregation
contract](docs/QUERY-AGGREGATION.md) for exact reduction and explanation semantics, and
[the arithmetic comparison contract](docs/ARITHMETIC-COMPARISONS.md) for numeric,
precedence, safety, and portability details. TypeScript consumers should read the
[0.2](docs/MIGRATING-0.2.md), [0.3](docs/MIGRATING-0.3.md),
[0.4](docs/MIGRATING-0.4.md), [0.5](docs/MIGRATING-0.5.md),
[0.8](docs/MIGRATING-0.8.md), [0.9](docs/MIGRATING-0.9.md), and
[0.14](docs/MIGRATING-0.14.md) migration notes as
applicable.
