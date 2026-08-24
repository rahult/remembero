# Changelog

Release notes previously embedded in the README are collected here in reverse
chronological order. Only releases that had a standalone README note are listed.

## Unreleased

Adds a generic SQLite semantic ledger for content-addressed objects, immutable version and
dependency graphs, first-class contracts, evaluation evidence, compatibility vectors,
promotion decisions, and append-only ref history. The optional Remembero bridge captures an
exact recorded knowledge state and reuses the existing semantic knowledge diff; the ledger
also operates standalone and contains no downstream-platform assumptions.

Adds `openRememberoDatabase(...)`, a Node database API that returns the native SQLite
connection with normal SQL behavior plus Datalog planning/query/explanation and a
`MemoryStore`-compatible `memory` surface. Governed memory files and history are durable
inside the same SQLite database, participate in caller-owned transactions, refresh across
connections, and reopen without a sidecar authority. Dynamic extension loading is locked
again after initialization; the derived semantic embedding cache remains outside the
durable SQLite claim.

Adds an executable agent-database scorecard with real stdio MCP proof, structured-query
cost boundaries, 100,000-fact scale gates, and a million-fact diagnostic. OpenRouter usage
metadata now flows into recall and extraction evals. The default detailed recall schema
slice is reduced from 32 to 8 after a complete live 26-case run preserved 100% correctness
while lowering charged recall cost by 25.8%; honest-negative widening remains unchanged.
Adds the first pinned external-stack comparison: LangGraph 1.2.10 with FastEmbed 0.8.0
and BGE-small runs through a manifest-declared, process-isolated bridge. The measured
retrieval-only adapter reaches 100% Recall@k/MRR without being misreported as an answer or
proof system.
Adds a second pinned retrieval adapter for LlamaIndex VectorMemory 0.14.23 using the same
FastEmbed model and capability disclosures, enabling a controlled three-way local comparison.
Adds a full Mem0 OSS 2.0.14 formation-and-retrieval adapter with native OpenRouter usage
accounting, plus Retrieval Precision@k so distractor-heavy top-k results cannot hide behind
perfect recall and MRR.
Adds a pinned Graphiti OSS 0.29.3 native bulk-formation and hybrid-retrieval adapter using
Luna, local FastEmbed, and embedded FalkorDBLite, including provider-native cost and an
explicit embedded-server shutdown contract.
Adds opt-in semantic knowledge search for recommendation and preference retrieval. The CLI
and MCP tool rerank a bounded lexical shortlist, enforce namespace/secret export policy,
report embedding usage/cost, and cache document vectors without changing proof authority.
The content-addressed derived cache now survives MCP and CLI restarts, is bounded to 2,000
entries, stores no source text, validates entry digests, and recomputes corrupt entries.
Adds explicit resumable semantic prewarming for post-review maintenance. It deduplicates
shared provenance, reports provider cost, and moves document embedding out of the next user
query without coupling mutation success to an external service.
Semantic ranking now scores bounded overlapping source chunks instead of one diluted
whole-session vector, improving all-preference LongMemEval Recall@5 from 66.7% to 73.3%
with a 13.3% increase in routed provider cost.
A conservative lexical score/margin guard prevents an obvious exact preference from being
demoted by max-chunk similarity without changing the benchmark rankings.
Selects the 0.6B semantic embedding default with a controlled development-only model
matrix. It outperformed the eligible 4B and 8B candidates on Recall@5 and MRR while costing
2.5–7.5x less; the free candidate was rejected rather than weakening provider data policy.
Embedding failures now include bounded, redacted provider diagnostics so configuration and
privacy-policy errors are actionable without exposing secrets.
Adds a product-native LongMemEval-S answer runner that forms each history in a fresh durable
store, performs bounded source retrieval, generates an answer, emits upstream-compatible
hypotheses, and applies a task-specific judge contract with provider-native cost. The locked
policy scores 75.4% over all 500 questions and 71.5% held-out. Factual and personalized
answers now use distinct grounding contracts; recommendation intent recognizes plural and
implicit-reason wording. Provider-boundary Unicode normalization repairs malformed source
text, and independent embedding batches run with bounded concurrency three.
The end-to-end reader now keeps full transcripts for assistant-memory questions but sends
only retrieved user turns for user facts, preferences, updates, temporal reasoning, and
multi-session synthesis. The 500-question post-hoc v2 result rises from 75.4% to 77.0%
while cutting reader tokens 76.7% and runtime provider cost 76.5%. Multi-session accuracy
rises from 56.4% to 61.7%. Global semantic multi-session routing, a low-score semantic gate,
top-six context, an aggregation prompt, and GPT-5.4 Mini were measured and rejected.
Adaptive answer context now retrieves five sessions only for multi-session questions and
keeps four elsewhere. It raises the composed post-hoc score to 77.6% and multi-session
accuracy to 63.9% at $0.000760 runtime provider cost per question. A role-aware top-six
rerun was rejected at 47/69 development answers versus top five's 51/69.
Temporal reasoning now also receives five sessions while ordinary questions remain at four.
Complete temporal runs improve the subtype from 71.4% to 79.7% and lift the composed
post-hoc score to 79.8% at $0.000786 runtime provider cost per question, without another
model or embedding call.
Multi-session semantic reranking is now gated by a deterministic local top-score ceiling of
315. Under role-aware top-five context it raises multi-session accuracy from 63.9% to 76.7%
and the composed post-hoc score from 79.8% to 83.2%. The gate avoids 18 high-confidence
local cases and 55 provider calls relative to global semantic routing.
The LongMemEval answer runner can explicitly prepare semantic document vectors before the
measured user turn. A 20-case check reduces user-turn p95 from 27.3 to 18.3 seconds and
query embedding tokens from 1,836,242 to 249, while disclosing the 19.7% increase in total
embedding cost as maintenance rather than hiding it.
Adds a pinned LongMemEval-V2 backend that splits browser-agent states into bounded chunks,
uses a state-level BM25 shortlist plus Remembero local chunk scoring, and conforms to the
official metadata-private `insert/query` interface. On ten fresh text-only enterprise
questions it improves the fixed reader from 0/10 to 3/10 at 121 ms memory-query p95 with
zero memory provider calls; multimodal and leaderboard claims remain explicitly out of scope.
The original v1 frozen adapter has a 50-question text-only enterprise expansion at 10/50 and
122 ms memory-query p95, with zero operational errors; it remains historical after the v2
chunk/diversity pass. The result remains a pilot because
web, multimodal, abstention, gotcha, and medium-tier questions are not yet covered.
Adds an opt-in prepared state-semantic LongMemEval-V2 lane using the existing 0.6B
embedding client. On the same ten-question pilot it reaches 6/10 versus the lexical
3/10, with 512 ms memory-query p95; 46.8 s maintenance and $0.005382 embedding cost are
reported separately from the user turn and the zero-provider-cost default.
The same prepared lane reaches 18/50 on the frozen 50-question expansion at 503 ms
memory-query p95, with $0.005221 embedding cost; no same-question no-retrieval control is
claimed for this expansion.
Adds a clean-install gate that packs the current package, installs it with lifecycle scripts
disabled into a fresh directory and empty npm cache, then times a real first write and
proof-carrying query with model credentials scrubbed.
Promotes the million-fact scale diagnostic into a separate repeated latency-and-memory gate.
The benchmark releases its source corpus after parsing, reports process max RSS, requires
indexed exact rows/proofs across three repetitions, and caps peak RSS at 2.5 GiB.
Adds a commit/hash-pinned 500-question LongMemEval-S source-retrieval runner and publishes
the full precision, recall, MRR, strict-coverage, abstention, latency, source-window, and
threshold evidence boundary. Local search now ignores conversational stopwords, indexes a
16 KiB source window by default, caps aggregate source ranking at 32 MiB, reports the
effective window, and lets trusted library callers set a minimum relevance score.

## 0.55.0

Renames the npm package and primary command-line tools to `remembero` and
`remembero-web`, matching the public product, repository, and domain. Installations of
the new package retain `rembero` and `rembero-web` executable aliases for compatibility.
Existing `REMBERO_*` environment variables, `.rembero` data directories, hook markers,
portable formats, and native SQLite symbols remain unchanged so existing memory and
integrations continue to work. See [the migration note](docs/MIGRATING-0.55.md).

## 0.54.0

Adds a public structured-memory benchmark that reports exact typed answers,
answerability, gold-evidence retrieval, proof citations, temporal updates, and trust views
as separate dimensions. `npm run bench:memory:check` is a deterministic release gate;
isolated JSON command adapters let external stacks participate without adding their
dependencies or credentials to Remembero. The first checked-in result measures Remembero and
transparent direct, lexical, and recency baselines only. See the
[benchmark](docs/research/MEMORY-STACK-BENCHMARK.md),
[Medium draft](docs/research/MEDIUM-DRAFT.md), and
[research paper](docs/research/paper/paper.md).

## 0.52.0

Makes failed recall immediately actionable. Opt-in related knowledge attaches
bounded deterministic local search results—with score reasons, provenance, trust, and a
retrieval graph—to `no_match`, `unanswerable`, and schema-budget results from the exact
same snapshot. It never changes answer authority or adds a model call. See
[the related-knowledge recall contract](docs/RELATED-KNOWLEDGE-RECALL.md).

## 0.51.0

Adds compact deterministic evidence answers. Positive recall can render
bindings, claims, rules, absences, aggregates, projections, trust, temporal lineage, and
durable sources locally, while negative recall keeps why-not summaries. No final phrasing
model call is made. See [the evidence-answer contract](docs/EVIDENCE-ANSWER-MODE.md).

## 0.50.0

Can enforce one portable check/coverage suite across every supported writer.
Strict mode requires a green candidate; `no_regressions` permits legacy debt and repairs
while rejecting newly failed checks or lower coverage. Enforcement runs under the mutation
lock and composes with integrity policy. See
[the all-writer check-enforcement contract](docs/CHECK-ENFORCEMENT.md).

## 0.49.0

Can bind portable regression checks and semantic coverage into a personal
memory proposal. Preview compares baseline/candidate results; reviewed apply re-runs the
normalized suite under the mutation lock and rejects failures before writing. See
[the reviewed memory check-gate contract](docs/MEMORY-CHECK-GATES.md).

## 0.48.0

Composes integrity, rule audit/topology, tentative review debt, identity
metadata, provenance completeness, and optional checks/coverage over one immutable current
or recorded snapshot. Stable findings produce healthy/review/violations status without an
LLM or semantic guessing. See [the knowledge-health contract](docs/KNOWLEDGE-HEALTH.md).

## 0.47.0

Gives relational answers explicit column authority. `select Answer where ...`
keeps helper join variables out of bindings, deduplicates and limits after projection, and
merges alternative proofs beneath the projected answer. Grounded recall now uses this form
for every variable-bearing query. See
[the relational projection contract](docs/RELATIONAL-PROJECTION.md).

## 0.46.0

Applies only an explicitly reviewed accepted-memory proposal. Exact facts,
rules, removals, and temporal archives commit together after current baseline, candidate
audit, and no-new-integrity checks pass under one mutation lock. The replayable change
retains provenance, fact history, recorded diffs, checkpoints, and idempotency. See
[the reviewed memory-application contract](docs/MEMORY-APPLICATION.md).

## 0.45.0

Lets natural-language memory remain a proposal before accepted mutation.
`propose-memory` uses the exact `remember` extraction and validation path, expands wildcard
corrections into exact baseline-bound changes, evaluates temporal, integrity, and rule
impact, and returns a content-addressed review artifact without writing. See
[the personal memory proposal contract](docs/MEMORY-PROPOSALS.md).

## 0.44.0

Applies only an explicitly reviewed digest-bound rule proposal. The operation
rechecks the exact current baseline, candidate audit, attached checks/coverage, and
no-new-integrity policy under one mutation lock, then journals one crash-safe idempotent
change. Recorded proposals and stale or tampered artifacts cannot apply. See
[the reviewed rule-application contract](docs/RULE-CHANGE-APPLICATION.md).

## 0.43.0

Previews rule changes before they become authority. `what-if` can add or
remove exact alpha-equivalent rules, compare query proofs, integrity, topology, and rule
health, then run the same knowledge checks and semantic coverage against both programs.
Current or exact recorded baselines remain immutable and proposed rules carry hypothetical
provenance. See [the rule-change impact contract](docs/RULE-CHANGE-IMPACT.md).

## 0.42.0

Can opt those paths into rule-derived relationships without turning inferred
edges into stored truth. `--include-derived` discovers semantic shortcuts through the
bounded fixpoint, then attaches sourced proofs and only the rules actually used by every
selected path claim. Recursion, negation, aggregates, aliases, tentative premises, and
recorded views keep the ordinary explanation semantics. See
[the proof-carrying knowledge-path contract](docs/KNOWLEDGE-PATHS.md).

## 0.41.0

Finds bounded shortest relationship paths between two explicit personal
knowledge entities. It returns ordered fact segments, argument positions, provenance,
aliases, trust, and recorded-view evidence, and distinguishes a fully disconnected
component from a depth-bounded miss. Equal shortest alternatives are complete or fail
closed. See [the deterministic knowledge-path contract](docs/KNOWLEDGE-PATHS.md).

## 0.40.0

Measures what natural-language memory actually changes. A 15-case live
extraction corpus runs through the real store and scores exact additions, removals,
corrections, rules, duplicates, tentative trust, authority no-ops, and local secret
rejection. Accepted mode now skips hedged claims; tentative caller authority is conveyed
to extraction while trust assignment remains local. See
[the evaluation contract](docs/EVALS.md).

## 0.39.0

Verifies grounded recall across four OpenRouter models and strengthens
bounded schema ranking for inverse grandparent/grandchild wording. Luna, Gemini 3.7 Flash,
and Claude Sonnet 5 passed all 26 current engine-backed cases; GPT-5.4 Mini remained
inconsistent by exposing rule-local helper variables. Luna stays the default because the
comparison covers recall translation, not memory extraction. See
[the model-compatibility checkpoint](docs/MODEL-COMPATIBILITY.md).

## 0.38.0

Profiles deterministic query work without wall-clock noise. `profile` returns
the normal proof/graph result plus relation lookups, indexed lookups, index facts processed,
and candidate facts visited. `--compare-scan` reruns without relation indexes and returns
only when complete explanations are byte-identical. See
[the deterministic-query-profile contract](docs/QUERY-PROFILING.md).

## 0.37.0

Adds schema-only `sqlite-plan` / `DatalogDatabase.datalogPlan(...)`.
It reports native/portable routing, execution boundary, result variables, derived
recursion, referenced tables/views, visible columns and declared types, optional native
SQL, and active bounds inside a savepoint without scanning rows. See
[SQLite Datalog planning](docs/SQLITE-PLANNING.md).

## 0.36.0

Adds semantic rule coverage to knowledge suites. Alpha-equivalent authored
rules form one coverage unit; primary, alternative, recursive, and aggregate proofs record
which named checks exercise it. An optional minimum percentage can fail CI after all row
checks pass. See [the semantic-rule-coverage contract](docs/RULE-COVERAGE.md).

## 0.35.0

Runs portable deterministic knowledge regression suites. Named checks can
expect empty, non-empty, exact ordered rows, or order-insensitive row sets against current
or recorded views. Failures include row deltas and proof or why-not evidence; CLI failures
exit `2` without storing test metadata. See
[the knowledge-check contract](docs/KNOWLEDGE-CHECKS.md).

## 0.34.0

Lets durable source vocabulary influence recall schema selection locally.
Predicate groups gain bounded source-word/phrase scores before dependency closure, while
source statements never enter the model prompt. Pruning diagnostics expose only the
selected predicate signatures that matched provenance. See
[the provenance-aware-recall contract](docs/PROVENANCE-AWARE-RECALL.md).

## 0.33.0

Exports raw namespace authority and durable provenance as deterministic,
content-addressed JSON. `bundle` includes trust/identity metadata and current or exact
recorded coordinates; `verify-bundle` checks canonical clauses, ordering, lineage, bounds,
and SHA-256 without importing anything. See
[the knowledge-bundle contract](docs/KNOWLEDGE-BUNDLES.md).

## 0.32.0

Browses the explicit stored personal graph without inventing a query.
`browse` seeds by entity and/or predicate, expands a bounded fact hypergraph through
shared entities, and returns claim/entity nodes, argument edges, provenance, aliases,
trust, namespaces, and recorded views. Rules and inferred claims remain query-scoped. See
[the explicit-graph-browse contract](docs/EXPLICIT-GRAPH-BROWSE.md).

## 0.31.0

Adds deterministic local search over facts, rules, constraints, and redacted
durable source text. `search` returns fixed integer scores with explicit match reasons,
provenance, trust/identity projection, recorded views, and a result/clause/predicate/entity
graph. It is lexical retrieval, not logical proof or vector similarity. See
[the local-knowledge-search contract](docs/LOCAL-KNOWLEDGE-SEARCH.md).

## 0.30.0

Adds opt-in deterministic rendering for successful natural-language recall.
`--answer-mode deterministic`, MCP `answerMode`, or
`REMBERO_RECALL_ANSWER_MODE=deterministic` formats exact boolean or binding rows locally,
labels tentative rows, and skips the final LLM phrasing call. Natural phrasing remains the
default. See [the deterministic-answer-mode contract](docs/DETERMINISTIC-ANSWER-MODE.md).

## 0.29.0

Makes negative natural-language recall deterministic after query review. A
final full-schema empty query receives complete why-not evidence and a local blocker
summary; Remembero no longer sends empty bindings to the LLM for phrasing. If diagnostic
limits are exceeded, recall returns an explicit `whyNotUnavailable` marker and an honest
generic negative answer. See
[the grounded-negative-recall contract](docs/GROUNDED-NEGATIVE-RECALL.md).

## 0.28.0

Audits rule health before a query fails. `audit-rules` reports undefined
closed-world negation, policy inputs without definitions, inert recursion, open positive
inputs, currently inactive derivations, alpha-equivalent duplicates, and arity overload.
Warnings exit `2`; informational findings remain successful. Every finding links into the
selected topology graph. See [the deterministic rule-audit contract](docs/RULE-AUDIT.md).

## 0.27.0

Turns grounded why-not blockers into verified proposal-only repair plans.
`repair` iteratively adds missing ground facts or retracts facts blocking negation, proves
the query against each candidate, removes redundant edits, and reports both strict and
no-new-violations policy safety. It never writes a proposed fact and returns a digest of
the exact baseline used for verification. See
[the repair-planning contract](docs/REPAIR-PLANNING.md).

## 0.26.0

Compares two exact recorded states as one coherent read transaction.
`diff` reports semantic fact/rule/policy additions and removals, provenance transitions,
topology node/edge impact, introduced or resolved integrity violations, and optional
before/after query proofs. This distinguishes an audit-only journal step from an actual
knowledge change and never orders history by timestamp. See
[the recorded-knowledge-diff contract](docs/RECORDED-KNOWLEDGE-DIFF.md).

## 0.25.0

Makes the rule system itself inspectable. `topology` maps predicates,
alpha-equivalent rule groups, policies, positive/negative/aggregate dependencies,
strata, recursive components, provenance, and undefined inputs. A predicate focus can
select its complete upstream requirements, downstream influence, or both while retaining
whole rule and policy nodes. See
[the knowledge-topology contract](docs/KNOWLEDGE-TOPOLOGY.md).

## 0.24.0

Explains deterministic failure rather than returning an opaque empty row
set. `why-not` follows conjunction bindings and every matching rule branch to missing
facts, present negated facts, false comparisons, recursive cycles, or aggregate output
mismatches. Nearby facts retain proofs and durable sources; a separate blocker graph
connects the query, attempted rules, failures, and observations. Empty `recall-explain`
results include the same `whyNot` evidence. See
[the why-not explanation contract](docs/WHY-NOT-EXPLANATIONS.md).

## 0.23.0

Adds deterministic counterfactual impact analysis. `what-if` evaluates
fact-only additions, removals, and corrections against a consistent current snapshot,
then returns before/after rows, changed proof evidence, introduced or resolved integrity
violations, and hypothetical provenance in the explanation graph. It never calls an LLM
or writes a namespace, source, or journal entry. See
[the counterfactual-impact contract](docs/COUNTERFACTUAL-IMPACT.md).

## 0.22.0

Removes the active journal's long-running growth bottleneck without deleting
history. `checkpoint` atomically rotates `journal.log` into an immutable, content-addressed
segment and publishes an exact clause/source checkpoint. Recorded sequence numbers remain
global across every segment and the active tail; reads reject missing, reordered, or
tampered artifacts. See [the journal-checkpoint contract](docs/JOURNAL-CHECKPOINTS.md).

## 0.21.0

Separates tentative claims from accepted knowledge. Tentative facts remain
explicit `.dl` declarations and journal entries but are excluded from reasoning by
default. Opt-in reads label their proofs, sources, and graph claims; deterministic
accept/reject operations preserve recorded-time and integrity authority. See
[the knowledge-trust contract](docs/TRUSTED-KNOWLEDGE.md).

## 0.20.0

Makes exact aggregation reusable inside rules. A clause such as
`team_size(Team, Count) :- count(*) as Count where member(Team, Person).` derives one
proof-carrying count per team for later rules, recall, integrity policy, graphs, and the
SQLite portable bridge. Aggregate dependency cycles fail stratification. See
[the aggregate-rule contract](docs/RULE-AGGREGATION.md).

## 0.19.0

No longer treats every non-empty natural-language query as semantically
correct. When deterministic local evidence finds a same-ground-anchor predicate
competitor or a historical query missing its named later state, recall performs one
bounded review before accepting the rows. The response records repeat, correction, or
unanswerable decisions in `queryReviews`; ordinary grounded recalls keep the one-call
path. See
[the recall disambiguation contract](docs/RECALL-DISAMBIGUATION.md).

## 0.18.0

Groups explicit integrity violations into focused personal conflict views.
`rembero conflicts [focus]` and MCP `conflict_views` combine every policy violation for
the same first alpha-stable constraint binding, with declaration provenance, fact proofs,
stable cluster IDs, and selectable evidence graphs. Canonical aliases and exact recorded
snapshots use the existing opt-in contracts; no conflict store or inferred subject schema
is added. See [the conflict-view contract](docs/CONFLICT-VIEWS.md).

## 0.17.0

Lazily indexes a relation's first argument whenever a rule or query has
already bound it. Selective joins, negation, recursion, aggregates, explanations, and the
SQLite portable bridge use the same insertion-ordered lookup without persisting a second
authority or reordering authored goals. Checked-in selective-join and recursive-growth
benchmarks require byte-identical rows and proofs, at least a 2x median speedup, and at
least a 100x reduction in deterministic relation work. Run `npm run bench:engine`; see
[the deterministic indexing contract](docs/ENGINE-INDEXING.md).

## 0.16.0

Makes that correction primitive directly available without an LLM. CLI
`supersede` and MCP `supersede_facts` atomically end up to 64 matched ground facts,
preserve each as `_until`, and add explicit replacement clauses under the same integrity,
retry, journal, and crash-recovery boundary as other writes. A caller-supplied `at` must
be a canonical UTC instant; it is descriptive valid-time metadata, never an ordering
authority. See [the temporal-correction contract](docs/TEMPORAL-CORRECTIONS.md).

## 0.14.0

Adds exact recorded-time snapshots across recall, query, explanation,
integrity audit, and listing. `--as-of-sequence 0` means before the journal; higher values
mean after that global journal entry. Rules, explicit identity, provenance, and graphs are
evaluated from the selected past view. Snapshot reads first reconcile the complete journal
with current files and fail closed if hand edits or legacy writes make history incomplete.
This is separate from descriptive valid-time timestamps and does not claim full bitemporal
interval algebra. See [the recorded-time snapshot contract](docs/RECORDED-TIME-SNAPSHOTS.md).

## 0.13.0

Makes raw assertions, retractions, and imports retry-safe. Supply a stable
`--op-id` in the CLI, `opId` in MCP, or `MutationContext.opId` in the library. Matching
retries return the first durable result even when it included duplicates or removed
facts; conflicting reuse fails with `OperationConflictError` (`operation_conflict`, CLI
exit `4`). See [the retry-safe write contract](docs/RETRY-SAFE-WRITES.md).

## 0.12.0

Adds bounded graph navigation to explanation, recall, integrity, and
write-rejection evidence. Use `--graph-result`, `--graph-support`, or
`--graph-neighbors` (with optional `--graph-depth`) in the CLI, or the equivalent
`graphSelector` object in MCP/library calls. Selection never changes result rows, proofs,
rules, or stored facts; it only projects the returned graph. See
[the graph-navigation contract](docs/GRAPH-NAVIGATION.md).

## 0.11.0

Can treat explicitly declared names as one entity without rewriting stored
facts. `rembero_alias(Alias, Canonical).` declares a chain and
`rembero_entity_position(Predicate, Arity, ZeroBasedPosition).` limits projection to
typed-by-policy argument positions. Raw reads remain literal by default; opt in with
`--entity-identity canonical`, `REMBERO_ENTITY_IDENTITY=canonical`, or the matching
library/MCP option. Proofs retain the exact literal source and graphs annotate canonical
entities with alias provenance. History always stays literal. See
[the entity identity contract](docs/ENTITY-IDENTITY.md).

## 0.10.0

Can promote those declarations into an opt-in atomic write boundary.
`strict` rejects any violating candidate; `no_new_violations` permits legacy findings to
remain or be repaired while rejecting new violation identities. Every rejection carries
the same bounded proof, source, and query-scoped graph evidence as `check`; CLI exit `3`
means no mutation was committed. All supported writers share one cross-process mutation
lock so a cross-namespace candidate cannot race another Remembero 0.10 writer. Audit remains
the default. See [the enforcement and migration contract](docs/INTEGRITY-ENFORCEMENT.md).
