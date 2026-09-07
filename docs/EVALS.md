# Evaluations

## Agent database scorecard

The top-level agent-facing release gate combines deterministic answer/proof conformance,
repeated engine latency, a real compiled stdio MCP process round trip, zero-provider-cost
structured query accounting, and the supported two-command setup contract:

```bash
npm run bench:agent-db
npm run bench:agent-db:check
npm run bench:agent-db -- --json --repetitions 20
npm run bench:agent-db:scale -- --check
npm run bench:agent-db:million
npm run bench:agent-db:cost
npm run bench:agent-db:install:check
```

The install check packs the current tree, uses a fresh directory and empty npm cache, disables
install lifecycle scripts, withholds LLM credentials, and requires a real first CLI write plus
proof query to return the expected answer and sources. It is intentionally not in
`prepublishOnly`: the cold registry download is a network diagnostic, while package smoke
remains the deterministic release gate.

The million-fact command is a separate heavy gate. It runs with a 4 GiB V8 ceiling, drops
the textual source corpus after parsing, performs three exact indexed query/proof repetitions,
and fails above 3.0 s parse/query/proof or 2.5 GiB process max RSS. It is kept out of
`prepublishOnly` so normal package release checks remain practical.

See the [current scorecard, gates, and evidence boundary](AGENT-DATABASE-SCORECARD.md).

## Agent-boundary benchmark

The agent-boundary benchmark serves one seeded SQLite database to a small local model under
two tool boundaries — model-authored read-only SQL vs model-authored Datalog behind the
Remembero write gate — across 28 gold-verified questions. The published v1 run
(llama3.2:1b) is an honest negative for query authorship (SQL 6/28, Remembero 1/28) and a
mechanism demonstration for the write gate: all four corrupting trap writes corrupted the
SQL condition's answers while the gate refused all four, preserving one correct answer
despite a garbage model query. The SQL-vs-Datalog comparison is confounded by
training-data prior and is disclosed as such; the gate rules were co-designed with the v1
traps and v2 adds frozen-rule traps plus a benign-write control. It requires a local Ollama
model and stays outside CI/prepublish:

```bash
npm run bench:agent-boundary -- --model llama3.2:1b
```

See the [full method, results, confound analysis, and v2 plan](research/AGENT-BOUNDARY.md).

The v2 rerun adds answer-set grading, a `sql-gated` arm isolating the gate from the query
language, a bridge-verified Datalog cheatsheet, frozen-rule traps t5/t6 plus a benign-write
control, and three seeds per model — with the gated arms routed through the shipped
`enforceIntegrityCandidate` primitive, test-pinned to the frozen rules. Across the full
four-model matrix (llama3.2 1b/3b, llama3.1:8b, qwen2.5-coder:7b) the gate refused 72/72
trap writes while raw SQL refused 0/72 and the benign control 0/36, and `sql-gated` is the
best condition at every scale (16/31 vs 11 at 3b, 21 vs 19 at 8b, 23 vs 17 at coder): the
supported claim is "the gate, not the language". Datalog authorship is scale- and
code-training-dependent — Remembero rises 1/31 at 1b to 22/31 at coder scale, beating raw
SQL there (22 vs 17) — so the v1 authorship negative is a 1b-only finding. See the
[v2 spec and results](research/AGENT-BOUNDARY-V2-DESIGN.md).

## LongMemEval-V2 fresh pilot

The pinned official adapter consumes browser-agent trajectories through the benchmark's
`Memory.insert` / `Memory.query` interface and returns bounded Remembero state evidence.
On ten fresh text-only enterprise questions with deterministic official scorers, the
original Remembero pilot improved the fixed Qwen3.5-9B reader from 0/10 without retrieval
to 3/10. Its memory-query p95 was 121 ms with zero memory model or embedding calls. The
current lexical v2 pilot remains 3/10 at 133 ms p95. The original v1 frozen 50-question
expansion scored 10/50 (20.0%) at 122 ms memory-query p95; it is retained as historical
v1 evidence. This is not a leaderboard claim; see the
[contract and limits](research/LONGMEMEVAL-V2.md).

The opt-in prepared state-semantic lane scores 6/10 on the same ten-question selection,
with 512 ms memory-query p95, 46.8 s one-time maintenance for 3,358 states, and $0.005382
embedding cost for the shared 100-trajectory haystack. It is reported as a separate
accuracy/cost operating point rather than blended into the zero-provider-cost lexical result.
The same prepared lane scores 18/50 (36.0%) on the frozen expansion at 503 ms memory-query
p95, with 43.8 s maintenance and $0.005221 embedding cost. There is no same-question
no-retrieval control for those 50 questions, so this is generalization evidence rather than
a causal comparison.

## LongMemEval-S source retrieval

The complete cleaned 500-question LongMemEval-S split is pinned by dataset commit and
SHA-256. It evaluates whether local deterministic source search retrieves the labelled
evidence sessions, with no model, embedding, remote call, or hidden answer data:

```bash
npm run bench:longmemeval:download
npm run bench:longmemeval
npm run bench:longmemeval -- --json
npm run bench:longmemeval:semantic
npm run bench:longmemeval:answer -- --split dev
```

The current recall-first configuration achieves 83.27% Recall@5, 80.96% MRR, and 75.32%
strict all-evidence coverage over 470 answerable questions at 10.78 ms p95 local search.
Precision@5 is 30.72% and abstention empty rate is 0%, which remain explicit gaps rather
than being hidden by answer generation. See the [full method, sweeps, and evidence
boundary](research/LONGMEMEVAL.md).

The v1 live semantic policy benchmark uses a deterministic SHA-256 development/held-out split
and routes only explicit recommendation/advice intent. On 15 held-out preference questions,
Recall@5 improved from 46.7% to 60.0% and MRR from 16.1% to 47.2%. It requires an embedding
provider and stays outside CI/prepublish. See the [semantic search contract and cost
evidence](SEMANTIC-KNOWLEDGE-SEARCH.md).

The default embedding model was selected separately on the development half with every
retrieval control held fixed. Perplexity 0.6B beat the eligible Qwen3 8B and Perplexity 4B
candidates on Recall@5, MRR, cost, and p95 latency; larger losing candidates were not run
on held-out data. The exact measurements and policy are preserved in the
[model-matrix artifact](research/results/semantic-model-matrix-v1-summary.json).

The live answer runner adds a real durable-store lifecycle and emits official-compatible
`{question_id, hypothesis}` JSONL when requested. Its locked policy achieved 78.9% on the
261-question development partition and 71.5% on 239 untouched held-out questions, with
zero final errors. Across all 500 questions it scored 75.4%. A role-aware v2 keeps full
transcripts only for assistant-memory questions and sends user turns for other reader
tasks. The post-hoc v2 result scores 77.0%, cuts reader tokens 76.7%, and lowers reader plus
embedding cost from $1.371122 to $0.322610 ($0.000645/question). It stays outside CI because
both generation and judging consume live provider budget. See the
[complete answer contract and limitations](research/LONGMEMEVAL.md).

Adaptive v3 keeps top four for ordinary questions and top five for multi-session synthesis.
It composes complete v2 non-multi runs with complete top-five multi-session runs and scores
77.6% overall, 73.6% on the post-hoc validation partition, and 63.9% on multi-session
questions. Runtime cost remains $0.000760/question. Top six was rejected at 47/69
development answers versus top five's 51/69.

Adaptive v4 also uses top five for temporal reasoning. Complete development and validation
temporal runs raise that subtype from 71.4% to 79.7%, producing a composed 79.8% overall
score. Runtime provider cost remains $0.000786/question and no additional model or
embedding call is introduced.

Gated semantic v5 reranks multi-session questions only when the local top score is at most 315. It routes 115 of 133 multi-session questions, reaches 76.7% multi-session and 83.2%
overall accuracy, and raises combined retrieval Recall to 89.7%. Runtime provider cost is
$0.000872/question; the semantic latency is a cold isolated-corpus measurement.

Pass `--prepare-semantic` to separate document-maintenance time from the measured user turn.
On 20 development multi-session questions, preparation reduced user-turn p50/p95 from
20.8/27.3 seconds to 12.1/18.3 seconds and query embedding tokens from 1,836,242 to 249.
The maintenance phase increased total embedding cost about 19.7%; the result is a latency
tradeoff, not a lower total-cost claim.

## Structured-memory comparison

The v0.54 benchmark separates exact answers, answerability, ranked evidence retrieval,
citations, trust views, and deterministic replay. It runs without a model or network:

```bash
npm run bench:memory
npm run bench:memory:check
npm run bench:memory -- --json
npm run bench:memory:langgraph
npm run bench:memory:llamaindex
npm run bench:memory:external
npm run bench:memory:mem0
npm run bench:memory:graphiti
```

The public v1 suite contains eight questions over direct facts, 100 distractors, rule and
recursive derivations, a temporal update, honest abstention, tentative knowledge, and an
integrity conflict. Remembero is gated at full answer and citation correctness. Direct-fact,
lexical, and recency baselines disclose which capabilities they do not implement rather
than receiving misleading zero or perfect scores.

External memory stacks can run through a bounded, one-process-per-case JSON protocol:

```bash
npm run bench:memory -- --adapters remembero --external mem0=/absolute/path/to/bridge
npm run bench:memory -- --adapters remembero --external-manifest /path/to/adapter.json
```

The manifest form is the publication path: it pins package/model versions, declares actual
capabilities, records storage/write/retrieval/cost disclosures, and supplies a shell-free
command plus timeout. The checked-in LangGraph manifest uses `InMemoryStore`, FastEmbed
0.8.0, and `BAAI/bge-small-en-v1.5`; its first measured run achieved 100% Retrieval
Recall@k and MRR with zero operational errors and zero provider calls. It is retrieval-only,
so answer and citation metrics remain not applicable.

The LlamaIndex manifest pins `llama-index-core` 0.14.23 and its FastEmbed integration 0.6.0.
It stores one event-tagged `ChatMessage` per fixture event in `VectorMemory`, retrieves with
the same question/top-k/model, and also achieved 100% Recall@k/MRR with no provider calls.
`bench:memory:external` runs Remembero and both external adapters together.

`bench:memory:mem0` is a separate live-provider measurement. It requires
`OPENROUTER_API_KEY`, pins Mem0 OSS 2.0.14 and FastEmbed 0.8.0, disables Mem0 telemetry,
runs native Mem0 inference on every event, and records OpenRouter-native calls, input/output
tokens, and charged cost. It is deliberately excluded from CI and prepublish because the
full 118-call run consumes live provider budget and takes several minutes.

`bench:memory:graphiti` is another live-provider measurement. It pins Graphiti OSS 0.29.3,
uses native bulk episode formation and hybrid edge search, embeds locally with FastEmbed,
and creates a fresh embedded FalkorDBLite graph per question. The measured full run used
275 model calls and $0.037854 provider cost. It is excluded from CI and prepublish.

See the [benchmark contract and current results](research/MEMORY-STACK-BENCHMARK.md), the
[Medium draft](research/MEDIUM-DRAFT.md), and the [research paper](research/paper/paper.md).
The current checked-in comparison measures Remembero, transparent baselines, and pinned
LangGraph, LlamaIndex, Mem0 OSS, and Graphiti OSS adapters. It does not claim an unexecuted
result for Letta or managed Zep.

## Natural-language extraction and recall

Remembero measures both personal-knowledge extraction and recall at deterministic evidence
boundaries. The extraction runner scores the exact store mutation after validation; the
recall runner scores exact engine bindings before answer phrasing.

Both live runners also aggregate OpenRouter's provider-native input/output tokens, cached
and reasoning tokens, and charged cost from each response. Usage metadata is observational
and never participates in answer authority. `npm run bench:agent-db:cost` runs the default
grounded recall cost measurement.

## Extraction

Run the default model against the labeled extraction corpus:

```bash
npm run eval:extract
```

Compare models or select cases:

```bash
npm run eval:extract -- --models openai/gpt-5.6-luna,another/model
npm run eval:extract -- --cases replace_current_fact,derived_colleague_rule
npm run --silent eval:extract -- --json
```

Each case gets a fresh real `MemoryStore` and runs through `rememberText`. The 15-case
corpus covers exact facts, quoted and numeric values, schema reuse among 100 distractor
predicates, duplicates, replacement, removal, positive and negated rules, tentative
caller authority, non-factual input, policy and identity no-ops, and secret rejection
before any model call.

Rules are compared by alpha-equivalent canonical form, so harmless variable renaming does
not fail. Signed mutation precision/recall/F1 scores both added and removed clauses without
inflating results from unchanged initial facts. Exact-case accuracy additionally requires
the complete final state, added clauses, duplicate/retraction counts, expected rejection,
and the zero-call secret boundary to match.

## Recall

For recall, the live model translates each labeled question into Datalog, the real engine
evaluates it against a fixed memory corpus, and the runner compares the returned binding
rows with the expected rows. This isolates retrieval quality from prose style.

Run the default model against both prompt variants:

```bash
npm run eval:recall
```

Compare models or select cases:

```bash
npm run eval:recall -- --models openai/gpt-5.6-luna,another/model
npm run eval:recall -- --cases direct_employer,derived_colleague
npm run --silent eval:recall -- --json
```

Both commands load `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` exactly like the product
CLI. Runs are sequential and use temperature zero, but they are still live-model
measurements: provider or model changes can move the result. The checked-in corpus,
labels, normalizer, and metric tests remain deterministic.

The v0.7 corpus adds 100 unrelated predicate families to the real memory program, so
every labeled case exercises the same bounded schema-selection path used by a scaled
personal knowledge base.

Version 0.19 adds four confusable non-empty cases where the wrong related predicate also
returns plausible rows: owner versus implementation language and home versus workplace.
These cases exercise the bounded query-review path as well as final evidence scoring.
Version 0.20 adds a reusable aggregate-predicate case, bringing the deterministic corpus
to 25 cases. Version 0.21 adds one explicitly tentative recall case, bringing it to 26;
that case runs with `include_tentative` while every other case retains the accepted view.
Version 0.47 corrects three labels that had treated dentist/year helper bindings as answer
columns and scores explicit relational projection from the authoritative selected variables.

## Recall metrics

- **Accuracy**: percentage of cases with the correct query/unanswerable decision and an
  exact binding-row set.
- **Precision**: micro-averaged correct rows divided by all returned rows.
- **Recall**: micro-averaged expected rows that were returned.
- **F1**: harmonic mean of precision and recall.
- **Answerability**: whether the model correctly distinguishes a query the schema can
  express from a structurally unanswerable question. A valid query with no matching fact
  remains answerable and is distinct from `unanswerable`.
- **Schema budget exhaustion**: count of cases where bounded schema context could not
  establish a complete negative result. Exhaustion is always scored as incorrect rather
  than being confused with an accurate `unanswerable` decision.

Variable names and row order do not affect scoring. Multi-variable values retain the
variables' first-appearance order in the generated query, so swapping semantic roles does
not pass. Rows are deduplicated before comparison. A ground query that is true is
represented by one empty binding row; a ground query that is false has no rows. Scored
facts are held out from the sample facts included in the model-visible schema summary.

## Last live comparison

### Extraction

Measured on 2026-08-17 AEST with the v0.40 extraction contract:

| Model                       | Cases |   Accuracy | Mutation precision | Mutation recall | Mutation F1 |     Safety | Unexpected errors |
| --------------------------- | ----: | ---------: | -----------------: | --------------: | ----------: | ---------: | ----------------: |
| `openai/gpt-5.6-luna`       |    15 | **100.0%** |         **100.0%** |      **100.0%** |  **100.0%** | **100.0%** |             **0** |
| `google/gemini-3.7-flash`   |    15 | **100.0%** |         **100.0%** |      **100.0%** |  **100.0%** | **100.0%** |             **0** |
| `anthropic/claude-sonnet-5` |    15 | **100.0%** |         **100.0%** |      **100.0%** |  **100.0%** | **100.0%** |             **0** |
| `openai/gpt-5.4-mini`       |    15 |      93.3% |              91.7% |           91.7% |       91.7% | **100.0%** |             **0** |

GPT-5.4 Mini changed `dr_chen` to `chen` in the quoted-city case. The other three
models produced the exact expected mutations in this run.

### Recall

Measured on 2026-08-20 AEST after reducing the default detailed schema slice from 32 to 8.
All 26 Luna cases remained exact:

| Model                 | Cases |   Accuracy | Input tokens | Output tokens | Seconds | Charged cost | Average/query |
| --------------------- | ----: | ---------: | -----------: | ------------: | ------: | -----------: | ------------: |
| `openai/gpt-5.6-luna` |    26 | **100.0%** |      122,686 |         1,934 |    92.8 |    $0.016735 |     $0.000644 |

The previous 32-predicate run cost $0.022549. The new default reduced charged cost by
25.8% without changing accuracy, precision, recall, or answerability.

Measured on 2026-08-17 AEST with the v0.47 grounded projection prompt and deterministic
schema ranker. All 26 current cases ran among 100 distractor predicates with no schema-budget
exhaustion or transport errors:

| Model                       | Cases |   Accuracy |  Precision |     Recall |         F1 | Answerability | Budget exhausted |
| --------------------------- | ----: | ---------: | ---------: | ---------: | ---------: | ------------: | ---------------: |
| `openai/gpt-5.6-luna`       |    26 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |    **100.0%** |            **0** |
| `google/gemini-3.7-flash`   |    26 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |    **100.0%** |            **0** |
| `anthropic/claude-sonnet-5` |    26 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |    **100.0%** |            **0** |
| `openai/gpt-5.4-mini`       |    26 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |    **100.0%** |            **0** |

Explicit projection removed the prior helper-variable failures for GPT-5.4 Mini. See
[model compatibility](MODEL-COMPATIBILITY.md) for the combined recall/extraction
recommendation, observed catalog prices, latency, and evidence boundary.

The earlier pre-0.4, pre-scale baseline comparison was:

| Prompt             |   Accuracy |  Precision |     Recall |         F1 | Answerability |
| ------------------ | ---------: | ---------: | ---------: | ---------: | ------------: |
| baseline           |      94.7% |      94.4% |     100.0% |      97.1% |         94.7% |
| grounded (default) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |    **100.0%** |

The baseline failure answered a causal “why” question with a related fact. The grounded
prompt now treats schema examples only as syntax evidence, keeps named entities fixed,
distinguishes yes/no questions from requested unknowns, and requires a causal predicate
for causal questions. The shared fallback prompt separately distinguishes valid empty
retrievals from structurally unanswerable questions.

The scaled run is a current-tree checkpoint, not a statistical guarantee. Provider or
model changes can move it, which is why the deterministic corpus and scorer remain the
release gate.

This corpus is deliberately small and diagnostic rather than statistically representative.
Add a labeled case whenever a real recall failure is found, and compare against the
baseline before changing the default prompt.
