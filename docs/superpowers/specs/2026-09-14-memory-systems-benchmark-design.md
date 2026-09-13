# Memory systems benchmark: one harness, one reader, one judge

Date: 2026-09-14. Status: approved design, not yet built.

## Why

Every published LongMemEval number comes from a different answer model and a different
judge: Mem0 reports 94.4%, Synap 92.0% with gpt-5-mini, Supermemory 85.2% with Gemini-3 Pro,
Zep 71.2% with gpt-4o. None of those is a statement about the memory layer. SWE-bench is
useful because the task set, the harness and the scoring are fixed and only the system under
test varies. This benchmark does the same for conversational memory: the same 500
LongMemEval-S questions, the same reader, the same judge, the same context builder, and only
the memory layer changes. Remembero is one row, measured exactly as every other system.

## What is held constant

| variable | value | why this value |
| --- | --- | --- |
| task set | LongMemEval-S cleaned, all 500 questions (`xiaowu0162/longmemeval-cleaned`, commit and sha256 already recorded by the harness) | the public set every vendor cites |
| iteration set | the 100-question stratified subset (`.cache/longmemeval/subset-100.txt`, seed 100) | one fifth of the cost; the 500 only for the final table |
| reader | GLM 5.3 Flash through Ollama Cloud (`glm-5.3-flash:cloud`, `http://127.0.0.1:11434/v1`), thinking off, `--reader-max-tokens` as the stored 440 run | the reader of Remembero's best stored run; answers from here today |
| judge | `deepseek-chat` (DeepSeek direct API), LongMemEval official-compatible protocol | the judge every stored run was re-judged with |
| top-k | 4, multi-session 15, temporal 10 | the composed policy the stored runs use |
| context | 24,576 bytes, 16,384 source characters per session, dated session headers (`--date-distances`) | as the stored runs |
| time-aware retrieval | off for everyone | it is a Remembero-side query rewrite; keeping it out isolates the memory layer |
| computed notes, structured evidence | off in the retrieval lane; Remembero's as-shipped row has them on | reader-side structure, not memory |

## Two lanes

**Retrieval lane.** The adapter ingests the question's sessions and returns ranked session
ids. The harness gives the reader the same top-k raw sessions, through the same context
builder, that it gives Remembero. What differs between rows is only which sessions were
found. Recall@k against the gold session ids is reported next to answer accuracy.

**As-shipped lane.** The adapter returns its own memory text, the material a user of that
system would put in front of a model, and the reader answers from that text alone, inside
the same byte budget. Remembero's as-shipped row is hybrid formation with the local writer
(r23), computed notes and structured evidence. Systems that cannot produce memory text
(pure retrievers) report the lane as unsupported rather than getting a proxy.

## Systems

| row | memory layer | LLM at ingest | embeddings |
| --- | --- | --- | --- |
| full context | the haystack truncated to the byte budget, newest first | none | none |
| BM25 | plain BM25 over whole sessions, in the harness | none | none |
| embedding top-k | nomic-embed-text through local Ollama, cosine over whole sessions | none | local |
| Remembero | its lexical source search (raw formation), and hybrid formation as shipped | r23 local writer (as-shipped row only) | none |
| LangGraph | `InMemoryStore` semantic search | none | FastEmbed bge-small local |
| LlamaIndex | `VectorMemory` over `SimpleVectorStore` | none | FastEmbed bge-small local |
| Mem0 OSS | native formation, search | DeepSeek V4 Flash | FastEmbed bge-small local |
| Graphiti OSS | `add_episode_bulk`, hybrid edge search | DeepSeek V4 Flash | FastEmbed bge-small local |

Package versions are pinned in each adapter manifest exactly as the conformance suite pins
them today. The three baselines live in the harness because they have no product to pin.

## Protocol: `rembero.memory-systems.v1`

One long-lived process per adapter per run, JSON lines on stdin and stdout, spawned without
a shell, stderr counted and discarded, a byte limit per response and a timeout per request.
The conformance suite's one-process-per-case shape is kept for that suite; 500 questions
cannot pay model load 500 times.

Request, one line per question:

```json
{"protocolVersion":"rembero.memory-systems.v1","questionId":"ba358f49_abs",
 "questionType":"temporal-reasoning","questionDate":"2023-07-10",
 "question":"...","topK":10,"lanes":["retrieval","memories"],
 "sessions":[{"id":"5a81277c","date":"2023-05-15","turns":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}]}
```

Response, one line:

```json
{"questionId":"ba358f49_abs",
 "retrieved":[{"sessionId":"5a81277c","rank":1,"score":0.83}],
 "memories":[{"text":"User's dentist is Dr Chen","sessionIds":["5a81277c"],"at":"2023-05-15"}],
 "unsupported":[],
 "usage":{"modelCalls":48,"inputTokens":120000,"outputTokens":3000,"costUsd":0.03},
 "wallMs":{"ingest":41000,"search":120}}
```

Rules: a fresh, isolated store per question (a distinct user or namespace inside one process
is acceptable, a shared store across questions is not); `retrieved` contains only ids from the
request; `memories` may be empty with `"memories"` listed in `unsupported`; the harness
truncates memory text to the context budget and records what it dropped. The adapter never
sees the question type's gold answer. Manifests carry the package pins, models, storage,
write and retrieval policy, and the provider-cost boundary, in the existing manifest schema
plus a `protocol` field.

## Harness changes

`run-longmemeval-answer.ts` gains `--memory-system <manifest>` and `--memory-lane
retrieval|memories`. When set, the per-question evaluation skips Remembero's formation and
search and calls the adapter instead; everything after retrieval is unchanged, so the result
JSON has the same schema, `retrieval` names the adapter id, and the DeepSeek re-judge sidecar
and the summary tables work as they do for every stored run. The three baselines are
`--memory-system builtin:full-context|bm25|embed` with no process. Adapter usage and wall
time land in the observation next to the reader and judge usage, so the cost column is
measured, not estimated.

## Outputs

- `docs/research/results/memory-systems-<system>-<lane>-subset100.json` and `-all500.json`,
  with the DeepSeek sidecars.
- `docs/research/MEMORY-SYSTEMS-BENCHMARK.md`: the constants table above, the leaderboard
  (accuracy per question type, recall@k, ingest cost, ingest and search time), what each
  system was and was not given, and the paragraph on what the benchmark cannot establish.
- A site section under Examples: the leaderboard and one sentence on what is held constant.
- `npm run bench:memory-systems` runs the subset for every system that needs no paid model;
  the Mem0 and Graphiti runs are documented commands that need `DEEPSEEK_API_KEY` and stay
  out of CI.

## Cost and time

About 24,000 session ingests per system on the 500 (47.7 sessions a question; 19,195
distinct sessions in the pool, but stores are per question). DeepSeek V4 Flash off-peak is
$0.22 per million input tokens and $0.66 output (peak is double; peak hours are 01:00-04:00
and 06:00-10:00 UTC). Mem0 at about 120k input tokens a question is roughly $15-25 for the
500; Graphiti makes two to three times the calls. Ingest wall time six to twelve hours per
system at concurrency four to eight. The subset is a fifth of all of that. Reader and judge
cost per run is what the stored runs already show: cents.

## Order of work

1. Protocol and harness option, with the three built-in baselines and a stub adapter test.
2. Remembero rows through the same option (raw retrieval; as-shipped) to prove the path
   reproduces the stored numbers within noise (about ±4 on the subset).
3. LangGraph and LlamaIndex bridges ported to the JSON-lines protocol (no paid model).
4. Mem0 and Graphiti bridges with DeepSeek; subset first, cost read from the observations.
5. The 500 for every system, the document, the site section.

## What this cannot establish

The reader is fixed at one open model; a system whose memory text is tuned for a frontier
model may read worse here. Ingest cost depends on the LLM the system was pointed at. The
retrieval lane rewards systems that return whole sessions and cannot score a system that only
returns facts, which is why the as-shipped lane exists. Judge noise on identical prompts is
about ±4 on 100 and ±7 on 500 questions; differences inside that band are not ranked.
