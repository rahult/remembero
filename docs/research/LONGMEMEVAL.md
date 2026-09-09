# LongMemEval-S retrieval and answer gates

Status: pinned retrieval and live end-to-end measurements, 20 August 2026 AEST

Remembero now runs the complete cleaned LongMemEval-S split: 500 questions over
multi-session conversational histories. The zero-provider runner measures deterministic
source retrieval. The live runner separately measures durable raw-session formation,
retrieval, answer generation, and task-specific answer judging so a strong retrieval score
cannot be presented as end-to-end accuracy.

## Reproduce it

```bash
npm run bench:longmemeval:download
npm run bench:longmemeval
npm run bench:longmemeval -- --json
npm run bench:longmemeval:answer -- --split dev --output /tmp/remembero-lme-dev.json
npm run bench:longmemeval:answer -- --split test --output /tmp/remembero-lme-test.json
```

The answer runner requires `LLM_API_KEY`, uses provider budget, and stays outside CI and
`prepublishOnly`. It defaults to the development partition. Run held-out only after the
policy is locked.

The download is pinned to dataset commit
`98d7416c24c778c2fee6e6f3006e7a073259d48f` and rejected unless its SHA-256 is
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The 277,383,467-byte dataset is cached under `.cache/`, never committed, and not needed by
normal tests or package installation.

## Current result

Default configuration: top five sessions, 16,384 source characters indexed per session,
minimum score 1.

| Metric                                   |          Result |
| ---------------------------------------- | --------------: |
| Questions                                |             500 |
| Answerable / abstention                  |        470 / 30 |
| Precision@5                              |          30.72% |
| Recall@5                                 |          83.27% |
| Mean reciprocal rank                     |          80.96% |
| All gold sessions retrieved              |          75.32% |
| Abstention queries returning no sessions |           0.00% |
| Local search p50 / p95                   | 9.90 / 10.78 ms |
| Model / embedding / remote calls         |       0 / 0 / 0 |

Three complete runs produced identical semantic metrics. Their p95 latency was
10.61–11.10 ms on an Apple M4 with Node 26.5. Timing is diagnostic and excludes dataset
loading and the one-time conversion of each question's sessions into source records.

The machine-readable measurement is
[`results/longmemeval-s-retrieval-v1-summary.json`](results/longmemeval-s-retrieval-v1-summary.json).

## End-to-end answer result v1

Each question gets a fresh real `MemoryStore`. Every timestamped session is committed as a
durable `longmem_session/1` fact with its transcript as source provenance. Remembero then
reconstructs the snapshot, retrieves four sessions, and sends at most 56 KiB of safe context
to the reader. Factual questions are history-only. Preference questions may combine recalled
personal context with general recommendation knowledge, but may not invent user details.

The policy was selected on a deterministic SHA-256 development partition, then run once on
the untouched held-out partition:

| Partition   |   Correct | Accuracy | Recall@4 | All-evidence answer accuracy | Errors |
| ----------- | --------: | -------: | -------: | ---------------------------: | -----: |
| Development | 206 / 261 |    78.9% |    85.8% |                        92.6% |      0 |
| Held-out    | 171 / 239 |    71.5% |    81.7% |                        85.9% |      0 |
| Combined    | 377 / 500 |    75.4% |    83.8% |                        89.5% |      0 |

The 42.3% combined accuracy when evidence was incomplete, versus 89.5% when it was
complete, identifies retrieval coverage—not polished answer prose—as the largest remaining
lever.

| Question type             | Questions | Accuracy |
| ------------------------- | --------: | -------: |
| Single-session user       |        70 |    94.3% |
| Single-session assistant  |        56 |    92.9% |
| Single-session preference |        30 |    90.0% |
| Knowledge update          |        78 |    80.8% |
| Temporal reasoning        |       133 |    70.7% |
| Multi-session             |       133 |    56.4% |

The live reader was `openai/gpt-5.6-luna`; recommendation intent alone could use the
selected Perplexity 0.6B embedder. The reader plus embeddings cost $1.371122 for all 500
questions, or $0.002742 per production-style answer. The separate
`openai/gpt-4o-2024-08-06` judge cost $0.161905. Provider-native usage—not catalog-price
multiplication—supplies every total.

Durable formation p95 was 180 ms and the 471 local retrievals had 11.5 ms p95. Cold
semantic preference retrieval was the slow path. After independent embedding batches were
bounded to concurrency three, a five-question development pilot retained 5/5 answers while
reducing semantic retrieval p50/p95 to 8.4/9.0 seconds and end-to-end p50/p95 to
16.3/24.5 seconds. Explicit `prepare_semantic_search` can move document work off the user
turn in a long-lived agent harness.

The machine-readable decision and result is
[`results/longmemeval-answer-v1-summary.json`](results/longmemeval-answer-v1-summary.json).
The runner can also emit the upstream two-field hypothesis JSONL. Its internal judge is
task-specific and official-compatible; this result was not independently rescored through
the upstream Python script. See the
[official benchmark and evaluator](https://github.com/xiaowu0162/LongMemEval).

## Role-aware answer result v2

The v1 reader repeatedly paid for long assistant replies even when the task asked about
facts supplied by the user. In the pinned dataset, 326 of 327 labelled multi-session answer
turns, all 259 temporal turns, all 144 knowledge-update turns, and all 44 preference turns
are user-authored. V2 therefore keeps full transcripts only for
`single-session-assistant`; every other reader prompt contains the retrieved user turns.
Retrieval still ranks full durable transcripts, so this changes neither session IDs nor
Recall@4.

| Partition   |   Correct | Accuracy | Reader tokens | Reader cost |  Median / p95 |
| ----------- | --------: | -------: | ------------: | ----------: | ------------: |
| Development | 211 / 261 |    80.8% |       645,845 |   $0.158195 | 11.7 / 38.4 s |
| Validation  | 174 / 239 |    72.8% |       613,282 |   $0.151552 | 13.4 / 56.7 s |
| Combined    | 385 / 500 |    77.0% |     1,259,127 |   $0.309746 | 12.6 / 55.4 s |

Compared with v1, overall accuracy rises 1.6 points, complete-evidence accuracy rises
3.1 points to 92.6%, reader tokens fall 76.7%, runtime provider cost falls 76.5%, median
latency falls 21.0%, and p95 falls 14.3%. Reader plus embedding cost is $0.322610, or
$0.000645 per question. The independent judge adds $0.160087 evaluation-only cost.

Multi-session accuracy rises from 56.4% to 61.7%; its reader tokens fall from 1,456,321 to
207,795 across 133 questions. Single-session assistant accuracy remains 92.9% because those
questions retain both roles.

The role-distribution audit examined answer-turn roles over the full dataset before v2 was
run. The 239-question v2 validation result is therefore post-hoc validation, not a new
pristine held-out claim. The original v1 held-out result remains the sealed result. See the
[v2 machine-readable evidence](results/longmemeval-answer-v2-summary.json).

Several development candidates were rejected. Semantic reranking raised multi-session
Recall@4 to 91.2% but did not improve complete answer accuracy after its extra latency and
cost. A low-local-score semantic gate also ended at the same 46/69 answers as local search.
Top-six context, an aggregation-specific prompt, and GPT-5.4 Mini each scored worse on the
fixed development slice. These negative results are retained in the v2 artifact.

## Adaptive top-k result v3

Role-aware context makes one additional session affordable for multi-session synthesis.
V3 keeps top four for every other question type and uses top five only for multi-session
questions. Complete development and validation multi-session runs improve from 82/133 to
85/133 correct and raise multi-session Recall from 74.2% to 78.1%.

The composed 500-question policy reaches 388/500 (77.6%), with 81.2% development and 73.6%
post-hoc validation accuracy. Runtime reader-plus-embedding cost is $0.379880, or
$0.000760/question—72.3% below v1. Reader tokens remain 75.9% below v1. Top-six was rerun
with role-aware context and rejected at 47/69 development answers versus top five's 51/69,
with two additional abstention regressions.

V3 is composed from the complete v2 non-multi observations and complete top-five
multi-session observations. It is not a fresh sealed run, and live reader/judge variance
still applies. See the
[v3 machine-readable evidence](results/longmemeval-answer-v3-summary.json).

## Temporal top-k result v4

Temporal questions also benefit from one additional dated session. Top five improves the
complete development temporal run from 48/66 to 55/66 and the post-hoc validation run from
47/67 to 51/67. Combined temporal accuracy rises from 71.4% to 79.7%, Recall rises from
76.9% to 80.9%, and all six temporal abstention questions remain correct.

V4 therefore uses top five for multi-session and temporal questions and top four elsewhere.
The composed policy reaches 399/500 (79.8%), with 83.9% development and 75.3% post-hoc
validation accuracy. Reader plus embedding cost is $0.392863, or $0.000786/question—71.3%
below v1—and no extra provider call is added. See the
[v4 machine-readable evidence](results/longmemeval-answer-v4-summary.json).

## Gated multi-session semantic result v5

Semantic multi-session retrieval was re-tested under role-aware top-five context. Global
semantic routing reaches 55/69 development and 46/64 validation answers, unlike the earlier
full-transcript experiment where higher Recall did not improve answers. A deterministic
local-score gate performs better: rerank only when the local leader scores at most 315.

The gate routes 60/69 development and 55/64 validation questions. It produces 56/69 and
46/64 correct answers, saves 55 embedding calls versus global routing, and raises combined
multi-session accuracy from 63.9% to 76.7%. Multi-session Recall reaches 93.0%.

The composed v5 policy reaches 416/500 (83.2%), with 85.8% development and 80.3% post-hoc
validation accuracy. Runtime reader-plus-embedding cost is $0.435901, or
$0.000872/question—68.2% below v1. The semantic timing is cold: every isolated benchmark
case recomputes document vectors; a prepared long-lived harness pays only for the query
embedding. See the [v5 evidence](results/longmemeval-answer-v5-summary.json).

## Prepared semantic user turns

`--prepare-semantic` runs the same bounded document preparation before measuring the user
turn. On the first 20 development multi-session questions, cold user-turn p50/p95 was
20.8/27.3 seconds. Prepared p50/p95 was 12.1/18.3 seconds, and query embedding tokens fell
from 1,836,242 to 249.

The work moves rather than disappears: maintenance used 2,196,997 tokens and total embedding
cost rose about 19.7%. Accuracy is not compared because live reader/judge outcomes varied
between runs while retrieval IDs and Recall stayed fixed. See the
[preparation evidence](results/semantic-preparation-v1-summary.json).

## What is actually indexed

Each LongMemEval session is represented by one opaque `longmem_session(session_N).` carrier
fact. The full transcript is attached as durable source provenance with the dataset's real
session ID. `searchKnowledge` ranks those sources against the question and returns the
source session IDs. No answer text, gold evidence ID, model-generated memory, embedding, or
handwritten case data is placed in the query.

This representation is also used by the answer runner, but through real durable store
writes and snapshot reconstruction rather than an in-memory source map shortcut. It proves
raw-session formation, not model-extracted structured-fact accuracy.

## Why the default source window changed

The former 4,096-character window truncated 97.6% of gold session occurrences. In the
pinned dataset, 198 labelled answer turns begin after character 4,096 and 89 begin after
8,192. The median session is 14,393 characters; p95 is 19,476 and the maximum is 28,108.

The same code and dataset produced:

| Characters per source | Precision@5 | Recall@5 |    MRR | All evidence |      p95 |
| --------------------: | ----------: | -------: | -----: | -----------: | -------: |
|                 4,096 |      29.21% |   79.67% | 78.10% |       70.00% |  5.30 ms |
|                 8,192 |      30.00% |   82.70% | 81.96% |       74.26% |  9.09 ms |
|                16,384 |      30.72% |   83.27% | 80.96% |       75.32% | 10.29 ms |
|                32,768 |      30.81% |   83.33% | 80.63% |       75.32% | 10.41 ms |

The 16 KiB default captures nearly all of the 32 KiB recall gain without making 32 KiB the
ordinary cost. Search also caps aggregate source text considered in one call at 32 MiB and
reports both requested and effective per-source limits.

Conversational stopwords are removed before ranking. On the same 4 KiB window, that change
raised Recall@5 from the pre-change 74.97% measurement to 79.67% and strict all-evidence
coverage from 63.40% to 70.00%.

## Precision and abstention are visible weaknesses

`minimumScore` is an explicit search option and benchmark flag. The threshold frontier is
published instead of selecting the best value after seeing the full test set:

| Minimum score | Precision@5 | Recall@5 |    MRR | All evidence | Abstention empty |
| ------------: | ----------: | -------: | -----: | -----------: | ---------------: |
|             1 |      30.72% |   83.27% | 80.96% |       75.32% |            0.00% |
|            90 |      32.67% |   82.46% | 80.67% |       74.47% |            3.33% |
|           135 |      38.34% |   78.79% | 78.86% |       70.64% |           20.00% |
|           180 |      42.46% |   70.33% | 72.44% |       61.91% |           36.67% |

The product keeps the recall-first default. Developers can raise the threshold when false
context is more expensive than missed context, but the current lexical search does not yet
provide reliable abstention.

In the retrieval-only baseline, preference questions are the largest gap: 43.33% Recall@5 versus 92.86% for
single-session assistant facts and 95.31% for single-session user facts. That makes learned
semantic retrieval or a purpose-built preference index a concrete next experiment rather
than an unmeasured feature claim.

That experiment is now measured. A locked recommendation-intent policy using the opt-in
semantic tool raises all-preference Recall@5 to 73.33% and held-out Recall@5 from 46.67% to
60.00%. It does not change the recall-first lexical default or use similarity for
abstention. See [semantic knowledge search](../SEMANTIC-KNOWLEDGE-SEARCH.md) for the split,
cost, cache, and export-safety boundary.

## Extraction formation: the product's own write path on the benchmark (2026-09-10)

Everything above forms memory as one placeholder fact per session carrying the raw transcript,
so the product's extraction (the fine-tuned 4B model writing Datalog) had never been exercised
here. `run-longmemeval-answer.js --formation` now offers three formations: `raw` (the above),
`extracted` (only what `rememberTranscriptText` writes for each session exists, one extraction
call per session), and `hybrid` (both, with the extracted facts under their own operation id
mapped to the session). Retrieval in the extracted formations fetches a wider pool and counts
top-k in distinct sessions. The extractor is its own OpenAI-compatible client
(`--extraction-model`, `--extraction-base-url`, `--extraction-api-key`), here the Qwen3.5-4B
fine-tune served from Modal. Assistant turns are 87% of the characters and never a source of
facts under the extraction contract, so `--extraction-assistant-characters 600` keeps only
their head. Reader Luna, judge GPT-4o, lexical retrieval only (`--local-only`), so formation is
the only variable.

The slice is the first eight development questions of each of the six types (48 questions,
`runs/modal/lme-cases48.txt`); the first forty development questions turned out to be almost
all single-session-user and scored 38/40 raw, which left nothing to measure.

| formation (extractor) | accuracy  | retrieval recall | k-update | multi | ss-asst | ss-pref | ss-user | temporal |
| --------------------- | --------- | ---------------- | -------- | ----- | ------- | ------- | ------- | -------- |
| raw                   | 32/48     | 74.1%            | 5/8      | 2/8   | 6/8     | 4/8     | 8/8     | 7/8      |
| hybrid (r13)          | **35/48** | **76.8%**        | 6/8      | 2/8   | 6/8     | 6/8     | 8/8     | 7/8      |
| extracted only (r12)  | 22/48     | 46.2%            | 6/8      | 0/8   | 2/8     | 3/8     | 8/8     | 3/8      |
| extracted only (r13)  | 23/48     | 38.5%            | 5/8      | 5/8   | 1/8     | 4/8     | 5/8     | 3/8      |
| hybrid (r14)          | **37/48** | **80.3%**        | 7/8      | 4/8   | 6/8     | 6/8     | 8/8     | 6/8      |
| extracted only (r14)  | 29/48     | 66.2%            | 5/8      | 3/8   | 4/8     | 4/8     | 7/8     | 6/8      |

Extraction cost per formation: about 2,450 calls (one per session), 1,100–1,400 facts from
680–830 sessions, 40–46 refused or malformed (under 2%; the first round-11 attempt had 39%
failing, mostly a 4096-token context limit since raised to 8192 and an operation-id collision
in hybrid since fixed).

What this says, with the caveat that 48 questions make three answers about one standard
error:

- **Hybrid beats raw.** Adding the small model's facts to the raw memory lifted three answers
  and retrieval recall by 2.7 points; the gains are in preference and knowledge-update
  questions, where a stored `prefers(user, …)` or the latest value matches the question's
  words better than the transcript does.
- **Facts alone are not enough for this benchmark, and never will be for two of its types.**
  Single-session-assistant questions ask what the assistant said, which the extraction
  contract deliberately ignores; temporal questions need the session dates, which the
  facts' `at` carries but the lexical reader path does not use. Extracted-only recall is
  under half of raw's.
- **Facts win where raw loses: aggregation.** Multi-session questions ("how many model kits
  have I bought") went from 2/8 raw to 5/8 with r13's facts alone, because five
  `bought(user, …)` facts across five sessions are retrievable together while five long
  transcripts are not.
- **The extractor was the bottleneck, and the data fixed most of it.** Probing r12 on the
  evidence sessions showed the misses were asides ("By the way, I just got back from a
  three-day trip to Big Sur") and an absent vocabulary for events. r13 (facts embedded in
  long requests, empty schemas) doubled multi-session; r14 added an event kind for exactly
  these asides and extracted facts from 63% of sessions instead of 28%, taking
  extracted-only from 23 to 29 of 48 (recall 38% to 66%) and hybrid from 35 to 37, five
  answers above raw. Round over round on the same slice: extracted-only 22 → 23 → 29,
  hybrid 32 (raw) → 35 → 37.

## Evidence boundary

The retrieval result remains the reproducible zero-provider baseline. The answer results add
live raw-session formation and QA evidence, but it does not establish:

- learned structured-fact formation accuracy over the complete transcript stream;
- superiority over another memory product under the same reader, judge, and policy;
- deterministic generation or judging—one development preference slice moved between
  repeated temperature-zero runs, so live-model variance remains real;
- reliable multi-session aggregation or learned abstention confidence—v5 reaches 76.7%
  multi-session accuracy but remains below the single-session results;
- hosted, concurrent, multi-tenant, or sustained-provider performance.

The held-out run initially contained one provider HTTP 400 caused by a lone UTF-16 high
surrogate in source text. Provider-boundary Unicode scalar normalization fixed the defect;
the exact case was retried without changing retrieval, prompts, models, or scoring. The
final 239-question held-out artifact has zero errors. This repair is disclosed in the
machine-readable result rather than silently dropping the case.
