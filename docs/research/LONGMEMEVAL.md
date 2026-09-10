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
| hybrid (Luna)         | 33/48     | 77.4%            | 6/8      | 2/8   | 5/8     | 5/8     | 8/8     | 7/8      |

Extraction cost per formation: about 2,450 calls (one per session), 1,100–1,400 facts from
680–830 sessions, 40–46 refused or malformed (under 2%; the first round-11 attempt had 39%
failing, mostly a 4096-token context limit since raised to 8192 and an operation-id collision
in hybrid since fixed).

The Luna row is the frontier model doing the same job under the same guards and prompt, with
a 4,096-token completion budget so its reasoning has room (at the 512-token default it
returned no content on 533 of 2,841 calls and scored 32/48). It wrote 8,474 facts against
r14's 2,329, 71 were refused, and the extraction alone cost $2.35 for the slice; r14's cost
is L4 time, about thirty cents.

### The full development split (261 questions)

The slice suggested hybrid formation was worth five answers. Two runs of the same r14 extractor
on the slice then agreed on retrieval in only 25 of 48 questions and differed by four answers,
so the whole development split was run: raw, hybrid r14 with the retrieved sessions' matched
facts listed to the reader under the session date, and hybrid r14 without them.

| formation (dev, 261)        | accuracy    | recall | k-update | multi | ss-asst | ss-pref | ss-user | temporal |
| --------------------------- | ----------- | ------ | -------- | ----- | ------- | ------- | ------- | -------- |
| raw                         | **214/261** | 83.4%  | 37/44    | 50/69 | 26/28   | 11/15   | 38/39   | 52/66    |
| hybrid r14, facts shown     | 211/261     | 85.2%  | 39/44    | 51/69 | 26/28   | 8/15    | 38/39   | 49/66    |
| hybrid r14, facts not shown | 209/261     | 84.9%  | 39/44    | 48/69 | 27/28   | 9/15    | 37/39   | 49/66    |

Hybrid gained 12 questions and lost 15 against raw; the standard error at this size is about
six answers, so the two are indistinguishable on accuracy. Retrieval recall is up 1.5–1.8
points, knowledge-update up two, multi-session up one, preference and temporal down three
each. Listing the matched facts to the reader changed two answers, which is nothing. Each
hybrid run made about 14,000 extraction calls, wrote about 12,200 facts from 8,300 of the
12,500 sessions, and had 4.9% of calls refused or malformed; on the L4 that is about $2 of GPU
time per run.

What this says:

- **On this benchmark, the product's extracted facts do not raise answer accuracy over raw
  transcript retrieval, and they do raise retrieval recall.** The five-answer gain on the
  48-question slice, and the four-answer lead over Luna as extractor there, were slice noise.
  The Luna comparison has not been run on the full split (it would cost about $13 of Luna
  extraction), so no claim about the 4B model beating the frontier model as an extractor
  survives; the claim that survives is that it does the same job for a fraction of the cost.
- **Where facts help and hurt is consistent across the slice and the split.** Knowledge
  updates and multi-session counting improve, because a stored latest value or five
  `bought(user, …)` facts match the question better than five transcripts do. Preference
  and temporal questions lose: a preference answer needs the texture of the conversation,
  not a `prefers(user, x)` fact, and the extra retrieved facts displace transcript context
  the reader needed for dates.
- **Facts alone are not enough for this benchmark, and never will be for two of its types.**
  Single-session-assistant questions ask what the assistant said, which the extraction
  contract deliberately ignores; extracted-only recall on the slice was under half of raw's.
- **The extractor was the bottleneck, and the data fixed most of it.** Probing r12 on the
  evidence sessions showed the misses were asides ("By the way, I just got back from a
  three-day trip to Big Sur") and an absent vocabulary for events. r13 (facts embedded in
  long requests, empty schemas) and r14 (an event kind for exactly these asides) took
  extraction from 28% to 63% of sessions and extracted-only from 22 to 29 of 48 on the slice.
- **Reserved retrieval was tried next and did worse.** `--hybrid-retrieval reserved` fills
  top-k from raw text exactly as raw formation does and appends up to 3k matched extracted
  facts as a dated block, so facts add context instead of competing for slots. On the full
  split: **206/261**, gained 7 and lost 15 against raw, with abstention accuracy down from
  95% to 84% and knowledge-update down two. The block gives the reader loosely matching facts
  from other sessions, which is exactly the material it needs to answer a question it should
  decline, and it shows stale values next to current ones because nothing supersedes a fact
  across sessions in this store (no `rembero_functional` declarations exist for invented
  predicates). Its 91% "retrieval recall" is inflated by counting every fact's session.

| formation (dev, 261)                               | accuracy    | recall | abstention | k-update | multi | ss-pref | temporal |
| -------------------------------------------------- | ----------- | ------ | ---------- | -------- | ----- | ------- | -------- |
| raw                                                | **214/261** | 83.4%  | 95%        | 37/44    | 50/69 | 11/15   | 52/66    |
| hybrid r14, shared                                 | 211/261     | 85.2%  | 95%        | 39/44    | 51/69 | 8/15    | 49/66    |
| hybrid r14, reserved                               | 206/261     | 91.2%* | 84%        | 35/44    | 49/69 | 10/15   | 50/66    |
| hybrid Gemma 4 E2B, shared                         | **219/261** | 85.4%  | 89%        | 38/44    | 55/69 | 10/15   | 52/66    |
| E2B shared, routed to k-update + multi             | 210/261     | 84.4%  | 84%        | 35/44    | 50/69 | 9/15    | 53/66    |
| E2B reserved ≥200, routed, framed                  | 215/261     | 84.2%  | 95%        | 37/44    | 49/69 | 10/15   | 54/66    |
| hybrid Gemma 4 E2B r17 (real-session data), shared | 201/261     | 84.1%  | –          | –        | –     | –       | –        |

\* counts the sessions of appended facts as retrieved.

- **A different extractor on the same data beats raw.** The Gemma 4 E2B fine-tune (see the
  base-model matrix in [EXTRACTION-BENCH.md](EXTRACTION-BENCH.md), where it ties Qwen3.5-4B on
  the benchmarks) run as the extractor in shared hybrid formation scored **219/261**: gained
  20 and lost 15 against raw, gained 24 and lost 16 against the r14 hybrid, with multi-session
  up from 50 to 55 and no type below raw by more than one. It wrote 12,837 facts from 5,777
  sessions, fewer sessions than r14 (8,331) but more facts each, in 16,734 calls (more
  retries) with 4.4% refused. The margin over raw is under one standard error, so this is
  the first hybrid that is at least as good as raw rather than proof it is better; it is
  also the first time two extractors trained on identical data differed by eight answers,
  which says the extractor's habits (which sessions it writes for, how many facts) matter
  as much as the benchmark scores that could not separate them.

- **Routing by type and thresholding the fact block, on the full split.** Two follow-ups
  with the Gemma 4 E2B extractor: shared hybrid only for knowledge-update and multi-session
  questions (the other types run raw, which also halves the extraction calls) scored
  **210/261**; reserved retrieval with a two-matched-word floor (`--reserved-min-score 200`)
  and the block framed as supplementary scored **215/261** with abstention back at 95%, so
  the framing repaired the reserved mode's abstention loss. Neither beat raw.

- **More extraction is not better extraction, for this retrieval.** Round 17, the Gemma 4 E2B
  recipe plus 3,000 Luna-labelled real sessions, nearly triples loose fact recall on real
  sessions (see [EXTRACTION-BENCH.md](EXTRACTION-BENCH.md)) and scores **201/261** here in
  shared hybrid formation, thirteen below raw. It wrote 40,671 facts against r16's 12,837 from
  about the same number of sessions; in shared retrieval those facts compete with raw sessions
  for the same slots and win them with low-value matches. r16 stays the served model. The
  fact-augmented-key formation below, where facts enrich a session's key instead of competing
  as documents, is the setting in which a more prolific extractor could pay off; it has not
  been run with r17.

- **Multi-session is a retrieval problem with a known fix: more slots.** When every evidence
  session is retrieved the reader answers 44 of 44 multi-session questions; with some missing
  it answers 3 of 16. Multi-session questions have two to five evidence sessions and the type
  had a top-k of five. Retrieval-only simulation on raw formation gives full-evidence
  retrieval for 40, 46, 48, 55 and 57 of the 69 questions at k = 5, 8, 10, 15 and 20. Runs
  on the 69 questions alone, Gemma 4 E2B as extractor, other settings unchanged:

| multi-session only (69)               | correct | recall |
| ------------------------------------- | ------: | -----: |
| raw, k=5 (from the full-split run)    |      50 |  73.0% |
| hybrid, k=5 (from the full-split run) |      55 |  78.0% |
| raw, k=15                             |      57 |  94.4% |
| hybrid, k=10                          |      57 |  93.6% |
| **hybrid, k=15**                      |  **61** |  93.6% |
| hybrid, k=5 + entity retrieval        |      48 |  85.9% |
| hybrid, k=15 + entity retrieval       |      57 |  95.1% |
| hybrid, k=15 + entity + 120KB context |      57 |  94.3% |

Raising this one type's k to 15 lifts it from 50 to 61 with nothing else touched, since
the flag is per type. The entity-keyed hop over the fact store
(`--entity-retrieval`, one hop through shared relation-and-subject or shared entity)
loses at every k: at k=5 its hits displace better lexical sessions, and at k=15 recall is
already 95% so it can only add noise. A wider reader context did not recover the
remaining reader errors. The extractor's contribution is the four answers between raw and
hybrid at k=15, both at about 94% recall: when both formations retrieve the evidence, the
facts listed under each session help the reader count.

### Patterns from the literature, composed (2026-09-11)

[LONGMEMEVAL-PATTERNS.md](LONGMEMEVAL-PATTERNS.md) reviews the paper's ablations and the
public systems' reports. Three patterns with numbers behind them were implemented on the
answer run and measured on the development split with the Gemma 4 E2B extractor, using a
per-session extraction cache so each variant cost reader time only:

| variant (dev, 261)                                         | accuracy    | recall | k-update | multi | ss-asst | ss-pref | ss-user | temporal |
| ---------------------------------------------------------- | ----------- | ------ | -------- | ----- | ------- | ------- | ------- | -------- |
| raw                                                        | 214/261     | 83.4%  | 37/44    | 50/69 | 26/28   | 11/15   | 38/39   | 52/66    |
| hybrid, multi-session k=15                                 | 219/261     | 86.5%  | 37/44    | 60/69 | 27/28   | 7/15    | 37/39   | 51/66    |
| keyed (facts in the session key), multi-session k=15       | 214/261     | 86.7%  | 36/44    | 55/69 | 26/28   | 10/15   | 38/39   | 49/66    |
| keyed + turn units (user turns only), k=15, 120KB context  | 218/261     | 86.0%  | 37/44    | 60/69 | 21/28   | 12/15   | 39/39   | 49/66    |
| keyed + turn units (all roles), multi-session k=15         | 216/261     | 87.7%  | 36/44    | 54/69 | 28/28   | 11/15   | 36/39   | 51/66    |
| **hybrid, multi-session k=15, temporal k=10 + time range** | **224/261** | 88.5%  | 37/44    | 60/69 | 28/28   | 8/15    | 36/39   | 55/66    |
| same, with turn units (all roles)                          | 218/261     | 86.5%  | 34/44    | 60/69 | 27/28   | 10/15   | 35/39   | 52/66    |

The composed policy gains 21 questions against raw and loses 11, ten net on 261 with a
standard error of about six, and keeps abstention at 95%. What each pattern did:

- **Multi-session top-k 15** is the whole multi-session gain (50 → 60) and the largest
  single lever found; see the multi-session section above.
- **Time-aware retrieval** (`--temporal-range-model`, Luna reads the absolute range off the
  question and question date, refusing when there is no cue, which it did for 46 of 66)
  with temporal top-k 10 takes temporal from 52 to 55–56 and temporal recall from 75.6% to
  87.5%; on the 66 questions alone raw scored 52, range 55, range + k=10 56.
- **Fact-augmented keys** (the paper's best BM25 pattern) did not beat shared hybrid here:
  keyed alone lost the multi-session gain, because on this benchmark the separate fact
  documents are what make a session with several matching facts outrank one with a long
  transcript. The pattern was measured with BM25 over whole sessions in the paper; our
  scorer already reads source text, so the key was not adding information.
- **Turn-level units** raise recall to the highest seen (87.7%) and fix assistant-memory
  questions once assistant turns are indexed (21 → 28 of 28), but cost multi-session and
  knowledge-update answers in the combination and net out below the session unit.
- **Entity-keyed retrieval** over the fact store lost at every k (multi-session section).

### The composed policy on all 500 questions

| all 500, lexical retrieval only                                          | accuracy    | recall | abstention | k-update | multi   | ss-asst | ss-pref | ss-user | temporal |
| ------------------------------------------------------------------------ | ----------- | ------ | ---------- | -------- | ------- | ------- | ------- | ------- | -------- |
| raw                                                                      | 391/500     | 82.7%  | 80%        | 66/78    | 83/133  | 52/56   | 21/30   | 65/70   | 104/133  |
| **hybrid (Gemma 4 E2B), multi-session k=15, temporal k=10 + time range** | **414/500** | 88.6%  | 90%        | 67/78    | 101/133 | 50/56   | 19/30   | 64/70   | 113/133  |
| same + v5 semantic routing (embeddings on 95 questions)                  | **416/500** | 92.4%  | 93%        | 66/78    | 103/133 | 51/56   | 23/30   | 63/70   | 110/133  |
| same + structured reading (dated notes, then an Answer line)             | 416/500     | 92.4%  | 90%        | 64/78    | 102/133 | 50/56   | 27/30   | 65/70   | 108/133  |

Gained 52 and lost 29 against raw; development 212 → 226, held-out test 179 → 188; about two
and a half standard errors on 500. The two types the patterns targeted carry it: multi-session
+18 and temporal +9. Preference and assistant-memory drift by two, inside the reader noise.
This is the first LongMemEval result in this document where the product's own extraction, run
by a 2.3B-effective model, moves the score, and it does so with lexical retrieval alone; the
earlier v5 policy (416/500) needed embedding-based routing on top of raw sessions. The two are
complementary and have not been combined.

Adding the v5 semantic route (embedding rerank for recommendation-intent preference questions
and for multi-session questions whose lexical leader scores at most 315; it fired on 95
questions, $0.04 of embeddings) gives **416/500**, development 223, test 193 (v5 alone: 416,
192 on test). Retrieval recall reaches 92.4%, the highest measured, yet accuracy moves two
answers: the reader's accuracy on questions whose evidence is fully in context fell from
93.4% (raw) to 89.4% as the contexts grew to fifteen sessions, and 43 fully-evidenced
questions are still wrong, 21 of them multi-session and 12 temporal. Retrieval is no longer
the limit; the reader's aggregation over many sessions is.

The paper's reading pattern (`--reading notes`: for multi-session, temporal and
knowledge-update questions the reader lists every relevant dated item, then gives an
"Answer:" line that alone is judged) was the natural next step and did nothing: 416 again,
with 14 gained and 21 lost on the three types it touched, and three preference answers moved
without the pattern applying to them. With Luna as reader, enumerating before answering does
not reduce its aggregation errors; a stronger reader or a two-call read (enumerate, then
answer from the enumeration only) is the untested remainder.

Cost of the composed run over 500 questions: 10,089 extraction calls on an L4 (the dev half
replayed from the cache), about $1 of GPU time, plus the reader and judge; the time-range
extractor added one Luna call for each of the 133 temporal questions.

- **The reader itself moves about six answers between identical runs.** The routed run
  presented 148 questions with exactly the same formation and retrieved sessions as the raw
  baseline; 6 of them still flipped. Luna at temperature zero plus a GPT-4o judge is about
  4% noisy per question, so on 261 questions any two runs differ by roughly six answers
  before formation changes anything. That reframes the table above: raw 214, E2B hybrid 219,
  routed 210, reserved 215 and Qwen hybrid 211 are one cluster, and the only result that
  stands outside it is the first reserved attempt at 206 with its abstention collapse.
  Separating formations on this benchmark needs either repeated runs with a deterministic
  reader, or a slice where formation is expected to matter (multi-session questions alone,
  where extracted facts have gained 1–5 answers in every run).

- **What would move the number.** Raw retrieval is at 83% recall and the reader answers 96%
  of questions whose evidence it sees, so the loss is retrieval on temporal and multi-session
  questions. Two things follow from the reserved result: appended facts need a relevance
  threshold well above lexical score 1 and a recency or supersession rule before the reader
  sees them, and the abstention cases need the reader told that the fact block is
  supplementary. Neither is a model change.

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
