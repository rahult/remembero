# Roadmap: run the memory system on our own models only

Goal: every model call the product makes at runtime goes to a model we trained or host, so the
marginal cost per memory operation is GPU time we control and no user text leaves our
infrastructure. Evaluation may keep using frontier models where a protocol fixes them (the
LongMemEval judge) or where they only produce training data offline.

This document lists each remaining external dependency, what the research docs already
measured about replacing it, the experiment that decides it, and the order to do them in.
Companion documents: [MODEL-COMPARISON.md](MODEL-COMPARISON.md) (what each model scored),
[RUN-MATRIX.md](RUN-MATRIX.md) (every run), [LONGMEMEVAL-PATTERNS.md](LONGMEMEVAL-PATTERNS.md)
(the literature).

## Where the product stands today

All product LLM traffic goes through one OpenAI-compatible client configured by `LLM_BASE_URL`,
`LLM_MODEL` and `LLM_API_KEY` (`src/llm/client.ts`), so every leg below can be repointed by
environment alone once a replacement exists.

| runtime leg                        | code                                              | model today                                     | ours? | measured replacement                                                              |
| ---------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| extraction (remember, transcript)  | `src/llm/pipeline.ts` remember paths              | Gemma 4 E2B r16 on Modal L4                     | yes   | 85–86/103 vs Luna 96; LongMemEval hybrid ties or beats Luna as extractor          |
| query authoring (recall → Datalog) | `src/llm/pipeline.ts` recall path                 | same adapter                                    | yes   | 27–28/31 vs Luna 29, GLM 30                                                       |
| answer phrasing (the "answer leg") | `PHRASING_SYSTEM_PROMPT`, pipeline                | whatever `LLM_MODEL` is (default Claude Sonnet) | no    | deterministic and evidence modes exist (`REMBERO_RECALL_ANSWER_MODE`), zero calls |
| embeddings (semantic route)        | `src/llm/embeddings.ts`, semantic-search          | `perplexity/pplx-embed-v1-0.6b` via OpenRouter  | no    | none local measured; `REMBERO_EMBEDDING_BASE_URL` can point anywhere              |
| reader over raw sessions           | not in the product; a LongMemEval-harness concept | GLM 5.3 Flash (Ollama Cloud) in evals           | no    | no sub-8B reader measured; the 3B answer leg was shown unreliable                 |
| time-range extraction              | not in the product; eval flag only                | Luna in evals                                   | no    | Luna refuses 46/66 and helps; the paper found a weak extractor hurts              |

Offline only: judge (GPT-4o, fixed by protocol), labeller (Luna, ~$6 per 3,400 sessions),
paraphraser for synthetic training data (Luna), evaluation readers.

The writer, which is the heart of the thesis, is already ours. What remains external at
runtime is the answer leg, the embeddings, and, if the product grows a raw-session reader the
way the evaluation has, that reader.

## The roadmap, in order

### 1. Make the answer leg local by default (days, no training)

**Done 2026-09-11.** `REMBERO_RECALL_ANSWER_MODE` defaults to `evidence`; `natural` is opt-in.

The product already has two zero-call answer modes, deterministic and evidence, and the
agent-boundary work showed the small-model prose leg is noise below 8B (the query was right,
the sentence about it was wrong). Switch the default to the evidence mode, which renders the
query's rows as the answer, and offer the LLM phrasing only as an opt-in. Measure with the
existing agent-boundary end-to-end column: the end-to-end score should rise to meet
query-correct, since the leg that lost answers is gone. Cost after: zero. Data after: none
leaves.

### 2. Local embeddings for the semantic route (one week)

The semantic route is worth about two answers on 500 and fires on a fifth of questions; it is
the only runtime call that sends user text to a third party besides the answer leg. Candidates
with open weights that a vLLM or Ollama process can serve next to the writer:
`Qwen3-Embedding-0.6B`, `bge-m3`, `nomic-embed-text`. Re-run the selection matrix in
[SEMANTIC-KNOWLEDGE-SEARCH.md](SEMANTIC-KNOWLEDGE-SEARCH.md) (development R@5, MRR, p95) with
each served locally through `REMBERO_EMBEDDING_BASE_URL`, then the full-500 LongMemEval run
from the extraction cache (40 minutes). Accept the first model within two points of
`pplx-embed-v1-0.6b` on R@5. Cost after: shared GPU time on the existing L4. This also
removes the 429 failures the hosted embedding provider produced.

### 3. A reader we own (two to four weeks; the only hard item)

**Proxy runs done 2026-09-11.** Gemma 4 31B 347/500 (over-abstains on date arithmetic),
Nemotron 3 Super 389/500 (over-answers, loses knowledge updates), against GLM 5.3 Flash 432.
Neither is a candidate under the shared prompt; the path is distillation or a per-model reader
prompt for Gemma. Details in [LONGMEMEVAL.md](LONGMEMEVAL.md).

Three findings frame this. The reader answers 96% of questions whose evidence is retrieved
at k ≤ 5 and 89–92% at k = 15; a 3B model's prose leg was unreliable; and swapping Luna for
GLM 5.3 Flash on aggregation questions was worth 17 answers on 500. So the reader matters and
small models have not been shown to do it. Two paths, cheapest first:

- **Proxy first.** Run the open-weight models on the Ollama subscription as readers before
  hosting anything: `gemma4:31b`, `qwen3.5:397b`, `nemotron-3-super`. A cached full-500 run
  is 45 minutes each and free on the plan. Whatever open model scores within the reader noise
  (about ten answers on 500) of GLM 5.3 Flash's 432 is a self-hosting candidate. Gemma 4 26B
  A4B (3.8B active, MoE) is the one to hope for: it would serve on a single A100 at about
  $2.50 an hour and only runs at recall time.
- **Distil a small reader.** If no open model within reach of an A100 clears the bar, train
  one: have GLM 5.3 Flash (on the plan, so free) answer a few thousand synthetic questions
  over real haystack sessions, using the same six question types, and fine-tune Gemma 4 E4B as
  a reader on those. The extraction cache and the labelled real sessions already provide the
  session pool; the question generator is the missing piece. Expect this to take two rounds.

Decide with the same rule used throughout: a candidate must clear the 432 baseline by less than
the noise to be called a tie, and the total cost of ownership (GPU-hour at recall volume) must
beat the subscription. Note the subscription is already flat-rate; the case for self-hosting
here is proprietary data handling, not money.

### 4. A time-range extractor from our writer (one week, optional)

**Interim step done 2026-09-11.** GLM 5.3 Flash on the subscription ties Luna in this role
(113 vs 116 of 133 temporal, 33 vs 36 ranges), so the evaluation configuration no longer
needs OpenRouter for it. Training our writer for the task remains the self-hosted end state.

Only needed if the product adopts time-aware retrieval. The paper's warning is that a weak
extractor that guesses ranges hurts; Luna helps because it refuses 70% of the time. The writer
adapter can learn the task from synthetic data (question, current date → range or "none") if
the data is refusal-heavy, and it is scored on the 66 development temporal questions against
Luna's 20 ranges and their effect (52 → 56). If it cannot match the refusal precision, keep the
feature off rather than route it externally.

### 5. Training data without frontier models (ongoing)

The paraphraser and labeller are offline and cheap, but they are the remaining way user text
could reach a third party if real sessions are labelled. Two changes: run the labeller on the
subscription (GLM 5.3 Flash) with a cap on facts per session, which also fixes the verbosity
that hurt r17; and generate paraphrases with the subscription model. Neither needs a
measurement beyond the next training round's benchmark scores.

### 6. Keep the judge external

GPT-4o is the LongMemEval protocol's judge and the reason our numbers are comparable to
published ones. A local judge can run alongside as a second opinion, never as the reported
number.

## Engine work that removes model calls altogether

The literature survey in [FACTS-AND-MEMORY-RESEARCH.md](FACTS-AND-MEMORY-RESEARCH.md) points the
same way as the model roadmap: every capability moved into the Datalog engine is one that
needs no model at all, ours or anyone's. Four of its items bear directly on the legs above.

- **Bi-temporal, append-only facts** (Zep/Graphiti, ROMEM). Most of this exists: with
  `REMBERO_VALID_TIME_MODE=archive_until` a superseded fact is kept as an `_until` fact
  carrying the instant it stopped being current, the journal records ingestion time, and
  `history` reads both ([TEMPORAL-HISTORY.md](../TEMPORAL-HISTORY.md)). The gaps are that
  archiving is opt-in (`remembero init` turns it on; the library default deletes), that
  facts have no explicit `valid_from`, and that the LongMemEval formation renders facts
  without their intervals. Closing them means defaulting to archive, stamping facts with the
  source session's date, and rendering "X, from May 3 until June 9" in the fact block. That
  is the deterministic answer to the knowledge-update questions and what would let the
  temporal route (roadmap item 4) work from stored intervals instead of a model guessing a
  date range from the question. First item in the survey's own priority order.
- **Solver-feedback repair on the query leg** (Logic-LM, SymbolLKG). Done 2026-09-11 on the
  engine side: `emptyResultFeedback` gives the repair turn per-goal match counts and a
  concrete argument-swap suggestion, in the product fallback and on the SQLite bridge. The
  fine-tune ignores it (27/31 either way) because it never saw a repair turn. The data
  generator now emits repair-turn conversations (`--repair-share`; a mutated program that
  runs empty, the engine's feedback, the verified program, and the repeat-unchanged case for
  legitimately empty answers); r18 trains on r14's data plus 1,383 of them
  ([QUERY-DIALECT-FINETUNE.md](QUERY-DIALECT-FINETUNE.md), finding 7).
- **Salience and decay as derived scores** (MemoryBank, Generative Agents). Ranking facts for
  a session brief by recency and reinforcement, computed in the engine and never deleting,
  replaces the semantic route for the common "what matters now" case and lowers how often
  roadmap item 2's embedding model is consulted.
- **Incremental rule maintenance** (DBSP, DRed). Keeps derived facts current under write-gated
  updates without recomputing; a cost item for the million-fact benchmark, not a model item,
  but it is what keeps the zero-model path fast enough to be the default.

The survey's evaluation suggestions fit the same programme: BeliefShift measures whether a
memory drifts under model pressure, which a write-gated store should resist by construction,
and LongMemEval-V2's best published result is 74.9%, so a proof-carrying system with our own
writer has room to place. Both are eval-only and can use the frontier judge.

## What does not need doing

- Replacing the writer: it is ours, it ties the frontier model on the synthetic benchmarks
  within the measured noise, and it beat Luna as an extractor on LongMemEval at a hundredth of
  the cost.
- Chasing the reader-side prompt patterns: both were measured negative with Luna; the gain
  came from a better model, not a better prompt.
- Entity-keyed retrieval and fact-augmented keys: measured, did not help; the retrieval
  levers that worked (per-type k, time range, semantic route) are already in.

## Cost and data posture, today and after

| leg                 | today                             | after roadmap                            |
| ------------------- | --------------------------------- | ---------------------------------------- |
| writer              | ours, L4 ≈$0.80/h, scales to zero | unchanged                                |
| answer leg          | frontier per call                 | zero calls (evidence mode)               |
| embeddings          | hosted, ≈$0.0006/question, 429s   | local model on the same GPU              |
| reader (if adopted) | subscription, text leaves         | open model on our GPU or a distilled E4B |
| labelling           | Luna, text leaves                 | subscription model, capped               |
| judge               | GPT-4o, eval only                 | unchanged                                |

Order of attack: 1 and 2 are cheap and independent and remove every runtime third-party call
except the reader; 3 is the real project; 4 and 5 follow if their features are wanted. The
bi-temporal engine work runs alongside, since it shrinks what 3 and 4 have to do.
