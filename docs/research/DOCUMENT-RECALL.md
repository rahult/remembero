# Reading a 1000-page document: extraction and recall at page scale

Status: first results · 2026-09-23
Harness: `npm run eval:doc-recall`, `npm run eval:doc-extract` (see `benchmarks/document-recall/README.md`)
Evidence: `results/docrecall-*.json`, `results/docextract-*.json`

## Why

Everything measured so far reads chat history. The question this answers is different: given a
document nobody has chunked, indexed or summarised for us — a 100, 500 or 1000-page public
report — can the platform find the page that answers a question, and is what it stores true?

## The corpus

**XL-DocBench** (arXiv:2608.00036) supplies the labels: gold answers, a deterministic
`verification_rule`, the **evidence page numbers**, a verbatim quote, and questions the document
cannot answer. Its PDFs are referenced by URL, not shipped, so `select_and_fetch.py` downloads
them, verifies each one's page count against the label with `pdfinfo`, records its sha256, and
skips hosts that serve unauthorised copies of copyrighted books.

A tier is a page band, and every document in it is a real document of that length:

| tier | documents | pages | questions (answerable + not) |
| --- | --- | --- | --- |
| 100p | 5 (95–123 pp) | 544 | 42 + 3 |
| 500p | 5 (448–597 pp) | 2,537 | 43 + 4 |
| 1000p | 5 (704–1,056 pp) | 4,404 | 26 + 4 |

The tiers are **not difficulty-matched** — they are different documents in different domains and
three languages. Comparing accuracy across tiers says little; comparing each tier to *its own*
ceiling says a lot, which is why the oracle arm exists.

## How it is scored

Accuracy is XL-DocBench's own rule, ported to TypeScript and checked against their `evaluate.py`:
over 123 questions the two agree on **every question** and on all three aggregates. So these
numbers sit beside their published baselines (SimpleDoc+GPT-5.4 44.0%, Claude Opus 4.6 40.5%,
GPT-5.4 39.5% — all whole-document pipelines, not retrieval).

Retrieval is scored in pages, which the benchmark itself does not do: **page hit** (a gold page
reached the reader), **page recall** (how many of them did), **window precision** (how much of
what was read was worth reading).

Three readings of correctness are reported, because one of them is unfair in a way worth
quantifying:

- **rule** — theirs, verbatim. Requires the gold string to be contained in the answer or within
  0.8 edit similarity. A correct answer phrased as a sentence often scores zero.
- **locale** — the same rule with a decimal comma read as a decimal point. Their
  `normalize_answer` deletes commas as thousands separators, so a Dutch "63,1%" becomes "631%"
  and a correct answer scores zero. Several of these documents are not in English.
- **judge** — DeepSeek asked whether the answer is right. The honest read of usefulness.

## Result 1: a 4.9 GB local model reads a thousand pages as well as a cloud model

Same retrieval, same depth (4), same questions; only the reader differs. The local arm is
reader v7 (Gemma 3n E4B, fine-tuned here) requantised to Q4_K_M, served by `llama-server` on one
Mac, nothing leaving the machine.

| tier | local v7 Q4 rule / judge | deepseek-chat rule / judge | local latency |
| --- | --- | --- | --- |
| 100p | **6.7% / 11.1%** | 2.2% / 13.3% | 22.5s |
| 500p | **14.9% / 31.9%** | 10.6% / 21.3% | 19.4s |
| 1000p | **10.3% / 27.6%** | 6.9% / 31.0% | 21.1s |

On the benchmark's own rule the local 4B model wins every tier; on the judge the two trade places
and the gaps are inside the noise of 30–47 questions. The interesting part is what it costs: about
21 seconds and zero dollars per answer, against 1.6 seconds and a per-token bill, over documents
that never leave the laptop. For a memory system that reads a customer's documents, that is the
trade that matters.

## Result 2: retrieval is the ceiling, and the ceiling is high

Reader: `deepseek-chat` throughout, so only retrieval changes between rows.

| arm | 100p rule / judge | 500p rule / judge | 1000p rule / judge |
| --- | --- | --- | --- |
| depth 4 (product default) | 2.2% / 13.3% | 10.6% / 21.3% | 6.9% / 31.0% |
| depth 12 | 2.3% / 31.8% | 14.9% / 36.2% | 7.4% / 29.6% |
| **oracle pages** (gold pages handed over) | **13.3% / 40.0%** | **21.3% / 46.8%** | **36.7% / 60.0%** |

The local reader shows the same shape, and its ceiling is no lower:

| arm (local v7 Q4) | 100p rule / judge | 500p rule / judge | 1000p rule / judge |
| --- | --- | --- | --- |
| depth 4 | 6.7% / 11.1% | 14.9% / 31.9% | 10.3% / 27.6% |
| **oracle pages** | **6.7% / 33.3%** | **27.7% / 55.3%** | **26.7% / 33.3%** |

On the 500-page tier the local model's ceiling (27.7% / 55.3%) is *above* the cloud model's
(21.3% / 46.8%), and it answers an oracle question in 9–15 seconds rather than 20 — a short
prompt of exactly the right pages is the case a small fine-tuned reader is good at.

Page-level retrieval behind those rows:

| arm | 100p hit / recall | 500p hit / recall | 1000p hit / recall |
| --- | --- | --- | --- |
| depth 4 | 66.7% / 42.1% | 69.8% / 52.1% | 44.0% / 29.3% |
| depth 12 | 90.2% / 63.5% | 83.7% / 65.2% | 60.9% / 40.6% |

Reading the two tables together:

- **Depth is the cheapest win available.** Going from 4 pages to 12 lifts the page hit rate by
  23, 14 and 17 points, and on the 1000-page tier lifts rule accuracy from 6.9% to 7.4% and the
  100-page tier's judge accuracy from 13.3% to 31.8%.
- **A thousand pages is where retrieval breaks.** At depth 12 the 100 and 500-page tiers find a
  gold page 84–90% of the time; the 1000-page tier manages 60.9%. Window precision falls to 7.7%
  — twelve pages read, one of them useful.
- **The reader is not the problem.** Handed the gold pages, the same model answers 60% of the
  1000-page tier's questions (judge). Retrieval at depth 12 delivers 29.6% of them — it recovers
  **half** the available ceiling at 1000 pages, against ~80% at 100 and 500 pages.
- **The strict rule undercounts by 2–4×.** 7.4% versus 29.6% on the same answers. Every published
  baseline pays the same tax, so the rule stays the headline, but the judge column is what the
  system is actually worth to a user.

### The depth lever runs out at about twenty pages

Depth 24 on the 1000-page tier reaches a **69.2%** page hit rate (from 60.9% at depth 12) and
47.4% page recall — but **14 of its 30 questions never ran**: `reading prompt exceeds 65536
bytes`. `MAX_INPUT_BYTES` in `src/safety.ts` bounds anything sent to an external LLM at 64 KB, so
at roughly 3 KB per page the product can show a reader about twenty pages and no more. Of the
questions that did run, judge accuracy was 26.7% — no better than depth 12. Brute force is
already close to exhausted; what the 1000-page tier needs is better ranking, not more pages.

(One further prompt was refused outright — "refusing to send sensitive reading prompt to the
external LLM" — because a page of a public government manual tripped the secret-shaped-span
check. Worth knowing before pointing this at a customer's documents.)

## Result 3: extraction is accurate and beside the point

The product's writer over sampled pages (every labelled evidence page, plus an even spread),
`deepseek-chat`, facts judged against the page they came from:

| tier | document | pages sampled | facts/page | precision | question coverage |
| --- | --- | --- | --- | --- | --- |
| 100p | doc_000156 | 23 | 0.9 | 81% | 0 of 10 |
| 500p | doc_000175 | 20 | 6.5 | 95% | 0 of 10 |
| 1000p | doc_000120 | 12 | 2.2 | 100% | 0 of 6 |

**What it stores is true** — 81–100% of sampled facts are supported by their page. **What it
stores is not what was asked**, and for two separate reasons:

- On the 1000-page document, **no question had a single fact from its evidence pages**: those
  pages are tables, abbreviation lists and structural front matter, and the writer correctly
  declines to turn a table of contents into facts.
- On the 500-page document, 8 of 10 questions *did* have facts from their evidence pages — 24 to
  33 of them — and still none contained the answer. The questions are comparisons ("which
  standard appears in X but not Y"), and extraction is not question-aware.

**The local writer extracts nothing at all.** `qwen2.5-coder:7b` — the local writer that scores
81.6% on the repo's own extraction bench — returns the NOTHING sentinel (`% nothing`) on every
sampled page of all three documents, including pages where `deepseek-chat` extracts 7 to 25
supported facts. Verified by calling it directly with `extractionSystemPrompt`: it declines, it is
not being rejected by the write-side guards. So on this evidence the local stack can *read* a
thousand-page document (Result 1) but cannot *remember* one.

| writer | facts/page (100p / 500p / 1000p) | precision |
| --- | --- | --- |
| deepseek-chat | 0.9 / 6.5 / 2.2 | 81% / 95% / 100% |
| qwen2.5-coder:7b (local) | 0 / 0 / 0 | n/a |

A separate negative result worth keeping: pointed at document prose, the **autocapture writer
extracts nothing at all** — zero facts on every page of all three documents, no errors. That is
correct behaviour, not a bug: `transcriptExtractionSystemPrompt` asks for "durable
personal-memory facts ... explicitly stated or confirmed by the USER". Documents need the general
writer (`rememberText`), which is what the table above uses. The product has no document ingest
path that reaches its own autocapture.

## Caveats

- **Sample sizes are small**: 45, 47 and 30 questions per tier. Two identical oracle runs differed
  by 6.6 points on the 100-page tier, so treat anything under ~10 points as noise.
- **XL-DocBench is a 2026 preprint** with licence `other` on an anonymous HF namespace. Fine for
  internal measurement; mirror the labels and get clarity before publishing scores.
- **Contamination**: these are famous public documents, so a model may know parts of them without
  reading. This flatters every no-retrieval baseline and understates what retrieval contributes.
- **The local reader runs at Q4_K_M**, requantised from the Q8 GGUF, because the 8 GB Q8 model
  loads on a 16 GB machine but fails at compute against the Metal wired-memory limit.

## Result 4: GPT-5.6 Sol, and the cost and speed lens (2026-09-23)

`openai/gpt-5.6-sol` via OpenRouter ($2 / $10 per M tokens, measured cost from the endpoint),
same retrieval, same questions, judge `deepseek-chat`. Costs for the other arms are their saved
token counts priced at DeepSeek's off-peak Flash rate ($0.15 / $0.60 per M, the runs were
off-peak) and $0 marginal for the local model. `benchmarks/document-recall/summarise.mjs` prints
the full table.

| reader, arm | judge 100p / 500p / 1000p | $ per answer | $ per right answer | p50 / p95 latency |
| --- | --- | --- | --- | --- |
| local v7 Q4, depth 4 | 11.1 / 31.9 / 27.6% | $0 | $0 | 16–23s / 34–39s |
| deepseek, depth 12 | 31.8 / 36.2 / 29.6% | $0.0013 | $0.0035–0.0042 | 1.3–1.9s / 3s |
| **Sol, depth 12** | **43.2 / 59.6 / 51.9%** | $0.020–0.022 | $0.034–0.051 | 5.3–6.0s / 12–17s |
| Sol, oracle pages | 62.2 / 70.2 / 70.0% | $0.005–0.008 | $0.007–0.012 | 4.0–4.6s / 7–15s |
| deepseek, oracle pages | 40.0 / 46.8 / 60.0% | $0.0003–0.0004 | $0.0005–0.0011 | ~1s / 2s |

By the benchmark's strict rule Sol at depth 12 scores 20.5 / 29.8 / 22.2% and at oracle
24.4 / 34.0 / 46.7% — the 1000-page oracle figure is past the published whole-document
baselines (39.5–44.0%). Total spend for the three Sol arms: $4.69.

What the lens shows:

- **With a strong reader the reader matters again.** At depth 12 Sol beats deepseek by 11–23
  judge points on the same pages. Retrieval still costs Sol a quarter of its ceiling at 100 and
  1000 pages (43 of 62, 52 of 70) and a seventh at 500 (60 of 70).
- **Precision is a cost lever, not only an accuracy one.** Sol's oracle prompt is 6–15 KB
  against 37–44 KB at depth 12, so the right pages are both more accurate *and* three to four
  times cheaper per answer. Every point of window precision is money.
- **Ranking time grows with the document** (0.1s, 0.4s, 0.8s per question at 100, 500, 1000
  pages) because the harness re-indexes per question. The product should index a document once.
- **Sol never declines.** It answered every unanswerable question on the 100 and 500-page tiers.
  Only 11 such questions exist, so the rate is imprecise, but the direction is consistent.

### Routing, replayed offline (`benchmarks/document-recall/cascade.mjs`)

A cascade answers with a cheap reader and escalates to Sol only when the cheap one declines.
Replayed from the saved per-question records, no new calls:

| cascade | judge 100p / 500p / 1000p | $ per question | vs Sol alone |
| --- | --- | --- | --- |
| deepseek d12 → Sol d12 | 47.7 / 59.6 / 40.7% | $0.011–0.015 | equal or better on 100p/500p at ~55–70% of the cost; worse on 1000p |
| local d4 → Sol d12 | 13.6 / 36.2 / 44.4% | $0.002–0.009 | cheap, but the local model rarely declines — it answers wrongly instead |

The cascade only works when the cheap reader knows when it doesn't know. The local reader
escalates 9–41% of questions while getting 68–89% wrong, so its miscalibration, not its
accuracy, is what blocks the zero-cost path.

## Result 5: a better page index — large retrieval gains, small answer gains (2026-09-23)

Two changes from Result 4's list. **Index once** (`src/evals/document-index.ts`): a per-document
BM25 index and a cached local-embedding index (nomic-embed-text via Ollama), replacing the
product's per-question `searchKnowledge` pass. **Rank better** (`src/evals/document-rankers.ts`):
BM25, dense, hybrid (reciprocal-rank fusion of the two), hybrid + TypeSafe re-rank, and
"decomposed" (an LLM splits the question into one query per passage it needs; each part's best
page is guaranteed a slot).

### Retrieval alone (`npm run eval:doc-retrieval`, no reader, 112 answerable questions)

| ranker | 100p recall@12 | 500p recall@12 | 1000p hit@4 | 1000p hit@12 / recall@12 | query time | build (1000p tier) |
| --- | --- | --- | --- | --- | --- | --- |
| product (before) | 62.9% | 65.2% | 34.6% | 61.5 / 41.7% | 96–401 ms | none |
| bm25 | 64.0% | 72.5% | 61.5% | 73.1 / 54.8% | 1 ms | 1 s |
| hybrid | 66.8% | 80.2% | 65.4% | 73.1 / 60.9% | ~50 ms | ~5 min, cached |
| hybrid + TypeSafe | **80.1%** | **82.9%** | 53.8% | 61.5 / 54.2% | ~1.6 s | as hybrid |
| decomposed | 65.3% | 72.5% | 61.5% | **80.8 / 66.0%** | ~0.85 s | as hybrid |

The product's flat word score is the problem at scale: IDF alone (bm25) nearly doubles 1000-page
hit@4 and makes a query 400× faster. TypeSafe re-ranking helps short documents and hurts the
1000-page tier; decomposition is the reverse. Hybrid is the robust default. TypeSafe cost for the
whole sweep: $0.22.

### What it does to answers (paired, pooled over all tiers, judge)

| change | judge | rule | per-question won / lost |
| --- | --- | --- | --- |
| Sol d12: product → hybrid | 50.4 → 52.2% | 23.5 → 26.1% | 13 / 11 |
| Sol d4: product → hybrid | 40.0 → 40.8% | 12.5 → 15.8% | 11 / 10 |
| deepseek d12: product → hybrid | 33.9 → 30.4% | 8.7 → 13.0% | 11 / 15 |
| deepseek d12: product → decomposed, **1000p only** | 29.6 → 40.7% | 7.4 → 22.2% | 4 / 1 |

Honest reading: **the retrieval gains are real, the answer gains are mostly inside the noise.**
The strict rule rises in every pairing (the right page makes the exact string likelier), and the
1000-page tier, where retrieval was worst, is where answers move. Everywhere else the product
already put a gold page in front of the reader at depth 12 (74% pooled hit), so the remaining
misses are not about finding a page: they are multi-page questions only half-retrieved (page
recall 60–80%) and a reader that misses even with every page in hand (Sol's oracle ceiling is
62–70%).

### The cost lever this does unlock

Sharper ranking lets a reader read fewer pages. On the 1000-page tier Sol at depth 4 with hybrid
matches Sol at depth 12 with the product ranking on judge (48.3% vs 51.9%, within noise) at
**$0.0088 instead of $0.022 per answer** and 3.9 s instead of 6.0 s median. It does not hold on
the shorter tiers (depth 4 loses 8–17 points there), so the rule is "depth by document size",
not "always shallow".

### What would move the answer numbers next

1. **More questions before the next decision.** 27–47 per tier cannot resolve the 2–5 point
   effects left. XL-DocBench has 455 / 270 / 85 questions in these bands; the retrieval-only
   sweep can run on all of them for the cost of downloading PDFs and local embeddings.
2. **Route the ranker by document size**: hybrid + TypeSafe under ~600 pages, decomposed above.
   Both halves are measured above; the switch is untested end to end.
3. **Multi-page questions**: 73% need more than one page and page recall is the metric still
   lagging hit rate. Decomposition is the one change that raised both at 1000 pages.
4. Items 1 and 4–5 of Result 4 (abstention training, the 64 KB cap, "cannot be determined").

OpenRouter balance after these runs: $1.98 — enough for one more small Sol arm, not a full one.

## Result 6: facts as pointers help; facts as the answer source do not (2026-09-23)

The question: if we read every page at ingest and store facts, shouldn't retrieving facts be
*more* relevant than retrieving pages? Tested directly.

**Extraction over the whole corpus** (`npm run eval:doc-facts`): all 7,430 pages through
`deepseek-chat`, ~270,000 facts, 0 failures, 15 minutes, **$4.49** ($0.0006 a page). Every fact
keeps its page.

- The **product's extraction prompt** is written for personal memory ("the speaker", "durable
  facts") and answers `% nothing` on **78% of pages** — all 100 pages of an FDA regulation. A
  document prompt with the same Datalog contract (`DOCUMENT_EXTRACTION_PROMPT`: entities, numbers
  with units, list items and table rows, `in_section`, `references`) yields 13–46 facts a page
  with a handful of empty pages. Only the document prompt's facts are used below.

**Fact pointers as a ranker** (search the facts, return the pages they came from; retrieval only):

| ranker | 100p hit / recall @12 | 500p hit / recall @12 | 1000p hit / recall @12 | query |
| --- | --- | --- | --- | --- |
| product | 88.4 / 62.9% | 83.7 / 65.2% | 61.5 / 41.7% | 89–384 ms |
| hybrid | 86.0 / 66.8% | 88.4 / 80.2% | 73.1 / 60.9% | ~50 ms |
| facts alone | 88.4 / 74.0% | 86.0 / 72.6% | 76.9 / 54.8% | 1–7 ms |
| **hybrid + facts** | **93.0 / 79.0%** | **90.7 / 79.8%** | **84.6 / 65.7%** | ~50 ms |

Hybrid + facts is best or tied-best on every tier — the only ranker so far that is, with no
per-query API cost (TypeSafe re-ranking matched it at 100p and lost at 1000p; decomposition the
reverse).

**Reading facts instead of pages** (same hybrid + facts retrieval, deepseek, depth 12, paired):
judge 31.4% reading pages → 27.3% reading only those pages' facts (7 won, 12 lost); the rule
drops 15.7 → 12.4%. Tied at 1000 pages. Extraction still loses what the question turns out to
need, and the reader can't get it back from a fact.

**End to end with Sol** (depth 4, paired against the product's ranking at depth 4):
judge 39.7 → 43.0% (18 won, 14 lost), rule 12.4 → 17.4%; on the **1000-page tier 37.9 → 51.7%**
(5 won, 1 lost), rule 13.8 → 27.6%.

And the cost lens: Sol reading **4** hybrid + facts pages matches Sol reading **12** product-
ranked pages on the 1000-page tier (53.3% vs 51.9% judge) and the 100-page tier (42.2% vs 43.2%)
at **$0.009 instead of $0.022 an answer** and 4.1 s instead of 6.0 s median. It does not hold at
500 pages (38.3% vs 59.6%), where depth still matters. At $0.013 saved per answer, the $4.49 of
extraction pays for itself after about 350 questions.

**So the design is facts as an index into the document, not a replacement for it**: extract at
ingest with a document prompt, keep the page on every fact, search facts and pages together, and
hand the reader the pages.
