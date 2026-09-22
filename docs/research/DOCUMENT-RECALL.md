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
