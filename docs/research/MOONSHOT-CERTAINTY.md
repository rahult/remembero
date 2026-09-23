# Moonshot: certain about what we know, honest about what we don't

Set 2026-09-23. Extends `MOONSHOT.md` (small trained models, no external model at runtime) with
the target that makes a memory product trustworthy in compliance work: **zero confident-wrong
answers**, every answer carrying its evidence, and answers that combine facts computed by the
engine rather than guessed by a model. Coverage — how many questions get a verified answer —
then climbs toward 100%.

"100% accuracy" on a public benchmark is not a target anyone can hit: labels are wrong or
ambiguous (4–8% of the misses measured below), and strict scorers fail correct paraphrases.
"Never confidently wrong, and right on everything the store can prove" is a target we can
hit and can prove we hit.

## Where we stand (measured, `docs/research/DOCUMENT-RECALL.md` Results 1–7)

| surface | best today | ceiling with perfect evidence | the gap is |
| --- | --- | --- | --- |
| chat memory, LongMemEval 500 | 451 (GLM reader), 428 (our reader) | ~90% | reading: counting, dates, latest value |
| documents, XL-DocBench, 100–1000 pp | Sol 52% judge (depth 12) | Sol 62–70% | ~21 pts retrieval, ~26 pts reading |
| page retrieval, 1000 pp, hit@12 | 84.6% (hybrid + fact pointers) | – | multi-page questions only partly found |

The misses, by cause (Sol, documents): partly found 36%, not found 29%, found but buried 13%;
with perfect pages, **declining although the evidence is there is 43% of what remains**, then
misread values, wrong passage, computation. With perfect pages, set-difference questions are
right 45% of the time, arithmetic 55%, comparison 60%, against 77% for everything else.

Two conclusions drive everything below:
1. The questions a Datalog engine answers exactly are the ones frontier readers fail most even
   when handed the right pages.
2. Nothing today measures that at scale — XL-DocBench has 10–20 questions of each such shape and
   its labels are hand-written. We need our own benchmark, with labels that are certain.

## The certainty contract

Every answer is one of three kinds, and says which:

- **Proved** — computed by the engine from stored facts; returns the Datalog proof and the page
  of every fact it used. Wrong only if extraction was wrong, and extraction is checked (below).
- **Read** — a model answered from pages; returns verbatim quotes, each checked by code to exist
  on the cited page, and a verifier's judgement that the answer follows from them.
- **Unknown** — the evidence does not settle it; says what was searched and what was missing.

A compliance answer is only ever Proved or Unknown. Remembro-v0's ALLOW / DENY / UNKNOWN
decision path (branch `remembro-v0`) is already this contract for authority questions.

## Architecture

1. **Ingest once.** Pages with structure kept (headings, tables as rows, lists). Facts extracted
   with the document prompt, each with its page; a section map; bm25, dense and fact-pointer
   indexes built once. *(Built in the benchmark; not yet in the product's ingest.)*
2. **Plan.** Classify the question's shape, split it into the pieces it needs, choose depth by
   document size, and — for engine shapes — draft the Datalog query it will need.
3. **Retrieve until complete.** Hybrid + fact pointers; then check each planned piece has
   evidence and search again for the ones that don't. Aimed at "partly found", the largest cause.
4. **Targeted extraction.** For engine shapes, extract facts again from just the retrieved pages,
   conditioned on the plan's predicates, and check each fact against its page.
5. **Compute or read.** Engine shapes run as Datalog over the checked facts → Proved. Everything
   else is read with quotes → Read. A reader may not decline until step 3 has searched for the
   missing piece (declining with the evidence present is the largest reading cause).
6. **Verify, then answer or say Unknown.** Quote check in code, entailment check by a verifier.

Modes: **Fast** (local, 4 pages, $0), **Balanced** (hybrid + facts, cheap reader), **Exact**
(all six steps, strong reader), **Audit** (Exact plus proofs and quotes). Route cheap-first and
escalate on a calibrated decline — replayed offline, this matches the strong reader at 55–70%
of its cost when the cheap reader knows when to decline.

## Our own benchmark: certain labels, composed questions, real haystacks

Working name **Compose**. Questions that need 2–6 facts spread across hundreds of pages —
identity, dates, validity, comparison, aggregation, absence, authority — with gold answers
**computed by the engine from the facts the documents were generated from**, so every label is
certain by construction.

**World.** A seeded generator builds an organisation as a Datalog fact base: people with aliases
and titles that change hands over time, units, suppliers, contracts with values and terms,
policies and amendments that supersede clauses, delegations of authority with limits and dates,
incidents, certifications with expiry dates, training records. Rules define the derived truths:
who held a role on a date, the effective clause after amendments, whether an approval was within
authority, which certificates had lapsed.

**Documents.** Each fact is written into a realistic section — board minutes, a policy and its
amendments, a delegation schedule, a contract register table, audit findings, an email appendix —
by a model paraphrasing a template, then checked: every name, number and date of the fact must
appear in the sentence, and an independent extraction must recover the fact. A sentence that
fails is rewritten or dropped. Hedged and proposed statements ("it was proposed that…") are
written deliberately and are *not* facts, which tests that nothing counts them.

**Haystacks.** The rendered sections are placed at chosen depths inside real public-document
filler (we hold 7,430 pages already) to reach 100, 500 and 1000 pages, with a collision check
that no filler page mentions a generated name. Distractors are planted on purpose: near-duplicate
names, superseded values, expired delegations, the same number in another unit.

**Question families**, each written from a Datalog query whose engine result is the gold:

| family | example | what makes it hard for a model |
| --- | --- | --- |
| needle | the notice period in clause 14.3 | one sentence on page 734 |
| identity | what "the CRO" approved in March, when the minutes name her only as "A. Osei" | aliases joined across documents |
| validity | who held signing authority on 14 March 2025 | a date inside ranges set and revoked on different pages |
| supersession | the current late-payment penalty after two amendments | the latest of several stated values |
| comparison | which supplier's total contract value is higher | values spread over a register and amendments |
| aggregate | how many incidents at sites with an expired certificate | a join and a count across pages |
| absence | which suppliers have no current certificate | proving something is not there |
| authority | may P approve a $250k contract with S on D? | chain of delegation, limits and dates → ALLOW / DENY / UNKNOWN |
| unanswerable | a question the facts cannot settle | the right answer is Unknown |

Every family is generated at 1, 2, 4 and 6 hops and at every page depth, so accuracy can be
plotted against both — the curve should show frontier readers falling as either grows while the
engine stays flat.

**Scoring.** Exact match against the engine: sets as sets, numbers exactly, dates as dates,
decisions as ALLOW / DENY / UNKNOWN. Three reported numbers: correct, **confident-wrong** (the
certainty metric), and Unknown. Plus page citations checked against where the facts were placed.

**Splits.** Worlds from disjoint seeds and disjoint name pools for train, dev and test. The train
worlds double as supervision for the planner and writer (question → Datalog query, page → facts)
with no leakage into test.

**Honesty checks.** A no-document baseline (to show the answers cannot be guessed), a
whole-document long-context baseline (to show the degradation), and a small human-audited sample
of rendered sections and questions.

## Roadmap

Each phase has an exit test. Order is by what the measurements say moves the numbers.

**Phase 1 — measure what matters (1–2 weeks).**
- Build Compose v0: generator, renderer with checks, haystack assembly, 9 families, 100/500/1000
  pages, ~600 test questions plus train/dev worlds.
- Run baselines: Sol, DeepSeek and the local reader through today's retrieval; Sol reading the
  whole document; the engine given the generating facts (upper bound = 100% by construction).
- Grow the XL-DocBench run from 123 to all 1,354 questions (retrieval-only on all; answers on a
  dev/test split) so a 3-point change is visible.
- *Exit:* published curves of accuracy against hops and page depth for each reader; the gap the
  engine has to close, in numbers.

**Phase 2 — retrieval to "complete" (2 weeks).**
- Move index-once, hybrid + fact pointers and the document extraction prompt into the product's
  ingest; give documents their own ingest path (autocapture extracts nothing from them today).
- Plan-driven retrieval: search again for every planned piece without evidence.
- Depth by document size; lift the 64 KB prompt cap for document reading.
- *Exit:* XL-DocBench 1000-page page recall@12 from 65.7% to ≥ 85%; "partly found" misses halved.

**Phase 3 — reading that doesn't guess or give up (2 weeks).**
- Quote-first answers with code-checked quotes; verifier; decline only after a targeted re-search.
- Prompt the "cannot be determined" wording the benchmark scorer accepts.
- *Exit:* Sol with gold pages ≥ 80% (from 67%); confident-wrong ≤ 2% on Compose and XL-DocBench.

**Phase 4 — the engine answers what it can prove (3–4 weeks).**
- Planner writes the Datalog query for engine shapes; targeted, checked extraction from the
  retrieved pages; engine answers with a proof; identity (alias) and validity-time rules as
  built-in closure predicates; Remembro-v0's decision path for authority questions.
- *Exit:* Compose engine families ≥ 98% correct with zero confident-wrong at 1000 pages and
  6 hops; XL-DocBench set-difference and arithmetic from 45% / 55% to ≥ 80%.

**Phase 5 — our own small models (ongoing, per `MOONSHOT.md`).**
- Reader v9: quotes and calibrated declining, trained on Sol traces from Compose train worlds and
  XL-DocBench train questions, plus our saved confident-wrong answers relabelled as declines.
- Planner (~2B): question → shape, pieces and Datalog query, supervised by Compose train worlds.
- Local document writer distilled from the 270,000 DeepSeek document facts we already hold.
- *Exit:* local-only Exact mode within 5 points of the Sol Exact mode on Compose test.

**Phase 6 — modes as product.** Fast / Balanced / Exact / Audit, cheap-first routing on
calibrated declines, per-answer cost and proof shown to the user.

## Cost of the plan's measurement

Phase 1 runs cheaply: generation and rendering ≈ $5–10 of DeepSeek; baselines ≈ $10–20 of Sol
(more for whole-document runs, which are ~700k tokens and ~$1.40 a question at 1000 pages, so
those are sampled); local runs are free. OpenRouter needs a top-up for anything beyond ~$1.

## Open questions for debate

- **Generated worlds vs real documents.** Generated labels are certain but the prose is ours;
  real documents are realistic but labels are hand-made. Proposal: Compose for certainty and
  training, XL-DocBench as the realism check; never train on either's test split.
- **Extract everything at ingest, or on demand?** Ingest-time facts are pointers (measured: they
  help find pages); question-time targeted extraction is what the engine computes on. Both.
- **How much the engine should own.** Only shapes whose query the planner writes with high
  confidence; otherwise read. Measured per family in Phase 4, not assumed.
- **When to train.** After the cause breakdown shows a stable largest cause (v8's lesson: training
  on guesses made it worse).
