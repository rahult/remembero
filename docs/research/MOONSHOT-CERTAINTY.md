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

## Phase 1 progress (2026-09-23)

**Miss causes on XL-DocBench** (`DOCUMENT-RECALL.md` Result 7): for Sol at depth 12, 25 of 56
misses are fixed by the gold pages alone and 31 are not; declining with the evidence present is
the largest reading cause; set-difference, arithmetic and comparison are the weakest shapes even
with perfect pages.

**Compose v0 built** (`benchmarks/compose/`): 2 test worlds, 93 questions, 100/500/1000 pages,
gold from the engine, deterministic scoring. First baselines (`deepseek-chat`, undated prompt):

| arm | right | confidently wrong | partial | declined |
| --- | --- | --- | --- | --- |
| depth 12, product ranking (100 / 500 / 1000 pp) | 67 / 66 / 67% | 13 / 14 / 12% | 10–15% | 5–11% |
| gold pages handed over | 82 / 82 / 82% | 5 / 4 / 3% | 4–9% | 5–9% |

By family, with the gold pages: lookups, validity, supersession, comparison and unanswerable
90–100%; **identity 10–30%, aggregate 50%, absence 71–86%, authority 81–88%**. With ordinary
retrieval the same four are 0%, 0–25%, 0–29% and 50–63%. The engine's target families are
exactly the ones a model fails even when it has every page.

Document length barely moves v0 (retrieval finds world pages 82–98% of the time because they
are lexically distinct from the filler) — the reasoning, not the haystack, is what is hard here.
v1 adds same-domain distractor organisations to make length bite.

Blocked: GPT-5.6 Sol runs stopped at the OpenRouter **key's** monthly limit (403 "Key limit
exceeded") although the account has credit; the key's limit must be raised in the OpenRouter
dashboard. The local reader run is in progress.

## First engine result: Proved-or-Unknown on Compose (2026-09-23)

`npm run compose:engine`: every page of each haystack through a schema-guided extraction
(`src/compose/extract.ts`, nine predicates, DeepSeek), each fact kept only if its dates, amounts
and names appear on its page, deterministic type checks (a contract reference must look like one
— swapped arguments are put back; a role title cannot be a person), then the Datalog engine
answers with Unknown whenever the facts do not settle the question (`src/compose/engine-answer.ts`).
The question → query step uses the question's parameters: an **oracle planner**, so this measures
extraction and reasoning, not question understanding.

| arm (1000 pages; 100 and 500 identical for the engine) | right | confidently wrong | Unknown / partial |
| --- | --- | --- | --- |
| local reader v7, depth 12 | 62% | 21% | 18% |
| deepseek-chat, depth 12 | 67% | 12% | 21% |
| deepseek-chat, gold pages handed over | 82% | 3% | 15% |
| **engine over extracted facts, test worlds** (checks tuned on these) | **100%** | **0%** | 0% |
| **engine over extracted facts, held-out dev worlds** | **97.8%** | **0%** | 2.2% |

- Extraction: recall 100% of the true facts on all four worlds; precision 97.5–100% (the extra
  facts are stray appointments, e.g. an "appointment" read off a resignation line). On the dev
  worlds those strays created a conflicting role holder and the engine answered one authority
  question per world Unknown — the contract working as designed: a conflict is never guessed.
- Cost: extracting 2,000 + 1,139 pages cost $0.41 + $0.23 ($0.0002 a page); filler pages yielded
  no schema facts at all, which is why document length does not move the engine. Answering is a
  query: milliseconds, no model.
- With perfectly extracted facts the rules and templates score 93/93 (a check of the engine
  alone, run before any extraction).

What this does **not** yet show:
- **Question understanding.** The planner is an oracle. The next measurement is an LLM (then our
  ~2B planner) turning the question text into the query or template + parameters.
- **Realistic prose.** Compose v0 pages come from templates; extraction on real documents is
  harder (XL-DocBench). v1: paraphrased rendering with fact-recovery checks.
- **Hostile filler.** Government-report filler never produces schema facts. v1: same-domain
  distractor organisations, so the extractor meets near-duplicate names and values.
- **Scale of the world.** ~125 facts per world; v1 grows the organisation with page count.

## Compose v1, a real planner, and what broke on the way (2026-09-24)

v1 makes the benchmark harder in the two ways v0 was soft: every question is **paraphrased** by an
LLM (kept only if every parameter survives — 369 of 369 did), and each haystack hides **two sister
organisations** with the same documents, role titles and formats but different people and
contracts. The planner is now `deepseek-chat` reading the question text (`src/compose/planner.ts`):
it picks one of nine query shapes and fills its parameters, and code rejects any parameter not
found in the question.

**Held-out worlds (103, 104), generated after the last change, 1000 pages:**

| arm | right | confidently wrong |
| --- | --- | --- |
| deepseek-chat reading 12 retrieved pages | 72% | 9% |
| deepseek-chat handed the gold pages | 77% | 9% |
| **engine: extraction + LLM planner + Datalog** | **100%** | **0%** |

Identical at 100 and 500 pages. Extraction recall 100%, precision 92–94% (the extras are
"appointments" read off the "Present: the Chair and directors" line, which no question uses).
Across all six worlds scored since the fixes, the engine path is 93.6–100% right with **zero**
confidently wrong answers; its misses are Unknowns.

Three lessons, each learned by the benchmark catching a real failure:

1. **Whose fact is it?** With sister organisations in the haystack the nine-predicate schema merged
   three companies: incident counts included other companies' incidents, authority checks used
   other companies' limits, and the engine was confidently wrong 4–13% of the time. Every fact now
   carries its organisation (grounded against the page, whose heading names it) and every rule
   joins within one.
2. **One extraction call is not evidence of absence.** The same page answered `% nothing` on one
   run and 32 certificate records on the previous one. A lost register does not produce Unknown —
   it produces a confidently wrong answer (a pre-amendment value returned as current). Record-like
   pages (three or more dates) now get a second independent pass, merged; recall went from 75–94%
   back to 100% on every world.
3. **Bound the rules, not only the engine.** Rules that paired every role term with every date in
   the corpus hit the engine's 5M-fact evaluation budget once three companies were loaded. Each
   rule now ranges only over the dates it can need (question, approval or incident dates); a world
   of three companies answers in about 4 seconds, from a single cached derivation per question.
   An evaluation the engine cannot finish is reported as Unknown with its reason, never guessed.

Cost of the whole engine path on this set: extraction ~$0.0002–0.0004 a page once (two passes on
record pages), planning ~$0.0001 a question, answering free.

## v2: reworded documents, and two more guards (2026-09-24)

**v2** rewrites every prose line of the minutes, amendments and site list with a model, keeping a
rewrite only if each fact the line states still passes the grounding check on that line alone
(559 of 563 kept); registers stay tables. On two fresh worlds (105, 106) at 1000 pages the engine
path — deepseek extraction, deepseek planner, Datalog — is **100% right, 0% confidently wrong,
100% extraction recall**.

Two guards added on the way, both general rather than benchmark-specific:

- **A fact must be stated by one line.** The extractor had been turning "Present: the Chair and
  directors" into appointments of whoever the minutes named next; the name, "Chair" and the
  meeting date were all on the page, so the page-level check passed them, and the phantom person
  made approvals unresolvable. Every argument except the organisation (which may come from the
  heading) must now appear on a single line. v0 extraction precision went to 100%; world 101 from
  93.6% to 100% right.
- **A gap in numbered amendments means Unknown.** If amendment 2 of a contract was extracted and
  amendment 1 was not, the record is provably incomplete, so value, comparison and authority
  answers about that contract are Unknown. A lost *last* amendment is invisible to this check;
  that is what the second extraction pass is for.

**State of the engine path, all runs since these guards (1000 pages, deepseek extraction and planner):**

| set | worlds | right | confidently wrong |
| --- | --- | --- | --- |
| v0 test / dev | 101, 102 / 201, 202 | 100, 100 / 97.8, 100% | 0% |
| v1 test / dev | 101, 102 / 201, 202 | 100, 97.8 / 95.6, 100% | 0% |
| v1 held out | 103, 104 | 100, 100% | 0% |
| v2 held out (reworded documents) | 105, 106 | 100, 100% | 0% |

Every miss is an Unknown. Readers on the same questions: 62–77% right and 9–21% confidently wrong.

**In progress:** Gemma 4 E2B LoRA on RunPod (H200, 319 steps, ~$4) trained on 20,386 rows from
250 train-split worlds — page → facts and question → plan — to run the whole engine path locally.

## The engine path on our own 2B model (2026-09-24)

`compose-engine-e2b`: Gemma 4 E2B with a LoRA trained on 20,386 rows from 250 train-split worlds
(page → schema facts, question → plan), 319 steps on an H200, 54 minutes, **$4.55**. One model does
both jobs; Datalog answers. No model API in the loop.

| held-out set, 1000 pages | world | right | confidently wrong | extraction recall | facts the page check dropped |
| --- | --- | --- | --- | --- | --- |
| v1 (paraphrased questions, sister organisations) | 103 | 100% | 0% | 100% | 63 |
| | 104 | 100% | 0% | 100% | 22 |
| v2 (documents reworded — never seen in training) | 105 | 97.7% | 0% | 99.3% | 68 |
| | 106 | 97.9% | 0% | 97.7% | 24 |

Identical at 100 and 500 pages. The 100-page runs were on the Mac (llama.cpp, Q8, ~10–20 s a page);
the 500 and 1000-page pages were extracted by the same weights under vLLM on a rented H100 (about
12 minutes, ~$1). Same questions through a reader: deepseek 72% right and 9% confidently wrong.

What the numbers say:
- **The small model invents facts on filler pages** — 22 to 68 per 1000-page haystack — and the
  page check (every argument on one line of the page) drops every one. The certainty is carried by
  the check, not by the model; that is what makes a 2B model usable at this scale.
- **It generalises past its templates.** Trained only on template-rendered pages, it reads the
  reworded v2 documents with 97.7–99.3% recall; the facts it misses become Unknowns, not wrong
  answers.
- **A malformed output is not an answer.** The model emitted an impossible date (month 13) on one
  page; that fact is now rejected as unsupported instead of crashing the run.

## v3: questions the documents cannot settle (2026-09-24)

v3 adds cases where the only correct answer is Unknown: two people in one organisation share a
surname and some approvals name the approver only as "Mr Rautio"; some contracts are tabled and
their approval deferred, so "was the approval within authority?" has no approval to judge. Built
on fresh worlds 107 and 108, on top of v2 (reworded documents, paraphrased questions, sister
organisations). 101 questions.

| 1000 pages | right | confidently wrong |
| --- | --- | --- |
| deepseek-chat reading 12 retrieved pages | 65% | 14% |
| deepseek-chat handed the gold pages | 80% | 7% |
| **engine (deepseek extraction and planner)** | **100%, 94%** | **0%** |

The reader declined every deferral correctly and answered the namesake questions with the bare
surname ("Mr Galloway") — scored partial, not wrong, since it names no one in full. The engine's
Unknowns in world 108 trace to one missed resignation: without it two people could hold the same
office, so the engine declines rather than choose.

Three refinements made on the way, each checked against every earlier run (all still 0%
confidently wrong, question ids unchanged):
- **Role-scoped names.** "Rautio stepped down as Chief Risk Officer" is ambiguous across the
  organisation but not within the office — only one Rautio held it — so resignations resolve by
  name within the role when the organisation-wide alias is ambiguous.
- **"Contracted" means holding a contract.** An absence question's gold had counted a supplier
  known only from the certificate register; gold and engine now both mean suppliers with a
  contract (3–9 absence labels per set changed; readers' saved answers are rescored from the spec).
- **Repeating an ambiguous surname is partial**, not confidently wrong, on an unanswerable
  identity question; naming either namesake in full is wrong.
