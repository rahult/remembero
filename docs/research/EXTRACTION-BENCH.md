# Extraction benchmark v1: baselines

Status: first results · 2026-09-08
Evidence: `results/extraction-bench-v1-{luna,llama3.2-3b,qwen2.5-coder-7b}.json` (before guards),
`results/extraction-bench-v1-{luna-guarded,qwen2.5-coder-7b-guarded,qwen2.5-coder-7b-closed,llama3.2-3b-closed}.json`
Run: `npm run eval:extract-bench -- --models <id> [--base-url http://127.0.0.1:11434/v1 --api-key ollama]`

## Why a new suite

The 15-case extraction eval is saturated: frontier models score 100 percent, and it has no
negation, coreference, first-person input, transcript mode, or realistic competitor
predicates. This suite has 100 cases across twelve phenomena, runs through the same
`rememberText` / `rememberTranscriptText` pipeline the product uses (prompt, validator,
gate, retry loop), reports accuracy per phenomenon, and measures **predicate drift**: added
facts whose predicate is neither expected nor already stored.

**Every case is schema-conditioned.** One neutral fact per expected predicate (entity `zed`,
values `sample_k`) is seeded into the store, because the product always sends the store's
schema with the prompt. On an empty store any predicate name is legitimate, so an
unconditioned run measures taste: Luna scored 62.1 percent unconditioned with 31 percent
"drift" that was mostly `prefers` versus the author's `prefers_language`. Conditioned, the
same model scores 83.5 percent with 3.9 percent drift, and every remaining miss is a real
phenomenon.

## Results (100 cases, exact canonical-set match after mutation)

| model                     | accuracy | mutation F1 | drift | pipeline errors |   cost |
| ------------------------- | -------: | ----------: | ----: | --------------: | -----: |
| openai/gpt-5.6-luna       |    83.5% |       0.876 |  3.9% |               0 | $0.025 |
| qwen2.5-coder:7b (Ollama) |    66.0% |       0.709 |  3.9% |               5 |  local |
| llama3.2:3b (Ollama)      |    24.3% |       0.444 |  3.9% |              48 |  local |

Per phenomenon (correct / cases):

| phenomenon           |  Luna | coder 7b | llama 3b |
| -------------------- | ----: | -------: | -------: |
| direct               | 12/12 |    11/12 |     7/12 |
| multi_fact           |   8/8 |      8/8 |      2/8 |
| negation             |  9/10 |     9/10 |     0/10 |
| coreference          |   8/8 |      7/8 |      4/8 |
| first_person         |   3/8 |      1/8 |      0/8 |
| supersession         |   8/8 |      8/8 |      6/8 |
| distractor_prose     |   6/8 |      7/8 |      0/8 |
| competitor_predicate |   8/8 |      4/8 |      2/8 |
| hedge                |   8/9 |      8/9 |      0/9 |
| transcript           |   6/8 |      2/8 |      2/8 |
| entity_normalization |   5/8 |      1/8 |      1/8 |
| date_number          |   5/8 |      2/8 |      1/8 |

## After the write-side guards (same day)

Four deterministic guards were added to the extraction pipeline
(`src/llm/extraction-guard.ts`): an output normalizer (list markers, missing periods, several
facts on one line, prose labels, an invented `assert` keyword), a configured self atom
(`REMBERO_SELF`, default `user`) that rewrites `me`/`the_user`/`you`/... after parsing,
constant grounding (every new constant must appear in the input, loosely matched, or already
be in the store), and predicate aliases (`rembero_predicate_alias(from, to).`) with an
optional closed vocabulary (`REMBERO_EXTRACTION_VOCABULARY=closed`, `--vocabulary closed`).

| model               | before | guards, open | guards, closed | drift (closed) |
| ------------------- | -----: | -----------: | -------------: | -------------: |
| openai/gpt-5.6-luna |  83.5% |        84.5% |              — |              — |
| qwen2.5-coder:7b    |  66.0% |        76.7% |      **81.6%** |           0.0% |
| llama3.2:3b         |  24.3% |            — |          34.0% |           0.0% |

Per phenomenon for the 7B coder model, before → closed: first_person 1/8 → 7/8,
transcript 2/8 → 6/8, competitor_predicate 4/8 → 7/8, distractor_prose 7/8 → 8/8, hedge
8/9 → 9/9; entity_normalization (2/8) and date_number (3/8) barely move, because those are
convention disagreements (quoted `'Mandarin Chinese'` versus `mandarin_chinese`) and
subject-slot choices, not grounding or vocabulary errors. The 3B model's remaining 22
pipeline errors are content: it retracts and re-asserts facts the input never stated
(`initech` for a negation, `sydney` for a hedge), which grounding rejects, so it fails
honestly rather than storing them.

A first version of the self-atom prompt line also said "use only names that appear in the
input", which pushed Luna toward verbatim surface forms (`tuesdays`, `'ledger service'`)
and cost five points; the deterministic guard already enforces grounding, so the sentence
now only forbids inference. Prompt text is not where these fixes live.

## A unified 3B model (query + extraction), first result

`npm run train:data -- --tasks query,extraction` renders facts from the synthetic worlds into
text with Luna and keeps a rendering only when it mentions every constant and no other
world entity, so the extraction gold is exact by construction (720 examples across six
kinds: state, first person, supersession, negation with and without a stored fact, hedge,
distractor). Merged with the query data (20,246 train lines) and trained as one LoRA on
`Llama-3.2-3B` base (round 5; held-out NLL 0.003 on unseen worlds).

| model                                   | extraction (closed) |   query-correct /31 |
| --------------------------------------- | ------------------: | ------------------: |
| llama3.2:3b instruct, untuned           |               24.3% |  11 (published arm) |
| llama3.2:3b instruct, guards            |               34.0% | 19 (closure prompt) |
| **Llama-3.2-3B unified fine-tune (r5)** |           **70.9%** |                  24 |
| Llama-3.2-3B query-only fine-tune (r4)  |                   — |                  26 |
| qwen2.5-coder:7b, guards                |               81.6% |                  26 |
| openai/gpt-5.6-luna                     |               84.5% |                  29 |

Per phenomenon for the unified 3B: direct 12/12, multi_fact 8/8, supersession 8/8,
distractor 8/8, hedge 8/9, negation 8/10, competitor 6/8, transcript 5/8, first_person 5/8,
coreference 4/8, entity_normalization 1/8, date_number 0/8. The three weak phenomena are
exactly the kinds the generator does not produce: no pronoun coreference, no multi-word or
capitalized constants, no dates or numbers. Pipeline errors fell from 22 to 9.

On the read side the unified checkpoint scores 24 query-correct against 26 for the
query-only round 4, with multi-hop 3/6 against 5/6. Two questions on a 31-question set is
within noise, and the misses are the familiar semantic ones plus one new dialect slip (a
goal with only a wildcard, `waits_on_plus(_, procurement_freeze)`, which the engine now
rejects with a fix). Whether the extraction data interferes with query authoring needs more
than one seed to answer.

### Round 6: three more kinds, and a lesson about single runs

The generator gained coreference (a second sentence pronominal), normalized constants (the
text shows `DB Primary` or `check-out`, the fact keeps `db_primary`) and numeric/date
relations (1,177 extraction examples across nine kinds; 20,658 lines total). Trained the
same way on `Llama-3.2-3B`:

| checkpoint              | extraction (closed) | query-correct /31 |
| ----------------------- | ------------------: | ----------------: |
| unified r5 (six kinds)  |               70.9% |                24 |
| unified r6 (nine kinds) |               61.2% |            **27** |

The query side reached the best 3B score in the series, so the mixed data does not hurt
query authoring and r5's 24 was variance. The extraction side fell: coreference 4/8 → 0/8,
distractor 8/8 → 3/8, competitor 6/8 → 2/8, while negation reached 10/10. The raw outputs
show over-abstention (`% nothing` on "Mira joined Acme last year. She lives in Melbourne."),
the schema sample's subject copied into a date fact (`filing_deadline(zed, ...)`), and an
unquoted `new_york`. Whether this is a regression caused by the new kinds or LoRA run
variance cannot be told from one run each; the honest reading is that single fine-tuning
runs on 100-question benchmarks move by ten points on their own, and the next experiment
needs an ablation (r6 data without the new kinds) and repeated seeds before any claim.

A `Qwen3.5-4B` run on the same round-6 data (after the Tinker balance was topped up):

| checkpoint                         | extraction (closed) | query-correct /31 | multi-hop /6 |
| ---------------------------------- | ------------------: | ----------------: | -----------: |
| Llama-3.2-3B unified r6            |               61.2% |                27 |            5 |
| **Qwen3.5-4B unified r6**          |               67.0% |            **28** |        **6** |
| qwen2.5-coder:7b, untuned + guards |               81.6% |                26 |            3 |
| GLM 5.3 / Luna                     |           — / 84.5% |           30 / 29 |        5 / 5 |

Two readings. On the query side the 4B fine-tune is the best small model measured and sits
within one to two questions of the frontier on a schema it never saw. On the extraction side
both base models trained on the round-6 data fail the same three phenomena the same way
(coreference 0–1/8, distractor 3–5/8, date_number 0–1/8, all with over-abstention), which
points at the data, not at run variance or model size: the round-6 kinds as rendered teach
something other than what the benchmark asks. Diagnosing that is the next step, with the
round-5 six-kind data as the control.

### Round 7: the distractor bug, found and fixed

Inspecting the round-6 training file showed **zero distractor examples**: the noise sentence
was passed to the renderer as a request field the real renderer never read, so the 120
"distractor" lines were plain facts (the test renderer had honoured the field itself and
masked the bug). With the prose prepended deterministically (108 lines carry it), the same
Qwen3.5-4B recipe gives:

| checkpoint                         | extraction (closed) | query-correct /31 | multi-hop /6 |
| ---------------------------------- | ------------------: | ----------------: | -----------: |
| Qwen3.5-4B unified r6 (bug)        |               67.0% |                28 |            6 |
| **Qwen3.5-4B unified r7**          |           **77.7%** |            **29** |            5 |
| qwen2.5-coder:7b, untuned + guards |               81.6% |                26 |            3 |
| Luna                               |               84.5% |                29 |            5 |

Distractors 5/8 → 8/8, coreference 1/8 → 7/8, negation 10/10, one pipeline error. On the
query side 29/31 ties Luna and is one behind GLM 5.3.

The remaining extraction misses are mostly a convention conflict in the training data, not a
capability gap: the normalization kind teaches "DB Primary → `db_primary`" because world
atoms are snake_case, while the product prompt (and the benchmark) quote _new_ multi-word
names, `'Blue Harbour Analytics'`, `'New York'`. Both are right in context; the data teaches
only the first, so the model writes `blue_harbour_analytics` and truncates to `chen`,
`dark`, `ledger`. Dates come out unquoted (`20261031`) and first-person subjects are
sometimes dropped. Next data change: a kind with a multi-word entity that is _not_ in the
schema, expected quoted, and quoted ISO dates in more than one relation.

### Rounds 8 and 9: the other half of the quoting convention

Round 8 added a kind that stores a multi-word name the schema does not know as a quoted
constant (`'Blue Harbour Analytics'`) and group-keyed ISO dates. Extraction fell to 49.5%
(query side 28/31, unchanged): the model began quoting **every** capitalized name,
`lives_in(user, 'Melbourne')`, `reports_to(user, 'Liam')`. The data had shown "multi-word
capitalized → quoted" but never "single capitalized word → lowercase atom", because Luna
renders world atoms in lowercase and English capitalizes names. Round 9 displays single-word
entity names capitalized in three quarters of ordinary examples with the fact unchanged
(1,393 extraction examples). Its training run aborted at about 40 percent when the Tinker
balance ran out for the second time; the data is generated and committed, the checkpoint does
not exist.

Two lessons from rounds 6–9 worth keeping. A rendering convention that is right for one class
of constants and silent about its neighbour teaches the wrong generalization; every
convention in the product prompt needs both its positive and its contrasting case in the
data. And a generator bug that a test double papers over (the distractor field) costs a whole
round; test doubles must be at least as ignorant as the real dependency.

### Round 10 and what the series says

Round 10 (single-word attribute values shown capitalized, quoted multi-word names, both
halves of the quoting convention present): **extraction 74.8%, query-correct 27/31**, the
best held-out loss in the series (0.0022). Entity normalization rose from 1/8 to 5/8, so the
convention landed; a few capitalized single words are still quoted (`'Globex'`, `'Liam'`) and
first person slipped to 3/8.

| Qwen3.5-4B checkpoint   | extraction (closed) | query-correct /31 |
| ----------------------- | ------------------: | ----------------: |
| r6 (generator bug)      |               67.0% |                28 |
| r7                      |           **77.7%** |            **29** |
| r8                      |               49.5% |                28 |
| r9                      |               64.1% |                26 |
| r10                     |               74.8% |                27 |
| r11 (= r10 data, Modal) |               76.7% |                27 |

Rounds 7 and 10 are the two coherent data sets and land within three points of each other on
extraction and two on queries. Data rounds have reached diminishing returns on this
benchmark; the gap to the untuned 7B coder (81.6%) and Luna (84.5%) is now first-person
naming, transcript mode, and dates, each a handful of cases.

### Round 11: the same data trained twice

Round 11 is not a data round. It is r10's exact training file trained again on a different
stack (a self-managed H100 on Modal, TRL + PEFT, merged weights served by vLLM; see
`benchmarks/modal/README.md`) instead of Tinker. It is therefore the first sample of the
training-run variance the paragraph below said was missing: **extraction 76.7% vs 74.8%,
query-correct 27 vs 27**, with 19 of the two runs' 24 and 26 extraction failures shared and
three of four query failures shared (j2, j5, m4; r10 also missed j4, r11 also missed a6).
Per phenomenon the two runs agree except for a case or two in negation, first person, and
entity normalization. Two consequences:

- The run-to-run noise on this benchmark is about two points of extraction and one query,
  so r7's 77.7% / 29 is not distinguishable from r10 and r11. The whole r7–r11 series is one
  plateau, and the next gain has to come from the data kinds the failures name (transcript
  mode 2/8, dates 1/8, first person 5/8), not from another retrain.
- The two training stacks agree closely enough that the cheaper one can be used from here:
  the Modal run cost about $1.50 of H100 time for 23 minutes wall, against about $5 on Tinker.

**Seeds.** Three evaluation seeds (7, 42, 123) on the query benchmark gave identical results
for both checkpoints (r7: 29, 29, 29; r10: 27, 27, 27) because the served model samples
greedily at temperature 0. Evaluation variance is therefore nil; the variance that matters is
between _training_ runs, and each sample of it costs a training run (~$5 on Tinker, ~$1
self-managed per [the provider matrix](FINETUNE-PROVIDER-MATRIX.md)). No such repeat has
been done, so every number in this document is one training run.

### Rounds 12 and 13: reading the failures instead of adding rounds

Round 11's failures were read case by case rather than counted. Five patterns accounted for
most of them: the model copied the schema sample's subject (`zed`) when the text named no
entity; it had never seen a generic subject (`the team`, `the project`); it stored what the
assistant said in a transcript; it mapped "we" to the self atom; and it swapped the order of a
three-place fact. Each became data (an `implicit_subject` kind, a `transcript` kind with
acknowledgements, guesses, summaries, tool output, confirmations and code, first-person
examples over relations, schedule facts in the state pool) and two of them became prompt
rules for every model: "we / our / my team" is the group, not the speaker, and an unnamed
subject is the generic noun the text describes, never a schema sample. Four benchmark inputs
whose gold subject did not appear in the text (which the grounding guard forbids, so every
model scored zero on them) now name it; this is benchmark v1.1.

Round 12: **87.4%** (90/103), query 27/31. Luna rerun on v1.1 with the new prompt: 93.2%.

Round 13 changed the data for the shape of real transcripts (see the LongMemEval section in
[LONGMEMEVAL.md](LONGMEMEVAL.md)): facts embedded in long requests with long assistant replies,
and a quarter of examples with an empty schema so predicates are named from the text. It
scored 83.5% before and **91.3%** (94/103) after one more write-side guard, and 28/31 on
queries. The guard exists because seven of the eleven new failures were the quoting flip
rounds 8 and 9 showed (`'Toronto'`, `'Liam'`): the convention is fragile in a 4B model and
moves between training runs, so the two unambiguous halves are now enforced in code. A quoted
single capitalized word becomes a lowercase atom and a quoted phrase of lowercase words (or
hyphenated words) becomes snake_case; multi-word proper names, acronyms, and anything with
digits or punctuation stay quoted. Luna is unaffected by the guard (93.2% either way).

| Qwen3.5-4B checkpoint | extraction (closed) | query-correct /31 | note                                       |
| --------------------- | ------------------: | ----------------: | ------------------------------------------ |
| r11                   |               76.7% |                27 | r10 data, Modal                            |
| r12                   |               87.4% |                27 | implicit subjects, transcripts, v1.1       |
| r13                   |               91.3% |                28 | embedded facts, empty schemas, + guard     |
| r14                   |               88.3% |                27 | + event asides (most facts on LongMemEval) |
| openai/gpt-5.6-luna   |               93.2% |                29 | v1.1, same prompt and guards               |

Round 14 added the event kind (first-person asides such as "by the way, I just got back
from a three-day trip to Big Sur", half as text and half as transcripts) for the LongMemEval
shape. On this benchmark it sits three cases under r13, inside the noise band, trading three
fixes for a few argument-order swaps and one multi-word name written snake_case instead of
quoted; on LongMemEval it extracts facts from far more sessions than r13 (63% against 28%)
and is the served default, though on the full development split hybrid formation with its
facts matches raw retrieval on accuracy rather than beating it (see that document).

### Base-model matrix and the size of the noise (2026-09-10)

Four Qwen3.5-4B runs on identical r14 data and two Gemma 4 runs on the same data, all rank-32
LoRA, one epoch, evaluated with the same prompts and guards:

| run                | base        | query /31 | extraction /103 | held-out loss | train (H100) |
| ------------------ | ----------- | --------: | --------------: | ------------: | -----------: |
| r14                | Qwen3.5-4B  |        27 |              91 |        0.0025 |       40 min |
| r14b               | Qwen3.5-4B  |        26 |              89 |        0.0039 |       40 min |
| r15 (batch 4 × 16) | Qwen3.5-4B  |        26 |              88 |        0.0028 |       46 min |
| r15 (batch 4 × 16) | Qwen3.5-4B  |        27 |              85 |        0.0036 |       44 min |
| r16                | Gemma 4 E4B |        24 |              84 |        0.0031 |       55 min |
| r16                | Gemma 4 E2B |        28 |           85–86 |        0.0037 |       43 min |

Four runs of one recipe span 26–27 queries and 85–91 extraction cases, so the training-run
noise is about one query and three extraction cases either side. Nothing in the table clears
it: Gemma 4 E2B, a 2.3B-effective model, ties Qwen3.5-4B on both benchmarks; Gemma 4 E4B is a
query below the band on one run. The base model is not where the remaining performance is.

Two engineering notes from the Gemma runs. Gemma 4 wraps each projection in a clipping module
PEFT cannot adapt, so the trainer now names the inner Linear layers; its KV-sharing layers
have no key/value parameters in transformers but vLLM requires them, so merged checkpoints are
re-exported text-only with Google's original tensors restored (`export_text_only`,
`restore_dropped_weights` in `benchmarks/modal/train_lora.py`). E4B needs more than an L4 to
serve with an 8k context. And the E4B run exposed an engine bug worth more than the run: it
writes a named rule plus an explicit `?- goal.` line, which the MCP query tool accepts but
the SQLite bridge sent to the native parser, scoring 16/31 until the bridge routed such
programs to the portable engine (24/31 after; r14 unchanged at 27).

The remaining nine r13 misses are one-offs: a dropped word (`dark` for `dark_mode`), a
hallucinated fact from CI noise, a manager/report direction, two generic-subject choices
(`deadline`, `engineers`), and two cases Luna also misses.

## What the failures are (first run, before guards)

- **First person has no name.** Luna wrote `the_user`, `me`, `you` and `user` for "I" across
  eight cases; the 7B model got one right; the 3B none. The pipeline has no self atom. This
  is the single largest frontier miss and is a configuration gap, not a model gap.
- **Negation is stored as a positive fact** by Luna in one case (`never_lived_in(ava, lisbon)`)
  and the 3B model stores the negated fact itself (`works_at(mira, initech)` for "no longer
  works at") in every case. Nothing in the pipeline checks a stored fact against the input.
- **Distractor prose becomes facts**: `closed_on(office_kitchen, friday)`,
  `build_status(build, red)`, `test_failures('cache.test.ts', 3)`. The prompt says to skip
  transient details; a small model does not.
- **Transcript mode** loses subjects (`prefers_package_manager(pnpm)`), capitalizes quoted
  constants (`'Dana'`, `'Melbourne'`), and on the 7B model stores assistant summaries.
- **Normalization and dates** drift in arity and subject: `filing_deadline('2026-10-31')`
  without a subject, `budget(250000)`, `mandarin_chinese` versus `'Mandarin Chinese'`.
- **The 3B model fails the output contract** before content: 28 of its 48 pipeline errors
  are markdown bullets in front of facts, 10 are missing periods between facts. The retry
  loop feeds back parser errors and the model does not recover.

## What this says about the write-side plan

In the order the adversarial review gave: a configured self atom fixes the largest frontier
gap outright; quote-grounding (every added fact must cite a substring of the input, and its
constants must appear there) would reject the distractor and negation-as-positive classes
for every model size; closed vocabulary with aliases is what the 7B model needs on
competitor predicates (4/8); functional-dependency supersession in the store already works
here because the prompt's retract instruction is being followed, but that is the instruction
most likely to be missed on real input. The 3B output-contract failures argue for a
structured output format (one fact per line, no prose) enforced by a lenient normalizer
before the parser, not for more prompt text.

## Limitations

- Expected outputs are the author's; the schema seeding removes predicate-name arbitrariness
  but argument order and constant form are still conventions of this suite.
- Single run per model at the pipeline's default temperature; local models via Ollama's
  OpenAI-compatible endpoint.
- Transcript cases use the `USER:` / `ASSISTANT:` block format the capture path produces,
  not real Claude Code transcripts.
