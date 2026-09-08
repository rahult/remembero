# Extraction benchmark v1: baselines

Status: first results · 2026-09-08
Evidence: `results/extraction-bench-v1-{luna,llama3.2-3b,qwen2.5-coder-7b}.json`
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

## What the failures are

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
