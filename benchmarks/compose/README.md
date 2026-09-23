# Compose: composed questions over generated worlds, with certain answers

Questions that combine 1–6 facts (identity, validity on a date, supersession, comparison,
aggregation, absence, authority) spread through 100, 500 and 1000-page documents, with every
gold answer **computed by the Datalog engine from the facts the documents were written from**.
Design and rationale: `docs/research/MOONSHOT-CERTAINTY.md`.

```bash
npm run compose:build -- test        # worlds 101 and 102, 93 questions, three tiers
npm run eval:doc-recall -- --spec benchmarks/compose/test/spec.json --top-k 12 \
  --pages-per-window 1 --context-bytes 49152 --reader-model deepseek-chat \
  --reader-base-url https://api.deepseek.com/v1 --reader-api-key "$DEEPSEEK_API_KEY" \
  --output results/compose-deepseek-product-d12.json
node benchmarks/compose/report.mjs results/compose-*.json   # rescored from the spec
```

- `src/compose/world.ts` — seeded organisation; names from disjoint train/dev/test pools.
- `src/compose/program.ts` — the world as Datalog facts plus the rules gold answers come from.
- `src/compose/render.ts` — board minutes, registers, amendments, certificate and incident logs;
  dates in four formats, money in three, approvers named by name, initial or role only;
  proposals that are deliberately not facts.
- `src/compose/questions.ts` — nine families, each question with its engine query, the gold,
  the accepted spellings and the distractors (plausible wrong answers).
- `src/compose/haystack.ts` — world pages at fixed relative depths inside real public-document
  filler; tiers are nested, so only the amount of surrounding text changes between them.
- `src/compose/score.ts` — no model in the loop: correct, **wrong** (confidently wrong),
  partial, or declined. The answer's claim is the candidate it names first, so explaining away a
  superseded value or the other contract is not penalised.
- `benchmarks/compose/test/w*.dl`, `w*.questions.jsonl` — the audit trail: every fact, every
  rule, every question's engine query.

Documents are presented undated (`undated` in the reading prompt): the harness's old chat
framing gave each page a synthetic 2020 date and a "current date", and on dated documents the
reader reasoned from it ("as of 2021 the 2023 contract was not yet signed").

Known v0 limits: world pages are lexically distinct from the filler, so retrieval finds them
82–98% of the time at every tier; v1 needs same-domain distractor organisations in the filler.
Seven templates per fact type; an LLM paraphrase pass with fact-recovery checks is planned.
