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

## The engine path (Proved or Unknown)

```bash
# v1: sister organisations in the filler and paraphrased questions; v2 also rewords the documents
npm run compose:build -- test --v1 --seeds 103,104 --name holdout-v1
npm run compose:build -- test --v2 --seeds 105,106 --name holdout-v2

# extraction (two passes on record-like pages) + planner + Datalog, deepseek for both models
npm run compose:engine -- --spec benchmarks/compose/holdout-v1/spec.json --planner llm \
  --output results/compose-engine-llmplanner-holdout-v1.json
```

- `src/compose/extract.ts` — the organisation-aware schema, the page check (every date, amount
  and name on the page), the same-line rule (every argument but the organisation stated by one
  line), and type repairs (a swapped contract reference is put back; a role title is not a person).
- `src/compose/engine-answer.ts` — rules for aliases, roles by date, values after amendments,
  authority and certificate validity, each ranging only over the dates it can need; Unknown on no
  row, conflicting rows, unresolved approvers, orphan incidents, or a gap in amendment numbering.
- `src/compose/planner.ts` — question → one of nine query shapes and its parameters; a parameter
  not found in the question is rejected.

## A local engine model

```bash
node dist/evals/run-compose-trainset.js --worlds 250 --paraphrase 3000   # data/compose-engine/
python benchmarks/runpod/volume.py put data/compose-engine/conversations.jsonl data/compose-engine-e2b/conversations.jsonl
python benchmarks/runpod/volume.py put data/compose-engine/heldout.jsonl data/compose-engine-e2b/heldout.jsonl
python benchmarks/runpod/pod.py create-train --run compose-engine-e2b --base-model google/gemma-4-E2B-it \
  --gpu "NVIDIA H200" --cloud SECURE --max-length 4096 --batch-size 16 --grad-accum 4 --max-hours 3
python benchmarks/runpod/volume.py get runs/compose-engine-e2b/compose-engine-e2b-Q8_0.gguf /Volumes/Atlas/models/rembero/compose-engine-e2b-Q8_0.gguf
llama-server -m /Volumes/Atlas/models/rembero/compose-engine-e2b-Q8_0.gguf --port 8085 -c 8192 -np 2 \
  --alias compose-engine-e2b --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
npm run compose:engine -- --spec benchmarks/compose/holdout-v1/spec.json --tiers 100p \
  --model compose-engine-e2b --base-url http://127.0.0.1:8085/v1 --api-key none --price 0,0 \
  --planner llm --planner-model compose-engine-e2b --planner-base-url http://127.0.0.1:8085/v1 --planner-api-key none
```

The first run: 20,386 rows from 250 train-split worlds, 319 steps on an H200, 54 minutes, $4.55,
held-out loss 0.00098. On a 16 GB Mac the model extracts at roughly one page per 10–20 seconds
(prompt processing dominates), so 1000-page haystacks are hours of local time.
