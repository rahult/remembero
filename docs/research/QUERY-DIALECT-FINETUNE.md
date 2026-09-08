# Fine-tuning a 3B model on the flat query dialect: findings

Status: first results · 2026-09-08 · single seed, one model family
Evidence: `results/agent-boundary-v2-finetune-llama3.2-3b-r{1,2,3}-remembero-closure-summary.json`,
`results/agent-boundary-v2-*-remembero-closure-summary.json` (untuned baselines),
`results/agent-boundary-v2-{z-ai-glm-5.3,openai-gpt-5.6-luna}-summary.json` (frontier, all
four conditions), and the published four-model matrix ([findings](AGENT-BOUNDARY-FINDINGS.md)).
Method: [closure predicates](../CLOSURE-PREDICATES.md), [training-data design](../superpowers/specs/2026-09-08-query-dialect-training-data-design.md),
[benchmarks/tinker/README.md](../../benchmarks/tinker/README.md).

## Question

The agent-boundary matrix showed the one Datalog skill small local models lack is authoring
recursive rules; every multi-hop miss was a recursion error. Two changes followed. The engine
now synthesizes transitive closures on demand (`p_plus`), so chains are one flat literal. And
a data generator emits execution-verified question → program pairs in that flat dialect
across synthetic worlds that share no predicate names with the benchmark. Does a 3B model
fine-tuned on that data author correct queries on the benchmark's unseen schema?

## Setup

- **Base model:** `meta-llama/Llama-3.2-3B` on Tinker (the base model; Tinker lists no
  Llama-3.2 instruct at this size and no Qwen 3.5 2B). LoRA rank 32, one epoch, lr 2e-4,
  batch 64, `role_colon` renderer, train on assistant messages.
- **Data:** `npm run train:data`, 60 worlds, 3 Luna paraphrases per templated example,
  whole worlds held out. Three rounds, each changing only the generator:
  1. baseline vocabulary (14,967 train lines, 3.9M tokens);
  2. mirrored-orientation relations so the same verbs appear with both argument orders
     (16,234 lines, 4.3M tokens);
  3. one-hop edge templates with explicit "direct / one step" wording (11,211 lines,
     3.0M tokens; the fixed example target was met in fewer rounds).
- **Evaluation:** the unchanged agent-boundary harness, `remembero-closure` condition,
  seed 7, served through the cookbook's OpenAI-compatible proxy (`--chat-api openai`). The
  31 benchmark questions never appear in training.
- **Two metrics.** _End-to-end_ is the published metric: the model also reads the rows back
  and its prose is graded. _Query-correct_ is new: the model-authored query is executed and
  its rows are checked against the gold answer set, so the answer step is excluded.

## Results (out of 31; multi-hop out of 6)

Query-correct is computed by `node dist/evals/regrade-agent-boundary.js <results.json>`,
which re-executes every stored query on a fresh seeded database and grades the rows with
`gradeQueryRows`: every expected entity present, no forbidden entity, **no seed entity
outside the gold set** (a superset fails), and yes/no by row presence with a lone
`No`/`0`/`false` value counted as no. An earlier draft of this table used a grader without
the superset and negative-value rules; it inflated several cells by one or two (the
fine-tune rounds 2 and 3 read 27, the coder model's gated-SQL arm read 26). The numbers
below are the corrected ones and are reproducible from the committed result files.

| model                                        | condition             | query-correct | end-to-end | multi-hop (query) |
| -------------------------------------------- | --------------------- | ------------: | ---------: | ----------------: |
| llama3.2:3b instruct                         | remembero (published) |            11 |         10 |                 2 |
| llama3.2:3b instruct                         | remembero-closure     |            19 |         17 |                 4 |
| llama3.1:8b instruct                         | remembero-closure     |            19 |         20 |                 4 |
| qwen2.5-coder:7b                             | sql-gated             |            24 |         23 |                 2 |
| qwen2.5-coder:7b                             | remembero-closure     |            26 |         25 |                 3 |
| **Llama-3.2-3B fine-tune, round 1**          | remembero-closure     |            24 |         23 |                 3 |
| **Llama-3.2-3B fine-tune, round 2**          | remembero-closure     |            26 |         21 |                 5 |
| Llama-3.2-3B fine-tune, round 3              | remembero-closure     |            26 |         25 |                 5 |
| Llama-3.2-3B fine-tune, round 3              | closure, v2 prompt    |            24 |         23 |                 4 |
| **Llama-3.2-3B fine-tune, round 4**          | closure, v2 prompt    |            26 |         26 |                 5 |
| Llama-3.2-3B unified r6 (query+extraction)   | closure, v2 prompt    |            27 |         27 |                 5 |
| **Qwen3.5-4B unified r6 (query+extraction)** | closure, v2 prompt    |        **28** |     **27** |             **6** |
| openai/gpt-5.6-luna (frontier)               | sql-gated             |            30 |         31 |                 6 |
| openai/gpt-5.6-luna (frontier)               | remembero-closure     |            29 |         29 |                 5 |
| z-ai/glm-5.3 (frontier)                      | sql-gated             |            30 |         31 |                 6 |
| z-ai/glm-5.3 (frontier)                      | remembero-closure     |            30 |         30 |                 5 |

Every fine-tuned round refused all six trap writes and made zero or one tool error across
31 questions; every program parsed and ran.

## Findings

1. **A 3B model fine-tuned on synthetic worlds matches the best untuned model measured on
   an unseen schema, and more than doubles the same-size instruct model.** 26/31
   query-correct, level with `qwen2.5-coder:7b` on the closure arm (26) and above its
   gated-SQL arm (24). The same-size instruct model scores 11 on the published arm and 19
   with the closure prompt. It does not beat the 7B coder model; an earlier draft of this
   document said it did, on the basis of the inflated grader.

2. **Direction was the residual error, and it was a data artefact.** Round 1 misses were
   `waits_on_plus(procurement_freeze, R)` and `reports_to_plus(dana, M)` with the anchor in
   the wrong position. In round 1's vocabulary the phrase "reports to" only ever occurred
   with `manages(Manager, Person)`, where the anchor does go first: the model learned the
   phrase, not the argument names. Adding mirrored relations (round 2) fixed both.

3. **The answer step is noise for small models in both directions.** With correct rows in
   hand, the 3B base model echoes the prompt, and the 3B instruct model says "no
   information". Round 2 scored 26 query-correct but 21 end-to-end for this reason alone. A
   decision engine should render rows plus proof deterministically; the harness now reports
   query-correct so the two legs are never conflated again.

4. **The remaining misses are coverage and semantics, not syntax.** Round-3 query misses:
   two join-predicate misreads (`blocker` versus `status`), a question that asks for yes/no
   _and_ the chain (the yes/no pattern hides the chain by design), an existence pattern
   ("manages at least one person") no template teaches, and "direct manager" answered with
   the whole chain in round 2 (the superset the corrected grader now catches).

5. **The frontier gap on query authoring is three to four questions.** GLM 5.3 and Luna,
   run through the same harness via OpenRouter (`--chat-api openai`), score 30 and 29
   query-correct on the closure arm against the fine-tune's 26. All three miss m4, which
   asks for yes/no _and_ the chain: a prompt-and-grading interaction shared by every model.
   Frontier models still make Datalog tool errors on the published `remembero` arm (GLM 5.3:
   5 across 31 questions) and none on SQL, the training-prior effect the v1 analysis
   predicted.

6. **Cost.** Each round trained in about seven minutes on 3 to 4.3 million tokens; the
   three rounds together were a few dollars of Tinker time plus roughly 5,000 cached Luna
   paraphrase calls.

## Round 4 (after the review)

Round 4 was trained after the adversarial review with every data defect below fixed and
three product changes that also changed the evaluation prompt ("v2 prompt"): the query
target is the sink rule rather than the first rule, a ground goal answers with one boolean
row (`?- p_plus(a, b).` → `yes = true|false`) instead of the `q(Y) :- …, Y = b.` idiom, and
the engine reports diagnostics instead of silent empties. Data: 2 fixed rounds per world
(19,598 train, 1,698 held-out lines), yes/no as ground goals, ternary schedule templates,
mirrored vocabulary, whole-word paraphrase filter. **True held-out NLL on unseen worlds:
0.0043**, the first honest held-out number in this series.

Result: 26/31 query-correct and 26/31 end-to-end, multi-hop 5/6, direct 6/6, all trap
writes refused. The round-3 checkpoint under the same v2 prompt scores 24 with five tool
errors, because it was trained on the old yes/no idiom: the retrain was needed to keep pace
with the dialect change, and it recovers the previous best while the product moved under
it. Remaining misses are unchanged in kind: two `blocker`/`status` join misreads, "manages
at least one person", the yes/no-plus-chain question m4, and one new failure where the
model used a helper name from the few-shot examples (`no_slot`) as if it were a stored
predicate.

## Defects found in review after these runs (fixed before round 4)

- **Two vocabulary entries had their up/down phrases inverted** (`supplied_by`, `follows`),
  so about 300 of the 11,211 round-3 training lines said the opposite of what their program
  computes. Fixed in `src/training/worlds.ts` with a per-relation direction test; the
  round-3 checkpoint was trained on the flawed data.
- **The "held-out NLL" reported during training was not held out.** The recipe passed
  `test_size=100`, which slices the training file; `heldout.jsonl` was never read. The
  round-3 value of 0.0001 therefore measures paraphrase siblings, not generalization. The
  benchmark numbers above are unaffected, since the 31 questions are on a foreign schema.
- **Held-out worlds share every predicate, entity and phrase template with training**,
  because worlds are drawn from three themes. They test new fact sets, not new vocabulary.
- **The paraphrase filter matched constants as substrings** ("search" satisfied by
  "searched"); seven generated lines lost their anchor entity. Fixed with whole-word
  matching.
- **`--examples` is a global pre-paraphrase cap**, so adding the one-hop template in round 3
  shrank the rest of the data (16,234 to 11,211 lines). Rounds 2 and 3 are therefore not
  a clean before/after for that template.

## Limitations

- Single seed at temperature 0; no variance measured. The 31-question benchmark makes
  differences of one or two questions indistinguishable from noise.
- The training system prompt (schema plus dialect card, no few-shots) differs from the
  evaluation prompt (cheatsheet plus eight few-shots), and the closure prompt contains a
  yes/no example one constant away from benchmark question m5.
- The fine-tune starts from the base model while the Ollama baselines are instruct models,
  so the same-size comparison is same parameter count, not same weights.
- One model family. Qwen3.5-4B is the next planned run.
- The training worlds are small and synthetic; nothing yet shows transfer to a real
  captured-memory store.
- The `remembero-closure` prompt was authored after the closure engine change and is not
  the published `remembero` prompt; the untuned baselines in the table above use it too.

## Reproduce

```bash
npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training
.venv/bin/python benchmarks/tinker/sl_query_dialect.py --data data/training/conversations.jsonl \
  --model Llama-3.2-3B --log-path runs/tinker/llama-3.2-3b-dialect
# then serve and evaluate per benchmarks/tinker/README.md
```

Checkpoints (Tinker, 7-day TTL): round 3 `tinker://4488229d-504c-5294-a3dc-d0274d214c9d:train:0/sampler_weights/final`,
round 4 `tinker://f428ffcf-055e-523b-9a19-297897eeaf7b:train:0/sampler_weights/final`.
