# Fine-tuning a 3B model on the flat query dialect: findings

Status: first results · 2026-09-08 · single seed, one model family
Evidence: `results/agent-boundary-v2-finetune-llama3.2-3b-r{1,2,3}-remembero-closure-summary.json`,
`results/agent-boundary-v2-*-remembero-closure-summary.json` (untuned baselines), and the
published four-model matrix ([findings](AGENT-BOUNDARY-FINDINGS.md)).
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

| model                               | condition                    | query-correct | end-to-end | multi-hop (query) |
| ----------------------------------- | ---------------------------- | ------------: | ---------: | ----------------: |
| llama3.2:3b instruct                | remembero (published)        |            13 |         10 |                 2 |
| llama3.2:3b instruct                | remembero-closure            |            19 |         17 |                 4 |
| llama3.1:8b instruct                | remembero-closure            |            20 |         20 |                 4 |
| qwen2.5-coder:7b                    | sql-gated (best untuned arm) |            26 |         23 |                 2 |
| qwen2.5-coder:7b                    | remembero-closure            |            26 |         25 |                 3 |
| **Llama-3.2-3B fine-tune, round 1** | remembero-closure            |            24 |         23 |                 3 |
| **Llama-3.2-3B fine-tune, round 2** | remembero-closure            |            27 |         21 |                 5 |
| **Llama-3.2-3B fine-tune, round 3** | remembero-closure            |        **27** |     **25** |             **5** |

Every fine-tuned round refused all six trap writes and made zero or one tool error across
31 questions; every program parsed and ran.

## Findings

1. **A 3B model fine-tuned on synthetic worlds authors better queries on an unseen schema
   than any untuned model measured, including the 7B coder model.** 27/31 query-correct
   against 26 for the best untuned arm. On the published metric for the same-size instruct
   model the gap is 25 versus 10.

2. **Direction was the residual error, and it was a data artefact.** Round 1 misses were
   `waits_on_plus(procurement_freeze, R)` and `reports_to_plus(dana, M)` with the anchor in
   the wrong position. In round 1's vocabulary the phrase "reports to" only ever occurred
   with `manages(Manager, Person)`, where the anchor does go first: the model learned the
   phrase, not the argument names. Adding mirrored relations (round 2) fixed both.

3. **The answer step is noise for small models in both directions.** With correct rows in
   hand, the 3B base model echoes the prompt, and the 3B instruct model says "no
   information". Round 2 scored 27 query-correct but 21 end-to-end for this reason alone. A
   decision engine should render rows plus proof deterministically; the harness now reports
   query-correct so the two legs are never conflated again.

4. **Every remaining miss names a template the generator lacks or a genuine misread.** The
   four round-3 query misses: two join-predicate misreads (`blocker` versus `status`), a
   question that asks for yes/no _and_ the chain (the yes/no pattern hides the chain by
   design), and an existence pattern ("manages at least one person") no template teaches.
   Held-out-world NLL was 0.007 or lower every round: the dialect itself is learned; what
   is left is coverage.

5. **Cost.** Each round trained in about seven minutes on 3 to 4.3 million tokens; the
   three rounds together were a few dollars of Tinker time plus roughly 5,000 cached Luna
   paraphrase calls.

## Limitations

- Single seed at temperature 0; no variance measured. The 31-question benchmark makes
  differences of one or two questions indistinguishable from noise.
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

Checkpoints (Tinker, 7-day TTL): round 2 `tinker://4208a773-116d-5356-8fa9-754d07ca2f24:train:0/sampler_weights/final`,
round 3 `tinker://4488229d-504c-5294-a3dc-d0274d214c9d:train:0/sampler_weights/final`.
