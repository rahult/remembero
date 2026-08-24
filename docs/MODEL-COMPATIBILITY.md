# Model compatibility

Remembero keeps rule evaluation, proofs, graph construction, schema selection, and final
deterministic answer rendering local. The configured model translates natural-language
memory and recall requests; it is not the reasoning authority.

Semantic preference retrieval has a separate embedding model. The default
`perplexity/pplx-embed-v1-0.6b` was selected on the deterministic LongMemEval-S
development split with routing, chunking, top-k, export safety, and the lexical guard held
fixed:

| Embedding model | Dev Recall@5 | Dev MRR | p95 | Charged cost | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `perplexity/pplx-embed-v1-0.6b` | 86.7% | 75.0% | 9.2 s | $0.004485 | default |
| `qwen/qwen3-embedding-8b` | 80.0% | 61.6% | 39.8 s | $0.011240 | rejected |
| `perplexity/pplx-embed-v1-4b` | 80.0% | 66.7% | 12.2 s | $0.033635 | rejected |
| `nvidia/nemotron-3-embed-1b:free` | not run | not run | not run | $0 | unavailable under account privacy policy |

Larger eligible models were stopped before held-out evaluation because none improved the
development result. The free route was rejected rather than weakening the account's data
policy. This is a dated provider measurement, not a universal model ranking; the raw
decision record is in
[`research/results/semantic-model-matrix-v1-summary.json`](research/results/semantic-model-matrix-v1-summary.json).

## Verified models

The following live OpenRouter comparisons were run on 2026-08-17 AEST. Every run used
temperature zero and the real storage/reasoning pipeline.

### Recall translation

The v0.47 grounded projection corpus has 26 cases and 100 distractor predicates:

| Model | Correct | Accuracy | Precision / recall / F1 | Budget exhausted | Errors | Observed seconds |
|---|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.6-luna` | 26/26 | 100.0% | 100.0% / 100.0% / 100.0% | 0 | 0 | 65.2 |
| `google/gemini-3.7-flash` | 26/26 | 100.0% | 100.0% / 100.0% / 100.0% | 0 | 0 | 129.6 |
| `anthropic/claude-sonnet-5` | 26/26 | 100.0% | 100.0% / 100.0% / 100.0% | 0 | 0 | 125.7 |
| `openai/gpt-5.4-mini` | 26/26 | 100.0% | 100.0% / 100.0% / 100.0% | 0 | 0 | 35.3 |

### Personal knowledge extraction

The v0.40 corpus has 15 exact mutation cases, including corrections, rules, tentative
trust, authority no-ops, and local secret rejection:

| Model | Correct | Accuracy | Mutation precision / recall / F1 | Safety | Errors | Observed seconds |
|---|---:|---:|---:|---:|---:|---:|
| `openai/gpt-5.6-luna` | 15/15 | 100.0% | 100.0% / 100.0% / 100.0% | 100.0% | 0 | 22.0 |
| `google/gemini-3.7-flash` | 15/15 | 100.0% | 100.0% / 100.0% / 100.0% | 100.0% | 0 | 38.7 |
| `anthropic/claude-sonnet-5` | 15/15 | 100.0% | 100.0% / 100.0% / 100.0% | 100.0% | 0 | 48.8 |
| `openai/gpt-5.4-mini` | 14/15 | 93.3% | 91.7% / 91.7% / 91.7% | 100.0% | 0 | 13.1 |

Observed duration is diagnostic only. Provider load, routing, and model revisions can
change it, so it is not a release threshold.

## Recommendation

- Use `anthropic/claude-sonnet-5` as the default for model-assisted writes and recall. In the
  21 August refresh it was the only tested model that reached 100% on both full checkpoints
  without an operational error, while costing less and completing faster than Gemini 3.1 Pro.
- Use `openai/gpt-5.6-luna` as the economy option. It remained about an order of magnitude
  cheaper and passed extraction, but one of 26 recall cases exhausted its repair attempts after
  emitting invalid syntax in the current run. Earlier 26/26 runs show live-model variance, not
  a deterministic regression.
- Use `google/gemini-3.1-pro-preview` when independent provider diversity matters; it also
  passed both current checkpoints but used more tokens, time, and cost than Sonnet 5.
- GPT-5.4 Mini now passes the recall checkpoint after explicit relational projection, but
  it remains below the combined recommendation because it changed `dr_chen` to `chen` in
  the separate extraction checkpoint and costs more than the default catalog snapshot.

## Provider-reported recall cost

Measured 20 August 2026 AEST over the same five representative recall cases with the
grounded prompt and eight detailed predicates. These values come from each response's
OpenRouter `usage.cost`, not catalog-price multiplication:

| Model | Accuracy | Input / output tokens | Seconds | Total cost | Average/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai/gpt-5.6-luna` | 100% | 29,014 / 339 | 23.7 | $0.001004 | $0.000201 |
| `google/gemini-3.7-flash` | 100% | 33,178 / 1,553 | 34.0 | $0.015354 | $0.003071 |
| `openai/gpt-5.4-mini` | 100% | 26,816 / 127 | 11.2 | $0.016018 | $0.003204 |
| `anthropic/claude-sonnet-5` | 100% | 40,109 / 156 | 32.9 | $0.081778 | $0.016356 |

Luna was about 15x cheaper per query than the next-lowest charged result in this slice.
Rerun `npm run bench:agent-db:cost` because provider routes and charges can change.

The OpenRouter catalog snapshot observed during the same session listed these prices per
million tokens:

| Model | Input | Output |
|---|---:|---:|
| `openai/gpt-5.6-luna` | $0.10 | $0.60 |
| `google/gemini-3.7-flash` | $0.375 | $1.875 |
| `openai/gpt-5.4-mini` | $0.75 | $4.50 |
| `anthropic/claude-sonnet-5` | $2.00 | $10.00 |

These prices are a dated catalog observation, not a billing guarantee. Check OpenRouter
before making a purchasing decision.

## Boundary of the evidence

These evaluations measure question-to-query translation, exact retrieved bindings, and
exact personal-knowledge mutations. They do not compare final prose phrasing or broad
open-domain semantic coverage beyond the labeled cases.

The ranker deterministically treats `grandchild` and `grandparent` as the same
kinship concept for schema selection. This makes the authored `grandparent` rule and its
dependencies visible under a bounded 100-predicate distractor load. The prompt then asks
the model to query a matching derived head instead of leaking rule-local helper variables.
Evaluation still runs the accepted query over the complete selected knowledge view.
Version 0.47 additionally makes answer columns explicit, so valid inlined joins cannot
leak helper variables even when a model does not choose the derived head predicate.

Run the checkpoint yourself:

```bash
LLM_API_KEY="$OPENROUTER_API_KEY" npm run eval:recall -- \
  --models openai/gpt-5.6-luna,google/gemini-3.7-flash,anthropic/claude-sonnet-5,openai/gpt-5.4-mini \
  --variants grounded
LLM_API_KEY="$OPENROUTER_API_KEY" npm run eval:extract -- \
  --models openai/gpt-5.6-luna,google/gemini-3.7-flash,anthropic/claude-sonnet-5,openai/gpt-5.4-mini
```

Live zero-temperature runs can still move when a provider changes routing or model
weights. The deterministic corpus and scorer, rather than this dated result, remain the
repeatable release evidence.

## 21 August 2026 frontier refresh

The current full-corpus runs compare the shipping default/economy model with three current
frontier families. Raw observations, provider usage, latency, and errors are frozen in:

- `research/results/economy-luna-recall-v1-summary.json`
- `research/results/economy-luna-extraction-v1-summary.json`
- `research/results/frontier-recall-v1-summary.json`
- `research/results/frontier-extraction-v1-summary.json`

| Model | Recall accuracy | Recall tokens | Recall cost | Recall time | Extraction accuracy | Extraction tokens | Extraction cost | Extraction time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anthropic/claude-sonnet-5` | 100% | 169,067 | $0.345182 | 145.4 s | 100% | 20,703 | $0.045646 | 54.0 s |
| `google/gemini-3.1-pro-preview` | 100% | 142,250 | $0.356040 | 147.1 s | 100% | 18,811 | $0.077842 | 62.7 s |
| `openai/gpt-5.4` | 100% | 110,721 | $0.227240 | 74.9 s | 93.3% | 13,321 | $0.035703 | 22.6 s |
| `openai/gpt-5.6-luna` | 96.2% | 120,454 | $0.020846 | 103.2 s | 100% | 13,507 | $0.003229 | 27.0 s |

All usage and cost values come from OpenRouter responses. The recall suite contains 26 cases
with 100 distractor predicates; extraction contains 15 exact mutation and safety cases. The
model is interpretation infrastructure only: accepted queries still execute through the local
deterministic engine, and guided document proofs use zero model calls, tokens, and provider cost.
