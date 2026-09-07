# Fine-tuning a small model on the Remembero query dialect

Everything here is manual and costs money. Nothing in `npm test`, `npm run build`, or CI
runs it. The TypeScript side (`npm run train:data`) is free apart from optional Luna
paraphrasing through OpenRouter.

## 1. Generate data

```bash
npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training
# or, with no model calls at all:
npm run train:data -- --no-paraphrase
```

Writes `data/training/conversations.jsonl` (train), `heldout.jsonl` (whole held-out
worlds, never trained on) and `manifest.json` (committed as the run record). Every line is
`{"messages":[system, user, assistant]}`; the assistant message is the gold program in the
flat dialect (see `docs/CLOSURE-PREDICATES.md`). Every example was executed on its world
before export.

## 2. Train

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r benchmarks/tinker/requirements.txt
python benchmarks/tinker/sl_query_dialect.py \
  --data data/training/conversations.jsonl \
  --model Llama-3.2-3B \
  --log-path runs/tinker/llama-3.2-3b-dialect
```

`TINKER_API_KEY` is read from `.env` at the repo root. Supported small models: `Llama-3.2-1B`,
`Llama-3.2-3B`, `Qwen3-4B-Instruct-2507`, `Qwen3.5-4B`. Train `Llama-3.2-3B` first: it is
in the published agent-boundary matrix (multi-hop 0/6, total 10/31) so before/after lands in
the same table. The trainer prints `tinker://<run>/sampler_weights/<name>` checkpoint paths.

## 3. Evaluate with the unchanged harness

The cookbook ships an OpenAI-compatible proxy for any `tinker://` checkpoint:

```bash
python -m tinker_cookbook.capture.proxy.serve --port 7462 \
  --base-model meta-llama/Llama-3.2-3B \
  --model-path tinker://<run>/sampler_weights/final
```

Then point the agent-boundary runner at it:

```bash
npm run build:core
OLLAMA_URL=http://127.0.0.1:7462 node dist/evals/run-agent-boundary.js \
  --chat-api openai --model tinker://<run>/sampler_weights/final \
  --conditions remembero-closure --seeds 7
```

Results land in `docs/research/results/agent-boundary-v2-<model>-remembero-closure-summary.json`,
the same shape as the published matrix. Compare against the pre-training baselines in the
same directory (`...-llama3.2-3b-remembero-closure-summary.json` etc.).

## Guardrails

- The 31 benchmark questions are never in the training data, and training worlds share no
  predicate names with the benchmark schema.
- Held-out worlds (`heldout.jsonl`) are the dialect generalization check; the benchmark is
  the task check.
- Do not commit `conversations.jsonl`; commit `manifest.json`.
