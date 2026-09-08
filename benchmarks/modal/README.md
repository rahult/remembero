# Training the query + extraction LoRA on Modal

A self-managed replacement for the Tinker recipe at about a dollar per run on an H100
(free within Modal's monthly starter credit), with the evaluation endpoint served from the
same volume. Cost model and alternatives: `docs/research/FINETUNE-PROVIDER-MATRIX.md`.

## One-time setup

```bash
VIRTUAL_ENV=.venv uv pip install modal      # already done
.venv/bin/modal setup                         # opens a browser login; writes ~/.modal.toml
```

## Check the data locally (no credentials, no GPU)

```bash
.venv/bin/python benchmarks/modal/train_lora.py --check data/training/conversations.jsonl data/training/heldout.jsonl
```

## Train

```bash
.venv/bin/modal run benchmarks/modal/train_lora.py --run r11 \
  --data data/training/conversations.jsonl --heldout data/training/heldout.jsonl
```

Uploads both files to the `rembero-finetune` volume under `data/r11/`, trains a rank-32 LoRA
on `Qwen/Qwen3.5-4B` (TRL `SFTTrainer`, loss on the assistant turn only, batch 8 × 8 grad
accumulation, lr 2e-4 linear, one epoch, bf16), evaluates held-out loss on `heldout.jsonl`,
saves the adapter and merged weights under `runs/r11/`, and prints `metrics.json`. Knobs:
`--epochs`, `--lr`, `--lora-rank`; env `MODAL_TRAIN_GPU` (default `H100`; `A100-80GB` is
cheaper and roughly three times slower), `BASE_MODEL`.

## Serve for evaluation

```bash
.venv/bin/modal deploy benchmarks/modal/train_lora.py     # prints https://...modal.run
```

vLLM serves the merged weights of the latest run (or `SERVE_RUN=r11`) as model `dialect`
on an L4, OpenAI-compatible, scaling to zero after five idle minutes. The existing harness
runs unchanged:

```bash
OLLAMA_URL=https://<app>.modal.run node dist/evals/run-agent-boundary.js \
  --chat-api openai --model dialect --conditions remembero-closure --seeds 7
node dist/evals/run-extraction-bench.js --base-url https://<app>.modal.run/v1 \
  --api-key x --models dialect --vocabulary closed
```

## Differences from the Tinker recipe

- Merged weights are served rather than a LoRA adapter, because vLLM's LoRA support for
  Qwen3.5's hybrid architecture is unverified; merging costs one extra model load per run.
- Held-out loss is TRL's `eval_loss` over `heldout.jsonl` (whole unseen worlds), the same
  set the Tinker recipe scores as `heldout/nll`.
- Chat rendering is the base model's own chat template through TRL, not the cookbook's
  renderer; assistant-only loss is TRL's prompt/completion masking.
- First run pulls the base model into the volume's `hf/` cache (a few minutes, once).

## Not yet verified

The first real run is the test of `transformers>=5` and `vllm>=0.11` support for
`Qwen3.5-4B`; if either fails, `BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507` is the fallback
the provider matrix confirms both libraries support.
