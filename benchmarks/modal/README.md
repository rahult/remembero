# Training the query + extraction LoRA on Modal

A self-managed replacement for the Tinker recipe: about a dollar per run on an H100 or
~$2.50 on an A10G (see the observations at the end), with the evaluation endpoint served
from the same volume. Cost model and alternatives: `docs/research/FINETUNE-PROVIDER-MATRIX.md`.

## One-time setup

```bash
VIRTUAL_ENV=.venv uv pip install modal      # already done
.venv/bin/modal setup                         # opens a browser login; writes ~/.modal.toml
# bearer key for the serve endpoint: one random value, kept in a Modal secret and in .env
.venv/bin/modal secret create rembero-vllm VLLM_API_KEY=<random>
echo 'MODAL_SERVE_API_KEY=<the same value>' >> .env
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
(alias `finetune/<base>-<run>-modal`, which is what names the result files) on an L4,
OpenAI-compatible, scaling to zero after five idle minutes. Every request must carry
`Authorization: Bearer $MODAL_SERVE_API_KEY`; anything else gets 401. The existing harness
runs unchanged:

```bash
set -a; . ./.env; set +a
OLLAMA_URL=https://<app>.modal.run CHAT_API_KEY=$MODAL_SERVE_API_KEY node dist/evals/run-agent-boundary.js \
  --chat-api openai --model finetune/qwen3.5-4b-r11-modal --conditions remembero-closure --seeds 7
node dist/evals/run-extraction-bench.js --base-url https://<app>.modal.run/v1 \
  --api-key $MODAL_SERVE_API_KEY --models finetune/qwen3.5-4b-r11-modal --vocabulary closed
```

A redeploy does not evict a container that is still receiving requests; if the old version
keeps answering, stop it with `modal container list` and `modal container stop -y <id>`.

## Differences from the Tinker recipe

- Merged weights are served rather than a LoRA adapter, because vLLM's LoRA support for
  Qwen3.5's hybrid architecture is unverified; merging costs one extra model load per run.
- Held-out loss is TRL's `eval_loss` over `heldout.jsonl` (whole unseen worlds), the same
  set the Tinker recipe scores as `heldout/nll`.
- Chat rendering is the base model's own chat template through TRL, not the cookbook's
  renderer; assistant-only loss is TRL's prompt/completion masking.
- First run pulls the base model into the volume's `hf/` cache (a few minutes, once).

## What the first runs taught (2026-09-09)

- **Account gating.** A fresh Starter workspace runs A10G functions without a card but
  refuses H100 ("Please add a payment method"), and after ~45 minutes of A10G time the
  workspace was disabled mid-run (`ConflictError: workspace ... is disabled`). Add a payment
  method on modal.com before relying on it; the monthly credit still applies.
- **Kernels.** `Qwen3.5-4B` is 24/32 Gated DeltaNet layers. `transformers>=5` loads it
  fine but without `flash-linear-attention` runs a pure-torch fallback at ~60 s/step on an
  A10G; with the kernels it is ~22–30 s/step (about 2.5 h per epoch, ~$2.50). The log's
  `flash-linear-attention: available` line confirms the fast path. An H100 should be well
  under 30 minutes.
- **Resume.** The adapter is checkpointed to the Volume every 25 steps (committed on save)
  and a rerun of the same `--run` resumes from the last checkpoint, so a preemption or a
  killed container costs at most 25 steps.
- **Loss mask.** TRL warns once per example that the tokenized prompt is not a prefix of
  prompt+completion. The Qwen3.5 template ends the generation prompt with `<think>\n`, and
  that newline merges with the completion's first newline into one token; the mask boundary
  is off by that one token, which is harmless.
- **Serving.** The template opens a `<think>` block, so vLLM runs `--reasoning-parser qwen3`
  and the harness receives only the text after `</think>` as `content`.
- Fallback base if anything else fails: `BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507`.
