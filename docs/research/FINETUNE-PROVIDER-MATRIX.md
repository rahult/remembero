# Fine-tuning provider matrix: cheaper alternatives to Tinker for a 4B LoRA

Status: research note · 2026-09-09 · prices read from primary sources on 2026-09-09 unless
stated; every wall-time and cost figure that is not a quoted list price is labelled
**estimate** with its assumption.
Evidence: URLs in [Sources](#evidence--sources); Vast.ai marketplace snapshot taken live via
the public `bundles` API on 2026-09-09; local hardware read with `sysctl` /
`system_profiler` on 2026-09-09.
Context: [query-dialect fine-tune findings](QUERY-DIALECT-FINETUNE.md),
[benchmarks/tinker/README.md](../../benchmarks/tinker/README.md) (the eval harness already
speaks to any OpenAI-compatible `/v1/chat/completions` endpoint via `--chat-api openai`).

## The workload being priced

| item                     | value                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| task                     | supervised LoRA, rank 32, one epoch                                                                                        |
| model                    | `Qwen/Qwen3.5-4B` preferred; also Llama-3.2-3B, Qwen3-4B-Instruct-2507, Llama-3.2-1B                                       |
| data                     | ~20,000 chat examples = **6,762,939** training tokens/epoch (Tinker `elapsed_tokens`, one run); ~338 tok/example           |
| Tinker wall time         | ~10 min                                                                                                                    |
| cadence                  | several runs per day                                                                                                       |
| evaluation               | 250 + 200 = **450** chat completions, ~700 prompt + ~50 completion tokens each = 315,000 prompt / 22,500 completion tokens |
| local machine (measured) | **Apple M4 Mac mini, 10-core CPU, 10-core GPU, 16 GB unified memory, 120 GB/s** — not the 64 GB assumed in the brief       |

## Summary

Tinker bills per token and today charges **$0.737 per million train tokens for Qwen3.5-4B**, so
one run is **≈ $4.98** and one 450-completion evaluation through its sampling API is **≈ $0.13**;
it is the only managed option whose turnaround (~10 min) we have measured. The two managed
competitors that still sell self-serve fine-tuning are only marginally cheaper: **Together**
lists Qwen3.5-4B for LoRA SFT at $0.48/M tokens but imposes a **$4.00 per-job minimum**, so our
run costs exactly $4.00 (a 20 % saving) and it has **discontinued serverless LoRA inference**, so
evaluating costs a dedicated H100 endpoint at $3.99/h or a local serve; **Fireworks** charges
$0.50/M ($3.38/run) but **does not list Qwen3.5-4B** for managed SFT (it lists Qwen3-4B-Instruct-2507
and Llama-3.2-3B-Instruct) and serves LoRAs only on on-demand GPUs starting at $8/h. **Predibase**
(acquired by Rubrik) and **OpenPipe** (folded into CoreWeave/W&B) no longer publish self-serve
pricing and are out. The real saving — 5–15× — comes from **self-managed Unsloth/TRL on rented
GPUs**: a FLOPs-based estimate (30 % MFU, checked against Tinker's own 10-minute run and an
independent multi-GPU A100 benchmark) puts the run at ~9 min on an H100, ~29 min on an A100 and
~55 min on an RTX 4090, i.e. **$0.30–$1.40 per run** on Modal, RunPod, Lambda, HF Jobs or Vast.ai,
with the same GPU serving the eval via vLLM for another ~$0.05–$0.35. **Modal's $30/month free
credit covers roughly 20–30 such runs at zero cost.** Local MLX on this 16 GB M4 is feasible only
as 4-bit QLoRA at batch 1 and would take an estimated **6–13 hours per run**, so it is an
overnight tool and a free eval server, not a daily iteration loop.

### Recommendation

- **Cheapest viable:** Unsloth (or TRL) LoRA script on a rented single GPU. Vast.ai RTX 4090
  (median $0.375/h live today) ≈ **$0.40/run**; Modal H100 ≈ **$0.92/run** and free within the
  $30/month Starter credit. Both are estimates from the throughput model below.
- **Fastest turnaround:** Tinker (~10 min, measured) or an H100 on Modal/RunPod (~14 min including
  container start, estimate). Together's and Fireworks' queue/wall times are not published.
- **Best for repeated daily iteration:** write the training job once as a Modal app (a train
  function plus a vLLM serve function, scale-to-zero). Per-run cost ≈ $1 on H100 or ≈ $1.40 on
  A100-80GB, ~15–35 min turnaround, and the eval endpoint is OpenAI-compatible so the existing
  harness works unchanged. If the engineering time is not available, staying on Tinker is
  defensible: the only drop-in managed alternative (Together) saves ~$1/run and makes evaluation
  harder. Note also that on Tinker **Qwen3-8B trains cheaper than Qwen3.5-4B** ($0.44 vs
  $0.737/M → $2.98/run).

## Matrix

Costs are for our 6.76M-token run and one 450-completion evaluation. "est." = estimate; see
per-provider sections for the arithmetic and assumptions.

| provider / option                          | Qwen3.5-4B supported?                                                | setup effort                                            | wall time per run                               | cost per run                                                | cost to serve one eval                                    | notes / caveats                                                                                                    |
| ------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Tinker** (Thinking Machines)             | Yes (`Qwen/Qwen3.5-4B`)                                              | low — already in use                                    | ~10 min (measured)                              | **$4.98** (6.763M × $0.737/M)                               | **$0.13** (prefill $0.104 + sample $0.023)                | Llama-3.2-1B/3B and Qwen3-4B-Instruct-2507 listed as **retired 2026-06-12**. Qwen3-8B is cheaper ($2.98/run).      |
| **Together AI**                            | Yes (`Qwen/Qwen3.5-4B`, LoRA, 131k ctx)                              | low–med — new SDK, dataset upload                       | not published                                   | **$4.00** ($3.25 raised to $4 job minimum)                  | ~$0.67 est. (dedicated H100 $3.99/h × 10 min) or $0 local | Serverless LoRA inference **discontinued**; adapters need a dedicated endpoint (H100/B200 only) or download.       |
| **Fireworks AI**                           | **No** — Qwen3-4B-Instruct-2507, Llama-3.2-3B-Instruct, Qwen3-8B yes | med — `firectl`, on-demand deployment for eval          | not published                                   | **$3.38** (6.763M × $0.50/M, no minimum stated)             | ~$1.33 est. (on-demand H100 $8/h × 10 min)                | LoRAs deployable **only** on on-demand GPUs; cheapest listed GPU is H100 at $8/h from 2026-09-01.                  |
| **Predibase** (Rubrik)                     | unknown — docs redirect to rubrik.com                                | high — "upgrade to Enterprise by contacting sales"      | unknown                                         | ~$3.38 est. from third-party rate card                      | ~$0.36 est. (L4 $2.14/h × 10 min, third-party)            | **No primary pricing source exists any more.** Not viable for self-serve.                                          |
| **Hugging Face Jobs** (TRL/Unsloth script) | Yes (any HF model; you bring the script)                             | med — uv script, 30-min default timeout, adapter upload | ~34 min A100 / ~80 min L4 est.                  | $1.42 A100-large · $1.07 L4 · $1.17 H200 est.               | $0.21 A100 · $0.07 L4 est. (vLLM in a second Job)         | AutoTrain is **no longer maintained**. Per-second billing, no minimum. Inference Endpoints L4 $0.80/h alternative. |
| **OpenPipe**                               | n/a                                                                  | n/a                                                     | n/a                                             | n/a                                                         | n/a                                                       | Site states the platform **"has migrated to Weights & Biases and CoreWeave"**; legacy training stopped 2026-07-30. |
| **Self-managed: Modal**                    | Yes                                                                  | med — ~150-line Modal app, then one command             | ~14 min H100 / ~34 min A100 est.                | $0.92 H100 · $1.42 A100-80GB est.                           | $0.33 H100 · $0.21 A100 est. (5 min vLLM)                 | **$30/month free credit** on Starter ≈ 20–30 runs free. Per-second billing, scale-to-zero serving.                 |
| **Self-managed: RunPod**                   | Yes                                                                  | med–high — pod lifecycle, volumes                       | ~14 min H100 / ~34 min A100 / ~60 min 4090 est. | $0.81 H100 SXM · $0.90 A100 · $0.74 RTX 4090 (Secure) est.  | ~$0.13 A100 est.                                          | Community Cloud ~20–50 % cheaper (4090 $0.34/h). No A10 listed.                                                    |
| **Self-managed: Lambda**                   | Yes                                                                  | med–high — VM, availability varies                      | ~14 min H100 / ~34 min A100-40GB est.           | $1.00 H100 SXM · $1.13 A100-40GB est.                       | ~$0.17 est.                                               | Per-minute billing, no egress fees. Only 40 GB A100 as 1×.                                                         |
| **Self-managed: Vast.ai**                  | Yes                                                                  | high — marketplace hosts, variable reliability          | ~60 min 4090 / ~34 min A100 est.                | **$0.38 RTX 4090 · $0.53 A100 SXM4 · $0.34–0.87 H100** est. | ~$0.03 4090 est.                                          | Live medians of the 30 cheapest verified 1-GPU on-demand offers on 2026-09-09; prices move hourly.                 |
| **Local MLX** (M4, 16 GB)                  | Yes — `mlx_lm/models/qwen3_5.py` exists                              | low–med — `pip install mlx-lm`, convert data, 1 command | **6–13 h** est. (4-bit QLoRA, batch 1)          | ≈ $0 (electricity ≈ $0.15)                                  | $0, ~25–30 min est. via `mlx_lm.server`                   | Fits only as QLoRA (4-bit weights), batch 1–2, grad-checkpoint. Different numerics from Tinker's bf16 LoRA.        |

## Throughput model used for the self-managed rows

No provider publishes an absolute tokens/sec figure for a 4B LoRA run, and Unsloth's README only
gives relative speedups ("Qwen3.5 (4B): 1.5x faster, 60% less VRAM"). I therefore estimated wall
time from arithmetic throughput, then sanity-checked it two ways.

**Assumptions (all estimates):**

- Training FLOPs per token for LoRA with gradient checkpointing ≈ 6 × N (forward 2N, backward
  activation-gradient 2N, recompute forward 2N; frozen base weights need no weight-gradient pass;
  LoRA parameters are negligible). N = 4.0 × 10⁹. Total = 6 × 4.0e9 × 6.763e6 = **1.62 × 10¹⁷ FLOPs**.
- Model FLOP utilisation (MFU) = **30 %** of the NVIDIA dense (non-sparse) BF16 tensor-core figure.
- Overheads added: **+5 min** per run for container start, weight download and dataset upload;
  **+5 min** for the eval (vLLM load + 450 requests; at this size vLLM completes 450 × 750 tokens in
  well under a minute once loaded).

| GPU           | dense BF16 TFLOPS (NVIDIA spec)                | effective @30 % | train time est. | ≈ tokens/s | billable min (train + 5) |
| ------------- | ---------------------------------------------- | --------------- | --------------- | ---------- | ------------------------ |
| H100 SXM      | 989.5 (1,979 with sparsity)                    | 297             | 9.1 min         | 12,400     | 14                       |
| H100 PCIe/NVL | 835.5 (1,671 with sparsity)                    | 251             | 10.8 min        | 10,400     | 16–17                    |
| L40S          | 362                                            | 109             | 24.9 min        | 4,500      | 30                       |
| A100 80/40GB  | 312                                            | 93.6            | 28.9 min        | 3,900      | 34                       |
| RTX 4090      | 165 (Ada whitepaper; not fetched this session) | 49.5            | 54.7 min        | 2,060      | 60                       |
| A10 (A10G)    | 125 (A10G is a lower-clocked variant)          | 37.5            | 72 min          | 1,560      | ~80                      |
| L4            | 121 (242 with sparsity)                        | 36.3            | 74.5 min        | 1,510      | 80                       |

**Sanity checks.** (1) Tinker completes the same 6.76M tokens in ~10 minutes, which matches the
single-H100-class estimate. (2) VESSL's independent LoRA benchmark measured 4,746 tokens/s on
8 × A100 for a 31B model (rank 16, seq 2,048, bf16) = 593 tokens/s per GPU; scaling by parameter
count (31/4) gives ~4,600 tokens/s for a 4B model, within 20 % of the 3,900 above. (3) Unsloth's
requirements page lists 8 GB VRAM for 16-bit LoRA on a 3B model, so bf16 LoRA on a 4B fits every
GPU in the table including 24 GB cards; QLoRA is not required.

## Per-provider detail

### 1. Tinker (Thinking Machines)

- **Pricing model:** per million tokens, four meters: _prefill_, _cached prefill_ (80 % off),
  _sample_, _train_ ("Forward and backward pass for gradient computation"). "All prices are per
  million tokens. Checkpoint storage is charged at $0.10 per GB per month."
- **Prices read 2026-09-09** from <https://tinker-docs.thinkingmachines.ai/tinker/models/>
  (machine-readable copy: `/tinker/models.json`):

  | model                  | prefill | cached | sample | train  |
  | ---------------------- | ------- | ------ | ------ | ------ |
  | `Qwen/Qwen3.5-4B`      | $0.33   | $0.066 | $1.005 | $0.737 |
  | `Qwen/Qwen3.5-9B`      | $0.66   | $0.132 | $1.995 | $1.463 |
  | `Qwen/Qwen3-8B`        | $0.195  | $0.039 | $0.60  | $0.44  |
  | `Qwen/Qwen3.6-35B-A3B` | $0.54   | $0.108 | $1.335 | $1.177 |

  The page lists **Qwen3-4B-Instruct-2507, Llama-3.2-3B, Llama-3.2-1B, Llama-3.1-8B(-Instruct),
  Qwen3.5-27B, Qwen3.5-35B-A3B** under "Retired Models — June 12, 2026" ("can no longer be used
  for training or inference"). Our earlier runs used Llama-3.2-3B; see "could not verify".

- **Our run:** 6,762,939 train tokens × $0.737/M = **$4.984**. Assumption: Tinker's billed train
  tokens equal the `elapsed_tokens` reported by the run (one forward-backward per token).
  With Qwen3-8B: 6.763M × $0.44 = **$2.98**.
- **Eval via Tinker sampling (the current setup, through the cookbook's OpenAI-compatible
  proxy):** 315,000 prompt tokens × $0.33/M = $0.104 + 22,500 sample tokens × $1.005/M =
  $0.023 → **$0.127**. Prompt caching (80 % off) would cut the prefill part if the system prompt
  is shared.
- **Storage:** a rank-32 LoRA for a 4B model is on the order of 0.1–0.3 GB → ≈ $0.01–0.03/month.
- Serverless inference is "currently in beta and available for Inkling and Inkling-Small only";
  LoRA sampling uses the per-token sample/prefill meters above.
- Free credits / minimums: none stated on the models page. (A web summary claimed $150 free
  credits for new users and a 2026-07-17 price rise; not found on a primary page — unverified.)

### 2. Together AI

- **Supported models** (read 2026-09-09, <https://docs.together.ai/docs/fine-tuning-models>):
  `Qwen/Qwen3.5-4B` — 131,072 ctx (SFT), LoRA supported; also Qwen3.5-2B/0.8B/9B/27B/35B-A3B/
  122B-A10B/397B-A17B, Qwen3.6-27B, Qwen3.8-27B. **Llama-3.2 (1B/3B) is not listed** (Llama 3.1/
  3.3/4 are).
- **Prices** (read 2026-09-09, <https://www.together.ai/pricing>): "Up to 16B — Supervised
  Fine-Tuning (LoRA) $0.48 / 1M tokens; DPO LoRA $0.54; full fine-tuning $1.35." "Each job is
  subject to a minimum charge of $4.00." Docs
  (<https://docs.together.ai/docs/fine-tuning-pricing>): "total_tokens = (n_epochs ×
  tokens_per_training_dataset) + (n_evals × tokens_per_validation_dataset)"; "Fine-tuning jobs
  have a $4.00 minimum charge. Some models are exempt."; with packing disabled tokens are
  `dataset_length × max_seq_length` (keep packing on).
- **Our run:** 6.763M × $0.48 = $3.246 → **billed $4.00** (minimum). Wall time: not published.
- **Serving the LoRA:** <https://docs.together.ai/docs/lora-inference> — "Serverless LoRA
  inference (including Serverless Multi-LoRA) has been discontinued." Adapters go on a dedicated
  endpoint, billed "per minute per running replica, even when idle", scale-to-zero supported,
  provisioning/cold-start time not billed. Hardware list
  (<https://docs.together.ai/docs/dedicated-endpoints/pricing>): `1xnvidia-h100-80gb` $3.99/h
  (pricing page shows "$5.49 → $3.99, promotional until 09/30/26"), `1xnvidia-b200-180gb` $8.99/h,
  H200/B300/GB300 "contact sales". No cheaper GPU is offered. **Eval estimate:** 10 min ready time
  × $3.99/h = **$0.67** (15 min → $1.00). Alternative: "download it for local inference" (merged
  weights) → serve on the Mac with mlx-lm for $0.
- Serverless prices for Qwen3.5-4B / Qwen3-4B / Llama-3.2 base models are **not listed** on the
  pricing page (only e.g. Qwen2.5 7B Instruct Turbo $0.30/$0.30).

### 3. Fireworks AI

- **Supported models** (read 2026-09-09, <https://docs.fireworks.ai/fine-tuning/models>):
  managed SFT LoRA = true for Qwen 3 4B Instruct 2507, Qwen 3 8B, Qwen 3 0.6B/14B/32B, Qwen 3.5
  27B, Llama 3.2 3B Instruct, Llama 3.1 8B Instruct, Qwen 2.5 32B. **Qwen3.5-4B, Qwen3-4B (base),
  Llama-3.2-1B are not listed.** LoRA rank "must be a power of 2 up to 32" (default 8); default 1 epoch.
- **Prices** (read 2026-09-09, <https://fireworks.ai/pricing>): "Models up to 16B parameters —
  LoRA SFT $0.50 / LoRA DPO $1.00 / Full SFT $1.00 per 1M training tokens." "Training tokens can
  be estimated with number of tokens in training dataset × number of epochs." No minimum charge
  is stated. "Serve fine-tuned models for the same price as base models." New accounts: "$1 in
  free credits."
- **Our run** (Qwen3-4B-Instruct-2507 or Llama-3.2-3B-Instruct, since Qwen3.5-4B is unavailable):
  6.763M × $0.50 = **$3.38**. Wall time: not published.
- **Serving the LoRA:** <https://docs.fireworks.ai/fine-tuning/deploying-loras> — "Trained LoRA
  models … can only be deployed to on-demand (dedicated) deployments. Serverless deployment is
  not supported for LoRA models." On-demand prices (pricing page): "Pay per GPU second, with no
  extra charges for start-up times"; H100 80 GB **$8.00/h from Sep 1** (was $7.00), H200 $8.00,
  B200 $13.00; no A100/L4 tier listed. **Eval estimate:** 10 min × $8/h = **$1.33**.
- Fireworks also lists a "Serverless Training API" with Tinker-style prefill/sample/train meters
  (e.g. Qwen 3.8 27B train $4.103/M — identical to Tinker's Qwen3.8-27B price) but only for four
  large launch models; no 4B model.
- Serverless base-model rates for reference (<https://docs.fireworks.ai/serverless/pricing>):
  "Under 4B $0.10; 4B–16B $0.20 per 1M tokens" — not applicable to LoRAs.

### 4. Predibase (Rubrik)

- `predibase.com/pricing` and `docs.predibase.com/*` **301-redirect to rubrik.com** (checked
  2026-09-09). Rubrik acquired Predibase in June 2025 (CNBC, TechTarget).
- The only rate card found is third-party (<https://www.usagepricing.com/blueprint/predibase>,
  "reconstructed from docs as of August 2026"): SFT LoRA ≤16B $0.50/M, Turbo LoRA $1.00/M;
  dedicated serving L4 $2.14/h, A10G $2.60/h, L40S $3.20/h, A100 $4.80/h; $25/30-day trial; and
  the quote "To add credits to your account, you must upgrade to Enterprise Tier by contacting
  our sales team."
- **If those numbers held:** run $3.38 est., eval on L4 10 min ≈ $0.36 est. Treat as
  **unverified**; the product is not self-serve.

### 5. Hugging Face

- **AutoTrain:** <https://github.com/huggingface/autotrain-advanced> README: "This project is no
  longer maintained. No new features will be added and bugs will not be fixed." (recommends
  Axolotl, TRL, `transformers.Trainer`). Software was free; you paid for Space hardware.
- **HF Jobs** (read 2026-09-09, <https://huggingface.co/docs/huggingface_hub/guides/jobs>):
  "pay-as-you-go: you only pay for the seconds you use"; "available to any user or organization
  with a positive credit balance"; **default timeout 30 minutes** — set `timeout="2h"`.
  Flavors/prices: `t4-small` $0.40/h · `l4x1` $0.80/h · `a10g-small` $1.00/h · `a10g-large`
  $1.50/h · `l40sx1` $1.80/h · `a100-large` (80 GB) $2.50/h · `rtx-pro-6000` (96 GB) $2.75/h ·
  `h200` $5.00/h. Same numbers on <https://huggingface.co/pricing>. The docs show exactly our
  pattern: `hf jobs uv run --flavor a10g-small .../trl/scripts/sft.py`.
- **Our run (estimates, throughput model above, +5 min overhead):**
  `a100-large` 34 min → **$1.42**; `l4x1` 80 min → **$1.07**; `a10g-small` 80 min → **$1.33**;
  `h200` (H100-class compute) 14 min → **$1.17**; `l40sx1` 30 min → **$0.90**.
- **Eval:** run vLLM in a second Job with `--image vllm/vllm-openai:latest` (the docs give this
  example) and point the harness at it over the Job's SSH tunnel, or push the adapter to the Hub
  and use **Inference Endpoints** (<https://huggingface.co/docs/inference-endpoints/pricing>:
  L4 $0.80/h, A10G $1.00/h, A100 $2.50/h; "actual cost is calculated by the minute"; no charge
  while scaled to zero; charged while initializing and running). Estimate 5 min A100 → $0.21;
  L4 → $0.07; an Endpoint incl. ~5–10 min initialisation on L4 → $0.10–0.20.

### 6. OpenPipe

- <https://openpipe.ai/> (fetched 2026-09-09): the page's meta description reads "The OpenPipe
  platform has migrated to Weights & Biases and CoreWeave. Browse migration details and the
  OpenPipe technical archive." `docs.openpipe.ai` no longer resolves (`ENOTFOUND`). Search
  summary of the blog: legacy training/inference ended **2026-07-30**. No pricing exists to
  quote. **Not an option.**

### 7. Self-managed Unsloth / Axolotl / TRL on rented GPUs

Prices read 2026-09-09. Run costs use the throughput table (train + 5 min); eval = 5 min vLLM
on the same GPU. All run/eval costs are **estimates**.

| provider | GPU (1×)                   | list price                                                     | run cost      | eval cost     | source                                      |
| -------- | -------------------------- | -------------------------------------------------------------- | ------------- | ------------- | ------------------------------------------- |
| Modal    | H100 SXM5                  | $0.001097/s ($3.95/h)                                          | $0.92         | $0.33         | <https://modal.com/pricing>                 |
| Modal    | A100 80 GB                 | $0.000694/s ($2.50/h)                                          | $1.42         | $0.21         | same                                        |
| Modal    | A100 40 GB                 | $0.000583/s ($2.10/h)                                          | $1.19         | $0.17         | same                                        |
| Modal    | L40S                       | $0.000542/s ($1.95/h)                                          | $0.98         | $0.16         | same                                        |
| Modal    | L4                         | $0.000222/s ($0.80/h)                                          | $1.07         | $0.07         | same                                        |
| Modal    | A10                        | $0.000306/s ($1.10/h)                                          | $1.47         | $0.09         | same                                        |
| RunPod   | H100 SXM Secure/Community  | $3.49 / $2.69 per h                                            | $0.81 / $0.63 | $0.29 / $0.22 | <https://www.runpod.io/pricing>             |
| RunPod   | A100 PCIe Secure/Community | $1.59 / $1.19 per h                                            | $0.90 / $0.67 | $0.13 / $0.10 | same                                        |
| RunPod   | L40S Secure/Community      | $1.09 / $0.79 per h                                            | $0.55 / $0.40 | $0.09 / $0.07 | same                                        |
| RunPod   | RTX 4090 Secure/Community  | $0.74 / $0.34 per h                                            | $0.74 / $0.34 | $0.06 / $0.03 | same                                        |
| RunPod   | L4 Secure/Community        | $0.49 / $0.44 per h                                            | $0.65 / $0.59 | $0.04         | same                                        |
| Lambda   | H100 SXM                   | $4.29/h (pay by the minute)                                    | $1.00         | $0.36         | <https://lambda.ai/service/gpu-cloud>       |
| Lambda   | H100 PCIe                  | $3.29/h                                                        | $0.93         | $0.27         | same                                        |
| Lambda   | A100 40 GB (PCIe or SXM)   | $1.99/h                                                        | $1.13         | $0.17         | same                                        |
| Lambda   | A10                        | $1.29/h                                                        | $1.72         | $0.11         | same                                        |
| Vast.ai  | RTX 4090                   | live: min $0.274 / **median $0.375** / max $0.696 per h (n=30) | **$0.38**     | $0.03         | `console.vast.ai/api/v0/bundles` 2026-09-09 |
| Vast.ai  | A100 SXM4                  | live: min $0.631 / median $0.936 / max $1.128 (n=10)           | $0.53         | $0.08         | same                                        |
| Vast.ai  | H100 SXM                   | live: min $1.469 / median $3.707 / max $7.356 (n=8)            | $0.34–0.87    | $0.12–0.31    | same                                        |
| Vast.ai  | L40S                       | live: min $0.468 / median $0.801 / max $0.869 (n=3)            | $0.40         | $0.07         | same                                        |

Notes:

- Modal Starter plan: "$30 / month free credits" → ~30 H100 runs or ~20 A100 runs per month at
  $0. CPU $0.0000131/core/s and memory $0.00000222/GiB/s add a few cents. Modal is also the
  easiest place to keep a scale-to-zero vLLM endpoint for the eval.
- Observed 2026-09-09: on a fresh Starter account Modal refuses H100 functions until a payment
  method is added ("Please add a payment method to use H100 GPU functions"), while A10G runs
  without one. So the free credit is usable card-free only on the smaller GPUs.
- Observed 2026-09-09: Qwen3.5-4B is 24/32 Gated DeltaNet layers. Without the
  `flash-linear-attention` Triton kernels, transformers 5 runs a pure-torch fallback that trained
  at ~60 s/step on an A10G (5+ hours for one epoch, ~$6). The estimates in this table assume
  the fast kernels are installed; the Modal app now installs them.
- RunPod lists no A10; storage $0.10/GB/mo container, $0.07/GB/mo network volume. Vast.ai prices
  are set by hosts ("Prices are set by the market, not by Vast"), per-second billing; the query
  filtered to `verified` hosts, `on-demand`, `num_gpus = 1`, sorted by price, 30-offer cap — the
  medians above are of the cheapest 30, not of the whole market.
- Unsloth (<https://github.com/unslothai/unsloth>) lists a Qwen3.5 (4B) notebook ("1.5x faster,
  60% less" VRAM) and Llama 3.2 notebooks; its requirements page says 16-bit LoRA on a 3B needs
  ~8 GB VRAM, so a 4B in bf16 fits any 24 GB card. Unsloth also claims macOS/Apple Silicon
  training support ("Training, MLX and GGUF inference are ALL supported") — not benchmarked here.
- vLLM serving: with `--enable-lora` the base model plus adapter loads in ~1–2 min on any of
  these GPUs; the 450-request eval is seconds of GPU time. The 5-minute eval allowance is
  conservative.

### 8. Local Apple Silicon with MLX (measured machine: M4, 16 GB)

- **Hardware (measured 2026-09-09):** `hw.memsize` = 17,179,869,184 (16 GB); `Apple M4`; Mac mini.
  Apple's spec sheet (<https://support.apple.com/en-us/121555>): "10-core CPU … 10-core GPU …
  120GB/s memory bandwidth", 16 GB base, configurable to 24/32 GB (64 GB requires an M4 Pro/Max
  machine).
- **Software support:** mlx-lm ships `mlx_lm/models/qwen3_5.py` and `qwen3_5_moe.py` (checked via
  GitHub API 2026-09-09) and a generic LoRA tuner (`mlx_lm/tuner/lora.py`). The LORA.md
  supported-family list (Mistral, Llama, Phi2, Mixtral, Qwen2, Gemma, OLMo, MiniCPM, InternLM2)
  is stale — it predates Qwen3 — but the tuner attaches adapters to any linear layer, and a
  community repo has trained Qwen3.5-4B with it (below).
- **Does it fit in 16 GB?** Estimate: **yes, as QLoRA only.** Qwen3.5-4B weights are ~8 GB in
  bf16 and ~2.5 GB at 4 bits. Our examples average 338 tokens, so activations are small.
  mlx-lm's own memory guidance: use a quantized model (QLoRA), `--batch-size 1` or `2` with
  `--grad-accumulation-steps`, `--num-layers` 8–16, `--grad-checkpoint`. A community measurement
  (<https://github.com/sciences44/mlx-lora-finetune>, M1 64 GB, rank 8, 16 layers, 600 iters):
  Qwen3.5-4B peak RAM **11.1 GB** at their default batch 4; the same project runs 16 GB Macs at
  batch 1. With macOS itself using 3–5 GB, a 4-bit model at batch 1–2 leaves headroom; bf16 LoRA
  at rank 32 would not fit comfortably.
- **Throughput (all from primary/near-primary sources, none on an M4 base):**
  - mlx-lm LORA.md: Mistral-7B LoRA, batch 1, 4 layers, "on an M1 Max with 32 GB runs at about
    250 tokens-per-second".
  - sciences44 (community, M1 64 GB): Qwen3.5-4B "~115 tokens/sec", Qwen3.5-2B ~180, 0.8B ~475.
  - **Estimate for this M4 (10-core GPU, 120 GB/s vs the M1's 8-core, 68 GB/s):** 150–300
    tokens/s for Qwen3.5-4B QLoRA at batch 1–2.
- **Wall time for 6,762,939 tokens (estimate):** 6.76M ÷ 300 = 6.3 h; ÷ 150 = 12.5 h; at the
  measured M1 rate (115) 16.3 h. **6–13 hours per epoch** — one run overnight, not several per day.
- **Cost:** electricity only. Mac mini M4 draws well under 65 W continuous; 10 h × 0.065 kW ≈
  0.65 kWh ≈ $0.10–0.25 at typical tariffs → **≈ $0**.
- **Serving for eval:** `mlx_lm.server --model <path>` exposes an OpenAI-style
  `/v1/chat/completions`; a request may pass `adapters` ("a string path to low-rank adapters …
  relative to the directory the server was started in"), or fuse first with `mlx_lm.fuse`. The
  docs warn "not recommended for production as it only implements basic security checks" — fine
  for a local eval. Estimate: 4-bit 4B on this machine ≈ 3–4 s per 700+50-token request →
  450 requests ≈ **25–30 min**, $0. This also works as a $0 eval server for adapters trained
  elsewhere (Together and Tinker both allow checkpoint download), subject to converting the
  PEFT adapter to mlx-lm's adapter format or fusing into HF weights and running `mlx_lm.convert -q`.
- **What a 64 GB machine would change (estimate):** memory stops being the constraint — bf16
  LoRA at rank 32 on all layers with batch 8–16 becomes possible, matching Tinker's numerics
  instead of 4-bit QLoRA. Speed scales with GPU cores and bandwidth, not RAM: an M4 Pro (20-core
  GPU, 273 GB/s) would be roughly 2× this machine (~3–6 h/run); an M4 Max (40-core GPU,
  546 GB/s) roughly 4× (~1.5–3 h/run). Still 10–20× slower than an H100, so even a 64 GB Mac is
  an overnight/one-run-a-day option rather than a replacement for Tinker's 10 minutes.

## What we could not verify

- **Together and Fireworks wall time/queue time** for a 6.8M-token job — neither publishes it.
- **Tinker's retirement of Llama-3.2-3B/1B and Qwen3-4B-Instruct-2507 (2026-06-12)** versus the
  repo's Llama-3.2-3B runs dated 2026-09-08; either the retirement list applies only to new
  training clients, or the runs predate enforcement. Check `tinker://` availability before
  relying on those models again. Tinker free credits ($150) and the 2026-07-17 price change came
  only from a search summary, not a primary page.
- **Predibase prices** — only a third-party reconstruction exists; every primary URL redirects
  to Rubrik.
- **Absolute tokens/sec for Unsloth on a 4B model** — Unsloth publishes only relative speedups;
  the self-managed wall times are a 30 % MFU FLOPs model with two independent cross-checks, not
  measurements. Real MFU on small batches of short (338-token) examples may be lower; packing helps.
- **RTX 4090 dense BF16 figure (165 TFLOPS)** is from memory of NVIDIA's Ada whitepaper and was
  not fetched this session; A10G (AWS variant) clocks lower than the A10 datasheet figure used.
- **vLLM and mlx-lm support for the Qwen3.5-4B hybrid (Gated-DeltaNet) architecture with LoRA
  adapters** — mlx-lm has the model file and a community LoRA run exists; vLLM LoRA support for
  this architecture was not checked.
- **MLX throughput on an M4 base** — no primary benchmark exists; the 150–300 tok/s range is
  extrapolated from M1/M1 Max measurements.
- **Vast.ai** medians are a 30-offer snapshot of verified hosts at one moment; not a stable price.
- **Fireworks minimum charge** — none stated on the pricing page, but not explicitly denied either.

## Evidence / Sources

All read on 2026-09-09.

- Tinker models & pricing: <https://tinker-docs.thinkingmachines.ai/tinker/models/> (JSON:
  <https://tinker-docs.thinkingmachines.ai/tinker/models.json>); landing page
  <https://thinkingmachines.ai/tinker/> ("All prices are in USD per million tokens").
- Together pricing: <https://www.together.ai/pricing>; fine-tuning models:
  <https://docs.together.ai/docs/fine-tuning-models>; fine-tuning pricing formula:
  <https://docs.together.ai/docs/fine-tuning-pricing>; LoRA inference discontinued:
  <https://docs.together.ai/docs/lora-inference>; dedicated endpoint pricing:
  <https://docs.together.ai/docs/dedicated-endpoints/pricing>; deploying a fine-tuned model:
  <https://docs.together.ai/docs/deploying-a-fine-tuned-model>.
- Fireworks pricing: <https://fireworks.ai/pricing>; fine-tuning models:
  <https://docs.fireworks.ai/fine-tuning/models>; fine-tuning guide:
  <https://docs.fireworks.ai/fine-tuning/fine-tuning-models>; deploying LoRAs:
  <https://docs.fireworks.ai/fine-tuning/deploying-loras>; serverless tiers:
  <https://docs.fireworks.ai/serverless/pricing>.
- Predibase: <https://predibase.com/pricing> and <https://docs.predibase.com/> (both 301 →
  rubrik.com); third-party rate card <https://www.usagepricing.com/blueprint/predibase>;
  acquisition: <https://www.cnbc.com/2025/06/25/rubrik-agrees-to-buy-ai-startup-predibase-for-over-100-million.html>.
- Hugging Face: Jobs guide and hardware table
  <https://huggingface.co/docs/huggingface_hub/guides/jobs>; pricing <https://huggingface.co/pricing>;
  Inference Endpoints pricing <https://huggingface.co/docs/inference-endpoints/pricing>;
  AutoTrain status <https://github.com/huggingface/autotrain-advanced>.
- OpenPipe: <https://openpipe.ai/> (meta description quoted above); <https://openpipe.ai/blog>.
- GPU rentals: <https://modal.com/pricing>; <https://www.runpod.io/pricing>;
  <https://lambda.ai/service/gpu-cloud>; Vast.ai public offers API
  `https://console.vast.ai/api/v0/bundles/?q=…` (rentable, verified, on-demand, 1 GPU, sorted
  by `dph_total`, limit 30) and <https://vast.ai/pricing> (billing terms).
- Unsloth: <https://github.com/unslothai/unsloth>;
  <https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements>.
- NVIDIA spec sheets (dense vs. sparsity figures): A100 <https://www.nvidia.com/en-us/data-center/a100/>;
  H100 <https://www.nvidia.com/en-us/data-center/h100/>; L4 <https://www.nvidia.com/en-us/data-center/l4/>;
  L40S <https://www.nvidia.com/en-us/data-center/l40s/>; A10 <https://www.nvidia.com/en-us/data-center/products/a10-gpu/>.
- Independent LoRA throughput cross-check: VESSL, "A100 vs H100 vs B200 LoRA cost benchmark"
  <https://vessl.ai/en/blog/lora-finetuning-cost-a100-h100-b200>.
- MLX: LoRA guide <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>; server guide
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>; model directory listing via
  <https://api.github.com/repos/ml-explore/mlx-lm/contents/mlx_lm/models>; community Qwen3.5
  LoRA-on-Mac measurements <https://github.com/sciences44/mlx-lora-finetune>; Mac mini (2024)
  spec <https://support.apple.com/en-us/121555>; local `sysctl -n hw.memsize
machdep.cpu.brand_string` and `system_profiler SPHardwareDataType`.
