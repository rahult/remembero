# Training a reader on RunPod Serverless

One job trains (or resumes) one reader run on an H100, exports it text-only, converts it to
GGUF and quantizes it, leaving `runs/<run>/<run>-Q8_0.gguf` and `runs/<run>/metrics.json` on a
network volume. The worker is `handler.py`; the recipe it calls is `benchmarks/train/reader_lora.py`,
the same code Modal ran, so a run started there resumes here from its checkpoint.

## What it costs

| Item | Price | Notes |
| --- | --- | --- |
| H100 80GB, flex (serverless) | $4.18–4.79 / h | billed per second of execution, idle workers cost nothing |
| Run A (resume reader v5 from checkpoint-50) | ≈ $2.50 | ≈ 26 min at ~65 s/step with batch 4 × accum 16, plus a few minutes of cold start |
| A fresh 4,700-row run at `--max-length 6912` | ≈ $6 | ≈ 80 min |
| Network volume, 40 GB | ≈ $2.80 / month | charged while it exists — delete it when the run is done |

The 16 GB base model is downloaded once into `HF_HOME=/runpod-volume/hf` and reused by every
later job, so only the first job pays for the download.

## One-time setup

Build and push the image from the repository root (the build is amd64; on an Apple Silicon Mac
it runs under emulation and takes 20–40 minutes):

```sh
docker login
docker build --platform linux/amd64 -f benchmarks/runpod/Dockerfile -t <dockerhub-user>/rembero-reader-train:v1 . && docker push <dockerhub-user>/rembero-reader-train:v1
```

The base image is `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (confirmed on
Docker Hub 2026-09-14; the built image is 10.6 GB). If it is gone, pick the newest
`2.x-py3.11-cuda12.x-cudnn-devel` tag from https://hub.docker.com/r/runpod/pytorch/tags and record
the replacement here.

The image deliberately does not install `llama.cpp/requirements/requirements-convert_hf_to_gguf.txt`:
it pins `transformers==4.57.6`, which cannot read the tokenizer config transformers 5 writes, and
`torch==2.11.0` from the CPU wheel index, which would replace the CUDA torch and leave training on
the CPU. The converter's imports (torch, numpy, transformers 5, sentencepiece, protobuf) are all in
the image already, and it puts the clone's own `gguf-py` on `sys.path`. Check with:

```sh
docker run --rm --platform linux/amd64 rembero-reader-train:local python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

In the RunPod console:

1. **Storage → Network Volume**: create one of **40 GB**; note the datacenter it lands in (e.g. `EU-RO-1`).
2. **Storage → S3 API keys**: create a key; keep the access key and secret.
3. **Serverless → New Endpoint**: GPU **H100 80GB**, **max workers 1**, **container disk 40 GB**,
   the network volume from step 1 attached, the image `<dockerhub-user>/rembero-reader-train:v1`
   from Docker Hub, **execution timeout 10800 s** (3 hours — the 600 s default kills the job).
4. **Settings → API keys**: create an API key.

Then put the six values in `.env` (already gitignored) and export them:

```sh
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_VOLUME_ID=...
export RUNPOD_DATACENTER=EU-RO-1
export RUNPOD_S3_ACCESS_KEY=...
export RUNPOD_S3_SECRET_KEY=...
```

```sh
.venv/bin/pip install runpod boto3
```

## Per run

Upload the training data (and, to resume a run started elsewhere, its checkpoint directory):

```sh
.venv/bin/python benchmarks/runpod/volume.py put data/training-reader-v5 data/reader-v5-gemma4-e4b
.venv/bin/python benchmarks/runpod/volume.py put /Volumes/Atlas/models/rembero/checkpoint-50 runs/reader-v5-gemma4-e4b/trainer/checkpoint-50
```

Submit the job and follow it (prints a line per stage; safe to Ctrl-C, the job keeps running):

```sh
.venv/bin/python benchmarks/runpod/submit.py reader-v5-gemma4-e4b
.venv/bin/python benchmarks/runpod/submit.py reader-v5-gemma4-e4b --max-length 6912 --no-liger
```

Pull the quantized model back when it finishes:

```sh
.venv/bin/python benchmarks/runpod/volume.py get runs/reader-v5-gemma4-e4b/reader-v5-gemma4-e4b-Q8_0.gguf /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf
.venv/bin/python benchmarks/runpod/volume.py get runs/reader-v5-gemma4-e4b/metrics.json docs/research/results/reader-v5-metrics.json
```

## Logs

Console → **Serverless** → the endpoint → **Requests** → the job: the worker's stdout is there,
including the per-stage progress and every line `reader_lora.py` prints. What to look for:

- `LoRA on 296 Linear layers (e.g. model.language_model.layers.0.self_attn.q_proj.linear)` —
  PEFT found the projections through Gemma 4's clipping wrappers.
- A **Liger fallback** looks like
  `liger unavailable for google/gemma-4-E4B-it (No Liger kernel for gemma4 ...); training without it`,
  followed by training that still runs but needs the logits tensor — expect it to be slower and to
  need a smaller `--max-length`. `"liger": false` in `metrics.json` says the same thing after the fact.
- `SFTConfig does not accept [...]; continuing without them` — a TRL/transformers upgrade changed the
  recipe's field names; check it before trusting the run.
- `resumed_from` in the job's JSON result names the checkpoint it picked up, or `null` for a fresh run.

## Serving the result locally

Same line as reader v4 (docs/research/READER-STRUCTURE.md, "Running the reader locally"):

```sh
llama-server -m /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf --port 8082 -c 12288 -np 1 -ngl 99 --alias rembero-reader \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
```

The harness points at it with `--reader-model rembero-reader --reader-base-url http://127.0.0.1:8082/v1`.
