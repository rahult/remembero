# Training a reader on RunPod Serverless

One job trains (or resumes) one reader run on an H100, exports it text-only, converts it to
GGUF and quantizes it, leaving `runs/<run>/<run>-Q8_0.gguf` and `runs/<run>/metrics.json` on a
network volume. The worker is `handler.py`; the recipe it calls is `benchmarks/train/reader_lora.py`,
the same code Modal ran, so a run started there resumes here from its checkpoint.

**What lives where.** The 40 GB network volume (`/runpod-volume`) holds only what has to outlive
the worker: the Hugging Face cache (`hf/`, the 16 GB base model), the input data
(`data/<run>/conversations.jsonl`, and `heldout.jsonl` if there is one), the newest trainer
checkpoint (`runs/<run>/trainer/checkpoint-N`, replaced each save so only one is kept), and the
outputs (`runs/<run>/<run>-Q8_0.gguf`, `runs/<run>/adapter/`, `runs/<run>/metrics.json`).
Everything heavy and disposable — the trainer's working directory, `merged/` and `merged-text/`
(~14 GB each), the f16 GGUF (~9 GB) — is written to the worker's **80 GB container disk** under
`/workspace/runs/<run>` and dies with the job. That is why the endpoint needs 80 GB of container
disk while the volume stays at 40 GB.

## What it costs

| Item | Price | Notes |
| --- | --- | --- |
| H100 80GB, flex (serverless) | $4.18–4.79 / h | billed per second of execution, idle workers cost nothing |
| Run A (resume reader v5 from checkpoint-50) | ≈ $2.50 | ≈ 26 min at ~65 s/step with batch 4 × accum 16, plus a few minutes of cold start |
| A fresh 4,700-row run at `--max-length 6912` | ≈ $6 | ≈ 80 min |
| Network volume, 40 GB | ≈ $2.80 / month | charged while it exists — delete it when the run is done |

**Measured.** The reader v5 resume — 24 steps at batch 4 × accum 16 — took **62 minutes** of wall
time including the bootstrap and cost **$4.96** at the H100 flex rate: **135 s per step**.

The 16 GB base model is downloaded once into `HF_HOME=/runpod-volume/hf` and reused by every
later job, so only the first job pays for the download.

## One-time setup

There are two ways in. **Pushing the image needs a fast uplink**: the built image is 10.6 GB, and
`docker push` moves all of it. This Mac pushed at ~117 KB/s — four to five hours — so unless the
uplink sustains several MB/s, use the bootstrap template below, which is what actually ran reader
v5.

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

1. **Storage → Network Volume**: create one of **40 GB**; note the datacenter it lands in (e.g. `US-GA-2`).
2. **Storage → S3 API keys**: create a key; keep the access key and secret.
3. **Serverless → New Endpoint**: GPU **H100 80GB**, **max workers 1**, **container disk 80 GB**
   (the merged checkpoints and the f16 GGUF are built there, not on the volume),
   the network volume from step 1 attached, the image `<dockerhub-user>/rembero-reader-train:v1`
   from Docker Hub, **execution timeout 10800 s** (3 hours — the 600 s default kills the job).
4. **Settings → API keys**: create an API key.

Then put the six values in `.env` (already gitignored) and export them:

```sh
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_VOLUME_ID=...
export RUNPOD_DATACENTER=US-GA-2
export RUNPOD_S3_ACCESS_KEY=...
export RUNPOD_S3_SECRET_KEY=...
```

`volume.py` takes the S3 key pair under those names or under the short `S3_ACCESS_KEY` /
`S3_SECRET_KEY`, whichever is set — the `.env` here uses the short ones. The datacenter and volume
id are always `RUNPOD_DATACENTER` and `RUNPOD_VOLUME_ID`.

```sh
uv pip install --python .venv/bin/python runpod boto3
```

### Bootstrap template (no image push)

Reader v5 was trained this way. Instead of building and pushing an image, point a serverless
template at the **public** base image and install the dependencies in the start command; the
worker's own code lives on the network volume. On the real run the bootstrap cost about **two
minutes** of worker time, once per cold start.

- Image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- `containerDiskInGb`: **80** (the merged checkpoints and the f16 GGUF are built there)
- Env: `HF_HOME=/runpod-volume/hf`, `PYTHONPATH=/runpod-volume/code`
- Start command, as `dockerStartCmd: ["bash","-lc", "<the line below>"]`:

```sh
set -e; export DEBIAN_FRONTEND=noninteractive PYTHONPATH=/runpod-volume/code HF_HOME=/runpod-volume/hf TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; pip install -q "transformers>=5.0,<6" "trl>=0.24" "peft>=0.17" "datasets>=3.0" "accelerate>=1.0" "liger-kernel>=0.8.2" sentencepiece protobuf runpod gguf; if [ ! -x /opt/llama.cpp/build/bin/llama-quantize ]; then apt-get update -qq && apt-get install -y -qq --no-install-recommends git cmake build-essential >/dev/null && git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null && cmake --build /opt/llama.cpp/build --target llama-quantize -j "$(nproc)" >/dev/null; fi; exec python -u /runpod-volume/code/benchmarks/runpod/handler.py
```

The code goes onto the volume with `volume.py put`, under `code/benchmarks/...`, plus an empty
`code/benchmarks/__init__.py` so `benchmarks` imports as a package:

```sh
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/train/__init__.py      code/benchmarks/train/__init__.py
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/train/reader_lora.py   code/benchmarks/train/reader_lora.py
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/runpod/__init__.py     code/benchmarks/runpod/__init__.py
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/runpod/handler.py      code/benchmarks/runpod/handler.py
touch /tmp/empty-init.py && .venv/bin/python benchmarks/runpod/volume.py put /tmp/empty-init.py code/benchmarks/__init__.py
```

### Datacenter

The network volume pins the endpoint to its datacenter, so the volume has to be created in a
datacenter that has **80 GB GPU stock** *and* **exposes the S3 API** — both, or the run cannot
start or the data cannot be uploaded. On 2026-09-14: **EU-RO-1** had no 80 GB stock, **EU-NL-1**
has no S3 endpoint, and **US-GA-2** had both (US-CA-2 also has S3 plus H100). Check each before
creating the volume:

```graphql
query { gpuTypes(input: {id: "NVIDIA H100 80GB HBM3"}) {
  lowestPrice(input: {gpuCount: 1, dataCenterId: "US-GA-2"}) { stockStatus }
} }
```

```sh
curl -sI https://s3api-us-ga-2.runpod.io/    # 401 = the endpoint exists; no connection = it does not
```

### Creating the template and the endpoint through the REST API

Both can be created without the console: `POST /v1/templates` (the image, `containerDiskInGb`,
the env pair and `dockerStartCmd` above), then `POST /v1/endpoints` with `templateId`,
`gpuTypeIds`, `networkVolumeId`, `dataCenterIds`, `workersMax` 1 and `executionTimeoutMs`
10800000. Widening `gpuTypeIds` to all five 80 GB types is what actually got a worker assigned:

```
NVIDIA H100 80GB HBM3, NVIDIA H100 PCIe, NVIDIA H100 NVL,
NVIDIA A100-SXM4-80GB, NVIDIA A100 80GB PCIe
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
- `train: resuming from checkpoint-50` is the worker copying that checkpoint off the volume onto the
  container disk before training; `resumed_from` in the job's JSON result names the checkpoint the
  trainer actually picked up, or `null` for a fresh run. Each `train: checkpoint saved` line means a
  new checkpoint has been copied back to the volume and the previous one deleted there — that is the
  state a re-submitted job resumes from if the worker dies.

## Serving the result locally

Same line as reader v4 (docs/research/READER-STRUCTURE.md, "Running the reader locally"):

```sh
llama-server -m /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf --port 8083 -c 12288 -np 1 -ngl 99 --alias rembero-reader-v5 \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
```

Port **8083** and alias **`rembero-reader-v5`**, not 8082 / `rembero-reader`: that slot belongs to
the v4 server, and a harness pointed at 8082 would measure v5 while labelling the numbers v4.

The harness points at it with `--reader-model rembero-reader-v5 --reader-base-url http://127.0.0.1:8083/v1`.
The reader contract pins the prompt but not the retrieval depth, so every paired run must also pass
`--top-k 4 --multi-session-top-k 15 --temporal-top-k 10` explicitly (the harness default is 5/5) —
without them the comparison is not paired.

## Reader serving pod

For paired harness runs against a fine-tuned reader without a local GPU: a community pod runs
vLLM over the reader's merged weights, rebuilt on the pod from the LoRA adapter, and the harness on
the Mac calls it through RunPod's public proxy. `pod.py` launches and manages the pod;
`prepare_reader.py` is what the pod runs before it serves.

**Where.** EUR-NO-1 (network volumes and an S3 endpoint both present), one **RTX 5090 32 GB**
community GPU ($0.69/h on 2026-09-15), the network volume mounted at **`/workspace`**
(`RUNPOD_VOLUME_ID` / `RUNPOD_DATACENTER` must point at the EUR-NO-1 volume for `volume.py`).
Stop the pod whenever no arm is running. RunPod has refused to stop pods that have a network
volume attached; if `stop` is refused, `terminate` and `create-serve` again: everything the pod
built lives on the volume, so the new pod skips the merge.

**Image.** `vllm/vllm-openai:v0.29.0-cu129`, the newest stable (non-nightly) vLLM release on
Docker Hub on 2026-09-15, pushed 2026-09-09; Gemma 4 support landed in vLLM in spring 2026. The
`-cu129` build (CUDA 12.9.1) rather than plain `v0.29.0` (CUDA 13.0.2) so hosts on a 12.9 driver
qualify as well; the payload sets `allowedCudaVersions` to 12.9 and 13.0 accordingly. Both builds
compile kernels for the 5090 (`TORCH_CUDA_ARCH_LIST` includes 12.0). The image's entrypoint is
`vllm serve`, so the pod overrides it with `dockerEntrypoint ["bash","-lc"]` and puts the whole
start command in `dockerStartCmd`. If the tag is gone, pin the newest `vX.Y.Z-cu129` from
https://hub.docker.com/r/vllm/vllm-openai/tags and record it here and in `pod.py` (`IMAGE`).

**What lives on the volume.** `code/benchmarks/{train,runpod}/*` (the code the pod imports,
`PYTHONPATH=/workspace/code`), `runs/<run>/adapter/` (uploaded from the Mac), `hf/` (the public
`google/gemma-4-E4B-it` base, downloaded on the first boot; `HF_HOME=/workspace/hf`), and after
the first boot `runs/<run>/merged-text/` with its `.prepared` marker. No Hugging Face or Modal
credential goes to the pod; the only secret in its env is its own `VLLM_API_KEY`.

**The three Gemma 4 steps.** `prepare_reader.py --root /workspace --run <run>` merges the adapter
into the base (`reader_lora.merge_adapter`, writing `merged/`), exports the text-only causal LM
(`export_text_only`, writing `merged-text/`), puts back the KV-sharing `self_attn` tensors
transformers drops on save (`restore_dropped_weights`; vLLM refuses the checkpoint without them),
copies any tokenizer file the export did not write from the adapter, refuses to finish without a
chat template, writes `merged-text/.prepared` and deletes `merged/`. With the marker present it
exits at once, so every boot runs it; an unmarked `merged-text/` from a boot that died halfway is
deleted and rebuilt. Expect the first boot to take the base download plus the merge (tens of
minutes); later boots go straight to vLLM.

**Serve flags.** The Modal serve of record, flag for flag: `--dtype bfloat16 --max-model-len 8192
--max-num-seqs 32 --gpu-memory-utilization 0.92 --reasoning-parser gemma4
--default-chat-template-kwargs '{"enable_thinking":false}'`, plus `--served-model-name`, and
`--api-key "$VLLM_API_KEY"` because the proxy URL is public (the start command stops if the key
is empty).

### Setup and use

```sh
# code and adapter onto the volume (S3 keys as in "One-time setup")
touch /tmp/empty-init.py && .venv/bin/python benchmarks/runpod/volume.py put /tmp/empty-init.py code/benchmarks/__init__.py
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/train  code/benchmarks/train
.venv/bin/python benchmarks/runpod/volume.py put benchmarks/runpod code/benchmarks/runpod
.venv/bin/python benchmarks/runpod/volume.py put <local adapter dir> runs/reader-v4-gemma4-e4b/adapter

.venv/bin/python benchmarks/runpod/pod.py create-serve --run reader-v4-gemma4-e4b --served-name rembero-reader-v4
.venv/bin/python benchmarks/runpod/pod.py wait --served-name rembero-reader-v4   # polls /v1/models every 20 s, up to 30 min
.venv/bin/python benchmarks/runpod/pod.py status                                  # the serving pod, or every pod if none is recorded
.venv/bin/python benchmarks/runpod/pod.py stop      # between arms
.venv/bin/python benchmarks/runpod/pod.py start     # prepare is a no-op now; vLLM loads in a few minutes
.venv/bin/python benchmarks/runpod/pod.py terminate
```

`create-serve` takes `--gpu`, `--datacenter` (default EUR-NO-1) and `--volume` (default
`RUNPOD_VOLUME_ID`). It reads `RUNPOD_API_KEY` from the environment or `.env`, generates a
`VLLM_API_KEY` once (`secrets.token_urlsafe(32)`) if `.env` has none, passes
`~/.ssh/id_ed25519.pub` as `PUBLIC_KEY` if it exists, prints the pod id and
`https://<pod>-8000.proxy.runpod.net/v1`, and writes both to `.env` as `RUNPOD_SERVE_POD_ID` and
`RUNPOD_SERVE_URL`; the other subcommands default `--pod` to that id. The harness points at the
pod with `--reader-model rembero-reader-v4 --reader-base-url "$RUNPOD_SERVE_URL"` and the
`--reader-api-key "$VLLM_API_KEY"`, plus the retrieval-depth flags every paired run passes.

Logs: console → **Pods** → the pod → **Logs** (the `prepare_reader` lines, then vLLM's startup).
The vLLM image runs no SSH daemon, so `22/tcp` and `PUBLIC_KEY` only help if one is added later;
use the web terminal or the logs.
