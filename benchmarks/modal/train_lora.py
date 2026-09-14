"""Train and serve Remembero's query+extraction LoRA on Modal instead of Tinker.

One file, three things:

  train   rank-32 LoRA SFT of a small open model on data/training/conversations.jsonl,
          held-out loss on heldout.jsonl, adapter (and merged weights) saved to a Volume.
  serve   vLLM OpenAI-compatible endpoint over the merged weights, scale-to-zero, so the
          existing harness runs unchanged with --chat-api openai.
  check   local, no GPU: validate the JSONL and print token/size estimates.

Usage (after `modal setup` once, which opens a browser login):

  .venv/bin/modal run benchmarks/modal/train_lora.py --run r11 \
      --data data/training/conversations.jsonl --heldout data/training/heldout.jsonl
  .venv/bin/modal deploy benchmarks/modal/train_lora.py        # prints the serve URL; requests need Authorization: Bearer $MODAL_SERVE_API_KEY
  OLLAMA_URL=<url> node dist/evals/run-agent-boundary.js --chat-api openai \
      --model dialect --conditions remembero-closure --seeds 7
  node dist/evals/run-extraction-bench.js --base-url <url>/v1 --api-key x --models dialect --vocabulary closed

Env knobs: MODAL_TRAIN_GPU (default H100), MODAL_SERVE_GPU (default L4), BASE_MODEL
(default Qwen/Qwen3.5-4B), SERVE_RUN (which run the endpoint serves; default "latest").
Cost model and alternatives: docs/research/FINETUNE-PROVIDER-MATRIX.md.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

# MODAL_APP_NAME lets a second serving app (another run, another GPU) coexist with the main one
APP_NAME = os.environ.get("MODAL_APP_NAME", "rembero-finetune")
VOLUME_NAME = "rembero-finetune"
VOL = "/vol"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")
TRAIN_GPU = os.environ.get("MODAL_TRAIN_GPU", "H100")
SERVE_GPU = os.environ.get("MODAL_SERVE_GPU", "L4")
SERVE_RUN = os.environ.get("SERVE_RUN", "latest")
SERVED_MODEL_NAME = "dialect"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8",
        # Qwen3.5 is 3/4 Gated DeltaNet layers; without these Triton kernels transformers
        # falls back to a pure-torch path that ran at ~60 s/step on an A10G (5+ hours).
        "flash-linear-attention>=0.3",
        "transformers>=5.0,<6",
        "trl>=0.24",
        "peft>=0.17",
        "datasets>=3.0",
        "accelerate>=1.0",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # the training recipe itself lives in benchmarks/train/reader_lora.py, which also runs on
    # a rented GPU; ship it into the container instead of keeping a second copy here
    .add_local_python_source("benchmarks")
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.11", "huggingface_hub")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "VLLM_LOGGING_LEVEL": "WARNING",
            # Image env is fixed at deploy time from the local shell, so the container sees the
            # run the deployer chose; reading os.environ inside serve() would see nothing.
            "SERVE_RUN": SERVE_RUN,
            # The slim image has no nvcc; FlashInfer's sampler JIT-compiles at startup and dies.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


# ---- data -----------------------------------------------------------------------------


def check_data(data: str, heldout: str | None) -> None:
    """Local validation, no GPU: shape, counts, and a size estimate."""
    from benchmarks.train.reader_lora import to_prompt_completion

    train = to_prompt_completion(data)
    held = to_prompt_completion(heldout) if heldout else []
    chars = sum(len(m["content"]) for r in train for m in [*r["prompt"], *r["completion"]])
    completion_chars = sum(len(m["content"]) for r in train for m in r["completion"])
    print(
        json.dumps(
            {
                "train_rows": len(train),
                "heldout_rows": len(held),
                "approx_train_tokens": chars // 4,
                "approx_completion_tokens": completion_chars // 4,
                "base_model": BASE_MODEL,
                "train_gpu": TRAIN_GPU,
            },
            indent=2,
        )
    )


# ---- train ----------------------------------------------------------------------------


def save_processor_files(base_model: str, target: Path) -> None:
    """Multimodal checkpoints (Gemma 4) need their processor configs next to the weights or
    vLLM refuses to load them. Copy the hub's small config files that the merge did not write
    (never weights, never the model config the merge produced)."""
    import shutil

    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(
            snapshot_download(
                base_model,
                allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"],
            )
        )
    except Exception as error:
        print(f"could not fetch processor files for {base_model}: {str(error)[:120]}")
        return
    copied = []
    for path in snapshot.iterdir():
        if path.is_dir() or path.name in {"config.json", "generation_config.json"}:
            continue
        if not (target / path.name).exists():
            shutil.copy2(path, target / path.name)
            copied.append(path.name)
    print(f"copied processor files into {target}: {copied}")


@app.function(image=train_image, volumes={VOL: volume}, timeout=30 * 60)
def export_text_only(run: str) -> str:
    """Re-save a merged multimodal checkpoint (Gemma 4) as its text-only causal LM.

    vLLM's loader mismatched the multimodal wrapper's weight names on the merged save; the
    plain Gemma4ForCausalLM layout loads cleanly, and this project never sends images or audio.
    """
    from benchmarks.train.reader_lora import export_text_only as core

    path = core(Path(VOL) / "runs" / run)
    volume.commit()
    return str(path)


gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential", "curl", "libcurl4-openssl-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF",
        "cmake --build /opt/llama.cpp/build --target llama-quantize -j 8",
        "pip install -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
        # the merged checkpoint's tokenizer config is written by transformers 5; the
        # converter's pinned transformers cannot read it
        "pip install 'transformers>=5.0,<6'",
    )
)


@app.function(image=gguf_image, volumes={VOL: volume}, timeout=60 * 60, memory=32768, cpu=8)
def export_gguf(run: str, quant: str = "Q8_0") -> str:
    """Convert a run's text-only merged checkpoint to GGUF and quantize it, for Ollama.

    The local-serving story: the writer runs under the user's own Ollama next to the
    embedding model, no Modal endpoint in the product path. Uses llama.cpp's converter;
    Gemma 4 text-only is a supported architecture (Ollama ships gemma4 GGUFs).
    """
    import subprocess

    text_dir = Path(VOL) / "runs" / run / "merged-text"
    if not text_dir.exists():
        raise SystemExit(f"{text_dir} missing; run export_text_only first")
    out_dir = Path(VOL) / "runs" / run / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{run}-f16.gguf"
    quantized = out_dir / f"{run}-{quant}.gguf"
    subprocess.run(
        [
            "python",
            "/opt/llama.cpp/convert_hf_to_gguf.py",
            str(text_dir),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run(
        ["/opt/llama.cpp/build/bin/llama-quantize", str(f16), str(quantized), quant],
        check=True,
    )
    f16.unlink()
    volume.commit()
    size = quantized.stat().st_size / 2**30
    print(f"wrote {quantized} ({size:.2f} GiB); download with: modal volume get rembero-finetune runs/{run}/gguf/{quantized.name} .")
    return str(quantized)


@app.function(image=train_image, volumes={VOL: volume}, timeout=30 * 60, memory=65536)
def restore_dropped_weights(run: str, base_model: str) -> str:
    """Put back tensors transformers drops on save but vLLM requires.

    Gemma 4's KV-sharing layers carry k_proj/v_proj/k_norm in Google's checkpoint; the
    transformers implementation has no such parameters there, so a re-saved model lacks them
    and vLLM refuses to load. LoRA never touched those layers, so the originals are exact.
    """
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    text_dir = Path(VOL) / "runs" / run / "merged-text"
    target = text_dir / "model.safetensors"
    exported = load_file(str(target))
    prefix = "model.language_model."
    added = []
    snapshot = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    for shard in glob.glob(f"{snapshot}/*.safetensors"):
        with safe_open(shard, "pt") as st:
            for key in st.keys():
                if not key.startswith(prefix):
                    continue
                new_key = "model." + key[len(prefix):]
                if new_key not in exported and ".self_attn." in new_key:
                    exported[new_key] = st.get_tensor(key)
                    added.append(new_key)
    save_file(exported, str(target), metadata={"format": "pt"})
    volume.commit()
    print(f"restored {len(added)} tensors, e.g. {added[:3]}")
    return str(target)


@app.function(image=train_image, volumes={VOL: volume}, timeout=15 * 60)
def add_processor(run: str, base_model: str) -> str:
    """Patch an already-merged run with its base model's processor files."""
    target = Path(VOL) / "runs" / run / "merged"
    save_processor_files(base_model, target)
    volume.commit()
    return str(target)


def reasoning_flags(base_model: str) -> list[str]:
    """Serving flags that depend on the base model's chat template."""
    lowered = base_model.lower()
    if "qwen3" in lowered:
        # Qwen3.5 opens a <think> block in the generation prompt; render the empty block the
        # training rows carried and route anything before </think> into reasoning_content.
        return [
            "--reasoning-parser",
            "qwen3",
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": False}),
        ]
    if "gemma-4" in lowered or "gemma4" in lowered:
        # Gemma 4 thinks only when asked; keep it off and parse just in case.
        return [
            "--reasoning-parser",
            "gemma4",
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": False}),
        ]
    return []


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    # reader rows at 8k tokens take ~80 s a step on an H100; 139 steps overran three hours
    # and Modal killed the function 15 minutes short (it resumed from checkpoint-100)
    timeout=6 * 60 * 60,
    volumes={VOL: volume},
)
def train(
    run: str,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 8,
    grad_accum: int = 8,
    max_length: int = 2048,
    merge: bool = True,
    base_model: str = BASE_MODEL,
) -> dict:
    # BASE_MODEL is read from the *local* environment at import; the container never sees
    # it, so the caller passes it in. (Two "Gemma" runs silently trained Qwen before this.)
    from benchmarks.train.reader_lora import train_lora

    metrics = train_lora(Path(VOL) / "data" / run, Path(VOL) / "runs" / run, base_model,
                         epochs=epochs, lr=lr, lora_rank=lora_rank, batch_size=batch_size,
                         grad_accum=grad_accum, max_length=max_length, merge=merge, on_save=volume.commit)
    metrics.update(run=run, gpu=TRAIN_GPU)
    if merge:
        save_processor_files(base_model, Path(VOL) / "runs" / run / "merged")
    (Path(VOL) / "runs" / "latest").write_text(run)
    volume.commit()
    return metrics


# ---- serve ----------------------------------------------------------------------------


@app.function(
    image=serve_image,
    gpu=SERVE_GPU,
    volumes={VOL: volume},
    # VLLM_API_KEY: the bearer token vLLM requires on every request. Create once with
    #   modal secret create rembero-vllm VLLM_API_KEY=<random>
    # and keep the same value in .env as MODAL_SERVE_API_KEY for the harness.
    secrets=[modal.Secret.from_name("rembero-vllm")],
    scaledown_window=5 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve() -> None:
    """OpenAI-compatible /v1/chat/completions over the merged weights of SERVE_RUN (default: latest)."""
    run = os.environ.get("SERVE_RUN", SERVE_RUN)
    if run == "latest":
        run = (Path(VOL) / "runs" / "latest").read_text().strip()
    merged = Path(VOL) / "runs" / run / "merged"
    # a text-only re-export (see export_text_only) is preferred when present
    if (Path(VOL) / "runs" / run / "merged-text").exists():
        merged = Path(VOL) / "runs" / run / "merged-text"
    if not merged.exists():
        raise RuntimeError(f"no merged weights at {merged}; train with merge=True first")
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError("VLLM_API_KEY missing: the endpoint is public without it; see the rembero-vllm secret")
    metrics_path = Path(VOL) / "runs" / run / "metrics.json"
    served_base = (
        json.loads(metrics_path.read_text()).get("base_model", BASE_MODEL)
        if metrics_path.exists()
        else BASE_MODEL
    )
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(merged),
        "--served-model-name",
        SERVED_MODEL_NAME,
        # run-specific alias so benchmark result files name the run, like the Tinker proxy did
        f"finetune/{served_base.split('/')[-1].lower()}-{run}-modal",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        # vLLM's default profiling run assumes 256 concurrent sequences; on a 24 GB L4 the
        # newer releases then find no memory left for the KV cache (2026-09-12). 32 is far
        # above what the benchmarks drive.
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.92",
        *reasoning_flags(served_base),
        "--api-key",
        api_key,
    ]
    subprocess.Popen(cmd)


# ---- local entrypoint -----------------------------------------------------------------------


@app.local_entrypoint()
def main(
    run: str,
    data: str = "data/training/conversations.jsonl",
    heldout: str = "data/training/heldout.jsonl",
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 8,
    grad_accum: int = 8,
    # Rows longer than this are truncated from the right, which for a prompt/completion row
    # drops the completion: 39% of the real-session extraction rows exceed 2048 tokens, so
    # runs that include them need 4096; reader rows (a whole rendered history) need 8192.
    max_length: int = 2048,
    check_only: bool = False,
) -> None:
    check_data(data, heldout if Path(heldout).exists() else None)
    if check_only:
        return
    with volume.batch_upload(force=True) as batch:
        batch.put_file(data, f"data/{run}/conversations.jsonl")
        if Path(heldout).exists():
            batch.put_file(heldout, f"data/{run}/heldout.jsonl")
    print(
        f"uploaded data for run {run}; training {BASE_MODEL} on {TRAIN_GPU} "
        f"(max_length {max_length}) ..."
    )
    metrics = train.remote(
        run=run,
        epochs=epochs,
        lr=lr,
        lora_rank=lora_rank,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_length=max_length,
        base_model=BASE_MODEL,
    )
    print(json.dumps(metrics, indent=2))
    print(
        "\nnext:\n"
        f"  .venv/bin/modal deploy benchmarks/modal/train_lora.py   # serve run '{run}' (latest)\n"
        "  OLLAMA_URL=<serve url> node dist/evals/run-agent-boundary.js --chat-api openai "
        f"--model {SERVED_MODEL_NAME} --conditions remembero-closure --seeds 7\n"
        f"  node dist/evals/run-extraction-bench.js --base-url <serve url>/v1 --api-key x --models {SERVED_MODEL_NAME} --vocabulary closed"
    )


if __name__ == "__main__":
    # `python benchmarks/modal/train_lora.py --check [data] [heldout]` validates the data
    # locally without Modal credentials.
    import sys

    # run as a script, sys.path[0] is this directory, so the repository root (which holds the
    # `benchmarks` package check_data imports the parser from) has to be added explicitly
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        check_data(
            sys.argv[2] if len(sys.argv) > 2 else "data/training/conversations.jsonl",
            sys.argv[3] if len(sys.argv) > 3 else "data/training/heldout.jsonl",
        )
    else:
        print(__doc__)
