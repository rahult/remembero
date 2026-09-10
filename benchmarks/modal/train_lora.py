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
import time
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


def _to_prompt_completion(path: str) -> list[dict]:
    """conversations.jsonl -> TRL conversational prompt/completion rows (loss on the assistant turn only)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            if messages[-1]["role"] != "assistant":
                raise ValueError("last message must be the assistant turn")
            rows.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    return rows


def check_data(data: str, heldout: str | None) -> None:
    """Local validation, no GPU: shape, counts, and a size estimate."""
    train = _to_prompt_completion(data)
    held = _to_prompt_completion(heldout) if heldout else []
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


def load_base_model(model_id: str, torch):
    """Text-only causal LM where the checkpoint offers one; multimodal wrapper (Gemma 4) otherwise."""
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError, OSError) as error:
        from transformers import AutoModelForImageTextToText

        print(f"AutoModelForCausalLM refused {model_id} ({str(error)[:80]}); loading the multimodal class")
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


LORA_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def lora_targets(model) -> list[str]:
    """Full names of the text model's attention/MLP Linear layers.

    Matching by short name ("q_proj") is enough for Qwen and Llama, but Gemma 4 wraps each
    projection in a clipping module PEFT cannot adapt, with the real Linear one level down
    (q_proj.linear). Naming the Linear modules explicitly works for both, and skips the
    vision and audio towers of multimodal checkpoints.
    """
    import torch.nn as nn

    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(tower in name for tower in ("vision", "audio", "embed_vision", "embed_audio")):
            continue
        parts = name.split(".")
        leaf, parent = parts[-1], (parts[-2] if len(parts) > 1 else "")
        if leaf in LORA_PROJECTIONS or (leaf == "linear" and parent in LORA_PROJECTIONS):
            names.append(name)
    if not names:
        raise RuntimeError("no LoRA target modules found; unexpected model layout")
    print(f"LoRA on {len(names)} Linear layers (e.g. {names[0]})")
    return names


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
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    merged_dir = Path(VOL) / "runs" / run / "merged"
    text_dir = Path(VOL) / "runs" / run / "merged-text"
    config = AutoConfig.from_pretrained(str(merged_dir))
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return f"{merged_dir} is already text-only"
    full = AutoModelForImageTextToText.from_pretrained(str(merged_dir), dtype=torch.bfloat16)
    language_model = getattr(full.model, "language_model", None) or getattr(full, "language_model")
    text_cls = getattr(__import__("transformers"), text_config.architectures[0] if getattr(text_config, "architectures", None) else "Gemma4ForCausalLM")
    text_model = text_cls(text_config)
    missing, unexpected = text_model.model.load_state_dict(language_model.state_dict(), strict=False)
    if hasattr(full, "lm_head") and hasattr(text_model, "lm_head"):
        text_model.lm_head.load_state_dict(full.lm_head.state_dict())
    print(f"text-only export: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    text_model.to(torch.bfloat16).save_pretrained(str(text_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(merged_dir)).save_pretrained(str(text_dir))
    volume.commit()
    return str(text_dir)


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
    timeout=3 * 60 * 60,
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
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    try:
        import fla  # noqa: F401

        print("flash-linear-attention: available (fast Gated DeltaNet path)")
    except ImportError:
        print("flash-linear-attention: MISSING; Gated DeltaNet layers will run the slow torch fallback")

    run_dir = Path(VOL) / "runs" / run
    data_dir = Path(VOL) / "data" / run
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(base_model, torch)

    train_rows = _to_prompt_completion(str(data_dir / "conversations.jsonl"))
    heldout_path = data_dir / "heldout.jsonl"
    held_rows = _to_prompt_completion(str(heldout_path)) if heldout_path.exists() else []
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(held_rows) if held_rows else None

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets(model),
    )
    # transformers 5.x keeps renaming/removing TrainingArguments fields (warmup_ratio,
    # group_by_length, ...); keep only the ones this SFTConfig accepts and say what was dropped.
    import inspect

    wanted = dict(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="linear",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size,
        bf16=True,
        max_length=max_length,
        logging_steps=10,
        # Checkpoint the adapter to the Volume so a preempted or killed container resumes
        # instead of starting over (an A10G epoch is ~2.5 h).
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        # Recompute activations everywhere: r13's long transcript examples (2k-token rows in
        # batches of 8) ran an 80 GB H100 out of memory without it. Costs ~30% speed.
        gradient_checkpointing=True,
        # Query and extraction prompts differ a lot in length; length-sorted batches cut padding.
        group_by_length=True,
        packing=False,
        # conversational prompt/completion rows: TRL puts the loss on the completion only
        completion_only_loss=True,
        # Render prompts with the empty <think></think> block so the tokenized prompt is an
        # exact prefix of prompt+completion (otherwise the template's trailing newline merges
        # with the completion's and TRL warns on every row).
        chat_template_kwargs={"enable_thinking": False},
    )
    accepted = inspect.signature(SFTConfig.__init__).parameters
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        print(f"SFTConfig does not accept {dropped}; continuing without them")
    config = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})
    from transformers import TrainerCallback

    class CommitVolume(TrainerCallback):
        """Modal only persists Volume writes on commit; do it right after each checkpoint."""

        def on_save(self, args, state, control, **kwargs):
            volume.commit()

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[CommitVolume()],
    )
    checkpoints = sorted(
        (run_dir / "trainer").glob("checkpoint-*"),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if checkpoints:
        print(f"resuming from {checkpoints[-1]}")
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    metrics: dict = {"run": run, "base_model": base_model, "gpu": TRAIN_GPU}
    if eval_ds is not None:
        evaluation = trainer.evaluate()
        metrics["heldout_loss"] = evaluation.get("eval_loss")
    # log_history ends with the eval entry; the train summary is the last entry that has train_loss
    metrics["train_loss"] = next(
        (h["train_loss"] for h in reversed(trainer.state.log_history) if "train_loss" in h), None
    )
    metrics["final_step_loss"] = next(
        (h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None
    )
    sample = train_rows[:200]
    sample_tokens = sum(
        len(tokenizer(tokenizer.apply_chat_template([*r["prompt"], *r["completion"]], tokenize=False))["input_ids"])
        for r in sample
    )
    metrics["train_tokens_estimate"] = sample_tokens * len(train_rows) // max(len(sample), 1)

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merge:
        merged_dir = run_dir / "merged"
        base = load_base_model(base_model, torch)
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        save_processor_files(base_model, merged_dir)
        metrics["merged_dir"] = str(merged_dir)
    (Path(VOL) / "runs" / "latest").write_text(run)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
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
        "--gpu-memory-utilization",
        "0.90",
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
    check_only: bool = False,
) -> None:
    check_data(data, heldout if Path(heldout).exists() else None)
    if check_only:
        return
    with volume.batch_upload(force=True) as batch:
        batch.put_file(data, f"data/{run}/conversations.jsonl")
        if Path(heldout).exists():
            batch.put_file(heldout, f"data/{run}/heldout.jsonl")
    print(f"uploaded data for run {run}; training {BASE_MODEL} on {TRAIN_GPU} ...")
    metrics = train.remote(
        run=run,
        epochs=epochs,
        lr=lr,
        lora_rank=lora_rank,
        batch_size=batch_size,
        grad_accum=grad_accum,
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

    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        check_data(
            sys.argv[2] if len(sys.argv) > 2 else "data/training/conversations.jsonl",
            sys.argv[3] if len(sys.argv) > 3 else "data/training/heldout.jsonl",
        )
    else:
        print(__doc__)
