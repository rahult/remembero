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
  .venv/bin/modal deploy benchmarks/modal/train_lora.py        # prints the serve URL
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

APP_NAME = "rembero-finetune"
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
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.11", "huggingface_hub")
    .env({"HF_HOME": f"{VOL}/hf", "VLLM_LOGGING_LEVEL": "WARNING"})
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
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    try:
        import fla  # noqa: F401

        print("flash-linear-attention: available (fast Gated DeltaNet path)")
    except ImportError:
        print("flash-linear-attention: MISSING; Gated DeltaNet layers will run the slow torch fallback")

    run_dir = Path(VOL) / "runs" / run
    data_dir = Path(VOL) / "data" / run
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    )

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = SFTConfig(
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
        # Recompute activations only where memory is tight (24 GB cards); it costs ~30% speed.
        gradient_checkpointing=TRAIN_GPU.upper().split(":")[0] in {"A10G", "A10", "L4", "T4"},
        # Query and extraction prompts differ a lot in length; length-sorted batches cut padding.
        group_by_length=True,
        packing=False,
        # conversational prompt/completion rows: TRL puts the loss on the completion only
        completion_only_loss=True,
    )
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
    metrics: dict = {"run": run, "base_model": BASE_MODEL, "gpu": TRAIN_GPU}
    if eval_ds is not None:
        evaluation = trainer.evaluate()
        metrics["heldout_loss"] = evaluation.get("eval_loss")
    metrics["train_loss"] = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
    metrics["train_tokens_estimate"] = sum(
        len(tokenizer.apply_chat_template([*r["prompt"], *r["completion"]], tokenize=True))
        for r in train_rows[:200]
    ) * len(train_rows) // max(len(train_rows[:200]), 1)

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merge:
        merged_dir = run_dir / "merged"
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
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
    scaledown_window=5 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve() -> None:
    """OpenAI-compatible /v1/chat/completions over the merged weights of SERVE_RUN (default: latest)."""
    run = SERVE_RUN
    if run == "latest":
        run = (Path(VOL) / "runs" / "latest").read_text().strip()
    merged = Path(VOL) / "runs" / run / "merged"
    if not merged.exists():
        raise RuntimeError(f"no merged weights at {merged}; train with merge=True first")
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(merged),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        # Qwen3.5's template opens a <think> block in the generation prompt; the parser moves
        # everything up to </think> into reasoning_content so the harness sees only the answer.
        "--reasoning-parser",
        "qwen3",
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
    check_only: bool = False,
) -> None:
    check_data(data, heldout if Path(heldout).exists() else None)
    if check_only:
        return
    with volume.batch_upload(force=True) as batch:
        batch.put_file(data, f"data/{run}/conversations.jsonl")
        if Path(heldout).exists():
            batch.put_file(heldout, f"data/{run}/heldout.jsonl")
    print(f"uploaded data for run {run}; training on {TRAIN_GPU} ...")
    metrics = train.remote(run=run, epochs=epochs, lr=lr, lora_rank=lora_rank)
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
