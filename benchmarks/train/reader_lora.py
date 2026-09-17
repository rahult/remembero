"""LoRA SFT of a small chat model on conversations.jsonl, the recipe Remembero's writer and
reader runs use, with nothing Modal-specific: paths are arguments and persistence is a
callback. benchmarks/modal/train_lora.py and benchmarks/runpod/handler.py both call this.

Launch the Modal app from the repository root: its `add_local_python_source("benchmarks")` has to
resolve `benchmarks` as a namespace package on the local path, which only holds from there."""

from __future__ import annotations

import inspect
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

LORA_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def to_prompt_completion(path: str) -> list[dict]:
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


def load_base_model(model_id: str, dtype=None):
    """Text-only causal LM where the checkpoint offers one; multimodal wrapper (Gemma 4) otherwise.
    dtype defaults to bf16 on CUDA and float32 on CPU (training); pass it to pin the weights'
    precision whatever the device (merging)."""
    import torch
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = dict(dtype=dtype, attn_implementation="sdpa")
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError, OSError) as error:
        from transformers import AutoModelForImageTextToText

        print(f"AutoModelForCausalLM refused {model_id} ({str(error)[:80]}); loading the multimodal class")
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


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


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "trainer").glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def train_lora(
    data_dir: Path,
    run_dir: Path,
    base_model: str,
    *,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 8,
    grad_accum: int = 8,
    max_length: int = 8192,
    merge: bool = True,
    save_steps: int = 25,
    liger: bool = False,
    on_save: Callable[[], None] | None = None,
    resume: bool = True,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(base_model)
    train_rows = to_prompt_completion(str(data_dir / "conversations.jsonl"))
    heldout = data_dir / "heldout.jsonl"
    held_rows = to_prompt_completion(str(heldout)) if heldout.exists() else []
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(held_rows) if held_rows else None
    peft_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
                             task_type="CAUSAL_LM", target_modules=lora_targets(model))
    # Two fields this recipe once carried are deliberately absent: `chat_template_kwargs`, which
    # TRL 1.x removed from SFTConfig (it is a per-example dataset column there), and
    # `group_by_length`, which transformers 5 removed. Reader v4 and v5 were trained without
    # them (the v5 checkpoint's saved SFTConfig on the Modal volume has neither, with
    # completion_only_loss=True, max_length=8192, use_liger_kernel=False, batch 4 x grad-accum 16,
    # seed 42), so leaving them out is what reproduces those runs. They are not passed at all
    # rather than filtered out below, so that a future TRL which accepts one again cannot quietly
    # change the recipe under a resume.
    wanted = dict(
        output_dir=str(run_dir / "trainer"), num_train_epochs=epochs, learning_rate=lr,
        lr_scheduler_type="linear", per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, per_device_eval_batch_size=batch_size,
        bf16=torch.cuda.is_available(), max_length=max_length, logging_steps=10,
        save_strategy="steps", save_steps=save_steps, save_total_limit=1,
        eval_strategy="epoch" if eval_ds is not None else "no", report_to=[],
        gradient_checkpointing=True, packing=False, completion_only_loss=True,
        # Liger's fused linear cross-entropy never materialises the logits tensor (34 GB at
        # batch 8 x 8k over Gemma's 262k vocabulary); it is what lets the run fit and go faster.
        use_liger_kernel=liger,
    )
    # transformers 5.x keeps renaming/removing TrainingArguments fields; keep only the ones this
    # SFTConfig accepts and say what was dropped, so a changed recipe is visible in the log.
    accepted = inspect.signature(SFTConfig.__init__).parameters
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        print(f"SFTConfig does not accept {dropped}; continuing without them")
    config = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})

    class Persist(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if on_save is not None:
                on_save()

    try:
        trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tokenizer, peft_config=peft_config, callbacks=[Persist()])
    except Exception as error:  # Liger has no patch for this architecture: fall back, say so
        missing_liger = isinstance(error, ImportError) or "liger" in str(error).lower()
        if not liger or not missing_liger:
            raise
        print(f"liger unavailable for {base_model} ({str(error)[:120]}); training without it")
        config = SFTConfig(**{k: v for k, v in {**wanted, "use_liger_kernel": False}.items() if k in accepted})
        model = load_base_model(base_model)
        trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tokenizer, peft_config=peft_config, callbacks=[Persist()])
    metrics_liger = bool(getattr(trainer.args, "use_liger_kernel", False))
    checkpoint = latest_checkpoint(run_dir) if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    metrics: dict = {"base_model": base_model, "resumed_from": str(checkpoint) if checkpoint else None, "liger": metrics_liger, "max_length": max_length}
    if eval_ds is not None:
        metrics["heldout_loss"] = trainer.evaluate().get("eval_loss")
    metrics["train_loss"] = next((h["train_loss"] for h in reversed(trainer.state.log_history) if "train_loss" in h), None)
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merge:
        merged_dir = run_dir / "merged"
        base = load_base_model(base_model)
        PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        metrics["merged_dir"] = str(merged_dir)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if on_save is not None:
        on_save()
    return metrics


@contextmanager
def default_dtype(dtype):
    """torch's default floating dtype set to `dtype` inside the block, restored afterwards."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def export_text_only(run_dir: Path) -> Path:
    """Re-save a merged multimodal checkpoint (Gemma 4) as its text-only causal LM, the layout
    llama.cpp's converter reads."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    merged_dir, text_dir = run_dir / "merged", run_dir / "merged-text"
    config = AutoConfig.from_pretrained(str(merged_dir))
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return merged_dir
    full = AutoModelForImageTextToText.from_pretrained(str(merged_dir), dtype=torch.bfloat16)
    language_model = getattr(full.model, "language_model", None) or getattr(full, "language_model")
    text_cls = getattr(transformers, text_config.architectures[0] if getattr(text_config, "architectures", None) else "Gemma4ForCausalLM")
    # Built under a bf16 default: the float32 default would allocate ~30 GB for ~8B parameters
    # next to the loaded bf16 multimodal model. The saved weights are the same, since the model
    # was cast to bf16 before saving either way.
    with default_dtype(torch.bfloat16):
        text_model = text_cls(text_config)
    # Gemma 4's KV-sharing layers have no counterpart here, so a non-empty list is expected
    # rather than an error (restore_dropped_weights puts those tensors back); print, never raise.
    missing, unexpected = text_model.model.load_state_dict(language_model.state_dict(), strict=False)
    if hasattr(full, "lm_head") and hasattr(text_model, "lm_head"):
        text_model.lm_head.load_state_dict(full.lm_head.state_dict())
    print(f"text-only export: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    text_model.to(torch.bfloat16).save_pretrained(str(text_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(merged_dir)).save_pretrained(str(text_dir))
    return text_dir


def merge_adapter(run_dir: Path, base_model: str) -> Path:
    """Fold run_dir/adapter into the base model and save run_dir/merged with the adapter's
    tokenizer: the same merge train_lora does at the end of a run, for a run whose adapter was
    trained elsewhere (only the adapter travels; the base comes from the hub)."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    import torch

    adapter_dir, merged_dir = run_dir / "adapter", run_dir / "merged"
    # bf16 explicitly: load_base_model's CPU default is float32, which would double the merged size
    base = load_base_model(base_model, dtype=torch.bfloat16)
    PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))
    return merged_dir


def restore_dropped_weights(run_dir: Path, base_model: str) -> int:
    """Put back tensors transformers drops on save but vLLM requires; returns how many.

    Gemma 4's KV-sharing layers carry k_proj/v_proj/k_norm in Google's checkpoint; the
    transformers implementation has no such parameters there, so a re-saved model lacks them
    and vLLM refuses to load. LoRA never touched those layers, so the originals are exact.
    Ported from benchmarks/modal/train_lora.py, which keeps its own copy.
    """
    import glob

    import huggingface_hub
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    target = run_dir / "merged-text" / "model.safetensors"
    if not target.exists():
        raise FileNotFoundError(f"{target} missing (a sharded export is not handled); run export_text_only first")
    exported = load_file(str(target))
    prefix = "model.language_model."
    added = []
    snapshot = huggingface_hub.snapshot_download(base_model, allow_patterns=["*.safetensors"])
    for shard in sorted(glob.glob(f"{snapshot}/*.safetensors")):
        with safe_open(shard, "pt") as st:
            for key in st.keys():
                if not key.startswith(prefix):
                    continue
                new_key = "model." + key[len(prefix):]
                if new_key not in exported and ".self_attn." in new_key:
                    exported[new_key] = st.get_tensor(key)
                    added.append(new_key)
    # Written beside the original and swapped in only once complete: a full disk mid-write
    # (it happened on a 60 GB volume) must leave the export intact, not a truncated file.
    temporary = target.with_name(target.name + ".restoring")
    try:
        save_file(exported, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"restored {len(added)} tensors, e.g. {added[:3]}")
    return len(added)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-length", type=int, default=8192); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--liger", action="store_true")
    a = ap.parse_args()
    print(json.dumps(train_lora(Path(a.data), Path(a.run), a.base_model, batch_size=a.batch_size,
                                grad_accum=a.grad_accum, max_length=a.max_length, merge=not a.no_merge, liger=a.liger), indent=2))
