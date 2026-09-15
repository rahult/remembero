"""Rebuild a reader's servable weights on the pod, from its LoRA adapter on the network volume.
  python3 -m benchmarks.runpod.prepare_reader --root /workspace --run reader-v4-gemma4-e4b [--base-model google/gemma-4-E4B-it]
Gemma 4 needs three steps before vLLM loads a fine-tune: merge the adapter into the base, export
the text-only causal LM, and put back the KV-sharing tensors transformers drops on save. The result
is <root>/runs/<run>/merged-text with a .prepared marker; a second call finds the marker and exits
at once, so the pod's start command can run this on every boot. The multimodal merged/ directory
(~16 GB) is deleted afterwards to save volume space; merged-text/fingerprint.json
(see fingerprint.py) is written just before the marker. HF_HOME defaults to <root>/hf, where the
public base model is cached after the first download."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from benchmarks.runpod.fingerprint import fingerprint
from benchmarks.train import reader_lora

MARKER = ".prepared"
# what AutoTokenizer.save_pretrained writes; the adapter directory has them from training
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja", "chat_template.json")


def is_prepared(run_dir: Path) -> bool:
    text_dir = run_dir / "merged-text"
    return (text_dir / "config.json").exists() and (text_dir / MARKER).exists()


def copy_tokenizer_files(adapter_dir: Path, text_dir: Path) -> list[str]:
    """Tokenizer files the export did not write, taken from the adapter (never overwriting)."""
    copied = []
    for name in TOKENIZER_FILES:
        source, target = adapter_dir / name, text_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def has_chat_template(text_dir: Path) -> bool:
    """vLLM's chat endpoint needs the template the reader was trained under; without it the
    server starts but renders prompts differently, which would silently change every score."""
    if (text_dir / "chat_template.jinja").exists() or (text_dir / "chat_template.json").exists():
        return True
    config = text_dir / "tokenizer_config.json"
    return config.exists() and bool(json.loads(config.read_text()).get("chat_template"))


def prepare(root: Path, run: str, base_model: str) -> Path:
    run_dir = root / "runs" / run
    text_dir = run_dir / "merged-text"
    if is_prepared(run_dir):
        print(f"{text_dir} already prepared")
        return text_dir
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}; upload it with volume.py put first")
    # a boot that died mid-prepare leaves an unmarked merged-text; start it clean so no stale
    # shard or half-written file from that attempt can end up next to the new weights
    shutil.rmtree(text_dir, ignore_errors=True)
    started = time.time()
    merged = reader_lora.merge_adapter(run_dir, base_model)
    print(f"merged into {merged} ({time.time() - started:.0f} s)")
    exported = reader_lora.export_text_only(run_dir)
    if exported != text_dir:
        raise SystemExit(f"export wrote {exported}, expected {text_dir}; is {base_model} a Gemma 4 checkpoint?")
    print(f"text-only export at {text_dir} ({time.time() - started:.0f} s)")
    restored = reader_lora.restore_dropped_weights(run_dir, base_model)
    print(f"restored {restored} dropped tensors ({time.time() - started:.0f} s)")
    print(f"tokenizer files copied from the adapter: {copy_tokenizer_files(adapter_dir, text_dir)}")
    if not has_chat_template(text_dir):
        raise SystemExit(f"{text_dir} has no chat template; refusing to mark it prepared")
    fp = fingerprint(text_dir)
    (text_dir / "fingerprint.json").write_text(json.dumps(fp, indent=2) + "\n")
    print(f"fingerprint: {fp['tensor_count']} tensors, {fp['parameter_count']} parameters, keys {fp['keys_sha256'][:12]}")
    (text_dir / MARKER).write_text(json.dumps({"base_model": base_model, "restored_tensors": restored,
                                               "seconds": round(time.time() - started, 1)}) + "\n")
    shutil.rmtree(run_dir / "merged", ignore_errors=True)
    return text_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace"); ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(a.root) / "hf"))
    prepare(Path(a.root), a.run, a.base_model)
