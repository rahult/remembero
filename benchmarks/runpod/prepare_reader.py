"""Rebuild a reader's servable weights on the pod, from its LoRA adapter on the network volume.
  python3 -m benchmarks.runpod.prepare_reader --root /workspace --run reader-v4-gemma4-e4b [--base-model google/gemma-4-E4B-it]
Gemma 4 needs three steps before vLLM loads a fine-tune: merge the adapter into the base, export
the text-only causal LM, and put back the KV-sharing tensors transformers drops on save. The result
is <root>/runs/<run>/merged-text with a .prepared marker; a second call finds the marker and exits
at once, so the pod's start command can run this on every boot. The multimodal merged/ directory
(~16 GB) is deleted as soon as the export succeeds, before the restore rewrites the weights;
merged-text/fingerprint.json (see fingerprint.py) is written just before the marker. Before any
model is loaded the volume's free space is checked against two copies of the base weights (plus
the HF cache when it is not there yet; see space_needed), so a full volume fails in seconds, not after a merge. HF_HOME defaults to <root>/hf, where the
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
# written into merged/ once merge_adapter returns, so a later boot can tell a finished merge from a torn one
MERGED_MARKER = ".merged"
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


def base_model_size(base_model: str) -> tuple[int, bool]:
    """Bytes of the base model's safetensors, and whether they are already in the HF cache.
    Read from the cache when present, else from the hub's file metadata (no download)."""
    import huggingface_hub

    try:
        snapshot = huggingface_hub.snapshot_download(base_model, allow_patterns=["*.safetensors"], local_files_only=True)
        shards = list(Path(snapshot).glob("*.safetensors"))
        if shards:
            return sum(shard.stat().st_size for shard in shards), True
    except Exception:
        pass
    info = huggingface_hub.HfApi().model_info(base_model, files_metadata=True)
    return sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors")), False


def space_needed(base_bytes: int, *, cached: bool, merged_on_disk: int) -> int:
    """Free space the rebuild needs: two more copies of the base's weights, less a finished merged/
    already written, plus the HF cache if it is not there yet.

    With merged/ deleted before the restore, at most three weight-sized files sit on the volume at
    once, the HF cache being one of them: cache + merged/ + merged-text/ during the export, then
    cache + merged-text/ + the restore's temporary file. (merged-text/ is ~0.94 x the base, so this
    leaves ~2 GB of slack for Gemma 4 E4B.) A cached base already counts against used space, not free."""
    return 2 * base_bytes - merged_on_disk + (0 if cached else base_bytes)


def directory_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def check_free_space(root: Path, base_model: str, merged_on_disk: int) -> None:
    """Fail before any GPU time is spent if the volume cannot hold the rebuild."""
    free = shutil.disk_usage(root).free
    try:
        base_bytes, cached = base_model_size(base_model)
    except Exception as error:  # the hub unreachable: say so, and let the rebuild try
        print(f"disk: {free / 1e9:.1f} GB free on {root}; could not size {base_model} ({str(error)[:120]}), not checking")
        return
    need = space_needed(base_bytes, cached=cached, merged_on_disk=merged_on_disk)
    detail = (f"2 x {base_bytes / 1e9:.1f} GB of base weights" + ("" if cached else " plus the Hugging Face cache")
              + (f", less {merged_on_disk / 1e9:.1f} GB already merged" if merged_on_disk else ""))
    print(f"disk: {free / 1e9:.1f} GB free on {root}, about {need / 1e9:.1f} GB needed ({detail})")
    if free < need:
        raise SystemExit(f"not enough space on {root}: {free / 1e9:.1f} GB free, about {need / 1e9:.1f} GB needed "
                         f"({detail}); grow the volume or clear old runs before preparing {base_model}")


def prepare(root: Path, run: str, base_model: str) -> Path:
    run_dir = root / "runs" / run
    text_dir, merged_dir = run_dir / "merged-text", run_dir / "merged"
    if is_prepared(run_dir):
        print(f"{text_dir} already prepared")
        return text_dir
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}; upload it with volume.py put first")
    # A boot that died mid-prepare leaves an unmarked merged-text: start it clean so no stale
    # shard or half-written file can end up next to the new weights. A merged/ carrying its
    # completion marker is kept (only export and restore run again); any other merged/ is redone.
    shutil.rmtree(text_dir, ignore_errors=True)
    reuse_merge = (merged_dir / MERGED_MARKER).exists()
    if merged_dir.exists() and not reuse_merge:
        shutil.rmtree(merged_dir)
    check_free_space(root, base_model, directory_bytes(merged_dir) if reuse_merge else 0)
    started = time.time()
    if reuse_merge:
        print(f"reusing the finished merge in {merged_dir}")
    else:
        merged = reader_lora.merge_adapter(run_dir, base_model)
        (merged_dir / MERGED_MARKER).write_text("")
        print(f"merged into {merged} ({time.time() - started:.0f} s)")
    exported = reader_lora.export_text_only(run_dir)
    if exported != text_dir:
        raise SystemExit(f"export wrote {exported}, expected {text_dir}; is {base_model} a Gemma 4 checkpoint?")
    print(f"text-only export at {text_dir} ({time.time() - started:.0f} s)")
    # merged/ has served its purpose; free its ~16 GB before the restore writes a second copy of
    # merged-text's weights (a 60 GB volume ran out of quota here with all three on it)
    shutil.rmtree(merged_dir, ignore_errors=True)
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
    return text_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace"); ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(a.root) / "hf"))
    prepare(Path(a.root), a.run, a.base_model)
