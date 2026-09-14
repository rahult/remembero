"""RunPod Serverless worker: train (or resume) one reader run from the network volume,
export it text-only, convert to GGUF and quantize, all on the volume. One job = one run.
Progress is reported per stage so `submit.py` can print it; the job result is metrics.json.

The volume is small (40 GB) and holds only what has to outlive the worker: the HF cache, the
input data, the newest trainer checkpoint and the outputs (GGUF, adapter, metrics.json). Every
heavy intermediate — the trainer's working directory, `merged/`, `merged-text/`, the f16 GGUF —
lives on the worker's container disk under /workspace and dies with the job."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

VOLUME = Path("/runpod-volume")
WORKSPACE = Path("/workspace")
STAGES = ("train", "export-text", "convert-f16", "quantize")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "F16")
DEFAULTS = {"base_model": "google/gemma-4-E4B-it", "max_length": 8192, "liger": True,
            "batch_size": 4, "grad_accum": 16, "quant": "Q8_0"}


def job_config(inp: dict) -> dict:
    run = str(inp.get("run", ""))
    if not RUN_NAME.match(run):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    cfg = {**DEFAULTS, **{k: v for k, v in inp.items() if k in DEFAULTS}, "run": run}
    if cfg["quant"] not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}, got {cfg['quant']!r}")
    cfg["data_dir"] = str(VOLUME / "data" / run)
    cfg["run_dir"] = str(WORKSPACE / "runs" / run)
    cfg["volume_run_dir"] = str(VOLUME / "runs" / run)
    return cfg


def latest_checkpoint_dir(trainer_dir: Path) -> Path | None:
    """The highest-numbered checkpoint-N directory under `trainer_dir`, or None."""
    if not trainer_dir.is_dir():
        return None
    checkpoints = [d for d in trainer_dir.glob("checkpoint-*") if d.is_dir() and d.name.split("-")[-1].isdigit()]
    return max(checkpoints, key=lambda d: int(d.name.split("-")[-1])) if checkpoints else None


def stage_resume_checkpoint(volume_run_dir: Path, run_dir: Path) -> Path | None:
    """Copy the newest checkpoint kept on the volume onto the container disk, where training runs.

    Returns the local checkpoint directory, or None when there is nothing to resume from."""
    source = latest_checkpoint_dir(volume_run_dir / "trainer")
    if source is None:
        return None
    target = run_dir / "trainer" / source.name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return target


def persist_checkpoint(run_dir: Path, volume_run_dir: Path) -> Path | None:
    """Copy the newest local checkpoint to the volume, keeping only that one there."""
    source = latest_checkpoint_dir(run_dir / "trainer")
    if source is None:
        return None
    trainer = volume_run_dir / "trainer"
    trainer.mkdir(parents=True, exist_ok=True)
    for old in trainer.glob("checkpoint-*"):
        if old.is_dir() and old.name != source.name:
            shutil.rmtree(old, ignore_errors=True)
    target = trainer / source.name
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def run_job(cfg: dict, progress=lambda msg: None) -> dict:
    from benchmarks.train.reader_lora import export_text_only, train_lora

    data_dir = Path(cfg["data_dir"])
    run_dir, volume_run_dir = Path(cfg["run_dir"]), Path(cfg["volume_run_dir"])
    if not (data_dir / "conversations.jsonl").exists():
        raise FileNotFoundError(f"{data_dir}/conversations.jsonl is not on the volume")
    run_dir.mkdir(parents=True, exist_ok=True)
    volume_run_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_resume_checkpoint(volume_run_dir, run_dir)
    if staged is not None:
        progress(f"train: resuming from {staged.name}")

    def on_save() -> None:
        persist_checkpoint(run_dir, volume_run_dir)
        progress("train: checkpoint saved")

    progress("train: starting")
    metrics = train_lora(data_dir, run_dir, cfg["base_model"], batch_size=cfg["batch_size"],
                         grad_accum=cfg["grad_accum"], max_length=cfg["max_length"],
                         merge=True, liger=cfg["liger"], on_save=on_save)
    progress("export-text")
    text_dir = export_text_only(run_dir)
    merged = run_dir / "merged"
    # The text-only export is a second full copy of the weights; the merged checkpoint is dead
    # weight on a container disk that still has to hold the f16 GGUF. (export_text_only returns
    # `merged` itself for a text-only base model — then there is nothing to delete.)
    if text_dir != merged and merged.exists():
        shutil.rmtree(merged)
        metrics["merged_dir"] = None
    f16 = run_dir / f"{cfg['run']}-f16.gguf"
    quantized = run_dir / f"{cfg['run']}-{cfg['quant']}.gguf"
    progress("convert-f16")
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(text_dir), "--outtype", "f16", "--outfile", str(f16)], check=True)
    progress("quantize")
    subprocess.run(["/opt/llama.cpp/build/bin/llama-quantize", str(f16), str(quantized), cfg["quant"]], check=True)
    f16.unlink()
    volume_gguf = volume_run_dir / f"{cfg['run']}-{cfg['quant']}.gguf"
    shutil.copy2(quantized, volume_gguf)
    adapter = run_dir / "adapter"
    if adapter.is_dir():
        shutil.copytree(adapter, volume_run_dir / "adapter", dirs_exist_ok=True)
    # The volume path, not the container one: it is what `volume.py get` fetches.
    metrics["gguf"] = str(volume_gguf)
    metrics["gguf_bytes"] = quantized.stat().st_size
    written = json.dumps(metrics, indent=2)
    (run_dir / "metrics.json").write_text(written)
    (volume_run_dir / "metrics.json").write_text(written)
    return metrics


def handler(job: dict) -> dict:
    import runpod

    cfg = job_config(job.get("input") or {})
    return run_job(cfg, progress=lambda msg: runpod.serverless.progress_update(job, msg))


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
