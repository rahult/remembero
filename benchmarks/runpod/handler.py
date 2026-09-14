"""RunPod Serverless worker: train (or resume) one reader run from the network volume,
export it text-only, convert to GGUF and quantize, all on the volume. One job = one run.
Progress is reported per stage so `submit.py` can print it; the job result is metrics.json."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

VOLUME = Path("/runpod-volume")
STAGES = ("train", "export-text", "convert-f16", "quantize")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
DEFAULTS = {"base_model": "google/gemma-4-E4B-it", "max_length": 8192, "liger": True,
            "batch_size": 4, "grad_accum": 16, "quant": "Q8_0", "merge": True}


def job_config(inp: dict) -> dict:
    run = str(inp.get("run", ""))
    if not RUN_NAME.match(run):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    cfg = {**DEFAULTS, **{k: v for k, v in inp.items() if k in DEFAULTS}, "run": run}
    cfg["data_dir"] = str(VOLUME / "data" / run)
    cfg["run_dir"] = str(VOLUME / "runs" / run)
    return cfg


def run_job(cfg: dict, progress=lambda msg: None) -> dict:
    from benchmarks.train.reader_lora import export_text_only, train_lora

    data_dir, run_dir = Path(cfg["data_dir"]), Path(cfg["run_dir"])
    if not (data_dir / "conversations.jsonl").exists():
        raise FileNotFoundError(f"{data_dir}/conversations.jsonl is not on the volume")
    progress("train: starting")
    metrics = train_lora(data_dir, run_dir, cfg["base_model"], batch_size=cfg["batch_size"],
                         grad_accum=cfg["grad_accum"], max_length=cfg["max_length"],
                         merge=cfg["merge"], liger=cfg["liger"], on_save=lambda: progress("train: checkpoint saved"))
    progress("export-text")
    text_dir = export_text_only(run_dir)
    f16 = run_dir / f"{cfg['run']}-f16.gguf"
    quantized = run_dir / f"{cfg['run']}-{cfg['quant']}.gguf"
    progress("convert-f16")
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(text_dir), "--outtype", "f16", "--outfile", str(f16)], check=True)
    progress("quantize")
    subprocess.run(["/opt/llama.cpp/build/bin/llama-quantize", str(f16), str(quantized), cfg["quant"]], check=True)
    f16.unlink()
    metrics["gguf"] = str(quantized)
    metrics["gguf_bytes"] = quantized.stat().st_size
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def handler(job: dict) -> dict:
    import runpod

    cfg = job_config(job.get("input") or {})
    return run_job(cfg, progress=lambda msg: runpod.serverless.progress_update(job, msg))


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
