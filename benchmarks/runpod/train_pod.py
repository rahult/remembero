"""Train one reader run on a RunPod *pod*, from the pod itself:
  python3 -m benchmarks.runpod.train_pod --run reader-v7-gemma4-e4b [--root /workspace]
      [--local-root /root] [--max-length 8192] [--batch-size 4] [--grad-accum 16] [--quant Q8_0] [--no-liger]

Same pipeline as the serverless worker (handler.run_job: train -> export text-only -> GGUF ->
quantize), only the layout differs. On a pod the network volume is mounted at `--root`
(/workspace), not at /runpod-volume, and the container disk is the pod's own filesystem, so
handler.job_config's paths cannot be reused: `build_config` supplies `data_dir`,
`volume_run_dir` (both on the volume) and `run_dir` (`--local-root`, the container disk, where
every heavy intermediate is written) explicitly.

Progress lines go to stdout as they happen, so `pod.py`'s start command can tee them to
runs/<run>/train.log on the volume and `pod.py train-log` can follow them without SSH. The last
line is `TRAIN_DONE <metrics json>` or, on any failure, `TRAIN_FAILED <error>` with a non-zero exit."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from benchmarks.runpod.handler import DEFAULTS, QUANTS, RUN_NAME, run_job

ROOT = "/workspace"          # the network volume, mounted here on a pod
LOCAL_ROOT = "/root"         # the container disk: the trainer's working dir, merged/, the f16 GGUF


def build_config(run: str, *, root: str = ROOT, local_root: str = LOCAL_ROOT, **overrides) -> dict:
    """The cfg `handler.run_job` expects, for the pod layout. Pure: it touches no filesystem."""
    if not RUN_NAME.match(str(run)):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; expected {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}, "run": run}
    if cfg["quant"] not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}, got {cfg['quant']!r}")
    cfg["data_dir"] = f"{root}/data/{run}"
    cfg["volume_run_dir"] = f"{root}/runs/{run}"
    cfg["run_dir"] = f"{local_root}/runs/{run}"
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=ROOT, help="the network volume's mount point on the pod")
    ap.add_argument("--local-root", default=LOCAL_ROOT, help="the container disk, for everything disposable")
    ap.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--quant", choices=QUANTS, default=DEFAULTS["quant"])
    ap.add_argument("--no-liger", action="store_true")
    a = ap.parse_args(argv)

    def say(message: str) -> None:
        print(message, flush=True)

    try:
        cfg = build_config(a.run, root=a.root, local_root=a.local_root, max_length=a.max_length,
                           batch_size=a.batch_size, grad_accum=a.grad_accum, quant=a.quant,
                           liger=not a.no_liger)
    except ValueError as error:
        say(f"TRAIN_FAILED {error}")
        raise SystemExit(2)
    say(f"train_pod {json.dumps(cfg, sort_keys=True)}")
    try:
        metrics = run_job(cfg, progress=say)
        # run_job already writes metrics.json; writing it here too keeps the promise of this CLI
        # even if that ever stops being true, and costs one small write.
        volume_run_dir = Path(cfg["volume_run_dir"])
        volume_run_dir.mkdir(parents=True, exist_ok=True)
        (volume_run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as error:  # noqa: BLE001 - every failure has to reach the log as one marker line
        traceback.print_exc(file=sys.stdout)
        say(f"TRAIN_FAILED {type(error).__name__}: {error}")
        raise SystemExit(1)
    say(f"TRAIN_DONE {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
