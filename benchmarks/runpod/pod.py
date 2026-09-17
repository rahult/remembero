"""Launch and manage the pod that serves a reader with vLLM, for the harness on the Mac.
  python benchmarks/runpod/pod.py create-serve --run reader-v4-gemma4-e4b --served-name rembero-reader-v4 [--max-hours 4] [--idle-minutes 45]
  python benchmarks/runpod/pod.py create-serve --run <a> --served-name <n> --run <b> --served-name <m>
  python benchmarks/runpod/pod.py wait [--pod <id>] [--served-name rembero-reader-v4] [--port 8001] [--no-stop-on-timeout]
  python benchmarks/runpod/pod.py stop|start|terminate [--pod <id>]
  python benchmarks/runpod/pod.py status [--pod <id>]
  python benchmarks/runpod/pod.py create-train --run reader-v7-gemma4-e4b [--gpu ...] [--max-length 8192] [--max-hours 8]
  python benchmarks/runpod/pod.py train-log --run reader-v7-gemma4-e4b [--tail 40]
The pod runs the official vLLM image on a community GPU with the network volume at /workspace:
its start command rebuilds the reader's merged weights from runs/<run>/adapter once
(prepare_reader.py) and then serves them. Two runs share one GPU: both are prepared first, then
the first is served on port 8000 and the second on 8001. Env or .env: RUNPOD_API_KEY,
RUNPOD_VOLUME_ID. create-serve writes VLLM_API_KEY (once), RUNPOD_SERVE_POD_ID, RUNPOD_SERVE_URL
and RUNPOD_SERVE_URL_2 (empty for one run) to .env; the other subcommands default --pod to
RUNPOD_SERVE_POD_ID. `wait` stops the pod when it times out (it needs RUNPOD_API_KEY for that) unless
--no-stop-on-timeout is given. No Hugging Face or Modal credential goes
to the pod: the base model is public and the adapter is already on the volume.

`create-train` is the other job this file does: when the serverless endpoint has no GPU capacity,
it trains a reader on a pod instead, on the public PyTorch image, with the same pipeline the
serverless worker runs (benchmarks.runpod.train_pod -> handler.run_job). It writes
RUNPOD_TRAIN_POD_ID to .env; `train-log` tails runs/<run>/train.log off the volume.

Every pod this file creates stops itself, so nothing a run leaves behind can keep billing: the
start commands end in `self_stop`, which logs why and then runs `runpodctl stop pod` from inside
the pod. A training pod stops when the run ends or after --max-hours (8); a serving pod stops on a
SERVE_FAILED, after --max-hours (4), or after --idle-minutes (45) with no request served. Check
`pod.py status` after a run anyway: a stop that fails leaves the pod parked for inspection."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
USER_AGENT = "rembero-runpod/1.0 (+https://github.com/rahult/remembero)"
# Newest stable vLLM release on 2026-09-15 (v0.29.0, pushed 2026-09-09). The -cu129 build rather
# than the default CUDA 13.0 one, so community hosts on a CUDA 12.9 driver qualify too; both carry
# kernels for the RTX 5090 (TORCH_CUDA_ARCH_LIST includes 12.0). Gemma 4 landed in vLLM in spring 2026.
IMAGE = "vllm/vllm-openai:v0.29.0-cu129"
CUDA_VERSIONS = ["12.9", "13.0"]
# network volumes may be Secure Cloud only; --cloud SECURE retries there without a code change
CLOUD_TYPES = ("COMMUNITY", "SECURE")
ROOT = "/workspace"
BASE_MODEL = "google/gemma-4-E4B-it"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
SERVED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,120}$")
# The Modal serve of record (benchmarks/modal/train_lora.py `serve`), flag for flag, except the
# context: thinking completions on 7k-token prompts overflow 8192.
SERVE_FLAGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "12288",
    "--max-num-seqs", "32",
    "--gpu-memory-utilization", "0.92",
    "--reasoning-parser", "gemma4",
    "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}, separators=(",", ":")),
]
# two readers side by side on one GPU each take this share of its memory
SHARED_GPU_MEMORY_UTILIZATION = "0.44"
PORTS = (8000, 8001)
# how long the start command waits for each server's /health before it parks
SERVE_HEALTH_TIMEOUT_S = 30 * 60
# two readers at 0.44 each need an 80 GB+ GPU: 0.44 of a 32 GB card is about the weights alone
LARGE_GPUS = ("NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB",
              "NVIDIA A100 80GB PCIe", "NVIDIA H200", "NVIDIA B200")
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# --- training on a pod (create-train) -------------------------------------------------------
# The image the serverless endpoint bootstraps from (benchmarks/runpod/README.md, "Bootstrap
# template"); confirmed on Docker Hub 2026-09-14. A pod runs it the same way, with the network
# volume at /workspace instead of /runpod-volume.
TRAIN_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
# The image is built for CUDA 12.8.1, so a 12.8 host qualifies as well as the serving pod's two.
TRAIN_CUDA_VERSIONS = ["12.8", "12.9", "13.0"]
# merged/, merged-text/ (~14 GB each) and the f16 GGUF (~9 GB) are built on the container disk
TRAIN_CONTAINER_DISK_GB = 80
TRAIN_GPU = "NVIDIA H100 80GB HBM3"
# the training volume lives in US-GA-2 (80 GB stock and an S3 endpoint both)
TRAIN_DATACENTER = "US-GA-2"
TRAIN_LOCAL_ROOT = "/root"
QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "F16")
# mirrors handler.DEFAULTS, which pod.py cannot import (it runs as a script, off sys.path);
# test_pod.py keeps the two in step
TRAIN_DEFAULTS = {"max_length": 8192, "batch_size": 4, "grad_accum": 16, "quant": "Q8_0"}
# The serverless bootstrap of record, copied verbatim from README.md "Bootstrap template" (a test
# asserts it is still a substring of that line): the pip installs and the llama.cpp build the
# handler's convert/quantize stages need. Everything around it — the exports, where the code is
# imported from, what is run — belongs to train_command.
TRAIN_BOOTSTRAP = (
    'pip install -q "transformers>=5.0,<6" "trl>=0.24" "peft>=0.17" "datasets>=3.0" "accelerate>=1.0" '
    '"liger-kernel>=0.8.2" sentencepiece protobuf runpod gguf; '
    'if [ ! -x /opt/llama.cpp/build/bin/llama-quantize ]; then apt-get update -qq && '
    'apt-get install -y -qq --no-install-recommends git cmake build-essential >/dev/null && '
    'git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp && '
    'cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null && '
    'cmake --build /opt/llama.cpp/build --target llama-quantize -j "$(nproc)" >/dev/null; fi'
)


# --- stopping the pod from inside (self_stop and the watchdogs) -----------------------------
# A pod stops itself with runpodctl, which inside a pod is authenticated by the pod itself: no
# RUNPOD_API_KEY ever goes into a pod's environment, and RUNPOD_POD_ID is already there. Neither
# image ships runpodctl, so self_stop installs it on demand from the release of record.
RUNPODCTL_INSTALL = ("wget -qO /usr/local/bin/runpodctl "
                     "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64"
                     " && chmod +x /usr/local/bin/runpodctl")
# how often each watchdog looks at the clock (and at serve.log's mtime)
WATCHDOG_INTERVAL_S = 60
# create-serve: the pod stops itself after this long whatever the servers are doing ...
SERVE_MAX_HOURS = 4
# ... and after this long without a request
SERVE_IDLE_MINUTES = 45
# create-train: a hung training run cannot bill longer than this
TRAIN_MAX_HOURS = 8


def _self_stop(*logs: str) -> str:
    """`self_stop <reason>`, the shell function both start commands park through.

    It writes `SELF_STOP: <reason>` to each of `logs` (shell expressions, already quoted) on the
    network volume first, so `pod.py train-log` and `volume.py get` still show why the pod went
    away; then it stops the pod from inside with `runpodctl stop pod "$RUNPOD_POD_ID"`, installing
    runpodctl first if the image has none. If the stop fails anyway, `SELF_STOP_FAILED: <reason>`
    goes to the same logs and the pod falls back to `exec sleep infinity`, the park of old, so it
    can still be inspected — an exited container is restarted by RunPod into a billed loop.
    The park is also what runs while RunPod acts on a *successful* stop."""
    def mark(marker: str) -> str:
        return " ".join(f'echo "{marker}: $1" | tee -a {log};' for log in logs)

    return ("self_stop() { " + mark("SELF_STOP")
            + f" command -v runpodctl >/dev/null 2>&1 || {{ {RUNPODCTL_INSTALL}; }};"
            + ' runpodctl stop pod "$RUNPOD_POD_ID" || { ' + mark("SELF_STOP_FAILED") + " };"
            + " exec sleep infinity; }")


def _watchdog(body: str, pid_var: str) -> str:
    """A background loop that ticks every WATCHDOG_INTERVAL_S and gives up when the start command
    it belongs to is gone (`$$` is the pod's main process, which the park keeps alive)."""
    return (f"( while sleep {int(WATCHDOG_INTERVAL_S)}; do"
            ' kill -0 "$$" 2>/dev/null || exit 0;'
            f" {body} done ) & {pid_var}=$!")


def _max_hours_watchdog(hours: float) -> str:
    """Stop the pod `hours` after boot, whatever it is doing: the cap on what one pod can bill."""
    return (f"MAX_DEADLINE=$(( $(date +%s) + {int(float(hours) * 3600)} )); "
            + _watchdog('[ "$(date +%s)" -lt "$MAX_DEADLINE" ] ||'
                        f' self_stop "max hours reached ({hours} h)";', "MAX_WATCHDOG_PID"))


def _idle_watchdog(minutes: float, *logs: str) -> str:
    """Stop the pod when no request has been served for `minutes`.

    Idleness is read off vLLM's own access logging: **it depends on vLLM logging a line for each
    request**, which moves that serve.log's mtime, so the newest mtime of the pod's serve.logs is
    the last time a request was served. IDLE_SINCE starts at boot, so the wait for the first
    request is counted from there rather than from a log that does not exist yet."""
    # -c is GNU stat (both images are Linux); -f is the BSD spelling, so this also runs on a Mac
    newest = " ".join(f'MTIME=$(stat -c %Y {log} 2>/dev/null || stat -f %m {log} 2>/dev/null || echo 0);'
                      ' [ "$MTIME" -gt "$IDLE_SINCE" ] && IDLE_SINCE="$MTIME";' for log in logs)
    return ("IDLE_SINCE=$(date +%s); "
            + _watchdog(f'{newest} [ "$(( $(date +%s) - IDLE_SINCE ))" -lt {int(float(minutes) * 60)} ]'
                        f' || self_stop "idle for {minutes} minutes";', "IDLE_WATCHDOG_PID"))


def serve_url(pod_id: str, port: int = PORTS[0]) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net/v1"


def _vllm(run_dir: str, served_name: str, port: int, gpu_memory_utilization: str | None = None) -> list[str]:
    flags = list(SERVE_FLAGS)
    if gpu_memory_utilization:
        flags[flags.index("--gpu-memory-utilization") + 1] = gpu_memory_utilization
    return ["vllm", "serve", f"{run_dir}/merged-text", *flags, "--served-model-name", served_name,
            "--host", "0.0.0.0", "--port", str(port)]


def _prepare_steps(run: str, *, root: str, base_model: str, install: bool) -> str:
    prepare = ["python3", "-m", "benchmarks.runpod.prepare_reader", "--root", root, "--run", run, "--base-model", base_model]
    return " && ".join([
        'echo "=== boot $(date -u +%Y-%m-%dT%H:%M:%SZ)"',
        f"cd {shlex.quote(f'{root}/code')}",
        # uv respects the image's /etc/uv-overrides.txt pins; plain pip is the fallback
        *(['(uv pip install --system -q "peft>=0.17" || python3 -m pip install -q "peft>=0.17")'] if install else []),
        shlex.join(prepare),
    ])


def _health_wait(run: str, port: int, pid_var: str, *, watch: tuple[tuple[str, str], ...] = ()) -> list[str]:
    """Wait for the server in $pid_var to answer /health on `port`, within SERVE_HEALTH_TIMEOUT_S.
    It parks through serve_fail when that server exits first, when a server in `watch` (run, pid
    variable) exits meanwhile, or at the deadline."""
    health = f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=5)"
    timeout = int(SERVE_HEALTH_TIMEOUT_S)
    watched = "".join(f' kill -0 "${var}" 2>/dev/null || {{ wait "${var}"; serve_fail "{name} exited with $?"; }};'
                      for name, var in watch)
    return [
        f"SERVE_DEADLINE=$((SECONDS + {timeout}))",
        f'until python3 -c {shlex.quote(health)} >/dev/null 2>&1; do'
        f' kill -0 "${pid_var}" 2>/dev/null || {{ wait "${pid_var}"; serve_fail "{run} exited with $? before it was healthy on {port}"; }};'
        f"{watched}"
        f' [ "$SECONDS" -lt "$SERVE_DEADLINE" ] || serve_fail "{run} was not healthy on {port} within {timeout} s";'
        " sleep 5; done",
    ]


def serve_command(run: str, served_name: str, *, second: tuple[str, str] | None = None,
                  root: str = ROOT, base_model: str = BASE_MODEL,
                  max_hours: float = SERVE_MAX_HOURS, idle_minutes: float = SERVE_IDLE_MINUTES) -> str:
    """The pod's whole start command, for `bash -lc`: install PEFT (the image has transformers but
    not PEFT), rebuild the weights once, then serve with vLLM without ever letting the container exit.

    - Prepare runs from <root>/code: the image's WORKDIR /vllm-workspace holds vLLM's own
      `benchmarks` package, which `python3 -m benchmarks...` would import from there instead.
    - Prepare's output goes to runs/<run>/prepare.log and vLLM's to runs/<run>/serve.log, on the
      volume, so both can be fetched with volume.py get (the image has no SSH).
    - A failed prepare (or an empty key) writes runs/<run>/PREPARE_FAILED and stops the pod through
      `self_stop`: an exited container is restarted by RunPod and would redo the merge.
    - vLLM runs in the background. If it is not healthy within SERVE_HEALTH_TIMEOUT_S, or it exits
      (before healthy or later), `serve_fail` writes "SERVE_FAILED: <reason>" to serve.log, stops
      the server if it is still up and stops the pod, as the two-run command does.
    - Two watchdogs run alongside the servers, so a pod nobody is using cannot keep billing:
      one stops it `max_hours` after boot, the other after `idle_minutes` without a request.
    - The API key is read from the pod's env, never written into the command.

    `second=(run, served_name)` serves a second reader on the same GPU: see _two_run_command."""
    for name, served in [(run, served_name), *([second] if second else [])]:
        if not RUN_NAME.match(name):
            raise ValueError(f"run must match {RUN_NAME.pattern}, got {name!r}")
        if not SERVED_NAME.match(served):
            raise ValueError(f"served name must match {SERVED_NAME.pattern}, got {served!r}")
    if second:
        if second[0] == run or second[1] == served_name:
            raise ValueError(f"the two readers need different runs and served names, got {run!r}/{served_name!r} twice")
        return _two_run_command((run, served_name), second, root=root, base_model=base_model,
                                max_hours=max_hours, idle_minutes=idle_minutes)
    run_dir = f"{root}/runs/{run}"
    vllm = _vllm(run_dir, served_name, PORTS[0])
    steps = _prepare_steps(run, root=root, base_model=base_model, install=True)
    return "; ".join([
        "set -o pipefail",
        "export PYTHONUNBUFFERED=1",
        f"RUN_DIR={shlex.quote(run_dir)}",
        'mkdir -p "$RUN_DIR"',
        'rm -f "$RUN_DIR/PREPARE_FAILED"',
        _self_stop('"$RUN_DIR/serve.log"'),
        'fail() { echo "$1" | tee -a "$RUN_DIR/prepare.log"; touch "$RUN_DIR/PREPARE_FAILED"; self_stop "$1"; }',
        'test -n "$VLLM_API_KEY" || fail "VLLM_API_KEY is empty; refusing to serve on a public URL without a key"',
        f'{{ {steps}; }} 2>&1 | tee -a "$RUN_DIR/prepare.log" || fail "prepare failed; see $RUN_DIR/prepare.log"',
        # serve_fail <reason>: SERVE_FAILED to serve.log, stop the server if it is still up, stop the pod
        'serve_fail() { echo "SERVE_FAILED: $1" | tee -a "$RUN_DIR/serve.log"; kill "$SERVE_PID" 2>/dev/null; self_stop "$1"; }',
        f'{shlex.join(vllm)} --api-key "$VLLM_API_KEY" > >(tee -a "$RUN_DIR/serve.log") 2>&1 & SERVE_PID=$!',
        _max_hours_watchdog(max_hours),
        _idle_watchdog(idle_minutes, '"$RUN_DIR/serve.log"'),
        *_health_wait(run, PORTS[0], "SERVE_PID"),
        f'wait "$SERVE_PID"; serve_fail "{run} exited with $?"',
    ])


def _two_run_command(first: tuple[str, str], second: tuple[str, str], *, root: str, base_model: str,
                     max_hours: float = SERVE_MAX_HOURS, idle_minutes: float = SERVE_IDLE_MINUTES) -> str:
    """Two readers on one GPU, each with SHARED_GPU_MEMORY_UTILIZATION of its memory.

    - Both runs are prepared, one after the other, before either server starts, so the two merges
      never hold host memory at once. Each prepare logs to its own run's prepare.log; a failure
      marks that run PREPARE_FAILED and parks the pod (an empty key marks both).
    - The first vLLM starts on 8000 and the second on 8001 only once the first answers /health, so
      the two never profile GPU memory at the same moment; each logs to its own run's serve.log.
    - The container never exits, because RunPod would restart it into a billed loop. If either
      server is not healthy within SERVE_HEALTH_TIMEOUT_S of its start, or either server exits
      (before healthy or later), `serve_fail` writes "SERVE_FAILED: <reason>" to both runs' serve.log, stops whatever
      server is still up and then stops the pod through `self_stop`, like PREPARE_FAILED.
    - The two watchdogs cover both servers: the idle one takes the newer of the two serve.logs."""
    (run_1, name_1), (run_2, name_2) = first, second
    dir_1, dir_2 = f"{root}/runs/{run_1}", f"{root}/runs/{run_2}"
    vllm_1 = _vllm(dir_1, name_1, PORTS[0], SHARED_GPU_MEMORY_UTILIZATION)
    vllm_2 = _vllm(dir_2, name_2, PORTS[1], SHARED_GPU_MEMORY_UTILIZATION)
    steps_1 = _prepare_steps(run_1, root=root, base_model=base_model, install=True)
    steps_2 = _prepare_steps(run_2, root=root, base_model=base_model, install=False)
    return "; ".join([
        "set -o pipefail",
        "export PYTHONUNBUFFERED=1",
        f"RUN_DIR={shlex.quote(dir_1)}",
        f"RUN_DIR_2={shlex.quote(dir_2)}",
        'mkdir -p "$RUN_DIR" "$RUN_DIR_2"',
        'rm -f "$RUN_DIR/PREPARE_FAILED" "$RUN_DIR_2/PREPARE_FAILED"',
        _self_stop('"$RUN_DIR/serve.log"', '"$RUN_DIR_2/serve.log"'),
        # fail <message> <run dir>...: log and mark each named run, then stop the pod
        'fail() { local message="$1" dir; shift; for dir in "$@"; do echo "$message" | tee -a "$dir/prepare.log";'
        ' touch "$dir/PREPARE_FAILED"; done; self_stop "$message"; }',
        'test -n "$VLLM_API_KEY" || fail "VLLM_API_KEY is empty; refusing to serve on a public URL without a key" "$RUN_DIR" "$RUN_DIR_2"',
        f'{{ {steps_1}; }} 2>&1 | tee -a "$RUN_DIR/prepare.log" || fail "prepare failed; see $RUN_DIR/prepare.log" "$RUN_DIR"',
        f'{{ {steps_2}; }} 2>&1 | tee -a "$RUN_DIR_2/prepare.log" || fail "prepare failed; see $RUN_DIR_2/prepare.log" "$RUN_DIR_2"',
        # serve_fail <reason>: SERVE_FAILED to both runs' serve.log, stop any server still up, stop the pod
        'serve_fail() { local dir pid; for dir in "$RUN_DIR" "$RUN_DIR_2"; do echo "SERVE_FAILED: $1" | tee -a "$dir/serve.log"; done;'
        ' for pid in $SERVE_PID $SERVE_PID_2; do kill "$pid" 2>/dev/null; done; self_stop "$1"; }',
        f'{shlex.join(vllm_1)} --api-key "$VLLM_API_KEY" > >(tee -a "$RUN_DIR/serve.log") 2>&1 & SERVE_PID=$!',
        _max_hours_watchdog(max_hours),
        _idle_watchdog(idle_minutes, '"$RUN_DIR/serve.log"', '"$RUN_DIR_2/serve.log"'),
        *_health_wait(run_1, PORTS[0], "SERVE_PID"),
        f'{shlex.join(vllm_2)} --api-key "$VLLM_API_KEY" > >(tee -a "$RUN_DIR_2/serve.log") 2>&1 & SERVE_PID_2=$!',
        *_health_wait(run_2, PORTS[1], "SERVE_PID_2", watch=((run_1, "SERVE_PID"),)),
        'while kill -0 "$SERVE_PID" 2>/dev/null && kill -0 "$SERVE_PID_2" 2>/dev/null; do sleep 30; done',
        f'kill -0 "$SERVE_PID" 2>/dev/null || {{ wait "$SERVE_PID"; serve_fail "{run_1} exited with $?"; }}',
        f'wait "$SERVE_PID_2"; serve_fail "{run_2} exited with $?"',
    ])


def train_command(run: str, *, root: str = ROOT, local_root: str = TRAIN_LOCAL_ROOT,
                  max_length: int = TRAIN_DEFAULTS["max_length"],
                  batch_size: int = TRAIN_DEFAULTS["batch_size"],
                  grad_accum: int = TRAIN_DEFAULTS["grad_accum"],
                  quant: str = TRAIN_DEFAULTS["quant"], liger: bool = True,
                  max_hours: float = TRAIN_MAX_HOURS) -> str:
    """The training pod's whole start command, for `bash -lc`: make the run directory on the
    volume, bootstrap the image, train, then stop the pod.

    - Every line of both stages is tee'd to <root>/runs/<run>/train.log **on the volume**, so
      `pod.py train-log` can follow the run without SSH and the log outlives the pod.
    - The bootstrap runs in a subshell under `set -e` so that a failed pip or llama.cpp build
      stops there instead of training against half an environment.
    - Training runs from <root>/code (PYTHONPATH is set to the same place) as
      `python3 -u -m benchmarks.runpod.train_pod`, which supplies handler.run_job with the pod's
      paths: data and outputs on the volume, everything heavy on the container disk.
    - Whatever the outcome, the pod stops itself through `self_stop` after the TRAIN_FAILED or
      TRAIN_DONE line, so a finished run cannot keep billing. It never simply exits: an exited
      container is restarted by RunPod, which would redo the training.
    - A watchdog stops the pod `max_hours` after boot, so a hung run cannot bill forever either."""
    if not RUN_NAME.match(run):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}, got {quant!r}")
    run_dir = f"{root}/runs/{run}"
    train = ["python3", "-u", "-m", "benchmarks.runpod.train_pod", "--run", run,
             "--root", root, "--local-root", local_root, "--max-length", str(max_length),
             "--batch-size", str(batch_size), "--grad-accum", str(grad_accum), "--quant", quant,
             *([] if liger else ["--no-liger"])]
    return "; ".join([
        "set -o pipefail",
        "export PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive TOKENIZERS_PARALLELISM=false"
        " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"export PYTHONPATH={shlex.quote(f'{root}/code')} HF_HOME={shlex.quote(f'{root}/hf')}",
        f"RUN_DIR={shlex.quote(run_dir)}",
        'mkdir -p "$RUN_DIR"',
        _self_stop('"$RUN_DIR/train.log"'),
        _max_hours_watchdog(max_hours),
        # fail <reason>: TRAIN_FAILED to the log on the volume, then stop the pod instead of exiting
        'fail() { echo "TRAIN_FAILED: $1" | tee -a "$RUN_DIR/train.log"; self_stop "$1"; }',
        f'( set -e; {TRAIN_BOOTSTRAP} ) 2>&1 | tee -a "$RUN_DIR/train.log"'
        ' || fail "bootstrap failed; see $RUN_DIR/train.log"',
        f'{{ cd {shlex.quote(f"{root}/code")} && {shlex.join(train)}; }} 2>&1 | tee -a "$RUN_DIR/train.log"'
        ' || fail "training failed; see $RUN_DIR/train.log"',
        f'echo "TRAIN_DONE: {run}; the pod is stopping itself"'
        ' | tee -a "$RUN_DIR/train.log"',
        'self_stop "training finished"',
    ])


def build_pod_payload(*, run: str, served_name: str, api_key: str, volume_id: str,
                      gpu: str = "NVIDIA GeForce RTX 5090", datacenter: str = "EUR-NO-1",
                      public_key: str | None = None, image: str = IMAGE, min_ram_gb: int = 48,
                      cloud_type: str = "COMMUNITY", second: tuple[str, str] | None = None,
                      max_hours: float = SERVE_MAX_HOURS, idle_minutes: float = SERVE_IDLE_MINUTES) -> dict:
    """PodCreateInput for the serving pod. Pure: everything it sends is in its arguments.
    `second=(run, served_name)` adds a second reader on port 8001 (serve_command); `max_hours` and
    `idle_minutes` are the two watchdogs that stop the pod from inside."""
    if cloud_type not in CLOUD_TYPES:
        raise ValueError(f"cloud type must be one of {CLOUD_TYPES}, got {cloud_type!r}")
    env = {"VLLM_API_KEY": api_key, "HF_HOME": f"{ROOT}/hf", "PYTHONPATH": f"{ROOT}/code"}
    if public_key:
        env["PUBLIC_KEY"] = public_key
    ports = [f"{port}/http" for port in PORTS[:2 if second else 1]]
    return {
        "name": (f"rembero-serve-{run}-{second[0]}" if second else f"rembero-serve-{run}")[:60],
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "allowedCudaVersions": CUDA_VERSIONS,
        # the merge holds the 16 GB bf16 multimodal checkpoint in host memory, and the text-only
        # export builds a second (bf16) copy next to it, before vLLM starts
        "minRAMPerGPU": min_ram_gb,
        "dataCenterIds": [datacenter],
        "networkVolumeId": volume_id,
        "volumeMountPath": ROOT,
        "containerDiskInGb": 40,
        "ports": [*ports, "22/tcp"],
        "env": env,
        "imageName": image,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [serve_command(run, served_name, second=second, max_hours=max_hours,
                                        idle_minutes=idle_minutes)],
    }


def build_train_payload(*, run: str, volume_id: str, gpu: str = TRAIN_GPU,
                        datacenter: str = TRAIN_DATACENTER, public_key: str | None = None,
                        image: str = TRAIN_IMAGE, min_ram_gb: int = 48, cloud_type: str = "COMMUNITY",
                        **train_flags) -> dict:
    """PodCreateInput for a training pod. Pure: everything it sends is in its arguments.
    No key of any kind goes to it — the base model is public and the data is on the volume."""
    if cloud_type not in CLOUD_TYPES:
        raise ValueError(f"cloud type must be one of {CLOUD_TYPES}, got {cloud_type!r}")
    env = {"HF_HOME": f"{ROOT}/hf", "PYTHONPATH": f"{ROOT}/code"}
    if public_key:
        env["PUBLIC_KEY"] = public_key
    return {
        "name": f"rembero-train-{run}"[:60],
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "allowedCudaVersions": TRAIN_CUDA_VERSIONS,
        # the merge holds a bf16 copy of the checkpoint in host memory and the text-only export
        # builds a second one next to it, as on the serving pod
        "minRAMPerGPU": min_ram_gb,
        "dataCenterIds": [datacenter],
        "networkVolumeId": volume_id,
        "volumeMountPath": ROOT,
        "containerDiskInGb": TRAIN_CONTAINER_DISK_GB,
        # nothing is served from a training pod; 22/tcp is there for a shell if PUBLIC_KEY is set
        "ports": ["22/tcp"],
        "env": env,
        "imageName": image,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [train_command(run, **train_flags)],
    }


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (optional `export `, optional matching quotes); comments and blanks skipped."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Replace each key's line in place, append the keys not there yet; leave every other line alone."""
    lines = path.read_text().splitlines() if path.exists() else []
    pending = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].removeprefix("export ").strip()
        if key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines.extend(f"{key}={value}" for key, value in pending.items())
    path.write_text("\n".join(lines) + "\n")


def setting(name: str, env: dict[str, str]) -> str | None:
    return os.environ.get(name) or env.get(name)


def build_request(url: str, key: str, *, method: str = "GET", body: dict | None = None) -> urllib.request.Request:
    """Every HTTP request pod.py makes is built here. RunPod sits behind Cloudflare, which rejects
    urllib's default User-Agent with 403 / error code 1010, so each request names this tool."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "Authorization": f"Bearer {key}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, method=method, data=data, headers=headers)


def api(method: str, path: str, key: str, body: dict | None = None) -> dict | list | None:
    request = build_request(f"{API}{path}", key, method=method, body=body)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {path} -> HTTP {error.code}: {error.read().decode()[:500]}")
    return json.loads(text) if text.strip() else None


def served_models(url: str, api_key: str) -> list[str]:
    with urllib.request.urlopen(build_request(f"{url}/models", api_key), timeout=30) as response:
        return [m.get("id") for m in json.loads(response.read().decode()).get("data", [])]


def create_serve(a, env: dict[str, str], key: str) -> None:
    api_key = setting("VLLM_API_KEY", env)
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        upsert_env(ENV_FILE, {"VLLM_API_KEY": api_key})
        print("generated VLLM_API_KEY and stored it in .env")
    volume = a.volume or setting("RUNPOD_VOLUME_ID", env)
    if not volume:
        raise SystemExit("no network volume: pass --volume or set RUNPOD_VOLUME_ID")
    public_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    public_key = public_key_path.read_text().strip() if public_key_path.exists() else None
    # a.run and a.served_name are parallel lists of one or two (main checks)
    second = (a.run[1], a.served_name[1]) if len(a.run) == 2 else None
    payload = build_pod_payload(run=a.run[0], served_name=a.served_name[0], api_key=api_key, volume_id=volume,
                                gpu=a.gpu, datacenter=a.datacenter, public_key=public_key, cloud_type=a.cloud,
                                second=second, max_hours=a.max_hours, idle_minutes=a.idle_minutes)
    pod = api("POST", "/pods", key, payload)
    pod_id = pod["id"]
    # the pod bills from here: its id reaches stdout before anything else can fail
    print(f"pod {pod_id} ({pod.get('costPerHr', '?')} $/h, {pod.get('desiredStatus', '?')})", flush=True)
    url = serve_url(pod_id)
    # a one-run pod blanks RUNPOD_SERVE_URL_2 so a stale second URL never points at a dead pod
    url_2 = serve_url(pod_id, PORTS[1]) if second else ""
    upsert_env(ENV_FILE, {"RUNPOD_SERVE_POD_ID": pod_id, "RUNPOD_SERVE_URL": url, "RUNPOD_SERVE_URL_2": url_2})
    print(f"the pod stops itself after {a.max_hours} h, or {a.idle_minutes} min without a request")
    print(f"{url}  ({a.served_name[0]})")
    if second:
        print(f"{url_2}  ({second[1]})")


def create_train(a, env: dict[str, str], key: str) -> None:
    """Create the training pod and record its id. Nothing waits for it: the run is followed with
    `train-log`, and the pod stops itself when the run ends (or after --max-hours)."""
    volume = a.volume or setting("RUNPOD_VOLUME_ID", env)
    if not volume:
        raise SystemExit("no network volume: pass --volume or set RUNPOD_VOLUME_ID")
    public_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    public_key = public_key_path.read_text().strip() if public_key_path.exists() else None
    payload = build_train_payload(run=a.run, volume_id=volume, gpu=a.gpu, datacenter=a.datacenter,
                                  public_key=public_key, cloud_type=a.cloud, max_length=a.max_length,
                                  batch_size=a.batch_size, grad_accum=a.grad_accum, quant=a.quant,
                                  liger=not a.no_liger, max_hours=a.max_hours)
    pod = api("POST", "/pods", key, payload)
    pod_id = pod["id"]
    # the pod bills from here: its id reaches stdout before anything else can fail
    print(f"pod {pod_id} ({pod.get('costPerHr', '?')} $/h, {pod.get('desiredStatus', '?')})", flush=True)
    upsert_env(ENV_FILE, {"RUNPOD_TRAIN_POD_ID": pod_id})
    print(f"follow it with: pod.py train-log --run {a.run}")
    print(f"it stops itself when the run ends, or after {a.max_hours} h; check: pod.py status --pod {pod_id}")


def train_log(a, env: dict[str, str]) -> None:
    """Print the tail of runs/<run>/train.log off the network volume, over the S3 API — the
    training pod's own console is the only other place these lines appear."""
    if not RUN_NAME.match(a.run):
        raise SystemExit(f"run must match {RUN_NAME.pattern}, got {a.run!r}")
    # volume.py reads its settings from the environment; .env is where they usually live
    for name in ("RUNPOD_VOLUME_ID", "RUNPOD_DATACENTER", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY",
                 "S3_ACCESS_KEY", "S3_SECRET_KEY"):
        value = setting(name, env)
        if value:
            os.environ[name] = value
    # pod.py is run as a script, so the repository root is not on sys.path by itself
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.runpod import volume

    remote = f"runs/{a.run}/train.log"
    try:
        text = volume.read_text(remote)
    except Exception as error:  # noqa: BLE001 - boto3 raises a family of errors; the message is what matters
        raise SystemExit(f"could not read {remote} from the volume: {type(error).__name__}: {error}")
    lines = text.splitlines()
    print("\n".join(lines[-a.tail:] if a.tail else lines))


def stop_pod(pod_id: str, key: str) -> None:
    api("POST", f"/pods/{pod_id}/stop", key)
    print(f"stop requested for {pod_id}")


def wait(a, env: dict[str, str]) -> None:
    api_key = setting("VLLM_API_KEY", env)
    if not api_key:
        raise SystemExit("no VLLM_API_KEY in the environment or .env")
    # a pod that never comes up still bills: wait stops it at the timeout, so it needs the key up front
    runpod_key = setting("RUNPOD_API_KEY", env)
    if a.stop_on_timeout and not runpod_key:
        raise SystemExit("no RUNPOD_API_KEY in the environment or .env; wait stops the pod on timeout "
                         "(pass --no-stop-on-timeout to wait without stopping it)")
    url, started, last = serve_url(a.pod, a.port), time.time(), None
    while time.time() - started < a.timeout:
        try:
            models = served_models(url, api_key)
            state = f"models {models}"
            if models and (a.served_name is None or a.served_name in models):
                print(f"ready after {(time.time() - started) / 60:.1f} min: {models}")
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
            state = f"not up ({getattr(error, 'code', None) or type(error).__name__})"
        if state != last:
            print(f"[{(time.time() - started) / 60:5.1f} min] {state}")
            last = state
        time.sleep(20)
    missed = f"{url} did not list {a.served_name or 'a model'} within {a.timeout / 60:.0f} min"
    if not a.stop_on_timeout:
        raise SystemExit(f"{missed}; pod {a.pod} is still running and billing (--no-stop-on-timeout)")
    print(f"{missed}; stopping pod {a.pod}")
    stop_pod(a.pod, runpod_key)
    raise SystemExit(f"{missed}; stopped pod {a.pod} (check runs/<run>/prepare.log and serve.log, then start it)")


def show(pod: dict) -> None:
    fields = ("id", "name", "desiredStatus", "costPerHr", "imageName")
    print(" ".join(f"{f}={pod.get(f)}" for f in fields) + f" gpu={(pod.get('gpu') or {}).get('displayName')}"
          f" dc={(pod.get('machine') or {}).get('dataCenterId')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-serve")
    # --run/--served-name pair up in order; a second pair serves a second reader on port 8001
    create.add_argument("--run", action="append", required=True)
    create.add_argument("--served-name", action="append", required=True)
    create.add_argument("--gpu", default="NVIDIA GeForce RTX 5090"); create.add_argument("--datacenter", default="EUR-NO-1")
    create.add_argument("--volume"); create.add_argument("--cloud", choices=CLOUD_TYPES, default="COMMUNITY")
    # the pod stops itself: after --max-hours whatever it is doing, or --idle-minutes with no request
    create.add_argument("--max-hours", type=float, default=SERVE_MAX_HOURS)
    create.add_argument("--idle-minutes", type=float, default=SERVE_IDLE_MINUTES)
    training = sub.add_parser("create-train")
    training.add_argument("--run", required=True)
    training.add_argument("--gpu", default=TRAIN_GPU); training.add_argument("--datacenter", default=TRAIN_DATACENTER)
    training.add_argument("--volume"); training.add_argument("--cloud", choices=CLOUD_TYPES, default="COMMUNITY")
    training.add_argument("--max-length", type=int, default=TRAIN_DEFAULTS["max_length"])
    training.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    training.add_argument("--grad-accum", type=int, default=TRAIN_DEFAULTS["grad_accum"])
    training.add_argument("--quant", choices=QUANTS, default=TRAIN_DEFAULTS["quant"])
    training.add_argument("--no-liger", action="store_true")
    training.add_argument("--max-hours", type=float, default=TRAIN_MAX_HOURS,
                          help="the pod stops itself after this long, however the run is going")
    tailing = sub.add_parser("train-log")
    tailing.add_argument("--run", required=True)
    tailing.add_argument("--tail", type=int, default=40, help="0 prints the whole log")
    waiting = sub.add_parser("wait")
    waiting.add_argument("--pod"); waiting.add_argument("--served-name"); waiting.add_argument("--timeout", type=int, default=30 * 60)
    waiting.add_argument("--port", type=int, choices=PORTS, default=PORTS[0])
    waiting.add_argument("--no-stop-on-timeout", dest="stop_on_timeout", action="store_false",
                         help="leave the pod running (and billing) when it does not come up in time")
    for name in ("stop", "start", "terminate", "status"):
        sub.add_parser(name).add_argument("--pod")
    a = ap.parse_args(argv)
    if a.command == "create-serve" and not (len(a.run) == len(a.served_name) <= 2):
        create.error("give one or two --run, each with its own --served-name")
    if a.command == "create-serve" and len(a.run) == 2 and a.gpu not in LARGE_GPUS:
        create.error(f"two readers at --gpu-memory-utilization {SHARED_GPU_MEMORY_UTILIZATION} each need an "
                     f"80 GB+ GPU; --gpu {a.gpu!r} is not one of: {', '.join(LARGE_GPUS)}")

    env = read_env(ENV_FILE)
    if a.command == "train-log":
        return train_log(a, env)  # reads the volume over S3; no RunPod API key needed
    if a.command not in ("create-serve", "create-train"):
        a.pod = a.pod or setting("RUNPOD_SERVE_POD_ID", env)
        if not a.pod and a.command != "status":
            raise SystemExit("no pod: pass --pod or set RUNPOD_SERVE_POD_ID")
    if a.command == "wait":
        return wait(a, env)
    key = setting("RUNPOD_API_KEY", env)
    if not key:
        raise SystemExit("no RUNPOD_API_KEY in the environment or .env")
    if a.command == "create-serve":
        create_serve(a, env, key)
    elif a.command == "create-train":
        create_train(a, env, key)
    elif a.command == "stop":
        stop_pod(a.pod, key)
    elif a.command == "start":
        api("POST", f"/pods/{a.pod}/start", key)
        print(f"start requested for {a.pod}")
    elif a.command == "terminate":
        api("DELETE", f"/pods/{a.pod}", key)
        print(f"terminated {a.pod}")
    elif a.pod:
        show(api("GET", f"/pods/{a.pod}", key))
    else:
        for pod in api("GET", "/pods", key) or []:
            show(pod)


if __name__ == "__main__":
    main()
