"""Launch and manage the pod that serves a reader with vLLM, for the harness on the Mac.
  python benchmarks/runpod/pod.py create-serve --run reader-v4-gemma4-e4b --served-name rembero-reader-v4
  python benchmarks/runpod/pod.py wait [--pod <id>] [--served-name rembero-reader-v4]
  python benchmarks/runpod/pod.py stop|start|terminate [--pod <id>]
  python benchmarks/runpod/pod.py status [--pod <id>]
The pod runs the official vLLM image on a community GPU with the network volume at /workspace:
its start command rebuilds the reader's merged weights from runs/<run>/adapter once
(prepare_reader.py) and then serves them. Env or .env: RUNPOD_API_KEY, RUNPOD_VOLUME_ID.
create-serve writes VLLM_API_KEY (once), RUNPOD_SERVE_POD_ID and RUNPOD_SERVE_URL to .env; the
other subcommands default --pod to RUNPOD_SERVE_POD_ID. No Hugging Face or Modal credential goes
to the pod: the base model is public and the adapter is already on the volume."""

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
# Newest stable vLLM release on 2026-09-15 (v0.29.0, pushed 2026-09-09). The -cu129 build rather
# than the default CUDA 13.0 one, so community hosts on a CUDA 12.9 driver qualify too; both carry
# kernels for the RTX 5090 (TORCH_CUDA_ARCH_LIST includes 12.0). Gemma 4 landed in vLLM in spring 2026.
IMAGE = "vllm/vllm-openai:v0.29.0-cu129"
CUDA_VERSIONS = ["12.9", "13.0"]
ROOT = "/workspace"
BASE_MODEL = "google/gemma-4-E4B-it"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
SERVED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,120}$")
# The Modal serve of record (benchmarks/modal/train_lora.py `serve`), flag for flag.
SERVE_FLAGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "8192",
    "--max-num-seqs", "32",
    "--gpu-memory-utilization", "0.92",
    "--reasoning-parser", "gemma4",
    "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}, separators=(",", ":")),
]
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def serve_url(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net/v1"


def serve_command(run: str, served_name: str, *, root: str = ROOT, base_model: str = BASE_MODEL) -> str:
    """The pod's whole start command, for `bash -lc`: install PEFT (the image has transformers but
    not PEFT), rebuild the weights once, then hand the process to vLLM. The API key is read from
    the pod's env, never written into the command, and an empty key stops the pod from serving."""
    if not RUN_NAME.match(run):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    if not SERVED_NAME.match(served_name):
        raise ValueError(f"served name must match {SERVED_NAME.pattern}, got {served_name!r}")
    model_dir = f"{root}/runs/{run}/merged-text"
    prepare = ["python3", "-m", "benchmarks.runpod.prepare_reader", "--root", root, "--run", run, "--base-model", base_model]
    vllm = ["vllm", "serve", model_dir, *SERVE_FLAGS, "--served-model-name", served_name, "--host", "0.0.0.0", "--port", "8000"]
    return " && ".join([
        "set -e",
        'test -n "$VLLM_API_KEY"',
        # uv respects the image's /etc/uv-overrides.txt pins; plain pip is the fallback
        '(uv pip install --system -q "peft>=0.17" || python3 -m pip install -q "peft>=0.17")',
        shlex.join(prepare),
        f'exec {shlex.join(vllm)} --api-key "$VLLM_API_KEY"',
    ])


def build_pod_payload(*, run: str, served_name: str, api_key: str, volume_id: str,
                      gpu: str = "NVIDIA GeForce RTX 5090", datacenter: str = "EUR-NO-1",
                      public_key: str | None = None, image: str = IMAGE, min_ram_gb: int = 32) -> dict:
    """PodCreateInput for the serving pod. Pure: everything it sends is in its arguments."""
    env = {"VLLM_API_KEY": api_key, "HF_HOME": f"{ROOT}/hf", "PYTHONPATH": f"{ROOT}/code"}
    if public_key:
        env["PUBLIC_KEY"] = public_key
    return {
        "name": f"rembero-serve-{run}"[:60],
        "cloudType": "COMMUNITY",
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "allowedCudaVersions": CUDA_VERSIONS,
        # the merge loads the 16 GB multimodal checkpoint into host memory before vLLM starts
        "minRAMPerGPU": min_ram_gb,
        "dataCenterIds": [datacenter],
        "networkVolumeId": volume_id,
        "volumeMountPath": ROOT,
        "containerDiskInGb": 40,
        "ports": ["8000/http", "22/tcp"],
        "env": env,
        "imageName": image,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [serve_command(run, served_name)],
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


def api(method: str, path: str, key: str, body: dict | None = None) -> dict | list | None:
    request = urllib.request.Request(f"{API}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {path} -> HTTP {error.code}: {error.read().decode()[:500]}")
    return json.loads(text) if text.strip() else None


def served_models(url: str, api_key: str) -> list[str]:
    request = urllib.request.Request(f"{url}/models", headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
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
    payload = build_pod_payload(run=a.run, served_name=a.served_name, api_key=api_key, volume_id=volume,
                                gpu=a.gpu, datacenter=a.datacenter, public_key=public_key)
    pod = api("POST", "/pods", key, payload)
    pod_id = pod["id"]
    url = serve_url(pod_id)
    upsert_env(ENV_FILE, {"RUNPOD_SERVE_POD_ID": pod_id, "RUNPOD_SERVE_URL": url})
    print(f"pod {pod_id} ({pod.get('costPerHr', '?')} $/h, {pod.get('desiredStatus', '?')})")
    print(url)


def wait(a, env: dict[str, str]) -> None:
    api_key = setting("VLLM_API_KEY", env)
    if not api_key:
        raise SystemExit("no VLLM_API_KEY in the environment or .env")
    url, started, last = serve_url(a.pod), time.time(), None
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
    raise SystemExit(f"{url} did not list {a.served_name or 'a model'} within {a.timeout / 60:.0f} min")


def show(pod: dict) -> None:
    fields = ("id", "name", "desiredStatus", "costPerHr", "imageName")
    print(" ".join(f"{f}={pod.get(f)}" for f in fields) + f" gpu={(pod.get('gpu') or {}).get('displayName')}"
          f" dc={(pod.get('machine') or {}).get('dataCenterId')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-serve")
    create.add_argument("--run", required=True); create.add_argument("--served-name", required=True)
    create.add_argument("--gpu", default="NVIDIA GeForce RTX 5090"); create.add_argument("--datacenter", default="EUR-NO-1")
    create.add_argument("--volume")
    waiting = sub.add_parser("wait")
    waiting.add_argument("--pod"); waiting.add_argument("--served-name"); waiting.add_argument("--timeout", type=int, default=30 * 60)
    for name in ("stop", "start", "terminate", "status"):
        sub.add_parser(name).add_argument("--pod")
    a = ap.parse_args(argv)

    env = read_env(ENV_FILE)
    if a.command != "create-serve":
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
    elif a.command in ("stop", "start"):
        api("POST", f"/pods/{a.pod}/{a.command}", key)
        print(f"{a.command} requested for {a.pod}")
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
