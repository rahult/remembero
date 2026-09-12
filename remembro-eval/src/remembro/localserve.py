"""Start llama-server for the local writer on demand, with the flags the writer needs.

Gemma 4's chat template turns thinking on by default; the writer was trained without it and
answers "[]" after thinking, so thinking is disabled. Parallel slots split the context, so each
slot gets a full 8k.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

FLAGS = ["-c", "16384", "-np", "2", "-ngl", "99", "--reasoning-budget", "0", "--chat-template-kwargs", '{"enable_thinking":false}']


def command(gguf: str | Path, port: int, alias: str = "rembero-writer") -> list[str]:
    return [shutil.which("llama-server") or "llama-server", "-m", str(gguf), "--port", str(port), "--alias", alias, *FLAGS]


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/").removesuffix("/v1") + "/health", timeout=2) as r:
            return b'"ok"' in r.read()
    except Exception:  # noqa: BLE001
        return False


def ensure(base_url: str, gguf: str | Path | None, timeout: float = 120.0) -> bool:
    """True when a server answers at base_url, starting one from gguf if it does not."""
    if healthy(base_url):
        return True
    if not gguf or not Path(gguf).expanduser().exists() or not shutil.which("llama-server"):
        return False
    port = int(base_url.rstrip("/").removesuffix("/v1").rsplit(":", 1)[-1])
    log = Path(os.environ.get("REMEMBRO_HOME", "~/.remembro/default")).expanduser() / "llama-server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        subprocess.Popen(command(Path(gguf).expanduser(), port), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if healthy(base_url):
            return True
        time.sleep(2)
    return False
