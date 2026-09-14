"""Copy files to and from a RunPod network volume through its S3-compatible API.
  python benchmarks/runpod/volume.py put <local-path> <volume-path>   (file or directory)
  python benchmarks/runpod/volume.py get <volume-path> <local-path>
Env: RUNPOD_VOLUME_ID, RUNPOD_DATACENTER (e.g. EU-RO-1), and the S3 key pair under either
RUNPOD_S3_ACCESS_KEY / RUNPOD_S3_SECRET_KEY or the short S3_ACCESS_KEY / S3_SECRET_KEY."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3


def credential(*names: str) -> str:
    """The first of these environment variables that is set (the console calls the S3 keys one
    thing, the .env here another), or a KeyError naming all of them."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise KeyError(f"set one of {', '.join(names)}")


def client():
    dc = os.environ["RUNPOD_DATACENTER"]
    return boto3.client("s3", endpoint_url=f"https://s3api-{dc.lower()}.runpod.io/", region_name=dc,
                        aws_access_key_id=credential("RUNPOD_S3_ACCESS_KEY", "S3_ACCESS_KEY"),
                        aws_secret_access_key=credential("RUNPOD_S3_SECRET_KEY", "S3_SECRET_KEY"))


def put(local: Path, remote: str) -> None:
    s3, bucket = client(), os.environ["RUNPOD_VOLUME_ID"]
    files = [local] if local.is_file() else sorted(p for p in local.rglob("*") if p.is_file())
    for f in files:
        key = remote if local.is_file() else f"{remote.rstrip('/')}/{f.relative_to(local)}"
        print(f"put {f} -> {key} ({f.stat().st_size / 2**20:.1f} MiB)")
        s3.upload_file(str(f), bucket, key)


def get(remote: str, local: Path) -> None:
    s3, bucket = client(), os.environ["RUNPOD_VOLUME_ID"]
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"get {remote} -> {local}")
    s3.download_file(bucket, remote, str(local))


if __name__ == "__main__":
    op, a, b = sys.argv[1:4]
    put(Path(a), b) if op == "put" else get(a, Path(b))
