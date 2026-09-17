"""Fingerprint a merged checkpoint directory, to show two copies hold the same weights.
  python benchmarks/runpod/fingerprint.py <merged-text dir>
Prints JSON: the tensor count, the parameter count, sha256 of the sorted tensor names, sha256 of
the raw bytes of five tensors picked by sorted name (the first, the last and three evenly spaced
between), the same for five of the attention/MLP projection weights LoRA adapts, and sha256 of
config.json, tokenizer_config.json and chat_template.jinja where present.
Reads safetensors headers directly and only the five sampled tensors' bytes, so it needs no torch
and runs in seconds on a 16 GB checkpoint. Compare the pod's merged-text against Modal's."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

FILES = ("config.json", "tokenizer_config.json", "chat_template.jinja")
# the Linear weights LoRA adapts; the five plain samples land on norms and embeddings, which a
# merge never changes, so these are the samples that show the adapter went in the same way
PROJECTION = re.compile(r"\.(q|k|v|o|gate|up|down)_proj(\.linear)?\.weight$")


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header (tensor name -> dtype, shape, data_offsets) and where the data starts."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def sample_indices(n: int) -> list[int]:
    return sorted({round(i * (n - 1) / 4) for i in range(5)}) if n else []


def fingerprint(directory: Path | str) -> dict:
    directory = Path(directory)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {directory}")
    where: dict[str, tuple[Path, int, dict]] = {}
    for shard in shards:
        header, data_start = read_header(shard)
        for key, info in header.items():
            where[key] = (shard, data_start, info)
    keys = sorted(where)
    parameters = 0
    for _, _, info in where.values():
        count = 1
        for size in info["shape"]:
            count *= size
        parameters += count
    def sample(chosen: list[str]) -> dict:
        out = {}
        for index in sample_indices(len(chosen)):
            key = chosen[index]
            shard, data_start, info = where[key]
            begin, end = info["data_offsets"]
            with open(shard, "rb") as handle:
                handle.seek(data_start + begin)
                data = handle.read(end - begin)
            out[key] = {"dtype": info["dtype"], "shape": info["shape"], "sha256": hashlib.sha256(data).hexdigest()}
        return out
    return {
        "tensor_count": len(keys),
        "parameter_count": parameters,
        "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "samples": sample(keys),
        "projection_samples": sample([k for k in keys if PROJECTION.search(k)]),
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in FILES if (directory / name).exists()},
    }


if __name__ == "__main__":
    print(json.dumps(fingerprint(sys.argv[1]), indent=2))
