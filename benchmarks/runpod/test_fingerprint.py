import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.runpod.fingerprint import fingerprint

ROOT = Path(__file__).resolve().parents[2]


def tiny_checkpoint(directory: Path, n: int = 9) -> dict:
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.layers.{i}.weight": torch.arange(i + 1, dtype=torch.float32) * (i + 1) for i in range(n)}
    tensors["model.layers.0.norm"] = torch.ones(3, 2, dtype=torch.bfloat16)
    tensors["model.layers.0.self_attn.q_proj.linear.weight"] = torch.full((2, 2), 3.0)
    tensors["model.layers.1.mlp.down_proj.weight"] = torch.full((2, 3), 5.0)
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text('{"architectures": ["Gemma4ForCausalLM"]}')
    (directory / "chat_template.jinja").write_text("{{ messages }}")
    return tensors


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fingerprint_counts_hashes_keys_and_samples_five_tensors(tmp_path):
    tensors = tiny_checkpoint(tmp_path / "merged-text")
    keys = sorted(tensors)

    fp = fingerprint(tmp_path / "merged-text")

    assert fp["tensor_count"] == 12
    assert fp["parameter_count"] == sum(t.numel() for t in tensors.values())
    assert fp["keys_sha256"] == sha("\n".join(keys).encode())
    n = len(keys)
    expected = [keys[i] for i in (0, round((n - 1) / 4), round((n - 1) / 2), round(3 * (n - 1) / 4), n - 1)]
    assert list(fp["samples"]) == expected
    first = fp["samples"][keys[0]]
    assert first["sha256"] == sha(tensors[keys[0]].contiguous().view(torch_uint8()).numpy().tobytes())
    last = fp["samples"][keys[-1]]
    assert last["dtype"] == "F32" and last["shape"] == list(tensors[keys[-1]].shape)
    assert last["sha256"] == sha(tensors[keys[-1]].numpy().tobytes())
    projections = ["model.layers.0.self_attn.q_proj.linear.weight", "model.layers.1.mlp.down_proj.weight"]
    assert list(fp["projection_samples"]) == projections
    assert fp["projection_samples"][projections[1]]["sha256"] == sha(tensors[projections[1]].numpy().tobytes())
    assert fp["files"] == {"config.json": sha((tmp_path / "merged-text" / "config.json").read_bytes()),
                           "chat_template.jinja": sha(b"{{ messages }}")}


def torch_uint8():
    import torch

    return torch.uint8


def test_the_same_weights_fingerprint_the_same_and_one_changed_tensor_does_not(tmp_path):
    import torch
    from safetensors.torch import save_file

    tensors = tiny_checkpoint(tmp_path / "a")
    tiny_checkpoint(tmp_path / "b")
    assert fingerprint(tmp_path / "a") == fingerprint(tmp_path / "b")
    last = sorted(tensors)[-1]
    save_file({**tensors, last: tensors[last] + 1}, str(tmp_path / "b" / "model.safetensors"), metadata={"format": "pt"})
    assert fingerprint(tmp_path / "a")["samples"][last] != fingerprint(tmp_path / "b")["samples"][last]


def test_fingerprint_reads_every_shard(tmp_path):
    import torch
    from safetensors.torch import save_file

    directory = tmp_path / "sharded"
    directory.mkdir()
    save_file({"a.weight": torch.zeros(2)}, str(directory / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": torch.zeros(3)}, str(directory / "model-00002-of-00002.safetensors"))
    fp = fingerprint(directory)
    assert fp["tensor_count"] == 2 and fp["parameter_count"] == 5 and list(fp["samples"]) == ["a.weight", "b.weight"]


def test_a_directory_without_weights_has_no_fingerprint(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint(tmp_path)


def test_the_cli_prints_the_same_json(tmp_path):
    tiny_checkpoint(tmp_path / "merged-text")
    out = subprocess.run([sys.executable, str(ROOT / "benchmarks/runpod/fingerprint.py"), str(tmp_path / "merged-text")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == fingerprint(tmp_path / "merged-text")
