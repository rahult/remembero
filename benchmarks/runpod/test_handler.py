from pathlib import Path

import pytest

from benchmarks.runpod import handler as handler_module
from benchmarks.runpod.handler import STAGES, job_config, stage_resume_checkpoint


def test_job_config_fills_defaults_and_validates():
    cfg = job_config({"run": "reader-v5-gemma4-e4b"})
    assert cfg["base_model"] == "google/gemma-4-E4B-it" and cfg["max_length"] == 8192
    assert cfg["liger"] is True and cfg["quant"] == "Q8_0" and cfg["batch_size"] == 4 and cfg["grad_accum"] == 16
    assert cfg["data_dir"] == "/runpod-volume/data/reader-v5-gemma4-e4b"
    assert cfg["run_dir"] == "/workspace/runs/reader-v5-gemma4-e4b"
    assert cfg["volume_run_dir"] == "/runpod-volume/runs/reader-v5-gemma4-e4b"


def test_job_config_rejects_a_run_name_that_escapes_the_volume():
    with pytest.raises(ValueError):
        job_config({"run": "../etc"})


def test_job_config_rejects_a_quantization_llama_quantize_does_not_take():
    with pytest.raises(ValueError):
        job_config({"run": "reader-v5-gemma4-e4b", "quant": "Q4_0_4_4; rm -rf /"})


def test_job_config_ignores_merge_from_the_job_input():
    assert "merge" not in job_config({"run": "reader-v5-gemma4-e4b", "merge": False})


def test_stage_order_is_train_export_convert_quantize():
    assert STAGES == ("train", "export-text", "convert-f16", "quantize")


def test_the_resume_copy_takes_the_highest_numbered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(handler_module, "VOLUME", tmp_path / "runpod-volume")
    monkeypatch.setattr(handler_module, "WORKSPACE", tmp_path / "workspace")
    cfg = job_config({"run": "reader-v5-gemma4-e4b"})
    volume_run_dir, run_dir = Path(cfg["volume_run_dir"]), Path(cfg["run_dir"])
    for step in (9, 50, 25):
        checkpoint = volume_run_dir / "trainer" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(str(step))

    staged = stage_resume_checkpoint(volume_run_dir, run_dir)

    assert staged == run_dir / "trainer" / "checkpoint-50"
    assert (staged / "trainer_state.json").read_text() == "50"


def test_nothing_to_resume_from_is_not_an_error(tmp_path):
    assert stage_resume_checkpoint(tmp_path / "volume-run", tmp_path / "run") is None
