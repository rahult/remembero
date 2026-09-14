import pytest

from benchmarks.runpod.handler import job_config, STAGES


def test_job_config_fills_defaults_and_validates():
    cfg = job_config({"run": "reader-v5-gemma4-e4b"})
    assert cfg["base_model"] == "google/gemma-4-E4B-it" and cfg["max_length"] == 8192
    assert cfg["liger"] is True and cfg["quant"] == "Q8_0" and cfg["batch_size"] == 4 and cfg["grad_accum"] == 16
    assert cfg["data_dir"] == "/runpod-volume/data/reader-v5-gemma4-e4b"
    assert cfg["run_dir"] == "/runpod-volume/runs/reader-v5-gemma4-e4b"


def test_job_config_rejects_a_run_name_that_escapes_the_volume():
    with pytest.raises(ValueError):
        job_config({"run": "../etc"})


def test_stage_order_is_train_export_convert_quantize():
    assert STAGES == ("train", "export-text", "convert-f16", "quantize")
