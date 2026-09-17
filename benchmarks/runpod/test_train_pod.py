import json

import pytest

from benchmarks.runpod import train_pod
from benchmarks.runpod.handler import DEFAULTS

RUN = "reader-v7-gemma4-e4b"


def test_the_cfg_splits_the_volume_from_the_container_disk():
    """On a pod the network volume is at /workspace, so the serverless /runpod-volume layout
    handler.job_config builds cannot be reused: the paths are passed in instead."""
    cfg = train_pod.build_config(RUN)
    assert cfg["run"] == RUN
    assert cfg["data_dir"] == f"/workspace/data/{RUN}"
    assert cfg["volume_run_dir"] == f"/workspace/runs/{RUN}"
    assert cfg["run_dir"] == f"/root/runs/{RUN}"  # the container disk, not the volume
    assert "/runpod-volume" not in json.dumps(cfg)


def test_the_cfg_starts_from_the_handlers_defaults():
    cfg = train_pod.build_config(RUN)
    assert {key: cfg[key] for key in DEFAULTS} == DEFAULTS


def test_every_flag_overrides_its_default_and_the_roots_move_together():
    cfg = train_pod.build_config(RUN, root="/vol", local_root="/disk", max_length=4096, batch_size=2,
                                 grad_accum=8, quant="Q6_K", liger=False)
    assert cfg["max_length"] == 4096 and cfg["batch_size"] == 2 and cfg["grad_accum"] == 8
    assert cfg["quant"] == "Q6_K" and cfg["liger"] is False
    assert cfg["base_model"] == DEFAULTS["base_model"]
    assert cfg["data_dir"] == f"/vol/data/{RUN}" and cfg["volume_run_dir"] == f"/vol/runs/{RUN}"
    assert cfg["run_dir"] == f"/disk/runs/{RUN}"


def test_a_run_name_that_escapes_the_volume_or_an_unknown_quant_is_refused():
    for bad in ("../etc", "", "a b"):
        with pytest.raises(ValueError):
            train_pod.build_config(bad)
    with pytest.raises(ValueError):
        train_pod.build_config(RUN, quant="Q3_K_S")


def fake_run_job(seen, metrics=None, error=None):
    def run_job(cfg, progress):
        seen["cfg"], seen["progress"] = cfg, progress
        progress("train: starting")
        if error:
            raise error
        return dict(metrics or {})

    return run_job


def test_a_finished_run_writes_metrics_to_the_volume_and_prints_train_done(tmp_path, capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(train_pod, "run_job", fake_run_job(seen, {"train_loss": 0.5}))
    train_pod.main(["--run", RUN, "--root", str(tmp_path), "--local-root", str(tmp_path / "disk"),
                    "--max-length", "4096", "--no-liger"])
    out = capsys.readouterr().out
    assert "train: starting" in out  # progress goes to stdout as it happens
    [done] = [line for line in out.splitlines() if line.startswith("TRAIN_DONE ")]
    assert json.loads(done.removeprefix("TRAIN_DONE ")) == {"train_loss": 0.5}
    written = json.loads((tmp_path / "runs" / RUN / "metrics.json").read_text())
    assert written == {"train_loss": 0.5}
    assert seen["cfg"]["max_length"] == 4096 and seen["cfg"]["liger"] is False
    assert seen["cfg"]["run_dir"] == str(tmp_path / "disk" / "runs" / RUN)


def test_a_failed_run_prints_train_failed_and_exits_non_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(train_pod, "run_job", fake_run_job({}, error=RuntimeError("CUDA out of memory")))
    with pytest.raises(SystemExit) as failed:
        train_pod.main(["--run", RUN, "--root", str(tmp_path)])
    assert failed.value.code not in (0, None)
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("TRAIN_FAILED ")]
    assert "CUDA out of memory" in out
    assert not (tmp_path / "runs" / RUN / "metrics.json").exists()


def test_a_bad_run_name_fails_before_any_training(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(train_pod, "run_job", lambda cfg, progress: called.append(cfg))
    with pytest.raises(SystemExit):
        train_pod.main(["--run", "../etc", "--root", str(tmp_path)])
    assert called == []
