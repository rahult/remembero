import json
import shlex

import pytest

from benchmarks.runpod import prepare_reader
from benchmarks.runpod.pod import IMAGE, build_pod_payload, read_env, serve_command, serve_url, upsert_env

RUN = "reader-v4-gemma4-e4b"
NAME = "rembero-reader-v4"


def payload(**overrides):
    args = dict(run=RUN, served_name=NAME, api_key="vllm-key-123", volume_id="vol-abc",
                gpu="NVIDIA GeForce RTX 5090", datacenter="EUR-NO-1", public_key="ssh-ed25519 AAAA me@mac")
    return build_pod_payload(**{**args, **overrides})


def test_the_pod_payload_asks_for_a_community_5090_on_the_volume():
    p = payload()
    assert p["cloudType"] == "COMMUNITY" and p["computeType"] == "GPU"
    assert p["gpuTypeIds"] == ["NVIDIA GeForce RTX 5090"] and p["gpuTypePriority"] == "availability"
    assert p["dataCenterIds"] == ["EUR-NO-1"]
    assert p["networkVolumeId"] == "vol-abc" and p["volumeMountPath"] == "/workspace"
    assert p["containerDiskInGb"] == 40
    assert p["ports"] == ["8000/http", "22/tcp"]
    assert p["imageName"] == IMAGE and IMAGE.startswith("vllm/vllm-openai:v")
    assert p["dockerEntrypoint"] == ["bash", "-lc"]
    assert p["dockerStartCmd"] == [serve_command(RUN, NAME)]
    assert p["env"] == {"VLLM_API_KEY": "vllm-key-123", "HF_HOME": "/workspace/hf",
                        "PYTHONPATH": "/workspace/code", "PUBLIC_KEY": "ssh-ed25519 AAAA me@mac"}


def test_the_pod_payload_leaves_public_key_out_when_there_is_none():
    assert "PUBLIC_KEY" not in payload(public_key=None)["env"]


def test_no_hugging_face_or_modal_credential_reaches_the_payload(monkeypatch):
    for name, value in {"HF_TOKEN": "hf-secret-1", "HUGGING_FACE_HUB_TOKEN": "hf-secret-2",
                        "MODAL_TOKEN_ID": "modal-secret-1", "MODAL_TOKEN_SECRET": "modal-secret-2",
                        "MODAL_SERVE_API_KEY": "modal-secret-3", "RUNPOD_API_KEY": "runpod-secret"}.items():
        monkeypatch.setenv(name, value)
    text = json.dumps(payload())
    for secret in ("hf-secret", "modal-secret", "runpod-secret", "HF_TOKEN", "HUGGING_FACE", "MODAL"):
        assert secret not in text


def test_the_serve_command_carries_every_serving_flag_of_record():
    command = serve_command(RUN, NAME)
    words = shlex.split(command)
    vllm = words[words.index("vllm"):]
    assert vllm[:3] == ["vllm", "serve", f"/workspace/runs/{RUN}/merged-text"]

    def flag(name):
        return vllm[vllm.index(name) + 1]

    assert flag("--dtype") == "bfloat16"
    assert flag("--max-model-len") == "8192"
    assert flag("--max-num-seqs") == "32"
    assert flag("--gpu-memory-utilization") == "0.92"
    assert flag("--reasoning-parser") == "gemma4"
    assert json.loads(flag("--default-chat-template-kwargs")) == {"enable_thinking": False}
    assert flag("--served-model-name") == NAME
    assert flag("--host") == "0.0.0.0" and flag("--port") == "8000"
    assert "--api-key" in vllm


def test_the_serve_command_prepares_the_weights_first_and_never_serves_without_a_key():
    command = serve_command(RUN, NAME)
    assert command.index("peft>=0.17") < command.index("benchmarks.runpod.prepare_reader") < command.index("exec vllm serve")
    assert f"--root /workspace --run {RUN}" in command
    assert '--api-key "$VLLM_API_KEY"' in command
    assert command.index('test -n "$VLLM_API_KEY"') < command.index("exec vllm serve")
    assert "vllm-key" not in command  # the key travels in env, not in the start command


def test_the_serve_command_refuses_a_run_name_that_escapes_the_volume():
    with pytest.raises(ValueError):
        serve_command("../etc", NAME)
    with pytest.raises(ValueError):
        serve_command(RUN, "name; rm -rf /")


def test_the_proxy_url_is_the_pods_port_8000():
    assert serve_url("abc123") == "https://abc123-8000.proxy.runpod.net/v1"


def test_upsert_env_replaces_existing_keys_and_appends_new_ones(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# secrets\nRUNPOD_API_KEY=keep-me\nRUNPOD_SERVE_POD_ID=old\n\nexport OTHER='quoted value'\n")
    upsert_env(env, {"RUNPOD_SERVE_POD_ID": "new-pod", "RUNPOD_SERVE_URL": "https://new-pod-8000.proxy.runpod.net/v1"})
    lines = env.read_text().splitlines()
    assert lines[:3] == ["# secrets", "RUNPOD_API_KEY=keep-me", "RUNPOD_SERVE_POD_ID=new-pod"]
    assert lines.count("RUNPOD_SERVE_POD_ID=new-pod") == 1
    assert lines[-1] == "RUNPOD_SERVE_URL=https://new-pod-8000.proxy.runpod.net/v1"
    assert read_env(env) == {"RUNPOD_API_KEY": "keep-me", "RUNPOD_SERVE_POD_ID": "new-pod", "OTHER": "quoted value",
                             "RUNPOD_SERVE_URL": "https://new-pod-8000.proxy.runpod.net/v1"}


def test_upsert_env_creates_a_missing_file(tmp_path):
    env = tmp_path / ".env"
    upsert_env(env, {"VLLM_API_KEY": "abc"})
    assert read_env(env) == {"VLLM_API_KEY": "abc"}


def test_read_env_of_a_missing_file_is_empty(tmp_path):
    assert read_env(tmp_path / "absent.env") == {}


def fake_steps(monkeypatch, calls):
    def merge(run_dir, base_model):
        calls.append("merge")
        (run_dir / "merged").mkdir(parents=True)
        (run_dir / "merged" / "model.safetensors").write_text("weights")
        return run_dir / "merged"

    def export(run_dir):
        calls.append("export")
        text = run_dir / "merged-text"
        text.mkdir()
        (text / "config.json").write_text("{}")
        (text / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
        return text

    def restore(run_dir, base_model):
        calls.append("restore")
        return 42

    monkeypatch.setattr(prepare_reader.reader_lora, "merge_adapter", merge)
    monkeypatch.setattr(prepare_reader.reader_lora, "export_text_only", export)
    monkeypatch.setattr(prepare_reader.reader_lora, "restore_dropped_weights", restore)


def adapter(root):
    adapter_dir = root / "runs" / RUN / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "tokenizer.json").write_text("{}")
    (adapter_dir / "chat_template.jinja").write_text("{{ messages }}")
    return adapter_dir


def test_prepare_runs_merge_export_restore_then_marks_and_drops_merged(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)

    text_dir = prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    run_dir = tmp_path / "runs" / RUN
    assert calls == ["merge", "export", "restore"]
    assert text_dir == run_dir / "merged-text"
    assert prepare_reader.is_prepared(run_dir)
    assert not (run_dir / "merged").exists()
    assert (text_dir / "tokenizer.json").exists() and (text_dir / "chat_template.jinja").exists()
    assert (run_dir / "adapter" / "adapter_config.json").exists()


def test_prepare_is_a_no_op_once_the_marker_is_written(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")
    calls.clear()

    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    assert calls == []


def test_a_half_finished_prepare_without_the_marker_is_redone(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    stale = tmp_path / "runs" / RUN / "merged-text"
    stale.mkdir(parents=True)
    (stale / "config.json").write_text("{}")
    (stale / "model-00002-of-00002.safetensors").write_text("left by a boot that died")

    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    assert calls == ["merge", "export", "restore"]
    assert not (stale / "model-00002-of-00002.safetensors").exists()


def test_a_marker_without_a_config_is_not_prepared(tmp_path):
    run_dir = tmp_path / "runs" / RUN
    (run_dir / "merged-text").mkdir(parents=True)
    (run_dir / "merged-text" / prepare_reader.MARKER).write_text("{}")
    assert not prepare_reader.is_prepared(run_dir)


def test_prepare_without_an_adapter_fails_before_loading_anything(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    with pytest.raises(SystemExit):
        prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")
    assert calls == []


def test_prepare_refuses_weights_with_no_chat_template(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)

    def export_without_template(run_dir):
        text = run_dir / "merged-text"
        text.mkdir()
        (text / "config.json").write_text("{}")
        (text / "tokenizer_config.json").write_text("{}")
        return text

    monkeypatch.setattr(prepare_reader.reader_lora, "export_text_only", export_without_template)
    adapter_dir = adapter(tmp_path)
    (adapter_dir / "chat_template.jinja").unlink()

    with pytest.raises(SystemExit):
        prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")
    assert not prepare_reader.is_prepared(tmp_path / "runs" / RUN)
