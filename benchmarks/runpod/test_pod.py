import json
import os
import shlex
import subprocess
import time

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
    assert p["containerDiskInGb"] == 40 and p["minRAMPerGPU"] == 48
    assert p["ports"] == ["8000/http", "22/tcp"]
    assert p["imageName"] == IMAGE and IMAGE.startswith("vllm/vllm-openai:v")
    assert p["dockerEntrypoint"] == ["bash", "-lc"]
    assert p["dockerStartCmd"] == [serve_command(RUN, NAME)]
    assert p["env"] == {"VLLM_API_KEY": "vllm-key-123", "HF_HOME": "/workspace/hf",
                        "PYTHONPATH": "/workspace/code", "PUBLIC_KEY": "ssh-ed25519 AAAA me@mac"}


def test_the_pod_payload_can_ask_for_secure_cloud():
    assert payload(cloud_type="SECURE")["cloudType"] == "SECURE"
    with pytest.raises(ValueError):
        payload(cloud_type="SPOT")


def test_create_serve_takes_a_cloud_option_defaulting_to_community(monkeypatch):
    from benchmarks.runpod import pod

    seen = {}
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_API_KEY": "k"})
    monkeypatch.setattr(pod, "create_serve", lambda a, env, key: seen.update(cloud=a.cloud))
    pod.main(["create-serve", "--run", RUN, "--served-name", NAME])
    assert seen == {"cloud": "COMMUNITY"}
    pod.main(["create-serve", "--run", RUN, "--served-name", NAME, "--cloud", "SECURE"])
    assert seen == {"cloud": "SECURE"}
    with pytest.raises(SystemExit):
        pod.main(["create-serve", "--run", RUN, "--served-name", NAME, "--cloud", "SPOT"])


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
    assert flag("--max-model-len") == "12288"
    assert flag("--max-num-seqs") == "32"
    assert flag("--gpu-memory-utilization") == "0.92"
    assert flag("--reasoning-parser") == "gemma4"
    assert json.loads(flag("--default-chat-template-kwargs")) == {"enable_thinking": False}
    assert flag("--served-model-name") == NAME
    assert flag("--host") == "0.0.0.0" and flag("--port") == "8000"
    assert "--api-key" in vllm


def test_the_serve_command_prepares_the_weights_first_and_never_serves_without_a_key():
    command = serve_command(RUN, NAME)
    assert command.index("peft>=0.17") < command.index("benchmarks.runpod.prepare_reader") < command.index("vllm serve")
    assert f"--root /workspace --run {RUN}" in command
    assert '--api-key "$VLLM_API_KEY"' in command
    assert command.index('test -n "$VLLM_API_KEY"') < command.index("vllm serve")
    assert "vllm-key" not in command  # the key travels in env, not in the start command


def test_the_serve_command_runs_prepare_from_the_code_directory_not_the_vllm_workdir():
    """The image's WORKDIR /vllm-workspace holds vLLM's own `benchmarks` package, which would
    shadow ours for `python3 -m benchmarks...` run from there."""
    command = serve_command(RUN, NAME)
    assert "cd /workspace/code" in command
    assert command.index("cd /workspace/code") < command.index("python3 -m benchmarks.runpod.prepare_reader")


def test_the_serve_command_logs_prepare_and_serve_to_the_volume_and_parks_on_failure():
    command = serve_command(RUN, NAME)
    run_dir = f"/workspace/runs/{RUN}"
    assert f"{run_dir}/prepare.log" in command or ("prepare.log" in command and run_dir in command)
    assert "serve.log" in command and "PREPARE_FAILED" in command and "sleep infinity" in command
    assert "pipefail" in command


def test_the_serve_command_is_valid_bash():
    subprocess.run(["bash", "-n", "-c", serve_command(RUN, NAME)], check=True)


def fake_bin(directory, name, body):
    path = directory / name
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


def boot(tmp_path, prepare_exit, key="test-key", vllm_then="", healthy=True):
    """Run the start command with stub python3/uv/vllm/sleep on PATH and the volume in tmp_path.
    vllm_then is shell run by the vllm stub after it logs; healthy=False fails the /health probe."""
    bin_dir, root = tmp_path / "bin", tmp_path / "volume"
    bin_dir.mkdir()
    (root / "code").mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    fake_bin(bin_dir, "uv", f'echo "uv $*" >> {calls}')
    probe = 'case "$1" in -c) exit 1;; esac; ' if not healthy else ""
    fake_bin(bin_dir, "python3", f'echo "python3 $* (cwd $PWD)" >> {calls}; {probe}echo "prepare says hello"; echo "prepare error" >&2; exit {prepare_exit}')
    fake_bin(bin_dir, "vllm", f'echo "vllm $*" >> {calls}; echo "vllm is serving"; {vllm_then}')
    fake_bin(bin_dir, "sleep", f'echo "sleep $*" >> {calls}')
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VLLM_API_KEY": key}
    subprocess.run(["bash", "-c", serve_command(RUN, NAME, root=str(root))], env=env, cwd=tmp_path, timeout=30)
    run_dir = root / "runs" / RUN
    for _ in range(50):  # the serve log is written by a tee that may outlive bash by a moment
        if prepare_exit or not key or (run_dir / "serve.log").exists() and "vllm is serving" in (run_dir / "serve.log").read_text():
            break
        time.sleep(0.05)
    return calls.read_text(), run_dir


def test_a_failed_prepare_is_logged_marked_and_parked_instead_of_restarting(tmp_path):
    calls, run_dir = boot(tmp_path, prepare_exit=1)
    assert (run_dir / "PREPARE_FAILED").exists()
    log = (run_dir / "prepare.log").read_text()
    assert "prepare says hello" in log and "prepare error" in log
    assert "sleep infinity" in calls and "vllm" not in calls


def test_a_good_prepare_serves_and_logs_vllm(tmp_path):
    calls, run_dir = boot(tmp_path, prepare_exit=0, vllm_then="/bin/sleep 1")
    assert "prepare says hello" in (run_dir / "prepare.log").read_text()
    assert not (run_dir / "PREPARE_FAILED").exists()
    assert f"(cwd {tmp_path / 'volume' / 'code'})" in calls
    assert "vllm serve" in calls and "8000/health" in calls
    assert "vllm is serving" in (run_dir / "serve.log").read_text()


def one_run_failures(run_dir):
    return [line for line in (run_dir / "serve.log").read_text().splitlines() if line.startswith("SERVE_FAILED")]


def test_one_run_never_exits_the_container_it_parks_with_a_serve_failed_reason():
    command = serve_command(RUN, NAME)
    assert "exec vllm" not in command and "exit 1" not in command and "SERVE_FAILED" in command
    assert command.endswith(f'wait "$SERVE_PID"; serve_fail "{RUN} exited with $?"')  # nothing after but parking
    assert command.count("SERVE_DEADLINE=$((SECONDS + 1800))") == 1  # the server gets 30 minutes to answer /health
    subprocess.run(["bash", "-n", "-c", command], check=True)


def test_one_run_server_that_exits_before_healthy_is_logged_and_parks(tmp_path):
    calls, run_dir = boot(tmp_path, prepare_exit=0, healthy=False, vllm_then="exit 3")
    assert one_run_failures(run_dir) == [f"SERVE_FAILED: {RUN} exited with 3 before it was healthy on 8000"]
    assert calls.splitlines()[-1] == "sleep infinity"


def test_one_run_server_not_healthy_within_the_cap_is_logged_and_parks(tmp_path, monkeypatch):
    from benchmarks.runpod import pod

    monkeypatch.setattr(pod, "SERVE_HEALTH_TIMEOUT_S", 0)
    calls, run_dir = boot(tmp_path, prepare_exit=0, healthy=False, vllm_then="/bin/sleep 3")
    assert one_run_failures(run_dir) == [f"SERVE_FAILED: {RUN} was not healthy on 8000 within 0 s"]
    assert calls.splitlines()[-1] == "sleep infinity"


def test_one_run_server_that_exits_later_is_logged_with_its_code_and_parks(tmp_path):
    calls, run_dir = boot(tmp_path, prepare_exit=0, vllm_then="exit 7")
    assert one_run_failures(run_dir) == [f"SERVE_FAILED: {RUN} exited with 7"]
    assert calls.splitlines()[-1] == "sleep infinity"


def test_an_empty_key_parks_the_pod_before_anything_runs(tmp_path):
    calls, run_dir = boot(tmp_path, prepare_exit=0, key="")
    assert (run_dir / "PREPARE_FAILED").exists() and "VLLM_API_KEY is empty" in (run_dir / "prepare.log").read_text()
    assert calls.splitlines() == ["sleep infinity"]


def test_the_serve_command_refuses_a_run_name_that_escapes_the_volume():
    with pytest.raises(ValueError):
        serve_command("../etc", NAME)
    with pytest.raises(ValueError):
        serve_command(RUN, "name; rm -rf /")


def test_the_proxy_url_is_the_pods_port_8000():
    assert serve_url("abc123") == "https://abc123-8000.proxy.runpod.net/v1"


RUN_2 = "reader-v8-gemma4-e4b"
NAME_2 = "rembero-reader-v8"


def vllm_invocations(command):
    """Each `vllm serve ...` in the command as its words, up to its redirection."""
    words = shlex.split(command)
    starts = [i for i, word in enumerate(words) if word == "vllm" and words[i + 1] == "serve"]
    return [words[start:words.index("2>&1", start)] for start in starts]


def test_the_one_run_command_keeps_todays_vllm_arguments_inside_the_park_wrapper():
    command = serve_command(RUN, NAME)
    assert command.count("vllm serve") == 1 and "8001" not in command and "RUN_DIR_2" not in command
    assert command.count("benchmarks.runpod.prepare_reader") == 1
    [vllm] = vllm_invocations(command)
    assert vllm[:vllm.index(">")] == [
        "vllm", "serve", f"/workspace/runs/{RUN}/merged-text",
        "--dtype", "bfloat16", "--max-model-len", "12288", "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.92", "--reasoning-parser", "gemma4",
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
        "--served-model-name", NAME, "--host", "0.0.0.0", "--port", "8000", "--api-key", "$VLLM_API_KEY",
    ]
    assert command.startswith(f"set -o pipefail; export PYTHONUNBUFFERED=1; RUN_DIR=/workspace/runs/{RUN}; ")
    assert f'vllm serve /workspace/runs/{RUN}/merged-text' in command and "& SERVE_PID=$!" in command


def test_two_runs_serve_one_vllm_each_on_8000_and_8001_with_the_same_flags_and_key():
    command = serve_command(RUN, NAME, second=(RUN_2, NAME_2))
    first, second = vllm_invocations(command)
    for vllm, run, name, port in ((first, RUN, NAME, "8000"), (second, RUN_2, NAME_2, "8001")):
        def flag(option):
            return vllm[vllm.index(option) + 1]

        assert vllm[:3] == ["vllm", "serve", f"/workspace/runs/{run}/merged-text"]
        assert flag("--dtype") == "bfloat16" and flag("--max-model-len") == "12288"
        assert flag("--max-num-seqs") == "32" and flag("--gpu-memory-utilization") == "0.44"
        assert flag("--reasoning-parser") == "gemma4"
        assert json.loads(flag("--default-chat-template-kwargs")) == {"enable_thinking": False}
        assert flag("--served-model-name") == name
        assert flag("--host") == "0.0.0.0" and flag("--port") == port
        assert flag("--api-key") == "$VLLM_API_KEY"
    assert command.count('--api-key "$VLLM_API_KEY"') == 2 and "vllm-key" not in command


def test_two_runs_prepare_both_before_either_server_starts_each_with_its_own_logs():
    command = serve_command(RUN, NAME, second=(RUN_2, NAME_2))
    first_serve = command.index("vllm serve")
    assert command.index(f"--run {RUN} ") < command.index(f"--run {RUN_2} ") < first_serve
    assert command.index('test -n "$VLLM_API_KEY"') < command.index("benchmarks.runpod.prepare_reader")
    assert command.count("uv pip install") == 1  # PEFT is installed once, before the first prepare
    assert 'tee -a "$RUN_DIR/prepare.log"' in command and 'tee -a "$RUN_DIR_2/prepare.log"' in command
    assert 'tee -a "$RUN_DIR/serve.log"' in command and 'tee -a "$RUN_DIR_2/serve.log"' in command
    assert f"RUN_DIR=/workspace/runs/{RUN};" in command and f"RUN_DIR_2=/workspace/runs/{RUN_2};" in command
    subprocess.run(["bash", "-n", "-c", command], check=True)


def test_two_runs_refuse_a_bad_or_repeated_second_run():
    with pytest.raises(ValueError):
        serve_command(RUN, NAME, second=("../etc", NAME_2))
    with pytest.raises(ValueError):
        serve_command(RUN, NAME, second=(RUN_2, "name; rm -rf /"))
    with pytest.raises(ValueError):
        serve_command(RUN, NAME, second=(RUN, NAME_2))
    with pytest.raises(ValueError):
        serve_command(RUN, NAME, second=(RUN_2, NAME))


def boot_two(tmp_path, failing_run=None, key="test-key", vllm_then="", healthy=True, unhealthy_port=None):
    """Boot the two-run command with stubs; the python3 stub fails when its arguments name failing_run.
    vllm_then is shell run by the vllm stub after it logs (e.g. stay up, or exit with a code);
    healthy=False makes the /health probe (python3 -c) fail."""
    bin_dir, root = tmp_path / "bin", tmp_path / "volume"
    bin_dir.mkdir()
    (root / "code").mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    fail_on = f'case "$* " in *"--run {failing_run} "*) exit 1;; esac; ' if failing_run else ""
    probe = 'case "$1" in -c) exit 1;; esac; ' if not healthy else ""
    if unhealthy_port:
        probe = f'case "$*" in -c*{unhealthy_port}/health*) exit 1;; esac; '
    fake_bin(bin_dir, "uv", f'echo "uv $*" >> {calls}')
    fake_bin(bin_dir, "python3", f'echo "python3 $*" >> {calls}; {probe}echo "prepare says $*"; {fail_on}exit 0')
    fake_bin(bin_dir, "vllm", f'echo "vllm $*" >> {calls}; echo "vllm is serving $*"; {vllm_then}')
    fake_bin(bin_dir, "sleep", f'echo "sleep $*" >> {calls}')
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VLLM_API_KEY": key}
    command = serve_command(RUN, NAME, second=(RUN_2, NAME_2), root=str(root))
    subprocess.run(["bash", "-c", command], env=env, cwd=tmp_path, timeout=30)
    dirs = root / "runs" / RUN, root / "runs" / RUN_2
    for _ in range(50):  # the serve logs are written by tees that may outlive bash by a moment
        logs = [d / "serve.log" for d in dirs]
        if failing_run or not key or all(log.exists() and "vllm is serving" in log.read_text() for log in logs):
            break
        time.sleep(0.05)
    return calls.read_text(), dirs


def test_two_runs_boot_prepares_both_then_serves_both_logging_each_to_its_run(tmp_path):
    calls, (dir_1, dir_2) = boot_two(tmp_path)
    lines = calls.splitlines()
    prepares = [i for i, line in enumerate(lines) if "benchmarks.runpod.prepare_reader" in line]
    serves = [i for i, line in enumerate(lines) if line.startswith("vllm serve")]
    assert len(prepares) == 2 and len(serves) == 2 and max(prepares) < min(serves)
    assert f"--run {RUN} " in lines[prepares[0]] and f"--run {RUN_2} " in lines[prepares[1]]
    assert f"--run {RUN} " in (dir_1 / "prepare.log").read_text()
    second_prepare = (dir_2 / "prepare.log").read_text()
    assert f"--run {RUN_2} " in second_prepare and f"--run {RUN} " not in second_prepare
    assert "--port 8000" in (dir_1 / "serve.log").read_text() and "--port 8001" in (dir_2 / "serve.log").read_text()
    assert "--port 8001" not in (dir_1 / "serve.log").read_text()
    assert not (dir_1 / "PREPARE_FAILED").exists() and not (dir_2 / "PREPARE_FAILED").exists()


def test_two_runs_park_when_the_second_prepare_fails_without_serving_either(tmp_path):
    calls, (dir_1, dir_2) = boot_two(tmp_path, failing_run=RUN_2)
    assert (dir_2 / "PREPARE_FAILED").exists() and not (dir_1 / "PREPARE_FAILED").exists()
    assert "prepare failed" in (dir_2 / "prepare.log").read_text()
    assert "sleep infinity" in calls and "vllm" not in calls


def test_two_runs_park_when_the_first_prepare_fails_before_the_second_starts(tmp_path):
    calls, (dir_1, dir_2) = boot_two(tmp_path, failing_run=RUN)
    assert (dir_1 / "PREPARE_FAILED").exists() and not (dir_2 / "PREPARE_FAILED").exists()
    assert f"--run {RUN_2} " not in calls and "vllm" not in calls and "sleep infinity" in calls


def test_two_runs_with_an_empty_key_mark_both_runs_and_park(tmp_path):
    calls, dirs = boot_two(tmp_path, key="")
    for run_dir in dirs:
        assert (run_dir / "PREPARE_FAILED").exists() and "VLLM_API_KEY is empty" in (run_dir / "prepare.log").read_text()
    assert calls.splitlines() == ["sleep infinity"]


def serve_failures(dirs):
    return [[line for line in (d / "serve.log").read_text().splitlines() if line.startswith("SERVE_FAILED")]
            for d in dirs]


def test_two_runs_never_exit_the_container_they_park_with_a_serve_failed_reason():
    command = serve_command(RUN, NAME, second=(RUN_2, NAME_2))
    assert "exit 1" not in command and "SERVE_FAILED" in command
    assert command.rsplit("; ", 1)[-1].startswith("serve_fail ")  # nothing after the watch but parking
    assert command.count("SERVE_DEADLINE=$((SECONDS + 1800))") == 2  # each server gets 30 minutes to answer /health
    assert "8001/health" in command
    subprocess.run(["bash", "-n", "-c", command], check=True)


def test_a_first_server_that_exits_before_healthy_is_logged_to_both_runs_and_parks(tmp_path):
    calls, dirs = boot_two(tmp_path, healthy=False, vllm_then='case "$*" in *"--port 8000"*) exit 3;; esac')
    assert serve_failures(dirs) == [[f"SERVE_FAILED: {RUN} exited with 3 before it was healthy on 8000"]] * 2
    assert "--port 8001" not in calls and calls.splitlines()[-1] == "sleep infinity"


def test_a_first_server_not_healthy_within_the_cap_is_logged_to_both_runs_and_parks(tmp_path, monkeypatch):
    from benchmarks.runpod import pod

    monkeypatch.setattr(pod, "SERVE_HEALTH_TIMEOUT_S", 0)
    calls, dirs = boot_two(tmp_path, healthy=False, vllm_then="/bin/sleep 3")
    assert serve_failures(dirs) == [[f"SERVE_FAILED: {RUN} was not healthy on 8000 within 0 s"]] * 2
    assert "--port 8001" not in calls and calls.splitlines()[-1] == "sleep infinity"


def test_a_second_server_not_healthy_within_the_cap_is_logged_to_both_runs_and_parks(tmp_path, monkeypatch):
    from benchmarks.runpod import pod

    monkeypatch.setattr(pod, "SERVE_HEALTH_TIMEOUT_S", 0)
    calls, dirs = boot_two(tmp_path, unhealthy_port=8001, vllm_then="/bin/sleep 3")
    assert serve_failures(dirs) == [[f"SERVE_FAILED: {RUN_2} was not healthy on 8001 within 0 s"]] * 2
    assert "8001/health" in calls and calls.splitlines()[-1] == "sleep infinity"


def test_a_second_server_that_exits_before_healthy_is_logged_to_both_runs_and_parks(tmp_path):
    calls, dirs = boot_two(tmp_path, unhealthy_port=8001,
                           vllm_then='case "$*" in *"--port 8001"*) exit 5;; *) /bin/sleep 3;; esac')
    assert serve_failures(dirs) == [[f"SERVE_FAILED: {RUN_2} exited with 5 before it was healthy on 8001"]] * 2
    assert calls.splitlines()[-1] == "sleep infinity"


def test_a_server_that_exits_later_is_logged_to_both_runs_with_its_code_and_parks(tmp_path):
    calls, dirs = boot_two(tmp_path, vllm_then='case "$*" in *"--port 8001"*) exit 7;; *) /bin/sleep 3;; esac')
    assert serve_failures(dirs) == [[f"SERVE_FAILED: {RUN_2} exited with 7"]] * 2
    assert calls.splitlines()[-1] == "sleep infinity"


def test_create_serve_with_two_runs_refuses_a_gpu_under_80_gb_before_any_api_call(monkeypatch, capsys):
    from benchmarks.runpod import pod

    called = []
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_API_KEY": "k", "VLLM_API_KEY": "v", "RUNPOD_VOLUME_ID": "vol"})
    monkeypatch.setattr(pod, "api", lambda *args, **kwargs: called.append(args) or {"id": "pod9"})
    monkeypatch.setattr(pod, "upsert_env", lambda path, updates: called.append(("upsert", updates)))
    two = ["create-serve", "--run", RUN, "--served-name", NAME, "--run", RUN_2, "--served-name", NAME_2]
    for gpu in (None, "NVIDIA GeForce RTX 5090", "NVIDIA L40S", "NVIDIA RTX A6000"):
        with pytest.raises(SystemExit) as refused:
            pod.main(two + ([] if gpu is None else ["--gpu", gpu]))
        assert refused.value.code != 0
        assert "80 GB" in capsys.readouterr().err
    assert called == []
    assert pod.LARGE_GPUS == ("NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB",
                              "NVIDIA A100 80GB PCIe", "NVIDIA H200", "NVIDIA B200")
    for gpu in pod.LARGE_GPUS:
        pod.main(two + ["--gpu", gpu])
    assert [c[0] for c in called if c[0] == "POST"] == ["POST"] * len(pod.LARGE_GPUS)
    pod.main(["create-serve", "--run", RUN, "--served-name", NAME])  # one run keeps the 5090 default


def test_the_two_run_payload_exposes_both_ports_and_starts_the_two_run_command():
    p = payload(second=(RUN_2, NAME_2))
    assert p["ports"] == ["8000/http", "8001/http", "22/tcp"]
    assert p["dockerStartCmd"] == [serve_command(RUN, NAME, second=(RUN_2, NAME_2))]
    assert p["name"].startswith("rembero-serve-") and len(p["name"]) <= 60
    assert payload()["ports"] == ["8000/http", "22/tcp"]


def test_the_second_readers_proxy_url_is_port_8001():
    assert serve_url("abc123", 8001) == "https://abc123-8001.proxy.runpod.net/v1"


def test_create_serve_takes_one_or_two_run_and_name_pairs(monkeypatch):
    from benchmarks.runpod import pod

    seen = []
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_API_KEY": "k"})
    monkeypatch.setattr(pod, "create_serve", lambda a, env, key: seen.append((a.run, a.served_name, a.gpu)))
    pod.main(["create-serve", "--run", RUN, "--served-name", NAME])
    pod.main(["create-serve", "--run", RUN, "--served-name", NAME, "--run", RUN_2, "--served-name", NAME_2,
              "--gpu", "NVIDIA H100 NVL"])
    assert seen == [([RUN], [NAME], "NVIDIA GeForce RTX 5090"), ([RUN, RUN_2], [NAME, NAME_2], "NVIDIA H100 NVL")]
    for bad in (["--run", RUN, "--served-name", NAME, "--run", RUN_2],
                ["--run", RUN, "--served-name", NAME, "--run", RUN_2, "--served-name", NAME_2,
                 "--run", "c", "--served-name", "d"]):
        with pytest.raises(SystemExit):
            pod.main(["create-serve", *bad])
    assert len(seen) == 2


def test_create_serve_with_two_runs_records_both_urls(monkeypatch, tmp_path):
    from argparse import Namespace

    from benchmarks.runpod import pod

    env_file, bodies = tmp_path / ".env", []
    env_file.write_text("VLLM_API_KEY=vllm-key-123\n")
    monkeypatch.setattr(pod, "ENV_FILE", env_file)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr(pod, "api", lambda method, path, key, body=None: bodies.append(body) or {"id": "pod9"})
    a = Namespace(run=[RUN, RUN_2], served_name=[NAME, NAME_2], volume="vol-abc", gpu="NVIDIA H100 NVL",
                  datacenter="EUR-NO-1", cloud="COMMUNITY")
    pod.create_serve(a, read_env(env_file), "runpod-key")
    [body] = bodies
    assert body["ports"] == ["8000/http", "8001/http", "22/tcp"] and body["gpuTypeIds"] == ["NVIDIA H100 NVL"]
    assert body["dockerStartCmd"] == [serve_command(RUN, NAME, second=(RUN_2, NAME_2))]
    values = read_env(env_file)
    assert values["RUNPOD_SERVE_POD_ID"] == "pod9"
    assert values["RUNPOD_SERVE_URL"] == "https://pod9-8000.proxy.runpod.net/v1"
    assert values["RUNPOD_SERVE_URL_2"] == "https://pod9-8001.proxy.runpod.net/v1"

    a.run, a.served_name = [RUN], [NAME]
    pod.create_serve(a, read_env(env_file), "runpod-key")
    assert bodies[-1]["dockerStartCmd"] == [serve_command(RUN, NAME)]
    assert read_env(env_file)["RUNPOD_SERVE_URL_2"] == ""  # a one-run pod leaves no stale second URL


def test_wait_can_poll_the_second_readers_port(monkeypatch):
    from benchmarks.runpod import pod

    polled = []
    monkeypatch.setattr(pod, "read_env", lambda path: {"VLLM_API_KEY": "v", "RUNPOD_SERVE_POD_ID": "pod9",
                                                       "RUNPOD_API_KEY": "k"})
    monkeypatch.setattr(pod, "served_models", lambda url, key: polled.append(url) or [NAME_2])
    pod.main(["wait", "--served-name", NAME_2, "--port", "8001"])
    pod.main(["wait", "--served-name", NAME_2])
    assert polled == ["https://pod9-8001.proxy.runpod.net/v1", "https://pod9-8000.proxy.runpod.net/v1"]


def never_up(monkeypatch, pod, env):
    calls = []
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(pod, "read_env", lambda path: env)
    monkeypatch.setattr(pod.time, "sleep", lambda s: None)

    def down(url, key):
        calls.append(("poll", url))
        raise pod.urllib.error.URLError("refused")

    monkeypatch.setattr(pod, "served_models", down)
    monkeypatch.setattr(pod, "api", lambda method, path, key, body=None: calls.append((method, path, key)))
    return calls


WAIT_ENV = {"VLLM_API_KEY": "v", "RUNPOD_SERVE_POD_ID": "pod9", "RUNPOD_API_KEY": "k"}


def test_wait_stops_the_pod_on_timeout_and_exits_non_zero_saying_so(monkeypatch, capsys):
    from benchmarks.runpod import pod

    calls = never_up(monkeypatch, pod, WAIT_ENV)
    with pytest.raises(SystemExit) as timed_out:
        pod.main(["wait", "--served-name", NAME, "--timeout", "1"])
    assert timed_out.value.code not in (0, None)
    assert calls[-1] == ("POST", "/pods/pod9/stop", "k")
    assert ("poll", "https://pod9-8000.proxy.runpod.net/v1") in calls
    said = str(timed_out.value.code) + capsys.readouterr().out
    assert "stopped pod pod9" in said


def test_wait_with_no_stop_on_timeout_leaves_the_pod_running(monkeypatch, capsys):
    from benchmarks.runpod import pod

    calls = never_up(monkeypatch, pod, {k: v for k, v in WAIT_ENV.items() if k != "RUNPOD_API_KEY"})
    with pytest.raises(SystemExit) as timed_out:
        pod.main(["wait", "--served-name", NAME, "--timeout", "1", "--no-stop-on-timeout"])
    assert timed_out.value.code not in (0, None)
    assert [c for c in calls if c[0] != "poll"] == []
    assert "still running" in str(timed_out.value.code)


def test_wait_that_would_stop_on_timeout_needs_a_runpod_key_before_polling(monkeypatch):
    from benchmarks.runpod import pod

    calls = never_up(monkeypatch, pod, {k: v for k, v in WAIT_ENV.items() if k != "RUNPOD_API_KEY"})
    with pytest.raises(SystemExit) as refused:
        pod.main(["wait", "--served-name", NAME, "--timeout", "1"])
    assert "RUNPOD_API_KEY" in str(refused.value.code) and calls == []


def test_wait_does_not_stop_a_pod_that_comes_up(monkeypatch):
    from benchmarks.runpod import pod

    calls = []
    monkeypatch.setattr(pod, "read_env", lambda path: WAIT_ENV)
    monkeypatch.setattr(pod, "served_models", lambda url, key: [NAME])
    monkeypatch.setattr(pod, "api", lambda *args, **kwargs: calls.append(args))
    pod.main(["wait", "--served-name", NAME])
    assert calls == []


def test_create_serve_prints_the_pod_id_before_writing_env(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from benchmarks.runpod import pod

    monkeypatch.setenv("VLLM_API_KEY", "vllm-key-123")
    monkeypatch.setattr(pod, "api", lambda method, path, key, body=None: {"id": "pod9", "costPerHr": 2.59})

    def broken(path, updates):
        raise OSError("disk full")

    monkeypatch.setattr(pod, "upsert_env", broken)
    a = Namespace(run=[RUN], served_name=[NAME], volume="vol-abc", gpu="NVIDIA H100 NVL",
                  datacenter="US-GA-2", cloud="COMMUNITY")
    with pytest.raises(OSError):
        pod.create_serve(a, {}, "runpod-key")
    assert "pod9" in capsys.readouterr().out


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


def tiny_weights(directory):
    import torch
    from safetensors.torch import save_file

    save_file({"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
              str(directory / "model.safetensors"))


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
        tiny_weights(text)
        return text

    def restore(run_dir, base_model):
        calls.append("restore" if not (run_dir / "merged").exists() else "restore-with-merged-still-on-disk")
        return 42

    monkeypatch.setattr(prepare_reader, "base_model_size", lambda base_model: (10**6, True))
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
    assert json.loads((text_dir / "fingerprint.json").read_text())["tensor_count"] == 1
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


def test_merged_is_deleted_before_the_restore_rewrites_the_weights(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")
    assert calls == ["merge", "export", "restore"]


def completed_merge(root):
    merged = root / "runs" / RUN / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_text("weights")
    (merged / prepare_reader.MERGED_MARKER).write_text("")
    return merged


def test_a_run_that_died_after_the_merge_reruns_only_export_and_restore(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    completed_merge(tmp_path)
    stale = tmp_path / "runs" / RUN / "merged-text"
    stale.mkdir(parents=True)
    (stale / "config.json").write_text("{}")

    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    assert calls == ["export", "restore"]
    assert prepare_reader.is_prepared(tmp_path / "runs" / RUN)


def test_a_merge_that_never_finished_is_redone(tmp_path, monkeypatch):
    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    merged = completed_merge(tmp_path)
    (merged / prepare_reader.MERGED_MARKER).unlink()

    prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    assert calls == ["merge", "export", "restore"]


def test_too_little_disk_fails_fast_naming_free_and_needed_space(tmp_path, monkeypatch):
    import collections

    calls: list[str] = []
    fake_steps(monkeypatch, calls)
    adapter(tmp_path)
    usage = collections.namedtuple("usage", "total used free")
    asked = []
    monkeypatch.setattr(prepare_reader, "base_model_size", lambda base_model: (16 * 10**9, False))
    monkeypatch.setattr(prepare_reader.shutil, "disk_usage", lambda path: asked.append(path) or usage(60 * 10**9, 50 * 10**9, 10 * 10**9))

    with pytest.raises(SystemExit) as error:
        prepare_reader.prepare(tmp_path, RUN, "google/gemma-4-E4B-it")

    message = str(error.value)
    assert "10.0 GB free" in message and "48.0 GB needed" in message
    assert calls == [] and asked == [tmp_path]
    assert not prepare_reader.is_prepared(tmp_path / "runs" / RUN)


def test_the_space_needed_counts_a_cached_base_once_and_a_finished_merge_as_already_written():
    gb = 10**9
    # a fresh volume: cache + merged + merged-text at the peak; the 60 GB volume that ran out held a
    # fourth copy (merged/ kept through the restore), which prepare no longer does
    assert prepare_reader.space_needed(16 * gb, cached=False, merged_on_disk=0) == 48 * gb
    assert prepare_reader.space_needed(16 * gb, cached=True, merged_on_disk=0) == 32 * gb
    assert prepare_reader.space_needed(16 * gb, cached=True, merged_on_disk=16 * gb) == 16 * gb


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


def test_every_request_carries_a_user_agent_cloudflare_accepts_and_asks_for_json():
    """Cloudflare answers urllib's default agent with 403 / error code 1010."""
    from benchmarks.runpod.pod import USER_AGENT, build_request

    request = build_request("https://rest.runpod.io/v1/pods", "k", method="POST", body={"a": 1})
    assert request.get_header("User-agent") == USER_AGENT == "rembero-runpod/1.0 (+https://github.com/rahult/remembero)"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") == "Bearer k"
    assert request.get_header("Content-type") == "application/json" and json.loads(request.data) == {"a": 1}
    assert request.get_method() == "POST"
    bare = build_request("https://pod-8000.proxy.runpod.net/v1/models", "k")
    assert bare.get_method() == "GET" and bare.data is None and bare.get_header("User-agent") == USER_AGENT


def test_the_rest_calls_and_the_wait_poll_both_send_those_headers(monkeypatch):
    import io

    from benchmarks.runpod import pod

    sent = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(request)
        return Response(json.dumps({"id": "pod1", "data": [{"id": NAME}]}).encode())

    monkeypatch.setattr(pod.urllib.request, "urlopen", fake_urlopen)
    pod.api("POST", "/pods", "runpod-key", {"x": 1})
    pod.api("DELETE", "/pods/pod1", "runpod-key")
    assert pod.served_models(pod.serve_url("pod1"), "vllm-key") == [NAME]
    assert [r.full_url for r in sent] == ["https://rest.runpod.io/v1/pods", "https://rest.runpod.io/v1/pods/pod1",
                                         "https://pod1-8000.proxy.runpod.net/v1/models"]
    for request in sent:
        assert request.get_header("User-agent") == pod.USER_AGENT
        assert request.get_header("Accept") == "application/json"


TRAIN_RUN = "reader-v7-gemma4-e4b"


def readme_bootstrap_line():
    """The `sh` block under "Bootstrap template (no image push)" in README.md, as one line."""
    from pathlib import Path

    readme = (Path(__file__).parent / "README.md").read_text()
    blocks = [b.split("```")[0] for b in readme.split("```sh\n")]
    [bootstrap] = [b.strip() for b in blocks if "llama-quantize" in b and "pip install" in b]
    return bootstrap


def train_command(**overrides):
    from benchmarks.runpod.pod import train_command as build

    return build(overrides.pop("run", TRAIN_RUN), **overrides)


def test_the_training_defaults_are_the_handlers_defaults():
    from benchmarks.runpod import handler, pod

    assert pod.QUANTS == handler.QUANTS
    assert pod.TRAIN_DEFAULTS == {k: handler.DEFAULTS[k] for k in pod.TRAIN_DEFAULTS}


def test_the_train_command_bootstraps_then_trains_then_parks():
    from benchmarks.runpod import pod

    command = train_command()
    assert pod.TRAIN_BOOTSTRAP in readme_bootstrap_line()  # the bootstrap of record, unchanged
    assert pod.TRAIN_BOOTSTRAP in command
    assert (command.index("mkdir -p") < command.index("pip install -q")
            < command.index("llama-quantize") < command.index("benchmarks.runpod.train_pod"))
    assert command.endswith("exec sleep infinity")  # the pod parks whatever the outcome
    assert "TRAIN_FAILED" in command and "TRAIN_DONE" in command
    assert command.index("TRAIN_FAILED") < command.index("benchmarks.runpod.train_pod")


def test_the_train_command_runs_train_pod_from_the_code_directory_with_the_run_flags():
    command = train_command(max_length=4096, batch_size=2, grad_accum=8, quant="Q6_K", liger=False)
    words = shlex.split(command)
    # the last word of the group carries bash's `;`, which shlex keeps attached
    train = [word.rstrip(";") for word in words[words.index("-m") - 2:]]
    assert train[:6] == ["python3", "-u", "-m", "benchmarks.runpod.train_pod", "--run", TRAIN_RUN]

    def flag(name):
        return train[train.index(name) + 1]

    assert flag("--root") == "/workspace" and flag("--local-root") == "/root"
    assert flag("--max-length") == "4096" and flag("--batch-size") == "2" and flag("--grad-accum") == "8"
    assert flag("--quant") == "Q6_K" and "--no-liger" in train
    assert "cd /workspace/code" in command
    assert command.index("cd /workspace/code") < command.index("python3 -u -m benchmarks.runpod.train_pod")
    assert "--no-liger" not in train_command()
    assert "--max-length 8192" in train_command()


def test_the_train_command_tees_everything_to_the_runs_log_on_the_volume():
    command = train_command()
    assert f"RUN_DIR=/workspace/runs/{TRAIN_RUN};" in command
    assert command.count('tee -a "$RUN_DIR/train.log"') >= 3  # bootstrap, training, and the final marker
    assert 'mkdir -p "$RUN_DIR"' in command
    assert "pipefail" in command


def test_the_train_command_is_valid_bash_and_refuses_a_bad_run_or_quant():
    subprocess.run(["bash", "-n", "-c", train_command()], check=True)
    for bad in ("../etc", "run; rm -rf /"):
        with pytest.raises(ValueError):
            train_command(run=bad)
    with pytest.raises(ValueError):
        train_command(quant="Q3_K_S")


def train_payload(**overrides):
    from benchmarks.runpod.pod import build_train_payload

    args = dict(run=TRAIN_RUN, volume_id="vol-abc", gpu="NVIDIA H100 NVL", datacenter="US-GA-2",
                public_key="ssh-ed25519 AAAA me@mac")
    return build_train_payload(**{**args, **overrides})


def test_the_train_payload_asks_for_the_training_image_80_gb_of_disk_and_the_volume():
    from benchmarks.runpod import pod

    p = train_payload()
    assert p["imageName"] == pod.TRAIN_IMAGE == "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    assert p["containerDiskInGb"] == 80  # merged/, merged-text/ and the f16 GGUF are built there
    assert p["networkVolumeId"] == "vol-abc" and p["volumeMountPath"] == "/workspace"
    assert p["gpuTypeIds"] == ["NVIDIA H100 NVL"] and p["dataCenterIds"] == ["US-GA-2"]
    assert p["computeType"] == "GPU" and p["gpuCount"] == 1 and p["cloudType"] == "COMMUNITY"
    assert p["ports"] == ["22/tcp"]  # nothing is served from the training pod
    assert p["env"] == {"HF_HOME": "/workspace/hf", "PYTHONPATH": "/workspace/code",
                        "PUBLIC_KEY": "ssh-ed25519 AAAA me@mac"}
    assert p["dockerEntrypoint"] == ["bash", "-lc"]
    assert p["dockerStartCmd"] == [train_command()]
    assert p["name"].startswith("rembero-train-") and len(p["name"]) <= 60
    assert "VLLM_API_KEY" not in json.dumps(p)


def test_the_train_payload_takes_the_run_flags_and_refuses_a_bad_cloud():
    assert "--max-length 4096" in train_payload(max_length=4096)["dockerStartCmd"][0]
    assert train_payload(cloud_type="SECURE")["cloudType"] == "SECURE"
    assert "PUBLIC_KEY" not in train_payload(public_key=None)["env"]
    with pytest.raises(ValueError):
        train_payload(cloud_type="SPOT")


def test_no_hugging_face_or_modal_credential_reaches_the_train_payload(monkeypatch):
    for name, value in {"HF_TOKEN": "hf-secret-1", "MODAL_TOKEN_ID": "modal-secret-1",
                        "RUNPOD_API_KEY": "runpod-secret"}.items():
        monkeypatch.setenv(name, value)
    text = json.dumps(train_payload())
    for secret in ("hf-secret", "modal-secret", "runpod-secret", "HF_TOKEN", "MODAL"):
        assert secret not in text


def test_create_train_prints_the_pod_id_before_writing_env_and_says_to_stop_the_pod(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from benchmarks.runpod import pod

    bodies, env_file = [], tmp_path / ".env"
    monkeypatch.setattr(pod, "ENV_FILE", env_file)
    monkeypatch.setattr(pod, "api", lambda method, path, key, body=None: bodies.append((method, path, body)) or {"id": "pod9", "costPerHr": 3.29})
    a = Namespace(run=TRAIN_RUN, volume="vol-abc", gpu="NVIDIA H100 NVL", datacenter="US-GA-2", cloud="COMMUNITY",
                  max_length=8192, batch_size=4, grad_accum=16, quant="Q8_0", no_liger=False)
    pod.create_train(a, {}, "runpod-key")
    [(method, path, body)] = bodies
    assert (method, path) == ("POST", "/pods")
    assert body["dockerStartCmd"] == [train_command()]
    said = capsys.readouterr().out
    assert "pod9" in said and "stop" in said and "train-log" in said
    assert read_env(env_file)["RUNPOD_TRAIN_POD_ID"] == "pod9"

    def broken(path, updates):
        raise OSError("disk full")

    monkeypatch.setattr(pod, "upsert_env", broken)
    with pytest.raises(OSError):
        pod.create_train(a, {}, "runpod-key")
    assert "pod9" in capsys.readouterr().out  # the id reaches stdout before .env can fail


def test_create_train_needs_a_volume_and_makes_no_call_without_one(monkeypatch):
    from argparse import Namespace

    from benchmarks.runpod import pod

    called = []
    monkeypatch.setattr(pod, "api", lambda *args, **kwargs: called.append(args))
    a = Namespace(run=TRAIN_RUN, volume=None, gpu="NVIDIA H100 NVL", datacenter="US-GA-2", cloud="COMMUNITY",
                  max_length=8192, batch_size=4, grad_accum=16, quant="Q8_0", no_liger=False)
    monkeypatch.delenv("RUNPOD_VOLUME_ID", raising=False)
    with pytest.raises(SystemExit):
        pod.create_train(a, {}, "runpod-key")
    assert called == []


def test_create_train_passes_its_flags_through_main(monkeypatch):
    from benchmarks.runpod import pod

    seen = []
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_API_KEY": "k"})
    monkeypatch.setattr(pod, "create_train", lambda a, env, key: seen.append(vars(a)))
    pod.main(["create-train", "--run", TRAIN_RUN])
    pod.main(["create-train", "--run", TRAIN_RUN, "--gpu", "NVIDIA H100 NVL", "--datacenter", "US-GA-2",
              "--max-length", "4096", "--batch-size", "2", "--grad-accum", "8", "--quant", "Q6_K", "--no-liger"])
    first, second = seen
    assert first["run"] == TRAIN_RUN and first["max_length"] == 8192 and first["no_liger"] is False
    assert first["quant"] == "Q8_0" and first["cloud"] == "COMMUNITY"
    assert second["gpu"] == "NVIDIA H100 NVL" and second["datacenter"] == "US-GA-2"
    assert second["max_length"] == 4096 and second["batch_size"] == 2 and second["grad_accum"] == 8
    assert second["quant"] == "Q6_K" and second["no_liger"] is True


def test_train_log_prints_the_tail_of_the_volume_log(monkeypatch, capsys):
    from benchmarks.runpod import pod, volume

    asked = []
    monkeypatch.setattr(volume, "read_text", lambda remote: asked.append(remote) or "\n".join(f"line {i}" for i in range(100)))
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_VOLUME_ID": "vol-abc", "RUNPOD_DATACENTER": "US-GA-2"})
    pod.main(["train-log", "--run", TRAIN_RUN])
    out = capsys.readouterr().out.splitlines()
    assert asked == [f"runs/{TRAIN_RUN}/train.log"]
    assert out[-1] == "line 99" and len(out) == 40  # --tail defaults to 40
    pod.main(["train-log", "--run", TRAIN_RUN, "--tail", "3"])
    assert capsys.readouterr().out.splitlines() == ["line 97", "line 98", "line 99"]
    assert os.environ["RUNPOD_VOLUME_ID"] == "vol-abc"  # the .env settings reach volume.py
    with pytest.raises(SystemExit):
        pod.main(["train-log", "--run", "../etc"])


def test_train_log_needs_no_runpod_api_key(monkeypatch, capsys):
    from benchmarks.runpod import pod, volume

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(volume, "read_text", lambda remote: "training\n")
    monkeypatch.setattr(pod, "read_env", lambda path: {"RUNPOD_VOLUME_ID": "vol-abc"})
    pod.main(["train-log", "--run", TRAIN_RUN])
    assert "training" in capsys.readouterr().out
