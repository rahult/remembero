import json
from pathlib import Path

import pytest

from benchmarks.train.reader_lora import to_prompt_completion, train_lora

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


def write_rows(path: Path, n: int = 6) -> None:
    rows = [
        {"messages": [
            {"role": "system", "content": "Answer only from the supplied history."},
            {"role": "user", "content": f"History chats:\n\n### Retrieved session 1\nUSER: my number is {i}.\n\nQuestion: what is my number?"},
            {"role": "assistant", "content": f"Your number is {i}."},
        ]}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_to_prompt_completion_puts_loss_on_the_assistant_turn(tmp_path):
    write_rows(tmp_path / "conversations.jsonl", 2)
    rows = to_prompt_completion(str(tmp_path / "conversations.jsonl"))
    assert rows[0]["completion"] == [{"role": "assistant", "content": "Your number is 0."}]
    assert [m["role"] for m in rows[0]["prompt"]] == ["system", "user"]


@pytest.mark.slow
def test_train_lora_runs_on_cpu_and_resumes(tmp_path):
    data = tmp_path / "data"; data.mkdir(); write_rows(data / "conversations.jsonl")
    run = tmp_path / "run"
    saves: list[int] = []
    metrics = train_lora(data, run, TINY, epochs=1, batch_size=1, grad_accum=1, max_length=128,
                         merge=False, on_save=lambda: saves.append(1), save_steps=2)
    assert metrics["train_loss"] is not None and (run / "adapter" / "adapter_config.json").exists()
    assert saves, "the save callback never fired"
    # a second call resumes from the last checkpoint and finishes immediately
    again = train_lora(data, run, TINY, epochs=1, batch_size=1, grad_accum=1, max_length=128, merge=False, save_steps=2)
    assert again["resumed_from"] is not None
    assert again["liger"] is False  # CPU smoke never asks for Liger


def test_restore_dropped_weights_puts_back_only_missing_attention_tensors(tmp_path, monkeypatch):
    import huggingface_hub
    import torch
    from safetensors.torch import load_file, save_file

    from benchmarks.train.reader_lora import restore_dropped_weights

    run_dir = tmp_path / "run"
    text_dir = run_dir / "merged-text"
    text_dir.mkdir(parents=True)
    fine_tuned = torch.full((2, 2), 7.0)
    save_file({"model.layers.0.self_attn.q_proj.weight": fine_tuned}, str(text_dir / "model.safetensors"))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file({
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),  # present: keep the merge's
        "model.language_model.layers.20.self_attn.k_proj.weight": torch.ones(2, 2),  # dropped by transformers
        "model.language_model.layers.20.self_attn.k_norm.weight": torch.ones(2),
        "model.language_model.layers.20.mlp.up_proj.weight": torch.ones(2, 2),        # not attention: skip
        "model.vision_tower.encoder.self_attn.q_proj.weight": torch.ones(2, 2),       # not the language model
    }, str(snapshot / "model-00001-of-00001.safetensors"))
    asked = {}

    def fake_download(repo_id, allow_patterns=None, **kwargs):
        asked.update(repo_id=repo_id, allow_patterns=allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)

    assert restore_dropped_weights(run_dir, "google/gemma-4-E4B-it") == 2

    restored = load_file(str(text_dir / "model.safetensors"))
    assert asked == {"repo_id": "google/gemma-4-E4B-it", "allow_patterns": ["*.safetensors"]}
    assert sorted(restored) == ["model.layers.0.self_attn.q_proj.weight", "model.layers.20.self_attn.k_norm.weight",
                                "model.layers.20.self_attn.k_proj.weight"]
    assert torch.equal(restored["model.layers.0.self_attn.q_proj.weight"], fine_tuned)
