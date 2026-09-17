import json
from pathlib import Path

from benchmarks.train.audit_lengths import audit_dir, audit_file


class WhitespaceTokenizer:
    """One token per role marker and per whitespace-separated word; the generation prompt opens the
    assistant turn with its role marker, so the prompt's ids are a prefix of prompt+completion's."""

    def apply_chat_template(self, messages, tokenize=True, return_dict=True, add_generation_prompt=False, **_):
        ids = []
        for message in messages:
            ids.append(message["role"])
            ids.extend(message["content"].split())
        if add_generation_prompt:
            ids.append("assistant")
        return {"input_ids": ids}


def row(prompt_words: int, completion_words: int, tag: str) -> dict:
    return {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": " ".join([tag] * prompt_words)},
        {"role": "assistant", "content": " ".join(["a"] * completion_words)},
    ]}


def write_split(directory: Path, name: str, rows: list[dict]) -> None:
    (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / f"{name}.meta.jsonl").write_text("".join(json.dumps({"i": i}) + "\n" for i in range(len(rows))))


def test_audit_file_counts_prompt_and_completion_tokens_like_the_trainer(tmp_path):
    # prompt = system + sys + user + words + assistant marker; completion = the assistant's words
    write_split(tmp_path, "conversations.jsonl", [row(5, 2, "x"), row(20, 3, "y"), row(4, 9, "z")])
    report, over = audit_file(tmp_path / "conversations.jsonl", WhitespaceTokenizer(), max_length=20, max_completion=6)
    assert report["rows"] == 3
    assert report["overLength"] == 1  # 24 + 3 = 27 > 20
    assert report["overCompletion"] == 1  # 9 > 6
    assert over == [1, 2]
    assert report["promptTokens"] == {"p50": 9, "p95": 24, "max": 24}
    assert report["completionTokens"] == {"p50": 3, "p95": 9, "max": 9}


def test_write_drops_over_limit_rows_with_their_meta_twins_and_keeps_order(tmp_path):
    rows = [row(3, 1, "keep0"), row(50, 1, "drop"), row(3, 1, "keep1"), row(2, 1, "keep2")]
    write_split(tmp_path, "conversations.jsonl", rows)
    write_split(tmp_path, "heldout.jsonl", [row(60, 1, "drop"), row(1, 1, "keep")])
    (tmp_path / "manifest.json").write_text(json.dumps({"rows": {"train": 4, "heldout": 2}}))

    result = audit_dir(tmp_path, WhitespaceTokenizer(), max_length=20, write=True)

    assert result["conversations.jsonl"]["overLength"] == 1
    assert result["heldout.jsonl"]["overLength"] == 1
    lines = (tmp_path / "conversations.jsonl").read_text().splitlines()
    assert [json.loads(l)["messages"][1]["content"].split()[0] for l in lines] == ["keep0", "keep1", "keep2"]
    meta = [json.loads(l)["i"] for l in (tmp_path / "conversations.jsonl.meta.jsonl").read_text().splitlines()]
    assert meta == [0, 2, 3]
    assert [json.loads(l)["i"] for l in (tmp_path / "heldout.jsonl.meta.jsonl").read_text().splitlines()] == [1]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rows"] == {"train": 4, "heldout": 2}  # untouched; lengthAudit carries the new counts
    audit = manifest["lengthAudit"]
    assert audit["maxLength"] == 20 and audit["maxCompletion"] is None
    assert audit["files"]["conversations.jsonl"]["overLength"] == 1
    assert audit["files"]["conversations.jsonl"]["removed"] == 1
    assert audit["files"]["conversations.jsonl"]["kept"] == 3


def test_without_write_nothing_changes(tmp_path):
    write_split(tmp_path, "conversations.jsonl", [row(50, 1, "drop"), row(1, 1, "keep")])
    before = (tmp_path / "conversations.jsonl").read_text()
    result = audit_dir(tmp_path, WhitespaceTokenizer(), max_length=20)
    assert result["conversations.jsonl"]["overLength"] == 1
    assert "heldout.jsonl" not in result  # an absent split is skipped
    assert (tmp_path / "conversations.jsonl").read_text() == before
    assert not (tmp_path / "manifest.json").exists()
