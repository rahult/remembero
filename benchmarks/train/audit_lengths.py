"""Token-length audit of a reader training directory, counted the way the trainer counts.

`reader_lora.train_lora` hands TRL prompt/completion rows with `max_length` truncation from the right,
so a row whose prompt plus completion runs past the limit silently loses the end of its completion.
This counts every row of `conversations.jsonl` and `heldout.jsonl` through the same path TRL's
SFTTrainer takes for conversational prompt/completion data: prompt ids from
`apply_chat_template(prompt, add_generation_prompt=True)`, prompt+completion ids from
`apply_chat_template(prompt + completion)`, completion tokens being the difference.

    python -m benchmarks.train.audit_lengths data/training-reader-v6 --max-length 6912
    python -m benchmarks.train.audit_lengths DIR --max-length 8192 --max-completion 768 --write

`--write` drops the rows over either limit from each file and, by line index, from its
`.meta.jsonl` twin, and records the counts in `manifest.json` under `lengthAudit`. The manifest's
own `rows` field is left as the generator wrote it."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from benchmarks.train.reader_lora import to_prompt_completion

SPLITS = ("conversations.jsonl", "heldout.jsonl")
DEFAULT_TOKENIZER = "google/gemma-4-E4B-it"


def _ids(tokenizer, messages: list[dict], **kwargs) -> list:
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kwargs)["input_ids"]


def _spread(values: list[int]) -> dict:
    """Nearest-rank p50/p95 and the max; zeros for an empty file."""
    if not values:
        return {"p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    rank = lambda q: ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"p50": rank(0.50), "p95": rank(0.95), "max": ordered[-1]}


def audit_file(path: Path, tokenizer, max_length: int, max_completion: int | None = None) -> tuple[dict, list[int]]:
    """Report for one split, plus the indices (among its non-blank rows) over either limit."""
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    over: list[int] = []
    over_length = over_completion = 0
    for index, row in enumerate(to_prompt_completion(str(path))):
        prompt = len(_ids(tokenizer, row["prompt"], add_generation_prompt=True))
        total = len(_ids(tokenizer, row["prompt"] + row["completion"]))
        completion = total - prompt
        prompt_tokens.append(prompt)
        completion_tokens.append(completion)
        too_long = total > max_length
        too_wordy = max_completion is not None and completion > max_completion
        over_length += too_long
        over_completion += too_wordy
        if too_long or too_wordy:
            over.append(index)
    report = {
        "rows": len(prompt_tokens),
        "overLength": over_length,
        "overCompletion": over_completion,
        "promptTokens": _spread(prompt_tokens),
        "completionTokens": _spread(completion_tokens),
    }
    return report, over


def _non_blank_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [line if line.endswith("\n") else line + "\n" for line in handle if line.strip()]


def _replace(path: Path, lines: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def _twin(path: Path) -> tuple[list[str], Path, list[str] | None]:
    lines = _non_blank_lines(path)
    meta_path = path.with_name(path.name + ".meta.jsonl")
    meta = _non_blank_lines(meta_path) if meta_path.exists() else None
    if meta is not None and len(meta) != len(lines):
        raise ValueError(f"{meta_path.name} has {len(meta)} rows but {path.name} has {len(lines)}")
    return lines, meta_path, meta


def _drop_rows(path: Path, over: list[int]) -> int:
    """Rewrite `path` and its `.meta.jsonl` twin without the rows at `over`; returns rows kept."""
    lines, meta_path, meta = _twin(path)
    dropped = set(over)
    _replace(path, [line for i, line in enumerate(lines) if i not in dropped])
    if meta is not None:
        _replace(meta_path, [line for i, line in enumerate(meta) if i not in dropped])
    return len(lines) - len(dropped)


def audit_dir(data_dir: Path, tokenizer, max_length: int, max_completion: int | None = None, *,
              write: bool = False, tokenizer_name: str | None = None) -> dict:
    data_dir = Path(data_dir)
    results: dict = {}
    overs: dict[str, list[int]] = {}
    for name in SPLITS:
        path = data_dir / name
        if path.exists():
            results[name], overs[name] = audit_file(path, tokenizer, max_length, max_completion)
    if not write:
        return results
    # Audit and check every split's meta twin before rewriting any, so a bad directory stays untouched.
    for name in results:
        _twin(data_dir / name)
    files = {}
    for name, report in results.items():
        kept = _drop_rows(data_dir / name, overs[name])
        files[name] = {**report, "removed": len(overs[name]), "kept": kept}
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["lengthAudit"] = {"maxLength": max_length, "maxCompletion": max_completion,
                               "tokenizer": tokenizer_name, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-completion", type=int, default=None)
    parser.add_argument("--write", action="store_true", help="drop over-limit rows and record lengthAudit")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="model id or saved tokenizer directory")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)  # what train_lora loads
    results = audit_dir(args.data_dir, tokenizer, args.max_length, args.max_completion,
                        write=args.write, tokenizer_name=args.tokenizer)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
