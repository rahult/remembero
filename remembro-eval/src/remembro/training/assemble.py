"""Assemble a writer training set: the previous round's rows plus the claims task.

  PYTHONPATH=src python -m remembro.training.assemble --base ../data/training-r19 \
      --claims data/claims/conversations.jsonl --claims-heldout data/claims/heldout.jsonl \
      --paraphrased data/claims/paraphrased.jsonl --out ../data/training-r21
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--claims-heldout", required=True)
    ap.add_argument("--paraphrased", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"base_train": 0, "claims_train": 0, "paraphrased": 0, "claims_heldout": 0, "duplicates_dropped": 0}
    with (out / "conversations.jsonl").open("w") as f:
        for line in open(Path(args.base) / "conversations.jsonl"):
            f.write(line)
            counts["base_train"] += 1
        for key, path in (("claims_train", args.claims), ("paraphrased", args.paraphrased)):
            if not path:
                continue
            for line in open(path):
                row = json.loads(line)
                user = row["messages"][1]["content"]
                if user in seen:
                    counts["duplicates_dropped"] += 1
                    continue
                seen.add(user)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[key] += 1
    with (out / "heldout.jsonl").open("w") as f:
        for line in open(Path(args.base) / "heldout.jsonl"):
            f.write(line)
        for line in open(args.claims_heldout):
            f.write(line)
            counts["claims_heldout"] += 1
    base_manifest = json.loads((Path(args.base) / "manifest.json").read_text()) if (Path(args.base) / "manifest.json").exists() else {}
    manifest = {
        "base": args.base,
        "baseNotes": base_manifest.get("notes"),
        **counts,
        "train": sum(1 for _ in open(out / "conversations.jsonl")),
        "heldout": sum(1 for _ in open(out / "heldout.jsonl")),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
