"""Paraphrase labelled spans with a chat model; keep a rewrite only if every label still
parses, grounds (amounts, dates, ceiling) and names its subject and object in the new text.

  EXTRACTION_API_KEY=… PYTHONPATH=src python -m remembro.training.paraphrase \
      --in data/claims/conversations.jsonl --out data/claims/paraphrased.jsonl --n 400 \
      --model glm-5.3-flash:cloud --base-url http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request

from remembro.training.synth import _valid, to_rows

PROMPT = "Rewrite the following sentence from a corporate delegation-of-authority policy in different words. Keep every person's name, role title, amount, currency, date and the exact meaning (including who may or may not do what, and any conditions). One sentence or two; formal register; no lists; return only the rewritten text."


def rewrite(text: str, model: str, base_url: str, key: str) -> str:
    body = json.dumps({"model": model, "temperature": 0.9, "max_tokens": 4000, "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return (data["choices"][0]["message"]["content"] or "").strip().strip('"')


def names_present(text: str, labels: list[dict]) -> bool:
    low = text.lower()
    for lab in labels:
        for field in ("subject", "object"):
            v = lab.get(field)
            if v and lab.get(f"{field}_kind") in ("person", "role") and v.lower() not in low:
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    key = os.environ.get("EXTRACTION_API_KEY") or os.environ.get("LLM_API_KEY", "")
    rows = [json.loads(l) for l in open(args.inp)]
    labelled = [r for r in rows if json.loads(r["messages"][2]["content"])]
    random.Random(args.seed).shuffle(labelled)
    kept: list[tuple[str, list[dict]]] = []
    tried = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(r):
        text = r["messages"][1]["content"]
        labels = json.loads(r["messages"][2]["content"])
        try:
            return text, labels, rewrite(text, args.model, args.base_url, key)
        except Exception as error:  # noqa: BLE001
            return text, labels, f"__error__ {str(error)[:80]}"

    with open(args.out, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one, r) for r in labelled[: args.n]]):
            text, labels, new = fut.result()
            tried += 1
            if new.startswith("__error__"):
                print("error:", new, flush=True)
                continue
            if not new or new == text or len(new) > 700:
                continue
            if _valid(new, labels) and names_present(new, labels):
                kept.append((new, labels))
                f.write(json.dumps(to_rows([(new, labels)])[0], ensure_ascii=False) + "\n")
                f.flush()
            if tried % 50 == 0:
                print(f"{tried} tried, {len(kept)} kept", flush=True)
    print(f"done: {tried} tried, {len(kept)} kept -> {args.out}")


if __name__ == "__main__":
    main()
