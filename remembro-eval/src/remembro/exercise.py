"""Scenario-based end-to-end exercises: a script of ingests, questions and register changes
played in order against a fresh workspace and the real extractor.

  PYTHONPATH=src python -m remembro.exercise exercises/leave-week.yaml [--extractor rules|llm] [--home DIR] [--keep]

Every `decide` step states what it expects; the report shows PASS/FAIL per step, the reasons
and quotes, and counts unjustified ALLOWs (an ALLOW where the script expected anything else)
separately, because that is the number that must stay at zero.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from remembro.workspace import Workspace


def load(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def run(script: dict, workspace: Workspace, base: Path | None = None) -> dict:
    results = []
    for n, step in enumerate(script.get("steps", []), start=1):
        kind, args = next(iter(step.items()))
        # YAML 1.1 reads a bare `on:` key as boolean True; it is the request date
        args = {("on" if k is True else k): v for k, v in dict(args or {}).items()}
        t = time.time()
        entry: dict = {"step": n, "kind": kind}
        try:
            if kind == "ingest":
                if "path" in args and base is not None and not Path(args["path"]).is_absolute():
                    args["path"] = str(base / args["path"])
                r = workspace.ingest(**args)
                entry.update(ok=True, document_id=r["document_id"], candidates=r["candidates"], accepted=r["accepted_claims"], rejected=r["rejected_at_boundary"], unread=len(r["unread_spans"]), proposed_categories=[p["mention"] for p in r["proposed_categories"]])
            elif kind == "decide":
                expect = args.pop("expect", None)
                because = args.pop("because", "")
                d = workspace.decide(**args)
                got = d["decision"]
                ok = expect is None or got == expect
                entry.update(ok=ok, expect=expect, got=got, because=because, question=d["question"], reasons=d["reasons"][-2:], evidence=[f"{e['document']} p{e['page']} ¶{e['paragraph']}: {e['quote'][:90]}" for e in d["evidence"][:3]], next_steps=d.get("next_steps", []), unjustified_allow=(got == "ALLOW" and expect not in (None, "ALLOW")))
            elif kind == "register":
                action = args.pop("action", "add")
                if action == "alias":
                    reg = workspace.register()
                    target = next(e for e in reg if e["canonical_name"].lower() == args["name"].lower())
                    workspace.add_entity(target["kind"], target["canonical_name"], args.get("aliases", []), target["id"])
                else:
                    workspace.add_entity(args["kind"], args["name"], args.get("aliases", []))
                entry.update(ok=True, **args)
            elif kind == "dismiss":
                workspace.dismiss(args["candidate_id"], args.get("reason", "exercise"))
                entry.update(ok=True, **args)
            elif kind == "forget":
                entry.update(ok=workspace.forget(args["document_id"]))
            else:
                entry.update(ok=False, error=f"unknown step kind {kind!r}")
        except Exception as error:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(error).__name__}: {error}")
        entry["seconds"] = round(time.time() - t, 1)
        results.append(entry)
    decided = [r for r in results if r["kind"] == "decide"]
    return {
        "name": script.get("name"),
        "extractor": workspace.status()["extractor"],
        "steps": results,
        "decisions": len(decided),
        "passed": sum(1 for r in decided if r["ok"]),
        "unjustified_allow": sum(1 for r in decided if r.get("unjustified_allow")),
        "errors": [r for r in results if r.get("error")],
    }


def format_report(report: dict) -> str:
    lines = [f"exercise {report['name']}  extractor {report['extractor']}", f"decisions {report['passed']} / {report['decisions']}   unjustified ALLOW {report['unjustified_allow']}   errors {len(report['errors'])}", ""]
    for r in report["steps"]:
        if r["kind"] == "decide":
            mark = "PASS" if r["ok"] else "FAIL"
            lines.append(f"  {mark} {r['step']:>2} {r['question']}  expected {r['expect']} got {r['got']}  ({r['seconds']}s)")
            lines.append(f"          because: {r['because']}")
            for reason in r["reasons"]:
                lines.append(f"          - {reason[:140]}")
            for ev in r["evidence"]:
                lines.append(f"          [{ev}]")
            for step in r.get("next_steps", []):
                lines.append(f"          next: {step[:140]}")
        elif r["kind"] == "ingest":
            lines.append(f"  {'ok  ' if r['ok'] else 'ERR '} {r['step']:>2} ingest {r.get('document_id')}: {r.get('candidates')} candidates, {r.get('accepted')} accepted, {r.get('rejected')} rejected, {r.get('unread')} unread  ({r['seconds']}s)" + (f"  proposed categories: {r['proposed_categories']}" if r.get("proposed_categories") else "") + (f"  {r.get('error')}" if r.get("error") else ""))
        else:
            lines.append(f"  {'ok  ' if r['ok'] else 'ERR '} {r['step']:>2} {r['kind']} {json.dumps({k: v for k, v in r.items() if k not in ('step', 'kind', 'ok', 'seconds')})[:120]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--extractor", default=os.environ.get("REMEMBRO_EXTRACTOR", "llm"))
    ap.add_argument("--home", default=None, help="workspace dir; default a fresh temporary one")
    ap.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    path = Path(args.script)
    script = load(path)
    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="remembro-exercise-"))
    ws = Workspace(home, extractor=args.extractor)
    report = run(script, ws, base=Path.cwd())
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
    if not args.home and not args.keep:
        shutil.rmtree(home, ignore_errors=True)
    else:
        print(f"\nworkspace kept at {home}")
    raise SystemExit(0 if report["passed"] == report["decisions"] and not report["errors"] else 1)


if __name__ == "__main__":
    main()
