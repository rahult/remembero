"""remembro ingest | evaluate | explain | inspect"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from remembro.beliefs.pipeline import build_state
from remembro.claims.models import Entity, Request
from remembro.decision.engine import evaluate as decide
from remembro.document.parser import MarkdownParser
from remembro.entities.resolver import EntityResolver
from remembro.evaluation.evaluate import format_report, load_gold, run
from remembro.extraction.extractors import extractor_from_args
from remembro.storage.sqlite import save_state


def ingest(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.documents] or [Path(d["path"]) for d in load_gold(Path(args.gold)).get("documents", [])]
    if not paths:
        raise SystemExit("no documents given and the gold file lists none")
    extractor = extractor_from_args(args.extractor, args.run_name, args.model, args.base_url)
    candidates: list[dict] = []
    spans = 0
    for path in paths:
        doc = MarkdownParser().parse(path.stem, path.read_bytes())
        spans += len(doc.spans)
        for c in extractor.extract(doc):
            # ids must stay unique across documents
            c["id"] = f"{path.stem}:{c['id']}" if len(paths) > 1 else c["id"]
            candidates.append(c)
    out = Path(args.out or f"fixtures/runs/{extractor.name}.claims.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"extractor": extractor.name, "documents": [p.stem for p in paths], "claims": candidates}, indent=2, default=str))
    errors = [c for c in candidates if "error" in c]
    print(f"{extractor.name}: {len(candidates) - len(errors)} candidate claims from {spans} spans in {len(paths)} document(s) ({len(errors)} span errors) -> {out}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    gold = load_gold(Path(args.gold))
    if args.run_name == "gold":
        candidates = gold["claims"]
        name = "gold (perfect extractor)"
    else:
        snapshot = json.loads(Path(f"fixtures/runs/{args.run_name}.claims.json").read_text())
        candidates = snapshot["claims"]
        name = args.run_name
    report, state = run(gold, candidates, name, match_threshold=args.match_threshold)
    print(format_report(report))
    if args.db:
        save_state(Path(args.db), state)
        print(f"state written to {args.db}")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))


def explain(args: argparse.Namespace) -> None:
    gold = load_gold(Path(args.gold))
    scenario = next(s for s in gold["scenarios"] if s["id"] == args.scenario)
    candidates = gold["claims"] if args.run_name == "gold" else json.loads(Path(f"fixtures/runs/{args.run_name}.claims.json").read_text())["claims"]
    entities = [Entity.model_validate(e) for e in gold["entities"]]
    resolver = EntityResolver(entities, match_threshold=args.match_threshold)
    dates = {d["id"]: d["effective_date"] for d in gold.get("documents", [])}
    state, _ = build_state(candidates, entities, resolver, document_dates=dates or None)
    decision = decide(Request.model_validate(scenario["request"]), state, resolver)
    print(f"DECISION: {decision.decision.value}")
    print(f"Question: {scenario['question']}")
    print("Reason:")
    for i, r in enumerate(decision.reasons, 1):
        print(f"  {i}. {r}")
    print("Evidence:")
    for eid in decision.evidence_used:
        for c in state.claims:
            for e in c.evidence:
                if e.id == eid:
                    print(f"  {e.document_id} page {e.page}, paragraph {e.paragraph} ({e.authority}): \"{e.quoted_text[:110]}\"")
    if not decision.evidence_used:
        print("  (none: the decision rests on the absence of authority or on an unresolved identity)")


def inspect(args: argparse.Namespace) -> None:
    gold = load_gold(Path(args.gold))
    candidates = gold["claims"] if args.run_name == "gold" else json.loads(Path(f"fixtures/runs/{args.run_name}.claims.json").read_text())["claims"]
    entities = [Entity.model_validate(e) for e in gold["entities"]]
    resolver = EntityResolver(entities, match_threshold=args.match_threshold)
    dates = {d["id"]: d["effective_date"] for d in gold.get("documents", [])}
    state, _ = build_state(candidates, entities, resolver, document_dates=dates or None)
    if args.what == "claim":
        print(state.claim(args.id).model_dump_json(indent=2))
    elif args.what == "belief":
        print(state.belief(args.id).model_dump_json(indent=2))
    elif args.what == "transitions":
        for t in state.transitions:
            print(f"{t.sequence:3d} {t.event:24s} {t.subject_id:28s} {t.reason}")
    elif args.what == "contradictions":
        for c in state.contradictions:
            print(c.model_dump_json())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="remembro")
    parser.add_argument("--gold", default="fixtures/delegation_policy_v1.gold.json")
    parser.add_argument("--run-name", default="gold", help="snapshot in fixtures/runs/<name>.claims.json, or 'gold'")
    parser.add_argument("--match-threshold", type=float, default=0.9)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("documents", nargs="*", help="documents to ingest; defaults to the gold file's documents")
    p.add_argument("--extractor", choices=["rules", "llm", "snapshot"], default="rules")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--out")
    p.set_defaults(func=ingest)
    p = sub.add_parser("evaluate")
    p.add_argument("--db")
    p.add_argument("--json")
    p.set_defaults(func=evaluate_cmd)
    p = sub.add_parser("explain")
    p.add_argument("scenario")
    p.set_defaults(func=explain)
    p = sub.add_parser("inspect")
    p.add_argument("what", choices=["claim", "belief", "transitions", "contradictions"])
    p.add_argument("id", nargs="?")
    p.set_defaults(func=inspect)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
