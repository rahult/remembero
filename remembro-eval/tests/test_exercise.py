from pathlib import Path

from remembro.exercise import load, run
from remembro.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def test_amendment_exercise_with_gold_claims_passes(tmp_path):
    """The exercise runner plays a script over time; with gold claims as the extractor, every
    expectation must hold, which pins the runner itself."""
    import json
    script = load(ROOT / "exercises/amendment.yaml")
    gold = json.loads((ROOT / "fixtures/harbourview_delegations_v2.gold.json").read_text())
    by_doc = {}
    for c in gold["claims"]:
        by_doc.setdefault(c["evidence"][0]["document_id"], []).append(c)
    for step in script["steps"]:
        if "ingest" in step:
            step["ingest"]["candidates"] = by_doc[Path(step["ingest"]["path"]).stem]
    report = run(script, Workspace(tmp_path / "w", extractor="snapshot"), base=ROOT)
    assert report["errors"] == []
    assert report["passed"] == report["decisions"], [s for s in report["steps"] if s["kind"] == "decide" and not s["ok"]]
    assert report["unjustified_allow"] == 0


def test_a_failed_expectation_is_reported_not_raised(tmp_path):
    script = {"name": "x", "steps": [{"ingest": {"path": "fixtures/delegation_policy_v1.md", "effective_date": "2026-01-01"}}, {"decide": {"actor": "Alice Morgan", "resource": "operational expenditure", "amount": 80000, "currency": "AUD", "on": "2026-06-01", "expect": "DENY"}}]}
    report = run(script, Workspace(tmp_path / "w", extractor="rules"), base=ROOT)
    d = report["steps"][1]
    assert d["ok"] is False and d["got"] == "ALLOW" and d["unjustified_allow"] is True
    assert report["unjustified_allow"] == 1
