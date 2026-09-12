"""The daily-use surface: a persistent workspace that ingests documents, keeps a register,
rebuilds state deterministically on every question, and answers with evidence."""
import json
from pathlib import Path

import pytest

from remembro.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "fixtures/delegation_policy_v1.md"
HV = ROOT / "fixtures/harbourview_delegations_v2.md"
HV_AMEND = ROOT / "fixtures/harbourview_delegations_v2_amendment1.md"


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "space", extractor="rules")


class TestIngest:
    def test_ingest_registers_people_and_roles_from_the_document(self, ws):
        report = ws.ingest(path=POLICY, effective_date="2026-01-01")
        assert report["document_id"] == "delegation_policy_v1"
        assert report["candidates"] > 15
        names = {e["canonical_name"] for e in ws.register()}
        assert {"Alice Morgan", "Bob Chen", "Carol Evans", "David Smith", "Chief Financial Officer", "Finance Director"} <= names
        assert "D. Smith" not in names  # an initial is never a register entry
        assert report["auto_registered"]

    def test_ingest_text_as_a_message(self, ws):
        report = ws.ingest(text="Alice Morgan may approve operational expenditure up to AUD 100,000 per transaction.", document_id="slack-1", effective_date="2026-09-12", kind="message")
        assert report["document_id"] == "slack-1"
        assert (ws.root / "documents" / "slack-1.md").exists()

    def test_documents_persist_and_reload(self, tmp_path):
        Workspace(tmp_path / "s", extractor="rules").ingest(path=POLICY, effective_date="2026-01-01")
        again = Workspace(tmp_path / "s", extractor="rules")
        assert [d["id"] for d in again.documents()] == ["delegation_policy_v1"]

    def test_forget(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        ws.forget("delegation_policy_v1")
        assert ws.documents() == []


class TestDecide:
    def test_allow_with_evidence(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        d = ws.decide(actor="Alice Morgan", resource="operational expenditure", amount=80000, currency="AUD", on="2026-06-01")
        assert d["decision"] == "ALLOW"
        assert d["evidence"] and all("quote" in e and "document" in e for e in d["evidence"])

    def test_unknown_person_denied_and_ambiguous_initial_unknown(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        assert ws.decide(actor="Erin Fox", resource="operational expenditure", amount=100, currency="AUD", on="2026-06-01")["decision"] == "DENY"
        assert ws.decide(actor="D. Smith", resource="utilities", amount=3000, currency="AUD", on="2026-09-10")["decision"] == "UNKNOWN"

    def test_unknown_category_is_unknown_until_registered(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        d = ws.decide(actor="Alice Morgan", resource="travel", amount=100, currency="AUD", on="2026-06-01")
        assert d["decision"] == "UNKNOWN"
        ws.add_entity(kind="category", name="travel expenditure", aliases=["travel"])
        d = ws.decide(actor="Alice Morgan", resource="travel", amount=100, currency="AUD", on="2026-06-01")
        assert d["decision"] == "DENY"  # registered, but nobody may approve it

    def test_a_later_document_changes_the_answer_from_its_date(self, tmp_path):
        # gold candidates stand in for a perfect extractor so this tests the workspace, not regex
        ws = Workspace(tmp_path / "s", extractor="snapshot")
        gold = json.loads((ROOT / "fixtures/harbourview_delegations_v2.gold.json").read_text())
        by_doc = {}
        for c in gold["claims"]:
            by_doc.setdefault(c["evidence"][0]["document_id"], []).append(c)
        ws.ingest(path=HV, effective_date="2026-03-01", candidates=by_doc["harbourview_delegations_v2"])
        before = ws.decide(actor="Owen Pritchard", resource="operational expenditure", amount=70000, currency="NZD", on="2026-11-15")
        ws.ingest(path=HV_AMEND, effective_date="2026-11-01", candidates=by_doc["harbourview_delegations_v2_amendment1"])
        after = ws.decide(actor="Owen Pritchard", resource="operational expenditure", amount=70000, currency="NZD", on="2026-11-15")
        assert (before["decision"], after["decision"]) == ("DENY", "ALLOW")


class TestInspect:
    def test_inspect_surfaces(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        assert ws.inspect("contradictions")
        assert ws.inspect("documents")[0]["id"] == "delegation_policy_v1"
        assert isinstance(ws.inspect("rejected"), list)
        assert any(t["event"] == "CLAIM_ACCEPTED" for t in ws.inspect("transitions"))


class TestPolish:
    def test_roles_get_acronym_aliases_when_auto_registered(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        reg = {e["canonical_name"]: e for e in ws.register()}
        assert "APL" in reg["Accounts Payable Lead"]["aliases"]
        assert "Bob" in reg["Bob Chen"]["aliases"]

    def test_status_overview(self, ws):
        ws.ingest(path=POLICY, effective_date="2026-01-01")
        st = ws.status()
        assert st["documents"] == 1 and st["people"] >= 4 and st["roles"] >= 4
        assert "open_contradictions" in st and "rejected_at_boundary" in st and st["extractor"]

    def test_ingest_missing_path_is_an_error_not_a_crash(self, ws):
        with pytest.raises(FileNotFoundError):
            ws.ingest(path="/nonexistent/policy.md")


class TestSettlingDoubt:
    def _doubtful(self, tmp_path):
        ws = Workspace(tmp_path / "s", extractor="snapshot")
        gold = json.loads((ROOT / "fixtures/delegation_policy_v1.gold.json").read_text())
        quote = "Bob Chen is off the roster. Bob Chen"
        ev = {"id": "e1", "document_id": "note", "page": 1, "paragraph": 1, "section": None, "start_offset": 0, "end_offset": len(quote), "quoted_text": quote, "authority": "body"}
        junk = {"id": "junk", "subject": "Bob Chen", "subject_kind": "person", "predicate": "off_roster", "object": None, "object_kind": "none", "modality": "assertion", "polarity": "negative", "constraints": {}, "valid_from": None, "valid_until": None, "confidence": 0.5, "evidence": [ev]}
        ws.ingest(path=POLICY, effective_date="2026-01-01", candidates=gold["claims"])
        ws.ingest(text=quote, document_id="note", effective_date="2026-06-01", candidates=[junk])
        return ws

    def test_unknown_carries_next_steps_and_dismissal_settles_it(self, tmp_path):
        ws = self._doubtful(tmp_path)
        d = ws.decide(actor="Bob Chen", resource="operational expenditure", amount=10000, currency="AUD", on="2026-06-05")
        assert d["decision"] == "UNKNOWN"
        assert d["next_steps"] and any("dismiss" in step for step in d["next_steps"])
        ws.dismiss("note:junk", reason="extractor noise: 'off the roster' is a scheduling remark")
        d = ws.decide(actor="Bob Chen", resource="operational expenditure", amount=10000, currency="AUD", on="2026-06-05")
        assert d["decision"] == "ALLOW"
        assert any(x["candidate_id"] == "note:junk" for x in ws.inspect("dismissed"))

    def test_identity_unknown_suggests_an_alias(self, tmp_path):
        ws = Workspace(tmp_path / "s", extractor="snapshot")
        gold = json.loads((ROOT / "fixtures/delegation_policy_v1.gold.json").read_text())
        ws.ingest(path=POLICY, effective_date="2026-01-01", candidates=gold["claims"])
        d = ws.decide(actor="D. Smith", resource="utilities", amount=3000, currency="AUD", on="2026-09-10")
        assert d["decision"] == "UNKNOWN" and any("alias" in s for s in d["next_steps"])
        ws.add_entity("person", "David Smith", aliases=["D. Smith"])
        assert ws.decide(actor="D. Smith", resource="utilities", amount=3000, currency="AUD", on="2026-09-10")["decision"] == "ALLOW"
