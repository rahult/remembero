import json
from pathlib import Path

import pytest

from remembro.beliefs.pipeline import build_state
from remembro.claims.models import Entity, EntityKind, MatchVerdict, Request
from remembro.decision.engine import evaluate as decide
from remembro.document.parser import MarkdownParser
from remembro.entities.resolver import EntityResolver
from remembro.evaluation.evaluate import run
from remembro.extraction.extractors import RuleExtractor

ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "fixtures/delegation_policy_v1.gold.json").read_text())
DOC = MarkdownParser().parse("delegation_policy_v1", (ROOT / "fixtures/delegation_policy_v1.md").read_bytes())


def resolver(threshold: float = 0.9) -> EntityResolver:
    return EntityResolver([Entity.model_validate(e) for e in GOLD["entities"]], match_threshold=threshold)


def state_from(candidates):
    entities = [Entity.model_validate(e) for e in GOLD["entities"]]
    s, rejected = build_state(candidates, entities, resolver())
    return s, rejected


class TestDocument:
    def test_pages_sections_and_appendix_authority(self):
        assert max(s.page for s in DOC.spans) == 5
        assert any(s.section and s.section.startswith("6.3") for s in DOC.spans)
        assert {s.authority for s in DOC.spans} == {"body", "appendix", "table"}
        span_texts = [s.text for s in DOC.spans]
        assert all(
            e["quoted_text"] in DOC.text or any(e["quoted_text"] in t for t in span_texts)
            for c in GOLD["claims"]
            for e in c["evidence"]
        )


class TestEntities:
    def test_canonical_and_alias_match(self):
        r = resolver()
        assert r.resolve("Alice Morgan", EntityKind.PERSON).verdict is MatchVerdict.MATCH
        assert r.resolve("Alice", EntityKind.PERSON).entity_id == "person_alice"
        assert r.resolve("CFO", EntityKind.ROLE).entity_id == "role_cfo"

    def test_initial_is_possible_not_match(self):
        r = resolver().resolve("D. Smith", EntityKind.PERSON)
        assert r.verdict is MatchVerdict.POSSIBLE_MATCH
        assert r.entity_id == "person_david"

    def test_unknown_person_is_no_match(self):
        assert resolver().resolve("Erin Fox", EntityKind.PERSON).verdict is MatchVerdict.NO_MATCH

    def test_lower_threshold_merges_the_initial(self):
        # the sensitivity property: lowering certainty turns POSSIBLE into MATCH
        assert resolver(0.6).resolve("D. Smith", EntityKind.PERSON).verdict is MatchVerdict.MATCH


class TestClaims:
    def test_schema_rejects_unknown_predicate(self):
        bad = dict(GOLD["claims"][0])
        bad = {**bad, "predicate": "likes"}
        _, rejected = state_from([bad])
        assert len(rejected) == 1

    def test_provenance_required(self):
        bad = {**GOLD["claims"][0], "evidence": []}
        _, rejected = state_from([bad])
        assert len(rejected) == 1


class TestTemporalAndConflicts:
    def test_revocation_supersedes_delegation_end(self):
        s, _ = state_from(GOLD["claims"])
        deleg = s.claim("claim_delegation_alice_bob")
        assert str(deleg.valid_until) == "2026-09-11"
        assert deleg.superseded_by == "claim_revocation"
        assert any(t.event == "CLAIM_SUPERSEDED" for t in s.transitions)

    def test_appendix_conflict_resolved_body_prevails(self):
        s, _ = state_from(GOLD["claims"])
        resolved = [c for c in s.contradictions if c.resolved]
        assert any(set(c.claims) == {"claim_pm_opex", "claim_appx_pm_opex"} and c.winner == "claim_pm_opex" for c in resolved)

    def test_body_body_conflict_stays_disputed(self):
        s, _ = state_from(GOLD["claims"])
        unresolved = [c for c in s.contradictions if not c.resolved]
        assert any(set(c.claims) == {"claim_fd_capex", "claim_fd_capex_s7"} for c in unresolved)
        disputed = [b for b in s.beliefs if b.status.value == "DISPUTED"]
        assert len(disputed) == 2


class TestDecisions:
    @pytest.mark.parametrize("scenario", GOLD["scenarios"], ids=lambda s: s["id"])
    def test_gold_scenarios(self, scenario):
        s, _ = state_from(GOLD["claims"])
        d = decide(Request.model_validate(scenario["request"]), s, resolver())
        assert d.decision.value == scenario["expected"]["decision"], d.reasons

    def test_zero_unjustified_allow_with_gold_and_rules(self):
        for candidates in (GOLD["claims"], RuleExtractor().extract(DOC)):
            report, _ = run(GOLD, candidates, "x")
            assert report.unjustified_allow == 0
            assert report.provenance_coverage == 1.0

    def test_allow_carries_evidence(self):
        s, _ = state_from(GOLD["claims"])
        d = decide(Request.model_validate(GOLD["scenarios"][2]["request"]), s, resolver())
        assert d.decision.value == "ALLOW"
        assert d.evidence_used and d.beliefs_used

    def test_threshold_sensitivity_exposes_the_risk(self):
        # at a permissive threshold the ambiguous identity becomes a match and scenario 7 flips
        s7 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_07")
        entities = [Entity.model_validate(e) for e in GOLD["entities"]]
        loose = EntityResolver(entities, match_threshold=0.6)
        s, _ = build_state(GOLD["claims"], entities, loose)
        d = decide(Request.model_validate(s7["request"]), s, loose)
        assert d.decision.value == "ALLOW"  # this is the unsafe merge the sensitivity report must show


class TestGrounding:
    """A claim that parses is not yet a claim the engine may act on: its numbers, dates and
    ceiling must be visible in its own evidence. These are the cases GLM Flash produced."""

    def _glm(self):
        return json.loads((ROOT / "fixtures/runs/glm-flash-1500tok.claims.json").read_text())["claims"]

    def test_gold_claims_are_all_grounded(self):
        from remembro.claims.grounding import ground
        kept, dropped = ground(GOLD["claims"])
        assert dropped == []
        assert len(kept) == len(GOLD["claims"])

    def test_amount_absent_from_evidence_is_rejected(self):
        from remembro.claims.grounding import ground
        c = json.loads(json.dumps(next(x for x in GOLD["claims"] if x["id"] == "claim_fd_opex")))
        c["constraints"]["maximum_amount"] = 60000
        _, dropped = ground([c])
        assert len(dropped) == 1 and "amount" in dropped[0][1]

    def test_date_absent_from_evidence_is_rejected(self):
        from remembro.claims.grounding import ground
        c = json.loads(json.dumps(next(x for x in GOLD["claims"] if x["id"] == "claim_delegation_alice_bob")))
        c["valid_until"] = "2026-09-25"
        _, dropped = ground([c])
        assert len(dropped) == 1 and "date" in dropped[0][1]

    def test_positive_permission_without_ceiling_needs_the_text_to_say_so(self):
        from remembro.claims.grounding import ground
        # GLM read "the delegation does not extend to capital expenditure" as an unlimited grant
        glm = self._glm()
        bad = [c for c in glm if c.get("predicate") == "may_approve" and c.get("polarity") == "positive" and c.get("constraints", {}).get("maximum_amount") is None and c.get("subject") == "Bob Chen"]
        assert bad, "fixture changed: the ungrounded unlimited grant is gone"
        _, dropped = ground(bad)
        assert len(dropped) == len(bad) and all("ceiling" in r for _, r in dropped)

    def test_grounding_removes_glm_unjustified_allow_for_the_conflict_scenario(self):
        # scenario 6 asked UNKNOWN; GLM's ungrounded grant made it ALLOW. Grounding must turn
        # that into a safe outcome (UNKNOWN or DENY), never ALLOW.
        report, _ = run(GOLD, self._glm(), "glm")
        s6 = next(d for d in report.decisions if d["id"] == "scenario_06")
        assert s6["got"] != "ALLOW"


class TestRestrictionsSurviveExtractorNoise:
    """Dropping a restriction fails toward permission. These cases came from GLM Flash."""

    def _glm(self):
        return json.loads((ROOT / "fixtures/runs/glm-flash.claims.json").read_text())["claims"]

    def test_suspension_ignores_a_meaningless_object(self):
        susp = json.loads(json.dumps(next(x for x in GOLD["claims"] if x["id"] == "claim_carol_suspended")))
        susp["object"], susp["object_kind"] = "approval authority", "category"
        s, _ = state_from([*[c for c in GOLD["claims"] if c["id"] != "claim_carol_suspended"], susp])
        assert s.claim("claim_carol_suspended").status.value == "ACCEPTED"
        s8 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_08")
        assert decide(Request.model_validate(s8["request"]), s, resolver()).decision.value == "DENY"

    def test_unresolved_restriction_on_a_known_person_makes_decisions_unknown(self):
        # a prohibition on Carol whose object nobody can resolve is not nothing: it is doubt
        bad = {**json.loads(json.dumps(GOLD["claims"][0])), "id": "odd", "subject": "Carol Evans", "subject_kind": "person",
               "predicate": "may_approve", "object": "gadgets", "object_kind": "category", "modality": "prohibition", "polarity": "negative",
               "constraints": {}, "valid_from": None, "valid_until": None}
        s, _ = state_from([*GOLD["claims"], bad])
        s1 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_01")  # Alice, unaffected
        assert decide(Request.model_validate(s1["request"]), s, resolver()).decision.value == "ALLOW"
        carol = Request.model_validate({**next(x for x in GOLD["scenarios"] if x["id"] == "scenario_08")["request"], "on": "2026-09-05"})
        assert decide(carol, s, resolver()).decision.value == "UNKNOWN"

    def test_revocation_ends_a_grant_that_restates_the_delegation(self):
        deleg = next(x for x in GOLD["claims"] if x["id"] == "claim_delegation_alice_bob")
        restated = {**json.loads(json.dumps(deleg)), "id": "restated", "subject": "Bob Chen", "subject_kind": "person",
                    "predicate": "may_approve", "object": "operational expenditure", "object_kind": "category", "modality": "permission"}
        s, _ = state_from([*GOLD["claims"], restated])
        assert str(s.claim("restated").valid_until) == "2026-09-11"
        s5 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_05")
        assert decide(Request.model_validate(s5["request"]), s, resolver()).decision.value == "DENY"

    def test_glm_snapshot_has_no_unjustified_allow(self):
        report, _ = run(GOLD, self._glm(), "glm")
        assert report.unjustified_allow == 0, [d for d in report.decisions if not d["ok"]]


class TestUnreadSpans:
    def test_an_unread_span_naming_the_actor_makes_decisions_unknown(self):
        # the truncated GLM run dropped §6.3 (the revocation) on the floor and allowed Bob;
        # once the harness records the drop, the engine must refuse to decide about Bob
        unread = {"id": "x_error_37", "error": "reply truncated at 1500 tokens", "span_text": "By notice dated 10 September 2026, Alice Morgan revoked the delegation of operational expenditure authority to Bob Chen described in section 5.2, with effect from 12 September 2026."}
        without_revocation = [c for c in GOLD["claims"] if c["id"] != "claim_revocation"]
        s, _ = state_from([*without_revocation, unread])
        s5 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_05")
        assert decide(Request.model_validate(s5["request"]), s, resolver()).decision.value == "UNKNOWN"
        s8 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_08")  # Carol is not named there
        assert decide(Request.model_validate(s8["request"]), s, resolver()).decision.value == "DENY"
