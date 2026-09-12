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


class TestSyntheticTrainingData:
    def test_every_label_parses_and_grounds_and_no_fixture_names(self):
        from remembro.training.synth import generate
        spans = generate(docs=20, per_doc=16, seed=3)
        assert len(spans) > 200
        assert sum(1 for _, l in spans if not l) > 10  # negatives are part of the lesson
        joined = " ".join(t for t, _ in spans)
        for name in ("Alice Morgan", "Bob Chen", "Carol Evans", "David Smith"):
            assert name not in joined
        predicates = {l["predicate"] for _, ls in spans for l in ls}
        assert predicates == {"holds_role", "may_approve", "delegates", "suspended", "revokes_delegation"}
        assert any(l["predicate"] == "suspended" and l["polarity"] == "negative" for _, ls in spans for l in ls)
        roles = {l["object"] for _, ls in spans for l in ls if l["predicate"] == "holds_role"}
        assert len(roles) > 30  # composed roles, so the writer copies role strings instead of learning a set
        assert any(l["predicate"] == "holds_role" and l["valid_until"] and not l["valid_from"] for _, ls in spans for l in ls)


class TestRegisterDrivenVocabulary:
    def test_roles_and_categories_resolve_from_the_register_not_the_code(self):
        reg = [
            Entity(id="role_coo", kind=EntityKind.ROLE, canonical_name="Chief Operating Officer", aliases=["COO"]),
            Entity(id="role_gm", kind=EntityKind.ROLE, canonical_name="Grants Manager", aliases=[]),
            Entity(id="grant_expenditure", kind=EntityKind.CATEGORY, canonical_name="grant expenditure", aliases=["grants", "grant payments"]),
        ]
        r = EntityResolver(reg)
        assert r.resolve("the COO", EntityKind.ROLE).entity_id == "role_coo"
        assert r.resolve("Chief Operating Officer", EntityKind.ROLE).entity_id == "role_coo"
        assert r.resolve("Grants Managers", EntityKind.ROLE).entity_id == "role_gm"
        assert r.resolve("Finance Director", EntityKind.ROLE).verdict is MatchVerdict.NO_MATCH
        assert r.resolve("grant payments", EntityKind.CATEGORY).entity_id == "grant_expenditure"
        assert r.resolve("grant expenditure", EntityKind.CATEGORY).entity_id == "grant_expenditure"


GOLD2 = json.loads((ROOT / "fixtures/harbourview_delegations_v2.gold.json").read_text())


def state2(candidates, threshold: float = 0.9):
    entities = [Entity.model_validate(e) for e in GOLD2["entities"]]
    r = EntityResolver(entities, match_threshold=threshold)
    dates = {d["id"]: d["effective_date"] for d in GOLD2["documents"]}
    s, rejected = build_state(candidates, entities, r, document_dates=dates)
    return s, r


class TestSecondDocument:
    """A different organisation, a shared surname, a third category, and an amendment dated
    later than the schedule it changes. The rules learned on fixture 1 must hold here unchanged."""

    def test_quotes_are_located_in_their_documents(self):
        from remembro.document.parser import MarkdownParser
        docs = {d["id"]: MarkdownParser().parse(d["id"], (ROOT / d["path"]).read_bytes()) for d in GOLD2["documents"]}
        for c in GOLD2["claims"]:
            for e in c["evidence"]:
                assert any(e["quoted_text"] in s.text for s in docs[e["document_id"]].spans), c["id"]

    def test_shared_surname_initial_is_possible_for_both(self):
        _, r = state2(GOLD2["claims"])
        res = r.resolve("J. Ford", EntityKind.PERSON)
        assert res.verdict is MatchVerdict.POSSIBLE_MATCH
        assert set(res.candidates) == {"person_julian", "person_julia"}

    def test_amendment_supersedes_the_schedule_from_its_effective_date(self):
        s, _ = state2(GOLD2["claims"])
        old = s.claim("c_coo_opex")
        new = s.claim("c_am_coo_opex")
        assert str(old.valid_until) == "2026-10-31" and old.superseded_by == "c_am_coo_opex"
        assert str(new.valid_from) == "2026-11-01"
        assert not any(set(c.claims) == {"c_coo_opex", "c_am_coo_opex"} for c in s.contradictions)

    def test_later_document_ends_a_role_and_a_suspension(self):
        s, _ = state2(GOLD2["claims"])
        assert str(s.claim("c_role_julian").valid_until) == "2026-10-31"
        assert str(s.claim("c_owen_suspended").valid_until) == "2026-08-31"

    def test_revocation_ends_a_personal_authorisation(self):
        s, _ = state2(GOLD2["claims"])
        assert str(s.claim("c_nadia_round").valid_until) == "2026-07-31"

    @pytest.mark.parametrize("scenario", GOLD2["scenarios"], ids=lambda s: s["id"])
    def test_scenarios(self, scenario):
        s, r = state2(GOLD2["claims"])
        d = decide(Request.model_validate(scenario["request"]), s, r)
        assert d.decision.value == scenario["expected"]["decision"], d.reasons

    def test_zero_unjustified_allow(self):
        report, _ = run(GOLD2, GOLD2["claims"], "gold2")
        assert report.unjustified_allow == 0 and report.decision_accuracy == 1.0


class TestObjectCategoryIsAuthoritative:
    def test_resolved_object_overrides_the_category_enum(self):
        # GLM wrote object "grant expenditure" but constraints.category "expenditure" because the
        # prompt's enum had no grant category; the quoted object is the grounded value
        c = json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_gm_grant")))
        c["constraints"]["category"] = "expenditure"
        s, _ = state2([x for x in GOLD2["claims"] if x["id"] != "c_gm_grant"] + [c])
        assert s.claim("c_gm_grant").constraints.category == "grant_expenditure"
        s9 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_09")
        _, r = state2([])
        assert decide(Request.model_validate(s9["request"]), s, r).decision.value == "ALLOW"


class TestLiftedSuspension:
    def test_negative_suspension_is_a_lift_not_a_suspension(self):
        lifted = {**json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_owen_suspended"))), "id": "lift", "polarity": "negative", "valid_from": "2026-09-01", "valid_until": None}
        lifted["evidence"] = [dict(next(x for x in GOLD2["claims"] if x["id"] == "c_am_owen_lifted")["evidence"][0])]
        base = [x for x in GOLD2["claims"] if x["id"] != "c_am_owen_lifted"]
        s, r = state2([*base, lifted])
        assert str(s.claim("c_owen_suspended").valid_until) == "2026-08-31"
        s17 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_17")
        assert decide(Request.model_validate(s17["request"]), s, r).decision.value == "ALLOW"
        s5 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_05")
        assert decide(Request.model_validate(s5["request"]), s, r).decision.value == "DENY"


class TestUnreadableClaims:
    def test_schema_rejected_claim_naming_the_actor_is_doubt(self):
        # r19 wrote predicate "ceased_to_hold" for Julian Ford; the schema dropped it, the role
        # end vanished, and Julian was allowed after he left. Unreadable speech about a person
        # is doubt about that person, not silence.
        base = [x for x in GOLD2["claims"] if x["id"] != "c_am_julian_ends"]
        bad = {**json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_am_julian_ends"))), "predicate": "ceased_to_hold"}
        s, r = state2([*base, bad])
        s14 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_14")
        assert decide(Request.model_validate(s14["request"]), s, r).decision.value == "UNKNOWN"
        s15 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_15")  # Marcus is not named in that claim
        assert decide(Request.model_validate(s15["request"]), s, r).decision.value == "ALLOW"

    def test_r19_snapshot_on_fixture_2_has_no_unjustified_allow(self):
        cands = json.loads((ROOT / "fixtures/runs/r19-writer-hv.claims.json").read_text())["claims"]
        report, _ = run(GOLD2, cands, "r19-hv")
        assert report.unjustified_allow == 0, [d for d in report.decisions if not d["ok"]]


class TestCategoryFieldIsResolved:
    def test_unknown_category_string_becomes_null_and_the_revocation_still_applies(self):
        # r21 wrote constraints.category "none" (a string) on the revocation; a category that
        # is not in the register is no category, and a revocation with no category applies to all
        rev = json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_revocation_nadia")))
        rev["constraints"]["category"] = "none"
        s, r = state2([x for x in GOLD2["claims"] if x["id"] != "c_revocation_nadia"] + [rev])
        assert s.claim("c_revocation_nadia").constraints.category is None
        assert str(s.claim("c_nadia_round").valid_until) == "2026-07-31"
        s8 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_08")
        assert decide(Request.model_validate(s8["request"]), s, r).decision.value == "DENY"

    def test_r21_snapshot_on_fixture_2_has_no_unjustified_allow(self):
        cands = json.loads((ROOT / "fixtures/runs/r21-writer-hv.claims.json").read_text())["claims"]
        report, _ = run(GOLD2, cands, "r21-hv")
        assert report.unjustified_allow == 0, [d for d in report.decisions if not d["ok"]]


class TestCategoryGrounding:
    def test_invented_category_on_a_revocation_is_dropped_so_it_applies_to_all(self):
        # r22 wrote category operational_expenditure on a revocation whose text names no
        # category; the revocation then missed the grant it was meant to end (unjustified ALLOW)
        rev = json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_revocation_nadia")))
        rev["constraints"]["category"] = "operational_expenditure"
        s, r = state2([x for x in GOLD2["claims"] if x["id"] != "c_revocation_nadia"] + [rev])
        assert s.claim("c_revocation_nadia").constraints.category is None
        s8 = next(x for x in GOLD2["scenarios"] if x["id"] == "scenario_08")
        assert decide(Request.model_validate(s8["request"]), s, r).decision.value == "DENY"

    def test_invented_category_on_a_permission_rejects_it(self):
        # the same invention on a grant would widen permission, so the grant is dropped instead
        from remembro.claims.grounding import ground
        g = json.loads(json.dumps(next(x for x in GOLD2["claims"] if x["id"] == "c_coo_opex")))
        g["constraints"]["category"] = "grant_expenditure"
        g["object"] = "grant expenditure"  # object and category agree, but the text says operational
        _, dropped = ground([g])
        assert dropped and "category" in dropped[0][1]

    def test_r22_snapshot_on_fixture_2_has_no_unjustified_allow(self):
        cands = json.loads((ROOT / "fixtures/runs/r22-writer-hv.claims.json").read_text())["claims"]
        report, _ = run(GOLD2, cands, "r22-hv")
        assert report.unjustified_allow == 0, [d for d in report.decisions if not d["ok"]]


class TestDailyUseLessons:
    """What the first live Slack message taught the boundary."""

    def test_date_ranges_sharing_a_month_ground_both_ends(self):
        from remembro.temporal.normalize import dates_in
        got = {d.isoformat() for d in dates_in("I'm on leave 21 to 25 September 2026, back 26 September 2026")}
        assert {"2026-09-21", "2026-09-25", "2026-09-26"} <= got
        got = {d.isoformat() for d in dates_in("between 1 and 15 September 2026; 3–7 October 2026; from 8 until 12 Nov 2026")}
        assert {"2026-09-01", "2026-09-15", "2026-10-03", "2026-10-07", "2026-11-08", "2026-11-12"} <= got

    def test_doubt_attaches_to_the_rejected_claims_subject_not_everyone_in_the_quote(self):
        # a Slack paragraph names Alice and Carol; a rejected restriction about Carol must not
        # make Alice's unrelated capex request UNKNOWN
        quote = "Heads up team, I'm on leave. Carol Evans can approve invoices up to AUD 30,000. Anything bigger waits for me. Alice Morgan"
        ev = {"id": "e1", "document_id": "slack", "page": 1, "paragraph": 1, "section": None, "start_offset": 0, "end_offset": len(quote), "quoted_text": quote, "authority": "body"}
        bad = {"id": "bad", "subject": "Carol Evans", "subject_kind": "person", "predicate": "may_not", "object": None, "object_kind": "none", "modality": "prohibition", "polarity": "negative", "constraints": {}, "valid_from": None, "valid_until": None, "confidence": 1.0, "evidence": [ev]}
        s, _ = state_from([*GOLD["claims"], bad])
        s1 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_01")  # Alice
        assert decide(Request.model_validate(s1["request"]), s, resolver()).decision.value == "ALLOW"
        carol = Request.model_validate({**next(x for x in GOLD["scenarios"] if x["id"] == "scenario_08")["request"], "on": "2026-09-05"})
        assert decide(carol, s, resolver()).decision.value == "UNKNOWN"

    def test_object_that_does_not_resolve_yields_to_a_category_that_does(self):
        c = json.loads(json.dumps(next(x for x in GOLD["claims"] if x["id"] == "claim_fd_opex")))
        c["object"] = "invoices"  # free text; the category field says operational_expenditure
        s, _ = state_from([x for x in GOLD["claims"] if x["id"] != "claim_fd_opex"] + [c])
        assert s.claim("claim_fd_opex").status.value == "ACCEPTED"
        assert s.claim("claim_fd_opex").object_entity == "operational_expenditure"

    def test_a_suspension_must_be_grounded_in_a_suspension_word(self):
        from remembro.claims.grounding import ground
        quote = "Heads up team, I'm on leave 21 to 25 September 2026. Anything bigger waits for me. Alice Morgan"
        ev = {"id": "e1", "document_id": "slack", "page": 1, "paragraph": 1, "section": None, "start_offset": 0, "end_offset": len(quote), "quoted_text": quote, "authority": "body"}
        susp = {"id": "s", "subject": "Alice Morgan", "subject_kind": "person", "predicate": "suspended", "object": None, "object_kind": "none", "modality": "assertion", "polarity": "positive", "constraints": {}, "valid_from": "2026-09-21", "valid_until": "2026-09-25", "confidence": 1.0, "evidence": [ev]}
        _, dropped = ground([susp])
        assert dropped and "suspend" in dropped[0][1]
        real = next(x for x in GOLD["claims"] if x["id"] == "claim_carol_suspended")
        kept, dropped = ground([real])
        assert kept and not dropped
        rev = next(x for x in GOLD["claims"] if x["id"] == "claim_revocation")
        kept, dropped = ground([rev])
        assert kept and not dropped


class TestTableHeadersAndRestatedCeilings:
    def test_table_rows_carry_their_header(self):
        rows = [s for s in DOC.spans if s.kind == "table_row"]
        assert rows and all(s.header for s in rows)
        appx = next(s for s in rows if s.text.startswith("Procurement Manager | AUD 25,000"))
        assert appx.header and "Operational" in appx.header and "Capital" in appx.header

    def test_llm_passage_includes_columns_but_evidence_quotes_the_row(self):
        from remembro.extraction.extractors import LlmExtractor
        appx = next(s for s in DOC.spans if s.text.startswith("Procurement Manager | AUD 25,000"))
        passage = LlmExtractor.passage_for(appx)
        assert passage.startswith("Table columns:") and appx.text in passage

    def test_a_ceiling_restated_as_a_prohibition_is_not_a_contradiction(self):
        pos = json.loads(json.dumps(next(x for x in GOLD["claims"] if x["id"] == "claim_carol_extension")))
        neg = {**json.loads(json.dumps(pos)), "id": "ceiling", "modality": "prohibition", "polarity": "negative"}
        s, _ = state_from([*GOLD["claims"], neg])
        assert not any("ceiling" in c.claims for c in s.contradictions)
        s9 = next(x for x in GOLD["scenarios"] if x["id"] == "scenario_09")
        assert decide(Request.model_validate(s9["request"]), s, resolver()).decision.value == "ALLOW"
