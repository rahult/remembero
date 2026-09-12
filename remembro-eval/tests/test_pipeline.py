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
