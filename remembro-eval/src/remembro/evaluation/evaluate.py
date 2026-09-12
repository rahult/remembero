"""Per-capability metrics. Nothing is collapsed into one accuracy score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from remembro.beliefs.pipeline import WorldState, build_state
from remembro.claims.models import Decision, DecisionOutcome, Entity, Request
from remembro.decision.engine import evaluate as decide
from remembro.entities.resolver import EntityResolver


def load_gold(path: Path) -> dict:
    return json.loads(path.read_text())


def claim_key(claim: dict) -> tuple:
    """What makes two claims 'the same' for extraction scoring: normalised subject, predicate,
    category, polarity and limit. Surface spelling of the subject is folded lightly."""
    subject = str(claim.get("subject", "")).lower().replace(".", "").strip()
    if subject.endswith("s") and subject.startswith("procurement manager"):
        subject = "procurement manager"
    if subject == "cfo":
        subject = "chief financial officer"
    constraints = claim.get("constraints") or {}
    obj = str(claim.get("object") or "").lower().strip()
    return (
        subject,
        claim.get("predicate"),
        constraints.get("category") or obj,
        claim.get("polarity", "positive"),
        constraints.get("maximum_amount"),
        str(claim.get("valid_from") or ""),
        str(claim.get("object") or "").lower() if claim.get("predicate") in ("delegates", "revokes_delegation", "holds_role") else "",
    )


@dataclass
class Report:
    extractor: str
    claim_precision: float
    claim_recall: float
    claims_extracted: int
    claims_gold: int
    schema_rejections: int
    entity_resolution_accuracy: float
    temporal_accuracy: float
    modality_accuracy: float
    contradiction_recall: float
    provenance_coverage: float
    decision_accuracy: float
    decisions: list[dict]
    unjustified_allow: int
    unjustified_allow_rate: float
    unknown_count: int

    def to_dict(self) -> dict:
        return self.__dict__


def run(gold: dict, candidates: list[dict], extractor_name: str, match_threshold: float = 0.9) -> tuple[Report, WorldState]:
    entities = [Entity.model_validate(e) for e in gold["entities"]]
    resolver = EntityResolver(entities, match_threshold=match_threshold)
    state, rejected = build_state(candidates, entities, resolver)

    # extraction precision / recall against gold claims
    gold_keys = {claim_key(c) for c in gold["claims"]}
    got_keys = {claim_key(c) for c in candidates if "predicate" in c}
    tp = len(gold_keys & got_keys)
    precision = tp / len(got_keys) if got_keys else 0.0
    recall = tp / len(gold_keys) if gold_keys else 0.0

    # entity resolution: gold entities' canonical names and aliases must MATCH; the ambiguous initial must not
    checks = 0
    correct = 0
    for e in entities:
        for name in [e.canonical_name, *e.aliases]:
            checks += 1
            r = resolver.resolve(name, e.kind)
            correct += int(r.verdict.value == "MATCH" and r.entity_id == e.id)
    for mention, expect in (("D. Smith", "POSSIBLE_MATCH"), ("Erin Fox", "NO_MATCH")):
        checks += 1
        correct += int(resolver.resolve(mention, entities[0].kind).verdict.value == expect)
    entity_accuracy = correct / checks

    # temporal: gold claims with a validity interval must be matched with the same interval
    gold_temporal = [c for c in gold["claims"] if c.get("valid_from") or c.get("valid_until")]
    got_by_key = {claim_key(c): c for c in candidates if "predicate" in c}
    t_ok = 0
    for c in gold_temporal:
        g = got_by_key.get(claim_key(c))
        if g and str(g.get("valid_from") or "") == str(c.get("valid_from") or "") and str(g.get("valid_until") or "") == str(c.get("valid_until") or ""):
            t_ok += 1
    temporal_accuracy = t_ok / len(gold_temporal) if gold_temporal else 1.0

    # modality: matched claims must carry the gold modality and polarity
    m_ok = 0
    m_n = 0
    for c in gold["claims"]:
        g = got_by_key.get(claim_key(c))
        if g:
            m_n += 1
            m_ok += int(g.get("modality", "assertion") == c.get("modality", "assertion") and g.get("polarity", "positive") == c.get("polarity", "positive"))
    modality_accuracy = m_ok / m_n if m_n else 0.0

    # contradictions: each gold contradiction is a pair of gold claim ids; found when a
    # detected contradiction covers both keys
    found = 0
    gold_by_id = {c["id"]: c for c in gold["claims"]}
    detected_pairs = []
    for con in state.contradictions:
        keys = {claim_key(state.claim(cid).model_dump(mode="json")) for cid in con.claims}
        detected_pairs.append(keys)
    for con in gold["contradictions"]:
        keys = {claim_key(gold_by_id[cid]) for cid in con["claims"]}
        if any(keys <= d for d in detected_pairs):
            found += 1
    contradiction_recall = found / len(gold["contradictions"]) if gold["contradictions"] else 1.0

    accepted = [c for c in state.claims if c.status.value in ("ACCEPTED", "CONFLICTED")]
    provenance = sum(1 for c in accepted if c.evidence and all(e.quoted_text and e.end_offset > e.start_offset for e in c.evidence)) / len(accepted) if accepted else 1.0

    decisions: list[dict] = []
    right = 0
    unjustified = 0
    unknown = 0
    for scenario in gold["scenarios"]:
        request = Request.model_validate(scenario["request"])
        decision: Decision = decide(request, state, resolver)
        expected = scenario["expected"]["decision"]
        ok = decision.decision.value == expected
        right += int(ok)
        unknown += int(decision.decision is DecisionOutcome.UNKNOWN)
        if decision.decision is DecisionOutcome.ALLOW and expected != "ALLOW":
            unjustified += 1
        decisions.append({"id": scenario["id"], "question": scenario["question"], "expected": expected, "got": decision.decision.value, "ok": ok, "reasons": decision.reasons, "beliefs_used": decision.beliefs_used, "evidence_used": decision.evidence_used})

    return Report(
        extractor=extractor_name,
        claim_precision=round(precision, 4),
        claim_recall=round(recall, 4),
        claims_extracted=len(got_keys),
        claims_gold=len(gold_keys),
        schema_rejections=len(rejected),
        entity_resolution_accuracy=round(entity_accuracy, 4),
        temporal_accuracy=round(temporal_accuracy, 4),
        modality_accuracy=round(modality_accuracy, 4),
        contradiction_recall=round(contradiction_recall, 4),
        provenance_coverage=round(provenance, 4),
        decision_accuracy=round(right / len(gold["scenarios"]), 4),
        decisions=decisions,
        unjustified_allow=unjustified,
        unjustified_allow_rate=round(unjustified / len(gold["scenarios"]), 4),
        unknown_count=unknown,
    ), state


def format_report(report: Report) -> str:
    lines = [
        f"extractor               {report.extractor}",
        f"Decision accuracy       {sum(d['ok'] for d in report.decisions)} / {len(report.decisions)}",
        f"Claim precision         {report.claim_precision*100:.1f}%  ({report.claims_extracted} extracted, {report.schema_rejections} rejected at the boundary)",
        f"Claim recall            {report.claim_recall*100:.1f}%  ({report.claims_gold} gold)",
        f"Entity resolution       {report.entity_resolution_accuracy*100:.1f}%",
        f"Temporal accuracy       {report.temporal_accuracy*100:.1f}%",
        f"Modality accuracy       {report.modality_accuracy*100:.1f}%",
        f"Contradictions detected {report.contradiction_recall*100:.0f}%",
        f"Provenance coverage     {report.provenance_coverage*100:.0f}%",
        f"Unjustified ALLOW       {report.unjustified_allow}",
        f"UNKNOWN decisions       {report.unknown_count}",
        "",
    ]
    for d in report.decisions:
        mark = "ok  " if d["ok"] else "MISS"
        lines.append(f"  {mark} {d['id']}: expected {d['expected']}, got {d['got']}")
    return "\n".join(lines)
