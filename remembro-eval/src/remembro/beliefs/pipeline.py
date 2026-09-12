"""The deterministic side of the trust boundary.

candidate claims → schema validation → entity resolution → conflict detection →
claim acceptance → belief update. Every transition is logged; nothing here
consults a model.
"""

from __future__ import annotations

from datetime import date

from dataclasses import dataclass, field

from pydantic import ValidationError

from remembro.claims.grounding import ground
from remembro.claims.models import (
    Belief,
    BeliefStatus,
    CandidateClaim,
    Claim,
    ClaimStatus,
    Contradiction,
    ContradictionType,
    Entity,
    EntityKind,
    MatchVerdict,
    Modality,
    Polarity,
    Predicate,
    Proposition,
    Resolution,
    StateTransition,
)
from remembro.entities.resolver import EntityResolver
from remembro.temporal.normalize import overlaps


@dataclass
class WorldState:
    entities: list[Entity]
    claims: list[Claim] = field(default_factory=list)
    resolutions: list[Resolution] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    unread_spans: list[dict] = field(default_factory=list)  # spans the extractor failed on
    unreadable_claims: list[dict] = field(default_factory=list)  # candidates the schema or grounding rejected

    def log(self, event: str, subject_id: str, reason: str, related_id: str | None = None) -> None:
        self.transitions.append(
            StateTransition(
                event=event,
                subject_id=subject_id,
                related_id=related_id,
                reason=reason,
                sequence=len(self.transitions) + 1,
            )
        )

    def claim(self, claim_id: str) -> Claim:
        return next(c for c in self.claims if c.id == claim_id)

    def belief(self, belief_id: str) -> Belief:
        return next(b for b in self.beliefs if b.id == belief_id)


def validate_candidates(raw: list[dict]) -> tuple[list[Claim], list[tuple[dict, str]]]:
    """Schema validation: a candidate that does not parse never enters the state."""
    claims: list[Claim] = []
    rejected: list[tuple[dict, str]] = []
    for item in raw:
        try:
            candidate = CandidateClaim.model_validate(item)
        except ValidationError as error:
            rejected.append((item, str(error.errors()[0].get("msg", "invalid"))))
            continue
        claims.append(Claim(**candidate.model_dump(), status=ClaimStatus.VALIDATED))
    return claims, rejected


def _kind(kind: str) -> EntityKind:
    return {"person": EntityKind.PERSON, "role": EntityKind.ROLE, "category": EntityKind.CATEGORY}[kind]


def resolve_claims(state: WorldState, resolver: EntityResolver) -> None:
    for claim in state.claims:
        if claim.status is not ClaimStatus.VALIDATED:
            continue
        subject = resolver.resolve(claim.subject, _kind(claim.subject_kind))
        state.resolutions.append(subject)
        obj: Resolution | None = None
        if claim.predicate in OBJECTLESS:
            # the predicate defines its arity: "X's approval authority is suspended" is about
            # X alone, whatever the extractor put in the object slot
            claim.object, claim.object_kind = None, "none"
        if claim.object is not None and claim.object_kind != "none":
            obj = resolver.resolve(claim.object, _kind(claim.object_kind))
            state.resolutions.append(obj)
        if subject.verdict is MatchVerdict.MATCH and (obj is None or obj.verdict is MatchVerdict.MATCH):
            claim.subject_entity = subject.entity_id
            claim.object_entity = obj.entity_id if obj else None
            if obj is not None and claim.object_kind == "category" and claim.constraints.category != obj.entity_id:
                # the object is quoted text that resolved; the category field is the extractor's
                # classification into a fixed list. The grounded one wins.
                if claim.constraints.category:
                    state.log("CLAIM_NORMALISED", claim.id, f"category '{claim.constraints.category}' replaced by resolved object '{obj.entity_id}'")
                claim.constraints.category = obj.entity_id
            claim.status = ClaimStatus.RESOLVED
            state.log("CLAIM_RESOLVED", claim.id, f"subject {subject.reason}" + (f"; object {obj.reason}" if obj else ""))
        else:
            # keep the claim, but it can never become a belief about a specific entity
            claim.status = ClaimStatus.REJECTED
            unresolved = subject if subject.verdict is not MatchVerdict.MATCH else obj
            # remember who this might be about: a POSSIBLE_MATCH grant must turn a later
            # decision about that person into UNKNOWN, never silently into DENY or ALLOW
            if subject.verdict in (MatchVerdict.POSSIBLE_MATCH, MatchVerdict.MATCH):
                claim.subject_entity = subject.entity_id
                claim.possible_subjects = subject.candidates
            if obj is not None and obj.verdict is MatchVerdict.POSSIBLE_MATCH:
                claim.object_entity = obj.entity_id
                claim.possible_objects = obj.candidates
            claim.rejection_reason = (
                f"unresolved {('subject' if unresolved is subject else 'object')} '{unresolved.mention}': "
                f"{unresolved.verdict.value} ({unresolved.reason})"
            )
            state.log("CLAIM_UNRESOLVED", claim.id, claim.rejection_reason)


OBJECTLESS = {Predicate.SUSPENDED}


def _same_paragraph(a: Claim, b: Claim) -> bool:
    return any((x.document_id, x.page, x.paragraph) == (y.document_id, y.page, y.paragraph) for x in a.evidence for y in b.evidence)


def _limit_key(claim: Claim) -> tuple | None:
    if claim.predicate not in (Predicate.MAY_APPROVE, Predicate.APPENDIX_LIMIT):
        return None
    return (claim.subject_entity, claim.constraints.category or claim.object_entity)


def detect_conflicts(state: WorldState) -> None:
    """Same subject and category, both current, different limit or polarity."""
    resolved = [c for c in state.claims if c.status is ClaimStatus.RESOLVED and _limit_key(c)]
    seen: set[tuple[str, str]] = set()
    n = 0
    for i, a in enumerate(resolved):
        for b in resolved[i + 1 :]:
            if _limit_key(a) != _limit_key(b):
                continue
            if not overlaps(a.valid_from, a.valid_until, b.valid_from, b.valid_until):
                continue
            pair = tuple(sorted((a.id, b.id)))
            if pair in seen:
                continue
            seen.add(pair)
            if a.polarity != b.polarity:
                ctype = ContradictionType.POLARITY_MISMATCH
            elif (a.constraints.maximum_amount or 0) != (b.constraints.maximum_amount or 0):
                ctype = ContradictionType.LIMIT_MISMATCH
            else:
                continue
            n += 1
            contradiction = Contradiction(id=f"contradiction_{n}", type=ctype, claims=[a.id, b.id])
            # the one resolution rule the document itself states: the body prevails over the appendix
            auth_a = a.evidence[0].authority
            auth_b = b.evidence[0].authority
            if auth_a != auth_b and "appendix" in (auth_a, auth_b):
                winner = a if auth_a == "body" else b
                loser = b if winner is a else a
                contradiction.resolved = True
                contradiction.resolution = "body prevails over appendix (Appendix A, stated precedence)"
                contradiction.winner = winner.id
                loser.status = ClaimStatus.CONFLICTED
                state.log("CLAIM_CONFLICTED", loser.id, contradiction.resolution, related_id=winner.id)
            else:
                a.status = ClaimStatus.CONFLICTED
                b.status = ClaimStatus.CONFLICTED
                state.log("CONTRADICTION_UNRESOLVED", a.id, f"{ctype.value} with {b.id}; equal authority", related_id=b.id)
            state.contradictions.append(contradiction)


def _same_key(a: Claim, b: Claim) -> bool:
    if a.predicate is not b.predicate or a.subject_entity != b.subject_entity:
        return False
    if a.predicate in (Predicate.MAY_APPROVE, Predicate.APPENDIX_LIMIT):
        return (a.constraints.category or a.object_entity) == (b.constraints.category or b.object_entity)
    return a.object_entity == b.object_entity


def apply_amendments(state: WorldState, document_dates: dict[str, date] | None) -> None:
    """Transaction time: a claim from a later document supersedes the same claim from an earlier
    one, from the later document's effective date. It does not contradict it. A later restatement
    that only adds an end date (a role that ceased, a suspension lifted) closes the earlier claim."""
    if not document_dates:
        return
    from datetime import timedelta

    def doc_date(c: Claim) -> date | None:
        return document_dates.get(c.evidence[0].document_id) if c.evidence else None

    resolved = [c for c in state.claims if c.status is ClaimStatus.RESOLVED]
    for later in resolved:
        d_later = doc_date(later)
        if d_later is None:
            continue
        for earlier in resolved:
            d_earlier = doc_date(earlier)
            if earlier is later or d_earlier is None or d_earlier >= d_later or not _same_key(earlier, later):
                continue
            if later.predicate in (Predicate.MAY_APPROVE, Predicate.APPENDIX_LIMIT):
                if (later.constraints.maximum_amount, later.polarity) == (earlier.constraints.maximum_amount, earlier.polarity):
                    continue  # a restatement
                if later.valid_from is None:
                    later.valid_from = d_later
                    state.log("CLAIM_DATED", later.id, f"valid from its document's effective date {d_later}")
                if earlier.valid_until is None or earlier.valid_until >= later.valid_from:
                    earlier.valid_until = later.valid_from - timedelta(days=1)
                    earlier.superseded_by = later.id
                    state.log("CLAIM_SUPERSEDED", earlier.id, f"amended by {later.id} from {later.valid_from}", related_id=later.id)
            else:
                # holds_role / suspended: the later claim's end date closes the earlier open claim
                if later.valid_until is not None and (earlier.valid_until is None or earlier.valid_until > later.valid_until):
                    earlier.valid_until = later.valid_until
                    earlier.superseded_by = later.id
                    state.log("CLAIM_SUPERSEDED", earlier.id, f"ended on {later.valid_until} by {later.id}", related_id=later.id)


def apply_lifts(state: WorldState) -> None:
    """'X is not suspended from D' is a lift: it ends every open suspension of X at D - 1. A
    negative suspension never becomes a suspension belief."""
    from datetime import timedelta

    for lift in state.claims:
        if lift.predicate is not Predicate.SUSPENDED or lift.polarity is not Polarity.NEGATIVE or lift.status is not ClaimStatus.RESOLVED or lift.valid_from is None:
            continue
        for susp in state.claims:
            if (
                susp.predicate is Predicate.SUSPENDED
                and susp.polarity is Polarity.POSITIVE
                and susp.status is ClaimStatus.RESOLVED
                and susp.subject_entity == lift.subject_entity
                and (susp.valid_from is None or susp.valid_from < lift.valid_from)
                and (susp.valid_until is None or susp.valid_until >= lift.valid_from)
            ):
                susp.valid_until = lift.valid_from - timedelta(days=1)
                susp.superseded_by = lift.id
                state.log("CLAIM_SUPERSEDED", susp.id, f"suspension lifted from {lift.valid_from} by {lift.id}", related_id=lift.id)


def apply_revocations(state: WorldState) -> None:
    """A revocation supersedes the end date of the delegation it names; if it names no delegation,
    it ends the personal authorisation of the person it names in that category."""
    from datetime import timedelta

    for rev in state.claims:
        if rev.predicate is not Predicate.REVOKES_DELEGATION or rev.status is not ClaimStatus.RESOLVED:
            continue
        matched = False
        for deleg in state.claims:
            if (
                deleg.predicate is Predicate.DELEGATES
                and deleg.status in (ClaimStatus.RESOLVED, ClaimStatus.ACCEPTED)
                and deleg.subject_entity == rev.subject_entity
                and deleg.object_entity == rev.object_entity
                and (rev.constraints.category in (None, deleg.constraints.category))
                and rev.valid_from is not None
            ):
                if deleg.valid_until is None or rev.valid_from <= deleg.valid_until:
                    # the delegation ends the day before the revocation takes effect
                    matched = True
                    deleg.valid_until = rev.valid_from - timedelta(days=1)
                    deleg.superseded_by = rev.id
                    state.log("CLAIM_SUPERSEDED", deleg.id, f"end date superseded by revocation effective {rev.valid_from}", related_id=rev.id)
                    # an extractor may restate "Alice delegates to Bob up to 100k" as a grant
                    # "Bob may approve up to 100k" from the same paragraph; that grant is the
                    # delegation, so the revocation ends it too
                    for grant in state.claims:
                        if (
                            grant.predicate is Predicate.MAY_APPROVE
                            and grant.status in (ClaimStatus.RESOLVED, ClaimStatus.ACCEPTED)
                            and grant.subject_entity == deleg.object_entity
                            and grant.polarity is Polarity.POSITIVE
                            and (grant.constraints.category or grant.object_entity) == (deleg.constraints.category or deleg.object_entity)
                            and _same_paragraph(grant, deleg)
                            and (grant.valid_until is None or rev.valid_from <= grant.valid_until)
                        ):
                            grant.valid_until = deleg.valid_until
                            grant.superseded_by = rev.id
                            state.log("CLAIM_SUPERSEDED", grant.id, f"restates delegation {deleg.id}; ended by revocation effective {rev.valid_from}", related_id=rev.id)
        if not matched and rev.valid_from is not None and rev.object_entity:
            for grant in state.claims:
                if (
                    grant.predicate is Predicate.MAY_APPROVE
                    and grant.status in (ClaimStatus.RESOLVED, ClaimStatus.ACCEPTED)
                    and grant.subject_entity == rev.object_entity
                    and grant.polarity is Polarity.POSITIVE
                    and (rev.constraints.category in (None, grant.constraints.category or grant.object_entity))
                    and grant.valid_from is not None  # a temporary authorisation, not a standing role limit
                    and (grant.valid_until is None or rev.valid_from <= grant.valid_until)
                ):
                    grant.valid_until = rev.valid_from - timedelta(days=1)
                    grant.superseded_by = rev.id
                    state.log("CLAIM_SUPERSEDED", grant.id, f"personal authorisation ended by revocation effective {rev.valid_from}", related_id=rev.id)
        rev.status = ClaimStatus.ACCEPTED
        state.log("CLAIM_ACCEPTED", rev.id, "revocation applied")


def accept_claims(state: WorldState) -> None:
    for claim in state.claims:
        if claim.status is ClaimStatus.RESOLVED:
            claim.status = ClaimStatus.ACCEPTED
            state.log("CLAIM_ACCEPTED", claim.id, "resolved, no unresolved conflict")


def update_beliefs(state: WorldState) -> None:
    """One belief per accepted or conflicted claim; conflicted ones are DISPUTED."""
    n = 0
    for claim in state.claims:
        if claim.status not in (ClaimStatus.ACCEPTED, ClaimStatus.CONFLICTED):
            continue
        n += 1
        contradicting = [
            other
            for c in state.contradictions
            if claim.id in c.claims
            for other in c.claims
            if other != claim.id
        ]
        winner_elsewhere = any(c.resolved and claim.id in c.claims and c.winner != claim.id for c in state.contradictions)
        if winner_elsewhere:
            status = BeliefStatus.SUPERSEDED
        elif claim.status is ClaimStatus.CONFLICTED:
            status = BeliefStatus.DISPUTED
        else:
            status = BeliefStatus.SUPPORTED
        state.beliefs.append(
            Belief(
                id=f"belief_{n}",
                proposition=Proposition(
                    subject=claim.subject_entity or claim.subject,
                    predicate=claim.predicate,
                    object=claim.object_entity if claim.object_entity else (claim.constraints.category if claim.object_kind == "none" else claim.object),
                    modality=claim.modality,
                    polarity=claim.polarity,
                    constraints=claim.constraints,
                ),
                status=status,
                supporting_claims=[claim.id],
                contradicting_claims=contradicting,
                valid_from=claim.valid_from,
                valid_until=claim.valid_until,
            )
        )
        state.log("BELIEF_" + status.value, f"belief_{n}", f"from {claim.id}", related_id=claim.id)


def build_state(raw_candidates: list[dict], entities: list[Entity], resolver: EntityResolver, document_dates: dict[str, str | date] | None = None) -> tuple[WorldState, list[tuple[dict, str]]]:
    unread = [c for c in raw_candidates if isinstance(c, dict) and "error" in c]
    claims, rejected = validate_candidates([c for c in raw_candidates if not (isinstance(c, dict) and "error" in c)])
    state = WorldState(entities=entities, claims=claims, unread_spans=unread)
    for item in unread:
        state.log("SPAN_UNREAD", str(item.get("id", "?")), f"extractor error: {item.get('error', '')[:120]}")
    for item, reason in rejected:
        state.log("CANDIDATE_REJECTED", str(item.get("id", "?")), f"schema: {reason}")
    grounded, ungrounded = ground([c.model_dump(mode="json") for c in claims])
    grounded_ids = {c["id"] for c in grounded}
    for item, reason in ungrounded:
        state.log("CANDIDATE_REJECTED", str(item.get("id", "?")), f"grounding: {reason}")
        rejected.append((item, reason))
    state.claims = [c for c in claims if c.id in grounded_ids]
    state.unreadable_claims = [item for item, _ in rejected]
    resolve_claims(state, resolver)
    dates = {k: (date.fromisoformat(v) if isinstance(v, str) else v) for k, v in (document_dates or {}).items()}
    apply_amendments(state, dates)
    apply_lifts(state)
    apply_revocations(state)
    detect_conflicts(state)
    accept_claims(state)
    update_beliefs(state)
    return state, rejected
