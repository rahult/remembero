"""Deterministic decision over beliefs. The invariant: uncertainty never becomes ALLOW."""

from __future__ import annotations

from remembro.beliefs.pipeline import WorldState
from remembro.claims.models import (
    Belief,
    BeliefStatus,
    Decision,
    DecisionOutcome,
    EntityKind,
    MatchVerdict,
    Modality,
    Polarity,
    Predicate,
    Request,
)
from remembro.entities.resolver import EntityResolver
from remembro.temporal.normalize import in_interval


def _current(belief: Belief, day) -> bool:
    return in_interval(day, belief.valid_from, belief.valid_until)


def evaluate(request: Request, state: WorldState, resolver: EntityResolver) -> Decision:
    reasons: list[str] = []
    beliefs_used: list[str] = []

    def used(*bs: Belief) -> None:
        for b in bs:
            if b.id not in beliefs_used:
                beliefs_used.append(b.id)

    def evidence_for(belief_ids: list[str]) -> list[str]:
        out: list[str] = []
        for bid in belief_ids:
            for cid in state.belief(bid).supporting_claims:
                for ev in state.claim(cid).evidence:
                    if ev.id not in out:
                        out.append(ev.id)
        return out

    def unknown(reason: str) -> Decision:
        reasons.append(reason)
        return Decision(decision=DecisionOutcome.UNKNOWN, request=request, reasons=reasons, beliefs_used=beliefs_used, evidence_used=evidence_for(beliefs_used), unknown_reason=reason)

    def deny(reason: str) -> Decision:
        reasons.append(reason)
        return Decision(decision=DecisionOutcome.DENY, request=request, reasons=reasons, beliefs_used=beliefs_used, evidence_used=evidence_for(beliefs_used))

    # 1. who is asking
    actor = resolver.resolve(request.actor, EntityKind.PERSON)
    if actor.verdict is MatchVerdict.POSSIBLE_MATCH:
        return unknown(f"actor identity unresolved: '{request.actor}' is a POSSIBLE_MATCH for {actor.entity_id} ({actor.reason}); identity must be established before authority is inferred")
    if actor.verdict is MatchVerdict.NO_MATCH:
        return deny(f"actor '{request.actor}' is not a known person; no authority can be attributed")
    person = actor.entity_id
    reasons.append(f"actor resolved to {person} ({actor.reason})")
    category = resolver.resolve(request.resource, EntityKind.CATEGORY)
    if category.verdict is not MatchVerdict.MATCH:
        return unknown(f"expenditure category '{request.resource}' is not one this policy defines")
    cat = category.entity_id

    # 2. hard prohibitions first
    if request.self_benefit:
        prohibitions = [b for b in state.beliefs if b.proposition.predicate is Predicate.MAY_APPROVE and b.proposition.modality is Modality.PROHIBITION and b.proposition.constraints.condition and "benefit" in b.proposition.constraints.condition]
        used(*prohibitions)
        return deny("the request benefits the approver; the policy prohibits approving expenditure from which the approver benefits")
    if request.own_expense_claim:
        return deny("an approver must not approve their own expense claims")

    # 3. suspension
    suspensions = [b for b in state.beliefs if b.proposition.predicate is Predicate.SUSPENDED and b.proposition.polarity is Polarity.POSITIVE and b.proposition.subject == person and _current(b, request.on) and b.status is BeliefStatus.SUPPORTED]
    if suspensions:
        used(*suspensions)
        return deny(f"{person}'s approval authority is suspended on {request.on}; a suspended person may not exercise authority held directly or by delegation")

    # 3a. a passage the extractor failed to read, which names this person, may hold the
    # restriction that changes the answer: an incomplete read is doubt, not absence
    entity = next(e for e in state.entities if e.id == person)
    names = [entity.canonical_name, *entity.aliases]
    unread = [u for u in state.unread_spans if any(n and n.lower() in (u.get("span_text") or "").lower() for n in names)]
    if unread:
        return unknown(f"the document was not fully read: span {unread[0].get('id')} names {person} but the extractor failed on it ({unread[0].get('error', '')[:80]})")

    # 3a'. a candidate the boundary rejected (unknown predicate, malformed, ungrounded) whose
    # evidence names this person: the extractor read something about them that the engine
    # could not; that is doubt, not silence
    def quotes(item: dict) -> str:
        return " ".join(e.get("quoted_text", "") for e in item.get("evidence") or [] if isinstance(e, dict))
    KNOWN = {p.value for p in Predicate}

    def might_restrict(item: dict) -> bool:
        # dropping a positive permission cannot create an ALLOW; dropping anything else might
        return (
            item.get("predicate") not in KNOWN
            or item.get("polarity") == "negative"
            or item.get("modality") == "prohibition"
            or item.get("predicate") in (Predicate.SUSPENDED.value, Predicate.REVOKES_DELEGATION.value)
        )

    def about_person(item: dict) -> bool:
        # the claim is about its subject and object, not about everyone its paragraph mentions
        for field in ("subject", "object"):
            mention, kind = item.get(field), item.get(f"{field}_kind")
            if mention and kind == "person":
                res = resolver.resolve(str(mention), EntityKind.PERSON)
                if res.entity_id == person and res.verdict in (MatchVerdict.MATCH, MatchVerdict.POSSIBLE_MATCH):
                    return True
        return False

    unreadable = [u for u in state.unreadable_claims if might_restrict(u) and about_person(u)]
    if unreadable:
        u = unreadable[0]
        return unknown(f"a claim about {person} was rejected at the boundary ({u.get('id')}: predicate {u.get('predicate')!r}); the engine will not decide past speech about a person it could not read")

    # 3b. a restriction on this person that could not be fully resolved is doubt, not absence:
    # nobody may be allowed past a prohibition the engine could not read
    unresolved_restrictions = [
        c for c in state.claims
        if c.status.value == "REJECTED" and c.subject_entity == person and c.rejection_reason and c.rejection_reason.startswith("unresolved object")
        and (c.predicate is Predicate.SUSPENDED or (c.predicate is Predicate.MAY_APPROVE and c.polarity is Polarity.NEGATIVE))
        and in_interval(request.on, c.valid_from, c.valid_until)
    ]
    if unresolved_restrictions:
        c = unresolved_restrictions[0]
        return unknown(f"a restriction on {person} could not be resolved ({c.id}: {c.rejection_reason}); the engine will not allow past a prohibition it cannot read")

    # 4. authority sources: direct via role, or by delegation
    roles = [b for b in state.beliefs if b.proposition.predicate is Predicate.HOLDS_ROLE and b.proposition.subject == person and _current(b, request.on) and b.status is BeliefStatus.SUPPORTED]
    limits: list[tuple[Belief, float | None, str]] = []  # (belief, max, source)
    disputed: list[Belief] = []
    for role in roles:
        used(role)
        reasons.append(f"{person} holds role {role.proposition.object}")
        for b in state.beliefs:
            p = b.proposition
            if p.predicate is Predicate.MAY_APPROVE and p.subject == role.proposition.object and (p.constraints.category or p.object) == cat and _current(b, request.on) and p.polarity is Polarity.POSITIVE:
                if b.status is BeliefStatus.DISPUTED:
                    disputed.append(b)
                elif b.status is BeliefStatus.SUPPORTED:
                    limits.append((b, p.constraints.maximum_amount, f"role {role.proposition.object}"))
    # person-level grants (temporary extensions written to a named person)
    for b in state.beliefs:
        p = b.proposition
        if p.predicate is Predicate.MAY_APPROVE and p.subject == person and (p.constraints.category or p.object) == cat and _current(b, request.on) and p.polarity is Polarity.POSITIVE:
            if b.status is BeliefStatus.DISPUTED:
                disputed.append(b)
            elif b.status is BeliefStatus.SUPPORTED:
                limits.append((b, p.constraints.maximum_amount, "personal grant"))
    # delegations to this person, bounded by the delegator's own limit
    for b in state.beliefs:
        p = b.proposition
        if p.predicate is Predicate.DELEGATES and p.object == person and (p.constraints.category or cat) == cat and _current(b, request.on) and b.status is BeliefStatus.SUPPORTED:
            delegator = p.subject
            delegator_roles = [r for r in state.beliefs if r.proposition.predicate is Predicate.HOLDS_ROLE and r.proposition.subject == delegator and _current(r, request.on)]
            delegator_limits = [
                l.proposition.constraints.maximum_amount
                for r in delegator_roles
                for l in state.beliefs
                if l.proposition.predicate is Predicate.MAY_APPROVE and l.proposition.subject == r.proposition.object and (l.proposition.constraints.category or l.proposition.object) == cat and _current(l, request.on) and l.status is BeliefStatus.SUPPORTED
            ]
            delegator_suspended = any(s.proposition.predicate is Predicate.SUSPENDED and s.proposition.polarity is Polarity.POSITIVE and s.proposition.subject == delegator and _current(s, request.on) for s in state.beliefs)
            if delegator_suspended:
                used(b)
                reasons.append(f"delegation from {delegator} is void on {request.on}: the delegator is suspended")
                continue
            cap = p.constraints.maximum_amount
            if delegator_limits:
                cap = min([c for c in [cap, *delegator_limits] if c is not None], default=cap)
            used(b, *delegator_roles)
            limits.append((b, cap, f"delegation from {delegator}"))

    # a grant to someone who might be this person is not authority, and not its absence:
    # it makes the decision UNKNOWN before any role-level rule can deny or allow it
    possible = [c for c in state.claims if c.status.value == "REJECTED" and c.rejection_reason and "POSSIBLE_MATCH" in c.rejection_reason and c.predicate in (Predicate.MAY_APPROVE, Predicate.DELEGATES) and c.polarity is Polarity.POSITIVE and (person in c.possible_subjects or person in c.possible_objects) and (c.constraints.category or cat) == cat and in_interval(request.on, c.valid_from, c.valid_until)]
    if possible and not any(m is None or request.amount <= m for _, m, _ in limits):
        reasons.append(f"a grant exists for '{possible[0].subject}' which may or may not be {person}: {possible[0].rejection_reason}")
        return unknown(f"authority for {person} over {cat} depends on whether '{possible[0].subject}' is the same person; identity not established")

    # 5. prohibitions on the category (e.g. procurement manager may not approve capital)
    for role in roles:
        for b in state.beliefs:
            p = b.proposition
            if p.predicate is Predicate.MAY_APPROVE and p.subject == role.proposition.object and (p.constraints.category or p.object) == cat and p.polarity is Polarity.NEGATIVE and _current(b, request.on) and b.status is BeliefStatus.SUPPORTED:
                used(b)
                # a role-level prohibition yields to a specific grant to the person: a
                # delegation from someone who may, or a named temporary extension
                if not any(src.startswith("delegation") or src == "personal grant" for _, _, src in limits):
                    return deny(f"role {role.proposition.object} may not approve {cat}")

    # 6. decide
    positive = [(b, m, src) for b, m, src in limits if b.proposition.polarity is Polarity.POSITIVE]
    if not positive and disputed:
        # every source of authority is disputed: decide only if every disputed limit agrees
        used(*disputed)
        values = [d.proposition.constraints.maximum_amount for d in disputed]
        verdicts = {request.amount <= (v if v is not None else float("inf")) for v in values}
        if len(verdicts) == 1:
            if verdicts == {True}:
                reasons.append(f"all disputed limits for {cat} ({sorted(set(str(v) for v in values))}) permit {request.amount}; the contradiction does not affect this request")
                return Decision(decision=DecisionOutcome.ALLOW, request=request, reasons=reasons, beliefs_used=beliefs_used, evidence_used=evidence_for(beliefs_used))
            return deny(f"all disputed limits for {cat} ({sorted(set(str(v) for v in values))}) are below {request.amount}")
        return unknown(f"conflicting limits for {cat} ({sorted(set(str(v) for v in values))}) and the amount {request.amount} falls between them; unresolved contradiction {[b.id for b in disputed]}")
    if not positive:
        return deny(f"no current authority found for {person} over {cat} on {request.on}")
    allowed_by = [(b, m, src) for b, m, src in positive if m is None or request.amount <= m]
    for b, m, src in positive:
        used(b)
        reasons.append(f"{src}: limit {m if m is not None else 'none stated'} {request.currency} for {cat}")
    if disputed:
        # a contradiction only matters if it could change the outcome
        disputed_limits = [d.proposition.constraints.maximum_amount for d in disputed]
        outcomes = {request.amount <= (m if m is not None else float('inf')) for m in [*disputed_limits, *[m for _, m, _ in positive]]}
        if len(outcomes) > 1:
            used(*disputed)
            return unknown(f"conflicting limits for {cat} ({sorted(set(str(m) for m in disputed_limits + [m for _, m, _ in positive]))}) and the amount {request.amount} falls between them; unresolved contradiction {[b.id for b in disputed]}")
    if any(b.proposition.constraints.currency and b.proposition.constraints.currency != request.currency for b, _, _ in positive):
        return unknown(f"currency mismatch: limits are stated in {positive[0][0].proposition.constraints.currency}, request in {request.currency}")
    if allowed_by:
        b, m, src = allowed_by[0]
        reasons.append(f"{request.amount} {request.currency} is within the {src} limit of {m}")
        return Decision(decision=DecisionOutcome.ALLOW, request=request, reasons=reasons, beliefs_used=beliefs_used, evidence_used=evidence_for(beliefs_used))
    return deny(f"{request.amount} {request.currency} exceeds every current limit for {person} over {cat} on {request.on}")
