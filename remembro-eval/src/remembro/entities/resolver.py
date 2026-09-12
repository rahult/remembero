"""Entity resolution with a verdict, never a silent merge.

Mentions score against the register (canonical names and aliases). Above
`match_threshold` is MATCH; between `possible_threshold` and it is
POSSIBLE_MATCH; below is NO_MATCH. A POSSIBLE_MATCH never mutates state and
makes any decision that depends on it UNKNOWN. The thresholds are the knob the
sensitivity test turns.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from remembro.claims.models import Entity, EntityKind, MatchVerdict, Resolution

ROLE_NORMALISATION = {
    "cfo": "chief financial officer",
    "chief financial officer": "chief financial officer",
    "finance director": "finance director",
    "acting finance director": "acting finance director",
    "procurement manager": "procurement manager",
    "procurement managers": "procurement manager",
    "accounts payable lead": "accounts payable lead",
    "board": "board",
    "the board": "board",
}

CATEGORY_NORMALISATION = {
    "operational expenditure": "operational_expenditure",
    "operational": "operational_expenditure",
    "opex": "operational_expenditure",
    "invoices for operational expenditure": "operational_expenditure",
    "capital expenditure": "capital_expenditure",
    "capital": "capital_expenditure",
    "capex": "capital_expenditure",
    "utilities invoices classified as operational expenditure": "operational_expenditure",
    "utilities": "operational_expenditure",
    "expenditure": "expenditure",
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", " ", text.lower())).strip()


def _role_key(text: str) -> str:
    """'the Procurement Managers' -> 'procurement manager'."""
    t = normalise(text)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    words = t.split()
    if words and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _name_parts(name: str) -> tuple[list[str], str | None]:
    tokens = [t for t in normalise(name).replace(".", " ").split() if t]
    if not tokens:
        return [], None
    return tokens[:-1], tokens[-1]


class EntityResolver:
    def __init__(
        self,
        register: list[Entity],
        match_threshold: float = 0.9,
        possible_threshold: float = 0.5,
    ) -> None:
        self.register = register
        self.match_threshold = match_threshold
        self.possible_threshold = possible_threshold

    def _score_person(self, mention: str, entity: Entity) -> tuple[float, str]:
        names = [entity.canonical_name, *entity.aliases]
        m_norm = normalise(mention).replace(".", "")
        best = 0.0
        reason = "no overlap"
        for name in names:
            n_norm = normalise(name).replace(".", "")
            if m_norm == n_norm:
                return 1.0, f"exact alias '{name}'"
            m_first, m_last = _name_parts(mention)
            n_first, n_last = _name_parts(name)
            if m_last and n_last and m_last == n_last:
                # same surname: an initial that matches the first name is evidence, not proof
                if m_first and n_first:
                    mi, ni = m_first[0], n_first[0]
                    if mi == ni:
                        best = max(best, 1.0)
                        reason = f"full name match '{name}'"
                    elif len(mi) == 1 and ni.startswith(mi):
                        best = max(best, 0.7)
                        reason = f"initial '{mi}.' compatible with '{name}', identity not established"
                    elif len(ni) == 1 and mi.startswith(ni):
                        best = max(best, 0.7)
                        reason = f"initial compatible with '{name}', identity not established"
                    else:
                        best = max(best, 0.4)
                        reason = f"surname only shared with '{name}'"
                elif not m_first and n_first:
                    # a bare surname or a bare first name
                    best = max(best, 0.6)
                    reason = f"single token shared with '{name}'"
            elif m_norm and n_first and m_norm == n_first[0]:
                best = max(best, 0.85 if len(names) else 0.6)
                reason = f"first name only, matches '{name}'"
            else:
                ratio = SequenceMatcher(None, m_norm, n_norm).ratio()
                if ratio > best:
                    best = ratio * 0.5
                    reason = f"string similarity {ratio:.2f} to '{name}'"
        return best, reason

    def resolve(self, mention: str, kind: EntityKind) -> Resolution:
        if kind is EntityKind.ROLE:
            key = _role_key(mention)
            # the register first: canonical names and aliases of the roles this document defines
            for entity in self.register:
                if entity.kind is EntityKind.ROLE and key in {_role_key(n) for n in (entity.canonical_name, *entity.aliases)}:
                    return Resolution(mention=mention, verdict=MatchVerdict.MATCH, entity_id=entity.id, score=1.0, reason=f"role register '{entity.canonical_name}'")
            # then the shared vocabulary (abbreviations, plurals)
            norm_key = ROLE_NORMALISATION.get(key)
            for entity in self.register:
                if norm_key and entity.kind is EntityKind.ROLE and _role_key(entity.canonical_name) == norm_key:
                    return Resolution(mention=mention, verdict=MatchVerdict.MATCH, entity_id=entity.id, score=1.0, reason="role vocabulary")
            return Resolution(mention=mention, verdict=MatchVerdict.NO_MATCH, entity_id=None, score=0.0, reason="unknown role")
        if kind is EntityKind.CATEGORY:
            m = normalise(mention)
            for entity in self.register:
                if entity.kind is EntityKind.CATEGORY and m in {normalise(n) for n in (entity.canonical_name, *entity.aliases)}:
                    return Resolution(mention=mention, verdict=MatchVerdict.MATCH, entity_id=entity.id, score=1.0, reason=f"category register '{entity.canonical_name}'")
            key = CATEGORY_NORMALISATION.get(m)
            if key is None:
                for phrase, cat in CATEGORY_NORMALISATION.items():
                    if phrase in m:
                        key = cat
                        break
            if key is None:
                for entity in self.register:
                    if entity.kind is EntityKind.CATEGORY and any(normalise(n) in m for n in (entity.canonical_name, *entity.aliases)):
                        key = entity.id
                        break
            if key:
                return Resolution(mention=mention, verdict=MatchVerdict.MATCH, entity_id=key, score=1.0, reason="category vocabulary")
            return Resolution(mention=mention, verdict=MatchVerdict.NO_MATCH, entity_id=None, score=0.0, reason="unknown category")
        # persons: the probabilistic-looking part, with deterministic thresholds
        scored: list[tuple[float, str, Entity]] = []
        for entity in self.register:
            if entity.kind is not EntityKind.PERSON:
                continue
            score, reason = self._score_person(mention, entity)
            scored.append((score, reason, entity))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            return Resolution(mention=mention, verdict=MatchVerdict.NO_MATCH, entity_id=None, score=0.0, reason="empty register")
        score, reason, entity = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        candidates = [e.id for sc, _, e in scored if sc >= self.possible_threshold]
        if score >= self.match_threshold and score - runner_up >= 0.2:
            return Resolution(mention=mention, verdict=MatchVerdict.MATCH, entity_id=entity.id, score=score, reason=reason, candidates=[entity.id])
        if score >= self.possible_threshold:
            if len(candidates) > 1 and score - runner_up < 0.2:
                reason = f"{reason}; also compatible with {', '.join(c for c in candidates if c != entity.id)}"
            return Resolution(mention=mention, verdict=MatchVerdict.POSSIBLE_MATCH, entity_id=entity.id, score=score, reason=reason, candidates=candidates)
        return Resolution(mention=mention, verdict=MatchVerdict.NO_MATCH, entity_id=None, score=score, reason=reason)
