"""Strict intermediate representations.

Everything the probabilistic layer produces is a CandidateClaim. Nothing below
the trust boundary accepts anything that does not parse into these models.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    PERMISSION = "permission"  # may, is authorised to
    OBLIGATION = "obligation"  # must, shall
    PROHIBITION = "prohibition"  # must not, may not, cannot
    ASSERTION = "assertion"  # is, holds, delegates, revokes


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Predicate(str, Enum):
    """The domain's small vocabulary. Anything else is rejected at validation."""

    HOLDS_ROLE = "holds_role"  # subject person, object role
    MAY_APPROVE = "may_approve"  # subject role|person, object expenditure category
    DELEGATES = "delegates"  # subject person, object person, constraints.category
    SUSPENDED = "suspended"  # subject person
    REVOKES_DELEGATION = "revokes_delegation"  # subject person (delegator), object person
    APPENDIX_LIMIT = "appendix_limit"  # subject role, object category (summary table)


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maximum_amount: float | None = None
    currency: str | None = None
    category: str | None = None  # operational_expenditure | capital_expenditure | utilities
    condition: str | None = None  # free text condition, kept for explanation


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    document_id: str
    page: int
    paragraph: int
    section: str | None = None
    start_offset: int
    end_offset: int
    quoted_text: str
    authority: Literal["body", "appendix", "table"] = "body"


class CandidateClaim(BaseModel):
    """What a source says, as the extractor read it. Not yet a fact."""

    model_config = ConfigDict(extra="forbid")
    id: str
    subject: str  # surface text: "Alice Morgan", "Chief Financial Officer", "D. Smith"
    subject_kind: Literal["person", "role"]
    predicate: Predicate
    object: str | None = None  # surface text
    object_kind: Literal["person", "role", "category", "none"] = "none"
    modality: Modality = Modality.ASSERTION
    polarity: Polarity = Polarity.POSITIVE
    constraints: Constraints = Field(default_factory=Constraints)
    valid_from: date | None = None
    valid_until: date | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: list[Evidence] = Field(min_length=1)


class ClaimStatus(str, Enum):
    RAW = "RAW"
    EXTRACTED = "EXTRACTED"
    VALIDATED = "VALIDATED"
    RESOLVED = "RESOLVED"
    ACCEPTED = "ACCEPTED"
    SUPERSEDED = "SUPERSEDED"
    REVOKED = "REVOKED"
    CONFLICTED = "CONFLICTED"
    REJECTED = "REJECTED"


class Claim(CandidateClaim):
    """A candidate that has crossed into the deterministic side. Carries resolved ids."""

    status: ClaimStatus = ClaimStatus.EXTRACTED
    subject_entity: str | None = None  # entity id once resolved
    object_entity: str | None = None
    possible_subjects: list[str] = Field(default_factory=list)  # POSSIBLE_MATCH candidates, never merged
    possible_objects: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
    superseded_by: str | None = None


class EntityKind(str, Enum):
    PERSON = "person"
    ROLE = "role"
    CATEGORY = "category"


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: EntityKind
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class MatchVerdict(str, Enum):
    MATCH = "MATCH"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    NO_MATCH = "NO_MATCH"


class Resolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mention: str
    verdict: MatchVerdict
    entity_id: str | None = None
    score: float
    reason: str
    candidates: list[str] = Field(default_factory=list)  # every entity within the possible threshold


class ContradictionType(str, Enum):
    LIMIT_MISMATCH = "LIMIT_MISMATCH"
    POLARITY_MISMATCH = "POLARITY_MISMATCH"


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: ContradictionType
    claims: list[str]
    resolved: bool = False
    resolution: str | None = None  # e.g. "body prevails over appendix"
    winner: str | None = None


class BeliefStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    DISPUTED = "DISPUTED"
    UNKNOWN = "UNKNOWN"
    SUPERSEDED = "SUPERSEDED"
    REVOKED = "REVOKED"


class Proposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str  # entity id
    predicate: Predicate
    object: str | None = None  # entity id or category
    modality: Modality = Modality.ASSERTION
    polarity: Polarity = Polarity.POSITIVE
    constraints: Constraints = Field(default_factory=Constraints)


class Belief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    proposition: Proposition
    status: BeliefStatus
    supporting_claims: list[str]
    contradicting_claims: list[str] = Field(default_factory=list)
    valid_from: date | None = None
    valid_until: date | None = None


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: str  # surface name; resolved by the engine
    action: Literal["approve"]
    resource: str  # category
    amount: float
    currency: str
    on: date
    self_benefit: bool = False
    own_expense_claim: bool = False


class DecisionOutcome(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    UNKNOWN = "UNKNOWN"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: DecisionOutcome
    request: Request
    reasons: list[str]
    beliefs_used: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    unknown_reason: str | None = None


class StateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: str
    subject_id: str
    related_id: str | None = None
    reason: str
    sequence: int
