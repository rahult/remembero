"""Synthetic delegation-policy spans with exact claim labels, for teaching a small writer the
six-predicate vocabulary.

Every span is produced from a template that knows the claim it encodes, so the label is exact
and free. Surface variety comes from many hand-written forms per predicate and, optionally, a
paraphrase model whose output is kept only when the grounding check still finds the amounts,
dates and names in it. Names are disjoint from the evaluation fixture; the roles and categories
are the policy vocabulary the resolver knows.

  PYTHONPATH=src python -m remembro.training.synth --out data/claims --docs 120 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from remembro.claims.grounding import ground
from remembro.claims.models import CandidateClaim
from remembro.extraction.extractors import EXTRACTION_PROMPT

FIRST = ["Priya", "Tomas", "Mei", "Jonas", "Amara", "Luca", "Sofia", "Kenji", "Ingrid", "Rafael", "Zanele", "Henrik", "Noor", "Mateo", "Aisha", "Declan", "Yuki", "Farid", "Elena", "Kwame", "Sigrid", "Arjun", "Beatriz", "Olamide", "Hana", "Piotr", "Leila", "Bjorn", "Rosa", "Tariq"]
LAST = ["Okafor", "Lindqvist", "Nakamura", "Petrov", "Haddad", "Moreau", "Adeyemi", "Kowalski", "Fernandes", "Iyer", "Brennan", "Sato", "Novak", "Mbeki", "Larsen", "Costa", "Rahman", "Vogel", "Duarte", "Osei", "Halvorsen", "Mehta", "Rossi", "Abiodun", "Tanaka", "Zielinski", "Farahani", "Eriksen", "Delgado", "Mansour"]
ROLES = [
    ("Chief Financial Officer", "CFO"),
    ("Finance Director", "FD"),
    ("Procurement Manager", "PM"),
    ("Accounts Payable Lead", "APL"),
    ("Chief Operating Officer", "COO"),
    ("Financial Controller", "FC"),
    ("Head of Operations", None),
    ("Treasurer", None),
    ("General Counsel", None),
    # multi-word roles the writer must copy whole, not truncate to their last word
    ("Programs Manager", None),
    ("Regional Sales Director", None),
    ("Payroll Team Lead", None),
    ("Head of People and Culture", None),
    ("Research Funding Officer", None),
    ("Deputy Chief Executive", None),
    ("Property Portfolio Manager", None),
    ("Community Programs Lead", None),
    ("Information Technology Manager", None),
    ("Supply Chain Director", None),
    ("Clinical Services Manager", None),
    ("Marketing and Communications Manager", None),
    ("Fleet Coordinator", None),
    ("Learning and Development Lead", None),
]
CATEGORIES = [("operational expenditure", "operational_expenditure"), ("capital expenditure", "capital_expenditure"), ("expenditure", "expenditure")]
CURRENCIES = ["AUD", "USD", "GBP", "EUR", "NZD", "SGD"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def fmt_amount(n: int) -> str:
    return f"{n:,}"


def fmt_date(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def iso(d: date) -> str:
    return d.isoformat()


def claim(subject, skind, predicate, obj=None, okind="none", modality="assertion", polarity="positive", constraints=None, vf=None, vu=None):
    return {"subject": subject, "subject_kind": skind, "predicate": predicate, "object": obj, "object_kind": okind, "modality": modality, "polarity": polarity, "constraints": constraints or {}, "valid_from": vf, "valid_until": vu, "confidence": 1.0}


class Gen:
    def __init__(self, rng: random.Random) -> None:
        self.r = rng
        self.currency = rng.choice(CURRENCIES)
        names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 6)
        roles = rng.sample(ROLES, 3) + [(r, None) for r in self._composed_roles(rng, 3)]
        self.people = list(zip(names, roles))  # (name, (role, abbrev))
        self.start = date(2025 + rng.randint(0, 2), rng.randint(1, 12), rng.randint(1, 28))

    @staticmethod
    def _composed_roles(rng: random.Random, n: int) -> list[str]:
        mods = ["Grants", "Programs", "Fleet", "Facilities", "Payroll", "Community", "Regional", "Digital", "Clinical", "Property", "Research", "Events", "Membership", "Volunteer", "Housing", "Capital Works", "Student Services", "Wellbeing", "Compliance", "Sustainability", "Partnerships", "Fundraising"]
        heads = ["Manager", "Coordinator", "Officer", "Lead", "Director", "Administrator", "Supervisor", "Adviser"]
        out: set[str] = set()
        # the evaluation fixtures' roles stay out of training, so the generality check stays honest
        excluded = {"Grants Manager", "Facilities Coordinator", "Grants Officer"}
        while len(out) < n:
            role = f"{rng.choice(mods)} {rng.choice(heads)}"
            if role not in excluded:
                out.add(role)
        return sorted(out)

    def amount(self, lo=5, hi=500) -> int:
        return self.r.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250, 300, 400, 500]) * 1000

    def day(self, base: date | None = None, spread=400) -> date:
        return (base or self.start) + timedelta(days=self.r.randint(0, spread))

    # ---- one span per method: returns (text, [labels]) --------------------------------------
    def role_row(self):
        name, (role, _) = self.r.choice(self.people)
        d = self.day()
        return f"{role} | {name} | {fmt_date(d)}", [claim(name, "person", "holds_role", role, "role", vf=iso(d))]

    def role_sentence(self):
        name, (role, abbr) = self.r.choice(self.people)
        d = self.day()
        forms = [
            f"{name} was appointed {role} with effect from {fmt_date(d)}.",
            f"With effect from {fmt_date(d)}, {name} holds the position of {role}.",
            f"{name} has served as {role} since {fmt_date(d)}.",
            f"The role of {role} is held by {name}, appointed {fmt_date(d)}.",
            f"{name} ({role}) took up the role on {fmt_date(d)}.",
        ]
        return self.r.choice(forms), [claim(name, "person", "holds_role", role, "role", vf=iso(d))]

    def limit(self):
        _, (role, _) = self.r.choice(self.people)
        cat_text, cat = self.r.choice(CATEGORIES[:2])
        amt = self.amount()
        cur = self.currency
        forms = [
            f"The {role} may approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction.",
            f"The {role} is authorised to approve {cat_text} not exceeding {cur} {fmt_amount(amt)} for any single transaction.",
            f"{cat_text.capitalize()} of up to {cur} {fmt_amount(amt)} per transaction may be approved by the {role}.",
            f"Approval authority for {cat_text} is delegated to the {role} to a limit of {cur} {fmt_amount(amt)}.",
            f"The {role}'s approval limit for {cat_text} is {cur} {fmt_amount(amt)} per transaction.",
            f"A {role} may commit the company to {cat_text} of no more than {cur} {fmt_amount(amt)} in a single transaction.",
            f"The {role} is authorised to approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction.",
            f"The {role} is authorised to approve {cat_text} up to {cur} {fmt_amount(amt)} per grant.",
        ]
        return self.r.choice(forms), [claim(role, "role", "may_approve", cat_text, "category", "permission", "positive", {"maximum_amount": amt, "currency": cur, "category": cat})]

    def limit_table_row(self):
        _, (role, _) = self.r.choice(self.people)
        a, b = self.amount(), self.amount()
        cur = self.currency
        labels = [
            claim(role, "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": a, "currency": cur, "category": "operational_expenditure"}),
            claim(role, "role", "may_approve", "capital expenditure", "category", "permission", "positive", {"maximum_amount": b, "currency": cur, "category": "capital_expenditure"}),
        ]
        if self.r.random() < 0.3:
            labels[1] = claim(role, "role", "may_approve", "capital expenditure", "category", "prohibition", "negative", {"category": "capital_expenditure"})
            return f"{role} | {cur} {fmt_amount(a)} | Not permitted", labels
        return f"{role} | {cur} {fmt_amount(a)} | {cur} {fmt_amount(b)}", labels

    def prohibition(self):
        _, (role, _) = self.r.choice(self.people)
        cat_text, cat = self.r.choice(CATEGORIES[:2])
        forms = [
            f"The {role} may not approve {cat_text} of any amount.",
            f"The {role} must not approve {cat_text}.",
            f"No {cat_text} may be approved by the {role}.",
            f"The {role} has no authority to approve {cat_text}; such requests must be referred upward.",
            f"{cat_text.capitalize()} is outside the authority of the {role} and may not be approved by that role.",
            f"The {role} shall not approve {cat_text}.",
        ]
        return self.r.choice(forms), [claim(role, "role", "may_approve", cat_text, "category", "prohibition", "negative", {"category": cat})]

    def conditional_limit(self):
        _, (role, _) = self.r.choice(self.people)
        cat_text, cat = CATEGORIES[0]
        amt = self.amount()
        cur = self.currency
        cond = self.r.choice(["the expenditure is within the approved annual budget", "a purchase order has been raised in advance", "the supplier is on the approved supplier list", "the expenditure relates to a project with an approved business case"])
        forms = [
            f"The {role} may approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction provided that {cond}.",
            f"Provided that {cond}, the {role} may approve {cat_text} not exceeding {cur} {fmt_amount(amt)}.",
            f"Where {cond}, {cat_text} up to {cur} {fmt_amount(amt)} may be approved by the {role}.",
        ]
        return self.r.choice(forms), [claim(role, "role", "may_approve", cat_text, "category", "permission", "positive", {"maximum_amount": amt, "currency": cur, "category": cat, "condition": cond})]

    def delegation(self):
        (a, (arole, _)), (b, _) = self.r.sample(self.people, 2)
        cat_text, cat = self.r.choice(CATEGORIES[:2])
        amt = self.amount()
        cur = self.currency
        s = self.day(spread=500)
        e = s + timedelta(days=self.r.randint(5, 45))
        forms = [
            f"During {a}'s period of leave from {fmt_date(s)} to {fmt_date(e)}, {a} delegates to {b} authority to approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction.",
            f"{a} delegates to {b}, for the period {fmt_date(s)} to {fmt_date(e)}, the authority to approve {cat_text} up to {cur} {fmt_amount(amt)}.",
            f"From {fmt_date(s)} until {fmt_date(e)}, {b} holds by delegation from {a} the authority to approve {cat_text} to a limit of {cur} {fmt_amount(amt)}.",
            f"By written instrument, {a} ({arole}) delegates to {b} approval of {cat_text} up to {cur} {fmt_amount(amt)} per transaction, effective {fmt_date(s)} and ending {fmt_date(e)}.",
        ]
        return self.r.choice(forms), [claim(a, "person", "delegates", b, "person", "assertion", "positive", {"maximum_amount": amt, "currency": cur, "category": cat}, iso(s), iso(e))]

    def personal_grant(self):
        name, _ = self.r.choice(self.people)
        cat_text, cat = CATEGORIES[0]
        amt = self.amount()
        cur = self.currency
        s = self.day(spread=500)
        e = s + timedelta(days=self.r.randint(5, 30))
        forms = [
            f"{name} may approve invoices for {cat_text} up to {cur} {fmt_amount(amt)} per transaction between {fmt_date(s)} and {fmt_date(e)}.",
            f"For the period {fmt_date(s)} to {fmt_date(e)}, {name}'s limit for {cat_text} is raised to {cur} {fmt_amount(amt)} per transaction.",
            f"A temporary extension applies to {name}: {cat_text} up to {cur} {fmt_amount(amt)} may be approved from {fmt_date(s)} to {fmt_date(e)}.",
        ]
        return self.r.choice(forms), [claim(name, "person", "may_approve", cat_text, "category", "permission", "positive", {"maximum_amount": amt, "currency": cur, "category": cat}, iso(s), iso(e))]

    def suspension(self):
        name, _ = self.r.choice(self.people)
        s = self.day(spread=500)
        reason = self.r.choice(["pending review of supplier onboarding procedures", "pending the outcome of an internal audit", "while a conflict-of-interest disclosure is reviewed", "pending completion of mandatory training"])
        forms = [
            f"{name}'s approval authority is suspended with effect from {fmt_date(s)} until further notice, {reason}.",
            f"With effect from {fmt_date(s)}, the approval authority of {name} is suspended {reason}.",
            f"{name} may not exercise any approval authority from {fmt_date(s)}, {reason}; the suspension remains in force until lifted in writing.",
        ]
        return self.r.choice(forms), [claim(name, "person", "suspended", None, "none", "assertion", "positive", {"condition": reason}, iso(s), None)]

    def suspension_with_end(self):
        name, _ = self.r.choice(self.people)
        s = self.day(spread=500)
        e = s + timedelta(days=self.r.randint(7, 60))
        text = f"{name}'s approval authority is suspended from {fmt_date(s)} to {fmt_date(e)} inclusive."
        return text, [claim(name, "person", "suspended", None, "none", "assertion", "positive", {}, iso(s), iso(e))]

    def revocation(self):
        (a, _), (b, _) = self.r.sample(self.people, 2)
        cat_text, cat = self.r.choice(CATEGORIES[:2])
        notice = self.day(spread=500)
        eff = notice + timedelta(days=self.r.randint(0, 5))
        forms = [
            f"By notice dated {fmt_date(notice)}, {a} revoked the delegation of {cat_text} authority to {b}, with effect from {fmt_date(eff)}.",
            f"{a} has revoked, effective {fmt_date(eff)}, the delegation to {b} of authority over {cat_text}.",
            f"The delegation from {a} to {b} in respect of {cat_text} is revoked with effect from {fmt_date(eff)}; this supersedes any end date previously stated.",
        ]
        return self.r.choice(forms), [claim(a, "person", "revokes_delegation", b, "person", "assertion", "positive", {"category": cat}, iso(eff), None)]

    def self_benefit(self):
        _, (role, _) = self.r.choice(self.people)
        forms = [
            (f"The {role} must not approve expenditure benefiting themselves or a related party.", role, "role"),
            (f"{role}s may not approve any expenditure from which they personally benefit.", role, "role"),
            (f"Approval by the {role} of expenditure in which the {role} has a personal interest is prohibited.", role, "role"),
        ]
        text, subj, kind = self.r.choice(forms)
        return text, [claim(subj, kind, "may_approve", "expenditure", "category", "prohibition", "negative", {"category": "expenditure", "condition": "benefiting themselves"})]

    def boilerplate(self):
        _, (role, _) = self.r.choice(self.people)
        name, _ = self.r.choice(self.people)
        days = self.r.choice(["two", "three", "five", "ten"])
        body = self.r.choice(["the Board", "the Audit Committee", "the Risk Committee", "the Executive Committee"])
        period = self.r.choice(["quarter", "month", "half-year"])
        cur = self.currency
        forms = [
            f"The {role} shall record every delegation in the delegation register within {days} business days.",
            f"Nothing in this policy creates an entitlement to any role, and {body} retains the right to vary any delegation at any time by resolution.",
            f"Questions about this policy should be directed to the office of the {role}.",
            "A delegate may not further delegate authority they hold by delegation.",
            "Transactions may not be split to bring them within an approval limit.",
            f"This policy is reviewed every {period} and republished on the intranet by the {role}.",
            f"Amounts in this policy are stated in {cur} and are exclusive of tax unless stated otherwise.",
            f"Internal Audit samples approved transactions each {period} and reports exceptions to {body}.",
            f"Requests to vary a limit must be submitted in writing to the {role} at least {days} business days in advance.",
            "Where a limit is stated per transaction, related invoices for the same purchase are treated as one transaction.",
            f"For queries relating to supplier payments contact {name}.",
            f"Every approval must be recorded in the finance system within {days} business days of the decision.",
            f"{body} approved this version of the policy at its meeting.",
            f"Definitions used in this policy are set out in the glossary maintained by the {role}.",
            f"The {role} notifies {body} of any breach of this policy within {days} business days.",
            f"Where this policy is silent, the matter must be referred to {body}.",
            f"A copy of each written notice under this section is retained by the {role} for seven years.",
        ]
        return self.r.choice(forms), []

    def suspension_lifted(self):
        name, _ = self.r.choice(self.people)
        s = self.day(spread=500)
        lift = s + timedelta(days=self.r.randint(7, 90))
        forms = [
            f"The suspension of {name}'s approval authority, which took effect on {fmt_date(s)}, was lifted with effect from {fmt_date(lift)}.",
            f"{name}'s approval authority, suspended from {fmt_date(s)}, is reinstated with effect from {fmt_date(lift)}.",
            f"With effect from {fmt_date(lift)}, {name} is no longer suspended and may exercise approval authority.",
            f"The suspension of {name} is lifted from {fmt_date(lift)}.",
            f"{name}'s suspension ended on {fmt_date(lift)}; approval authority resumed on that date.",
        ]
        # a lift is a negative suspension from the lift date: no date arithmetic for the writer
        return self.r.choice(forms), [claim(name, "person", "suspended", None, "none", "assertion", "negative", {}, iso(lift), None)]

    def role_ended(self):
        name, (role, _) = self.r.choice(self.people)
        e = self.day(spread=600)
        forms = [
            f"{name} ceased to hold the role of {role} on {fmt_date(e)}.",
            f"{name} stepped down as {role} with effect from the close of business on {fmt_date(e)}.",
            f"The appointment of {name} as {role} ended on {fmt_date(e)}.",
        ]
        return self.r.choice(forms), [claim(name, "person", "holds_role", role, "role", vu=iso(e))]

    def role_replaced(self):
        (a, (role, _)), (b, _) = self.r.sample(self.people, 2)
        e = self.day(spread=600)
        start = e + timedelta(days=1)
        forms = [
            f"{a} ceased to hold the role of {role} on {fmt_date(e)}. {b} was appointed {role} with effect from {fmt_date(start)}.",
            f"With effect from {fmt_date(start)}, {b} replaces {a} as {role}; {a}'s appointment ended on {fmt_date(e)}.",
        ]
        return self.r.choice(forms), [
            claim(a, "person", "holds_role", role, "role", vu=iso(e)),
            claim(b, "person", "holds_role", role, "role", vf=iso(start)),
        ]

    def amended_limit(self):
        _, (role, _) = self.r.choice(self.people)
        cat_text, cat = self.r.choice(CATEGORIES[:2])
        amt = self.amount()
        cur = self.currency
        forms = [
            f"The {role} is authorised to approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction. This replaces the limit previously stated.",
            f"Clause 3 is amended so that the {role} may approve {cat_text} up to {cur} {fmt_amount(amt)} per transaction.",
            f"The {role}'s limit for {cat_text} is varied to {cur} {fmt_amount(amt)} per transaction.",
        ]
        return self.r.choice(forms), [claim(role, "role", "may_approve", cat_text, "category", "permission", "positive", {"maximum_amount": amt, "currency": cur, "category": cat})]

    def span(self):
        kind = self.r.choices(
            [self.role_row, self.role_sentence, self.limit, self.limit_table_row, self.prohibition, self.conditional_limit, self.delegation, self.personal_grant, self.suspension, self.suspension_with_end, self.revocation, self.self_benefit, self.boilerplate, self.suspension_lifted, self.role_ended, self.role_replaced, self.amended_limit],
            weights=[6, 6, 14, 6, 8, 6, 12, 8, 6, 3, 9, 3, 18, 6, 4, 4, 5],
        )[0]
        return kind()


def _valid(text: str, labels: list[dict]) -> bool:
    """Every label must parse and be grounded in the text it came from."""
    for lab in labels:
        item = {**lab, "id": "x", "evidence": [{"id": "e", "document_id": "d", "page": 1, "paragraph": 1, "section": None, "start_offset": 0, "end_offset": len(text), "quoted_text": text, "authority": "body"}]}
        try:
            CandidateClaim.model_validate(item)
        except Exception:  # noqa: BLE001
            return False
        _, dropped = ground([item])
        if dropped:
            return False
    return True


def generate(docs: int, per_doc: int, seed: int) -> list[tuple[str, list[dict]]]:
    rng = random.Random(seed)
    out: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    for _ in range(docs):
        g = Gen(rng)
        for _ in range(per_doc):
            text, labels = g.span()
            if text in seen or not _valid(text, labels):
                continue
            seen.add(text)
            out.append((text, labels))
    return out


def to_rows(spans: list[tuple[str, list[dict]]]) -> list[dict]:
    rows = []
    for text, labels in spans:
        rows.append({"messages": [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": text}, {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)}]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--per-doc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heldout", type=float, default=0.06)
    args = ap.parse_args()
    spans = generate(args.docs, args.per_doc, args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(spans)
    cut = int(len(spans) * args.heldout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("heldout.jsonl", spans[:cut]), ("conversations.jsonl", spans[cut:])):
        with (out / name).open("w") as f:
            for row in to_rows(part):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    empties = sum(1 for _, l in spans if not l)
    print(f"{len(spans)} spans ({empties} empty-label), train {len(spans) - cut}, heldout {cut} -> {out}")


if __name__ == "__main__":
    main()
