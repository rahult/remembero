"""Grounding: a candidate that parses is not yet a claim the engine may act on.

Schema validation catches malformed output. It does not catch a well-formed claim whose
content is not in its evidence: an amount the passage never states, a date that is not there,
or a permission with no ceiling read out of a sentence that actually withheld it. Each check
here is a string comparison between a field and the claim's own quoted evidence, so it needs
no model and no gold. A claim that fails is rejected with the reason, and never enters state.
"""

from __future__ import annotations

import re
from datetime import timedelta

from remembro.temporal.normalize import dates_in, parse_date

NO_CEILING = re.compile(r"\b(any amount|no limit|without limit|unlimited|of any value|regardless of (the )?amount)\b", re.IGNORECASE)


def _amount_in(text: str, amount: float) -> bool:
    digits = str(int(amount)) if float(amount).is_integer() else str(amount)
    return digits in text.replace(",", "").replace(" ", "")


def _date_in(text: str, value: str, exclusive_end: bool = False) -> bool:
    wanted = parse_date(value)
    if wanted is None:
        return False
    found = dates_in(text)
    if wanted in found:
        return True
    # "suspended until 1 September" / "lifted with effect from 1 September": the last day of
    # validity is the day before a date the text does state
    return exclusive_end and (wanted + timedelta(days=1)) in found


def ground(candidates: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    for c in candidates:
        evidence = c.get("evidence") or []
        text = " ".join(e.get("quoted_text", "") for e in evidence if isinstance(e, dict))
        constraints = c.get("constraints") or {}
        reason: str | None = None
        amount = constraints.get("maximum_amount")
        if amount is not None and not _amount_in(text, amount):
            reason = f"ungrounded amount: {amount} does not appear in the evidence"
        for field in ("valid_from", "valid_until"):
            value = c.get(field)
            if reason is None and value and not _date_in(text, value, exclusive_end=(field == "valid_until")):
                reason = f"ungrounded date: {field}={value} does not appear in the evidence"
        if (
            reason is None
            and c.get("predicate") == "may_approve"
            and c.get("polarity", "positive") == "positive"
            and c.get("modality", "permission") in ("permission", "assertion")
            and amount is None
            and not NO_CEILING.search(text)
        ):
            reason = "ungrounded ceiling: a positive permission with no maximum_amount, and the evidence does not say the authority is unlimited"
        if reason:
            dropped.append((c, reason))
        else:
            kept.append(c)
    return kept, dropped
