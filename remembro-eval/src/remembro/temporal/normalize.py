"""Temporal normalisation: dates in prose to ISO days, and interval algebra.

V0 stores valid time only. Every claim carries `valid_from`/`valid_until` so a
transaction-time column can be added without touching the claim shape.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}

DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def parse_date(text: str) -> date | None:
    text = text.strip()
    iso = ISO_RE.search(text)
    if iso:
        return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    match = DATE_RE.search(text)
    if match:
        month = MONTHS.get(match.group(2).lower())
        if month is None:
            return None
        return date(int(match.group(3)), month, int(match.group(1)))
    return None


def dates_in(text: str) -> list[date]:
    out: list[date] = []
    for m in DATE_RE.finditer(text):
        month = MONTHS.get(m.group(2).lower())
        if month:
            out.append(date(int(m.group(3)), month, int(m.group(1))))
    for m in ISO_RE.finditer(text):
        out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def in_interval(day: date, start: date | None, end: date | None) -> bool:
    """Closed on both ends; None is open."""
    if start is not None and day < start:
        return False
    if end is not None and day > end:
        return False
    return True


def overlaps(a_start: date | None, a_end: date | None, b_start: date | None, b_end: date | None) -> bool:
    lo = max([d for d in (a_start, b_start) if d is not None], default=None)
    hi = min([d for d in (a_end, b_end) if d is not None], default=None)
    if lo is None or hi is None:
        return True
    return lo <= hi
