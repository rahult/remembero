"""Temporal normalisation: dates in prose to ISO days, and interval algebra.

V0 stores valid time only. Every claim carries `valid_from`/`valid_until` so a
transaction-time column can be added without touching the claim shape.
"""

from __future__ import annotations

import re
from datetime import date

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
MONTHS = {m: i for i, m in enumerate(_MONTH_NAMES, start=1)}
MONTHS.update({m[:3]: i for i, m in enumerate(_MONTH_NAMES, start=1)})
MONTHS["sept"] = 9

DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
# "21 to 25 September 2026", "1 and 15 September 2026", "3–7 October 2026", "8 until 12 Nov 2026":
# the first day borrows the month and year of the second
RANGE_RE = re.compile(r"\b(\d{1,2})\s*(?:to|and|until|through|till|-|–|—)\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
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
    for m in RANGE_RE.finditer(text):
        month = MONTHS.get(m.group(3).lower())
        if month:
            try:
                out.append(date(int(m.group(4)), month, int(m.group(1))))
            except ValueError:
                pass
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
