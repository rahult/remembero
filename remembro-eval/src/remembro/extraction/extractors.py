"""The probabilistic boundary. Every extractor returns raw candidate dicts; the
pipeline validates them. Three backends:

- SnapshotExtractor: reads fixtures/runs/<name>.claims.json, so any model's
  output can be replayed against the same deterministic downstream.
- RuleExtractor: regular expressions over spans, zero model calls, the
  deterministic baseline.
- LlmExtractor: any OpenAI-compatible chat endpoint (the fine-tuned Rembero
  writer on Modal, a local GGUF, OpenRouter models), one call per span, JSON out.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Protocol

from remembro.document.parser import ParsedDocument, Span
from remembro.temporal.normalize import dates_in


class ClaimExtractor(Protocol):
    name: str

    def extract(self, document: ParsedDocument) -> list[dict]: ...


class SnapshotExtractor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.stem.replace(".claims", "")

    def extract(self, document: ParsedDocument) -> list[dict]:
        data = json.loads(self.path.read_text())
        return data["claims"] if isinstance(data, dict) else data


def _evidence(span: Span, document: ParsedDocument, eid: str, quote: str | None = None) -> dict:
    text = quote or span.text
    off = document.text.find(text)
    return {
        "id": eid,
        "document_id": document.id,
        "page": span.page,
        "paragraph": span.paragraph,
        "section": span.section,
        "start_offset": off if off >= 0 else span.start_offset,
        "end_offset": (off + len(text)) if off >= 0 else span.end_offset,
        "quoted_text": text,
        "authority": span.authority,
    }


AMOUNT = re.compile(r"AUD\s*([\d,]+)")
ROLE = r"(Chief Financial Officer|Finance Director|Procurement Managers?|Accounts Payable Lead|CFO)"
PERSON = r"([A-Z]\.\s?[A-Z][a-z]+|[A-Z][a-z]+\s[A-Z][a-z]+|[A-Z][a-z]+)"
CATEGORY = r"(operational expenditure|capital expenditure|invoices for operational expenditure|utilities invoices classified as operational expenditure|expenditure)"


class RuleExtractor:
    """Sentence patterns for the delegation domain. Coverage is rule-bounded by design."""

    name = "rules"

    def extract(self, document: ParsedDocument) -> list[dict]:
        out: list[dict] = []
        n = 0

        def emit(span: Span, **fields) -> None:
            nonlocal n
            n += 1
            cid = f"rule_{n}"
            out.append({"id": cid, "confidence": 1.0, "evidence": [_evidence(span, document, f"ev_{cid}", fields.pop("quote", None))], **fields})

        for span in document.spans:
            text = span.text
            sentences = re.split(r"(?<!\b[A-Z])(?<=\.)\s+(?=[A-Z])", text) if span.kind != "table_row" else [text]
            for s in sentences:
                dates = dates_in(s)
                # role table rows: "Role | Holder | Since"
                m = re.match(rf"^{ROLE}\s*\|\s*{PERSON}\s*\|\s*(.*)$", s)
                if m and span.kind == "table_row":
                    emit(span, subject=m.group(2), subject_kind="person", predicate="holds_role", object=m.group(1), object_kind="role", valid_from=(dates[0].isoformat() if dates else None), quote=s)
                    continue
                # appendix limit rows: "Role | AUD x | AUD y / Not permitted"
                m = re.match(rf"^{ROLE}\s*\|\s*(AUD\s*[\d,]+|Not permitted)\s*\|\s*(AUD\s*[\d,]+|Not permitted)$", s)
                if m and span.kind == "table_row":
                    for category, cell in (("operational_expenditure", m.group(2)), ("capital_expenditure", m.group(3))):
                        amt = AMOUNT.search(cell)
                        if amt:
                            emit(span, subject=m.group(1), subject_kind="role", predicate="may_approve", object=category.replace("_", " "), object_kind="category", modality="permission", polarity="positive", constraints={"maximum_amount": float(amt.group(1).replace(",", "")), "currency": "AUD", "category": category}, quote=s)
                        else:
                            emit(span, subject=m.group(1), subject_kind="role", predicate="may_approve", object=category.replace("_", " "), object_kind="category", modality="prohibition", polarity="negative", constraints={"category": category}, quote=s)
                    continue
                # "The <Role> may approve <category> up to AUD N"
                m = re.search(rf"The {ROLE} may approve {CATEGORY} up to AUD ([\d,]+)", s)
                if m:
                    cat = "capital_expenditure" if "capital" in m.group(2) else "operational_expenditure"
                    cond = None
                    pm = re.search(r"provided (.*?)\.?$", s)
                    if pm:
                        cond = pm.group(1)
                    emit(span, subject=m.group(1), subject_kind="role", predicate="may_approve", object=m.group(2), object_kind="category", modality="permission", polarity="positive", constraints={"maximum_amount": float(m.group(3).replace(",", "")), "currency": "AUD", "category": cat, **({"condition": cond} if cond else {})}, quote=s)
                    continue
                # "The <Role> may not approve <category>"
                m = re.search(rf"The {ROLE} may not approve {CATEGORY}", s)
                if m:
                    emit(span, subject=m.group(1), subject_kind="role", predicate="may_approve", object=m.group(2), object_kind="category", modality="prohibition", polarity="negative", constraints={"category": "capital_expenditure" if "capital" in m.group(2) else "operational_expenditure"}, quote=s)
                    continue
                # "Procurement Managers must not approve expenditure benefiting themselves"
                m = re.search(rf"{ROLE} must not approve expenditure benefiting themselves", s)
                if m:
                    emit(span, subject=m.group(1), subject_kind="role", predicate="may_approve", object="expenditure", object_kind="category", modality="prohibition", polarity="negative", constraints={"category": "expenditure", "condition": "benefiting themselves"}, quote=s)
                    continue
                # "<Person> delegates to <Person> ... up to AUD N" (two-sentence paragraph handled at span level below)
                m = re.search(rf"{PERSON} delegates to {PERSON} her authority to approve {CATEGORY}", s)
                if m:
                    amt = AMOUNT.search(text)
                    span_dates = dates_in(text)
                    emit(span, subject=m.group(1), subject_kind="person", predicate="delegates", object=m.group(2), object_kind="person", constraints={"category": "operational_expenditure" if "operational" in m.group(3) else "capital_expenditure", **({"maximum_amount": float(amt.group(1).replace(",", "")), "currency": "AUD"} if amt else {})}, valid_from=span_dates[0].isoformat() if len(span_dates) > 1 else None, valid_until=span_dates[1].isoformat() if len(span_dates) > 1 else None, quote=text)
                    continue
                # "<Person> may approve <category> up to AUD N ... between/from D to D"
                m = re.search(rf"^{PERSON} may approve {CATEGORY} up to AUD ([\d,]+)", s)
                if m:
                    emit(span, subject=m.group(1), subject_kind="person", predicate="may_approve", object=m.group(2), object_kind="category", modality="permission", polarity="positive", constraints={"maximum_amount": float(m.group(3).replace(",", "")), "currency": "AUD", "category": "operational_expenditure"}, valid_from=dates[0].isoformat() if len(dates) > 1 else None, valid_until=dates[1].isoformat() if len(dates) > 1 else None, quote=s)
                    continue
                # suspension
                m = re.search(rf"{PERSON}'s approval authority is suspended with effect from (\d{{1,2}} [A-Za-z]+ \d{{4}})", s)
                if m:
                    emit(span, subject=m.group(1), subject_kind="person", predicate="suspended", valid_from=dates[0].isoformat() if dates else None, quote=s)
                    continue
                # revocation
                m = re.search(rf"{PERSON} revoked the delegation of {CATEGORY} authority to {PERSON} .*? with effect from (\d{{1,2}} [A-Za-z]+ \d{{4}})", s)
                if m:
                    eff = dates_in(m.group(4))
                    emit(span, subject=m.group(1), subject_kind="person", predicate="revokes_delegation", object=m.group(3), object_kind="person", constraints={"category": "operational_expenditure" if "operational" in m.group(2) else "capital_expenditure"}, valid_from=eff[0].isoformat() if eff else None, quote=s)
                    continue
                # "Capital expenditure approved by the Finance Director is capped at AUD N"
                m = re.search(rf"{CATEGORY} approved by the {ROLE} is capped at AUD ([\d,]+)", s, re.IGNORECASE)
                if m:
                    emit(span, subject=m.group(2), subject_kind="role", predicate="may_approve", object=m.group(1).lower(), object_kind="category", modality="permission", polarity="positive", constraints={"maximum_amount": float(m.group(3).replace(",", "")), "currency": "AUD", "category": "capital_expenditure" if "capital" in m.group(1).lower() else "operational_expenditure"}, quote=s)
                    continue
        return out


EXTRACTION_PROMPT = """You extract structured claims from one passage of a corporate delegation-of-authority policy. Return a JSON array (possibly empty). Each element:
{"subject": string, "subject_kind": "person"|"role", "predicate": "holds_role"|"may_approve"|"delegates"|"suspended"|"revokes_delegation", "object": string|null, "object_kind": "person"|"role"|"category"|"none", "modality": "permission"|"obligation"|"prohibition"|"assertion", "polarity": "positive"|"negative", "constraints": {"maximum_amount": number|null, "currency": string|null, "category": "operational_expenditure"|"capital_expenditure"|"expenditure"|null, "condition": string|null}, "valid_from": "YYYY-MM-DD"|null, "valid_until": "YYYY-MM-DD"|null, "confidence": number 0..1}
Rules: quote nothing you cannot see in the passage; "may not" and "must not" are prohibition with polarity negative; a delegation's object is the delegate and its constraints carry the delegated category and limit; a revocation's valid_from is the date it takes effect; a suspension's valid_from is the effective date; dates as ISO. Table rows arrive as "cell | cell | cell". Return only the JSON array."""


class LlmExtractor:
    def __init__(self, base_url: str, model: str, api_key: str, name: str | None = None, timeout: int = 180, max_tokens: int = 6000) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = name or model.replace("/", "-")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _complete(self, passage: str) -> str:
        # one retry at double the budget: a reasoning model that loops on a short table row
        # usually settles the second time; if not, the span is recorded as unread
        try:
            return self._complete_once(passage, self.max_tokens)
        except RuntimeError as error:
            if "truncated" not in str(error):
                raise
            return self._complete_once(passage, self.max_tokens * 2)

    def _complete_once(self, passage: str, max_tokens: int) -> str:
        # reasoning models spend most of the budget thinking before the JSON; a tight cap
        # truncates the answer and the span silently yields nothing, which is the one failure
        # a boundary must not hide
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": passage}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"reply truncated at {max_tokens} tokens")
        return choice["message"]["content"] or ""

    def extract(self, document: ParsedDocument) -> list[dict]:
        out: list[dict] = []
        n = 0
        for span in document.spans:
            if span.kind == "table_row" and "|" not in span.text:
                continue
            try:
                reply = self._complete(span.text)
            except Exception as error:  # noqa: BLE001 - the boundary must not crash the pipeline
                out.append({"id": f"{self.name}_error_{span.paragraph}", "error": str(error)[:200], "span_text": span.text})
                continue
            match = re.search(r"\[[\s\S]*\]", reply)
            if not match:
                out.append({"id": f"{self.name}_error_{span.paragraph}", "error": f"no JSON array in reply: {reply[:120]!r}", "span_text": span.text})
                continue
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError as error:
                out.append({"id": f"{self.name}_error_{span.paragraph}", "error": f"unparseable JSON: {error}", "span_text": span.text})
                continue
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                n += 1
                cid = f"{self.name}_{n}"
                item = {k: v for k, v in item.items() if k in {"subject", "subject_kind", "predicate", "object", "object_kind", "modality", "polarity", "constraints", "valid_from", "valid_until", "confidence"}}
                item.setdefault("constraints", {})
                if isinstance(item["constraints"], dict):
                    item["constraints"] = {k: v for k, v in item["constraints"].items() if v is not None and k in {"maximum_amount", "currency", "category", "condition"}}
                item["id"] = cid
                item["evidence"] = [_evidence(span, document, f"ev_{cid}")]
                out.append(item)
        return out


def extractor_from_args(kind: str, run_name: str | None = None, model: str | None = None, base_url: str | None = None) -> ClaimExtractor:
    if kind == "rules":
        return RuleExtractor()
    if kind == "snapshot":
        return SnapshotExtractor(Path("fixtures/runs") / f"{run_name}.claims.json")
    if kind == "llm":
        return LlmExtractor(base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"), model or os.environ["LLM_MODEL"], os.environ.get("EXTRACTION_API_KEY") or os.environ.get("LLM_API_KEY", ""), name=run_name)
    raise SystemExit(f"unknown extractor {kind}")
