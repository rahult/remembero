"""A persistent workspace for daily use: documents in, decisions with evidence out.

Layout under `root/`:
  register.json          entities (persons, roles, categories) with aliases
  documents/<id>.md      the document text as ingested
  documents/<id>.json    metadata: effective_date, kind, source, ingested_at, extractor
  candidates/<id>.json   the extractor's raw candidates for that document

Nothing below the trust boundary is cached: every question rebuilds the world state from the
stored candidates and the current register, so a register change (a new alias, a new
category) changes answers immediately and every past answer is reproducible.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

from remembro.beliefs.pipeline import build_state
from remembro.claims.models import Entity, EntityKind, MatchVerdict, Request
from remembro.decision.engine import evaluate as decide_request
from remembro.document.parser import MarkdownParser
from remembro.entities.resolver import EntityResolver
from remembro.extraction.extractors import LlmExtractor, RuleExtractor

DEFAULT_CATEGORIES = [
    {"id": "operational_expenditure", "kind": "category", "canonical_name": "operational expenditure", "aliases": ["operational", "opex", "operating expenditure", "invoices for operational expenditure", "utilities", "utilities invoices", "facilities maintenance", "maintenance"]},
    {"id": "capital_expenditure", "kind": "category", "canonical_name": "capital expenditure", "aliases": ["capital", "capex"]},
    {"id": "expenditure", "kind": "category", "canonical_name": "expenditure", "aliases": ["expense", "expenses", "spend", "payments"]},
    {"id": "role_board", "kind": "role", "canonical_name": "Board", "aliases": ["the Board", "Board of Directors", "Board of Trustees", "the Board of Trustees"]},
]

FULL_NAME = re.compile(r"^[A-Z][a-z]+(?:[-' ][A-Z][a-z]+)+$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class Workspace:
    def __init__(self, root: Path | str, extractor: str = "llm", model: str | None = None, base_url: str | None = None, api_key: str | None = None, match_threshold: float = 0.9) -> None:
        self.root = Path(root).expanduser()
        (self.root / "documents").mkdir(parents=True, exist_ok=True)
        (self.root / "candidates").mkdir(parents=True, exist_ok=True)
        self.extractor_kind = extractor
        self.model = model or os.environ.get("REMEMBRO_MODEL", "glm-5.3-flash:cloud")
        self.base_url = base_url or os.environ.get("REMEMBRO_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.environ.get("REMEMBRO_API_KEY") or os.environ.get("EXTRACTION_API_KEY", "ollama")
        self.match_threshold = match_threshold
        if not (self.root / "register.json").exists():
            self._save_register(DEFAULT_CATEGORIES)

    # ---- register --------------------------------------------------------------------------
    def register(self) -> list[dict]:
        return json.loads((self.root / "register.json").read_text())

    def _save_register(self, entities: list[dict]) -> None:
        (self.root / "register.json").write_text(json.dumps(entities, indent=2, ensure_ascii=False) + "\n")

    def entities(self) -> list[Entity]:
        return [Entity.model_validate(e) for e in self.register()]

    def resolver(self) -> EntityResolver:
        return EntityResolver(self.entities(), match_threshold=self.match_threshold)

    def add_entity(self, kind: str, name: str, aliases: list[str] | None = None, entity_id: str | None = None) -> dict:
        reg = self.register()
        eid = entity_id or f"{kind}_{_slug(name)}" if kind != "category" else (entity_id or _slug(name))
        for e in reg:
            if e["id"] == eid or (e["kind"] == kind and e["canonical_name"].lower() == name.lower()):
                for a in aliases or []:
                    if a not in e["aliases"]:
                        e["aliases"].append(a)
                self._save_register(reg)
                return e
        entity = {"id": eid, "kind": kind, "canonical_name": name, "aliases": list(aliases or [])}
        reg.append(entity)
        self._save_register(reg)
        return entity

    def remove_entity(self, entity_id: str) -> bool:
        reg = self.register()
        kept = [e for e in reg if e["id"] != entity_id]
        self._save_register(kept)
        return len(kept) < len(reg)

    def _auto_register(self, candidates: list[dict]) -> list[dict]:
        """People and roles a document itself asserts (holds_role) join the register; an initial
        or a single token never does, and categories are only proposed, never added."""
        added: list[dict] = []
        resolver = self.resolver()
        for c in candidates:
            if c.get("predicate") != "holds_role":
                continue
            subject, obj = c.get("subject") or "", c.get("object") or ""
            if c.get("subject_kind") == "person" and FULL_NAME.match(subject) and "." not in subject:
                if resolver.resolve(subject, EntityKind.PERSON).verdict is not MatchVerdict.MATCH:
                    added.append(self.add_entity("person", subject, aliases=[subject.split()[0]] if len(subject.split()) == 2 else []))
                    resolver = self.resolver()
            if c.get("object_kind") == "role" and obj and len(obj.split()) <= 5 and obj[0].isupper():
                if resolver.resolve(obj, EntityKind.ROLE).verdict is not MatchVerdict.MATCH:
                    added.append(self.add_entity("role", obj))
                    resolver = self.resolver()
        return added

    def _propose_categories(self, candidates: list[dict]) -> list[dict]:
        resolver = self.resolver()
        seen: dict[str, str] = {}
        for c in candidates:
            # only things someone may approve with a stated limit are worth a category
            constraints = c.get("constraints") or {}
            if c.get("predicate") != "may_approve" or c.get("polarity") == "negative" or constraints.get("maximum_amount") is None:
                continue
            if constraints.get("category") and resolver.resolve(str(constraints["category"]), EntityKind.CATEGORY).verdict is MatchVerdict.MATCH:
                continue  # the category field already places it
            if c.get("object_kind") == "category" and c.get("object"):
                mention = c["object"]
                if mention.lower() not in seen and resolver.resolve(mention, EntityKind.CATEGORY).verdict is not MatchVerdict.MATCH:
                    seen[mention.lower()] = (c.get("evidence") or [{}])[0].get("quoted_text", "")[:160]
        return [{"mention": m, "quote": q} for m, q in seen.items()]

    # ---- documents -------------------------------------------------------------------------
    def documents(self) -> list[dict]:
        out = []
        for meta in sorted((self.root / "documents").glob("*.json")):
            out.append(json.loads(meta.read_text()))
        return sorted(out, key=lambda d: (d.get("effective_date") or "", d["id"]))

    def _extractor(self):
        if self.extractor_kind == "rules":
            return RuleExtractor()
        return LlmExtractor(self.base_url, self.model, self.api_key, name=_slug(self.model))

    def ingest(self, *, path: Path | str | None = None, text: str | None = None, document_id: str | None = None, effective_date: str | None = None, kind: str = "policy", candidates: list[dict] | None = None) -> dict:
        if path is not None:
            path = Path(path).expanduser()
            text = path.read_text()
            document_id = document_id or path.stem
        if text is None or not document_id:
            raise ValueError("ingest needs a path, or text with a document_id")
        document_id = _slug(document_id) if not re.fullmatch(r"[A-Za-z0-9_.-]+", document_id) else document_id
        effective = effective_date or self._effective_date_from(text) or date.today().isoformat()
        doc = MarkdownParser().parse(document_id, text.encode())
        if candidates is None:
            extractor = self._extractor()
            candidates = extractor.extract(doc)
            extractor_name = extractor.name
        else:
            extractor_name = "provided"
        (self.root / "documents" / f"{document_id}.md").write_text(text)
        meta = {"id": document_id, "kind": kind, "effective_date": effective, "source": str(path) if path else "text", "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "extractor": extractor_name, "spans": len(doc.spans)}
        (self.root / "documents" / f"{document_id}.json").write_text(json.dumps(meta, indent=2))
        (self.root / "candidates" / f"{document_id}.json").write_text(json.dumps({"extractor": extractor_name, "document": document_id, "claims": candidates}, indent=2, default=str))
        added = self._auto_register(candidates)
        proposed = self._propose_categories(candidates)
        state = self._state()
        errors = [c for c in candidates if "error" in c]
        rejected = [t for t in state.transitions if t.event == "CANDIDATE_REJECTED" and t.subject_id.startswith(document_id + ":") or (t.event == "CANDIDATE_REJECTED" and len(self.documents()) == 1)]
        return {
            "document_id": document_id,
            "effective_date": effective,
            "kind": kind,
            "spans": len(doc.spans),
            "candidates": len(candidates) - len(errors),
            "unread_spans": [{"id": c.get("id"), "error": c.get("error"), "text": (c.get("span_text") or "")[:160]} for c in errors],
            "accepted_claims": sum(1 for c in state.claims if c.status.value in ("ACCEPTED", "CONFLICTED") and c.evidence and c.evidence[0].document_id == document_id),
            "rejected_at_boundary": len(rejected),
            "contradictions": len(state.contradictions),
            "auto_registered": added,
            "proposed_categories": proposed,
            "note": "register changes apply to every past document immediately; use inspect('rejected') to see what the boundary refused",
        }

    @staticmethod
    def _effective_date_from(text: str) -> str | None:
        from remembro.temporal.normalize import parse_date

        m = re.search(r"\*\*Effective date:\*\*\s*([^\n]+)|Effective date:\s*([^\n]+)|Effective\s+([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})", text)
        if not m:
            return None
        d = parse_date(next(g for g in m.groups() if g))
        return d.isoformat() if d else None

    def forget(self, document_id: str) -> bool:
        found = False
        for p in (self.root / "documents" / f"{document_id}.md", self.root / "documents" / f"{document_id}.json", self.root / "candidates" / f"{document_id}.json"):
            if p.exists():
                p.unlink()
                found = True
        return found

    # ---- state -----------------------------------------------------------------------------
    def _all_candidates(self) -> tuple[list[dict], dict[str, str]]:
        cands: list[dict] = []
        dates: dict[str, str] = {}
        for meta in self.documents():
            dates[meta["id"]] = meta["effective_date"]
            path = self.root / "candidates" / f"{meta['id']}.json"
            if path.exists():
                for c in json.loads(path.read_text())["claims"]:
                    if "id" in c and not str(c["id"]).startswith(meta["id"] + ":"):
                        c["id"] = f"{meta['id']}:{c['id']}"
                    cands.append(c)
        return cands, dates

    def _state(self):
        cands, dates = self._all_candidates()
        state, _ = build_state(cands, self.entities(), self.resolver(), document_dates=dates or None)
        return state

    # ---- questions -------------------------------------------------------------------------
    def decide(self, *, actor: str, resource: str, amount: float, currency: str = "AUD", on: str | None = None, action: str = "approve", self_benefit: bool = False, own_expense_claim: bool = False) -> dict:
        state = self._state()
        request = Request(actor=actor, action=action, resource=resource, amount=amount, currency=currency, on=on or date.today().isoformat(), self_benefit=self_benefit, own_expense_claim=own_expense_claim)
        decision = decide_request(request, state, self.resolver())
        evidence = []
        for eid in decision.evidence_used:
            for c in state.claims:
                for e in c.evidence:
                    if e.id == eid:
                        evidence.append({"document": e.document_id, "page": e.page, "paragraph": e.paragraph, "section": e.section, "authority": e.authority, "quote": e.quoted_text})
        return {
            "decision": decision.decision.value,
            "question": f"May {actor} {action} {amount:,.0f} {currency} of {resource} on {request.on}?",
            "reasons": decision.reasons,
            "unknown_reason": decision.unknown_reason,
            "evidence": evidence,
            "beliefs_used": decision.beliefs_used,
            "documents_consulted": [d["id"] for d in self.documents()],
        }

    def inspect(self, what: str, query: str | None = None) -> list[dict]:
        state = self._state()
        q = (query or "").lower()
        if what == "documents":
            return self.documents()
        if what == "register":
            return [e for e in self.register() if not q or q in json.dumps(e).lower()]
        if what == "claims":
            return [c.model_dump(mode="json") for c in state.claims if c.status.value != "REJECTED" and (not q or q in c.model_dump_json().lower())]
        if what == "rejected":
            rej = [{"id": c.id, "predicate": c.predicate.value, "subject": c.subject, "object": c.object, "reason": c.rejection_reason, "quote": c.evidence[0].quoted_text[:160] if c.evidence else ""} for c in state.claims if c.status.value == "REJECTED"]
            rej += [{"id": u.get("id"), "predicate": u.get("predicate"), "subject": u.get("subject"), "object": u.get("object"), "reason": "rejected at the boundary (schema or grounding)", "quote": ((u.get("evidence") or [{}])[0].get("quoted_text") or "")[:160]} for u in state.unreadable_claims]
            return [r for r in rej if not q or q in json.dumps(r, default=str).lower()]
        if what == "unread":
            return state.unread_spans
        if what == "beliefs":
            return [b.model_dump(mode="json") for b in state.beliefs if not q or q in b.model_dump_json().lower()]
        if what == "contradictions":
            out = []
            for con in state.contradictions:
                claims = [state.claim(cid) for cid in con.claims]
                out.append({"id": con.id, "type": con.type.value, "resolved": con.resolved, "resolution": con.resolution, "winner": con.winner, "claims": [{"id": c.id, "subject": c.subject, "limit": c.constraints.maximum_amount, "polarity": c.polarity.value, "quote": c.evidence[0].quoted_text[:160], "document": c.evidence[0].document_id} for c in claims]})
            return out
        if what == "transitions":
            return [t.model_dump(mode="json") for t in state.transitions if not q or q in t.model_dump_json().lower()]
        raise ValueError(f"unknown inspect target {what!r}; one of documents, register, claims, rejected, unread, beliefs, contradictions, transitions")
