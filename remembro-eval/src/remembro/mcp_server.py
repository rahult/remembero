"""Remembro as an MCP server: documents in, ALLOW / DENY / UNKNOWN with evidence out.

  REMEMBRO_HOME=~/.remembro/default REMEMBRO_EXTRACTOR=llm \
  REMEMBRO_MODEL=glm-5.3-flash:cloud REMEMBRO_BASE_URL=http://127.0.0.1:11434/v1 REMEMBRO_API_KEY=ollama \
  python -m remembro.mcp_server

Everything below the trust boundary is deterministic and rebuilt from stored candidates on every
call; the extractor is the only model in the loop, and it runs only at ingest time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from remembro.workspace import Workspace

INSTRUCTIONS = """Remembro turns documents (policies, delegation schedules, amendments, Slack messages that grant,
suspend or revoke authority) into evidence-bearing claims and answers "may X approve Y" with ALLOW,
DENY or UNKNOWN. UNKNOWN is a real answer: it means the documents do not settle the question (an
ambiguous identity, an unresolved contradiction, a passage the extractor could not read). Never
turn an UNKNOWN into permission. Every ALLOW and DENY carries the quotes it rests on; show them.

Typical flow: remembro_ingest a document (path or pasted text, with its effective date), check
proposed_categories and register them with remembro_register if they matter, then remembro_decide.
Use remembro_inspect("rejected") to see what the boundary refused and why."""

server = MCPServer(
    name="remembro",
    title="Remembro decisions",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


def _safe(fn):
    """Tool errors come back as data the agent can act on, never as a dead tool call."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            return {"error": f"{type(error).__name__}: {error}"}

    return wrapper


def _workspace() -> Workspace:
    return Workspace(
        Path(os.environ.get("REMEMBRO_HOME", "~/.remembro/default")),
        extractor=os.environ.get("REMEMBRO_EXTRACTOR", "llm"),
        model=os.environ.get("REMEMBRO_MODEL"),
        base_url=os.environ.get("REMEMBRO_BASE_URL"),
        api_key=os.environ.get("REMEMBRO_API_KEY"),
        match_threshold=float(os.environ.get("REMEMBRO_MATCH_THRESHOLD", "0.9")),
    )


@server.tool(name="remembro_ingest", description="Ingest a document into the workspace: a file path, or pasted text (a policy, an amendment, a Slack message that grants, delegates, suspends or revokes approval authority). Extracts claims with the configured model, auto-registers the people and roles the document itself names, proposes categories it does not know, and reports what the boundary refused. Re-ingesting the same document_id replaces it.")
@_safe
def remembro_ingest(path: str | None = None, text: str | None = None, document_id: str | None = None, effective_date: str | None = None, kind: str = "policy") -> dict[str, Any]:
    """effective_date is ISO (YYYY-MM-DD); when omitted it is read from an 'Effective date:' line or defaults to today. kind is free text: policy, amendment, message, email."""
    return _workspace().ingest(path=path, text=text, document_id=document_id, effective_date=effective_date, kind=kind)


@server.tool(name="remembro_decide", description="Answer 'may this person approve this expenditure on this date?' from the ingested documents. Returns ALLOW, DENY or UNKNOWN with the reasoning chain and the exact quotes (document, page, paragraph) it rests on. UNKNOWN means the documents do not settle it; do not treat it as permission.")
@_safe
def remembro_decide(actor: str, resource: str, amount: float, currency: str = "AUD", on: str | None = None, self_benefit: bool = False, own_expense_claim: bool = False) -> dict[str, Any]:
    """actor: a person's name as written; resource: the expenditure category (operational expenditure, capital expenditure, grants, or any registered category); on: ISO date, default today."""
    return _workspace().decide(actor=actor, resource=resource, amount=amount, currency=currency, on=on, self_benefit=self_benefit, own_expense_claim=own_expense_claim)


@server.tool(name="remembro_register", description="Manage the entity register: list it, add a person / role / category (with aliases), add aliases to an existing entry, or remove an entry. Registering an alias is how an ambiguous mention becomes a match; registering a category is how a proposed category becomes decidable. Changes apply to every ingested document immediately.")
@_safe
def remembro_register(action: str = "list", kind: str | None = None, name: str | None = None, aliases: list[str] | None = None, entity_id: str | None = None) -> dict[str, Any]:
    """action: list | add | alias | remove. add needs kind (person|role|category) and name; alias needs name (or entity_id) and aliases; remove needs entity_id."""
    ws = _workspace()
    if action == "list":
        return {"entities": ws.register()}
    if action == "add":
        if not (kind and name):
            return {"error": "add needs kind and name"}
        return {"added": ws.add_entity(kind, name, aliases or [], entity_id)}
    if action == "alias":
        reg = ws.register()
        target = next((e for e in reg if e["id"] == entity_id or (name and e["canonical_name"].lower() == name.lower())), None)
        if not target:
            return {"error": f"no entity {entity_id or name!r}"}
        return {"updated": ws.add_entity(target["kind"], target["canonical_name"], aliases or [], target["id"])}
    if action == "remove":
        return {"removed": ws.remove_entity(entity_id or "")}
    return {"error": f"unknown action {action!r}"}


@server.tool(name="remembro_inspect", description="Look inside the workspace: documents, register, claims, rejected (what the boundary refused and why), unread (spans the extractor failed on), dismissed, beliefs, contradictions, transitions (the full audit log). Optional query filters by substring.")
@_safe
def remembro_inspect(what: str = "documents", query: str | None = None, limit: int = 50) -> dict[str, Any]:
    items = _workspace().inspect(what, query)
    return {"what": what, "count": len(items), "items": items[:limit]}


@server.tool(name="remembro_status", description="One-call overview of the workspace: documents, people, roles, categories, accepted claims, what the boundary refused, unread spans and open contradictions. Call this first in a session.")
@_safe
def remembro_status() -> dict[str, Any]:
    return _workspace().status()


@server.tool(name="remembro_dismiss", description="Dismiss a rejected candidate as extractor noise, with a reason, so it stops casting doubt on decisions. Use only after reading it with remembro_inspect('rejected'): a dismissal is a human judgement on the record, never a way around a real restriction. undo=true restores it.")
@_safe
def remembro_dismiss(candidate_id: str, reason: str = "", undo: bool = False) -> dict[str, Any]:
    ws = _workspace()
    if undo:
        return {"restored": ws.undismiss(candidate_id), "candidate_id": candidate_id}
    if not reason.strip():
        return {"error": "a dismissal needs a reason; it goes on the record"}
    return {"dismissed": ws.dismiss(candidate_id, reason)}


@server.tool(name="remembro_forget", description="Remove an ingested document and its claims from the workspace. The register is untouched.")
@_safe
def remembro_forget(document_id: str) -> dict[str, Any]:
    return {"forgotten": _workspace().forget(document_id), "document_id": document_id}


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
