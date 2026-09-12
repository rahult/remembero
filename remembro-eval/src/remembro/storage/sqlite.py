"""SQLite persistence of the deterministic state. Boring by design; a graph can be derived later."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from remembro.beliefs.pipeline import WorldState

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, kind TEXT, canonical_name TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases (entity_id TEXT, alias TEXT);
CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, status TEXT, subject TEXT, subject_entity TEXT, predicate TEXT, object TEXT, object_entity TEXT, modality TEXT, polarity TEXT, constraints TEXT, valid_from TEXT, valid_until TEXT, confidence REAL, rejection_reason TEXT, superseded_by TEXT);
CREATE TABLE IF NOT EXISTS claim_evidence (claim_id TEXT, evidence_id TEXT, document_id TEXT, page INTEGER, paragraph INTEGER, section TEXT, start_offset INTEGER, end_offset INTEGER, quoted_text TEXT, authority TEXT);
CREATE TABLE IF NOT EXISTS contradictions (id TEXT PRIMARY KEY, type TEXT, claims TEXT, resolved INTEGER, resolution TEXT, winner TEXT);
CREATE TABLE IF NOT EXISTS beliefs (id TEXT PRIMARY KEY, status TEXT, proposition TEXT, valid_from TEXT, valid_until TEXT);
CREATE TABLE IF NOT EXISTS belief_support (belief_id TEXT, claim_id TEXT, role TEXT);
CREATE TABLE IF NOT EXISTS state_transitions (sequence INTEGER PRIMARY KEY, event TEXT, subject_id TEXT, related_id TEXT, reason TEXT);
"""


def save_state(path: Path, state: WorldState) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for e in state.entities:
        con.execute("INSERT INTO entities VALUES (?,?,?)", (e.id, e.kind.value, e.canonical_name))
        for a in e.aliases:
            con.execute("INSERT INTO entity_aliases VALUES (?,?)", (e.id, a))
    for c in state.claims:
        con.execute("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (c.id, c.status.value, c.subject, c.subject_entity, c.predicate.value, c.object, c.object_entity, c.modality.value, c.polarity.value, c.constraints.model_dump_json(), str(c.valid_from) if c.valid_from else None, str(c.valid_until) if c.valid_until else None, c.confidence, c.rejection_reason, c.superseded_by))
        for e in c.evidence:
            con.execute("INSERT INTO claim_evidence VALUES (?,?,?,?,?,?,?,?,?,?)", (c.id, e.id, e.document_id, e.page, e.paragraph, e.section, e.start_offset, e.end_offset, e.quoted_text, e.authority))
    for x in state.contradictions:
        con.execute("INSERT INTO contradictions VALUES (?,?,?,?,?,?)", (x.id, x.type.value, json.dumps(x.claims), int(x.resolved), x.resolution, x.winner))
    for b in state.beliefs:
        con.execute("INSERT INTO beliefs VALUES (?,?,?,?,?)", (b.id, b.status.value, b.proposition.model_dump_json(), str(b.valid_from) if b.valid_from else None, str(b.valid_until) if b.valid_until else None))
        for cid in b.supporting_claims:
            con.execute("INSERT INTO belief_support VALUES (?,?,?)", (b.id, cid, "supports"))
        for cid in b.contradicting_claims:
            con.execute("INSERT INTO belief_support VALUES (?,?,?)", (b.id, cid, "contradicts"))
    for t in state.transitions:
        con.execute("INSERT INTO state_transitions VALUES (?,?,?,?,?)", (t.sequence, t.event, t.subject_id, t.related_id, t.reason))
    con.commit()
    con.close()
