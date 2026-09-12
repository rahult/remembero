"""Builds harbourview_delegations_v2.gold.json: two documents (a schedule and a later amendment),
five people with a shared surname, three expenditure categories in NZD, and the amendment's
transaction-time semantics. Each quote is located in its parsed document."""
import json, sys
sys.path.insert(0, 'src')
from remembro.document.parser import MarkdownParser

DOCS = {
  "harbourview_delegations_v2": "2026-03-01",
  "harbourview_delegations_v2_amendment1": "2026-11-01",
}
parsed = {d: MarkdownParser().parse(d, open(f"fixtures/{d}.md", "rb").read()) for d in DOCS}

def ev(eid, quote, doc="harbourview_delegations_v2", authority=None):
    p = parsed[doc]
    for s in p.spans:
        if quote in s.text:
            off = p.text.find(quote)
            return {"id": eid, "document_id": doc, "page": s.page, "paragraph": s.paragraph, "section": s.section,
                    "start_offset": off if off >= 0 else s.start_offset, "end_offset": (off + len(quote)) if off >= 0 else s.end_offset,
                    "quoted_text": quote, "authority": authority or s.authority}
    raise SystemExit(f"quote not found in {doc}: {quote[:60]}")

def claim(cid, subject, skind, predicate, obj=None, okind="none", modality="assertion", polarity="positive", constraints=None, vf=None, vu=None, evidence=None):
    return {"id": cid, "subject": subject, "subject_kind": skind, "predicate": predicate, "object": obj, "object_kind": okind,
            "modality": modality, "polarity": polarity, "constraints": constraints or {}, "valid_from": vf, "valid_until": vu, "confidence": 1.0, "evidence": evidence}

NZD = "NZD"
OPEX, CAPEX, GRANT = "operational_expenditure", "capital_expenditure", "grant_expenditure"
def lim(cid, role, cat_text, cat, amount, quote, cond=None, doc="harbourview_delegations_v2"):
    c = {"maximum_amount": amount, "currency": NZD, "category": cat}
    if cond: c["condition"] = cond
    return claim(cid, role, "role", "may_approve", cat_text, "category", "permission", "positive", c, evidence=[ev("ev_" + cid, quote, doc)])
def forbid(cid, role, cat_text, cat, quote, doc="harbourview_delegations_v2"):
    return claim(cid, role, "role", "may_approve", cat_text, "category", "prohibition", "negative", {"category": cat}, evidence=[ev("ev_" + cid, quote, doc)])

A = "harbourview_delegations_v2_amendment1"
claims = [
  claim("c_role_helen", "Helen Marsh", "person", "holds_role", "Chief Executive Officer", "role", vf="2024-02-01", evidence=[ev("ev_role_helen", "Chief Executive Officer | Helen Marsh | 1 February 2024")]),
  claim("c_role_owen", "Owen Pritchard", "person", "holds_role", "Chief Operating Officer", "role", vf="2025-09-12", evidence=[ev("ev_role_owen", "Chief Operating Officer | Owen Pritchard | 12 September 2025")]),
  claim("c_role_nadia", "Nadia Kessler", "person", "holds_role", "Grants Manager", "role", vf="2025-03-03", evidence=[ev("ev_role_nadia", "Grants Manager | Nadia Kessler | 3 March 2025")]),
  claim("c_role_julian", "Julian Ford", "person", "holds_role", "Facilities Coordinator", "role", vf="2026-01-20", evidence=[ev("ev_role_julian", "Facilities Coordinator | Julian Ford | 20 January 2026")]),
  claim("c_role_julia", "Julia Ford", "person", "holds_role", "Grants Officer", "role", vf="2025-05-05", evidence=[ev("ev_role_julia", "Grants Officer | Julia Ford | 5 May 2025")]),
  lim("c_ceo_opex", "Chief Executive Officer", "operational expenditure", OPEX, 150000, "The Chief Executive Officer is authorised to approve operational expenditure up to NZD 150,000 per transaction."),
  lim("c_ceo_capex", "Chief Executive Officer", "capital expenditure", CAPEX, 300000, "The Chief Executive Officer is authorised to approve capital expenditure up to NZD 300,000 per transaction."),
  lim("c_ceo_grant", "Chief Executive Officer", "grant expenditure", GRANT, 200000, "The Chief Executive Officer is authorised to approve grant expenditure up to NZD 200,000 per grant."),
  lim("c_coo_opex", "Chief Operating Officer", "operational expenditure", OPEX, 60000, "The Chief Operating Officer is authorised to approve operational expenditure up to NZD 60,000 per transaction."),
  lim("c_coo_capex", "Chief Operating Officer", "capital expenditure", CAPEX, 40000, "The Chief Operating Officer is authorised to approve capital expenditure up to NZD 40,000 per transaction."),
  forbid("c_coo_grant", "Chief Operating Officer", "grant expenditure", GRANT, "The Chief Operating Officer shall not approve grant expenditure."),
  lim("c_gm_grant", "Grants Manager", "grant expenditure", GRANT, 25000, "The Grants Manager is authorised to approve grant expenditure up to NZD 25,000 per grant provided the grant falls within a program approved by the Board.", cond="the grant falls within a program approved by the Board"),
  lim("c_gm_opex", "Grants Manager", "operational expenditure", OPEX, 10000, "The Grants Manager is authorised to approve operational expenditure up to NZD 10,000 per transaction."),
  forbid("c_gm_capex", "Grants Manager", "capital expenditure", CAPEX, "The Grants Manager shall not approve capital expenditure."),
  lim("c_fc_opex", "Facilities Coordinator", "facilities maintenance classified as operational expenditure", OPEX, 5000, "The Facilities Coordinator is authorised to approve facilities maintenance classified as operational expenditure up to NZD 5,000 per transaction.", cond="facilities maintenance"),
  claim("c_delegation_helen_owen", "Helen Marsh", "person", "delegates", "Owen Pritchard", "person", "assertion", "positive", {"maximum_amount": 150000, "currency": NZD, "category": OPEX}, "2026-07-01", "2026-08-31", evidence=[ev("ev_delegation", "For the period 1 July 2026 to 31 August 2026, while Helen Marsh is on sabbatical, Owen Pritchard is delegated the Chief Executive Officer's authority for operational expenditure up to NZD 150,000 per transaction.")]),
  claim("c_nadia_round", "Nadia Kessler", "person", "may_approve", "grant expenditure", "category", "permission", "positive", {"maximum_amount": 40000, "currency": NZD, "category": GRANT}, "2026-07-15", "2026-08-15", evidence=[ev("ev_nadia_round", "Nadia Kessler is authorised, from 15 July 2026 to 15 August 2026, to approve grant expenditure up to NZD 40,000 per grant")]),
  claim("c_jford_works", "J. Ford", "person", "may_approve", "facilities maintenance invoices", "category", "permission", "positive", {"maximum_amount": 8000, "currency": NZD, "category": OPEX}, "2026-07-01", "2026-07-31", evidence=[ev("ev_jford", "J. Ford may approve facilities maintenance invoices up to NZD 8,000 per transaction between 1 July 2026 and 31 July 2026")]),
  claim("c_owen_suspended", "Owen Pritchard", "person", "suspended", None, "none", "assertion", "positive", {"condition": "pending the outcome of an internal review"}, "2026-08-10", None, evidence=[ev("ev_owen_susp", "The approval authority of Owen Pritchard is suspended with effect from 10 August 2026 pending the outcome of an internal review")]),
  claim("c_revocation_nadia", "Helen Marsh", "person", "revokes_delegation", "Nadia Kessler", "person", "assertion", "positive", {"category": GRANT}, "2026-08-01", None, evidence=[ev("ev_revocation", "Helen Marsh revoked the authorisation granted to Nadia Kessler under clause 5.2 with effect from 1 August 2026")]),
  lim("c_coo_capex_s7", "Chief Operating Officer", "capital expenditure", CAPEX, 50000, "the Chief Operating Officer's limit for capital expenditure is NZD 50,000 per transaction"),
  # Appendix A rows: agreeing rows restate, one contradicts the body, two add prohibitions
  lim("c_appx_gm_opex", "Grants Manager", "operational expenditure", OPEX, 15000, "Grants Manager | NZD 15,000 | Not permitted | NZD 25,000"),
  forbid("c_appx_fc_capex", "Facilities Coordinator", "capital expenditure", CAPEX, "Facilities Coordinator | NZD 5,000 | Not permitted | Not permitted"),
  forbid("c_appx_fc_grant", "Facilities Coordinator", "grant expenditure", GRANT, "Facilities Coordinator | NZD 5,000 | Not permitted | Not permitted"),
  # Amendment No. 1, effective 1 November 2026
  lim("c_am_coo_opex", "Chief Operating Officer", "operational expenditure", OPEX, 80000, "The Chief Operating Officer is authorised to approve operational expenditure up to NZD 80,000 per transaction.", doc=A),
  lim("c_am_gm_grant", "Grants Manager", "grant expenditure", GRANT, 20000, "The Grants Manager is authorised to approve grant expenditure up to NZD 20,000 per grant.", doc=A),
  claim("c_am_julian_ends", "Julian Ford", "person", "holds_role", "Facilities Coordinator", "role", vu="2026-10-31", evidence=[ev("ev_am_julian", "Julian Ford ceased to hold the role of Facilities Coordinator on 31 October 2026.", A)]),
  claim("c_am_marcus", "Marcus Bell", "person", "holds_role", "Facilities Coordinator", "role", vf="2026-11-01", evidence=[ev("ev_am_marcus", "Marcus Bell was appointed Facilities Coordinator with effect from 1 November 2026.", A)]),
  claim("c_am_owen_lifted", "Owen Pritchard", "person", "suspended", None, "none", "assertion", "positive", {}, "2026-08-10", "2026-08-31", evidence=[ev("ev_am_lifted", "The suspension of Owen Pritchard's approval authority under clause 6.1 of the Schedule, which took effect on 10 August 2026, was lifted with effect from 1 September 2026.", A)]),
]
entities = [
  {"id": "person_helen", "kind": "person", "canonical_name": "Helen Marsh", "aliases": ["Helen"]},
  {"id": "person_owen", "kind": "person", "canonical_name": "Owen Pritchard", "aliases": ["Owen"]},
  {"id": "person_nadia", "kind": "person", "canonical_name": "Nadia Kessler", "aliases": ["Nadia"]},
  {"id": "person_julian", "kind": "person", "canonical_name": "Julian Ford", "aliases": []},
  {"id": "person_julia", "kind": "person", "canonical_name": "Julia Ford", "aliases": []},
  {"id": "person_marcus", "kind": "person", "canonical_name": "Marcus Bell", "aliases": []},
  {"id": "role_ceo", "kind": "role", "canonical_name": "Chief Executive Officer", "aliases": ["CEO"]},
  {"id": "role_coo", "kind": "role", "canonical_name": "Chief Operating Officer", "aliases": ["COO"]},
  {"id": "role_gm", "kind": "role", "canonical_name": "Grants Manager", "aliases": []},
  {"id": "role_fc", "kind": "role", "canonical_name": "Facilities Coordinator", "aliases": []},
  {"id": "role_go", "kind": "role", "canonical_name": "Grants Officer", "aliases": []},
  {"id": "grant_expenditure", "kind": "category", "canonical_name": "grant expenditure", "aliases": ["grants", "grant payments", "grant"]},
  {"id": "operational_expenditure", "kind": "category", "canonical_name": "operational expenditure", "aliases": ["facilities maintenance", "facilities maintenance invoices", "operational"]},
  {"id": "capital_expenditure", "kind": "category", "canonical_name": "capital expenditure", "aliases": ["capital"]},
]
contradictions = [
  {"id": "gold_contradiction_1", "claims": ["c_coo_capex", "c_coo_capex_s7"], "type": "LIMIT_MISMATCH", "resolvable": False, "note": "body versus body: 40,000 in 3.2, 50,000 in 7"},
  {"id": "gold_contradiction_2", "claims": ["c_gm_opex", "c_appx_gm_opex"], "type": "LIMIT_MISMATCH", "resolvable": True, "winner": "c_gm_opex", "note": "body prevails over Appendix A"},
]
def sc(sid, question, actor, resource, amount, on, expected, reason, currency="NZD", **flags):
    return {"id": sid, "question": question, "request": {"actor": actor, "action": "approve", "resource": resource, "amount": amount, "currency": currency, "on": on, **flags}, "expected": {"decision": expected, "reason": reason}}
scenarios = [
  sc("scenario_01", "Can Helen approve NZD 120,000 operational expenditure on 2026-05-01?", "Helen Marsh", "operational_expenditure", 120000, "2026-05-01", "ALLOW", "CEO operational limit 150,000"),
  sc("scenario_02", "Can Owen approve NZD 45,000 capital expenditure on 2026-05-01?", "Owen Pritchard", "capital_expenditure", 45000, "2026-05-01", "UNKNOWN", "COO capital limit is 40,000 in 3.2 and 50,000 in 7; the amount falls between"),
  sc("scenario_03", "Can Owen approve NZD 30,000 capital expenditure on 2026-05-01?", "Owen Pritchard", "capital_expenditure", 30000, "2026-05-01", "ALLOW", "both disputed limits permit 30,000"),
  sc("scenario_04", "Can Owen approve NZD 120,000 operational expenditure on 2026-07-15?", "Owen Pritchard", "operational_expenditure", 120000, "2026-07-15", "ALLOW", "sabbatical delegation 150,000"),
  sc("scenario_05", "Can Owen approve NZD 120,000 operational expenditure on 2026-08-15?", "Owen Pritchard", "operational_expenditure", 120000, "2026-08-15", "DENY", "suspended from 10 August"),
  sc("scenario_06", "Can Owen approve NZD 5,000 grant expenditure on 2026-07-15?", "Owen Pritchard", "grant_expenditure", 5000, "2026-07-15", "DENY", "COO may not approve grants; the delegation does not extend to grants"),
  sc("scenario_07", "Can Nadia approve a NZD 35,000 grant on 2026-07-20?", "Nadia Kessler", "grant_expenditure", 35000, "2026-07-20", "ALLOW", "grant round authorisation 40,000"),
  sc("scenario_08", "Can Nadia approve a NZD 35,000 grant on 2026-08-05?", "Nadia Kessler", "grant_expenditure", 35000, "2026-08-05", "DENY", "authorisation revoked from 1 August; standing limit 25,000"),
  sc("scenario_09", "Can Nadia approve a NZD 22,000 grant on 2026-10-01?", "Nadia Kessler", "grant_expenditure", 22000, "2026-10-01", "ALLOW", "standing limit 25,000 before the amendment"),
  sc("scenario_10", "Can Nadia approve a NZD 22,000 grant on 2026-12-01?", "Nadia Kessler", "grant_expenditure", 22000, "2026-12-01", "DENY", "amendment lowers the limit to 20,000 from 1 November"),
  sc("scenario_11", "Can Nadia approve NZD 12,000 operational expenditure on 2026-05-01?", "Nadia Kessler", "operational_expenditure", 12000, "2026-05-01", "DENY", "body limit 10,000 prevails over Appendix A 15,000"),
  sc("scenario_12", "Can J. Ford approve NZD 6,000 of facilities maintenance on 2026-07-10?", "J. Ford", "facilities maintenance", 6000, "2026-07-10", "UNKNOWN", "J. Ford may be Julian Ford or Julia Ford"),
  sc("scenario_13", "Can Julian Ford approve NZD 4,000 of facilities maintenance on 2026-10-15?", "Julian Ford", "facilities maintenance", 4000, "2026-10-15", "ALLOW", "Facilities Coordinator limit 5,000"),
  sc("scenario_14", "Can Julian Ford approve NZD 4,000 of facilities maintenance on 2026-11-15?", "Julian Ford", "facilities maintenance", 4000, "2026-11-15", "DENY", "ceased to hold the role on 31 October"),
  sc("scenario_15", "Can Marcus Bell approve NZD 4,000 of facilities maintenance on 2026-11-15?", "Marcus Bell", "facilities maintenance", 4000, "2026-11-15", "ALLOW", "appointed Facilities Coordinator from 1 November"),
  sc("scenario_16", "Can Owen approve NZD 70,000 operational expenditure on 2026-10-15?", "Owen Pritchard", "operational_expenditure", 70000, "2026-10-15", "DENY", "COO limit 60,000 before the amendment; suspension lifted 1 September"),
  sc("scenario_17", "Can Owen approve NZD 70,000 operational expenditure on 2026-11-15?", "Owen Pritchard", "operational_expenditure", 70000, "2026-11-15", "ALLOW", "amendment raises the COO limit to 80,000"),
  sc("scenario_18", "Can Helen approve NZD 100,000 operational expenditure that benefits her on 2026-05-01?", "Helen Marsh", "operational_expenditure", 100000, "2026-05-01", "DENY", "no delegate may approve a payment to themselves", self_benefit=True),
  sc("scenario_19", "Can Helen approve AUD 100,000 operational expenditure on 2026-05-01?", "Helen Marsh", "operational_expenditure", 100000, "2026-05-01", "UNKNOWN", "limits are stated in NZD", currency="AUD"),
  sc("scenario_20", "Can Julia Ford approve NZD 1,000 of facilities maintenance on 2026-07-10?", "Julia Ford", "facilities maintenance", 1000, "2026-07-10", "UNKNOWN", "the Grants Officer has no standing authority, but the J. Ford grant may be hers; identity not established"),
]
gold = {"documents": [{"id": d, "path": f"fixtures/{d}.md", "effective_date": eff} for d, eff in DOCS.items()],
        "entities": entities, "claims": claims, "contradictions": contradictions, "scenarios": scenarios}
json.dump(gold, open("fixtures/harbourview_delegations_v2.gold.json", "w"), indent=2)
print(len(claims), "claims", len(scenarios), "scenarios")
