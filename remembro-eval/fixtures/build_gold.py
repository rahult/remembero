"""Builds delegation_policy_v1.gold.json from quoted evidence, locating each quote in the parsed document."""
import json, sys
sys.path.insert(0, 'src')
from remembro.document.parser import MarkdownParser

doc = MarkdownParser().parse('delegation_policy_v1', open('fixtures/delegation_policy_v1.md', 'rb').read())

def ev(eid, quote, authority=None):
    for s in doc.spans:
        if quote in s.text:
            off = doc.text.find(quote)
            return {"id": eid, "document_id": doc.id, "page": s.page, "paragraph": s.paragraph, "section": s.section,
                    "start_offset": off if off >= 0 else s.start_offset, "end_offset": (off + len(quote)) if off >= 0 else s.end_offset,
                    "quoted_text": quote, "authority": authority or s.authority}
    raise SystemExit(f"quote not found: {quote[:60]}")

def claim(cid, subject, skind, predicate, obj=None, okind="none", modality="assertion", polarity="positive", constraints=None, vf=None, vu=None, evidence=None):
    c = {"id": cid, "subject": subject, "subject_kind": skind, "predicate": predicate, "object": obj, "object_kind": okind,
         "modality": modality, "polarity": polarity, "constraints": constraints or {}, "valid_from": vf, "valid_until": vu, "confidence": 1.0, "evidence": evidence}
    return c

AUD = "AUD"
claims = [
  claim("claim_role_alice", "Alice Morgan", "person", "holds_role", "Chief Financial Officer", "role", vf="2024-03-01", evidence=[ev("ev_role_alice", "Chief Financial Officer | Alice Morgan | 1 March 2024")]),
  claim("claim_role_bob", "Bob Chen", "person", "holds_role", "Finance Director", "role", vf="2025-07-15", evidence=[ev("ev_role_bob", "Finance Director | Bob Chen | 15 July 2025")]),
  claim("claim_role_carol", "Carol Evans", "person", "holds_role", "Procurement Manager", "role", vf="2025-02-02", evidence=[ev("ev_role_carol", "Procurement Manager | Carol Evans | 2 February 2025")]),
  claim("claim_role_david", "David Smith", "person", "holds_role", "Accounts Payable Lead", "role", vf="2025-11-11", evidence=[ev("ev_role_david", "Accounts Payable Lead | David Smith | 11 November 2025")]),
  claim("claim_cfo_opex", "Chief Financial Officer", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 100000, "currency": AUD, "category": "operational_expenditure"}, evidence=[ev("ev_cfo_opex", "The Chief Financial Officer may approve operational expenditure up to AUD 100,000 per transaction.")]),
  claim("claim_fd_opex", "Finance Director", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 50000, "currency": AUD, "category": "operational_expenditure"}, evidence=[ev("ev_fd_opex", "The Finance Director may approve operational expenditure up to AUD 50,000 per transaction.")]),
  claim("claim_pm_opex", "Procurement Manager", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 20000, "currency": AUD, "category": "operational_expenditure", "condition": "goods or services procured under an approved supplier agreement"}, evidence=[ev("ev_pm_opex", "The Procurement Manager may approve operational expenditure up to AUD 20,000 per transaction")]),
  claim("claim_cfo_capex", "Chief Financial Officer", "role", "may_approve", "capital expenditure", "category", "permission", "positive", {"maximum_amount": 250000, "currency": AUD, "category": "capital_expenditure"}, evidence=[ev("ev_cfo_capex", "The Chief Financial Officer may approve capital expenditure up to AUD 250,000 per project.")]),
  claim("claim_fd_capex", "Finance Director", "role", "may_approve", "capital expenditure", "category", "permission", "positive", {"maximum_amount": 75000, "currency": AUD, "category": "capital_expenditure"}, evidence=[ev("ev_fd_capex", "The Finance Director may approve capital expenditure up to AUD 75,000 per project.")]),
  claim("claim_pm_capex_prohibited", "Procurement Manager", "role", "may_approve", "capital expenditure", "category", "prohibition", "negative", {"category": "capital_expenditure"}, evidence=[ev("ev_pm_capex", "The Procurement Manager may not approve capital expenditure of any amount.")]),
  claim("claim_pm_self_benefit", "Procurement Manager", "role", "may_approve", "expenditure", "category", "prohibition", "negative", {"category": "expenditure", "condition": "benefiting themselves"}, evidence=[ev("ev_pm_self", "Procurement Managers must not approve expenditure benefiting themselves under any circumstances")]),
  claim("claim_delegation_alice_bob", "Alice Morgan", "person", "delegates", "Bob Chen", "person", "assertion", "positive", {"maximum_amount": 100000, "currency": AUD, "category": "operational_expenditure"}, "2026-09-01", "2026-09-20", evidence=[ev("ev_delegation", "During Alice Morgan's period of leave from 1 September 2026 to 20 September 2026, Alice delegates to Bob Chen her authority to approve operational expenditure. Under this delegation Bob Chen may approve operational expenditure up to AUD 100,000 per transaction.")]),
  claim("claim_carol_extension", "Carol Evans", "person", "may_approve", "invoices for operational expenditure", "category", "permission", "positive", {"maximum_amount": 30000, "currency": AUD, "category": "operational_expenditure"}, "2026-09-01", "2026-09-15", evidence=[ev("ev_carol_ext", "Carol Evans may approve invoices for operational expenditure up to AUD 30,000 per transaction between 1 September 2026 and 15 September 2026")]),
  claim("claim_dsmith_utilities", "D. Smith", "person", "may_approve", "utilities invoices classified as operational expenditure", "category", "permission", "positive", {"maximum_amount": 5000, "currency": AUD, "category": "operational_expenditure"}, "2026-09-01", "2026-09-30", evidence=[ev("ev_dsmith", "D. Smith may approve utilities invoices classified as operational expenditure up to AUD 5,000 per transaction from 1 September 2026 to 30 September 2026")]),
  claim("claim_carol_suspended", "Carol Evans", "person", "suspended", None, "none", "assertion", "positive", {}, "2026-09-08", None, evidence=[ev("ev_carol_susp", "Carol Evans's approval authority is suspended with effect from 8 September 2026 until further notice")]),
  claim("claim_revocation", "Alice Morgan", "person", "revokes_delegation", "Bob Chen", "person", "assertion", "positive", {"category": "operational_expenditure"}, "2026-09-12", None, evidence=[ev("ev_revocation", "Alice Morgan revoked the delegation of operational expenditure authority to Bob Chen described in section 5.2, with effect from 12 September 2026.")]),
  claim("claim_fd_capex_s7", "Finance Director", "role", "may_approve", "capital expenditure", "category", "permission", "positive", {"maximum_amount": 100000, "currency": AUD, "category": "capital_expenditure"}, evidence=[ev("ev_fd_capex_s7", "Capital expenditure approved by the Finance Director is capped at AUD 100,000 per project")]),
  claim("claim_appx_pm_opex", "Procurement Manager", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 25000, "currency": AUD, "category": "operational_expenditure"}, evidence=[ev("ev_appx_pm", "Procurement Manager | AUD 25,000 | Not permitted", "appendix")]),
  claim("claim_appx_cfo_opex", "Chief Financial Officer", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 100000, "currency": AUD, "category": "operational_expenditure"}, evidence=[ev("ev_appx_cfo", "Chief Financial Officer | AUD 100,000 | AUD 250,000", "appendix")]),
  claim("claim_appx_apl_opex", "Accounts Payable Lead", "role", "may_approve", "operational expenditure", "category", "prohibition", "negative", {"category": "operational_expenditure"}, evidence=[ev("ev_appx_apl_opex", "Accounts Payable Lead | Not permitted | Not permitted", "appendix")]),
  claim("claim_appx_apl_capex", "Accounts Payable Lead", "role", "may_approve", "capital expenditure", "category", "prohibition", "negative", {"category": "capital_expenditure"}, evidence=[ev("ev_appx_apl_capex", "Accounts Payable Lead | Not permitted | Not permitted", "appendix")]),
  claim("claim_appx_fd_opex", "Finance Director", "role", "may_approve", "operational expenditure", "category", "permission", "positive", {"maximum_amount": 50000, "currency": AUD, "category": "operational_expenditure"}, evidence=[ev("ev_appx_fd", "Finance Director | AUD 50,000 | AUD 75,000", "appendix")]),
]
entities = [
  {"id": "person_alice", "kind": "person", "canonical_name": "Alice Morgan", "aliases": ["Alice"]},
  {"id": "person_bob", "kind": "person", "canonical_name": "Bob Chen", "aliases": ["Bob"]},
  {"id": "person_carol", "kind": "person", "canonical_name": "Carol Evans", "aliases": ["Carol"]},
  {"id": "person_david", "kind": "person", "canonical_name": "David Smith", "aliases": []},
  {"id": "role_cfo", "kind": "role", "canonical_name": "Chief Financial Officer", "aliases": ["CFO"]},
  {"id": "role_fd", "kind": "role", "canonical_name": "Finance Director", "aliases": []},
  {"id": "role_pm", "kind": "role", "canonical_name": "Procurement Manager", "aliases": ["Procurement Managers"]},
  {"id": "role_apl", "kind": "role", "canonical_name": "Accounts Payable Lead", "aliases": []},
]
def sc(sid, question, actor, resource, amount, on, expected, reason, currency="AUD", **flags):
    return {"id": sid, "question": question, "request": {"actor": actor, "action": "approve", "resource": resource, "amount": amount, "currency": currency, "on": on, **flags}, "expected": {"decision": expected, "reason": reason}}
scenarios = [
  sc("scenario_01", "Can Alice approve AUD 80,000 operational expenditure on 2026-06-01?", "Alice Morgan", "operational_expenditure", 80000, "2026-06-01", "ALLOW", "CFO operational limit 100,000"),
  sc("scenario_02", "Can Alice approve AUD 120,000 operational expenditure on 2026-06-01?", "Alice Morgan", "operational_expenditure", 120000, "2026-06-01", "DENY", "exceeds CFO limit; Board approval required"),
  sc("scenario_03", "Can Bob approve AUD 90,000 operational expenditure on 2026-09-05?", "Bob Chen", "operational_expenditure", 90000, "2026-09-05", "ALLOW", "delegation from Alice in force"),
  sc("scenario_04", "Can Bob approve AUD 90,000 operational expenditure on 2026-09-25?", "Bob Chen", "operational_expenditure", 90000, "2026-09-25", "DENY", "delegation ended; Finance Director limit 50,000"),
  sc("scenario_05", "Can Bob approve AUD 70,000 operational expenditure on 2026-09-15?", "Bob Chen", "operational_expenditure", 70000, "2026-09-15", "DENY", "delegation revoked effective 12 September"),
  sc("scenario_06", "Can Bob approve AUD 90,000 capital expenditure on 2026-10-01?", "Bob Chen", "capital_expenditure", 90000, "2026-10-01", "UNKNOWN", "Finance Director capital limit stated as 75,000 (s4.2) and 100,000 (s7); unresolved contradiction"),
  sc("scenario_07", "Can David Smith approve AUD 4,000 of utilities invoices on 2026-09-05?", "David Smith", "utilities", 4000, "2026-09-05", "UNKNOWN", "permission is granted to 'D. Smith'; identity with David Smith not established"),
  sc("scenario_08", "Can Carol approve AUD 15,000 operational expenditure on 2026-09-10?", "Carol Evans", "operational_expenditure", 15000, "2026-09-10", "DENY", "suspended from 8 September"),
  sc("scenario_09", "Can Carol approve AUD 25,000 operational expenditure on 2026-09-05?", "Carol Evans", "operational_expenditure", 25000, "2026-09-05", "ALLOW", "temporary extension to 30,000 in force"),
  sc("scenario_10", "Can Carol approve AUD 25,000 operational expenditure on 2026-06-01?", "Carol Evans", "operational_expenditure", 25000, "2026-06-01", "DENY", "Procurement Manager limit 20,000; the appendix's 25,000 is overridden by the body"),
  sc("scenario_11", "Can Bob approve AUD 60,000 capital expenditure on 2026-10-01?", "Bob Chen", "capital_expenditure", 60000, "2026-10-01", "ALLOW", "within both stated Finance Director capital limits"),
  sc("scenario_12", "Can Carol approve AUD 10,000 capital expenditure on 2026-06-01?", "Carol Evans", "capital_expenditure", 10000, "2026-06-01", "DENY", "Procurement Manager may not approve capital expenditure"),
  sc("scenario_13", "Can Alice approve AUD 50,000 operational expenditure that benefits her on 2026-06-01?", "Alice Morgan", "operational_expenditure", 50000, "2026-06-01", "DENY", "self-benefit prohibition", self_benefit=True),
  sc("scenario_14", "Can Bob approve USD 90,000 operational expenditure on 2026-09-05?", "Bob Chen", "operational_expenditure", 90000, "2026-09-05", "UNKNOWN", "limits are in AUD; currency mismatch", currency="USD"),
  sc("scenario_15", "Can Erin Fox approve AUD 1,000 operational expenditure on 2026-06-01?", "Erin Fox", "operational_expenditure", 1000, "2026-06-01", "DENY", "unknown person"),
  sc("scenario_16", "Can Alice approve AUD 80,000 operational expenditure on 2026-09-05 while on leave?", "Alice Morgan", "operational_expenditure", 80000, "2026-09-05", "ALLOW", "leave does not remove authority; delegation adds Bob's"),
  sc("scenario_17", "Can Bob approve AUD 90,000 operational expenditure on 2026-09-11?", "Bob Chen", "operational_expenditure", 90000, "2026-09-11", "ALLOW", "day before revocation takes effect"),
]
gold = {"document": {"id": doc.id, "language": "en"}, "entities": entities, "claims": claims,
        "contradictions": [{"type": "LIMIT_MISMATCH", "claims": ["claim_pm_opex", "claim_appx_pm_opex"], "resolved": True, "resolution": "body prevails over appendix"},
                           {"type": "LIMIT_MISMATCH", "claims": ["claim_fd_capex", "claim_fd_capex_s7"], "resolved": False}],
        "scenarios": scenarios}
json.dump(gold, open('fixtures/delegation_policy_v1.gold.json', 'w'), indent=2)
print(len(claims), 'claims', len(scenarios), 'scenarios')
