# Transcript: examples/approvals-desk/run.sh

One run through the local writer (r23, Q8_0 GGUF under llama.cpp on a 16 GB M-series Mac), captured 2026-09-14. Ingesting the five-page schedule took 58 seconds, the Slack message 3, the amendment 9; every decision is instant.

```
exercise approvals-desk  extractor llm:rembero-writer
decisions 13 / 13   unjustified ALLOW 0   errors 0

  ok    1 ingest harbourview_delegations_v2: 35 candidates, 29 accepted, 1 rejected, 0 unread  (57.9s)  proposed categories: ['grants']
  ok    2 category {"name": "grant expenditure", "aliases": ["grants", "grant payments", "grant"]}
  PASS  3 May Nadia Kessler approve 18,000 NZD of grants on 2026-04-10?  expected ALLOW got ALLOW  (0.0s)
          because: Grants Manager, 25,000 per grant
          - role role_grants_manager: limit 25000.0 NZD for grant_expenditure
          - 18000.0 NZD is within the role role_grants_manager limit of 25000.0
          [harbourview_delegations_v2 p1 ¶7: Grants Manager | Nadia Kessler | 3 March 2025]
          [harbourview_delegations_v2 p2 ¶13: The Grants Manager is authorised to approve grant expenditure up to NZD 25,000 per grant p]
          [harbourview_delegations_v2 p4 ¶27: Grants Manager | NZD 15,000 | Not permitted | NZD 25,000]
  PASS  4 May Nadia Kessler approve 12,000 NZD of operational expenditure on 2026-04-10?  expected DENY got DENY  (0.0s)
          because: her operational limit is 10,000
          - role role_grants_manager: limit 10000.0 NZD for operational_expenditure
          - 12000.0 NZD exceeds every current limit for person_nadia_kessler over operational_expenditure on 2026-04-10
          [harbourview_delegations_v2 p1 ¶7: Grants Manager | Nadia Kessler | 3 March 2025]
          [harbourview_delegations_v2 p2 ¶13: The Grants Manager is authorised to approve grant expenditure up to NZD 25,000 per grant p]
  PASS  5 May Owen Pritchard approve 55,000 NZD of operational expenditure on 2026-07-01?  expected ALLOW got ALLOW  (0.0s)
          because: COO limit 60,000
          - personal grant: limit 150000.0 NZD for operational_expenditure
          - 55000.0 NZD is within the role role_chief_operating_officer limit of 60000.0
          [harbourview_delegations_v2 p1 ¶6: Chief Operating Officer | Owen Pritchard | 12 September 2025]
          [harbourview_delegations_v2 p2 ¶12: The Chief Operating Officer is authorised to approve operational expenditure up to NZD 60,]
          [harbourview_delegations_v2 p4 ¶26: Chief Operating Officer | NZD 60,000 | NZD 40,000 | Not permitted]
  PASS  6 May Owen Pritchard approve 55,000 NZD of operational expenditure on 2026-08-20?  expected DENY got DENY  (0.0s)
          because: authority suspended from 10 August pending review
          - actor resolved to person_owen_pritchard (exact alias 'Owen Pritchard')
          - person_owen_pritchard's approval authority is suspended on 2026-08-20; a suspended person may not exercise authority held directly or by del
          [harbourview_delegations_v2 p3 ¶20: The approval authority of Owen Pritchard is suspended with effect from 10 August 2026 pend]
  PASS  7 May Helen Marsh approve 90,000 NZD of capital expenditure on 2026-05-01?  expected ALLOW got ALLOW  (0.0s)
          because: CEO capital limit 300,000
          - role role_chief_executive_officer: limit 300000.0 NZD for capital_expenditure
          - 90000.0 NZD is within the role role_chief_executive_officer limit of 300000.0
          [harbourview_delegations_v2 p1 ¶5: Chief Executive Officer | Helen Marsh | 1 February 2024]
          [harbourview_delegations_v2 p2 ¶11: The Chief Executive Officer is authorised to approve operational expenditure up to NZD 150]
          [harbourview_delegations_v2 p4 ¶25: Chief Executive Officer | NZD 150,000 | NZD 300,000 | NZD 200,000]
  PASS  8 May Helen Marsh approve 250,000 NZD of grants on 2026-05-01?  expected DENY got DENY  (0.0s)
          because: CEO grant limit 200,000 per grant
          - role role_chief_executive_officer: limit 200000.0 NZD for grant_expenditure
          - 250000.0 NZD exceeds every current limit for person_helen_marsh over grant_expenditure on 2026-05-01
          [harbourview_delegations_v2 p1 ¶5: Chief Executive Officer | Helen Marsh | 1 February 2024]
          [harbourview_delegations_v2 p2 ¶11: The Chief Executive Officer is authorised to approve operational expenditure up to NZD 150]
          [harbourview_delegations_v2 p4 ¶25: Chief Executive Officer | NZD 150,000 | NZD 300,000 | NZD 200,000]
  ok    9 ingest slack-helen-2026-08-12: 1 candidates, 1 accepted, 0 rejected, 0 unread  (3.3s)
  PASS 10 May Nadia Kessler approve 15,000 NZD of operational expenditure on 2026-09-01?  expected ALLOW got ALLOW  (0.0s)
          because: the CEO's delegation to 20,000 until 30 September
          - personal grant: limit 20000.0 NZD for operational_expenditure
          - 15000.0 NZD is within the personal grant limit of 20000.0
          [harbourview_delegations_v2 p1 ¶7: Grants Manager | Nadia Kessler | 3 March 2025]
          [harbourview_delegations_v2 p2 ¶13: The Grants Manager is authorised to approve grant expenditure up to NZD 25,000 per grant p]
          [harbourview_delegations_v2 p1 ¶5: Chief Executive Officer | Helen Marsh | 1 February 2024]
  PASS 11 May Nadia Kessler approve 15,000 NZD of operational expenditure on 2026-10-05?  expected DENY got DENY  (0.0s)
          because: the delegation ended 30 September; her own limit is 10,000
          - role role_grants_manager: limit 10000.0 NZD for operational_expenditure
          - 15000.0 NZD exceeds every current limit for person_nadia_kessler over operational_expenditure on 2026-10-05
          [harbourview_delegations_v2 p1 ¶7: Grants Manager | Nadia Kessler | 3 March 2025]
          [harbourview_delegations_v2 p2 ¶13: The Grants Manager is authorised to approve grant expenditure up to NZD 25,000 per grant p]
  PASS 12 May Sam Oduya approve 500 NZD of operational expenditure on 2026-09-01?  expected DENY got DENY  (0.0s)
          because: not a known person; no authority can be attributed
          - actor 'Sam Oduya' is not a known person; no authority can be attributed
  PASS 13 May J. Ford approve 6,000 NZD of facilities maintenance on 2026-07-10?  expected UNKNOWN got UNKNOWN  (0.0s)
          because: Julian Ford and Julia Ford are both in the register; an initial is not an identity
          - actor identity unresolved: 'J. Ford' is a POSSIBLE_MATCH for person_julian_ford (initial 'j.' compatible with 'Julian Ford', identity not es
          next: if the mention and the register entry are the same person, add the mention as an alias with remembro_register(action='alias', name=<canonica
  PASS 14 May Owen Pritchard approve 55,000 NZD of operational expenditure on 2026-07-01?  expected DENY got DENY  (0.0s)
          because: a person may not approve expenditure in which they have an interest
          - actor resolved to person_owen_pritchard (exact alias 'Owen Pritchard')
          - the request benefits the approver; the policy prohibits approving expenditure from which the approver benefits
  ok   15 ingest harbourview_delegations_v2_amendment1: 5 candidates, 5 accepted, 0 rejected, 0 unread  (8.9s)
  PASS 16 May Owen Pritchard approve 70,000 NZD of operational expenditure on 2026-11-15?  expected ALLOW got ALLOW  (0.0s)
          because: 80,000 from 1 November; suspension lifted 1 September
          - role role_chief_operating_officer: limit 80000.0 NZD for operational_expenditure
          - 70000.0 NZD is within the role role_chief_operating_officer limit of 80000.0
          [harbourview_delegations_v2 p1 ¶6: Chief Operating Officer | Owen Pritchard | 12 September 2025]
          [harbourview_delegations_v2_amendment1 p1 ¶3: The Grants Manager is authorised to approve grant expenditure up to NZD 20,000 per grant. ]
          [harbourview_delegations_v2 p1 ¶5: Chief Executive Officer | Helen Marsh | 1 February 2024]
  PASS 17 May Nadia Kessler approve 22,000 NZD of grants on 2026-12-01?  expected DENY got DENY  (0.0s)
          because: grant limit lowered to 20,000
          - role role_grants_manager: limit 20000.0 NZD for grant_expenditure
          - 22000.0 NZD exceeds every current limit for person_nadia_kessler over grant_expenditure on 2026-12-01
          [harbourview_delegations_v2 p1 ¶7: Grants Manager | Nadia Kessler | 3 March 2025]
          [harbourview_delegations_v2_amendment1 p1 ¶4: Julian Ford ceased to hold the role of Facilities Coordinator on 31 October 2026. Marcus B]
          [harbourview_delegations_v2 p1 ¶6: Chief Operating Officer | Owen Pritchard | 12 September 2025]
```
