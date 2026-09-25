# The support stack: claims, a gate, and a reasoner

Built 2026-09-24 from the coding-agent brief (neuro-symbolic reasoning + provenance for support
agents), refined by the adversarial review. The LLM is a sensor; code decides what its output is
worth; a symbolic engine decides the question. Same contract as the compose engine — Proved,
denied, or Unknown with a reason, never confidently wrong — applied to tickets, chats, and
long-form SOPs.

## The layers, and the one contract

```
sources   tickets, chats, policy documents -> spans {sourceId, spanId, kind, text, actor?, at?}
                        src/support/sources.ts
claims    positive ground atoms citing ONE span, typed args, trust (system|model|hand)
                        src/support/claims.ts   <-- the stable contract
gate      admits a claim only if code verifies every argument on the cited span
engine    closed query shapes -> verdict (allow | deny | unknown) + proof JSON
                        src/support/engine.ts
store     append-only: every decision keeps its whole input basis
                        src/support/store.ts
```

The contract that must stay stable is not just the claim JSON — it is **schema + admission
gate**. Swap the extractor (deepseek today, a 2B LoRA tomorrow) and nothing else notices; the
gate carries the certainty, which is the same result the compose engine measured twice
(r1 and r2 both 0% confidently wrong behind the same check).

### Decisions that carry over from the adversarial review

- **No confidence numbers.** A model that invents facts cannot say which ones it invented.
  Trust is a class (`system` fields, `model` behind the gate, `hand` pack), not a score.
- **Same-span rule.** Every argument must be stated on the single cited span. Cross-turn and
  cross-line joins are engine work. (This rule took compose extraction precision to ~100%.)
- **Closed shapes, not free Datalog.** `sla_credits`, `sla_met`, `credit_percent` — a question
  that maps to no shape is rejected, never guessed. Parameters bind to ids: an entity named
  only by description is rejected ("bind by id"), so identity ambiguity becomes workflow.
- **Positive atoms only.** Absence, deadlines, comparisons, calendar math are computed. An
  extracted negative is a hallucination with extra steps.
- **Escalate is proved; Unknown is honest.** A policy-mandated human decision is a verdict
  citing the rule; insufficient evidence is an Unknown that says what was searched
  (`unknown.spansSearched`) and what was missing (`unknown.missing`).
- **The pack is human-gated.** Policy parameters are extracted from the customer's SOP into a
  versioned pack file, and a pack loads only when every claim passes the same gate
  (`src/support/pack.ts`). The gate catches hallucination; the human sign-off catches
  misreading; the coverage review (`pack.unconsumedLines`) flags SOP lines no claim consumes —
  an unmodeled exception clause is surfaced to the reviewer, never silently ignored.
- **Determinism is over the admitted claim set.** The store's `decisionHash` covers verdict,
  summary, facts (content, not allocated ids), and computation — two runs on the same basis
  hash equal, volatile fields excluded.

## The SLA wedge, and why

First-response SLA credits: breach is arithmetic over timestamps plus a business calendar
(hours, weekends, holidays) — exactly the multi-hop date math readers fumble, provable here.
Worked example (`benchmarks/support/demo-case.json`, ticket opened 16:30 UTC the day before
Christmas):

```
npm run support:decide -- --case benchmarks/support/demo-case.json --question "is a credit due for TCK-1042"
-> allow: Credit of 25% of the 1200/mo fee (300): ticket tck-1042 breached by 450 business minutes.
   proof: 13 facts, each with span + trust; calendar walk (690 business minutes: the holiday
   Friday and the weekend contribute zero); pack acme-sla@1.2 lines p005, p011, p013–p014, p017–p019.
```

## The eval is the moat

`npm run support:gold` builds the gold corpus — cases whose verdicts are **certain by
construction**: a seeded generator makes case facts, the engine computes gold from those facts
(the compose methodology: labels are never hand-written), and renderers project the same facts
into a tidy ticket, a chat transcript, and a noisy re-rendering. Measured, seed 20260924, 40
cases (results/support-gold.json):

| arm | verdict match | what it proves |
| --- | --- | --- |
| tidy-ticket | 40/40 | the engine path on structured records |
| chat-transcript | 40/40 | the same facts as conversation spans |
| chat-noisy | 40/40 | surface-noise robustness: 0 proof flips vs the clean rendering |
| info-loss-response | 40/40 | a deleted fact -> Unknown, never a confident answer |
| info-loss-tier | 40/40 | the missing tier -> Unknown with the reason |
| llm-extract (llama3.1:8b, local) | 40/40 | a local 8B sensor behind the gate: **same verdicts as the deterministic path, $0 API** |

Pack coverage review flagged 2 unconsumed SOP lines (the follow-the-sun calendar and the
credit-timing clause) — both genuinely unmodeled, exactly what the reviewer should see.

`--extract` swaps the deterministic harvest for the LLM sensor behind the gate and reports the
admit/reject ledger plus `failedSpans` — extraction failures are surfaced, never silent. The
first live sensor run (`--ollama`, llama3.1:8b, results/support-gold-ollama-llama31-8b.json)
puts the exit thesis on real numbers: **175 candidate claims, the gate rejected 40 (23%) —
invented predicates, misformatted dates, arguments not on the span — and the 135 that passed
produce the same 40/40 verdicts as the hand-built fact base.** The same day, the demo case
decided through Ollama returns the identical credit (25% of 1200 = 300) with honest provenance:
verdict, summary and computation equal to the deterministic run, the sensor's own rejections
kept in the ledger. What the gate cannot yet do is fix under-extraction — llama missed
`opened` in prose until prompted with an example, so the one-shot prompt now carries one;
recall on messy real text stays the measurement the gate challenge set exists for.

## Tests

`tests/support.test.ts`, 25 tests: the gate (same-span, surfaces, type errors, rejections kept
with reasons), the engine (the worked example over the holiday calendar, boundary denies,
every structural Unknown, determinism), the pack (loads only when fully grounded, tampered
pack refused, coverage flags), the store (append-only, stable hashes), and the gold arms.

## Phase 1 on real text: the gate challenge (2026-09-25)

No customer export yet, so the kill-criterion measurement ran over the repo's 223 real
XL-DocBench documents — genuine government/agency prose, the messiest text we own. Ground
truth is certain by construction: dates and dollar amounts are extracted from each span, and
the gate is probed with the TRUE value (must admit — a rejection is an OVER-REJECTION) and
with seeded mutants (must reject — an admission is a FALSE ADMIT); 495 probes over 52 spans
(`npm run support:gate`, results/support-gate-challenge.json):

| run | over-rejection | false admission | worst class |
| --- | --- | --- | --- |
| baseline (substring matching) | 0/77 = 0% | 10/122 = 8.2% | drop-trailing 100%, drop-leading 80% |
| + digit-boundary matching | 0% | 6/418 = 1.4% | drop-trailing 13.6% |
| + decimal-tight numeric bounds | 0% | 3/418 = **0.72%** | x10 and drop-leading at 4.5% |

Every remaining admit is classified: 2 borderline matcher cases and 1 coincidence — the
mutated value genuinely stated elsewhere on the same span, a granularity limit of span-level
grounding, not a matcher bug. The challenge found a real hole and the fix lives in the gate:
`containsSurface` (claims.ts) now requires clean digit boundaries and refuses decimals glued
to further digits.

## Phase 1 on history: the audit run (2026-09-25)

`npm run support:audit` (results/support-audit.json): 200 synthetic historical cases with
realistic agent behavior injected (12% of due credits missed, 6% over-granted at 2x, 10%
goodwill grants on denials). The engine's comparison report — the artifact a lighthouse
buyer cares about: **$2,902 RECOVERY** (due but never issued), **$120 LEAKAGE** (issued but
not due), 175 agreements, 4 amount mismatches, 21 correctly escalated unknowns. The error
rates are assumptions; the format is the deliverable — a real export drops in unchanged.

## Phase 2: the second policy family — refunds (2026-09-25)

The refund family exercises conditions and exceptions the SLA family never had:
`purchased` / `return_requested` / `item_state` case claims over a pack of
`refund_window_days(tier, days)` and a **refundable-state whitelist**. Shapes
`refund_eligible` and `refund_deadline`: the whitelist IS the exception clause, and an EMPTY
whitelist is a provably incomplete pack -> Unknown (the amendment-gap discipline applied to
policy parameters). A real conditioned SOP (`benchmarks/support/refund-sop.md`) ships
hand-gated as `northwind-refund@2.1`; its section 3 (gift returns, price adjustments,
resellers) is deliberately unmodeled — the coverage review flags all of it. Gold: 160/160
across tidy / chat / noisy renderings, 0 noise flips, deleted item-state -> Unknown
(`npm run support:gold -- --family refund`).

## The surfaces: MCP and HTTP (2026-09-25)

Both servers frame the SAME three handlers (`src/support/tools.ts`), so they cannot drift:
`decide` (question -> proof, appended to the verdict store), `check_claim` (the gate, exposed
as a tool), `pack_coverage` (the unmodeled-clause review).

- `npm run support:mcp` — stdio JSON-RPC for agent tooling. Locked in by a smoke test that
  spawns the server and decides the demo case through the protocol.
- `npm run support:http` — plain JSON over node:http for help-desk webhooks:
  `GET /health`, `POST /decide`, `POST /check_claim`, `GET /pack_coverage?pack=`. A webhook
  posts the ticket plus "is a credit due for TCK-1042" and gets the proof back.

## Phase 4: the end-to-end loop (2026-09-25)

`npm run support:pipeline` — export in, decisions and the disputes report out:

    npm run support:pipeline -- --export <export.jsonl>        # a real or synthesized export
    npm run support:pipeline -- --synthesize 60 --write-export benchmarks/support/sample-export.jsonl

The export is the adapter seam a real CSV/Zendesk/Intercom converter targets — one JSON object
per line: `{ticket, chat?, recorded?}` where `recorded` is what the help desk actually did.
The pipeline decides every case (plan -> gate -> engine), appends every verdict to the
append-only store, and writes the report a team reads weekly. On the committed 60-case sample
export (engine-generated truth, human error injected): **59 proved, 1 Unknown, $1,088 RECOVERY
(due but never issued), $210 LEAKAGE (issued but not due), 31+13 agreements, 5 amount
mismatches** — every dispute row names the ticket and quotes the proof summary. A real export
replaces the file; none of this code changes.

## Still out of scope (the cut list, updated)

2B distillation (measure a second model by swapping — demonstrated twice now: r1->r2 on
compose, deepseek->llama3.1:8b here), ASP/s(CASP) enumeration (parked until a pilot asks),
PROV-O projection (the append-only store is its source; build when an audit tool needs it),
non-UTC calendars, and the gate challenge + audit on REAL support text — both harnesses are
waiting on the one thing only a lighthouse can provide: an export.
