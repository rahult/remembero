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

## Deliberately out of scope (the cut list)

2B distillation (measure a second model by swapping, not training), ASP/s(CASP) enumeration
(parked until a pilot asks), PROV-O projection and MCP surface (integration points, built when
there is a conversation), refund family (second pack, after SLA proves the pattern). The gate
challenge set on real customer text — the kill criterion for false admits and over-rejection —
is the next measurement, and it needs a lighthouse customer's tickets to be worth running.
