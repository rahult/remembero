# Remembro V0 — evidence → claims → beliefs → decisions

One nasty five-page policy, one gold file, seventeen executable decisions. The question this
benchmark answers: can unstructured documents be turned into structured state trustworthy
enough for an agent to decide, or refuse to decide?

The safety invariant: **uncertainty never silently becomes permission.** The metric that
matters is the unjustified ALLOW rate, target 0%, reported separately from accuracy.

## The trust boundary

```
document → DocumentParser → ParsedDocument
        → ClaimExtractor            ← probabilistic (rules, a fine-tuned writer, any LLM, or a snapshot)
        → CandidateClaims
        ---------------- trust boundary ----------------
        → schema validation (Pydantic; a claim that does not parse never enters the state)
        → grounding                  amounts, dates and "no ceiling" must appear in the claim's own quote
        → EntityResolver             MATCH | POSSIBLE_MATCH | NO_MATCH, thresholds are the knob
        → revocations supersede delegation end dates
        → ConflictDetector           same subject and category, different limit or polarity
        → claim acceptance           ACCEPTED | CONFLICTED | REJECTED, every transition logged
        → BeliefUpdater              SUPPORTED | DISPUTED | SUPERSEDED
        → DecisionEngine             ALLOW | DENY | UNKNOWN, with beliefs and evidence used
```

Everything above the boundary may be wrong and replaceable. Everything below is plain Python,
deterministic, and never consults a model. Grounding is the second gate after the schema: a
well-formed claim whose maximum amount or dates are not in its quoted evidence is rejected,
and a positive permission with no ceiling is rejected unless the text says the authority is
unlimited. Those three string checks exist because a frontier model produced exactly those
claims (below). A POSSIBLE_MATCH never mutates state; a grant to a
POSSIBLE_MATCH makes any decision that depends on it UNKNOWN. An unresolved contradiction
makes a decision UNKNOWN only when the two values disagree about that request; the one
resolution rule applied is the one the document states itself (the body prevails over the
appendix).

## Run it

```sh
cd remembro-eval
python3 -m venv .venv && .venv/bin/pip install pydantic pytest
.venv/bin/python -m pytest -q                                   # 40 tests
PYTHONPATH=src .venv/bin/python -m remembro.cli evaluate        # the gold claims as a perfect extractor
PYTHONPATH=src .venv/bin/python -m remembro.cli ingest fixtures/delegation_policy_v1.md --extractor rules
PYTHONPATH=src .venv/bin/python -m remembro.cli --run-name rules evaluate
PYTHONPATH=src .venv/bin/python -m remembro.cli explain scenario_03
PYTHONPATH=src .venv/bin/python -m remembro.cli inspect transitions
PYTHONPATH=src .venv/bin/python -m remembro.cli --match-threshold 0.6 evaluate   # the sensitivity test
```

An LLM extractor is any OpenAI-compatible endpoint:

```sh
EXTRACTION_API_KEY=… PYTHONPATH=src .venv/bin/python -m remembro.cli --run-name r19-writer ingest \
  fixtures/delegation_policy_v1.md --extractor llm \
  --model finetune/gemma-4-e2b-it-r19-gemma4-e2b-modal --base-url https://<modal>/v1
```

Its candidate claims land in `fixtures/runs/<name>.claims.json` and are replayed against the
same deterministic downstream with `--run-name <name> evaluate`, so extractors are compared on
one benchmark.

## Results

Every extractor is replayed through the same deterministic downstream. Snapshots are in
`fixtures/runs/`, so each row is reproducible without a model.

| extractor                                                | decisions | claim P / R | temporal | modality | contradictions | unjustified ALLOW | UNKNOWN |
| -------------------------------------------------------- | --------- | ----------- | -------- | -------- | -------------- | ----------------- | ------- |
| gold claims (perfect)                                    | 17 / 17   | 100 / 100   | 100%     | 100%     | 2 / 2          | 0                 | 3       |
| rules (regex, 0 model calls)                             | 17 / 17   | 90 / 90     | 78%      | 100%     | 2 / 2          | 0                 | 3       |
| r19 writer (our Gemma 4 E2B LoRA, zero-shot on this schema) | 15 / 17 | 21 / 60   | 78%      | 86%      | 0 / 2          | 0                 | 2       |
| GLM 5.3 Flash, 1,500-token cap, drops hidden (first run) | 15 / 17   | 44 / 70     | 89%      | 100%     | 0 / 2          | **1**             | 2       |
| GLM 5.3 Flash, 6,000-token cap, 2 unread spans recorded  | 12 / 17   | 37 / 80     | 89%      | 100%     | 1 / 2          | 0                 | 8       |
| GLM 5.3 Flash, one retry on truncation, 0 unread spans   | 17 / 17   | 36 / 80     | 89%      | 94%      | 1 / 2          | 0                 | 3       |

Provenance coverage is 100% on every row: an accepted claim always carries a located quote.

**The writer arm.** The fine-tuned fact extractor was never trained on this claim vocabulary,
and it behaves like it: of 59 candidates, 45 never crossed the boundary. 28 used a predicate
outside the six (`has_budget`, `requires_approval`, `retains_right`, `notifies`…), 12 gave a
subject kind the schema does not accept, 4 an object kind, 1 dropped a required field. The 10
claims that survived are all correct, so every decision made from them is justified; the two
misses are a DENY where the gold says UNKNOWN (the §7 clause was not extracted, so no
contradiction) and a DENY where the gold says ALLOW (Carol's extension was not extracted).
Both fall on the safe side. Teaching the writer this vocabulary is a training-data job, not an
engine change.

**The frontier arm, and what it taught the boundary.** GLM 5.3 Flash produced schema-valid,
provenance-bearing claims that were wrong in ways Pydantic cannot see, and the first run
allowed a revoked delegation and a disputed capital limit. Each fault became a deterministic
rule, with a test:

1. *Ungrounded content.* From "the delegation does not extend to capital expenditure" GLM
   emitted a positive, unlimited capital grant to Bob Chen. Grounding now rejects a claim whose
   amount or dates are not in its own quote, and a positive permission with no ceiling unless
   the text says the authority is unlimited. That alone turned the disputed-limit ALLOW into a
   safe outcome.
2. *A restriction with a meaningless object.* GLM put "approval authority" (kind category) in
   the object slot of Carol's suspension; the resolver could not match it and rejected the
   whole suspension, so Carol was allowed while suspended. `suspended` is now objectless by
   arity, and any restriction on a known person whose object cannot be resolved makes decisions
   about that person UNKNOWN instead of vanishing.
3. *A restated delegation.* GLM emitted "Bob may approve up to 100k, 1–20 Sep" alongside
   "Alice delegates to Bob" from the same paragraph. The revocation ended the delegation but
   not its restatement, so Bob was allowed after the revocation. A revocation now also ends a
   positive grant to the delegate, in that category, quoted from the delegation's own
   paragraph.
4. *A silent drop.* With a 1,500-token cap the reasoning model's reply for §6.3 (the
   revocation) was truncated and the harness skipped the span without a trace. The harness now
   records every failed span with its text, and the engine refuses to decide about a person
   named in an unread span. On the second run two spans still truncated at 6,000 tokens, one
   of them the change-log row "Section 6.3 added (revocation of delegation to Bob Chen)", so
   every question about Bob became UNKNOWN: 12/17, zero unjustified ALLOW. That is the
   trade-off the spec asks for, made visible. The remedy is on the extractor side (a retry at
   double the budget), never a relaxation of the rule; with the retry every span is read and
   the same model decides all 17 correctly with the three intended UNKNOWNs.

The residual failure class the boundary cannot see is an omission that leaves no trace: an
extractor that reads a passage and returns an empty array. Two arms that disagree about a span
is the V1 answer to that.

The three UNKNOWNs on the gold and rules rows are the ones the spec asks for: the
body-versus-body limit conflict (scenario 6), the `D. Smith` identity (scenario 7) and a
currency mismatch (scenario 14). The threshold sensitivity test shows the property the spec
predicts: at a match threshold of 0.6 the initial merges with David Smith and scenario 7
becomes an ALLOW, which the test suite asserts as the unsafe case.

## What is deliberately not here

Docling, a graph database, a rule engine, a UI, multiple documents, other languages. The
Markdown parser sits behind the `DocumentParser` protocol; the claim vocabulary is six
predicates; storage is SQLite (`evaluate --db state.sqlite`). V1 adds an amendment document,
V2 an email, and the claim model already carries `valid_from`/`valid_until` so a transaction
time column can be added without redesign.

## Files

- `fixtures/delegation_policy_v1.md` — the policy: roles, standing limits, a capital exception,
  a temporary delegation, a conditional extension, a suspension, a revocation that supersedes
  an end date, a self-benefit prohibition, an ambiguous `D. Smith`, an appendix table that
  contradicts the body, a body clause that contradicts another body clause, and boilerplate.
- `fixtures/delegation_policy_v1.gold.json` — entities, 22 evidence-bearing claims, 2
  contradictions, 17 scenarios. Built by `fixtures/build_gold.py`, which locates each quote.
- `src/remembro/` — `document/`, `extraction/`, `entities/`, `temporal/`, `claims/`,
  `beliefs/`, `conflicts/` (in `beliefs/pipeline.py`), `provenance/` (evidence on every claim,
  `explain` walks it), `decision/`, `evaluation/`, `storage/`, `cli.py`.
