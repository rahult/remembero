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
        → amendments                 a later document supersedes an earlier one from its effective date
        → revocations supersede delegation end dates (or a personal authorisation, if no delegation is named)
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
.venv/bin/python -m pytest -q                                   # 74 tests
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
| r19 writer (our Gemma 4 E2B LoRA, zero-shot on this schema) | 11 / 17 | 21 / 60   | 78%      | 86%      | 0 / 2          | 0                 | 8       |
| GLM 5.3 Flash, 1,500-token cap, drops hidden (first run) | 15 / 17   | 44 / 70     | 89%      | 100%     | 0 / 2          | **1**             | 2       |
| GLM 5.3 Flash, 6,000-token cap, 2 unread spans recorded  | 12 / 17   | 37 / 80     | 89%      | 100%     | 1 / 2          | 0                 | 8       |
| GLM 5.3 Flash, one retry on truncation, 0 unread spans   | 17 / 17   | 36 / 80     | 89%      | 94%      | 1 / 2          | 0                 | 3       |
| **r21 writer** (r19 data + 2,221 claim spans, one LoRA round) | **17 / 17** | 67 / 90 | 100%    | 100%     | 2 / 2          | 0                 | 3       |

Provenance coverage is 100% on every row: an accepted claim always carries a located quote.

**The writer arm.** The fine-tuned fact extractor was never trained on this claim vocabulary,
and it behaves like it: of 59 candidates, 45 never crossed the boundary. 28 used a predicate
outside the six (`has_budget`, `requires_approval`, `retains_right`, `notifies`…), 12 gave a
subject kind the schema does not accept, 4 an object kind, 1 dropped a required field. The 10
claims that survived are all correct, so every decision made from them is justified. Before
the doubt rules below, the writer scored 15/17 with both misses on the safe side; once
rejected speech about a person counts as doubt (rule 5), six of its decisions become UNKNOWN,
because it said things like `retains_right(Alice…)` that the engine could not read. That is
the honest price of an extractor that does not speak the vocabulary, and it is the number the
retrained writer (r21, below) has to beat. Teaching the writer this vocabulary is a
training-data job, not an engine change.

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

5. *Unreadable speech about a person.* On the second document the writer described "Julian
   Ford ceased to hold the role" with an invented predicate `ceased_to_hold`; the schema dropped
   it, the role end vanished, and Julian was allowed after he left. A candidate the boundary
   rejected is now doubt about anyone named in its evidence, when what was rejected could have
   been a restriction: an unknown predicate, a negative polarity, a prohibition, a suspension
   or a revocation. A rejected *positive permission* is not doubt, because dropping a
   permission cannot create an ALLOW.

The residual failure class the boundary cannot see is an omission that leaves no trace: an
extractor that reads a passage and returns an empty array. Two arms that disagree about a span
is the V1 answer to that.

The three UNKNOWNs on the gold and rules rows are the ones the spec asks for: the
body-versus-body limit conflict (scenario 6), the `D. Smith` identity (scenario 7) and a
currency mismatch (scenario 14). The threshold sensitivity test shows the property the spec
predicts: at a match threshold of 0.6 the initial merges with David Smith and scenario 7
becomes an ALLOW, which the test suite asserts as the unsafe case.

## Teaching the writer the vocabulary (r21)

The r19 writer lost 45 of 59 candidates at the schema gate because it had never seen the six
predicates. r21 is the same Gemma 4 E2B LoRA recipe with 2,221 claim-extraction rows added to
r19's 27,853: 1,754 spans generated from templates that know the claim they encode (thirteen
span kinds, 150 boilerplate negatives, names disjoint from both fixtures) and 467 GLM 5.3 Flash
paraphrases of those spans, kept only when the labels still parse and ground in the rewritten
text (`src/remembro/training/`). One epoch, 75 minutes on an H100, held-out loss 0.022.

On fixture 1 the writer goes from 11/17 to 17/17: claim recall 60% → 90%, precision 21% → 67%,
temporal and modality 100%, both contradictions found, zero unjustified ALLOW. On fixture 2,
which it never saw a word of, 18/20 with zero unjustified ALLOW after one more boundary rule
(the `constraints.category` field is resolved against the register like any other mention, so
the string "none" the writer wrote on a revocation becomes null and the revocation applies to
every category). The two misses are safe-side and are training gaps, not engine gaps: it wrote
the unseen two-word role "Grants Manager" as "Manager", so the role was dropped and a standing
limit with it; and it read "lifted with effect from 1 September" as a suspension starting on
1 September. Neither pattern is in the generator. The regression benches held: extraction
85/103 (r19 86), query 25/31 (r19 26, r20 25).

## The second document

`fixtures/harbourview_delegations_v2.md` is a different organisation in a different register:
"is authorised to", "shall not", a not-for-profit with a Board of Trustees, three expenditure
categories (grants added) in NZD, five people including Julian Ford and Julia Ford so that
`J. Ford` is compatible with two register entries, a delegation that excludes two categories,
a temporary authorisation that is revoked, an open-ended suspension, a body-versus-body
conflict and an appendix that contradicts the body. `…_amendment1.md` is a second document
dated eight months later: it raises one limit, lowers another, ends one appointment and makes
another, and lifts the suspension. Twenty scenarios, six of them on either side of the
amendment's effective date.

The gold file lists its documents with effective dates; `ingest` with no document argument
reads them all. Three rules were added for it, each with a test: a claim from a later document
supersedes the same claim from an earlier one from the later document's effective date (a
supersession, logged, never a contradiction); a later restatement that only adds an end date
closes the earlier open claim (a role that ceased, a suspension lifted); a revocation that
names no delegation ends the personal authorisation of the person it names. Grounding accepts
an end date that is the day before a quoted date, because "lifted with effect from 1 September"
ends on 31 August. Roles and categories now resolve from the gold register rather than a
vocabulary in code. Resolution carries every candidate within the possible threshold, so a
grant to `J. Ford` makes decisions about *both* Fords UNKNOWN.

| extractor                                   | decisions | claim P / R | contradictions | unjustified ALLOW | UNKNOWN |
| ------------------------------------------- | --------- | ----------- | -------------- | ----------------- | ------- |
| gold claims (perfect)                       | 20 / 20   | 100 / 100   | 2 / 2          | 0                 | 4       |
| rules (regex written for fixture 1)         | 9 / 20    | 0 / 0       | 0 / 2          | 0                 | 1       |
| r19 writer (zero-shot)                      | 9 / 20    | 21 / 36     | 0 / 2          | 0                 | 13      |
| GLM 5.3 Flash                               | 20 / 20   | 35 / 61     | 1 / 2          | 0                 | 4       |
| **r21 writer**                              | **18 / 20** | 53 / 57   | 1 / 2          | 0                 | 4       |

The regex arm extracts nothing here: it matched fixture 1's phrasing, not English. That is the
result the second document was written to produce, and it is why the neural arm exists. Every
rule learned on fixture 1 held unchanged on fixture 2, and GLM Flash's first replay scored
17/20 with all three misses on the safe side. Two more deterministic rules took it to 20/20
without touching the snapshot: the resolved object ("grant expenditure", quoted) is
authoritative over the extractor's `category` field (the prompt's enum had no grant category,
so GLM wrote "expenditure"); and a `suspended` claim with negative polarity is a lift, which
ends the open suspension the day before, and is never itself a suspension.

```sh
G=fixtures/harbourview_delegations_v2.gold.json
PYTHONPATH=src .venv/bin/python -m remembro.cli --gold $G evaluate
PYTHONPATH=src .venv/bin/python -m remembro.cli --gold $G --run-name glm-flash-hv ingest --extractor llm \
  --model glm-5.3-flash:cloud --base-url http://127.0.0.1:11434/v1 --out fixtures/runs/glm-flash-hv.claims.json
PYTHONPATH=src .venv/bin/python -m remembro.cli --gold $G --run-name glm-flash-hv evaluate
```

## What is deliberately not here

Docling, a graph database, a rule engine, a UI, other languages. The
Markdown parser sits behind the `DocumentParser` protocol; the claim vocabulary is six
predicates; storage is SQLite (`evaluate --db state.sqlite`). The amendment document (V1) is in; V2 is an email, and a queue of documents arriving out of
order is the next transaction-time case.

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
