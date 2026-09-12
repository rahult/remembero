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
        → EntityResolver             MATCH | POSSIBLE_MATCH | NO_MATCH, thresholds are the knob
        → revocations supersede delegation end dates
        → ConflictDetector           same subject and category, different limit or polarity
        → claim acceptance           ACCEPTED | CONFLICTED | REJECTED, every transition logged
        → BeliefUpdater              SUPPORTED | DISPUTED | SUPERSEDED
        → DecisionEngine             ALLOW | DENY | UNKNOWN, with beliefs and evidence used
```

Everything above the boundary may be wrong and replaceable. Everything below is plain Python,
deterministic, and never consults a model. A POSSIBLE_MATCH never mutates state; a grant to a
POSSIBLE_MATCH makes any decision that depends on it UNKNOWN. An unresolved contradiction
makes a decision UNKNOWN only when the two values disagree about that request; the one
resolution rule applied is the one the document states itself (the body prevails over the
appendix).

## Run it

```sh
cd remembro-eval
python3 -m venv .venv && .venv/bin/pip install pydantic pytest
.venv/bin/python -m pytest -q                                   # 30 tests
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

| extractor                    | decisions | claim P / R | temporal | modality | contradictions | provenance | unjustified ALLOW | UNKNOWN |
| ---------------------------- | --------- | ----------- | -------- | -------- | -------------- | ---------- | ----------------- | ------- |
| gold claims (perfect)        | 17 / 17   | 100 / 100   | 100%     | 100%     | 2 / 2          | 100%       | 0                 | 3       |
| rules (regex, 0 model calls) | 17 / 17   | 90 / 90     | 78%      | 100%     | 2 / 2          | 100%       | 0                 | 3       |

The three UNKNOWNs are the ones the spec asks for: the body-versus-body limit conflict
(scenario 6), the `D. Smith` identity (scenario 7) and a currency mismatch (scenario 14). The
threshold sensitivity test shows the property the spec predicts: at a match threshold of 0.6
the initial merges with David Smith and scenario 7 becomes an ALLOW, which the test suite
asserts as the unsafe case.

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
