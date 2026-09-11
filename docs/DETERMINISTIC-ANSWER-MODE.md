# Deterministic positive answer mode

Remembero 0.30 can render successful natural-language recall locally from exact Datalog
bindings. Query translation may still use the configured model, but the model no longer
needs authority over the final wording when deterministic mode is selected.

## Enable

```bash
remembero recall 'Who works at Acme?' --answer-mode deterministic
remembero recall-explain 'What may Mira status be?' \
  --trust include_tentative --answer-mode deterministic

export REMBERO_RECALL_ANSWER_MODE=deterministic
```

Library callers use `RecallOptions.answerMode` or the `PipelineDeps.recallAnswerMode`
default. MCP `recall` and `recall_explain` accept `answerMode`. A per-call value overrides
the server/environment default.

`evidence` is the default since 0.57; `natural` phrasing is opt-in (`REMBERO_RECALL_ANSWER_MODE=natural`
or a per-call `answerMode`). Before 0.57 `natural` was the default.

## Rendering contract

`deterministicRecallAnswer(...)` preserves returned row order and binding insertion order:

```text
The query project(atlas) is supported.

Result for works_at(maya, Company): Company = acme.

Results for works_at(Person, acme):
1. Person = rahul
2. Person = maya

Tentative result for status(mira, State): State = paused.
```

Multi-row tentative results prefix the corresponding row with `[tentative]`. Values are
the same canonical serialized terms returned in `bindings`; rendering does not apply
locale, title case, entity labels, or model interpretation. Aggregate output variables
are ordinary bindings and use the same format.

The answer is checked against the existing 16 MiB output bound before return.

## Model-call boundary

Successful natural mode retains two calls: validated query generation and grounded answer
phrasing. Successful deterministic mode uses query generation only. Empty queries retain
their one translation-review fallback; final no-match answers remain local under v0.29 in
both modes. Unanswerable, schema-budget, and empty-store messages are already local.

Proofs, source provenance, graph construction, schema pruning/widening, query review,
identity, trust, recorded snapshots, and safety checks are unchanged. Deterministic answer
mode does not make model-generated query translation deterministic; callers needing no
model at all should use raw `query`, `explain`, `why-not`, `what-if`, or `repair`.
