# Compact deterministic evidence answer mode

Remembero 0.51 adds `answerMode: "evidence"` for people who want a readable local answer
and its trust basis without inspecting full explanation JSON or granting an LLM phrasing
authority.

## Use

```bash
remembero recall "Who are Rahul's colleagues?" --answer-mode evidence
remembero recall-explain "Who are Rahul's colleagues?" \
  --answer-mode evidence --proof-limit 4
```

MCP `recall` and `recall_explain` accept `answerMode: "evidence"`. Library callers use the
same `RecallOptions`; `evidenceRecallAnswer(...)` is also exported. Since 0.57 evidence is the
process default; set `REMBERO_RECALL_ANSWER_MODE=natural` to restore LLM phrasing.

## Rendering contract

Evidence mode performs the ordinary bounded retrieval with explanation enabled, then
renders each selected result locally:

- ordered projected bindings or `supported` for a true ground query;
- visible tentative row label;
- deduplicated grounded claims from the complete selected proof trees;
- authored rule numbers and canonical clauses;
- closed-world absence patterns;
- aggregate operator/input/value evidence;
- literal claims used through identity/trust projection; and
- durable namespace, operation ID, timestamp, trust/temporal metadata, and redacted source
  statement.

All sets are sorted deterministically. Requested alternative proofs are included and
deduplicated beneath the same result. Ground atoms use canonical Datalog serialization,
so quoted multi-word values remain unambiguous.

## Model and negative-answer behavior

The model is used only for the existing question-to-query translation. Evidence mode
forces explanation retrieval and never performs the final phrasing call.

Empty successful queries retain deterministic why-not summaries from v0.29. Structurally
unanswerable and schema-budget-exhausted statuses keep their honest local messages. Thus
evidence mode never asks a model to narrate either positive or negative evidence.

## Bounds and privacy

The renderer consumes only already-bounded explanation rows, proofs, aggregate
contributors, alternatives, and sources. The final text retains the 16 MiB output bound.

Source statements are the locally stored, already redacted provenance text. Evidence mode
is intended for the caller who requested recall; it does not send sources to another model
or service.
