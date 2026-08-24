# Document intelligence showcase evaluation

Status: executable real-PDF prototype gate, measured 21 August 2026 AEST.

The showcase evaluates a complete evidence-to-decision path:

```text
downloaded PDF bytes
  -> hashed rendered page
  -> extracted text regions
  -> human review
  -> accepted facts + proposed claims
  -> deterministic recall and rules
  -> answer + proof
  -> exact page region
```

It is deliberately more than retrieval. Supported questions must return accepted proof leaves
and reviewed rules; questions backed only by raw or proposed evidence must abstain.

## Run

```bash
npm run eval:documents
```

The command exits non-zero if parsing, expected answer/status, source-region recall, proof
grounding, abstention, or idempotent re-parsing drifts.

## Real corpus

| Source document | Selected page | Reasoning proof | Abstention boundary |
| --- | ---: | --- | --- |
| IRS Form W-9, English | 1 | Safe recipient and TIN/name match | Filing deadline |
| IRS Form W-9, Spanish | 1 | Recipient and TIN/name concordance | Complete legal effect |
| MathBridge arXiv paper | 4 | Dataset scale and relative field limits | Training cost |
| UN *Why It Matters: Multilingualism* | 20 | Daily Hindi coverage and named outreach languages | Future coverage guarantee |

Every source URL, original PDF hash, byte size, page count, rendered-page hash, render setting,
and rights note is recorded in `benchmarks/document-ocr/real/provenance.json`. The UI displays
the actual rendered page under selectable evidence coordinates.

## Current deterministic result

| Metric | Result |
| --- | ---: |
| Documents | 4 / 4 passed |
| Questions | 12 / 12 passed |
| Parse coverage | 100% |
| Expected answer and status accuracy | 100% |
| Expected source-region recall | 100% |
| Proof grounding | 100% |
| Correct abstention | 4 / 4 |
| Idempotent re-parse | 4 / 4 |

Timings are displayed by the executable report and remain machine diagnostics, not production
latency claims. These scores prove the reviewed vertical slice, not general OCR accuracy.

## Authority boundary

- Downloaded bytes and their hashes establish source identity.
- PDF text-layer extraction and coordinates are evidence, not authority.
- Proposed claims never enter ordinary proof.
- Human-accepted facts and reviewed deterministic rules are the only proof-bearing inputs.
- Every accepted proof leaf must resolve to a selected source-page region.
- Re-running the parse is idempotent within a document-specific namespace.

The live model boundary and its current provider-quota status are documented in
[UNLIMITED-OCR-LIVE-EVALUATION.md](UNLIMITED-OCR-LIVE-EVALUATION.md).
