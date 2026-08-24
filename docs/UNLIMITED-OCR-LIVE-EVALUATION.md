# Unlimited-OCR live evaluation

Status: real public corpus prepared; authenticated official-provider inference is quota-blocked,
measured 21 August 2026 AEST.

## Integration surfaces

Remembero provides:

- `UnlimitedOcrClient` for a caller-operated OpenAI-compatible vLLM `/v1` endpoint; and
- `eval:ocr:live` for the official Baidu Hugging Face ZeroGPU Space.

The vLLM client implements the recipe's request contract: literal `<image>` prompt prefix,
`skip_special_tokens: false`, n-gram size 35, window 128 for one page and 1024 for multiple
pages. Grounding tokens are parsed into text blocks while retaining category and coordinates.

Official references:

- <https://recipes.vllm.ai/baidu/Unlimited-OCR>
- <https://github.com/baidu/Unlimited-OCR>
- <https://huggingface.co/baidu/Unlimited-OCR>

## Real corpus

The evaluator uses four pages rasterized from downloaded public PDFs:

| Page | Stress |
| --- | --- |
| IRS W-9 English, page 1 | Dense government form and instruction columns |
| IRS W-9 Spanish, page 1 | Multilingual dense form |
| MathBridge paper, page 4 | Two columns, formulas, and a four-column table |
| UN multilingualism publication, PDF page 20 | Editorial layout and embedded multilingual screenshots |

See `benchmarks/document-ocr/real/provenance.json` for source URLs and hashes, and
`benchmarks/document-ocr/real-ground-truth.json` for human-reviewed fields and order anchors.

## Run

```bash
npm run eval:ocr:live -- --check --json
```

`HF_TOKEN` is loaded from `.env`; it is never written to results. A failed provider run will
not replace an existing output unless `--allow-failed-output` is explicitly supplied.

For a caller-operated endpoint:

```ts
const client = new UnlimitedOcrClient({
  baseUrl: 'http://127.0.0.1:8000/v1',
  allowPrivateNetwork: true,
});
const parsed = await client.parse([{ bytes: pagePng, mimeType: 'image/png' }]);
```

Private or loopback endpoints require explicit opt-in. Public endpoints require HTTPS. Embedded
credentials, queries, fragments, and non-`/v1` paths are rejected before bytes or keys are sent.

## Gates

- Required-field recall: at least 95%.
- Reading-order anchor recall: at least 90%.
- Ordered anchor transitions: at least 90%.
- Required fields with grounding coordinates: at least 70%.
- Table classification must match.
- No provider or transport errors.

Provider failures are excluded from quality denominators. An operational failure cannot become
a misleading 0% or 100% OCR score.

## Current real-page attempt

The authenticated requests reached the official Baidu Space, but inference did not start. The
provider returned: ZeroGPU quota requested 90 seconds, available 0 seconds. The frozen result is
`docs/research/results/unlimited-ocr-real-v1-summary.json`.

| Evidence | Result |
| --- | ---: |
| Real pages attempted | 4 |
| Model completions | 0 |
| Provider errors | 4 |
| Model quality metrics | Not measured |

The UI reports this as **provider quota blocked** and does not display fabricated quality
percentages. The deterministic showcase remains runnable because it uses real downloaded PDFs,
their text layers, human-reviewed facts, and local proof—not prior synthetic OCR outputs.

## Evidence boundary

A future successful live run can replace the operational snapshot and populate measured field,
order, grounding, table, and latency results. Even then, OCR output remains parser evidence and
cannot enter deterministic proof without explicit review.
