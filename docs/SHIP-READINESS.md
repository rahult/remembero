# Ship readiness

Status: **ready for a local-first alpha with live Unlimited-OCR disabled as a quality claim**.

Remembero now has one tested local system across CLI, MCP, web, real-PDF document intelligence,
deterministic proof, and Memorg interoperability. `npm run ship:check` is the release gate.

## Golden path

```bash
npm install
npm run web
```

Open `http://127.0.0.1:4318/`. Guided memory and document questions are local and
deterministic. Configure `LLM_API_KEY` only for arbitrary natural-language writes or questions.
The selected model default is `anthropic/claude-sonnet-5`; override it with `LLM_MODEL`.

Installed-package surfaces:

```bash
remembero serve
remembero document-memorg > document-intelligence.memorg.json
remembero verify-document-memorg document-intelligence.memorg.json
remembero-web
```

## Current release evidence

- Full suite: 55 test files, 747 passed, 1 skipped after the final model-default change.
- Packed install smoke test: passed.
- Package policy: 2.99 MB observed tarball (8 MiB gate), no original PDFs, no Python bytecode/cache files, and
  asset-specific third-party notices included.
- Four real PDFs: 12/12 deterministic questions; 100% answer, source recall, proof grounding,
  abstention, and idempotency.
- Deterministic document questions: 1.152 ms average, zero model calls, zero tokens, zero provider
  cost on the measured machine.
- Memorg: 66/66 items stored and indexed in a clean Memorg 0.1.2 SQLite/FTS round trip.
- Dependency audit: zero known vulnerabilities.
- Browser: desktop/mobile render, interactions, download, and overflow checks passed with no
  fresh-tab console warnings or errors.

The machine-readable decision record is
`docs/research/results/product-ship-v1-summary.json`.

## Frontier-model checkpoint

All models ran temperature-zero through the same real Remembero ingestion/recall pipeline.
Recall has 26 cases and 100 distractor predicates; extraction has 15 exact mutation/safety
cases. Provider responses supplied token and cost values.

| Model | Recall | Recall tokens / cost / time | Extraction | Extraction tokens / cost / time |
| --- | ---: | --- | ---: | --- |
| Claude Sonnet 5 | 100% | 169,067 / $0.345182 / 145.4 s | 100% | 20,703 / $0.045646 / 54.0 s |
| Gemini 3.1 Pro Preview | 100% | 142,250 / $0.356040 / 147.1 s | 100% | 18,811 / $0.077842 / 62.7 s |
| GPT-5.4 | 100% | 110,721 / $0.227240 / 74.9 s | 93.3% | 13,321 / $0.035703 / 22.6 s |
| GPT-5.6 Luna | 96.2% | 120,454 / $0.020846 / 103.2 s | 100% | 13,507 / $0.003229 / 27.0 s |

Claude Sonnet 5 is the shipping default because it was the only tested model with 100% on both
current checkpoints and was cheaper/faster than Gemini. Luna remains the economy option; one
current recall case exhausted repair after invalid generated syntax, despite earlier 26/26 runs.

Raw results:

- `docs/research/results/frontier-recall-v1-summary.json`
- `docs/research/results/frontier-extraction-v1-summary.json`
- `docs/research/results/economy-luna-recall-v1-summary.json`
- `docs/research/results/economy-luna-extraction-v1-summary.json`
- `docs/research/results/document-showcase-real-v1-summary.json`

## Explicit release boundary

The official Baidu Unlimited-OCR transport is implemented, but its real-page run remains blocked
by ZeroGPU quota. The UI and release summary therefore show **provider quota blocked** and no OCR
quality percentage. Shipping the local alpha does not enable or claim live OCR quality.

The web server is loopback-only and is not a hosted multi-user deployment. Remote authentication,
tenant isolation, and a successful real-page OCR provider run are post-alpha gates.
