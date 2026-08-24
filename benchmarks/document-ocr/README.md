# Real-PDF OCR evaluation corpus

This corpus contains four public, source-attributed PDFs. No synthetic document is part of the
benchmark or showcase.

| Document | Publisher | Selected page | Layout stress |
| --- | --- | ---: | --- |
| Form W-9 (English) | Internal Revenue Service | 1 | Dense form, boxes, labels, two-column instructions |
| Form W-9 (Spanish) | Internal Revenue Service | 1 | Multilingual dense form |
| MathBridge paper | arXiv | 4 | Two-column paper, formulas, table |
| *Why It Matters: Multilingualism* | United Nations | 20 | Editorial layout, embedded screenshots, linked text |

`real/provenance.json` records the publisher URL, retrieval time, original PDF hash, byte size,
page count, selected page, render settings, rendered PNG hash, and rights note. The original PDFs
are retained under `real/`; representative 150-DPI pages are under `real/pages/`.

The rendered pages were produced reproducibly with:

```bash
pdftoppm -f 1 -l 1 -singlefile -png -r 150 real/irs-form-w9-en.pdf real/pages/irs-form-w9-en-p1
pdftoppm -f 1 -l 1 -singlefile -png -r 150 real/irs-form-w9-es.pdf real/pages/irs-form-w9-es-p1
pdftoppm -f 4 -l 4 -singlefile -png -r 150 real/mathbridge-paper.pdf real/pages/mathbridge-paper-p4
pdftoppm -f 20 -l 20 -singlefile -png -r 150 real/un-multilingualism.pdf real/pages/un-multilingualism-p20
```

`real-ground-truth.json` contains human-reviewed required fields and reading-order anchors. It
does not contain model-generated answers. Run the authenticated adapter with:

```bash
npm run eval:ocr:live -- --json --check
```

The current frozen attempt is
`docs/research/results/unlimited-ocr-real-v1-summary.json`. The official provider accepted the
authenticated requests but rejected inference because the available ZeroGPU quota was zero.
That file is operational evidence, not an OCR quality score.

These files are for local, non-commercial evaluation. Follow each source's own terms before
redistributing the originals.
