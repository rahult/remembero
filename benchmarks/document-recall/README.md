# Page-scale document benchmark (100 / 500 / 1000 pages)

What a local memory system does with a document nobody has chunked for it: can it find the page
that answers a question, and is what it stores true? Three tiers of real public documents, gold
answers with page-level evidence, and no labels leaked into the answering path.

## The corpus

Labels come from **XL-DocBench** (arXiv:2608.00036, HF `anonymous12123/XL-DocBench`, ungated,
2.7 MB). Each question carries a gold answer, a deterministic `verification_rule`, the evidence
**page numbers** and a verbatim quote; 199 of its 1,354 single-document questions are
deliberately unanswerable. The PDFs are not in the release — they are referenced by public URL,
so `select_and_fetch.py` downloads them and records each file's sha256.

```bash
mkdir -p .cache/xl-docbench && cd .cache/xl-docbench
for f in data/documents.jsonl data/qa_single_doc.jsonl evaluate.py; do
  curl -sL -O "https://huggingface.co/datasets/anonymous12123/XL-DocBench/resolve/main/$f"
done
cd - && python3 benchmarks/document-recall/select_and_fetch.py --per-tier 5 --attempts 10
```

A tier is a page **band**, and every document in it is a real document of about that length —
not a pile of short ones concatenated. The fetcher skips hosts that serve unauthorised copies of
copyrighted books, drops a download whose `pdfinfo` page count disagrees with the label, and drops
an answerable question that carries no evidence page (it could not score retrieval).

## Running it

```bash
# recall: retrieval → reader → score, per document, aggregated per tier
npm run eval:doc-recall -- \
  --reader-base-url http://127.0.0.1:8084/v1 --reader-api-key none \
  --reader-timeout-ms 600000 --max-tokens 512 --top-k 4 \
  --output results/document-recall-depth4.json \
  --predictions results/document-recall-depth4.predictions.jsonl

# extraction: the product writer over sampled pages, judged for support and coverage
npm run eval:doc-extract -- \
  --writer-base-url http://127.0.0.1:8084/v1 --writer-api-key none \
  --judge-model deepseek-chat --judge-base-url https://api.deepseek.com/v1 \
  --judge-api-key "$DEEPSEEK_API_KEY" --sample-pages 10 --output results/document-extraction.json
```

`scripts/serve-reader-v7.sh` serves the local reader. On a 16 GB machine the Q8 GGUF loads but
fails at compute (Metal wired-memory limit), so the local arm runs the Q4_K_M requantisation:

```bash
llama-quantize --allow-requantize \
  /Volumes/Atlas/models/rembero/reader-v7-gemma4-e4b-Q8_0.gguf \
  /Volumes/Atlas/models/rembero/reader-v7-gemma4-e4b-Q4_K_M.gguf Q4_K_M 6
```

## How it is scored, and why you can trust the number

Accuracy is **XL-DocBench's own rule**, ported to TypeScript in `src/evals/xl-docbench-score.ts`:
relaxed accuracy by answer type (numeric within 5%, gold contained in the prediction, or 0.8
normalised edit similarity), plus token F1 and ANLS. No judge model decides correctness, so the
numbers sit beside the benchmark's published baselines (SimpleDoc+GPT-5.4 44.0%, Claude Opus 4.6
40.5%, GPT-5.4 39.5%).

The port is checked rather than trusted. `--predictions` writes the file `evaluate.py` reads:

```bash
python3 .cache/xl-docbench/evaluate.py \
  --predictions results/document-recall-depth4.predictions.jsonl \
  --gold-files .cache/xl-docbench/qa_single_doc.jsonl --ignore-missing
```

Over 123 questions the port and `evaluate.py` agree on **every** question, and on all three
aggregates to two decimal places.

Retrieval is scored separately, in pages, which the benchmark itself does not do:

| number | meaning |
| --- | --- |
| page hit rate | at least one gold evidence page reached the reader |
| page recall | fraction of the gold evidence pages that reached it |
| window precision | fraction of the chosen windows that held a gold page |
| false answer rate | answered a question the document cannot answer |

`--judge-model` adds a second opinion on the answers; it is optional, and the rule is the
headline either way.

## Caveats a reader of the numbers should know

- **Retrieval, not reading, is what the tiers test.** One window is 4 pages or 12 KB; depth 4
  therefore shows the reader ~16 pages of a 1000-page document.
- **XL-DocBench is a 2026 preprint** with licence `other` and an anonymous HF namespace. Fine for
  internal measurement; get clarity before publishing scores, and mirror the labels.
- **Contamination cuts both ways.** These are famous public documents; a frontier model may know
  parts of them without reading. It flatters a no-retrieval baseline and understates retrieval.
- **The 1000-page tier is thin** — the band holds 21 documents and 85 questions in the whole
  dataset, so a few points there is noise. A tier may set `assemblePages` to concatenate its
  documents to an exact page count for a larger-sample second reading.
- **Extraction is sampled**, and every number carries the sample it came from.
