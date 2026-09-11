# Semantic knowledge search

Remembero keeps structured query and proof local, deterministic, and free. For fuzzy
recommendation, preference, and advice questions, an opt-in semantic tool can rerank a
bounded lexical shortlist without making embeddings answer authority.

## Use it

```bash
export LLM_API_KEY=...
remembero semantic-search 'Can you recommend accessories for my camera setup?'
```

Agent harnesses can call `semantic_search_knowledge` over MCP. The result returns ranked
clauses, durable sources, the original lexical rank, cosine score, cache hits/misses, model,
provider tokens, and provider-reported cost.

After reviewed writes, move document work off the first user-facing search with:

```bash
remembero semantic-index --namespaces default --search-limit 100
```

Agent harnesses use `prepare_semantic_search`. Each call prepares at most 100 deterministic
documents and returns `nextCursor` when more remain; pass that cursor back as `after` until
status is `complete`. Re-running a prepared page reports cache hits and makes no provider
call.

Use this tool for recommendation/advice intent. Use `query`, `explain_query`,
`recall_explain`, or evidence answer mode when the answer must be logically established.
Similarity is retrieval evidence, never proof and never an abstention decision.

## Boundary

The semantic path:

1. runs deterministic local search first;
2. selects at most 100 candidates;
3. uses at most 16,384 source characters per candidate, split into at most ten overlapping
   2,048-character chunks;
4. rejects detected secrets before any network call;
5. permits only namespaces allowed by `REMBERO_LLM_ALLOWED_NAMESPACES`;
6. sends batches of at most 100 document chunks with at most three provider calls in flight;
7. ranks each candidate by its best chunk cosine similarity, breaking ties by lexical rank;
8. preserves a lexical leader only when it scores at least 120 and has a 1.5x margin; and
9. caches document vectors by model and content hash in memory and in a bounded derived
   cache under the memory root.

Preparation uses the same redaction, allowlist, model/content/chunk key, cache, and source window
as search. Documents sharing one provenance source are embedded once per batch. Preparation
does not write memory, change trust, infer facts, or establish proof.

The lexical guard prevents max-chunk overconfidence from demoting an unusually strong exact
match. It changed none of the LongMemEval development or held-out rankings, while preserving
an explicit Sony accessory preference over a semantically tempting camera-cake distractor.

The default model is `perplexity/pplx-embed-v1-0.6b` through the configured OpenAI-compatible
embedding endpoint. Override it with `REMBERO_EMBEDDING_MODEL` and the endpoint with
`REMBERO_EMBEDDING_BASE_URL`. `LLM_API_KEY` is used by default; `OPENROUTER_API_KEY` is an
accepted fallback.

## Model selection

The default was selected on the deterministic development split with the routing, chunks,
top-k, source safety, and lexical guard held fixed:

| Model                   | Dev Recall@5 | Dev MRR |     p95 |      Cost | Decision                                 |
| ----------------------- | -----------: | ------: | ------: | --------: | ---------------------------------------- |
| Perplexity 0.6B         |        86.7% |   75.0% |   9.2 s | $0.004485 | default                                  |
| Qwen3 8B                |        80.0% |   61.6% |  39.8 s | $0.011240 | rejected                                 |
| Perplexity 4B           |        80.0% |   66.7% |  12.2 s | $0.033635 | rejected                                 |
| NVIDIA Nemotron 1B free |      not run | not run | not run |        $0 | unavailable under account privacy policy |

The larger eligible models were less accurate, slower, and more expensive, so they were
stopped before held-out evaluation. The free endpoint was rejected rather than weakening
privacy settings. See the
[machine-readable matrix](research/results/semantic-model-matrix-v1-summary.json).

### Local embedding models (2026-09-11)

The same selection benchmark, rerun under today's routing (15 development and 14 held-out
preference questions routed) with open-weight models served by a local Ollama daemon through
`REMBERO_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1`, against the hosted default rerun the
same day:

| model                         | where        | dev R@5 | dev MRR | held-out R@5 | held-out MRR | p95 (Mac M-series) |
| ----------------------------- | ------------ | ------: | ------: | -----------: | -----------: | -----------------: |
| perplexity/pplx-embed-v1-0.6b | OpenRouter   |  100.0% |   83.9% |        86.7% |        75.6% |                3 s |
| **nomic-embed-text** (274 MB) | local Ollama |  100.0% |   78.0% |        86.7% |        75.6% |               29 s |
| qwen3-embedding:0.6b          | local Ollama |   86.7% |   65.2% |        86.7% |        66.7% |               75 s |
| bge-m3                        | local Ollama |       – |       – |            – |            – |     runner crashed |

nomic-embed-text ties the hosted model on held-out recall and MRR and is the local
replacement; the wall time is the laptop embedding 1.7M tokens of candidate chunks, not the
model, and disappears on a GPU or with the derived-vector cache warm. Qwen3-Embedding-0.6B is
behind on MRR. The Ollama runner intermittently dropped requests mid-batch (HTTP 400, "Post
…/tokenize: EOF"); the client now retries that as transient. Results in
`runs/local/semantic-*.json` were produced by `node dist/evals/run-longmemeval-semantic.js`
with `REMBERO_EMBEDDING_MODEL` set to each tag.

The first cache layer is a 2,000-entry in-process LRU. The second is
`.semantic-embeddings/`, a restart-safe derived cache containing only vectors, content/model
hashes, and integrity metadata—never source text. Files are written atomically with `0600`
permissions in a real `0700` directory. Corrupt, oversized, symlinked, or digest-mismatched
entries become misses and are recomputed. Changing source text or model changes the key.
The directory is bounded to 2,000 entries and may be deleted at any time; the journal,
program, and provenance remain the sole authority.

## Measured preference gate v1

The pinned LongMemEval-S policy routes only explicit recommendation/advice intent to
semantic retrieval and leaves every other question on local lexical search. Question IDs
are split deterministically by SHA-256 before policy selection.

| Preference metric | Development lexical | Development policy | Held-out lexical | Held-out policy |
| ----------------- | ------------------: | -----------------: | ---------------: | --------------: |
| Precision@5       |                8.0% |              17.3% |             9.3% |           12.0% |
| Recall@5          |               40.0% |              86.7% |            46.7% |           60.0% |
| MRR               |               30.6% |              75.0% |            16.1% |           47.2% |

Across all 30 preference questions, Recall@5 improves from 43.3% to 73.3% and MRR from
23.3% to 61.1%. The 22 routed requests used 2,266,050 provider tokens and cost $0.0090642,
or $0.000412 per routed question, while recomputing every candidate chunk vector.

A live MCP restart probe over two documents measured 744 ms on the initial request and
399 ms after closing and restarting the server. Both document vectors survived as cache
hits; provider input fell from 32 to 9 tokens and charged cost from $0.000000128 to
$0.000000036. These tiny timings are a boundary check, not a production latency claim.

An adversarial prewarm probe split a long preference into ten cached chunks during a 994 ms
maintenance call. After closing and restarting the MCP server, the first semantic query took
412 ms, hit all ten vectors, sent only the 9-token question for $0.000000036, and retained
the correct lexical leader over a semantically tempting distractor.

A larger end-to-end preparation check uses the first 20 development multi-session cases
under the v5 score gate. Cold user-turn p50/p95 was 20.8/27.3 seconds. Explicit preparation
moved document embedding into maintenance and reduced user-turn p50/p95 to 12.1/18.3
seconds; query embedding tokens fell from 1,836,242 to 249. Preparation consumed 2,196,997
tokens and raised total embedding cost about 19.7%, so this is a latency shift, not free
work. See the
[preparation evidence](research/results/semantic-preparation-v1-summary.json).

Run the reproducible live-provider evaluation after downloading LongMemEval-S:

```bash
npm run bench:longmemeval:download
npm run bench:longmemeval:semantic
npm run bench:longmemeval:answer -- --split dev --prepare-semantic
```

The machine-readable result is
[`research/results/semantic-preference-v1-summary.json`](research/results/semantic-preference-v1-summary.json).
OpenRouter documents batching, caching, cosine comparison, model selection, and its
OpenAI-compatible embeddings endpoint:
<https://openrouter.ai/docs/api/reference/embeddings>.

The later end-to-end answer gate expands intent detection to ordinary plural wording such
as “recommendations” and “suggestions,” plus implicit “could there be a reason” questions.
With top four answer-facing sessions it retrieved 100.0% of development preference evidence
and 86.7% held-out; judged answer accuracy was 86.7% and 93.3%, respectively. Across all 30
preference questions, Recall@4 was 93.3% and answer accuracy was 90.0%. This v2 policy was
locked before the held-out run and is recorded in the
[end-to-end result](research/results/longmemeval-answer-v1-summary.json).

Cold embedding batches were initially serial. Running up to three independent batches in
parallel preserved a 5/5 development pilot while reducing semantic retrieval p50/p95 from
the full run's 48.6/67.0 seconds to 8.4/9.0 seconds. Prepared caches still reduce a later
query to its question embedding only.

The first full-transcript multi-session experiment raised Recall but reduced answer accuracy,
so it was rejected. Role-aware top-five context changes that result. Under v5, semantic
reranking is enabled only when the local leader scores at most 315. It routes 115/133
multi-session questions, reaches 93.0% Recall and 76.7% answer accuracy, and saves 55
embedding calls versus global routing. Strong local leaders remain embedding-free. See the
[v5 end-to-end evidence](research/results/longmemeval-answer-v5-summary.json).

## Remaining limits

- Held-out Recall@4 is now 86.7% in the end-to-end policy, but the remaining misses still
  matter and semantic similarity cannot prove absence.
- Cold evaluation deliberately uses isolated corpora. MCP and CLI reuse the derived disk
  cache across processes, but first access still embeds shortlisted documents.
- The cache is populated by explicit preparation or search, never automatically during a
  memory mutation.
- Preparation is explicit by design; harnesses that skip it still pay document embedding on
  the first matching search.
- A lexical miss outside the 100-candidate shortlist cannot be recovered semantically.
- Embedding-provider pricing and routing can change; rely on returned usage, not this snapshot.
- Providers can revise a model behind a stable ID; change `REMBERO_EMBEDDING_MODEL` (or clear
  `.semantic-embeddings/`) before mixing vectors from a known model revision change.
