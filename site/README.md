# Remembero product site, labs, and playground

The hosted product site at `/`, three labs under `/labs/`, and the
browser-contained proof playground at `/playground` for
[Remembero](https://github.com/rahult/remembero). Designed under the **Ledger**
system — tokens, components, and the no-served-models rule are documented in
[DESIGN.md](DESIGN.md).

**No trained model weights are hosted, served, or required by this site.** The
latest fine-tuned readers and writers stay research artifacts in the repository;
the site links to their measured runs instead of shipping them.

- `/labs/reading-recall` is the deterministic showcase of the reading pipeline:
  lexical retrieval and a re-rank stand-in, context tiering (abstracts for all,
  full text for the top two), and a computed-notes block **generated live by code
  in the visitor's browser** over a fixed fictional history. The two reader arms
  are stamped replays of a recorded paired run — same reader, same evidence, with
  and without the notes — which is the unit the research scores.
- `/labs/chat-memory` runs a model → tool → SQLite → tool result → model loop twice
  over one browser-local database, on four questions chosen so the lanes structurally
  diverge rather than tie: a recursive root-cause chain (steelmanned against
  `WITH RECURSIVE` — rows agree, only one lane carries a checkable proof), a
  contradictory write that SQL applies silently while the Remembero constraint refuses
  and rolls it back, a proven absence versus a NULL cell, and a why-not diagnosis that
  names each failing premise where SQL returns an empty set. Both calls, commands,
  results, raw Hermes 7B WebLLM output, and answer contract remain visible.
- `/labs/grounded-agent` keeps request facts, both model packets, the proposed action,
  deterministic gate rule, complete decision proof, and measured gate time visible together.
- `/playground` measures SQLite + Wasm boot, rule + proof, SQL, and insert operations in
  the current browser while keeping the extension build identity inspectable.
- `/#research` tells the reading-recall arc with a live-shaped computed-notes sample,
  the first paired proof (318 → 359 on the 500, judge named), and the delta ledger of
  measured increments (+29 computed notes, +10 thinking step trained, +17 turn-level
  retrieval, +13 TypeSafe re-rank) to reader v7's 425/500 against the teacher's 440.
- `/#models` states which models Remembero runs on and what they score — the writer (a 2.3B
  Gemma 4 E2B fine-tune served as a Q8_0 GGUF under llama.cpp; extraction 87/103, query 27/31,
  policy decisions 20/20 unseen with zero unjustified ALLOW) and the reader (a Gemma 4 E4B
  fine-tune distilled from GLM 5.3 Flash; LongMemEval 425/500 with computed notes against the
  teacher's 440 under one judge) — under an explicit "not served here" boundary. Numbers come
  from `docs/research/READER-STRUCTURE.md` and `docs/research/MODEL-COMPARISON.md` in the main
  repository; update both together.
- `/#examples` links the four executable examples with their numbers: the Remembro
  decisions pipeline (`remembro-eval/README.md`), the scenario exercises, the LongMemEval
  runs, and the extraction and agent-boundary benchmarks.
- `/guides/agent-harness` turns the lab into a portable MCP and agent-harness integration
  recipe with a bounded Query tool, validation, proof-aware synthesis, review-gated writes,
  and the executable agent database scorecard.

The reading-recall lab and the agent lab use the browser-safe Remembero TypeScript
engine. The chat lab and playground run SQLite 3.53.4 as WebAssembly with Remembero's
C extension linked into the same binary. All four experiences use fictional fixtures,
perform no remote model calls or mutations, store no browser data, and reset on
refresh. Developers can load the optional Hermes 2 Pro Mistral 7B WebLLM model on
demand (a third-party model cached by the visitor's browser); compatible native
Prompt API models are also supported. When neither is ready, model text is
deliberately scripted so the tool and policy boundary remains the only changing
variable.

Hermes uses WebLLM's native `tools` and forced `tool_choice` fields. The model emits an
assistant `tool_calls` message, the app validates and executes it, and the exact call plus
tool result become a visible final-answer prompt. No source rows are embedded before the call.

```bash
npm install
npm run dev
npm test
```

`npm run build:pages` creates the static artifact published by GitHub Pages at
[remembero.rahultrikha.com](http://remembero.rahultrikha.com/). D1, R2, and the former
ChatGPT Sites deployment are intentionally absent.
