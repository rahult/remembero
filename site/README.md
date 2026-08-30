# Remembero product site, labs, and playground

The hosted product site at `/`, two real-life labs under `/labs/`, and the
browser-contained proof playground at `/playground` for
[Remembero](https://github.com/rahult/remembero).

- `/labs/chat-memory` runs a real model → tool → SQLite → tool result → model loop twice
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
- `/guides/agent-harness` turns the lab into a portable MCP and agent-harness integration
  recipe with a bounded Query tool, validation, proof-aware synthesis, review-gated writes,
  and the executable agent database scorecard.

The agent lab uses the browser-safe Remembero TypeScript engine. The chat lab and playground
run SQLite 3.53.4 as WebAssembly with Remembero's C extension linked into the same binary. All three
experiences use fictional fixtures, perform no remote model calls or mutations, store no
browser data, and reset on refresh. Developers can load the optional Hermes 2 Pro Mistral 7B WebLLM
model on demand; compatible native Prompt API models are also supported. When neither is
ready, model text is deliberately scripted so the tool and policy boundary remains the
only changing variable.

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
