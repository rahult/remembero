# Remembero product site, labs, and playground

The hosted product site at `/`, the evidence page at `/research`, four labs under
`/labs/`, and the browser-contained proof playground at `/playground` for
[Remembero](https://github.com/rahult/remembero). Designed under the **Ledger**
system — tokens, components, and the no-served-models rule are documented in
[DESIGN.md](DESIGN.md).

The site is built for a first-time visitor with zero context and reads as a
progressive walk — **problem → idea → try → evidence**:

- `/` opens with the promise and one proof-carrying answer card, states the
  forgets/misremembers problem in plain language, then proves the idea with a
  **sixty-second live demo** (`app/first-proof-demo.tsx`): the visitor says two
  things, stores them as facts, compiles one rule, and asks one question — every
  step executed by Remembero's deterministic engine in the browser via
  `lib/first-proof-demo.ts`, with zero models. A **paragraph ladder** section then
  takes one realistic team-update paragraph apart in four layers of rising
  complexity — facts (replayed writer reading), dates resolved by live code,
  implied arithmetic and a conditional rule, and validity timelines — and the try
  section offers four labs plus the IDE showcase, a three-metric evidence teaser,
  and the models-translate-rules-decide boundary. Extracted knowledge renders
  English-first everywhere, with predicate syntax demoted to a small "machine
  form" line; visitor-facing copy carries no internal version jargon.
- `/research` carries the full measured story in plain language: what the small
  reader got wrong (with the failure list), the code-written notes fix with the
  first controlled before/after pair (318 → 359 of 500, grader named), the
  delta ledger of four measured improvements to 425/500 against the frontier
  teacher's 440, the two small open models ("the translator", "the answerer")
  under an explicit **not served here** boundary, and the worked examples.

**No trained model weights are hosted, served, or required by this site.** The
fine-tuned readers and writers stay research artifacts in the repository; the
site links to their measured runs instead of shipping them.

- `/labs/writer-reader` shows the whole loop on genuinely messy history: six
  months of one person's chat containing a three-deep supersession chain
  (dentist), a flip-back at a changed price (dashboard tool), an effective-dated
  handover, a contract that ends without renewal, relative dates, and one
  question whose answer was never said. Act 1 replays the writer's claims, then
  **code in the visitor's browser** resolves, dates, and supersedes them into
  validity timelines (current / superseded / ended). Act 2 asks four questions
  over that moving truth — retrieval scoring and structured-evidence notes are
  live code; both reader arms (raw chats vs structured evidence) are stamped
  replays, including an honest "not in memory" against a confabulated number.
- `/labs/reading-recall` is the deterministic showcase of the reading pipeline:
  lexical retrieval and a re-rank stand-in, context tiering (abstracts for all,
  full text for the top two), and a computed-notes block **generated live by code
  in the visitor's browser** over a fixed fictional history. The two reader arms
  are stamped replays of a recorded before/after run — same reader, same
  evidence, with and without the notes.
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
- `/guides/agent-harness` turns the lab into a portable MCP and agent-harness integration
  recipe with a bounded Query tool, validation, proof-aware synthesis, review-gated writes,
  and the executable agent database scorecard.

The reading-recall lab, the writer-reader lab, the first-proof demo, and the
agent lab use the browser-safe Remembero TypeScript engine or their own
deterministic fixture code. The chat lab and playground run SQLite 3.53.4 as
WebAssembly with Remembero's C extension linked into the same binary. All
experiences use fictional fixtures, perform no remote model calls or mutations,
store no browser data, and reset on refresh. Developers can load the optional Hermes 2 Pro Mistral 7B WebLLM model on
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
