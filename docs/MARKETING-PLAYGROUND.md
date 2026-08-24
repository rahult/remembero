# Hosted marketing labs and playground

Remembero's product marketing site, real-life labs, and deterministic browser playground
are deployed at:

<http://remembero.rahultrikha.com/>

## What the public-facing experience proves

- `/labs/chat-memory` runs two real model tool loops over one shared browser-local SQLite
  database: a typed prepared-SQL call versus a Remembero Datalog query + proof call. The
  model calls, executed commands, tool results, raw output, and answer contract stay visible;
- `/labs/grounded-agent` keeps both model packets beside the proposed-action fact, gate
  query, active policy rule, proof chain, deterministic authorization, and measured gate time;
- `/playground` exposes current-browser SQLite + Wasm boot, rule + proof, SQL, and insert
  timings beside the linked extension identity;
- the playground scenario picker includes project follow-ups, team collaboration,
  dependency paths, customer-support escalation, release readiness, and team-based
  document access; each scenario resets to a small fictional SQLite fixture and exposes
  the exact Datalog question and proof for that workflow;
- `/guides/agent-harness` provides the portable MCP and model-tool-loop integration contract,
  including validation, proof-aware synthesis, fail-closed behavior, and review-gated writes;
- supported queries execute through the actual browser-safe Remembero Datalog engine;
- derived answers show their exact leaf claims, authored rule, and fictional source;
- the gift question returns an explicit non-answer and visibly separate related context;
- adding `prefers_gift(maya, notebook).` is session-only and changes that query to a
  directly supported answer; and
- reset restores the immutable fictional Atlas fixture.

No remote model API, API route, D1 database, R2 bucket, cookie, browser storage, or private
Remembero store participates. Developers may explicitly download the optional Hermes 2 Pro Mistral 7B
WebLLM weights; inference then runs locally and the browser caches those weights. A
ready browser-native language model may also generate the comparison proposals locally;
unsupported or zero-download sessions use explicitly labeled simulated prose.
The facts, rules, results, and proofs are always evaluated by the real browser-safe engine.
Node-backed store, source, SQLite, and server modules stay outside the hosted lab clients.

## Source

The deployable vinext project is under `site/`. The committed design references are:

- `docs/assets/rembero-marketing-hero-concept.png`
- `docs/assets/rembero-marketing-playground-concept.png`
- `docs/assets/rembero-marketing-lower-concept.png`
- `docs/assets/rembero-marketing-mobile-concept.png`
- `docs/assets/rembero-chat-memory-lab-concept.png`
- `docs/assets/rembero-grounded-agent-lab-concept.png`

The social card is generated specifically for the finished site at `site/public/og.png`.
