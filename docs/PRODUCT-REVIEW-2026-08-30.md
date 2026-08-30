# Remembero — Daily-Driver Product Review

*2026-08-30 · Based on a full codebase, docs, benchmark, and live-store audit (v0.55.0, 44.6k LOC, 766 tests)*

> **Status update (same day):** Phases A–E below are implemented. Fixed: auto-capture
> transcript windowing (verified live — first real capture happened mid-session; the
> extraction also hallucinated a fact from assistant text, which was pruned via `review`
> and remains a watch item), engine p95 warmup gate, test timeouts, README model claim,
> journal append cost (stat-validated cache), Luhn-gated card detection. Added:
> `session-brief` + SessionStart hook, `serve --profile core` (12 tools),
> `serve -n <namespace>` server default namespace, `remembero init` one-command setup,
> real-memory-first web console (`--demo` for the sandbox), `backup`/`restore`, zero-key
> guidance in tool descriptions, and a README quickstart.

---

## Verdict in one paragraph

Remembero is a technically excellent product that nobody uses — including you. Two weeks
produced 55 releases, 57 completed roadmap phases, 43 MCP tools, a native SQLite
extension, science-grade benchmarks (LongMemEval 83.2% at $0.0009/question), and a
beautiful Wasm playground. Meanwhile your live store at `~/.rembero/memory/` still
contains only the 7 demo facts from build night (Aug 16), and the auto-capture hook —
installed and firing this very morning — has **never captured a single fact** because of a
transcript-windowing bug. The project has been in *evidence-production mode* when it needs
to be in *adoption mode*. The core is daily-driver-capable today; what blocks it is
packaging, defaults, and one real bug. Roughly 5–7 focused days close the gap.

---

## 1. What is working well

**The engine and store are genuinely solid.**
- Zero-dep Datalog engine with stratified negation, aggregation, arithmetic, proofs;
  deterministic and honest (`answered` / `no_match` / `unanswerable` + why-not evidence).
- Storage discipline is top-decile: `O_EXCL` locking with dev/inode-checked release,
  crash-recoverable rename protocol, append-only journal with SHA-addressed rotation,
  human-editable `.dl` files. Symlink refusal, `0o600` modes, loopback-only web.
- Strict TS, zero TODO/FIXME across 44.6k lines, 766 tests mapping ~1:1 onto modules,
  CI on Node 20/22/24 + Windows install check + million-fact scale gate.

**The hot path is already fast and cheap where it matters.**
- Only 5 of 43 MCP tools touch an LLM; everything else is pure local Datalog (MCP explain
  round trip ≈ 8 ms). `remember` is a single hands-free LLM call ($0.0002); recall is
  2 calls best case with honest fallbacks. Empty store short-circuits with zero calls.

**The evidence discipline is rare and valuable.**
- Every claim has an executable gate and an explicit "what this does not prove" boundary.
  Negative results are published. The LongMemEval v1→v5 arc (75.4% → 83.2%, cost cut 68%)
  is defensible, sealed-split work. Keep this culture — it is the moat for the
  "proof-carrying memory" positioning.

**The marketing site is the best positioning artifact.**
- Real SQLite Wasm + the C extension in the browser, no fake metrics, mechanism shown
  rather than asserted. This is done; it needs no more investment right now.

---

## 2. What needs to be fixed

Priority-ordered. Items 1–3 are the reasons the tool has zero real memories.

| # | Problem | Evidence | Fix shape |
|---|---------|----------|-----------|
| 1 | **Auto-capture captures nothing in real sessions.** The byte-bounded transcript tail (`src/autocapture/transcript.ts:243-297`, ~192KB read window → 24KB message tail) is swamped by tool output in agentic sessions, so user turns never appear → every attempt journals `skipped: no_user_text` (see `~/.rembero/memory/journal.log`, two skips this morning). | Verified live today | Select messages by **role**, not byte position: scan backward for the last N user messages; always include the current session's user turns. |
| 2 | **No recall at session start.** Capture is Stop-hook-only; nothing ever injects remembered context into a new session. Memory that is written but never read is a write-only log. | `src/autocapture/hooks.ts` targets only `hooks.Stop` | Add a `SessionStart` hook that emits a bounded, deterministic context brief (no LLM call needed — `list_memories`/`query` are free). |
| 3 | **43 unfilterable MCP tools.** Permanent context tax on every session; 7 semantic-ledger tools (product release governance, not memory versioning) actively confuse agents and error out when unconfigured (`src/mcp/server.ts:531`). | `src/mcp/server.ts:538-2347` | Tool profiles: `serve --profile core` exposing ~11 tools (Clusters A+B below) as default; `--profile full` for knowledge engineering. |
| 4 | **Every write rewrites the whole journal.** `appendJournalUnlocked` (`src/store/store.ts:1180`) reads, JSON-parses, re-hashes, and rewrites the full journal (up to 16 MB) synchronously per fact — blocks the MCP event loop and stresses the 2 s lock timeout. | Agent-measured | True `appendFileSync` with a cached tail digest. |
| 5 | **The suite isn't green.** 8/766 failures: 7 are subprocess tests tripping vitest's default 5 s timeout (no `testTimeout` in `vitest.config.ts`); 1 is real — the engine's own p95 gate failing 5× (125.6 ms vs 25 ms budget, `tests/memory-stack-evals.test.ts:76`). | Suite run 2026-08-30 | Set `testTimeout`; then profile the engine regression against the gate before trusting any perf claim. |
| 6 | **README contradicts the code on the default model.** README line 8 says GPT-5.6 Luna; the actual default is `anthropic/claude-sonnet-5` (`src/llm/client.ts:49`) — a ~16× recall-cost difference per your own frontier table. | `docs/SHIP-READINESS.md` | One-line fix; decide the default deliberately (see §5). |
| 7 | **Secret-scan false positives can wedge a namespace.** Any 13–19 digit run (build IDs, timestamps) trips `src/safety.ts:6-13`; the *schema summary* is scanned too, so one poisoned stored fact blocks all future LLM writes to that namespace. | Agent-verified | Luhn-check card-number candidates; scope schema scanning to literal values, not summaries; give the error a remediation hint. |
| 8 | Smaller: no `fsync` (survives process crash, not power loss); `export`/`import` exist (`src/cli.ts:2217-2228`) but appear in no doc; native recursive queries can block ~20 s before the step bound trips; no automated migration despite 54 migration guides. | | Batch into a hardening pass. |

---

## 3. What's missing from MVP

The original MVP bar (`docs/ROADMAP.md`) was right: *"a stranger on a fresh machine gets
working logical memory in Claude Code in under 5 minutes."* v0.1 arguably met it. The
55 releases since served a different persona — the technical evaluator — and the
daily-driver persona got nothing new. Missing, in order of pain:

1. **A read loop.** Session-start context injection (fix #2). Without it the agent must
   *choose* to call `recall`, and it mostly won't.
2. **A working capture loop.** Fix #1.
3. **Project scoping.** Namespaces are flat and global; every project writes to
   `default.dl` unless each CLAUDE.md is hand-edited. Need cwd→namespace derivation with
   a personal namespace that is always in scope.
4. **A memory inspector you'll actually open.** The web console defaults to a demo
   sandbox (`.rembero-web/`, auto-seeded fictional data); pointing it at real memory
   takes three undocumented env vars. Invert the default: real memory first, demo behind
   a flag.
5. **One-command onboarding.** `remembero init` should: register the MCP server with the
   core profile, install both hooks, write the CLAUDE.md snippet, and set a recommended
   env profile — including `REMBERO_VALID_TIME_MODE=archive` (today's default `delete`
   destroys the history the temporal features were built to preserve).
6. **A manual.** 123 docs and no quickstart. The README opens with benchmark deltas;
   `Install` is at line 119. Need: 30-line quickstart, "start with these four commands,"
   latency expectations (0.03 ms structured vs 12–27 s semantic), backup/restore/sync
   page (a git repo in `~/.rembero` is a fine v1 answer — say so).
7. **A zero-key mode** (see vision). The two headline tools require a paid OpenRouter key
   to run Sonnet *inside* a session already powered by Claude.

---

## 4. Updated product vision

> **Remembero is the memory your coding agent actually uses every session — and can
> prove.** Facts, rules, and constraints live in readable local files; recall is
> deterministic inference with citations; models translate at the edges but never decide.

Three deliberate shifts from the current `PRODUCT.md` (which describes the *showcase*,
not the product):

**Shift 1 — Primary persona is you.** A developer running Claude Code daily who wants
durable, inspectable, correctable memory across sessions and projects. The evaluator
persona stays, but is served by the (finished) site and research docs, not by new tools.

**Shift 2 — The loop is the product.** capture → recall-at-start → correct/prune →
accumulate. Every feature is judged by whether it strengthens this loop. The 32
knowledge-engineering tools remain as the power surface behind a profile flag; they stop
taxing the default experience.

**Shift 3 — The calling agent is the translator.** Claude Code *is* a frontier model
already; requiring a second paid LLM to translate NL→Datalog is architecturally redundant
for the MCP use case. Well-written tool descriptions can teach the client agent to call
the LLM-free `query`/`assert_facts`/`search_knowledge` surface directly, with server-side
translation (`remember`/`recall`) kept as the fallback for thin clients and the CLI. This
makes the daily driver **zero-key, zero-marginal-cost, and faster** — and it is honest to
the tagline: *models translate, rules decide* — whichever model happens to be in the room.

**Explicit non-goals (unchanged):** hosted multi-user service, vectors as primary
retrieval, full Prolog.

---

## 5. Capabilities map

| Cluster | Tools | State | Daily-driver role |
|---------|------:|-------|-------------------|
| A. Core memory (`remember`, `recall`, `recall_explain`, `list_memories`, `forget`, `history`) | 6 | ✅ solid | **Core** — default profile |
| B. Raw Datalog (`query`, `explain_query`, `assert_facts`, `supersede_facts`, `check_integrity`) | 5 | ✅ solid, LLM-free, ~8 ms | **Core** — the zero-key path |
| C. Search (`search_knowledge` lexical; `semantic_*` embedding) | 5 | ✅ / semantic is 12–27 s | Lexical: core. Semantic: opt-in, never in-turn |
| D. Review gates (propose/apply, tentative, what-if) | 7 | ✅ solid | Power profile — opt-in governance |
| E. Diagnostics (health, why-not, audit, topology, repair…) | 8 | ✅ solid | Power profile — "DBA tools" |
| F. Time travel (diff, checkpoints, recorded views) | 4 | ✅ solid | Power profile |
| G. Bundles (export/verify) | 2 | ✅ solid | Reframe as **backup/restore** |
| H. Semantic version ledger | 7 | ✅ but misnamed | **Hide from MCP** — it's product release governance, not user memory |
| Auto-capture (Stop hook) | — | 🐛 broken in real sessions | **Core** after fix #1 |
| Session-start recall | — | ❌ absent | **Core** — biggest missing piece |
| Web console | — | ✅ but demo-first | Core after inverting the default |
| CLI (~55 cmds), site, benchmarks | — | ✅ | Done; freeze investment |

---

## 6. Roadmap to daily driver

Each phase independently shippable; total ≈ 5–7 focused days. **Phase order is the
opposite of the last two weeks: adoption first, evidence later.**

### Phase A — Make it green and honest *(½ day)*
1. `testTimeout` in `vitest.config.ts` (clears 7 of 8 failures).
2. Investigate the engine p95 regression against its own 25 ms gate.
3. Fix the README model claim; document `export`/`import`.

### Phase B — Close the loop *(2 days)* ← the payoff phase
1. **Fix auto-capture windowing** (role-aware user-message selection).
2. **SessionStart hook**: deterministic, bounded context brief injected at session start
   (personal + project namespace summary; zero LLM calls).
3. **Tool profiles**: `serve --profile core` (Clusters A+B+lexical search, ~12 tools) as
   the documented default; `full` keeps everything.
4. **cwd→namespace derivation** with `personal` always in scope.
5. **`remembero init`**: MCP registration + both hooks + CLAUDE.md snippet + recommended
   env profile (`VALID_TIME_MODE=archive`, `RECALL_ANSWER_MODE=deterministic` for agents).

### Phase C — Make it trustworthy under real load *(1–1½ days)*
1. O(1) journal append (fix #4) — also relieves lock-timeout pressure.
2. Secret-scan false-positive fix (#7).
3. Web console: real memory by default, demo behind `--demo`; surface `review`/prune in
   the UI so pruning bad captures is a click, not a CLI incantation.
4. `remembero backup` / `restore` as named wrappers over bundles + a sync doc.

### Phase D — Zero-key mode *(1 day)*
1. Rewrite Cluster B tool descriptions to teach the calling agent direct Datalog use
   (schema discovery via `list_memories`, then `query`/`assert_facts`).
2. Measure: a week of your own sessions — capture rate, recall hits, cost. This replaces
   LongMemEval as the success metric that matters now.

### Phase E — Tell the truth in the docs *(½–1 day)*
Quickstart (30 lines), four-command tour, latency expectations table, backup/sync page,
"which of the 55 commands you actually need." Move benchmark prose below the fold.

### Success metric

Not another LongMemEval point. **In 14 days: ≥50 real facts in `~/.rembero`, ≥1 session
per day that reads memory at start, zero manual setup on your second machine.** The
scorecard culture the project already has should be pointed at this: an executable
"daily-driver check" that fails while the store contains only demo facts.
