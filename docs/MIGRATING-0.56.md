# Migrating to 0.56

0.56 is the daily-driver release: it turns the existing engine and store into a memory
loop a single developer actually lives with in Claude Code.

## New

- `remembero init` — one-command setup: installs the managed Stop capture hook **and** a
  new SessionStart hook, registers the MCP server (`core` profile, `archive_until`
  supersession) through the claude CLI, and prints the CLAUDE.md snippet.
- `remembero session-brief` — a bounded, deterministic, zero-LLM summary of one
  namespace. The SessionStart hook injects it so every session starts with remembered
  context.
- MCP tool profiles — `serve --profile core` (or `REMBERO_MCP_PROFILE=core`) registers
  only the 12 daily-driver tools. The default remains `full`.
- Server default namespace — `serve -n <namespace>` routes namespace-less tool calls to
  that namespace instead of `default`.
- `remembero backup <file>` / `remembero restore <file>` — verified, tamper-rejecting,
  idempotent whole-store backup on top of knowledge bundles. See
  [SYNC-AND-BACKUP.md](SYNC-AND-BACKUP.md).

## Changed behavior

- **Auto-capture now works in real agentic sessions.** The transcript reader widens its
  backward scan (up to 16 MiB) until it finds user text instead of returning
  `no_user_text` whenever tool output filled the fixed window.
- **Extraction sees user-authored text only.** Assistant text (including
  `last_assistant_message`) no longer reaches the extraction prompt, so captured facts
  cannot originate from the assistant's own words. Facts merely "confirmed" by a short
  user reply to assistant prose may now be skipped — state them explicitly or use manual
  `remember`.
- **`remembero-web` opens your real memory** (`REMBERO_HOME`, namespace `default`).
  The seeded fictional workspace moved behind `--demo` / `REMBERO_WEB_DEMO=true`; the
  repo's `npm run web` / `npm run web:dev` scripts stay in demo mode.
- **`init-hooks` installs (and `remove-hooks` removes) two managed hooks** — the Stop
  capture hook and the SessionStart brief hook. Re-run `remembero init-hooks` (or
  `remembero init`) after upgrading to pick up the SessionStart hook; removal still
  touches only Remembero's managed entries.
- **13–19-digit runs block only when Luhn-valid.** Timestamps and build IDs no longer
  trip the card-number guard; real card numbers still do.
- **Journal appends no longer re-parse the whole journal.** A stat-validated in-process
  cache removes the per-write read-parse-hash pass; the crash-safe rename protocol and
  cross-process lock semantics are unchanged.
- The agent-database scorecard runs untimed warmup passes (reported as
  `speed.warmupRepetitions`) so the engine p95 gate measures steady-state latency, not
  first-call JIT.

## Compatibility

- Storage format, journal format, checkpoint segments, bundles, and all `REMBERO_*`
  compatibility contracts are unchanged; no data migration is required.
- The MCP tool list under the default `full` profile is unchanged. Registrations created
  by `remembero init` use `core`; pass `--profile full` if an agent needs the complete
  knowledge-engineering surface.
