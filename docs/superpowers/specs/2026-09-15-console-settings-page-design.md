# A Settings page for the local web console

Status: design, 2026-09-15. Requested as: _"Add settings page so we can use it to change
defaults of the app, install folio skill, command line etc. Change default folder etc."_

## Assumptions and open questions

The user was not available to clarify. Each assumption below is load-bearing; the "if wrong"
column says what the spec loses.

| # | Assumption | If it is wrong |
|---|---|---|
| A1 | **"the app" is the local web console** — `remembero-web` (`src/web/server.ts`, `src/web/service.ts`, `web/src/app.tsx` and `web/src/components/*`), not the marketing site in `site/`. | The whole spec moves. A settings page on a static marketing site has no process to configure; nothing here transfers. |
| A2 | **"install folio skill" means one-click install of the Claude Code integration** — exactly what `remembero init` does today: the managed `Stop` + `SessionStart` hooks (`src/autocapture/hooks.ts`), the core-profile MCP registration via `claude mcp add`, and the CLAUDE.md snippet (`src/init.ts`). "Folio" is read as a mis-transcription of the Claude Code integration Remembero already ships. | **Open question, flagged for the user.** There is a Folio app on this Mac — a Markdown review tool with its own skill. If "folio skill" means *a Remembero skill authored for Folio* (a `SKILL.md` installed into Folio's skill directory), then §3 Integrations is the wrong feature: that would need a new skill artifact, a Folio skill-directory discovery path, and a separate install action. Everything else in this spec (config layer, storage/models/recall/safety sections, API, security, testing) stands unchanged either way. **Do not implement §3's "Folio" labelling until this is answered; the hooks + MCP + CLI actions are correct regardless of the answer.** |
| A3 | **"command line" means installing or verifying the `remembero` CLI on PATH** (npm global), with the page showing the detected version and the install command to copy. | If it means an in-browser terminal or a command palette, §3's CLI card is wrong (and an in-browser shell is refused outright on the grounds in §5). |
| A4 | **"change default folder" means the memory root** (`REMBERO_HOME`, so `~/.rembero/memory` by default; `defaultRoot()` at `src/store/store.ts:783`) **and the default namespace**. | If it means the console's *demo sandbox* root (`REMBERO_WEB_ROOT`) or an export/backup directory, the Storage section changes shape but the config layer does not. |
| A5 | **Settings are per-machine, single-user, local-only.** No multi-profile, no sync, no per-namespace overrides. | Multi-profile would need a profile selector and a config-set concept; the flat file in §1 would become a directory. |
| A6 | **An `LLM_API_KEY` is a secret that stays in the environment/`.env`** and is never written to the config file nor returned to the browser. | If the user wants the key editable from the page, §5 has to gain an OS-keychain or encrypted-at-rest story. Presence-only is the shipped behaviour. |

## Problem

Every setting Remembero has is an environment variable read at process start. There is no
config file: `src/env.ts` reads `process.env` per call, `defaultRoot()` reads `REMBERO_HOME`,
`lazyClientFromEnv()` reads `LLM_*`. Changing a default means editing a shell profile or a
`.env`, then restarting the CLI, the MCP server and the console separately — and the three
processes can silently disagree. Onboarding (`remembero init`) is CLI-only, so a user who
starts at the console has no path to the hooks and the MCP registration.

Two things are therefore needed, in this order: a persisted config layer shared by all three
entry points, and a console view over it.

## Design

### 1. The config layer

**File.** `~/.rembero/config.json`, overridable by `REMBERO_CONFIG`. Note it is **not** under
`REMBERO_HOME`: the memory root is itself a setting, so a config file inside it would be a
cycle. The path is `join(homedir(), '.rembero', 'config.json')` unconditionally.

**Precedence: env var > config file > built-in default.** This is deliberately implemented by
*layering the file underneath `process.env`* rather than by rewriting every reader:

```ts
// src/config.ts
export function bootstrapConfig(env = process.env): ConfigLoadResult;
```

`bootstrapConfig()` reads and validates the file, then for each key whose env var is **unset**
writes the file's value into `process.env`. It returns `{ settings, sources, path, warnings }`,
where `sources: Record<SettingKey, 'env' | 'config' | 'default'>` records what won — the map
the console renders. Called once per process, immediately after `loadEnv()`, in three places:
`src/cli.ts` `main()`, the MCP `serve` path, and `startWebServer()`.

Why this shape: every existing `*FromEnv()` validator in `src/env.ts`, every
`process.env.REMBERO_*` read in `pipeline.ts`, `embeddings.ts`, `safety.ts` and `hooks.ts`
keeps working untouched, and precedence falls out for free. The cost is one global mutation of
`process.env` at startup — acceptable because it happens once, before any store or client is
constructed, and because tests inject their own `env` object.

**Schema.** One version field and five sections. Every key maps to exactly one existing env
var and reuses that env var's existing validator; the config loader adds no new validation
semantics, only a JSON-shape check (type, enum, range) before the value is stringified into
`process.env` so that a bad file fails with a config-flavoured message instead of an env one.

| Config key | Env var | Default | Validation | Applies |
|---|---|---|---|---|
| `storage.memoryRoot` | `REMBERO_HOME`¹ | `~/.rembero/memory` | absolute path; parent exists; not a symlink | restart |
| `storage.namespace` | — ² | `default` | `/^[a-z0-9_-]+$/` (`hooks.ts` `NAMESPACE_RE`) | restart |
| `models.llmBaseUrl` | `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | `http:`/`https:` URL | live |
| `models.llmModel` | `LLM_MODEL` | `anthropic/claude-sonnet-5` | non-empty, ≤ 200 chars | live |
| `models.embeddingModel` | `REMBERO_EMBEDDING_MODEL` | `perplexity/pplx-embed-v1-0.6b` | non-empty | live |
| `models.embeddingBaseUrl` | `REMBERO_EMBEDDING_BASE_URL` | inherits `LLM_BASE_URL` | URL | live |
| `recall.answerMode` | `REMBERO_RECALL_ANSWER_MODE` | `evidence` | `evidence｜deterministic｜natural` | live |
| `recall.computedNotes` | `REMBERO_COMPUTED_NOTES` | `true` | boolean (serialized as `1`/`0`) | live |
| `recall.schemaPredicateLimit` | `REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT` | `8` | integer 1–256 | live |
| `safety.integrityMode` | `REMBERO_INTEGRITY_MODE` | `no_new_violations` | `off｜strict｜no_new_violations` | live |
| `safety.validTimeMode` | `REMBERO_VALID_TIME_MODE` | `delete` | `delete｜archive_until` | live |
| `safety.selfAtom` | `REMBERO_SELF` | `user` | `/^[a-z][a-z0-9_]*$/` | live |
| `safety.extractionVocabulary` | `REMBERO_EXTRACTION_VOCABULARY` | `open` | `open｜closed` | live |
| `capture.dailyCap` | `REMBERO_AUTO_CAPTURE_DAILY_CAP` | `10` | integer 1–100 (`validateAutoCaptureDailyCap`) | reinstall hooks |
| `capture.tailBytes` | `REMBERO_AUTO_CAPTURE_TAIL_BYTES` | `24576` | integer 1024–49152 (`validateAutoCaptureTailBytes`) | reinstall hooks |
| `mcp.profile` | `REMBERO_MCP_PROFILE` | `full` | `core｜full` | restart |

¹ `REMBERO_HOME` names the *parent* (`$REMBERO_HOME/memory` is the root). `storage.memoryRoot`
names the memory directory itself. Resolution order in the new `defaultRoot()`:
`process.env.REMBERO_HOME` → `join(that, 'memory')`; else `config.storage.memoryRoot` verbatim;
else `join(homedir(), '.rembero', 'memory')`. The env var keeps winning, so nothing that works
today breaks.

² The namespace has no env var today — the CLI takes `-n`, `init` defaults to `personal`,
`init-hooks` to `default`, the console to `default`. `storage.namespace` becomes the single
default those three fall back to when no flag is given. An explicit `-n` still wins.

**Deliberately not in the file:** `LLM_API_KEY` / `OPENAI_API_KEY` (secrets, §5);
`REMBERO_WEB_*` (bind host, port, demo — server-lifecycle flags, and putting the demo switch in
a browser-writable file would let the page talk itself out of the sandbox);
`REMBERO_LLM_ALLOWED_NAMESPACES`, `REMBERO_CHECK_MODE`/`_SUITE`/`_NAMESPACES`,
`REMBERO_INTEGRITY_NAMESPACES`, `REMBERO_ENTITY_IDENTITY`, `REMBERO_SQLITE_EXTENSION`,
`REMBERO_ENGINE_DEBUG` (operator/CI knobs; no evidence of a user wanting them in a GUI — YAGNI).

**Writing.** `writeConfig(partial)` does read → merge → validate → atomic write (temp file in
the same directory, `mode: 0o600`, `renameSync`), the same pattern `hooks.ts` `writeSettings()`
already uses. Unknown keys are rejected, not silently kept. A key whose env var is currently
set is still written to the file, and the response tells the caller the env var is shadowing it.

**Applying a change to a running console.** Three tiers, shown per field:

- **live** — `bootstrapConfig()` re-runs after the write and re-stamps `process.env`; these keys
  are read per call by the pipeline, so the next recall uses the new value. The LLM client is
  `lazyClientFromEnv()` and is rebuilt on the next call, so base URL and model are live too.
- **restart** — memory root, namespace, MCP profile. The `MemoryStore` and the MCP server are
  constructed at boot. **The change never moves data**: it writes the new path and nothing else.
  The response carries `restartRequired: true` and the console shows a persistent banner.
- **reinstall hooks** — the capture cap and tail bytes are baked into the hook argv in
  `~/.claude/settings.json`. Changing them writes the config *and* re-runs `installClaudeHook()`
  with the new values, but only if the managed hooks are already installed.

### 2. The Settings view

New `web/src/components/settings-view.tsx`, `NavigationView` gains `'settings'`, a nav item with
the existing `GearIcon`, and the currently-dead gear button in the desktop topbar
(`web/src/app.tsx:~907`) is wired to `navigateTo('settings')`. Styling follows
`rules-view.tsx`: a `view-header`, `panel-section` blocks, `pill` chips.

Every field row renders: label, control, the effective value, a **source chip**
(`env` / `config` / `default` / `demo`), inline validation, and — once saved — a
**restart needed** badge. `env`-sourced fields render the control disabled with the text
"set by `LLM_MODEL` in your environment; unset it to edit here", because a config write that an
env var shadows is a silent no-op and is the single most likely source of a bad bug report.

Sections:

- **Storage** — memory root (text input for an absolute path; the server reports back
  `exists`, `writable`, and `containsStore` — whether `journal.log` or `.journal-segments/`
  is already there — so the user can see whether they are pointing at an existing store or an
  empty folder). Copy states plainly: *"Points Remembero at this folder. Your existing memory
  is not moved or copied."* Plus the default namespace.
- **Models** — LLM base URL, model, API-key **presence only** (`Configured (env)` /
  `Not configured`, with the `export LLM_API_KEY=…` line to copy), embedding model and base
  URL, and a **local writer** one-click preset that fills base URL `http://127.0.0.1:8081/v1`,
  model `rembero-writer` (the README "Run the model locally" values) and shows the
  `llama-server` command to copy. The preset only fills the form; the user still saves.
- **Recall** — answer mode (three radio options carrying the `src/env.ts` rationale: evidence is
  local and model-free, natural is the only mode that sends recalled facts to a model),
  computed notes, schema predicate limit.
- **Safety** — integrity mode, valid-time mode, auto-capture daily cap and tail bytes.
- **Integrations** — three cards:
  - *Claude Code hooks + MCP*: detected state (`Stop` hook, `SessionStart` hook, MCP server
    registered) with **Install** / **Remove** buttons, and the settings path being written.
  - *Command line*: detected `remembero` on `PATH`, its version, whether it matches the running
    console's version, and `npm install -g remembero` to copy when it is missing.
  - *CLAUDE.md snippet*: `claudeMdSnippet(namespace)` rendered in a `<pre>` with a copy button.
    **The console never edits the user's CLAUDE.md** — it cannot know which one — so this stays
    copy-only, and the Install button's success text says so.

Save is **per section**, POSTing only dirty keys. No optimistic update: after a successful POST
the view re-fetches `GET /api/settings` and renders the authoritative values and sources, so a
shadowing env var or a server-side clamp is visible immediately.

### 3. API

Four routes in `apiResponse()` (`src/web/server.ts`). Non-GET already goes through
`assertSameOrigin()`; the 64 KiB input and 16 MiB output bounds already apply.

| Route | Body / response |
|---|---|
| `GET /api/settings` | `{ readOnly, restartRequired, configPath, fields: { <key>: { value, source, editable, appliesAt, validation? } }, secrets: { llmApiKey: { present, source } } }` |
| `POST /api/settings` | `{ <dotted key>: value, … }` (partial) → same shape as GET, plus `{ changed: string[], restartRequired, warnings }` |
| `GET /api/integrations/status` | `{ hooks: { stop, sessionStart, settingsPath }, mcp: 'registered'｜'not_registered'｜'claude_cli_missing', cli: { onPath, path?, version?, matchesConsole }, claudeMdSnippet }` |
| `POST /api/integrations/claude-code/install` \| `/remove` | `{}` → `{ hooks: HookChangeResult, registration?: InitRegistration, status: <the GET shape> }` |

Install calls `runInit()` unchanged. One wiring detail that will bite: `runInit` needs
`cliPath`, and `process.argv[1]` in the console process is the `remembero-web` entry, not
`cli.js` — resolve it from the module (`resolve(dirname(fileURLToPath(import.meta.url)),
'../cli.js')`), `existsSync`-check it, and fail with `cli_not_found` rather than writing a hook
that points at nothing. Remove calls `removeClaudeHook()` and leaves the MCP registration alone
(removing an MCP server is `claude mcp remove`, a different blast radius — "Not in this version").

Detection needs one small addition to `src/autocapture/hooks.ts`: an exported, read-only
`claudeHookStatus({ settingsPath })` reusing the existing `readSettings()` / `isManagedHandler()`
helpers. MCP detection runs `claude mcp get remembero` with a 5 s timeout, result cached 10 s.
**CLI detection does not execute anything**: it walks `PATH` for a `remembero` entry and reads
the `package.json` next to its `realpath`. Executing a binary that happens to be named
`remembero` on an unknown `PATH` because a browser asked is not a trade worth making for a
version string.

Demo mode (`--demo` / `REMBERO_WEB_DEMO=true`): `resolveWebConfig()`'s `demo` flag is threaded
into `RemberoWebService`. All four POST routes throw `WebServiceError('demo_read_only', 403)`;
the two GETs work and return `readOnly: true`, with `storage.memoryRoot` reported with source
`demo` and the sandbox path — so the console still cannot touch real memory in demo mode, which
is the rule `docs/WEB-CONSOLE.md` already sets.

### 4. Data flow

```
startup:  loadEnv() → bootstrapConfig() → process.env stamped → MemoryStore(defaultRoot())
GET:      browser → /api/settings → service.settings() → readConfig() + sources + detection
POST:     browser → /api/settings → validate partial → writeConfig() (0600, atomic)
                                  → bootstrapConfig() re-stamp → re-read → respond
install:  browser → /api/integrations/claude-code/install → runInit() → ~/.claude/settings.json
                                  → spawnSync('claude', [...argv]) → re-detect → respond
```

### 5. Security

- **API keys never reach the browser.** `GET /api/settings` returns presence and source only.
  `POST /api/settings` rejects any key in the secret deny-list with
  `WebServiceError('secret_not_stored', 400)` and a message pointing at `.env`. This holds the
  existing `docs/WEB-CONSOLE.md` line — *"The browser never receives `LLM_API_KEY`"*.
- **Config file `0600`**, parent directory `0700`, atomic temp+rename, symlink refused
  (`lstatSync().isSymbolicLink()`), size-bounded read — the same posture `hooks.ts` already
  applies to `~/.claude/settings.json`.
- **No shell from the browser.** The only process the API can start is the whitelisted
  `claude mcp add …` argv that `runInit()` builds in library code, via `spawnSync` with no
  shell, a 30 s timeout, and a fixed argv whose only interpolated value is the namespace —
  already validated against `NAMESPACE_RE`. No user-supplied path, flag, or command is ever
  passed through. (Pre-existing behaviour worth restating: `runInit` forwards `LLM_API_KEY` to
  `claude mcp add -e`. The key travels server-side only, never through the browser.)
- **Path handling.** `storage.memoryRoot` must be absolute and non-symlink, and its parent must
  exist; the store creates the directory itself at `0700`. The setting only ever *points*.
- Same-origin, loopback-only bind, and the existing security headers are unchanged.

### 6. Error handling

Everything fails closed through the existing `WebServiceError` → JSON `{ error, message }` path.
New codes: `invalid_setting` (400, names the key and the constraint), `unknown_setting` (400),
`secret_not_stored` (400), `config_write_failed` (500), `config_unreadable` (500, malformed file
— read is refused rather than silently reset, so a hand-edited file is never destroyed),
`demo_read_only` (403), `cli_not_found` (500), `claude_cli_missing` (returned as *status*, never
an error — a missing `claude` CLI is a normal state the card explains). A failed MCP
registration follows `runInit`'s existing contract: reported, never thrown, with the command to
run by hand.

### 7. Testing

**`tests/config.test.ts`** (new): precedence for each tier (env beats file beats default);
`REMBERO_CONFIG` redirection; a malformed file raises `config_unreadable` and does not delete;
each validator's boundary (`schemaPredicateLimit` 0/1/256/257, `dailyCap` 0/1/100/101,
`selfAtom` casing, non-absolute `memoryRoot`); unknown key rejected; round-trip write→read;
file mode is `0600`; `defaultRoot()` resolution order including the `REMBERO_HOME`-wins case.

**`tests/web.test.ts`** (extended, matching the file's existing style — `mkdtempSync` roots,
`webService(label)` helper, `startWebServer({ root, port: 0 })` + `fetch` with an `Origin`
header): `settings()` reports every field with its source; `updateSettings()` persists and
reflects a changed value; an env-shadowed key returns a warning and `source: 'env'`; a bad value
rejects with `invalid_setting` and writes nothing; a secret key rejects with `secret_not_stored`;
demo mode allows GET and refuses POST with `demo_read_only`; cross-origin POST is 403;
integrations status reports hook presence from a temp settings file, and install/remove
round-trips it.

For that to be testable without touching the real `~/.rembero` or `~/.claude`,
`RemberoWebService` options gain four injection seams: `configPath`, `env`,
`claudeSettingsPath`, and `exec` (the `InitExec` that `runInit` already accepts). No test may
depend on the developer's home directory.

## Order of work

1. `src/config.ts` + `defaultRoot()` change + `bootstrapConfig()` at the three entry points, with `tests/config.test.ts`. Ships value on its own: a config file for the CLI and MCP server.
2. `claudeHookStatus()` in `hooks.ts`, and the service methods `settings()`, `updateSettings()`, `integrationsStatus()`, `installClaudeCode()`, `removeClaudeCode()`, with service tests.
3. The four routes + demo gating + HTTP tests.
4. `settings-view.tsx`, the nav item, the gear-button wiring, `api.ts` types and normalizers.
5. `docs/WEB-CONSOLE.md` Configuration table and README Quickstart updated to mention the file.

## Not in this version

- Editing or storing `LLM_API_KEY` (or any secret) from the browser; no keychain integration.
- Moving, copying, or migrating memory when the root changes. It points; it never moves.
- Editing the user's `CLAUDE.md`. Copy-to-clipboard only.
- `claude mcp remove` from the page. Install and hook-removal only.
- Per-namespace or per-project setting overrides; multiple profiles; import/export of settings.
- Editing `REMBERO_WEB_HOST`/`PORT`/`DEMO`, the check-suite knobs, `REMBERO_LLM_ALLOWED_NAMESPACES`, `REMBERO_ENTITY_IDENTITY`, `REMBERO_SQLITE_EXTENSION`, `REMBERO_ENGINE_DEBUG`.
- A "restart the server" button. The banner tells the user; the user restarts.
- Live-reloading the config file on external edits (no watcher). Re-read happens on write and at startup.
- A settings audit log or change history.
- A Remembero skill for the Folio Markdown app — blocked on open question A2.
