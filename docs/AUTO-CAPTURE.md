# Opt-in Claude Code auto-capture

Remembero can review the end of each Claude Code turn for durable personal facts. This is
an optional ingestion lane around the same deterministic parser, store, and journal used
by manual `remember`; it does not change the Datalog engine or make generated text
authoritative by itself.

## Install or remove the hook

Install Remembero globally, configure `LLM_API_KEY`, then opt in:

```bash
remembero init-hooks --namespace personal
```

By default this merges one managed Stop hook into `~/.claude/settings.json` (or the
directory named by `CLAUDE_CONFIG_DIR`). The existing settings and unrelated hooks are
preserved. The generated hook uses Claude Code's shell-free command/argument form and
`"async": true`, so the external LLM call does not hold up the conversation.

Use a different settings scope explicitly when required:

```bash
remembero init-hooks --settings .claude/settings.local.json --namespace project
```

Remove only Remembero's managed entry with either command:

```bash
remembero init-hooks --remove
remembero remove-hooks
```

Re-run `init-hooks` after moving or reinstalling the package because the safe exec-form
hook records the resolved Node and Remembero CLI paths. Installation and removal are
idempotent.

Claude Code documents the current Stop payload and settings scopes in its
[hooks reference](https://code.claude.com/docs/en/hooks). In non-interactive `claude -p`
sessions, Claude Code may terminate an unfinished async hook during teardown; interactive
turns are the supported auto-capture path.

## What is allowed to become memory

The hook passes JSON to `remembero remember --batch` on standard input. Remembero then:

1. accepts only a Claude `Stop` event;
2. resolves a regular `.jsonl` transcript beneath the configured Claude `projects/`
   directory and rejects symbolic links, hard links, changed file identities, or paths
   outside that root;
3. reads only a bounded tail and extracts **user-authored text only** — tool results,
   private thinking, system notices, fenced code, and all assistant text (including
   `last_assistant_message`) are excluded from the extraction prompt, so the model cannot
   mint "facts" from the assistant's own words. Because agentic sessions bury user turns
   under large tool results, the backward scan widens (up to a 16 MiB window) until it
   finds user text; the extracted message tail itself stays bounded by `--tail-bytes`;
4. rejects credential-like text before any external LLM call;
5. asks the model for at most 12 stable facts explicitly stated or confirmed by the user;
6. accepts additive ground facts only. Rules, variables, retractions, comparisons, and
   negation fail validation and get one corrective retry before the batch fails;
7. sends validated facts through the normal atomic, locked, append-journaled store.

Assistant proposals, task summaries, source code, errors, temporary debugging details,
and pleasantries are context, not memory authority. Auto-capture never silently retracts
or supersedes an existing fact. Use manual `remember` for an explicit update, or prune an
unwanted capture during review.

The raw transcript is not copied into `journal.log` or attached to each fact. Assertion
provenance uses a neutral source label, while capture records retain only bounded metadata
such as the session ID, byte count, operation IDs, and a SHA-256 fingerprint.

## Quotas and duplicate suppression

The default limit is 10 unique capture attempts per namespace per UTC calendar day. A
transcript fingerprint is reserved in the journal before the LLM call, so duplicate Stop
events and over-limit attempts spend no additional model request. Every valid attempt is
journaled as `started`, `captured`, `empty`, `failed`, or `skipped`; interrupted background
work remains visibly `started` rather than disappearing.

If the primary journal is temporarily locked or unavailable, the worker records a
bounded `capture-errors.log` entry under the same memory root using an independent lock.
`remembero review` merges those fallback failures into its status view, so contention is
still inspectable after the primary journal becomes available.

Configure the hook at installation time:

```bash
remembero init-hooks \
  --namespace personal \
  --daily-cap 6 \
  --tail-bytes 16384
```

Equivalent environment defaults are `REMBERO_AUTO_CAPTURE_DAILY_CAP` and
`REMBERO_AUTO_CAPTURE_TAIL_BYTES`. The tail must be between 1 KiB and 48 KiB; the daily
cap must be between 1 and 100. `REMBERO_LLM_ALLOWED_NAMESPACES` applies unchanged, so a
local-only namespace cannot be exported by the hook.

## Review and prune

Review the last seven days:

```bash
remembero review --namespace personal
```

The output shows every capture status and numbers each captured fact. A fact is marked
current only while the live clause still has that capture's operation as its latest
source; a later manual remove/re-add is never pruned through the older capture. Remove
selected facts explicitly in one command:

```bash
remembero review --namespace personal --forget 2,5
```

Selection numbers are resolved and pruned in the same process against journal-backed
capture identities. The retraction and review action receive their own operation ID and
journal records. For automation, add `--json`; change the window with `--days N`.

No raw Datalog query, review, or prune operation calls an LLM.

Version 0.21 does not silently reclassify existing ambient capture. Auto-capture still
accepts only high-confidence facts explicitly grounded in user text and retains this
capture-specific review/prune flow. Use manual `remember --trust tentative`, CLI
`assert --trust tentative`, or MCP `assert_tentative` when the caller intends a tentative
claim lifecycle.
