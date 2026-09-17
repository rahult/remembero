# Session store and reading recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** give Remembero a local store of captured conversation turns and a `sessions` recall mode that retrieves, re-ranks and reads them, so the product gets the accuracy the LongMemEval harness measures (reader v7: 425/500).

**Architecture:** a new `src/sessions/` store writes append-only JSONL per session under `~/.rembero/sessions/<namespace>/`, off unless `REMBERO_SESSIONS=on`. A new `src/knowledge/session-retrieval.ts` holds the retrieval and prompt code the harness has today (turn ranking, session aggregation, TypeSafe re-rank, byte budget, date distances, computed notes) so harness and product call one implementation, proven by a byte-for-byte parity test. `recallQuestion` gains an opt-in `sessions` answer mode that classifies the question, retrieves, re-ranks, reads with a configurable reader and returns the answer plus the sessions it read.

**Tech Stack:** TypeScript (Node 24, vitest), existing `MemoryStore` conventions, `searchKnowledge` lexical search, TypeSafe Jev through `src/evals/typesafe-rerank.ts`, local reader v7 through an OpenAI-compatible endpoint.

**Spec:** `docs/superpowers/specs/2026-09-18-session-store-and-reading-recall-design.md`

## Global Constraints

- Nothing is stored unless `REMBERO_SESSIONS=on`; the flag is read once per process through `src/env.ts` helpers, following the existing `REMBERO_*` validation style.
- Every turn is masked before it is written: `maskSensitiveSpans` replaces the matched secret; a turn that still trips `containsSensitiveText` is stored as `REDACTED_SOURCE`.
- Today's recall behaviour is untouched: `REMBERO_RECALL_ANSWER_MODE` keeps `natural | deterministic | evidence` as the default set, and every existing test passes unchanged.
- Session files are append-only. A session's turns are identified by `sha256(role + '\n' + text)`; a turn whose hash is already in the file is skipped.
- Retrieval depth, unit, time-range lookup, reading format, personalisation prompt and assistant-turn visibility all come from `questionKindFromText` (or `typesafeQuestionKind`); no code in `src/` reads a LongMemEval `question_type`.
- A reader that is not on `127.0.0.1`/`localhost` must pass `assertSafeForExternalLlm` before the prompt is sent; a refusal is an error naming the setting, never a silent fallback to another model.
- Tests never touch the network: the TypeSafe client and reader client are injected and stubbed.
- Commit trailers, at the end of every commit message:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01JKn8pk6RZdP6dV42QfA5rA`

## File structure

| File | Responsibility |
|---|---|
| `src/sessions/store.ts` (new) | Session and turn types, append, read, per-namespace index, size cap, delete |
| `src/sessions/import.ts` (new) | Parse whole Claude Code transcript files into sessions |
| `src/safety.ts` (modify) | `maskSensitiveSpans` beside `redactSensitiveText` |
| `src/env.ts` (modify) | `REMBERO_SESSIONS`, `REMBERO_SESSION_CAP_BYTES`, `REMBERO_READER_*` |
| `src/autocapture/capture.ts`, `src/autocapture/transcript.ts` (modify) | Append captured turns; export `parseTranscriptMessages` |
| `src/knowledge/session-retrieval.ts` (new) | Shared retrieval + prompt building for harness and product |
| `src/llm/pipeline.ts` (modify) | `sessions` answer mode in `recallQuestion` |
| `src/mcp/tools.ts`, `src/mcp/server.ts` (modify) | `forget_sessions` tool |
| `src/cli.ts` (modify) | `sessions import`, `sessions forget`, `sessions list` |

---

### Task 1: Span masking in safety.ts

**Files:** Modify `src/safety.ts`. Test: `tests/safety.test.ts`.

**Interfaces:**
- Produces: `maskSensitiveSpans(value: string): { text: string; masked: number }` — every match of the existing `SENSITIVE_TEXT_PATTERNS` and of a Luhn-valid card run is replaced by `[redacted]`; `masked` counts replacements. Unchanged text returns `{ text: value, masked: 0 }`.

- [ ] **Step 1: Write the failing test**

```ts
import { maskSensitiveSpans, containsSensitiveText } from '../src/safety.js';

it('keeps the readable half of a turn and hides the secret', () => {
  const result = maskSensitiveSpans('I set my api key = sk-abc123456789 for the deploy');
  expect(result.masked).toBe(1);
  expect(result.text).toBe('I set my [redacted] for the deploy');
  expect(containsSensitiveText(result.text)).toBe(false);
});

it('leaves ordinary text alone', () => {
  expect(maskSensitiveSpans('I rode 40 km on the new bike')).toEqual({
    text: 'I rode 40 km on the new bike',
    masked: 0,
  });
});
```

- [ ] **Step 2: Run it and watch it fail** — `npx vitest run tests/safety.test.ts`, expected: `maskSensitiveSpans is not a function`.
- [ ] **Step 3: Implement** `maskSensitiveSpans` by running the existing patterns with the global flag over a copy of the text and replacing each match with `[redacted]`, then the card candidates whose digits are Luhn-valid. Export it. Do not change `redactSensitiveText` or `containsSensitiveText`.
- [ ] **Step 4: Run the test and the full file** — `npx vitest run tests/safety.test.ts`, expected: PASS.
- [ ] **Step 5: Commit** `git add src/safety.ts tests/safety.test.ts`, message `feat: mask only the secret span in a turn`.

### Task 2: The session store

**Files:** Create `src/sessions/store.ts`. Modify `src/env.ts`. Test: `tests/sessions-store.test.ts`.

**Interfaces:**
- Consumes: `maskSensitiveSpans` (Task 1), `containsSensitiveText`, `REDACTED_SOURCE`.
- Produces:
  - `interface SessionTurn { index: number; role: 'user' | 'assistant'; ts: string; text: string; hash: string }`
  - `interface SessionHeader { version: 1; source: 'claude-code' | 'remember' | 'import'; sourceSessionId: string; startedAt: string; cwd?: string }`
  - `interface SessionIndexEntry { key: string; source: SessionHeader['source']; startedAt: string; lastTs: string; turns: number; bytes: number }`
  - `class SessionStore { constructor(options?: { root?: string; capBytes?: number }); sessionsEnabled(): boolean; sessionKey(source, sourceSessionId): string; appendTurns(namespace: string, header: SessionHeader, turns: Array<{ role: 'user'|'assistant'; ts: string; text: string }>): { appended: number; skipped: number; masked: number; dropped: string[] }; readSession(namespace: string, key: string): { header: SessionHeader; turns: SessionTurn[] } | undefined; list(namespace: string): SessionIndexEntry[]; deleteSession(namespace: string, key: string): boolean; deleteNamespace(namespace: string): number }`
  - `sessionsRoot(env?): string` — `${REMBERO_HOME ?? ~/.rembero}/sessions`
  - `sessionsEnabledFromEnv(env?): boolean`, `sessionCapBytesFromEnv(env?): number` (default `200 * 1024 * 1024`, integer, 1 MB minimum) in `src/env.ts`.

- [ ] **Step 1: Write the failing tests** (in a `mkdtempSync` root)

```ts
const store = new SessionStore({ root: tmp, capBytes: 1024 * 1024 });
const header = { version: 1, source: 'claude-code', sourceSessionId: 'abc', startedAt: '2026-09-18T00:00:00.000Z' } as const;

it('appends turns and skips ones it already has', () => {
  const first = store.appendTurns('default', header, [
    { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    { role: 'assistant', ts: '2026-09-18T00:00:01.000Z', text: 'Nice, which model?' },
  ]);
  expect(first.appended).toBe(2);
  const second = store.appendTurns('default', header, [
    { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    { role: 'user', ts: '2026-09-18T00:00:02.000Z', text: 'Rode 40 km today' },
  ]);
  expect(second).toMatchObject({ appended: 1, skipped: 1 });
  const key = store.sessionKey('claude-code', 'abc');
  expect(store.readSession('default', key)!.turns.map((t) => t.index)).toEqual([0, 1, 2]);
});

it('masks a secret and records it', () => {
  const result = store.appendTurns('default', { ...header, sourceSessionId: 'sec' }, [
    { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'my api key = sk-abc123456789' },
  ]);
  expect(result.masked).toBe(1);
  const key = store.sessionKey('claude-code', 'sec');
  expect(store.readSession('default', key)!.turns[0].text).toContain('[redacted]');
});

it('drops the oldest session when the cap is exceeded', () => {
  const small = new SessionStore({ root: tmp2, capBytes: 400 });
  small.appendTurns('default', { ...header, sourceSessionId: 'old', startedAt: '2026-09-01T00:00:00.000Z' }, [
    { role: 'user', ts: '2026-09-01T00:00:00.000Z', text: 'x'.repeat(300) },
  ]);
  const result = small.appendTurns('default', { ...header, sourceSessionId: 'new', startedAt: '2026-09-18T00:00:00.000Z' }, [
    { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'y'.repeat(300) },
  ]);
  expect(result.dropped).toEqual([small.sessionKey('claude-code', 'old')]);
  expect(small.list('default').map((e) => e.key)).toEqual([small.sessionKey('claude-code', 'new')]);
});

it('deletes a session and its index entry', () => { /* deleteSession returns true, list() is empty, readSession undefined */ });
it('rejects a namespace that is not [a-z0-9_-]+', () => { expect(() => store.list('../etc')).toThrow(); });
```

- [ ] **Step 2: Run them and watch them fail** — `npx vitest run tests/sessions-store.test.ts`.
- [ ] **Step 3: Implement the store.** JSONL per session at `<root>/<namespace>/<key>.jsonl` (header line then turn lines), `index.json` per namespace rewritten atomically (`.tmp` + `rename`) after each append, `bytes` from the file size, cap enforced oldest-first by `startedAt`, namespace validated against `/^[a-z0-9_-]+$/`, key = `sha256(source + '\0' + sourceSessionId)` sliced to 32 chars, turn hash = `sha256(role + '\n' + text)`.
- [ ] **Step 4: Run the tests** — all PASS, then `npx vitest run` once.
- [ ] **Step 5: Commit** `feat: session store with masking, an index and a size cap`.

### Task 3: Capture and `remember` write sessions

**Files:** Modify `src/autocapture/transcript.ts` (export `parseTranscriptMessages`), `src/autocapture/capture.ts`, `src/llm/pipeline.ts` (`rememberText`). Test: `tests/sessions-capture.test.ts`.

**Interfaces:**
- Consumes: `SessionStore` (Task 2), `sessionsEnabledFromEnv`.
- Produces: `AutoCaptureDeps.sessions?: SessionStore` and `PipelineDeps.sessions?: SessionStore`; `AutoCaptureResult.sessionTurns?: { appended: number; skipped: number }`.

- [ ] **Step 1: Write the failing tests.** With a `SessionStore` on a temp root: a stop-hook capture over a two-turn transcript appends two turns; a second capture over the same transcript tail appends none; `rememberText(deps, 'I bought a road bike')` writes a one-turn session whose `source` is `remember`; with no `sessions` dependency nothing is written and the existing capture assertions still hold.
- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Implement.** Export `parseTranscriptMessages`. In `autoCaptureClaudeStop`, after the existing extraction, when `deps.sessions` is present append the parsed turns under header `{ version: 1, source: 'claude-code', sourceSessionId: input.sessionId, startedAt: <first turn ts or now>, cwd: input.cwd }`, timestamps from the transcript entries when present else the capture time. In `rememberText`, when `deps.sessions` is present append one `user` turn under `{ source: 'remember', sourceSessionId: <sha256 of the text sliced to 32> }`. Wire `deps.sessions` in `src/mcp/server.ts` and `src/cli.ts` only when `sessionsEnabledFromEnv()`.
- [ ] **Step 4: Run the tests plus `tests/autocapture.test.ts` and `tests/llm.test.ts`.**
- [ ] **Step 5: Commit** `feat: capture and remember write turns to the session store`.

### Task 4: Import, list and forget

**Files:** Create `src/sessions/import.ts`. Modify `src/cli.ts`, `src/mcp/tools.ts`, `src/mcp/server.ts`. Test: `tests/sessions-import.test.ts`, additions to `tests/tools.test.ts`.

**Interfaces:**
- Produces: `importClaudeTranscript(store: SessionStore, namespace: string, path: string): { key: string; appended: number; skipped: number }`; CLI `remembero sessions import <files...> [--namespace ns]`, `remembero sessions list [--namespace ns]`, `remembero sessions forget <key|--all> [--namespace ns]`; MCP tool `forget_sessions` with input `{ namespace?: string; key?: string; all?: boolean }` returning `{ deleted: number }`.

- [ ] **Step 1: Write the failing tests.** Importing a two-line transcript file creates one session with both turns; importing the same file twice appends nothing the second time; `forget_sessions` with a key deletes one session and returns `{ deleted: 1 }`; with `all: true` deletes the namespace; with neither it throws.
- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Implement.** `importClaudeTranscript` reads the whole file, reuses `parseTranscriptMessages`, and appends under `{ source: 'import', sourceSessionId: <file basename without extension> }`. CLI subcommands follow the existing `src/cli.ts` command style and print a one-line summary each. The MCP tool follows the shape of `forgetTool` in `src/mcp/tools.ts` and is registered in `server.ts`'s tool list.
- [ ] **Step 4: Run the tests plus `tests/tools.test.ts` and `tests/mcp.test.ts`.**
- [ ] **Step 5: Commit** `feat: import, list and forget sessions`.

### Task 5: Shared retrieval and prompt module

**Files:** Create `src/knowledge/session-retrieval.ts`. Modify `src/evals/longmemeval-answer.ts` to call it. Test: `tests/session-retrieval.test.ts`.

**Interfaces:**
- Consumes: `searchKnowledge`, `questionKindFromText` / `typesafeQuestionKind` (`src/knowledge/question-kind.ts`), `isTemporalQuestion`, the TypeSafe client in `src/evals/typesafe-rerank.ts`, `buildComputedNotes` from `src/knowledge/computed-notes.ts`.
- Produces:
  - `interface RetrievableSession { id: string; date: string; turns: Array<{ role: 'user' | 'assistant'; text: string }> }`
  - `interface SessionRetrievalOptions { kind: QuestionKind; topK: number; contextBytes: number; dateDistances: boolean; computedNotes: boolean; rerank?: { client: TypesafeClient; pool: number; sessionChars: number }; timeRange?: { start: string; end: string } }`
  - `retrieveSessions(question: string, askedAt: Date, sessions: readonly RetrievableSession[], options: SessionRetrievalOptions): Promise<{ ranked: string[]; chosen: string[]; rerank?: RerankUsage }>`
  - `buildReadingPrompt(question: string, askedAt: Date, chosen: readonly RetrievableSession[], options: SessionRetrievalOptions): { system: string; user: string }`

- [ ] **Step 1: Write the failing parity test.** On a three-session fixture, `buildReadingPrompt` must equal the prompt `buildLongMemEvalAnswerContext` produces for the same sessions, question, date and options (compare the system and user strings exactly).
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement** by moving the turn indexing, session aggregation, re-rank call, depth cut, byte budget, date-distance headers and computed-notes block out of `longmemeval-answer.ts` into the new module, and having the harness call it. Keep the harness's public behaviour identical.
- [ ] **Step 4: Run `npx vitest run`** — every existing harness test passes unchanged, plus the parity test.
- [ ] **Step 5: Verify on real data.** `node dist/evals/run-longmemeval-answer.js --retrieval-only --question-types multi-session,temporal-reasoning --split all --formation raw --local-only --top-k 4 --retrieval-unit turn --turn-unit-unless-temporal --multi-session-top-k 12 --temporal-top-k 10 --context-bytes 24576 --date-distances --computed-notes --output results/retrieval-only/r266-after-extract.json` and confirm `evidenceTurnsCompleteRate` equals the 84.0% recorded in `results/retrieval-only/r266-det-range.json`.
- [ ] **Step 6: Commit** `refactor: harness and product share one session retrieval module`.

### Task 6: The `sessions` answer mode

**Files:** Modify `src/llm/pipeline.ts`, `src/env.ts`, `src/mcp/tools.ts`. Test: `tests/sessions-recall.test.ts`.

**Interfaces:**
- Consumes: Tasks 2 and 5, `assertSafeForExternalLlm`.
- Produces: `RecallAnswerMode` gains `'sessions'`; `RecallResult` gains `sessionsRead?: Array<{ key: string; date: string; excerpts: string[] }>` and `readerModel?: string`; `RecallOptions` gains `reader?: { baseUrl: string; model: string; apiKey: string; maxTokens: number; temperature: number; timeoutMs: number }`; `readerFromEnv(env?)` in `src/env.ts` returns that shape or `undefined`.

- [ ] **Step 1: Write the failing tests** with a stub reader client:
  - a question whose evidence is in one stored session returns `status: 'answered'`, the reader's final `Answer:` line as `answer`, and that session in `sessionsRead`;
  - an empty namespace returns `status: 'no_evidence'` and the stub reader is never called;
  - a reader replying `Answer: I do not know` returns `status: 'unknown'`;
  - a non-local reader with a sensitive prompt throws an error naming `REMBERO_READER_BASE_URL`;
  - `answerMode: 'evidence'` still takes the Datalog path (existing tests unchanged).
- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Implement.** In `recallQuestion`, when the mode is `sessions`: classify with `questionKindFromText`, load the namespace's sessions through `SessionStore`, call `retrieveSessions` then `buildReadingPrompt`, send it to the reader from `options.reader ?? readerFromEnv() ?? deps.llm`, take `finalAnswerLine` when the reading format applies, and fill `sessionsRead` from the chosen sessions with their matching turn excerpts. `recall_explain` adds the reader's whole reply and the kind flags.
- [ ] **Step 4: Run the tests plus `tests/llm.test.ts`, `tests/tools.test.ts`, `tests/mcp.test.ts`.**
- [ ] **Step 5: Commit** `feat: sessions answer mode reads retrieved conversations`.

### Task 7: The label-free benchmark arm (ops, about $2 of pod time)

- [ ] Run GLM 5.3 Flash on the 266 and the 500 with `--classify text` and with `--classify typesafe` (free, Ollama subscription) on top of `--formation raw --split all --local-only --top-k 4 --retrieval-unit turn --multi-session-top-k 12 --temporal-top-k 10 --context-bytes 24576 --date-distances --computed-notes --rerank typesafe --reading notes`.
- [ ] Run reader v7 on the 500 with `--classify typesafe`, paired against the 425 row, on one self-stopping pod (`pod.py create-serve --run reader-v7-gemma4-e4b --served-name rembero-reader-v7 --gpu "NVIDIA H100 NVL" --datacenter US-GA-2 --cloud SECURE --max-hours 1 --idle-minutes 20`).
- [ ] Record both tables and each classifier flag's precision and recall in `docs/research/READER-STRUCTURE.md`, and state that this score, not 425, is what a product can reach.

### Task 8: End-to-end smoke (ops)

- [ ] `REMBERO_SESSIONS=on remembero sessions import ~/.claude/projects/<one real transcript>.jsonl --namespace smoke`
- [ ] Serve reader v7 locally (`scripts/serve-reader-v7.sh`), then ask five questions through `recall` with `REMBERO_RECALL_ANSWER_MODE=sessions` and `REMBERO_READER_BASE_URL=http://127.0.0.1:8084/v1`, checking each answer cites the right session.
- [ ] `remembero sessions forget --all --namespace smoke`, then record the five questions, answers and citations in the plan's ledger.
