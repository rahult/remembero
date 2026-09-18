# Session store and reading recall

**Status:** approved in conversation on 2026-09-17/18 (four design sections), spec written 2026-09-18.

## Why

Remembero answers a question by writing a Datalog query over the facts it extracted and evaluating
it. It keeps no conversation text: `rememberTranscriptText` states plainly that the raw transcript
is never persisted, and the turns the autocapture parser produces are flattened away
(`src/autocapture/transcript.ts`, `src/llm/pipeline.ts`). Every gain measured on LongMemEval this
month comes from reading retrieved sessions instead: reader v7 with computed notes, turn-routed
retrieval and TypeSafe re-ranking scores 425/500, where the fact-query path cannot compete because
it has nothing to read. This design gives the product the same material and the same reading step,
without disturbing the fact store.

## Decisions taken with the user

1. Store both user and assistant turns, redacted.
2. Keep sessions until forgotten, with a per-namespace size cap; nothing silently expires.
3. The reader is configurable and defaults to the product's existing LLM; `REMBERO_READER_*`
   points at a dedicated reader such as local v7.
4. Session reading is a new opt-in answer mode. Today's fact-query modes stay the default.
5. Past conversations enter the store only through an explicit import command.

## 1. The session store

- **Location** `~/.rembero/sessions/<namespace>/`, beside `memory/`, honouring `REMBERO_HOME`.
- **One file per session**, append-only. Line 1 is a header (`{version, source, sourceSessionId,
  startedAt, cwd?}`); every later line is a turn (`{index, role: 'user'|'assistant', ts, text,
  hash}`). `index.json` per namespace holds, per session, its first and last timestamp, turn count
  and byte size.
- **File naming** is `sha256(source + '\0' + sourceSessionId)` truncated to 32 hex characters, so no
  session id can escape the directory.
- **Off by default.** Nothing is written unless `REMBERO_SESSIONS=on`.
- **Redaction** uses a new `maskSensitiveSpans(text)` beside today's `redactSensitiveText`
  (`src/safety.ts`): the matched secret is replaced with `[redacted]` and the rest of the turn
  survives. A turn that still trips `containsSensitiveText` after masking is stored as the existing
  `REDACTED_SOURCE` placeholder.
- **Capture** extends the Claude Code stop-hook path: the turns `parseTranscriptMessages` already
  produces are appended, skipping any whose hash is stored, because consecutive stop hooks re-read
  overlapping transcript tails. A `remember` call becomes a one-turn session.
- **Import** `remembero sessions import <files...> --namespace <ns>` reads whole Claude Code
  transcript files, applies the same redaction, and is idempotent.
- **Cap** `REMBERO_SESSION_CAP_BYTES` (default 200 MB per namespace). After each append, whole
  sessions are dropped oldest-first until the namespace is under the cap; each drop is logged.
- **Forgetting** a new `forget_sessions` tool and `remembero sessions forget` remove one session or
  a namespace's sessions. The existing `forget` keeps its fact-only meaning.

## 2. Question kinds, retrieval and the shared module

- **One classifier.** `questionKindFromText` returns `{aggregation, temporal, update, preference,
  assistantRecall}`; `--classify typesafe` replaces it with one TypeSafe request carrying five noul
  questions, falling back to the text rules when the call fails. The product uses the same function.
- **The kind decides** retrieval depth (12 aggregation, 10 temporal, else 4), the retrieval unit
  (turn-level unless temporal), whether the time-range lookup runs, whether the Notes-then-Answer
  reading applies, whether the personalisation prompt is used, and whether assistant turns reach the
  reader. Nothing reads a dataset label.
- **Re-ranking.** The top 30 sessions from the lexical, turn-routed search go to TypeSafe, two noul
  questions each (talks about the question's subject; holds a user statement the answer needs), and
  are reordered by `0.7 × evidence + 0.3 × relevance` before the depth cut. Measured: answer turns
  in context 84% → 95%, reader v7 412 → 425/500, $0.28 per 500 questions, answers cached on disk.
- **Shared module.** Turn ranking, session aggregation, re-ranking, the byte budget, date distances
  and computed notes move out of `src/evals/longmemeval-answer.ts` into `src/knowledge/` so the
  harness and the product call one implementation. A test asserts both build the same prompt for a
  fixture, byte for byte.
- **Index.** Ranking reads one namespace's sessions, so the module keeps an in-memory turn index,
  rebuilt when `index.json` changes.

## 3. The sessions answer mode

- `REMBERO_RECALL_ANSWER_MODE=sessions`, or the mode passed on one `recall` call. Existing modes
  are untouched.
- Per question: classify, retrieve (allowed namespaces only), re-rank, build the prompt through the
  shared module, call the reader, return the final answer line.
- **Reading budget.** `REMBERO_READING_CONTEXT_BYTES`, default **24576** — the
  `--context-bytes 24576` the measured arm ran, and the prompt size reader v7 was trained on. The
  shared module's `DEFAULT_READING_CONTEXT_BYTES` (56 KB) stays as the harness's fallback for arms
  that pass no flag; the product does not use it, because the published number would then describe
  a prompt the product never builds.
- **Candidate cap.** The product ranks at most the 500 most recent sessions and their 20,000 most
  recent turns. Retrieval indexes one clause per turn and the lexical search refuses more than
  100,000, and the store's own 200 MB cap allows roughly ten times that, so without a cap a large
  store made every sessions recall fail until the sessions were deleted. Loading *and* ranking sit
  inside one degrade path: any failure becomes `no_evidence` with the reason recorded, never a
  thrown recall.
- **Reader** from `REMBERO_READER_BASE_URL`, `_MODEL`, `_API_KEY`, `_MAX_TOKENS`, `_TEMPERATURE`,
  `_TIMEOUT_MS`; unset means the product's configured LLM.
- **Result** carries the answer, a status (`answered`, `unknown`, `no_evidence`) and the evidence:
  each session used, its date, and the matching turn excerpts. `recall_explain` adds the reader's
  Notes working and the classifier's flags; plain `recall` never shows the working.
- **No evidence** short-circuits before the reader. A reader that says it does not know yields
  `unknown`, never a guess.
- **Privacy.** Stored text is masked at write time. **Masking is best-effort pattern matching, not
  a guarantee.** The detectors recognise credential words (including `reset_password`-style
  identifier segments), `bearer`/`sk-`/`gh*` tokens, Luhn-valid card runs, PEM private-key blocks,
  AWS access key ids, JWTs and credentials embedded in a URL. A secret shaped like none of those —
  a bare high-entropy string, an internal hostname, a person's address — is stored as written.
  `assertSafeForExternalLlm` is not an independent second opinion either: it shares that one pattern
  set, so it can only refuse what the masker would already have caught. On the benchmark it refused
  2 of 500 prompts.

  A reader is therefore gated on where it runs, not on what the text looks like. A reader that is
  not on localhost — including the `deps.llm` fallback, because recall cannot tell whether
  `LLM_BASE_URL` points at a local server — is refused unless the user sets
  `REMBERO_READER_ALLOW_REMOTE=1`. Without it, nothing is rendered and nothing is sent, and the
  error names the setting. With it, the non-local reader is still gated on
  `assertSafeForExternalLlm`, and that refusal names `REMBERO_READER_BASE_URL`.
- **Failure** of the reader is an error naming the setting to check. No silent fallback to another
  model.

## 4. Testing and measurement

- Unit tests: session file format; stop-hook re-reads never duplicate a turn; the masker keeps the
  readable part; the cap drops oldest-first and logs; `forget_sessions`; import idempotence; each
  classifier flag; the mode's `no_evidence`, `unknown` and reader-error paths.
- Parity test: product prompt equals harness prompt, byte for byte, on a fixture.
- Classifier accuracy per flag against the dataset labels on the 500, recorded in
  `docs/research/READER-STRUCTURE.md`.
- A fully label-free benchmark arm (`--classify text` and `--classify typesafe`): GLM 5.3 Flash free
  on the 266 and the 500, then reader v7 on the 500 on one self-stopping pod (about $2). That score,
  not 425, is the product's number.
- End-to-end smoke: import one real Claude Code transcript into a scratch namespace, ask five
  questions through `recall` with local v7, check the cited sessions, delete the namespace.

## Out of scope

The SQLite-backed store mode; retraining v7 on the new notes lines; a facts-then-sessions fallback;
automatic backfill. Each can follow once the label-free number exists.
