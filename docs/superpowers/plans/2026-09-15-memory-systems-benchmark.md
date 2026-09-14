# Memory systems benchmark implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run-longmemeval-answer` able to swap its memory layer for any external system through one JSON-lines protocol, run eight systems over the same 500 LongMemEval-S questions with the same reader, judge, context builder and retrieval depth, and publish the leaderboard.

**Architecture:** One protocol module (`src/evals/memory-systems-protocol.ts`) names the request, the response, the client interface and the observation record; every later task imports it and nothing re-declares those shapes. A long-lived child process per adapter speaks that protocol over stdin/stdout. Inside `evaluateLongMemEvalAnswerInstance` a single `if (options.memorySystem === undefined) { …existing lexical search… } else { …ask the adapter… }` branch sits after formation and before `buildLongMemEvalAnswerContext`, so the reader, the judge, the result schema, the DeepSeek re-judge sidecar and the summary tables are untouched. Three baselines and Remembero's two rows are built-in clients with no process at all.

**Tech Stack:** TypeScript (vitest, `tsc -p tsconfig.json`), Node 22 `child_process.spawn` without a shell, Python 3.12 PEP 723 single-file bridges run by `uv`, FastEmbed bge-small on CPU, Ollama (`nomic-embed-text`, `glm-5.3-flash:cloud`), DeepSeek direct API.

**Spec:** `docs/superpowers/specs/2026-09-14-memory-systems-benchmark-design.md` — read it before Task 1; this plan argues from its "What is held constant", "Two lanes", "Protocol", "Harness changes", "Outputs" and "Order of work" sections.

## Global Constraints

- Protocol name, fixed: `rembero.memory-systems.v1`. One long-lived process per adapter per run, JSON lines on stdin and stdout, spawned **without a shell**, stderr counted and discarded, a byte limit per response and a timeout per request. The conformance suite's one-process-per-case bridges (`benchmarks/adapters/*/bridge.py`) are **not** modified; the new bridges are new files beside them.
- Task set: LongMemEval-S cleaned, `xiaowu0162/longmemeval-cleaned`, commit `98d7416c24c778c2fee6e6f3006e7a073259d48f`, sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. Iteration set: the 100-question stratified subset in `.cache/longmemeval/subset-100.txt` (one comma-separated line, no trailing newline, so `--cases "$(cat .cache/longmemeval/subset-100.txt)"` is the literal form).
- Reader for every fixed-harness row: GLM 5.3 Flash through Ollama Cloud — `--reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 --reader-api-key ollama --reader-max-tokens 440`. Thinking is on by default and the harness already strips reasoning content; do not pass a thinking flag.
- Judge for every row: `deepseek-chat` with `LLM_BASE_URL=https://api.deepseek.com/v1` and `LLM_API_KEY=$DEEPSEEK_API_KEY`, LongMemEval official-compatible protocol.
- Retrieval depth must be passed explicitly on every command: `--top-k 4 --multi-session-top-k 15 --temporal-top-k 10`. The harness default is 5/5 for multi-session and temporal; a run without these flags is a different run and is not a benchmark row.
- Context: `--context-bytes 24576`, `--date-distances`, source characters 16,384 (the harness constant). Computed notes are **off** in the retrieval lane and **on** only where a row's command says so.
- Time-aware retrieval (`--temporal-range-model`), entity retrieval and engine recall are off for everyone; the runner refuses them together with `--memory-system`.
- `--local-only` on every memory-system command: the memory layer under test does the retrieval, so the harness's own semantic route must be empty.
- DeepSeek V4 Flash at ingest is reached at `https://api.deepseek.com/v1` with model `deepseek-chat` (it aliases the current flash model), $0.22 per million input tokens and $0.66 output off-peak; peak is double and peak hours are 01:00–04:00 and 06:00–10:00 UTC. Run Mem0 and Graphiti outside those windows.
- `uv` runs the PEP 723 bridges; package pins live in the script header **and** in the manifest's `disclosures.packages`, and the two must agree.
- Judge noise on identical prompts is about ±4 on 100 questions and ±7 on 500. No claim in the document ranks two systems inside that band.
- Every run writes `docs/research/results/memory-systems-<system>-<lane>-subset100.json` or `-all500.json` and keeps the stored result schema exactly: the top-level fields and `settings` of `docs/research/results/longmemeval-raw-reader-v4-notes-ds-mt-all266.json`.

---

### Task 1: The protocol and the long-lived process client

**Cost:** $0. No model calls; the stub adapter is a Node one-liner.

**Files:**
- Create: `src/evals/memory-systems-protocol.ts`
- Create: `src/evals/memory-systems-process.ts`
- Test: `tests/memory-systems.test.ts`

**Interfaces:**
- Produces, from `memory-systems-protocol.ts`: `MEMORY_SYSTEMS_PROTOCOL_VERSION`, `MEMORY_SYSTEM_MAX_SESSIONS = 100`, `MEMORY_SYSTEM_MAX_MEMORIES = 500`, `MEMORY_SYSTEM_MAX_MEMORY_CHARACTERS = 4000`, `MEMORY_SYSTEM_UNSOURCED_SESSION = 'memory:unsourced'`, `MEMORY_SYSTEM_MEMORY_HEADROOM_BYTES = 2048`; types `MemorySystemLane`, `MemorySystemTurn`, `MemorySystemSession`, `MemorySystemRequest`, `MemorySystemRetrieved`, `MemorySystemMemory`, `MemorySystemUsage`, `MemorySystemWallMs`, `MemorySystemResponse`, `MemorySystemClient`, `MemorySystemObservation`; functions `memorySystemDate(value: string): string`, `memorySystemRequestFor(instance: LongMemEvalInstance, lane: MemorySystemLane, topK: number): MemorySystemRequest`, `parseMemorySystemResponse(value: unknown, request: MemorySystemRequest): MemorySystemResponse`, `summarizeMemorySystemUsage(observations: ReadonlyArray<{ memorySystem?: MemorySystemObservation }>): MemorySystemUsageSummary`.
- Produces, from `memory-systems-process.ts`: `interface MemorySystemProcessOptions { id: string; executable: string; args?: string[]; workingDirectory?: string; env?: Record<string, string>; timeoutMs?: number; maxResponseBytes?: number }` and `createMemorySystemProcess(options: MemorySystemProcessOptions): MemorySystemClient`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/memory-systems.test.ts
import { describe, expect, it } from 'vitest';
import {
  MEMORY_SYSTEMS_PROTOCOL_VERSION,
  memorySystemDate,
  memorySystemRequestFor,
  parseMemorySystemResponse,
  summarizeMemorySystemUsage,
} from '../src/evals/memory-systems-protocol.js';
import { createMemorySystemProcess } from '../src/evals/memory-systems-process.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';

function instance(): LongMemEvalInstance {
  return {
    question_id: 'ba358f49_abs',
    question_type: 'temporal-reasoning',
    question: 'How long ago did I graduate?',
    answer: 'Business Administration, 2019',
    question_date: '2023/07/10 (Mon) 09:00',
    haystack_session_ids: ['noise', 'evidence'],
    haystack_dates: ['2023/05/15 (Mon) 02:21', '2023/06/01 (Thu) 11:04'],
    haystack_sessions: [
      [
        { role: 'user', content: 'Compare credit card rewards.' },
        { role: 'assistant', content: 'Here are three.' },
      ],
      [
        { role: 'user', content: 'I graduate with Business Administration.' },
        { role: 'assistant', content: 'Congratulations.' },
      ],
    ],
    answer_session_ids: ['evidence'],
  };
}

// A stub adapter: one process, one JSON line per question, noise on stderr.
const STUB = `
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const request = JSON.parse(line);
  process.stderr.write('adapter chatter that must never be parsed\\n');
  const hits = request.sessions.filter((session) =>
    session.turns.some((turn) => turn.content.includes('graduate')));
  process.stdout.write(JSON.stringify({
    questionId: request.questionId,
    retrieved: hits.map((session, index) => ({
      sessionId: session.id, rank: index + 1, score: 1 - index / 10,
    })),
    memories: hits.map((session) => ({
      text: 'the user graduated with Business Administration',
      sessionIds: [session.id],
      at: session.date,
    })),
    unsupported: [],
    usage: { modelCalls: 2, inputTokens: 100, outputTokens: 10, costUsd: 0.0001 },
    wallMs: { ingest: 41, search: 2 },
  }) + '\\n');
});
`;

const SILENT = 'setInterval(() => {}, 1000);';

describe('rembero.memory-systems.v1', () => {
  it('normalizes dataset dates and never shows the adapter the gold answer', () => {
    expect(memorySystemDate('2023/07/10 (Mon) 09:00')).toBe('2023-07-10');
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(request.protocolVersion).toBe(MEMORY_SYSTEMS_PROTOCOL_VERSION);
    expect(request.topK).toBe(4);
    expect(request.lanes).toEqual(['retrieval']);
    expect(request.sessions.map(({ id, date }) => [id, date])).toEqual([
      ['noise', '2023-05-15'],
      ['evidence', '2023-06-01'],
    ]);
    const serialized = JSON.stringify(request);
    expect(serialized).not.toContain('Business Administration, 2019');
    expect(serialized).not.toContain('answer_session_ids');
  });

  it('rejects a session id that was not in the request', () => {
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(() =>
      parseMemorySystemResponse(
        { questionId: request.questionId, retrieved: [{ sessionId: 'invented', rank: 1 }] },
        request,
      ),
    ).toThrow(/not in the request/);
  });

  it('rejects an answer to a different question and surfaces a reported error', () => {
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(() => parseMemorySystemResponse({ questionId: 'other' }, request)).toThrow(
      /expected ba358f49_abs/,
    );
    expect(() =>
      parseMemorySystemResponse({ questionId: request.questionId, error: 'qdrant died' }, request),
    ).toThrow(/qdrant died/);
  });

  it('keeps one process alive across questions and discards stderr', async () => {
    const client = createMemorySystemProcess({
      id: 'stub',
      executable: process.execPath,
      args: ['-e', STUB],
      timeoutMs: 10_000,
    });
    try {
      const first = await client.request(memorySystemRequestFor(instance(), 'retrieval', 4));
      const second = await client.request(memorySystemRequestFor(instance(), 'memories', 4));
      expect(first.retrieved).toEqual([{ sessionId: 'evidence', rank: 1, score: 1 }]);
      expect(second.memories?.[0]?.sessionIds).toEqual(['evidence']);
      expect(second.usage).toEqual({
        modelCalls: 2,
        inputTokens: 100,
        outputTokens: 10,
        costUsd: 0.0001,
      });
      expect(second.wallMs).toEqual({ ingest: 41, search: 2 });
    } finally {
      await client.close();
    }
  });

  it('kills a silent adapter at the timeout', async () => {
    const client = createMemorySystemProcess({
      id: 'silent',
      executable: process.execPath,
      args: ['-e', SILENT],
      timeoutMs: 400,
    });
    try {
      await expect(
        client.request(memorySystemRequestFor(instance(), 'retrieval', 4)),
      ).rejects.toThrow(/timed out after 400ms/);
    } finally {
      await client.close();
    }
  });

  it('totals adapter usage across observations', () => {
    const summary = summarizeMemorySystemUsage([
      {
        memorySystem: {
          id: 'stub',
          lane: 'retrieval',
          usage: { modelCalls: 2, inputTokens: 100, outputTokens: 10, costUsd: 0.01 },
          wallMs: { ingest: 40, search: 2 },
          returnedSessions: 1,
          returnedMemories: 0,
          memoryBytes: { supplied: 0, kept: 0, dropped: 0 },
          unsupported: [],
        },
      },
      {
        memorySystem: {
          id: 'stub',
          lane: 'retrieval',
          usage: { modelCalls: 3, inputTokens: 200, outputTokens: 20, costUsd: 0.02 },
          wallMs: { ingest: 60, search: 4 },
          returnedSessions: 1,
          returnedMemories: 0,
          memoryBytes: { supplied: 0, kept: 0, dropped: 0 },
          unsupported: [],
        },
      },
      {},
    ]);
    expect(summary).toEqual({
      questions: 2,
      modelCalls: 5,
      inputTokens: 300,
      outputTokens: 20,
      costUsd: 0.03,
      medianIngestMs: 50,
      medianSearchMs: 3,
      droppedMemoryBytes: 0,
    });
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run tests/memory-systems.test.ts`
Expected: FAIL, `Cannot find module '../src/evals/memory-systems-protocol.js'`.

- [ ] **Step 3: Write the protocol module**

```ts
// src/evals/memory-systems-protocol.ts
/**
 * rembero.memory-systems.v1 — the wire between the LongMemEval answer harness and a memory
 * layer under test. One JSON line per question in, one JSON line out, over a long-lived
 * process. The harness gives the adapter the question and every session of that question's
 * haystack; the adapter gives back either the sessions it would retrieve ('retrieval' lane)
 * or its own memory text ('memories' lane). The adapter never sees the gold answer or the
 * gold evidence session ids.
 */
import type { LongMemEvalInstance } from './longmemeval.js';

export const MEMORY_SYSTEMS_PROTOCOL_VERSION = 'rembero.memory-systems.v1' as const;

/** Sessions the harness will accept from one response, whatever the adapter returns. */
export const MEMORY_SYSTEM_MAX_SESSIONS = 100;
export const MEMORY_SYSTEM_MAX_MEMORIES = 500;
export const MEMORY_SYSTEM_MAX_MEMORY_CHARACTERS = 4_000;
/** Pseudo-session for memory text the adapter did not attribute to a session. */
export const MEMORY_SYSTEM_UNSOURCED_SESSION = 'memory:unsourced';
/** Kept free inside the context budget for headers, the question and the notes block. */
export const MEMORY_SYSTEM_MEMORY_HEADROOM_BYTES = 2_048;

export type MemorySystemLane = 'retrieval' | 'memories';

export interface MemorySystemTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface MemorySystemSession {
  id: string;
  /** YYYY-MM-DD. */
  date: string;
  turns: MemorySystemTurn[];
}

export interface MemorySystemRequest {
  protocolVersion: typeof MEMORY_SYSTEMS_PROTOCOL_VERSION;
  questionId: string;
  questionType: string;
  /** YYYY-MM-DD. */
  questionDate: string;
  question: string;
  /** The retrieval depth this question is scored at; the adapter is expected to respect it. */
  topK: number;
  lanes: MemorySystemLane[];
  sessions: MemorySystemSession[];
}

export interface MemorySystemRetrieved {
  sessionId: string;
  rank: number;
  score?: number;
}

export interface MemorySystemMemory {
  text: string;
  sessionIds?: string[];
  /** YYYY-MM-DD. */
  at?: string;
}

export interface MemorySystemUsage {
  modelCalls: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
}

export interface MemorySystemWallMs {
  ingest: number;
  search: number;
}

export interface MemorySystemResponse {
  questionId: string;
  retrieved?: MemorySystemRetrieved[];
  memories?: MemorySystemMemory[];
  unsupported?: MemorySystemLane[];
  usage?: MemorySystemUsage;
  wallMs?: MemorySystemWallMs;
  error?: string;
}

/** One adapter, alive across questions. Requests are answered in the order they are made. */
export interface MemorySystemClient {
  readonly id: string;
  request(request: MemorySystemRequest): Promise<MemorySystemResponse>;
  close(): Promise<void>;
}

/** What lands in the run observation next to readerUsage and judgeUsage. */
export interface MemorySystemObservation {
  id: string;
  lane: MemorySystemLane;
  usage: MemorySystemUsage | null;
  wallMs: MemorySystemWallMs | null;
  /** Sessions the adapter returned, before the harness capped them. */
  returnedSessions: number;
  /** Memory entries the adapter returned (memories lane). */
  returnedMemories: number;
  /** Memory text bytes offered, kept inside the context budget, and dropped. */
  memoryBytes: { supplied: number; kept: number; dropped: number };
  unsupported: MemorySystemLane[];
}

export interface MemorySystemUsageSummary {
  questions: number;
  modelCalls: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  medianIngestMs: number;
  medianSearchMs: number;
  droppedMemoryBytes: number;
}

/** "2023/07/10 (Mon) 09:00" and "2023-07-10T09:00:00.000Z" both become "2023-07-10". */
export function memorySystemDate(value: string): string {
  const match = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(value);
  return match === null ? value.slice(0, 10) : `${match[1]}-${match[2]}-${match[3]}`;
}

export function memorySystemRequestFor(
  instance: LongMemEvalInstance,
  lane: MemorySystemLane,
  topK: number,
): MemorySystemRequest {
  return {
    protocolVersion: MEMORY_SYSTEMS_PROTOCOL_VERSION,
    questionId: instance.question_id,
    questionType: instance.question_type,
    questionDate: memorySystemDate(instance.question_date),
    question: instance.question,
    topK,
    lanes: [lane],
    // instance.answer and instance.answer_session_ids are deliberately absent
    sessions: instance.haystack_sessions.map((turns, index) => ({
      id: instance.haystack_session_ids[index]!,
      date: memorySystemDate(instance.haystack_dates[index]!),
      turns: turns.map(({ role, content }) => ({
        role: role === 'assistant' ? ('assistant' as const) : ('user' as const),
        content,
      })),
    })),
  };
}

function objectOf(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function arrayOf(value: unknown, label: string, maximum: number): unknown[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  if (value.length > maximum) {
    throw new Error(`${label} must hold at most ${maximum} entries, got ${value.length}`);
  }
  return value;
}

function stringOf(value: unknown, label: string, maxLength: number): string {
  if (typeof value !== 'string' || value === '' || value.length > maxLength) {
    throw new Error(`${label} must be a non-empty string up to ${maxLength} characters`);
  }
  return value;
}

function finiteOf(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function nonNegativeOf(value: unknown, label: string): number {
  const parsed = finiteOf(value, label);
  if (parsed < 0) throw new Error(`${label} must not be negative`);
  return parsed;
}

function laneOf(value: unknown, label: string): MemorySystemLane {
  if (value !== 'retrieval' && value !== 'memories') {
    throw new Error(`${label} must be "retrieval" or "memories"`);
  }
  return value;
}

/**
 * Every field an adapter may send, checked. A response that names a session outside the
 * request, answers the wrong question, or reports its own error is a failed question, not a
 * silently degraded one.
 */
export function parseMemorySystemResponse(
  value: unknown,
  request: MemorySystemRequest,
): MemorySystemResponse {
  const payload = objectOf(value, 'memory system response');
  if (payload.questionId !== request.questionId) {
    throw new Error(
      `memory system answered ${String(payload.questionId)}, expected ${request.questionId}`,
    );
  }
  if (typeof payload.error === 'string' && payload.error !== '') {
    throw new Error(`memory system reported: ${payload.error.slice(0, 300)}`);
  }
  const known = new Set(request.sessions.map(({ id }) => id));
  const retrieved = arrayOf(payload.retrieved, 'retrieved', MEMORY_SYSTEM_MAX_SESSIONS).map(
    (entry, index) => {
      const item = objectOf(entry, `retrieved[${index}]`);
      const sessionId = stringOf(item.sessionId, `retrieved[${index}].sessionId`, 200);
      if (!known.has(sessionId)) {
        throw new Error(`memory system returned session ${sessionId}, which is not in the request`);
      }
      return {
        sessionId,
        rank: Math.trunc(nonNegativeOf(item.rank, `retrieved[${index}].rank`)),
        ...(item.score === undefined
          ? {}
          : { score: finiteOf(item.score, `retrieved[${index}].score`) }),
      };
    },
  );
  const memories = arrayOf(payload.memories, 'memories', MEMORY_SYSTEM_MAX_MEMORIES).map(
    (entry, index) => {
      const item = objectOf(entry, `memories[${index}]`);
      const sessionIds = arrayOf(
        item.sessionIds,
        `memories[${index}].sessionIds`,
        MEMORY_SYSTEM_MAX_SESSIONS,
      ).map((id, position) => {
        const sessionId = stringOf(id, `memories[${index}].sessionIds[${position}]`, 200);
        if (!known.has(sessionId)) {
          throw new Error(
            `memory system cited session ${sessionId}, which is not in the request`,
          );
        }
        return sessionId;
      });
      return {
        text: stringOf(item.text, `memories[${index}].text`, MEMORY_SYSTEM_MAX_MEMORY_CHARACTERS),
        ...(sessionIds.length === 0 ? {} : { sessionIds }),
        ...(item.at === undefined
          ? {}
          : { at: memorySystemDate(stringOf(item.at, `memories[${index}].at`, 40)) }),
      };
    },
  );
  const unsupported = arrayOf(payload.unsupported, 'unsupported', 2).map((lane, index) =>
    laneOf(lane, `unsupported[${index}]`),
  );
  const usage =
    payload.usage === undefined
      ? undefined
      : (() => {
          const source = objectOf(payload.usage, 'usage');
          return {
            modelCalls: nonNegativeOf(source.modelCalls ?? 0, 'usage.modelCalls'),
            inputTokens: nonNegativeOf(source.inputTokens ?? 0, 'usage.inputTokens'),
            outputTokens: nonNegativeOf(source.outputTokens ?? 0, 'usage.outputTokens'),
            costUsd: nonNegativeOf(source.costUsd ?? 0, 'usage.costUsd'),
          };
        })();
  const wallMs =
    payload.wallMs === undefined
      ? undefined
      : (() => {
          const source = objectOf(payload.wallMs, 'wallMs');
          return {
            ingest: nonNegativeOf(source.ingest ?? 0, 'wallMs.ingest'),
            search: nonNegativeOf(source.search ?? 0, 'wallMs.search'),
          };
        })();
  return {
    questionId: request.questionId,
    retrieved,
    memories,
    unsupported,
    ...(usage === undefined ? {} : { usage }),
    ...(wallMs === undefined ? {} : { wallMs }),
  };
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

export function summarizeMemorySystemUsage(
  observations: ReadonlyArray<{ memorySystem?: MemorySystemObservation }>,
): MemorySystemUsageSummary {
  const present = observations.flatMap(({ memorySystem }) =>
    memorySystem === undefined ? [] : [memorySystem],
  );
  return {
    questions: present.length,
    modelCalls: present.reduce((total, { usage }) => total + (usage?.modelCalls ?? 0), 0),
    inputTokens: present.reduce((total, { usage }) => total + (usage?.inputTokens ?? 0), 0),
    outputTokens: present.reduce((total, { usage }) => total + (usage?.outputTokens ?? 0), 0),
    costUsd: present.reduce((total, { usage }) => total + (usage?.costUsd ?? 0), 0),
    medianIngestMs: median(present.flatMap(({ wallMs }) => (wallMs === null ? [] : [wallMs.ingest]))),
    medianSearchMs: median(present.flatMap(({ wallMs }) => (wallMs === null ? [] : [wallMs.search]))),
    droppedMemoryBytes: present.reduce((total, { memoryBytes }) => total + memoryBytes.dropped, 0),
  };
}
```

- [ ] **Step 4: Write the long-lived process client**

```ts
// src/evals/memory-systems-process.ts
/**
 * One adapter process for a whole run. Spawned without a shell, fed one JSON line per
 * question, read back one JSON line per question. Requests queue, so an adapter never sees
 * two questions at once and its per-question store stays isolated. stderr is counted and
 * discarded; a response over the byte limit or past the timeout kills the child.
 * A child that dies is respawned on the next request: the protocol requires a fresh store
 * per question anyway, so nothing carries over.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import {
  parseMemorySystemResponse,
  type MemorySystemClient,
  type MemorySystemRequest,
  type MemorySystemResponse,
} from './memory-systems-protocol.js';

const DEFAULT_TIMEOUT_MS = 900_000;
const DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const CLOSE_GRACE_MS = 5_000;

export interface MemorySystemProcessOptions {
  id: string;
  executable: string;
  args?: string[];
  workingDirectory?: string;
  env?: Record<string, string>;
  /** Per request, not per process. */
  timeoutMs?: number;
  maxResponseBytes?: number;
}

export function createMemorySystemProcess(
  options: MemorySystemProcessOptions,
): MemorySystemClient {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxResponseBytes = options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES;
  let child: ChildProcessWithoutNullStreams | undefined;
  let buffer = '';
  let diagnosticBytes = 0;
  let pending:
    | {
        request: MemorySystemRequest;
        resolve: (value: MemorySystemResponse) => void;
        reject: (error: Error) => void;
        timer: NodeJS.Timeout;
      }
    | undefined;
  let queue: Promise<unknown> = Promise.resolve();

  const settle = (error?: Error, value?: MemorySystemResponse): void => {
    const current = pending;
    if (current === undefined) return;
    pending = undefined;
    clearTimeout(current.timer);
    if (error !== undefined) current.reject(error);
    else current.resolve(value!);
  };

  const consume = (): void => {
    let newline = buffer.indexOf('\n');
    while (newline >= 0) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      const current = pending;
      if (line.trim() !== '' && current !== undefined) {
        try {
          settle(undefined, parseMemorySystemResponse(JSON.parse(line) as unknown, current.request));
        } catch (error) {
          settle(error instanceof Error ? error : new Error(String(error)));
        }
      }
      newline = buffer.indexOf('\n');
    }
  };

  const start = (): ChildProcessWithoutNullStreams => {
    if (child !== undefined) return child;
    buffer = '';
    const spawned = spawn(options.executable, options.args ?? [], {
      shell: false,
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: options.workingDirectory,
      env: {
        PATH: process.env.PATH ?? '/usr/bin:/bin',
        HOME: process.env.HOME ?? '',
        LANG: process.env.LANG ?? 'C.UTF-8',
        ...options.env,
      },
    });
    spawned.stdout.setEncoding('utf8');
    spawned.stdout.on('data', (chunk: string) => {
      buffer += chunk;
      if (Buffer.byteLength(buffer, 'utf8') > maxResponseBytes) {
        buffer = '';
        spawned.kill('SIGKILL');
        settle(new Error(`${options.id} response exceeded ${maxResponseBytes} bytes`));
        return;
      }
      consume();
    });
    spawned.stderr.on('data', (chunk: Buffer) => {
      diagnosticBytes += chunk.length;
    });
    // A broken stdin pipe is not the real failure; the close handler reports that.
    spawned.stdin.on('error', () => {});
    spawned.on('error', (error) => {
      child = undefined;
      settle(error);
    });
    spawned.on('close', (code, signal) => {
      child = undefined;
      settle(
        new Error(
          `${options.id} exited with ${signal ?? code}; stderr suppressed (${diagnosticBytes} bytes)`,
        ),
      );
    });
    child = spawned;
    return spawned;
  };

  return {
    id: options.id,
    async request(request) {
      const run = queue.then(
        async () =>
          await new Promise<MemorySystemResponse>((resolve, reject) => {
            const spawned = start();
            pending = {
              request,
              resolve,
              reject,
              timer: setTimeout(() => {
                spawned.kill('SIGKILL');
                settle(
                  new Error(`${options.id} timed out after ${timeoutMs}ms on ${request.questionId}`),
                );
              }, timeoutMs),
            };
            spawned.stdin.write(`${JSON.stringify(request)}\n`);
          }),
      );
      queue = run.then(
        () => undefined,
        () => undefined,
      );
      return await run;
    },
    async close() {
      const spawned = child;
      if (spawned === undefined) return;
      child = undefined;
      spawned.stdin.end();
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          spawned.kill('SIGKILL');
          resolve();
        }, CLOSE_GRACE_MS);
        spawned.on('close', () => {
          clearTimeout(timer);
          resolve();
        });
      });
    },
  };
}
```

- [ ] **Step 5: Run the tests and the build**

Run: `npx vitest run tests/memory-systems.test.ts && npm run build:core`
Expected: 6 tests pass; `tsc` clean.

- [ ] **Step 6: Commit**

```bash
git add src/evals/memory-systems-protocol.ts src/evals/memory-systems-process.ts tests/memory-systems.test.ts
git commit -m "rembero.memory-systems.v1: the protocol and one long-lived adapter process per run"
```

---

### Task 2: The harness seam, the runner options and the three baselines

**Cost:** three subset-100 runs for the baselines. Reader is on the Ollama Cloud subscription ($0), embeddings are local ($0), judge is `deepseek-chat` at about 300 input tokens and 1 output token per question ≈ $0.01 a run. **≈ $0.03 total.**

**Files:**
- Create: `src/evals/memory-systems-builtin.ts`
- Modify: `src/evals/longmemeval-answer.ts` — the option block around `:739-779`, the `let` declarations around `:844`, the observation interface `:130-167`, the run interface `:220-268`, the search block `:1041-1351`, both return objects `:1496-1530` and `:1532-1567`, `longMemEvalAnswerRun` `:1766-1846`
- Modify: `src/evals/run-longmemeval-answer.ts` — `Args` `:27-77`, `USAGE` `:79-160`, `parseArgs` `:183-443`, `main` `:560-582` (extractor), `:585-663` (the map), `:664-701` (the run), `:728-783` (the summary)
- Modify: `src/evals/memory-stack-external.ts:13-24` and `:26-30`, `:167-192`
- Test: `tests/memory-systems.test.ts` (append)

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces, from `memory-systems-builtin.ts`: `BUILTIN_MEMORY_SYSTEMS`, `type BuiltinMemorySystemId`, `isBuiltinMemorySystem(value: string): value is BuiltinMemorySystemId`, `interface MemorySystemFactoryOptions { embeddings?: EmbeddingClient; extractor?: LongMemEvalCompletionClient; extractionCacheDir?: string; extractionCharacters?: number; extractionMaxTokens?: number; timeoutMs?: number }`, `createBuiltinMemorySystem(id: BuiltinMemorySystemId, options: MemorySystemFactoryOptions): MemorySystemClient`, `openMemorySystem(spec: string, options: MemorySystemFactoryOptions): Promise<MemorySystemClient>`.
- Produces, in `longmemeval-answer.ts`: `evaluateLongMemEvalAnswerInstance` gains `options.memorySystem?: { client: MemorySystemClient; lane: MemorySystemLane }`; `LongMemEvalAnswerObservation` gains `memorySystem?: MemorySystemObservation`; `LongMemEvalAnswerRun['retrieval']` widens with `` `memory-system:${string}` ``; `LongMemEvalAnswerRun['settings']` gains `memorySystem?: string | null` and `memoryLane?: MemorySystemLane | null`; `longMemEvalAnswerRun` gains `options.memorySystemId?: string`.
- Produces, in `memory-stack-external.ts`: `LoadedExternalAdapterManifest` gains `protocol?: 'rembero.memory-stack.v1' | 'rembero.memory-systems.v1'`; `ALLOWED_EXTERNAL_ENVIRONMENT` gains `DEEPSEEK_API_KEY` and `OLLAMA_HOST`.
- Produces, on the command line: `--memory-system <id|manifest>` and `--memory-lane <retrieval|memories>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/memory-systems.test.ts`:

```ts
import {
  evaluateLongMemEvalAnswerInstance,
  longMemEvalAnswerRun,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  createBuiltinMemorySystem,
  isBuiltinMemorySystem,
} from '../src/evals/memory-systems-builtin.js';
import type {
  MemorySystemClient,
  MemorySystemResponse,
} from '../src/evals/memory-systems-protocol.js';

class ScriptedClient implements LongMemEvalCompletionClient {
  readonly calls: ChatMessage[][] = [];
  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    this.calls.push(structuredClone(messages));
    const content = this.outputs.shift();
    if (content === undefined) throw new Error('scripted completion exhausted');
    return {
      content,
      model: this.model,
      usage: {
        promptTokens: 10,
        completionTokens: 2,
        totalTokens: 12,
        cachedPromptTokens: 0,
        reasoningTokens: 0,
        costUsd: 0.001,
      },
    };
  }
}

function fixedClient(id: string, response: Omit<MemorySystemResponse, 'questionId'>): MemorySystemClient {
  return {
    id,
    async request(request) {
      return { questionId: request.questionId, ...response };
    },
    async close() {},
  };
}

describe('the memory-system seam', () => {
  it('replaces the lexical search with the adapter ranking in the retrieval lane', async () => {
    const reader = new ScriptedClient('reader', ['Business Administration.']);
    const judge = new ScriptedClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(instance(), reader, judge, {
      topK: 4,
      contextBytes: 24_576,
      semanticQuestionTypes: new Set<string>(),
      memorySystem: {
        lane: 'retrieval',
        client: fixedClient('stub', {
          retrieved: [{ sessionId: 'evidence', rank: 1, score: 0.83 }],
          memories: [],
          unsupported: ['memories'],
          usage: { modelCalls: 48, inputTokens: 120_000, outputTokens: 3_000, costUsd: 0.03 },
          wallMs: { ingest: 41_000, search: 120 },
        }),
      },
    });
    expect(observation.status).toBe('judged');
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.retrieval?.recallAtK).toBe(1);
    expect(observation.memorySystem).toMatchObject({
      id: 'stub',
      lane: 'retrieval',
      usage: { modelCalls: 48, costUsd: 0.03 },
      wallMs: { ingest: 41_000, search: 120 },
      returnedSessions: 1,
    });
    expect(reader.calls[0]?.[1]?.content).toContain('I graduate with Business Administration');
    expect(reader.calls[0]?.[1]?.content).not.toContain('credit card rewards');
  });

  it('reads from memory text alone in the memories lane and records what it dropped', async () => {
    const reader = new ScriptedClient('reader', ['Business Administration.']);
    const judge = new ScriptedClient('judge', ['yes']);
    const filler = 'x'.repeat(3_000);
    const observation = await evaluateLongMemEvalAnswerInstance(instance(), reader, judge, {
      topK: 4,
      // small on purpose: the second memory cannot fit, so it is dropped and counted
      contextBytes: 4_096,
      semanticQuestionTypes: new Set<string>(),
      memorySystem: {
        lane: 'memories',
        client: fixedClient('stub', {
          retrieved: [],
          memories: [
            {
              text: "User's degree is Business Administration",
              sessionIds: ['evidence'],
              at: '2023-06-01',
            },
            { text: filler, sessionIds: ['noise'], at: '2023-05-15' },
          ],
          unsupported: [],
          usage: { modelCalls: 3, inputTokens: 10, outputTokens: 1, costUsd: 0.001 },
          wallMs: { ingest: 10, search: 1 },
        }),
      },
    });
    expect(observation.status).toBe('judged');
    expect(reader.calls[0]?.[1]?.content).toContain(
      "MEMORY: User's degree is Business Administration",
    );
    expect(reader.calls[0]?.[1]?.content).not.toContain('I graduate with Business Administration');
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.memorySystem?.returnedMemories).toBe(2);
    expect(observation.memorySystem?.memoryBytes.dropped).toBeGreaterThan(2_000);
    expect(observation.memorySystem?.memoryBytes.kept).toBeLessThan(4_096);
  });

  it('fails the question when the memories lane comes back empty', async () => {
    const reader = new ScriptedClient('reader', ['unused']);
    const judge = new ScriptedClient('judge', ['unused']);
    const observation = await evaluateLongMemEvalAnswerInstance(instance(), reader, judge, {
      topK: 4,
      semanticQuestionTypes: new Set<string>(),
      memorySystem: {
        lane: 'memories',
        client: fixedClient('retriever-only', {
          retrieved: [{ sessionId: 'evidence', rank: 1 }],
          memories: [],
          unsupported: ['memories'],
        }),
      },
    });
    expect(observation.status).toBe('error');
    expect(observation.error).toMatch(/memories lane is unsupported/);
  });

  it('ranks with BM25 and returns the whole haystack newest first for full context', async () => {
    expect(isBuiltinMemorySystem('builtin:bm25')).toBe(true);
    expect(isBuiltinMemorySystem('builtin:nothing')).toBe(false);
    const request = memorySystemRequestFor(instance(), 'retrieval', 2);
    const bm25 = createBuiltinMemorySystem('builtin:bm25', {});
    const ranked = await bm25.request(request);
    expect(ranked.retrieved?.[0]?.sessionId).toBe('evidence');
    expect(ranked.unsupported).toEqual(['memories']);
    const full = createBuiltinMemorySystem('builtin:full-context', {});
    const everything = await full.request(request);
    expect(everything.retrieved?.map(({ sessionId }) => sessionId)).toEqual(['evidence', 'noise']);
    await bm25.close();
    await full.close();
  });

  it('names the memory system in the run artifact without changing the schema', () => {
    const run = longMemEvalAnswerRun([], 'glm-5.3-flash:cloud', 'deepseek-chat', {
      memorySystemId: 'builtin:bm25',
      topK: 4,
      multiSessionTopK: 15,
      temporalTopK: 10,
      contextBytes: 24_576,
      settings: {
        aggregationReaderModel: null,
        temporalRangeModel: null,
        readingStrategy: 'direct',
        hybridRetrieval: 'shared',
        retrievalUnit: 'session',
        entityRetrieval: false,
        hybridQuestionTypes: null,
        factsInContext: true,
        readerMaxTokens: 440,
        memorySystem: 'builtin:bm25',
        memoryLane: 'retrieval',
      },
    });
    expect(run.retrieval).toBe('memory-system:builtin:bm25');
    expect(run.schemaVersion).toBe('remembero.longmemeval-answer.v1');
    expect(run.settings?.memoryLane).toBe('retrieval');
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run tests/memory-systems.test.ts`
Expected: FAIL, `Cannot find module '../src/evals/memory-systems-builtin.js'` and `Object literal may only specify known properties, and 'memorySystem' does not exist`.

- [ ] **Step 3: Widen the observation and run types**

In `src/evals/longmemeval-answer.ts`, add to the import block near `:55`:

```ts
import {
  MEMORY_SYSTEM_MAX_SESSIONS,
  MEMORY_SYSTEM_MEMORY_HEADROOM_BYTES,
  MEMORY_SYSTEM_UNSOURCED_SESSION,
  memorySystemRequestFor,
  type MemorySystemClient,
  type MemorySystemLane,
  type MemorySystemMemory,
  type MemorySystemObservation,
} from './memory-systems-protocol.js';
```

In `LongMemEvalAnswerObservation`, after the `engineRecall` field (`:158`):

```ts
  /** Present when an external memory layer replaced Remembero's formation and search. */
  memorySystem?: MemorySystemObservation;
```

In `LongMemEvalAnswerRun['settings']`, after `readerMaxTokens` (`:252`):

```ts
    memorySystem?: string | null;
    memoryLane?: MemorySystemLane | null;
```

Replace the `retrieval` field (`:254-255`):

```ts
  retrieval:
    | 'remembero-local-source-search'
    | 'remembero-adaptive-source-search'
    | `memory-system:${string}`;
```

In `longMemEvalAnswerRun`'s options (`:1784`) add `memorySystemId?: string;`, and replace its `retrieval` expression (`:1817-1820`):

```ts
    retrieval:
      options.memorySystemId === undefined
        ? options.embeddingModel === undefined || options.embeddingModel === null
          ? 'remembero-local-source-search'
          : 'remembero-adaptive-source-search'
        : (`memory-system:${options.memorySystemId}` as const),
```

- [ ] **Step 4: Add the option and the seam**

In the options object of `evaluateLongMemEvalAnswerInstance`, after `extractionAssistantCharacters` (`:778`):

```ts
    /**
     * An external memory layer replaces Remembero's formation and search. The haystack loop
     * still runs (it is what teaches the harness every session's text, roles and date), but
     * nothing is written into the store and the lexical search never happens: the adapter is
     * asked for this question's ranked sessions ('retrieval') or its own memory text
     * ('memories'). Everything after retrieval — the context builder, the reader, the judge
     * and the observation schema — is unchanged.
     */
    memorySystem?: { client: MemorySystemClient; lane: MemorySystemLane };
```

Next to `let engineRecall: LongMemEvalEngineRecall | undefined;` (`:844`) add:

```ts
  let memorySystem: MemorySystemObservation | undefined;
```

Change the store-write guard at `:988` from `if (formation !== 'extracted') {` to:

```ts
      // a memory system does its own formation; the loop above still fills sessionRecords
      // and userSourceText, which the context builder needs whichever layer retrieves
      if (formation !== 'extracted' && options.memorySystem === undefined) {
```

Replace `:1041` (`const snapshot = store.knowledgeSnapshot(['longmemeval']);`) with the hoisted declarations and the opening brace:

```ts
    type RankedSource = {
      opId: string;
      ts: string;
      text?: string;
      redacted?: true;
      focusCharacterOffset?: number;
      facts?: string[];
    };
    let rankedSources: RankedSource[] = [];
    const extraFacts: Array<{ clause: string; ts: string }> = [];
    let topScore = 0;
    if (options.memorySystem === undefined) {
      const snapshot = store.knowledgeSnapshot(['longmemeval']);
```

Then, inside that block (the former `:1042-1345`), make exactly three edits and indent the block one level:
- `const rankedSources = search.results.flatMap((result) => {` (`:1202`) becomes `rankedSources = search.results.flatMap((result) => {`
- delete the declaration `const extraFacts: Array<{ clause: string; ts: string }> = [];` (`:1317`), keeping the `if (reserved && factClauses.length > 0)` block that fills it
- `const topScore =` (`:1340`) becomes `topScore =`

Close the block after the `topScore` assignment and add the external branch, then the shared scoring line (replacing `:1346-1351`):

```ts
    } else {
      const { client, lane } = options.memorySystem;
      const request = memorySystemRequestFor(instance, lane, effectiveTopK);
      const askedAt = performance.now();
      const reply = await client.request(request);
      retrievalMs = performance.now() - askedAt;
      const seen = new Set<string>();
      const ranked = [...(reply.retrieved ?? [])]
        .sort((left, right) => left.rank - right.rank)
        .filter(({ sessionId }) => {
          if (seen.has(sessionId)) return false;
          seen.add(sessionId);
          return true;
        });
      topScore = ranked[0]?.score ?? 0;
      let memoryBytes = { supplied: 0, kept: 0, dropped: 0 };
      const memories = reply.memories ?? [];
      if (lane === 'retrieval') {
        rankedSources = ranked.slice(0, MEMORY_SYSTEM_MAX_SESSIONS).flatMap(({ sessionId }) => {
          const record = sessionRecords.get(sessionId);
          return record === undefined
            ? []
            : [{ opId: sessionId, ts: record.ts, text: record.text, facts: [] }];
        });
        retrievedSessionIds = rankedSources.map(({ opId }) => opId);
      } else {
        if (memories.length === 0) {
          throw new Error(
            `${client.id} returned no memory text for ${instance.question_id}: the memories lane is unsupported for it`,
          );
        }
        // Keep memory text in the order the system ranked it until the context budget is
        // full, and say exactly how many bytes were left behind.
        const budget = Math.max(1_024, contextBytes - MEMORY_SYSTEM_MEMORY_HEADROOM_BYTES);
        const kept: MemorySystemMemory[] = [];
        for (const memory of memories) {
          // "MEMORY: " and the newline the line is rendered with
          const bytes = Buffer.byteLength(memory.text, 'utf8') + 9;
          memoryBytes.supplied += bytes;
          if (memoryBytes.kept + bytes > budget) {
            memoryBytes.dropped += bytes;
            continue;
          }
          memoryBytes.kept += bytes;
          kept.push(memory);
        }
        // one pseudo-session per cited session, so the context builder, the byte budget, the
        // date distances and the computed notes work over memory text exactly as over chats
        const grouped = new Map<string, { ts: string; lines: string[] }>();
        for (const memory of kept) {
          const sessionId = memory.sessionIds?.[0] ?? MEMORY_SYSTEM_UNSOURCED_SESSION;
          const record = sessionRecords.get(sessionId);
          const ts =
            record?.ts ??
            (memory.at === undefined
              ? datasetDate(instance.question_date).toISOString()
              : `${memory.at}T09:00:00.000Z`);
          const group = grouped.get(sessionId) ?? { ts, lines: [] };
          group.lines.push(memory.text);
          grouped.set(sessionId, group);
        }
        rankedSources = [...grouped.entries()]
          .slice(0, MEMORY_SYSTEM_MAX_SESSIONS)
          .map(([sessionId, group]) => ({
            opId: sessionId,
            ts: group.ts,
            text: group.lines.map((line) => `MEMORY: ${line}`).join('\n'),
            facts: [],
          }));
        retrievedSessionIds = [
          ...new Set([
            ...kept.flatMap(({ sessionIds }) => sessionIds ?? []),
            ...ranked.map(({ sessionId }) => sessionId),
          ]),
        ];
      }
      memorySystem = {
        id: client.id,
        lane,
        usage: reply.usage ?? null,
        wallMs: reply.wallMs ?? null,
        returnedSessions: ranked.length,
        returnedMemories: memories.length,
        memoryBytes,
        unsupported: reply.unsupported ?? [],
      };
    }
    retrieval = scoreLongMemEvalRetrievedSessions(
      instance,
      retrievedSessionIds,
      retrievalMs,
      topScore,
    );
```

Finally, in **both** return objects, add next to `...(engineRecall === undefined ? {} : { engineRecall }),` (`:1522` and `:1558`):

```ts
      ...(memorySystem === undefined ? {} : { memorySystem }),
```

- [ ] **Step 5: Write the built-in memory systems and the resolver**

```ts
// src/evals/memory-systems-builtin.ts
/**
 * The rows that have no product to pin: they live in the harness because there is nothing to
 * install. full-context hands back the whole haystack newest first and lets the context
 * budget do the cutting; bm25 is textbook BM25 over whole sessions; embed is cosine over
 * whole-session embeddings from a local nomic-embed-text through Ollama. None of them calls
 * a paid model. openMemorySystem also resolves an adapter manifest to a long-lived process.
 */
import { performance } from 'node:perf_hooks';
import type { EmbeddingClient } from '../llm/embeddings.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';
import { loadExternalAdapterManifest } from './memory-stack-external.js';
import { createMemorySystemProcess } from './memory-systems-process.js';
import {
  MEMORY_SYSTEM_MAX_SESSIONS,
  MEMORY_SYSTEMS_PROTOCOL_VERSION,
  type MemorySystemClient,
  type MemorySystemRequest,
  type MemorySystemResponse,
} from './memory-systems-protocol.js';

export const BUILTIN_MEMORY_SYSTEMS = [
  'builtin:full-context',
  'builtin:bm25',
  'builtin:embed',
] as const;

export type BuiltinMemorySystemId = (typeof BUILTIN_MEMORY_SYSTEMS)[number];

export function isBuiltinMemorySystem(value: string): value is BuiltinMemorySystemId {
  return (BUILTIN_MEMORY_SYSTEMS as readonly string[]).includes(value);
}

export interface MemorySystemFactoryOptions {
  embeddings?: EmbeddingClient;
  extractor?: LongMemEvalCompletionClient;
  extractionCacheDir?: string;
  extractionCharacters?: number;
  extractionMaxTokens?: number;
  timeoutMs?: number;
}

const ZERO_USAGE = { modelCalls: 0, inputTokens: 0, outputTokens: 0, costUsd: 0 } as const;
const BM25_K1 = 1.2;
const BM25_B = 0.75;
/** nomic-embed-text takes 8192 tokens; 8,000 characters is comfortably inside it. */
const EMBED_CHARACTERS = 8_000;
const EMBED_BATCH = 16;

export function sessionText(session: MemorySystemRequest['sessions'][number]): string {
  return session.turns.map(({ role, content }) => `${role}: ${content}`).join('\n');
}

function words(text: string): string[] {
  return (text.toLowerCase().normalize('NFKC').match(/[\p{L}\p{N}_]+/gu) ?? []).filter(
    (word) => word.length > 1,
  );
}

function fullContext(request: MemorySystemRequest): MemorySystemResponse {
  // newest first; the context builder's byte budget is what truncates the haystack. The cap
  // is the protocol's own limit, and also what buildLongMemEvalAnswerContext will accept.
  const ordered = [...request.sessions]
    .sort((left, right) => right.date.localeCompare(left.date) || left.id.localeCompare(right.id))
    .slice(0, MEMORY_SYSTEM_MAX_SESSIONS);
  return {
    questionId: request.questionId,
    retrieved: ordered.map(({ id }, index) => ({
      sessionId: id,
      rank: index + 1,
      score: ordered.length - index,
    })),
    memories: [],
    unsupported: ['memories'],
    usage: { ...ZERO_USAGE },
    wallMs: { ingest: 0, search: 0 },
  };
}

function bm25(request: MemorySystemRequest): MemorySystemResponse {
  const started = performance.now();
  const documents = request.sessions.map((session) => ({
    id: session.id,
    terms: words(sessionText(session)),
  }));
  const averageLength =
    documents.reduce((total, { terms }) => total + terms.length, 0) /
    Math.max(1, documents.length);
  const documentFrequency = new Map<string, number>();
  for (const { terms } of documents) {
    for (const term of new Set(terms)) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
  }
  const query = [...new Set(words(request.question))];
  const ranked = documents
    .map(({ id, terms }) => {
      const counts = new Map<string, number>();
      for (const term of terms) counts.set(term, (counts.get(term) ?? 0) + 1);
      let score = 0;
      for (const term of query) {
        const frequency = counts.get(term);
        if (frequency === undefined) continue;
        const n = documentFrequency.get(term) ?? 0;
        const idf = Math.log(1 + (documents.length - n + 0.5) / (n + 0.5));
        score +=
          idf *
          ((frequency * (BM25_K1 + 1)) /
            (frequency +
              BM25_K1 * (1 - BM25_B + (BM25_B * terms.length) / Math.max(1, averageLength))));
      }
      return { id, score };
    })
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id))
    .slice(0, request.topK);
  return {
    questionId: request.questionId,
    retrieved: ranked.map(({ id, score }, index) => ({ sessionId: id, rank: index + 1, score })),
    memories: [],
    unsupported: ['memories'],
    usage: { ...ZERO_USAGE },
    wallMs: { ingest: 0, search: performance.now() - started },
  };
}

function cosine(left: readonly number[], right: readonly number[]): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  const magnitude = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return magnitude === 0 ? 0 : dot / magnitude;
}

async function embedded(
  request: MemorySystemRequest,
  embeddings: EmbeddingClient,
): Promise<MemorySystemResponse> {
  const ingestStarted = performance.now();
  const texts = request.sessions.map((session) => sessionText(session).slice(0, EMBED_CHARACTERS));
  const vectors: number[][] = [];
  for (let index = 0; index < texts.length; index += EMBED_BATCH) {
    const batch = await embeddings.embed(texts.slice(index, index + EMBED_BATCH));
    vectors.push(...batch.vectors);
  }
  const ingest = performance.now() - ingestStarted;
  const searchStarted = performance.now();
  const query = (await embeddings.embed([request.question])).vectors[0] ?? [];
  const ranked = request.sessions
    .map((session, index) => ({ id: session.id, score: cosine(query, vectors[index] ?? []) }))
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id))
    .slice(0, request.topK);
  return {
    questionId: request.questionId,
    retrieved: ranked.map(({ id, score }, index) => ({ sessionId: id, rank: index + 1, score })),
    memories: [],
    unsupported: ['memories'],
    // local Ollama: tokens are not billed and the provider reports no cost
    usage: { ...ZERO_USAGE },
    wallMs: { ingest, search: performance.now() - searchStarted },
  };
}

export function createBuiltinMemorySystem(
  id: BuiltinMemorySystemId,
  options: MemorySystemFactoryOptions,
): MemorySystemClient {
  if (id === 'builtin:embed' && options.embeddings === undefined) {
    throw new Error('builtin:embed needs an embedding client');
  }
  return {
    id,
    async request(request) {
      if (id === 'builtin:full-context') return fullContext(request);
      if (id === 'builtin:bm25') return bm25(request);
      return await embedded(request, options.embeddings!);
    },
    async close() {},
  };
}

/**
 * `builtin:*` or a path to an adapter manifest whose `protocol` is
 * rembero.memory-systems.v1. The manifest's own timeout and output cap are honoured.
 */
export async function openMemorySystem(
  spec: string,
  options: MemorySystemFactoryOptions,
): Promise<MemorySystemClient> {
  if (spec.startsWith('builtin:')) {
    if (!isBuiltinMemorySystem(spec)) {
      throw new Error(
        `unknown built-in memory system ${spec}; known: ${BUILTIN_MEMORY_SYSTEMS.join(', ')}`,
      );
    }
    return createBuiltinMemorySystem(spec, options);
  }
  const manifest = await loadExternalAdapterManifest(spec);
  if (manifest.protocol !== MEMORY_SYSTEMS_PROTOCOL_VERSION) {
    throw new Error(
      `${spec} declares protocol ${String(manifest.protocol)}; --memory-system needs ${MEMORY_SYSTEMS_PROTOCOL_VERSION}`,
    );
  }
  return createMemorySystemProcess({
    id: manifest.descriptor.id,
    executable: manifest.command.executable,
    ...(manifest.command.args === undefined ? {} : { args: manifest.command.args }),
    ...(manifest.command.workingDirectory === undefined
      ? {}
      : { workingDirectory: manifest.command.workingDirectory }),
    ...(manifest.command.env === undefined ? {} : { env: manifest.command.env }),
    ...(options.timeoutMs === undefined
      ? manifest.command.timeoutMs === undefined
        ? {}
        : { timeoutMs: manifest.command.timeoutMs }
      : { timeoutMs: options.timeoutMs }),
    ...(manifest.command.maxOutputBytes === undefined
      ? {}
      : { maxResponseBytes: manifest.command.maxOutputBytes }),
  });
}
```

- [ ] **Step 6: Teach the manifest loader the protocol field**

In `src/evals/memory-stack-external.ts`, add to `ALLOWED_EXTERNAL_ENVIRONMENT` (`:13-24`) two entries — `DEEPSEEK_API_KEY` (Task 5's Mem0 and Graphiti bridges) and `OLLAMA_HOST`:

```ts
  'DEEPSEEK_API_KEY',
  'OLLAMA_HOST',
```

Add to `LoadedExternalAdapterManifest` (`:26-30`):

```ts
  /** Absent means the conformance suite's one-process-per-case shape. */
  protocol?: 'rembero.memory-stack.v1' | 'rembero.memory-systems.v1';
```

And in the returned object of `loadExternalAdapterManifest` (before `manifestPath` at `:191`):

```ts
    ...(root.protocol === undefined
      ? {}
      : {
          protocol: ((): 'rembero.memory-stack.v1' | 'rembero.memory-systems.v1' => {
            if (
              root.protocol !== 'rembero.memory-stack.v1' &&
              root.protocol !== 'rembero.memory-systems.v1'
            ) {
              throw new Error(`unsupported adapter protocol: ${String(root.protocol)}`);
            }
            return root.protocol;
          })(),
        }),
```

- [ ] **Step 7: Add the runner options**

In `src/evals/run-longmemeval-answer.ts`:

Imports:

```ts
import { openMemorySystem } from './memory-systems-builtin.js';
import {
  summarizeMemorySystemUsage,
  type MemorySystemClient,
  type MemorySystemLane,
} from './memory-systems-protocol.js';
```

`Args` gains:

```ts
  memorySystem: string | undefined;
  memoryLane: MemorySystemLane;
```

`USAGE` gains, after the `--local-only` line:

```
  --memory-system <spec>  Replace Remembero's formation and search with another memory
                         layer: builtin:full-context, builtin:bm25, builtin:embed, or a path
                         to an adapter manifest whose protocol is rembero.memory-systems.v1.
                         One adapter process per --concurrency worker, alive for the run
  --memory-lane <retrieval|memories>  retrieval (default): the adapter ranks sessions and the
                         reader sees the same raw sessions it sees for Remembero. memories:
                         the adapter returns its own memory text and the reader answers from
                         that alone, inside the same byte budget
```

Defaults in `parseArgs`: `memorySystem: undefined,` and `memoryLane: 'retrieval',`. Parsing, next to `--local-only`:

```ts
    } else if (arg === '--memory-system') {
      args.memorySystem = requiredValue(argv, index++, arg);
    } else if (arg === '--memory-lane') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'retrieval' && value !== 'memories') {
        throw new Error('--memory-lane must be retrieval or memories');
      }
      args.memoryLane = value;
```

After `parseArgs` returns in `main`, add the refusals (the constants the benchmark holds fixed):

```ts
  if (args.memorySystem !== undefined) {
    if (args.formation !== 'raw') {
      throw new Error('--memory-system does its own formation; drop --formation');
    }
    if (args.engineRecall || args.entityRetrieval) {
      throw new Error('--memory-system cannot be combined with --engine-recall or --entity-retrieval');
    }
    if (args.temporalRangeModel !== undefined) {
      throw new Error('--memory-system cannot be combined with --temporal-range-model: time-aware retrieval is off for every system');
    }
    if (args.semanticQuestionTypes.size > 0) {
      throw new Error('--memory-system needs --local-only: the memory layer under test does the retrieval');
    }
  }
```

Open one client per concurrency worker (an adapter answers one question at a time, so a single process would serialize the whole run), replacing the `let completed = 0;` line at `:585`:

```ts
  // one adapter process per worker: each client queues its own requests, so a question is
  // never interleaved with another inside one store
  const memorySystems: MemorySystemClient[] =
    args.memorySystem === undefined
      ? []
      : await Promise.all(
          Array.from({ length: args.concurrency }, async () =>
            await openMemorySystem(args.memorySystem!, {
              ...(embeddings === undefined ? {} : { embeddings }),
            }),
          ),
        );
  let completed = 0;
```

Change the `mapConcurrent` callback signature at `:589` from `async (instance) => {` to `async (instance, index) => {` and add to the options object passed to `evaluateLongMemEvalAnswerInstance`, next to `structuredEvidence: args.structuredEvidence,`:

```ts
          ...(memorySystems.length === 0
            ? {}
            : {
                memorySystem: {
                  client: memorySystems[index % memorySystems.length]!,
                  lane: args.memoryLane,
                },
              }),
```

Wrap the `await mapConcurrent(...)` assignment in `try { … } finally { await Promise.all(memorySystems.map(async (client) => await client.close())); }`.

In the `longMemEvalAnswerRun` call, add `...(args.memorySystem === undefined ? {} : { memorySystemId: args.memorySystem }),` and inside `settings`:

```ts
      memorySystem: args.memorySystem ?? null,
      memoryLane: args.memorySystem === undefined ? null : args.memoryLane,
```

In the summary block, after the judge line:

```ts
    if (args.memorySystem !== undefined) {
      const adapter = summarizeMemorySystemUsage(observations);
      console.log(`memory system: ${args.memorySystem} (lane ${args.memoryLane})`);
      console.log(
        `memory system calls/tokens/cost: ${adapter.modelCalls} / ${adapter.inputTokens + adapter.outputTokens} / $${adapter.costUsd.toFixed(6)}`,
      );
      console.log(
        `memory system ingest/search p50: ${adapter.medianIngestMs.toFixed(1)} / ${adapter.medianSearchMs.toFixed(1)} ms; dropped memory bytes ${adapter.droppedMemoryBytes}`,
      );
      const overreach = observations.filter(
        (observation) =>
          observation.memorySystem !== undefined &&
          observation.memorySystem.returnedSessions > args.multiSessionTopK,
      ).length;
      if (overreach > 0) {
        console.log(
          `note: ${overreach} questions returned more sessions than the requested top-k (full context does this by design)`,
        );
      }
    }
```

- [ ] **Step 8: Run the tests and the build**

Run: `npx vitest run tests/memory-systems.test.ts tests/longmemeval-answer.test.ts && npm run build:core`
Expected: all pass; `tsc` clean.

- [ ] **Step 9: Run the three baselines on the subset**

Start the reader and the embedding model once (`ollama serve` must be running and signed in to Ollama Cloud; `ollama pull nomic-embed-text` if it is not local yet). Then:

```bash
set -a; source .env; set +a
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=$DEEPSEEK_API_KEY
export REMBERO_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
export REMBERO_EMBEDDING_MODEL=nomic-embed-text
COMMON="--split all --cases $(cat .cache/longmemeval/subset-100.txt) --local-only \
  --top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --context-bytes 24576 --date-distances \
  --reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 \
  --reader-api-key ollama --reader-max-tokens 440 --judge-model deepseek-chat --concurrency 4"
for system in full-context bm25 embed; do
  node dist/evals/run-longmemeval-answer.js $COMMON \
    --memory-system builtin:$system --memory-lane retrieval \
    --output docs/research/results/memory-systems-$system-retrieval-subset100.json
done
```

Expected: 100 judged per run, `summary.errors` 0, `retrieval` reading `memory-system:builtin:<system>` in each file. BM25 and embed report recall@k well under 1; full context reports recall@k at or near 1 with every session in the context and each cut to roughly 500 bytes — that is the point of the row, and the document says so. Note the three accuracies; they are the floor the real systems must clear.

- [ ] **Step 10: Commit**

```bash
git add src/evals/memory-systems-builtin.ts src/evals/longmemeval-answer.ts \
  src/evals/run-longmemeval-answer.ts src/evals/memory-stack-external.ts \
  tests/memory-systems.test.ts docs/research/results/memory-systems-*-retrieval-subset100.json
git commit -m "--memory-system swaps the memory layer inside the answer harness; full-context, BM25 and embedding baselines measured on the subset"
```

---

### Task 3: Remembero's own rows through the same option

**Cost:** four subset-100 runs, judge only ≈ $0.04. The hybrid rows replay `.cache/longmemeval/extraction-r23` (4,515 cached extractions), so the r23 writer makes no new calls if the cache covers the subset; misses cost local GPU time, not money.

**Files:**
- Modify: `src/evals/memory-systems-builtin.ts` (two more ids and the shared formation helper)
- Test: `tests/memory-systems.test.ts` (append)
- Create: `scripts/compare-retrieved-sessions.mjs`
- Create: `docs/research/results/memory-systems-remembero-{retrieval-subset100,memories-subset100}.json` and the two paired stock-path runs

**Interfaces:**
- Consumes: `MemorySystemFactoryOptions`, `createBuiltinMemorySystem`, `openMemorySystem` (Task 2); `MemoryStore`, `searchKnowledge`, `rememberTranscriptText`, `LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS`, `DEFAULT_LONGMEMEVAL_EXTRACTION_CHARACTERS` from the product.
- Produces: `BUILTIN_MEMORY_SYSTEMS` gains `'builtin:remembero-raw'` and `'builtin:remembero-hybrid'`; `scripts/compare-retrieved-sessions.mjs <a.json> <b.json>` exits non-zero and prints the differing question ids when the two runs did not retrieve the same sessions.

**Ambiguity resolved here:** no stored run uses this benchmark's reader, so "reproduces the stored numbers within ±4" is measured against a *paired stock-path run made in this task* — same reader, judge, depth and flags, with and without `--memory-system`. The raw lane has a stricter gate than ±4: it must retrieve the **identical** session ids on every question, because `builtin:remembero-raw` calls the same `searchKnowledge` over the same documents.

- [ ] **Step 1: Write the failing test**

Append to `tests/memory-systems.test.ts`:

```ts
describe('Remembero as a memory system', () => {
  it('ranks with its own lexical source search in the retrieval lane', async () => {
    const client = createBuiltinMemorySystem('builtin:remembero-raw', {});
    try {
      const reply = await client.request(memorySystemRequestFor(instance(), 'retrieval', 4));
      expect(reply.retrieved?.[0]?.sessionId).toBe('evidence');
      expect(reply.unsupported).toEqual(['memories']);
      expect(reply.usage).toEqual({
        modelCalls: 0,
        inputTokens: 0,
        outputTokens: 0,
        costUsd: 0,
      });
    } finally {
      await client.close();
    }
  });

  it('returns the writer facts it matched as memory text in the memories lane', async () => {
    const extractor = new ScriptedClient('writer', [
      JSON.stringify({ facts: ['graduated(user, business_administration).'] }),
      JSON.stringify({ facts: [] }),
    ]);
    const client = createBuiltinMemorySystem('builtin:remembero-hybrid', { extractor });
    try {
      const reply = await client.request(memorySystemRequestFor(instance(), 'memories', 4));
      expect(reply.memories?.some(({ text }) => text.includes('business_administration'))).toBe(
        true,
      );
      expect(reply.memories?.[0]?.sessionIds).toEqual(['evidence']);
      expect(reply.retrieved?.[0]?.sessionId).toBe('evidence');
    } finally {
      await client.close();
    }
  });

  it('refuses the hybrid row without a writer', () => {
    expect(() => createBuiltinMemorySystem('builtin:remembero-hybrid', {})).toThrow(
      /needs an extraction client/,
    );
  });
});
```

Note: the scripted writer's output shape must match what `rememberTranscriptText` parses; if the two scripted strings do not satisfy the parser, read `tests/extraction-evals.test.ts` for the exact JSON the pipeline accepts and use that instead. The assertions on `reply.memories` do not change.

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run tests/memory-systems.test.ts -t "Remembero as a memory system"`
Expected: FAIL, `unknown built-in memory system builtin:remembero-raw`.

- [ ] **Step 3: Implement the two Remembero rows**

In `src/evals/memory-systems-builtin.ts`, extend the id list:

```ts
export const BUILTIN_MEMORY_SYSTEMS = [
  'builtin:full-context',
  'builtin:bm25',
  'builtin:embed',
  'builtin:remembero-raw',
  'builtin:remembero-hybrid',
] as const;
```

Add the imports:

```ts
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { searchKnowledge } from '../knowledge/search.js';
import { rememberTranscriptText } from '../llm/pipeline.js';
import { MemoryStore } from '../store/store.js';
import {
  DEFAULT_LONGMEMEVAL_EXTRACTION_CHARACTERS,
  LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS,
} from './longmemeval-answer.js';
import type { ChatMessage } from '../llm/client.js';
```

And the implementation:

```ts
/**
 * Remembero as a row in its own benchmark: a fresh MemoryStore per question, one placeholder
 * fact per session carrying the transcript as source text (raw formation), optionally the
 * r23 writer's extracted facts on top (hybrid formation), then the product's own lexical
 * source search. These are the same MemoryStore, rememberTranscriptText and searchKnowledge
 * the native harness path calls, with the same limit, minimum score and source character
 * limit, so the raw row must retrieve exactly what the native path retrieves.
 */
async function remberoMemory(
  request: MemorySystemRequest,
  options: MemorySystemFactoryOptions,
  hybrid: boolean,
): Promise<MemorySystemResponse> {
  const root = mkdtempSync(join(tmpdir(), 'remembero-memory-system-'));
  const usage = { modelCalls: 0, inputTokens: 0, outputTokens: 0, costUsd: 0 };
  try {
    const store = new MemoryStore(root);
    const ingestStarted = performance.now();
    const sessionOf = new Map<string, string>();
    const extractor = options.extractor;
    const writer =
      extractor === undefined
        ? undefined
        : {
            complete: async (messages: ChatMessage[]): Promise<string> => {
              const completion = await extractor.completeWithUsage(messages, {
                maxTokens: options.extractionMaxTokens ?? 512,
              });
              usage.modelCalls += 1;
              usage.inputTokens += completion.usage.promptTokens;
              usage.outputTokens += completion.usage.completionTokens;
              usage.costUsd += completion.usage.costUsd;
              return completion.content;
            },
          };
    for (const [index, session] of request.sessions.entries()) {
      const opId = `longmemeval:${index}:${session.id}`;
      sessionOf.set(opId, session.id);
      const text = sessionText(session);
      const at = new Date(`${session.date}T09:00:00.000Z`);
      if (hybrid && writer !== undefined) {
        const factsOperationId = `${opId}:facts`;
        sessionOf.set(factsOperationId, session.id);
        const transcript = text.slice(
          0,
          options.extractionCharacters ?? DEFAULT_LONGMEMEVAL_EXTRACTION_CHARACTERS,
        );
        const cachePath =
          options.extractionCacheDir === undefined
            ? undefined
            : join(
                options.extractionCacheDir,
                `${createHash('sha256')
                  .update(`${extractor!.model}\n${transcript}`)
                  .digest('hex')
                  .slice(0, 40)}.json`,
              );
        const cached =
          cachePath !== undefined && existsSync(cachePath)
            ? (JSON.parse(readFileSync(cachePath, 'utf8')) as { facts: string[]; error?: string })
            : undefined;
        if (cached !== undefined) {
          if (cached.error === undefined && cached.facts.length > 0) {
            store.assert('longmemeval', cached.facts.join('\n'), {
              opId: factsOperationId,
              sourceText: text,
              at,
            });
          }
        } else {
          try {
            const result = await rememberTranscriptText(
              { store, llm: writer },
              transcript,
              'longmemeval',
              { captureId: opId, opId: factsOperationId, sourceText: text, origin: 'manual', at },
            );
            if (cachePath !== undefined) {
              mkdirSync(options.extractionCacheDir!, { recursive: true });
              writeFileSync(cachePath, JSON.stringify({ facts: result.added }));
            }
          } catch (error) {
            // a refused extraction leaves the session raw, exactly as the native path does
            if (cachePath !== undefined) {
              mkdirSync(options.extractionCacheDir!, { recursive: true });
              writeFileSync(
                cachePath,
                JSON.stringify({ facts: [], error: error instanceof Error ? error.message : 'error' }),
              );
            }
          }
        }
      }
      store.assert('longmemeval', `longmem_session(session_${index}).`, {
        opId,
        sourceText: text,
        at,
      });
    }
    const ingest = performance.now() - ingestStarted;
    const searchStarted = performance.now();
    const snapshot = store.knowledgeSnapshot(['longmemeval']);
    const search = searchKnowledge(snapshot.clauses, request.question, snapshot.sources, {
      limit: Math.min(100, hybrid ? request.topK * 8 : request.topK),
      minimumScore: 1,
      kinds: ['fact'],
      sourceCharacterLimit: LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS,
    });
    const seen = new Set<string>();
    const retrieved: MemorySystemResponse['retrieved'] = [];
    const memories: MemorySystemResponse['memories'] = [];
    for (const result of search.results) {
      const source = result.sources[0];
      if (source === undefined || source.redacted === true) continue;
      const sessionId = sessionOf.get(source.opId) ?? source.opId;
      const isPlaceholder = result.clause.startsWith('longmem_session(');
      if (!isPlaceholder && memories.length < 200) {
        memories.push({
          text: result.clause,
          sessionIds: [sessionId],
          at: source.ts.slice(0, 10),
        });
      }
      if (seen.has(sessionId) || retrieved.length >= request.topK) continue;
      seen.add(sessionId);
      retrieved.push({ sessionId, rank: retrieved.length + 1, score: result.score });
    }
    return {
      questionId: request.questionId,
      retrieved,
      memories,
      unsupported: hybrid ? [] : ['memories'],
      usage,
      wallMs: { ingest, search: performance.now() - searchStarted },
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}
```

In `createBuiltinMemorySystem`, replace the guard and the dispatch:

```ts
  if (id === 'builtin:embed' && options.embeddings === undefined) {
    throw new Error('builtin:embed needs an embedding client');
  }
  if (id === 'builtin:remembero-hybrid' && options.extractor === undefined) {
    throw new Error('builtin:remembero-hybrid needs an extraction client (--extraction-model)');
  }
  return {
    id,
    async request(request) {
      if (id === 'builtin:full-context') return fullContext(request);
      if (id === 'builtin:bm25') return bm25(request);
      if (id === 'builtin:embed') return await embedded(request, options.embeddings!);
      return await remberoMemory(request, options, id === 'builtin:remembero-hybrid');
    },
    async close() {},
  };
```

In `run-longmemeval-answer.ts`, the extractor is currently created only when the formation is not raw; the Remembero hybrid row uses raw formation with `--extraction-model`. Change the condition at `:560-562` from

```ts
  const extractor =
    args.formation === 'raw'
      ? undefined
```

to

```ts
  const extractor =
    args.formation === 'raw' && args.extractionModel === undefined
      ? undefined
```

and pass the extraction settings into `openMemorySystem`:

```ts
            await openMemorySystem(args.memorySystem!, {
              ...(embeddings === undefined ? {} : { embeddings }),
              ...(extractor === undefined ? {} : { extractor }),
              ...(args.extractionCacheDir === undefined
                ? {}
                : { extractionCacheDir: args.extractionCacheDir }),
              ...(args.extractionCharacters === undefined
                ? {}
                : { extractionCharacters: args.extractionCharacters }),
              ...(args.extractionMaxTokens === undefined
                ? {}
                : { extractionMaxTokens: args.extractionMaxTokens }),
            }),
```

- [ ] **Step 4: Write the retrieval comparison script**

```js
// scripts/compare-retrieved-sessions.mjs
// Usage: node scripts/compare-retrieved-sessions.mjs <run-a.json> <run-b.json>
// Exits 1 and lists the question ids whose retrieved session ids differ.
import { readFileSync } from 'node:fs';

const [a, b] = process.argv.slice(2);
if (a === undefined || b === undefined) {
  console.error('usage: compare-retrieved-sessions.mjs <run-a.json> <run-b.json>');
  process.exit(2);
}
const load = (path) =>
  new Map(
    JSON.parse(readFileSync(path, 'utf8')).observations.map((observation) => [
      observation.questionId,
      observation.retrievedSessionIds.join(','),
    ]),
  );
const left = load(a);
const right = load(b);
const differing = [...left.entries()].filter(([id, ids]) => right.get(id) !== ids);
console.log(`${left.size} questions; ${differing.length} differ`);
for (const [id, ids] of differing.slice(0, 20)) {
  console.log(`  ${id}\n    a: ${ids}\n    b: ${right.get(id) ?? '(absent)'}`);
}
process.exit(differing.length === 0 ? 0 : 1);
```

- [ ] **Step 5: Run the tests and the build**

Run: `npx vitest run tests/memory-systems.test.ts && npm run build:core`
Expected: all pass; `tsc` clean.

- [ ] **Step 6: The raw lane, paired**

```bash
set -a; source .env; set +a
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=$DEEPSEEK_API_KEY
COMMON="--split all --cases $(cat .cache/longmemeval/subset-100.txt) --local-only \
  --top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --context-bytes 24576 --date-distances \
  --reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 \
  --reader-api-key ollama --reader-max-tokens 440 --judge-model deepseek-chat --concurrency 4"
node dist/evals/run-longmemeval-answer.js $COMMON \
  --output docs/research/results/memory-systems-remembero-native-raw-subset100.json
node dist/evals/run-longmemeval-answer.js $COMMON \
  --memory-system builtin:remembero-raw --memory-lane retrieval \
  --output docs/research/results/memory-systems-remembero-retrieval-subset100.json
node scripts/compare-retrieved-sessions.mjs \
  docs/research/results/memory-systems-remembero-native-raw-subset100.json \
  docs/research/results/memory-systems-remembero-retrieval-subset100.json
```

Expected: `100 questions; 0 differ` and exit 0 — the seam retrieves exactly what the native path retrieves. Accuracy differs only by reader sampling; a gap beyond ±4 means the context builder is seeing different text, so diff one question's `contextSessionIds` before going on.

- [ ] **Step 7: The as-shipped lane, paired**

Serve the r23 writer on port 8081 first (see `docs/research/READER-STRUCTURE.md` for the serve line), then:

```bash
WRITER="--extraction-model rembero-writer --extraction-base-url http://127.0.0.1:8081/v1 \
  --extraction-cache .cache/longmemeval/extraction-r23 --extraction-max-tokens 512"
node dist/evals/run-longmemeval-answer.js $COMMON --formation hybrid $WRITER \
  --computed-notes --structured-evidence \
  --output docs/research/results/memory-systems-remembero-native-hybrid-subset100.json
node dist/evals/run-longmemeval-answer.js $COMMON $WRITER \
  --memory-system builtin:remembero-hybrid --memory-lane memories --computed-notes \
  --output docs/research/results/memory-systems-remembero-memories-subset100.json
```

Expected: both 100 judged, `summary.errors` 0. The stored r23 hybrid subset numbers under reader v4 were 74 baseline, 73 notes, 71 evidence, 75 both — context, not a target, because the reader differs. The gate here is that the memories-lane run lands within ±4 of the native hybrid run made two commands earlier. If it is far below, read one prompt: the memories lane shows fact lines only, and the likely cause is that too few facts matched, which the run's `memorySystem.returnedMemories` shows directly.

- [ ] **Step 8: Commit**

```bash
git add src/evals/memory-systems-builtin.ts src/evals/run-longmemeval-answer.ts \
  scripts/compare-retrieved-sessions.mjs tests/memory-systems.test.ts \
  docs/research/results/memory-systems-remembero-*.json
git commit -m "Remembero measured as a row in its own benchmark: raw retrieval and as-shipped memories through --memory-system"
```

---

### Task 4: The LangGraph and LlamaIndex bridges on the long-lived protocol

**Cost:** $0 in provider spend (local CPU embeddings, no LLM). Judge for two subset runs ≈ $0.02. Wall time: the first run of each bridge downloads the pinned 67 MB bge-small model once; ingest is about 47 sessions of local embedding per question, so budget 45–90 minutes per subset run at `--concurrency 4`.

**Files:**
- Create: `benchmarks/adapters/langgraph-fastembed/memory_systems_bridge.py`, `benchmarks/adapters/langgraph-fastembed/memory-systems.json`
- Create: `benchmarks/adapters/llamaindex-fastembed/memory_systems_bridge.py`, `benchmarks/adapters/llamaindex-fastembed/memory-systems.json`

**Interfaces:**
- Consumes: the request and response shapes of `rembero.memory-systems.v1` (Task 1), `--memory-system <manifest>` (Task 2), `loadExternalAdapterManifest`'s `protocol` field (Task 2).
- Produces: two manifests whose `command.args` are `["run", "--script", "memory_systems_bridge.py"]` and whose `protocol` is `"rembero.memory-systems.v1"`. Both bridges answer the retrieval lane and report `"unsupported": ["memories"]` — they are pure retrievers and get no proxy.

- [ ] **Step 1: Write the LangGraph bridge**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "fastembed==0.8.0",
#   "langgraph==1.2.10",
# ]
# ///
"""rembero.memory-systems.v1 bridge for LangGraph's InMemoryStore semantic search.

One process for the whole run, one JSON line per question on stdin, one on stdout. A fresh
namespace per question is a fresh store: nothing a question ingests is visible to the next.
No LLM: the memory is the session text, indexed by local FastEmbed bge-small."""

import json
import os
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

PROTOCOL_VERSION = "rembero.memory-systems.v1"
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
ZERO_USAGE = {"modelCalls": 0, "inputTokens": 0, "outputTokens": 0, "costUsd": 0.0}


class FastEmbedEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.model = TextEmbedding(model_name=MODEL_ID, providers=["CPUExecutionProvider"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed([text])).tolist()


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def answer(request: dict[str, Any], embeddings: FastEmbedEmbeddings) -> dict[str, Any]:
    ingest_started = perf_counter()
    store = InMemoryStore(
        index={"dims": EMBEDDING_DIMENSIONS, "embed": embeddings, "fields": ["text"]}
    )
    namespace = ("rembero-memory-systems", str(request["questionId"]))
    for session in request["sessions"]:
        store.put(
            namespace,
            str(session["id"]),
            {"text": session_text(session), "date": str(session["date"])},
        )
    ingest_ms = (perf_counter() - ingest_started) * 1000
    search_started = perf_counter()
    top_k_value = request.get("topK", 5)
    top_k = top_k_value if isinstance(top_k_value, int) and top_k_value > 0 else 5
    matches = store.search(namespace, query=str(request["question"]), limit=top_k)
    search_ms = (perf_counter() - search_started) * 1000
    return {
        "questionId": str(request["questionId"]),
        "retrieved": [
            {
                "sessionId": str(item.key),
                "rank": index + 1,
                "score": float(getattr(item, "score", 0.0) or 0.0),
            }
            for index, item in enumerate(matches)
        ],
        "memories": [],
        "unsupported": ["memories"],
        "usage": dict(ZERO_USAGE),
        "wallMs": {"ingest": ingest_ms, "search": search_ms},
    }


def main() -> None:
    embeddings = FastEmbedEmbeddings()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        question_id = str(request.get("questionId", ""))
        if request.get("protocolVersion") != PROTOCOL_VERSION:
            response: dict[str, Any] = {
                "questionId": question_id,
                "error": f"unsupported protocol version {request.get('protocolVersion')!r}",
            }
        else:
            try:
                response = answer(request, embeddings)
            except Exception as error:  # one bad question must not end the run
                response = {
                    "questionId": question_id,
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the LangGraph manifest**

```json
{
  "schemaVersion": "rembero.memory-stack-adapter.v1",
  "protocol": "rembero.memory-systems.v1",
  "adapter": {
    "id": "langgraph-inmemory-fastembed",
    "version": "langgraph-1.2.10_fastembed-0.8.0_bge-small-en-v1.5",
    "capabilities": {
      "answerRows": false,
      "rankedRetrieval": true,
      "citations": false,
      "rules": false,
      "temporalUpdates": false,
      "trustViews": false
    },
    "disclosures": {
      "packages": { "langgraph": "1.2.10", "fastembed": "0.8.0" },
      "embeddingModel": "BAAI/bge-small-en-v1.5 (FastEmbed ONNX, 384 dimensions, CPU)",
      "storage": "LangGraph InMemoryStore; a fresh namespace per benchmark question inside one long-lived process",
      "writePolicy": "Store each haystack session's whole transcript under its session id; no LLM runs at ingest",
      "retrievalPolicy": "LangGraph vector search over the text field with the question as query and the request's topK",
      "providerCostBoundary": "Local CPU embeddings only; no LLM, no API key, no remote inference provider",
      "notes": [
        "Retrieval lane only: a pure retriever returns unsupported for the memories lane rather than a proxy.",
        "First execution downloads the pinned 67 MB embedding model; that download is outside the reported wallMs."
      ]
    }
  },
  "command": {
    "executable": "uv",
    "args": ["run", "--script", "memory_systems_bridge.py"],
    "workingDirectory": ".",
    "timeoutMs": 900000,
    "maxOutputBytes": 4000000
  }
}
```

- [ ] **Step 3: Write the LlamaIndex bridge**

Identical structure; only `answer` and the imports differ:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "fastembed==0.8.0",
#   "llama-index-core==0.14.23",
#   "llama-index-embeddings-fastembed==0.6.0",
# ]
# ///
"""rembero.memory-systems.v1 bridge for LlamaIndex VectorMemory over SimpleVectorStore.

One process for the whole run; a fresh VectorMemory per question. No LLM."""

import json
import os
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import VectorMemory
from llama_index.embeddings.fastembed import FastEmbedEmbedding

PROTOCOL_VERSION = "rembero.memory-systems.v1"
MODEL_ID = "BAAI/bge-small-en-v1.5"
ZERO_USAGE = {"modelCalls": 0, "inputTokens": 0, "outputTokens": 0, "costUsd": 0.0}


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def answer(request: dict[str, Any], embedding: FastEmbedEmbedding) -> dict[str, Any]:
    top_k_value = request.get("topK", 5)
    top_k = top_k_value if isinstance(top_k_value, int) and top_k_value > 0 else 5
    ingest_started = perf_counter()
    memory = VectorMemory.from_defaults(
        embed_model=embedding, retriever_kwargs={"similarity_top_k": top_k}
    )
    for session in request["sessions"]:
        memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=session_text(session),
                additional_kwargs={"session_id": str(session["id"])},
            )
        )
    ingest_ms = (perf_counter() - ingest_started) * 1000
    search_started = perf_counter()
    matches = memory.get(input=str(request["question"]))
    search_ms = (perf_counter() - search_started) * 1000
    session_ids: list[str] = []
    for message in matches:
        session_id = message.additional_kwargs.get("session_id")
        if session_id is not None and str(session_id) not in session_ids:
            session_ids.append(str(session_id))
    return {
        "questionId": str(request["questionId"]),
        "retrieved": [
            {"sessionId": session_id, "rank": index + 1}
            for index, session_id in enumerate(session_ids[:top_k])
        ],
        "memories": [],
        "unsupported": ["memories"],
        "usage": dict(ZERO_USAGE),
        "wallMs": {"ingest": ingest_ms, "search": search_ms},
    }


def main() -> None:
    embedding = FastEmbedEmbedding(
        model_name=MODEL_ID, providers=["CPUExecutionProvider"], doc_embed_type="passage"
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        question_id = str(request.get("questionId", ""))
        if request.get("protocolVersion") != PROTOCOL_VERSION:
            response: dict[str, Any] = {
                "questionId": question_id,
                "error": f"unsupported protocol version {request.get('protocolVersion')!r}",
            }
        else:
            try:
                response = answer(request, embedding)
            except Exception as error:
                response = {
                    "questionId": question_id,
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the LlamaIndex manifest**

The LangGraph manifest with these substitutions: `adapter.id` `llamaindex-vectormemory-fastembed`, `adapter.version` `llama-index-core-0.14.23_fastembed-0.8.0_bge-small-en-v1.5`, `disclosures.packages` `{"llama-index-core": "0.14.23", "llama-index-embeddings-fastembed": "0.6.0", "fastembed": "0.8.0"}`, `storage` `"LlamaIndex VectorMemory over SimpleVectorStore; a fresh VectorMemory per benchmark question inside one long-lived process"`, `retrievalPolicy` `"VectorMemory.get with the question as input and similarity_top_k set to the request's topK; returned chat messages are deduplicated back to session ids in score order"`. Everything else is identical.

- [ ] **Step 5: Smoke-test each bridge by hand before spending an hour on a run**

```bash
printf '%s\n' '{"protocolVersion":"rembero.memory-systems.v1","questionId":"smoke","questionType":"single-session-user","questionDate":"2023-07-10","question":"what degree did I graduate with","topK":2,"lanes":["retrieval"],"sessions":[{"id":"noise","date":"2023-05-15","turns":[{"role":"user","content":"Compare credit card rewards."}]},{"id":"evidence","date":"2023-06-01","turns":[{"role":"user","content":"I graduate with Business Administration."}]}]}' \
  | (cd benchmarks/adapters/langgraph-fastembed && uv run --script memory_systems_bridge.py)
```

Expected: one JSON line whose first `retrieved` entry is `evidence`, `unsupported` is `["memories"]`, and `wallMs.ingest` is a positive number. Repeat with `llamaindex-fastembed`. The first run resolves dependencies and downloads the model; allow several minutes.

- [ ] **Step 6: Run both on the subset**

```bash
set -a; source .env; set +a
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=$DEEPSEEK_API_KEY
COMMON="--split all --cases $(cat .cache/longmemeval/subset-100.txt) --local-only \
  --top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --context-bytes 24576 --date-distances \
  --reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 \
  --reader-api-key ollama --reader-max-tokens 440 --judge-model deepseek-chat --concurrency 4"
node dist/evals/run-longmemeval-answer.js $COMMON \
  --memory-system benchmarks/adapters/langgraph-fastembed/memory-systems.json --memory-lane retrieval \
  --output docs/research/results/memory-systems-langgraph-retrieval-subset100.json
node dist/evals/run-longmemeval-answer.js $COMMON \
  --memory-system benchmarks/adapters/llamaindex-fastembed/memory-systems.json --memory-lane retrieval \
  --output docs/research/results/memory-systems-llamaindex-retrieval-subset100.json
```

Expected: 100 judged each, `summary.errors` 0, `memory system calls/tokens/cost: 0 / 0 / $0.000000`, `memory system ingest/search p50` in the tens of seconds for ingest and single-digit milliseconds for search. Compare recall@k with `builtin:embed` from Task 2: both are dense retrievers over whole sessions, so a large gap means a chunking or a query-prefix difference worth one sentence in the document.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/adapters/langgraph-fastembed/memory_systems_bridge.py \
  benchmarks/adapters/langgraph-fastembed/memory-systems.json \
  benchmarks/adapters/llamaindex-fastembed/memory_systems_bridge.py \
  benchmarks/adapters/llamaindex-fastembed/memory-systems.json \
  docs/research/results/memory-systems-{langgraph,llamaindex}-retrieval-subset100.json
git commit -m "LangGraph and LlamaIndex on the long-lived memory-systems protocol, measured on the subset"
```

---

### Task 5: The Mem0 and Graphiti bridges with DeepSeek

**Cost, and the first real spend.** Off-peak DeepSeek at $0.22/M input and $0.66/M output. Mem0 is about 120k input tokens a question, so the 100-question subset is roughly 12M input and 1M output tokens ≈ **$3–5**. Graphiti makes two to three times the calls ≈ **$7–15** on the subset. Judge ≈ $0.04. **Budget $10–20 for this task**, and read the true figure out of each run's `memory system calls/tokens/cost` line rather than estimating it again. Do not start inside the peak windows (01:00–04:00 and 06:00–10:00 UTC), where the rate doubles. Ingest wall time is one to two and a half hours per subset run at `--concurrency 4`.

**Files:**
- Create: `benchmarks/adapters/mem0-deepseek-fastembed/memory_systems_bridge.py`, `benchmarks/adapters/mem0-deepseek-fastembed/memory-systems.json`
- Create: `benchmarks/adapters/graphiti-deepseek-fastembed/memory_systems_bridge.py`, `benchmarks/adapters/graphiti-deepseek-fastembed/memory-systems.json`

**Interfaces:**
- Consumes: the protocol (Task 1), `--memory-system <manifest>` and the `DEEPSEEK_API_KEY` entry in `ALLOWED_EXTERNAL_ENVIRONMENT` (Task 2).
- Produces: two adapters that answer **both** lanes — `retrieved` from their native search and `memories` from their own memory text (Mem0's `memory` strings, Graphiti's edge `fact` strings), each memory citing the sessions it came from — and that report real `usage` with a cost computed from the DeepSeek rate constants in the bridge.

- [ ] **Step 1: Write the Mem0 bridge**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "fastembed==0.8.0",
#   "mem0ai==2.0.14",
# ]
# ///
"""rembero.memory-systems.v1 bridge for Mem0 OSS with DeepSeek at ingest.

One process for the whole run; a fresh Qdrant collection, history database and user id per
question, deleted when the question is done. Mem0 runs its own extraction and update
decision over every session, then its own search. Both lanes are answered: retrieval maps
memories back to the sessions they were written from; memories returns Mem0's own text."""

import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ["MEM0_TELEMETRY"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Mem0's openai provider reads OPENAI_API_KEY; DeepSeek is OpenAI-compatible.
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

from mem0 import Memory

PROTOCOL_VERSION = "rembero.memory-systems.v1"
LLM_MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
# DeepSeek publishes no per-response cost, so the bridge prices its own tokens.
INPUT_USD_PER_MTOK = 0.22
OUTPUT_USD_PER_MTOK = 0.66


class Usage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def callback(self, _client: Any, response: Any, _params: dict[str, Any]) -> None:
        usage = (response.model_dump() or {}).get("usage") or {}
        with self._lock:
            self.model_calls += 1
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)

    def take(self) -> dict[str, Any]:
        with self._lock:
            payload = {
                "modelCalls": self.model_calls,
                "inputTokens": self.input_tokens,
                "outputTokens": self.output_tokens,
                "costUsd": self.input_tokens / 1e6 * INPUT_USD_PER_MTOK
                + self.output_tokens / 1e6 * OUTPUT_USD_PER_MTOK,
            }
            self.model_calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            return payload


def safe_id(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-") or "question"


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def create_memory(root: Path, identifier: str, usage: Usage) -> Memory:
    return Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"rembero-{identifier}",
                    "path": str(root / f"qdrant-{identifier}"),
                    "embedding_model_dims": EMBEDDING_DIMENSIONS,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "max_tokens": 512,
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                    "openai_base_url": DEEPSEEK_URL,
                    "response_callback": usage.callback,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": EMBEDDING_MODEL,
                    "embedding_dims": EMBEDDING_DIMENSIONS,
                },
            },
            "history_db_path": str(root / f"history-{identifier}.db"),
        }
    )


def close_memory(memory: Memory) -> None:
    close = getattr(getattr(getattr(memory, "vector_store", None), "client", None), "close", None)
    if callable(close):
        close()


def answer(root: Path, request: dict[str, Any], usage: Usage) -> dict[str, Any]:
    identifier = safe_id(request["questionId"])
    user_id = f"rembero-{identifier}"
    memory = create_memory(root, identifier, usage)
    try:
        ingest_started = perf_counter()
        for session in request["sessions"]:
            memory.add(
                session_text(session),
                user_id=user_id,
                metadata={"session_id": str(session["id"]), "at": str(session["date"])},
            )
        ingest_ms = (perf_counter() - ingest_started) * 1000
        search_started = perf_counter()
        top_k_value = request.get("topK", 5)
        top_k = top_k_value if isinstance(top_k_value, int) and top_k_value > 0 else 5
        response = memory.search(str(request["question"]), filters={"user_id": user_id}, limit=top_k)
        search_ms = (perf_counter() - search_started) * 1000
        results = response.get("results", []) if isinstance(response, dict) else []
        session_ids: list[str] = []
        memories: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") if isinstance(result, dict) else None
            session_id = str(metadata["session_id"]) if isinstance(metadata, dict) and "session_id" in metadata else None
            text = str(result.get("memory") or "") if isinstance(result, dict) else ""
            if text:
                memories.append(
                    {
                        "text": text[:4000],
                        **({"sessionIds": [session_id]} if session_id else {}),
                        **(
                            {"at": str(metadata["at"])}
                            if isinstance(metadata, dict) and "at" in metadata
                            else {}
                        ),
                    }
                )
            if session_id is not None and session_id not in session_ids:
                session_ids.append(session_id)
        return {
            "questionId": str(request["questionId"]),
            "retrieved": [
                {"sessionId": session_id, "rank": index + 1}
                for index, session_id in enumerate(session_ids[:top_k])
            ],
            "memories": memories,
            "unsupported": [],
            "usage": usage.take(),
            "wallMs": {"ingest": ingest_ms, "search": search_ms},
        }
    finally:
        close_memory(memory)


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required")
    usage = Usage()
    with tempfile.TemporaryDirectory(prefix="rembero-mem0-") as temp:
        root = Path(temp)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            question_id = str(request.get("questionId", ""))
            if request.get("protocolVersion") != PROTOCOL_VERSION:
                response: dict[str, Any] = {
                    "questionId": question_id,
                    "error": f"unsupported protocol version {request.get('protocolVersion')!r}",
                }
            else:
                try:
                    response = answer(root, request, usage)
                except Exception as error:
                    response = {
                        "questionId": question_id,
                        "error": f"{type(error).__name__}: {error}"[:300],
                    }
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the Mem0 manifest**

```json
{
  "schemaVersion": "rembero.memory-stack-adapter.v1",
  "protocol": "rembero.memory-systems.v1",
  "adapter": {
    "id": "mem0-oss-deepseek-fastembed",
    "version": "mem0ai-2.0.14_deepseek-chat_fastembed-0.8.0_bge-small-en-v1.5",
    "capabilities": {
      "answerRows": false,
      "rankedRetrieval": true,
      "citations": false,
      "rules": false,
      "temporalUpdates": true,
      "trustViews": false
    },
    "disclosures": {
      "packages": { "mem0ai": "2.0.14", "fastembed": "0.8.0" },
      "llmModel": "deepseek-chat (DeepSeek V4 Flash) at https://api.deepseek.com/v1, temperature 0, max_tokens 512",
      "embeddingModel": "BAAI/bge-small-en-v1.5 (Mem0 FastEmbed provider, local CPU, 384 dimensions)",
      "storage": "Mem0 OSS with a fresh local Qdrant collection and SQLite history database per benchmark question, inside one long-lived process",
      "writePolicy": "Mem0's native add per haystack session with the session id and date in metadata; Mem0 runs its own extraction and update decision",
      "retrievalPolicy": "Mem0 search with the question, the question's user filter and the request's topK; returned memories are deduplicated back to their source sessions in score order",
      "providerCostBoundary": "DeepSeek chat completions are provider-charged; tokens come from the native response usage and cost is computed in the bridge at $0.22/M input and $0.66/M output (off-peak). Embeddings, vector storage and search are local",
      "notes": [
        "Both lanes are answered: the memories lane returns Mem0's own memory text, which is what a Mem0 user puts in front of a model.",
        "Mem0 telemetry is disabled and spaCy is intentionally not installed, so Mem0 uses its built-in fallback.",
        "Peak DeepSeek pricing (01:00-04:00 and 06:00-10:00 UTC) is double; runs are made outside those windows and the document says so."
      ]
    }
  },
  "command": {
    "executable": "uv",
    "args": ["run", "--script", "memory_systems_bridge.py"],
    "workingDirectory": ".",
    "requiredEnvironment": ["DEEPSEEK_API_KEY"],
    "timeoutMs": 900000,
    "maxOutputBytes": 4000000
  }
}
```

- [ ] **Step 3: Write the Graphiti bridge**

Take `benchmarks/adapters/graphiti-openrouter-fastembed/bridge.py` as the starting point and change these things and nothing else:

- header docstring and `PROTOCOL_VERSION = "rembero.memory-systems.v1"`;
- `LLM_MODEL = "deepseek-chat"`, `DEEPSEEK_URL = "https://api.deepseek.com/v1"`, and `AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_URL, http_client=http_client)` plus the same base URL and key in `LLMConfig`;
- `Usage.response_hook` keeps counting tokens but computes cost the way the Mem0 bridge does (`INPUT_USD_PER_MTOK = 0.22`, `OUTPUT_USD_PER_MTOK = 0.66`), and gains a `take()` that returns the totals and resets them per question;
- `run_question` becomes `answer(root, request, usage, llm_client, embeddings)`: the identifier is `safe_id(request["questionId"])`, the episodes are the haystack sessions —

```python
        formed = await graphiti.add_episode_bulk(
            [
                RawEpisode(
                    name=str(session["id"]),
                    content="\n".join(
                        f"{turn['role']}: {turn['content']}" for turn in session["turns"]
                    ),
                    source_description="Rembero memory-systems benchmark session",
                    source=EpisodeType.text,
                    reference_time=datetime.fromisoformat(f"{session['date']}T09:00:00+00:00"),
                )
                for session in request["sessions"]
            ],
            group_id=database,
        )
        episode_ids = {
            str(episode.uuid): str(session["id"])
            for episode, session in zip(formed.episodes, request["sessions"], strict=True)
        }
        edges = await graphiti.search(
            str(request["question"]), group_ids=[database], num_results=max(top_k * 3, top_k)
        )
```

— and the return is the protocol response, with the memories lane filled from the edges' own facts:

```python
        session_ids = event_ids_from_edges(edges, episode_ids, top_k)
        memories = [
            {
                "text": str(getattr(edge, "fact", ""))[:4000],
                "sessionIds": [
                    episode_ids[str(uuid)]
                    for uuid in getattr(edge, "episodes", [])
                    if str(uuid) in episode_ids
                ],
            }
            for edge in edges
            if str(getattr(edge, "fact", ""))
        ]
        return {
            "questionId": str(request["questionId"]),
            "retrieved": [
                {"sessionId": session_id, "rank": index + 1}
                for index, session_id in enumerate(session_ids)
            ],
            "memories": memories[:200],
            "unsupported": [],
            "usage": usage.take(),
            "wallMs": {"ingest": ingest_ms, "search": search_ms},
        }
```

- `async_main` becomes the line loop: read stdin line by line inside `async for`-free plain iteration, `await answer(...)` per line, write one JSON line, `sys.stdout.flush()`, and catch every exception into `{"questionId": …, "error": …}` exactly as the Mem0 bridge does. The `httpx.AsyncClient`, the `OpenAIGenericClient`, the `FastEmbedClient` and the `TemporaryDirectory` are created once, outside the loop.
- `custom_extraction_instructions` is dropped: this benchmark measures each system as shipped, and a hand-written extraction instruction is a thumb on the scale. Say so in the manifest notes.

- [ ] **Step 4: Write the Graphiti manifest**

The Mem0 manifest with: `adapter.id` `graphiti-oss-deepseek-fastembed`, `adapter.version` `graphiti-core-0.29.3_deepseek-chat_fastembed-0.8.0_bge-small-en-v1.5_falkordblite`, `disclosures.packages` `{"graphiti-core": "0.29.3", "fastembed": "0.8.0", "falkordblite": "0.10.0", "httpx": "0.28.1", "openai": "3.3.1", "redis": "8.1.0"}`, `storage` `"Graphiti OSS with a fresh embedded FalkorDBLite graph and group id per benchmark question, inside one long-lived process"`, `writePolicy` `"Graphiti add_episode_bulk with one episode per haystack session and no custom extraction instructions; episode identity comes from the returned native episode objects"`, `retrievalPolicy` `"Graphiti hybrid edge search with the question and the question's group id at three times the request's topK, then the ranked edges' native episode uuids mapped back to session ids"`, and a note that the no-op cross-encoder is supplied because the basic RRF path does not rerank, so no unmeasured provider client exists.

- [ ] **Step 5: Smoke-test both bridges on one question**

```bash
set -a; source .env; set +a
printf '%s\n' '{"protocolVersion":"rembero.memory-systems.v1","questionId":"smoke","questionType":"single-session-user","questionDate":"2023-07-10","question":"what degree did I graduate with","topK":2,"lanes":["retrieval","memories"],"sessions":[{"id":"noise","date":"2023-05-15","turns":[{"role":"user","content":"Compare credit card rewards."}]},{"id":"evidence","date":"2023-06-01","turns":[{"role":"user","content":"I graduate with Business Administration."}]}]}' \
  | (cd benchmarks/adapters/mem0-deepseek-fastembed && uv run --script memory_systems_bridge.py)
```

Expected: one line with `evidence` ranked first, a non-empty `memories` array, and `usage.modelCalls` above zero with a `costUsd` in the small fractions of a cent. Repeat for `graphiti-deepseek-fastembed`. A DeepSeek authentication failure shows as `error` in the line, not as a crash. **Do not go on to Step 6 until both smoke lines look right — Step 6 spends real money.**

- [ ] **Step 6: Run both on the subset, both lanes**

Check the clock first: it must be outside 01:00–04:00 and 06:00–10:00 UTC.

```bash
set -a; source .env; set +a
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=$DEEPSEEK_API_KEY
COMMON="--split all --cases $(cat .cache/longmemeval/subset-100.txt) --local-only \
  --top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --context-bytes 24576 --date-distances \
  --reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 \
  --reader-api-key ollama --reader-max-tokens 440 --judge-model deepseek-chat --concurrency 4"
for system in mem0 graphiti; do
  for lane in retrieval memories; do
    node dist/evals/run-longmemeval-answer.js $COMMON \
      --memory-system benchmarks/adapters/$system-deepseek-fastembed/memory-systems.json \
      --memory-lane $lane \
      --output docs/research/results/memory-systems-$system-$lane-subset100.json
  done
done
```

Expected: 100 judged per run; the `memory system calls/tokens/cost` line is the measured ingest bill. Read it and record it: it is the cost column of the leaderboard and the input to Task 6's decision about the 500. If Mem0's subset bill is above $8 or Graphiti's above $20, stop and say so before Task 6 — the 500 is five times that.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/adapters/mem0-deepseek-fastembed benchmarks/adapters/graphiti-deepseek-fastembed \
  docs/research/results/memory-systems-{mem0,graphiti}-{retrieval,memories}-subset100.json
git commit -m "Mem0 and Graphiti on the memory-systems protocol with DeepSeek at ingest; subset measured with the cost read from the observations"
```

---

### Task 6: The 500, the document, the script and the site section

**Cost.** Free systems (full context, BM25, embed, Remembero raw and as-shipped, LangGraph, LlamaIndex): judge only, ≈ $0.05 a run, ≈ $0.45 for nine runs. Mem0 on the 500 ≈ **$15–25**; Graphiti ≈ **$35–75**. **Total $50–100, and the paid half needs the user's go-ahead.** Step 2 stops and asks with the measured subset figures in hand. Wall time: six to twelve hours per system at `--concurrency 4`, so plan the free systems overnight and the two paid systems one at a time.

**Files:**
- Create: `scripts/bench-memory-systems.mjs`, `scripts/memory-systems-table.mjs`
- Create: `docs/research/MEMORY-SYSTEMS-BENCHMARK.md`
- Create: `docs/research/results/memory-systems-*-all500.json` (and the rejudge sidecars)
- Modify: `package.json` (one script), `site/app/page.tsx:174-205`, `site/app/globals.css` (after the `.examples-grid` rules near `:743`)

**Interfaces:**
- Consumes: every result file from Tasks 2–5, `src/evals/rejudge-longmemeval.ts` (already built), `--memory-system` / `--memory-lane`.
- Produces: `npm run bench:memory-systems` runs every free system's subset row; `node scripts/memory-systems-table.mjs` prints the leaderboard rows as Markdown so no number is transcribed by hand.

- [ ] **Step 1: Write the free-systems script and wire it into package.json**

```js
// scripts/bench-memory-systems.mjs
// Every memory system that needs no paid model, on the 100-question subset, one after the
// other. Mem0 and Graphiti are deliberately absent: they need DEEPSEEK_API_KEY and money,
// and their commands live in docs/research/MEMORY-SYSTEMS-BENCHMARK.md.
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const cases = readFileSync('.cache/longmemeval/subset-100.txt', 'utf8').trim();
const common = [
  '--split', 'all',
  '--cases', cases,
  '--local-only',
  '--top-k', '4',
  '--multi-session-top-k', '15',
  '--temporal-top-k', '10',
  '--context-bytes', '24576',
  '--date-distances',
  '--reader-model', 'glm-5.3-flash:cloud',
  '--reader-base-url', 'http://127.0.0.1:11434/v1',
  '--reader-api-key', 'ollama',
  '--reader-max-tokens', '440',
  '--judge-model', 'deepseek-chat',
  '--concurrency', '4',
];

const rows = [
  ['full-context', 'retrieval', ['--memory-system', 'builtin:full-context']],
  ['bm25', 'retrieval', ['--memory-system', 'builtin:bm25']],
  ['embed', 'retrieval', ['--memory-system', 'builtin:embed']],
  ['remembero', 'retrieval', ['--memory-system', 'builtin:remembero-raw']],
  ['langgraph', 'retrieval', ['--memory-system', 'benchmarks/adapters/langgraph-fastembed/memory-systems.json']],
  ['llamaindex', 'retrieval', ['--memory-system', 'benchmarks/adapters/llamaindex-fastembed/memory-systems.json']],
];

let failed = 0;
for (const [system, lane, extra] of rows) {
  const output = `docs/research/results/memory-systems-${system}-${lane}-subset100.json`;
  console.log(`\n=== ${system} (${lane}) -> ${output}`);
  const result = spawnSync(
    process.execPath,
    ['dist/evals/run-longmemeval-answer.js', ...common, ...extra, '--memory-lane', lane, '--output', output],
    { stdio: 'inherit' },
  );
  if (result.status !== 0) {
    failed += 1;
    console.error(`${system} exited with ${result.status}`);
  }
}
process.exit(failed === 0 ? 0 : 1);
```

In `package.json`, next to `"bench:longmemeval:answer"`:

```json
    "bench:memory-systems": "npm run build:core && node scripts/bench-memory-systems.mjs",
```

- [ ] **Step 2: Decide the paid 500 with the user**

Print the two subset bills:

```bash
for f in docs/research/results/memory-systems-{mem0,graphiti}-retrieval-subset100.json; do
  python3 -c "
import json,sys
run=json.load(open('$f'))
usage=[o['memorySystem']['usage'] for o in run['observations'] if o.get('memorySystem')]
print('$f', run['summary']['correct'], '/', run['summary']['questions'],
      'calls', sum(u['modelCalls'] for u in usage),
      'in', sum(u['inputTokens'] for u in usage),
      'out', sum(u['outputTokens'] for u in usage),
      'cost \$%.2f' % sum(u['costUsd'] for u in usage))
"
done
```

Multiply each cost by five: that is the 500-question ingest bill for that system. **Stop here and ask the user to approve the Mem0 and Graphiti 500 runs with those two numbers in the question.** The free systems' 500 runs need no approval; start them while waiting.

- [ ] **Step 3: Run the 500 for every system**

Same `COMMON` as Task 5 Step 6 with `--cases` removed (so `--split all` selects all 500) and `--output …-all500.json`:

```bash
set -a; source .env; set +a
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=$DEEPSEEK_API_KEY
COMMON="--split all --local-only --top-k 4 --multi-session-top-k 15 --temporal-top-k 10 \
  --context-bytes 24576 --date-distances --reader-model glm-5.3-flash:cloud \
  --reader-base-url http://127.0.0.1:11434/v1 --reader-api-key ollama --reader-max-tokens 440 \
  --judge-model deepseek-chat --concurrency 4"
WRITER="--extraction-model rembero-writer --extraction-base-url http://127.0.0.1:8081/v1 \
  --extraction-cache .cache/longmemeval/extraction-r23 --extraction-max-tokens 512"

node dist/evals/run-longmemeval-answer.js $COMMON --memory-system builtin:full-context --memory-lane retrieval --output docs/research/results/memory-systems-full-context-retrieval-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON --memory-system builtin:bm25 --memory-lane retrieval --output docs/research/results/memory-systems-bm25-retrieval-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON --memory-system builtin:embed --memory-lane retrieval --output docs/research/results/memory-systems-embed-retrieval-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON --memory-system builtin:remembero-raw --memory-lane retrieval --output docs/research/results/memory-systems-remembero-retrieval-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON $WRITER --memory-system builtin:remembero-hybrid --memory-lane memories --computed-notes --output docs/research/results/memory-systems-remembero-memories-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON --memory-system benchmarks/adapters/langgraph-fastembed/memory-systems.json --memory-lane retrieval --output docs/research/results/memory-systems-langgraph-retrieval-all500.json
node dist/evals/run-longmemeval-answer.js $COMMON --memory-system benchmarks/adapters/llamaindex-fastembed/memory-systems.json --memory-lane retrieval --output docs/research/results/memory-systems-llamaindex-retrieval-all500.json
# only after Step 2's approval, and outside 01:00-04:00 and 06:00-10:00 UTC
for system in mem0 graphiti; do
  for lane in retrieval memories; do
    node dist/evals/run-longmemeval-answer.js $COMMON \
      --memory-system benchmarks/adapters/$system-deepseek-fastembed/memory-systems.json \
      --memory-lane $lane --output docs/research/results/memory-systems-$system-$lane-all500.json
  done
done
node dist/evals/rejudge-longmemeval.js --judge-model deepseek-chat --files '^memory-systems-.*-all500\.json$'
```

Expected: 500 judged per run, `summary.errors` 0, and one `.rejudged-deepseek-chat.json` sidecar per run. A run that dies mid-way is resumed with `--offset`; the observations of the two slices are concatenated by hand only if the run genuinely cannot be redone, and the document says so if it happens.

- [ ] **Step 4: Write the table generator**

```js
// scripts/memory-systems-table.mjs
// Prints the leaderboard rows of docs/research/MEMORY-SYSTEMS-BENCHMARK.md as Markdown, so
// no number is transcribed by hand.
//   node scripts/memory-systems-table.mjs [subset100|all500]
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const suffix = process.argv[2] ?? 'all500';
const directory = 'docs/research/results';
const types = ['single-session-user', 'single-session-assistant', 'single-session-preference',
  'multi-session', 'temporal-reasoning', 'knowledge-update'];

const rows = readdirSync(directory)
  .filter((name) => name.startsWith('memory-systems-') && name.endsWith(`-${suffix}.json`) && !name.includes('.rejudged-'))
  .sort()
  .map((name) => {
    const run = JSON.parse(readFileSync(join(directory, name), 'utf8'));
    const usage = run.observations.flatMap((o) => (o.memorySystem ? [o.memorySystem] : []));
    const total = (field) => usage.reduce((sum, u) => sum + (u.usage?.[field] ?? 0), 0);
    const ingestSeconds = usage.reduce((sum, u) => sum + (u.wallMs?.ingest ?? 0), 0) / 1000;
    return {
      system: run.settings?.memorySystem ?? 'native',
      lane: run.settings?.memoryLane ?? 'native',
      correct: run.summary.correct,
      questions: run.summary.questions,
      recall: run.summary.retrievalRecallAtK,
      types: types.map((type) => run.byQuestionType[type]?.correct ?? 0),
      calls: total('modelCalls'),
      costUsd: total('costUsd'),
      ingestSeconds,
      searchMs: usage.reduce((sum, u) => sum + (u.wallMs?.search ?? 0), 0) / Math.max(1, usage.length),
    };
  });

console.log(`| memory layer | lane | correct | ${types.join(' | ')} | recall@k | ingest calls | ingest cost | ingest h | search p̄ ms |`);
console.log(`| --- |${' --- |'.repeat(types.length + 7)}`);
for (const row of rows) {
  console.log(
    `| ${row.system} | ${row.lane} | ${row.correct}/${row.questions} | ${row.types.join(' | ')} | ` +
      `${(row.recall * 100).toFixed(1)}% | ${row.calls} | $${row.costUsd.toFixed(2)} | ` +
      `${(row.ingestSeconds / 3600).toFixed(1)} | ${row.searchMs.toFixed(1)} |`,
  );
}
```

Run it and keep the output: `node scripts/memory-systems-table.mjs all500 > /tmp/leaderboard.md`.

- [ ] **Step 5: Write the document**

Create `docs/research/MEMORY-SYSTEMS-BENCHMARK.md` with, in this order and nothing else:

1. Title and a two-sentence why, naming the published numbers the spec names (Mem0 94.4%, Synap 92.0% with gpt-5-mini, Supermemory 85.2% with Gemini-3 Pro, Zep 71.2% with gpt-4o) and the point that none of them is a statement about the memory layer.
2. **What is held constant** — the spec's table copied verbatim, with the reader row updated to the fact fixed since: GLM 5.3 Flash through Ollama Cloud (`glm-5.3-flash:cloud`, `http://127.0.0.1:11434/v1`), thinking on, `--reader-max-tokens 440`.
3. **The two lanes**, the spec's two paragraphs, plus one sentence each on the two harness decisions this plan made: the memories lane groups memory text into one pseudo-session per cited session so the same context builder, byte budget and date distances apply; and `builtin:full-context` is the only row that returns more sessions than the requested top-k, by design, which is why its recall@k is 1.0 and its per-session text is about 500 bytes.
4. **Leaderboard** — the rows printed by `scripts/memory-systems-table.mjs all500`, then the same table for `subset100`.
5. **What each system was and was not given** — one row per system from its manifest's `disclosures` (`llmModel`, `embeddingModel`, `storage`, `writePolicy`, `retrievalPolicy`, `providerCostBoundary`), and, for the three baselines and Remembero's two rows, the same six facts written out.
6. **Cost and time** — the measured `ingest cost`, `ingest calls` and `ingest h` columns, the DeepSeek rate and the peak windows, and the note that the reader is on a subscription and the judge costs cents.
7. **How to reproduce** — `npm run bench:memory-systems` for the free systems, and the two literal Mem0 and Graphiti commands from Task 5 Step 6 with `DEEPSEEK_API_KEY` named as the requirement and the peak windows named as the thing to avoid. State that these two stay out of CI.
8. **What this cannot establish** — the spec's closing paragraph verbatim, plus the sentence that any two rows inside ±7 on the 500 are not ranked.

- [ ] **Step 6: Add the site section**

In `site/app/page.tsx`, add the link constant next to the others at the top:

```tsx
const memorySystemsDoc = `${github}/blob/main/docs/research/MEMORY-SYSTEMS-BENCHMARK.md`;
```

and, inside `<section className="examples section" id="examples">`, immediately after the closing `</div>` of `examples-grid` (line 203):

```tsx
          <div className="systems-board">
            <h3>One harness, one reader, one judge.</h3>
            <p>
              Eight memory layers over the same 500 LongMemEval questions, the same reader
              (GLM 5.3 Flash), the same judge (DeepSeek), the same context builder and the
              same retrieval depth. Only the memory layer changes. Remembero is one row,
              measured exactly as every other system.
            </p>
            <div className="systems-scroll">
              <table>
                <thead>
                  <tr><th>Memory layer</th><th>Correct</th><th>Recall@k</th><th>Ingest cost</th></tr>
                </thead>
                <tbody>
                  {/* one <tr> per row of `node scripts/memory-systems-table.mjs all500`,
                      taking the memory layer, correct/500, recall@k and ingest cost columns */}
                </tbody>
              </table>
            </div>
            <a href={memorySystemsDoc}>Every run, every disclosure, and what this cannot establish</a>
          </div>
```

Fill the `<tbody>` from the generator's output before committing; the comment must not survive. Then append to `site/app/globals.css`, after the `.examples-grid` rules:

```css
.systems-board { margin-top: 56px; padding-top: 32px; border-top: 1px solid var(--line); display: grid; gap: 14px; }
.systems-board h3 { margin: 0; font-size: 1.5rem; }
.systems-board p { margin: 0; max-width: 680px; color: var(--muted); }
.systems-scroll { overflow-x: auto; }
.systems-board table { width: 100%; min-width: 460px; border-collapse: collapse; font-size: 0.95rem; }
.systems-board th, .systems-board td { padding: 10px 16px 10px 0; border-bottom: 1px solid var(--line); text-align: left; }
.systems-board th { color: var(--muted); font-weight: 500; }
.systems-board td:first-child { font-weight: 600; }
.systems-board > a { justify-self: start; color: var(--amber-dark); text-decoration: underline; }
@media (max-width: 620px) { .systems-board th, .systems-board td { padding-right: 12px; } }
```

- [ ] **Step 7: Verify the site and the scripts**

Run: `cd site && npm run typecheck && npm run build:pages && node --test tests/rendered-html.test.mjs`
Expected: typecheck clean, build succeeds, the rendered-HTML test passes.

Run: `node scripts/memory-systems-table.mjs subset100`
Expected: a Markdown table whose row count equals the number of `-subset100.json` result files, with no `NaN` and no `$0.00` on the Mem0 and Graphiti rows.

- [ ] **Step 8: Commit**

```bash
git add scripts/bench-memory-systems.mjs scripts/memory-systems-table.mjs package.json \
  docs/research/MEMORY-SYSTEMS-BENCHMARK.md docs/research/results/memory-systems-*-all500*.json \
  site/app/page.tsx site/app/globals.css
git commit -m "Memory systems benchmark: the 500 for every system, the leaderboard, npm run bench:memory-systems and the site section"
```

---

## Self-review

**Spec coverage.** Constants table → Global Constraints and Task 6 Step 5 item 2. Retrieval lane → Task 2 Step 4's `lane === 'retrieval'` branch, recall@k already reported by `scoreLongMemEvalRetrievedSessions`. As-shipped lane → Task 2's `memories` branch, Task 3's `builtin:remembero-hybrid`, Task 5's Mem0 and Graphiti memories rows; pure retrievers report `unsupported: ["memories"]` and Task 2's third test proves the harness fails such a question rather than giving it a proxy. Systems table → Tasks 2–5, one row each. Protocol (long-lived process, no shell, stderr counted and discarded, byte limit, per-request timeout, fresh store per question, ids only from the request, memory text truncated with the drop recorded, the adapter never sees the gold answer) → Task 1's module and its five tests, Task 2's `memoryBytes`, each bridge's per-question namespace. Manifest schema plus a `protocol` field → Task 2 Step 6. Harness changes (`--memory-system`, `--memory-lane`, the three builtins with no process, unchanged result schema, adapter usage and wall time in the observation, the DeepSeek sidecar and the summary tables still working) → Task 2 Steps 3, 4, 5, 7 and Task 6 Step 3's rejudge line. Outputs (result files and sidecars, the document, the site section, the npm script with Mem0 and Graphiti out of CI) → Task 6. Cost and time → the cost line of every task and Task 6 Steps 2 and 5. Order of work → Tasks 1+2, 3, 4, 5, 6.

**Placeholder scan.** The only deferred values are the leaderboard numbers in Task 6 Steps 5 and 6 and the approval figures in Step 2; each is produced by the step immediately before it and the plan says which command prints it. Task 3 Step 1 flags one genuine unknown — the exact JSON `rememberTranscriptText` accepts from a scripted writer — and names the file to read for it rather than leaving it blank. No "TBD", no "add error handling", no "similar to Task N".

**Type consistency.** `MemorySystemClient` has `id`, `request`, `close` in Tasks 1, 2, 3 and the runner. `MemorySystemObservation` carries `id`, `lane`, `usage`, `wallMs`, `returnedSessions`, `returnedMemories`, `memoryBytes`, `unsupported` in the protocol module, the seam, the summary helper and both Task 1 and Task 2 tests. `memorySystemRequestFor(instance, lane, topK)` is called the same way in Task 1's tests, the seam and Task 3's tests. `createBuiltinMemorySystem(id, options)` and `openMemorySystem(spec, options)` share `MemorySystemFactoryOptions` across Tasks 2 and 3, and Task 3 only extends `BUILTIN_MEMORY_SYSTEMS` and the dispatch inside `createBuiltinMemorySystem` — it does not rename either. `sessionText(session)` is the one session renderer, exported from the builtin module and mirrored by the identical Python `session_text` in all four bridges. `run.settings.memorySystem` / `run.settings.memoryLane` are written in Task 2 Step 7 and read in Task 6's generator under exactly those names. `unsupported` is always an array of lanes, never a boolean.

**Fixes made during review.** The runner opens one adapter client per `--concurrency` worker rather than one per run, because a single long-lived process would serialize all 500 ingests; this is what makes the spec's "concurrency four to eight" reachable, and the observation still records per-question wall time. The `--memory-system` refusals were added after noticing that `--engine-recall` and `--temporal-range-model` read the store the memory-system path never fills.
