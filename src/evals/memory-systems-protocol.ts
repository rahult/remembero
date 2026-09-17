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
  /**
   * The adapter returned more sessions than the question's retrieval depth asked for. The
   * harness still scores it at that depth; this says the system did not respect topK.
   */
  overRequestedDepth?: boolean;
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
          throw new Error(`memory system cited session ${sessionId}, which is not in the request`);
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
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
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
    medianIngestMs: median(
      present.flatMap(({ wallMs }) => (wallMs === null ? [] : [wallMs.ingest])),
    ),
    medianSearchMs: median(
      present.flatMap(({ wallMs }) => (wallMs === null ? [] : [wallMs.search])),
    ),
    droppedMemoryBytes: present.reduce((total, { memoryBytes }) => total + memoryBytes.dropped, 0),
  };
}
