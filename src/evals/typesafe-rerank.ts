/**
 * Experimental: a TypeSafe ("System One") relevance judgment per candidate session, used to
 * re-rank the lexical session list before the harness cuts it to top-k.
 * https://docs.typesafe.ai — POST a state plus typed questions, get calibrated probabilities.
 * It is good at relevance, bad at counting, dates and arithmetic, and loses accuracy on a
 * large state full of irrelevant text; hence one short state per session.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { STOPWORDS } from '../knowledge/computed-notes.js';
import { assertSafeForExternalLlm } from '../safety.js';

export const TYPESAFE_URL = 'https://api.typesafe.ai/v1/systemone';
export const DEFAULT_TYPESAFE_MODEL = 'jev-1.13.0';
export const TYPESAFE_USD_PER_MILLION_INPUT_TOKENS = 0.042;
export const DEFAULT_RERANK_POOL = 30;
export const DEFAULT_RERANK_SESSION_CHARS = 8_000;
export const DEFAULT_RERANK_CONCURRENCY = 16;
export const DEFAULT_RERANK_KEY_ENV = 'TYPESAFE_AI_API_KEY';
export const DEFAULT_TYPESAFE_CACHE_DIR = '.cache/typesafe';
const MAX_ATTEMPTS = 5;
const MAX_RETRY_AFTER_MS = 60_000;

export interface TypesafeQuestion {
  type: 'noul';
  instructions: string;
}

export const RERANK_QUESTIONS = {
  relevant: {
    type: 'noul',
    instructions:
      'The session talks about the specific thing the question asks about.',
  },
  evidence: {
    type: 'noul',
    instructions:
      'The session contains a statement from the user that is needed to answer the question (an event, a date, an amount, a name, or a preference the answer depends on).',
  },
} as const satisfies Record<string, TypesafeQuestion>;

export interface TypesafeResult {
  nouls: Record<string, number>;
  /** The provider's input_tokens for this answer (also reported when it came from the cache). */
  inputTokens: number;
  cached: boolean;
}

export type TypesafeNouls = (
  state: unknown,
  questions: Record<string, TypesafeQuestion>,
) => Promise<TypesafeResult>;

export type Limiter = <T>(task: () => Promise<T>) => Promise<T>;

/** At most `concurrency` tasks in flight; the rest wait in arrival order. */
export function createLimiter(concurrency: number): Limiter {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1) {
    throw new Error('limiter concurrency must be a positive integer');
  }
  let active = 0;
  const waiting: Array<() => void> = [];
  return async <T>(task: () => Promise<T>): Promise<T> => {
    if (active >= concurrency) {
      await new Promise<void>((release) => waiting.push(release));
    } else {
      active += 1;
    }
    try {
      return await task();
    } finally {
      const next = waiting.shift();
      // hand the slot straight to the next task, so active never drops and climbs again
      if (next !== undefined) next();
      else active -= 1;
    }
  };
}

export interface TypesafeClientOptions {
  apiKey: string;
  model?: string;
  fetchFn?: typeof fetch;
  timeoutMs?: number;
  /** Cache directory (default .cache/typesafe under the working directory); null: no cache. */
  cacheDir?: string | null;
  limiter?: Limiter;
  sleep?: (ms: number) => Promise<void>;
  url?: string;
}

function retryAfterMs(header: string | null): number | undefined {
  if (header === null || header.trim() === '') return undefined;
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(MAX_RETRY_AFTER_MS, seconds * 1_000);
  }
  const at = Date.parse(header);
  if (Number.isNaN(at)) return undefined;
  return Math.min(MAX_RETRY_AFTER_MS, Math.max(0, at - Date.now()));
}

function parseAnswer(
  data: unknown,
  questions: Record<string, TypesafeQuestion>,
): { nouls: Record<string, number>; inputTokens: number } {
  const body = data as {
    answers?: Record<string, { noul?: unknown }>;
    usage?: { input_tokens?: unknown };
  };
  const nouls: Record<string, number> = {};
  for (const name of Object.keys(questions)) {
    const value = body.answers?.[name]?.noul;
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
      throw new Error(`TypeSafe response has no probability for "${name}"`);
    }
    nouls[name] = value;
  }
  const tokens = body.usage?.input_tokens;
  return {
    nouls,
    inputTokens:
      typeof tokens === 'number' && Number.isFinite(tokens) && tokens >= 0 ? tokens : 0,
  };
}

/** One TypeSafe System One call returning the noul probabilities, cached on disk. */
export async function typesafeNouls(
  state: unknown,
  questions: Record<string, TypesafeQuestion>,
  options: TypesafeClientOptions,
): Promise<TypesafeResult> {
  const model = options.model ?? DEFAULT_TYPESAFE_MODEL;
  const cacheDir =
    options.cacheDir === null
      ? undefined
      : resolve(options.cacheDir ?? DEFAULT_TYPESAFE_CACHE_DIR);
  const cachePath =
    cacheDir === undefined
      ? undefined
      : join(
          cacheDir,
          `${createHash('sha256')
            .update(`${model}\n${JSON.stringify(state)}\n${JSON.stringify(questions)}`)
            .digest('hex')}.json`,
        );
  if (cachePath !== undefined && existsSync(cachePath)) {
    const hit = JSON.parse(readFileSync(cachePath, 'utf8')) as {
      nouls: Record<string, number>;
      inputTokens: number;
    };
    return { nouls: hit.nouls, inputTokens: hit.inputTokens, cached: true };
  }
  if (options.apiKey.trim() === '') throw new Error('TypeSafe API key is empty');
  const fetchFn = options.fetchFn ?? fetch;
  const sleep =
    options.sleep ?? ((ms: number) => new Promise<void>((done) => setTimeout(done, ms)));
  const limit = options.limiter ?? (<T>(task: () => Promise<T>) => task());
  const body = JSON.stringify({ state, model, questions });
  let lastError: Error | null = null;
  let waitMs = 0;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) await sleep(waitMs);
    waitMs = 500 * 2 ** (attempt - 1);
    let reply: { status: number; retryAfter: string | null; text: string };
    try {
      // the body is read inside the limiter too: a slot is free only once the reply is in
      reply = await limit(async () => {
        const response = await fetchFn(options.url ?? TYPESAFE_URL, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${options.apiKey}`,
            'Content-Type': 'application/json',
          },
          body,
          signal: AbortSignal.timeout(options.timeoutMs ?? 60_000),
        });
        return {
          status: response.status,
          retryAfter: response.headers.get('retry-after'),
          text: await response.text(),
        };
      });
    } catch (error) {
      // network failures and timeouts are transient; anything else is ours to see
      lastError = error instanceof Error ? error : new Error(String(error));
      if (/abort|timeout|fetch failed|network|ECONN/i.test(lastError.message)) continue;
      throw lastError;
    }
    if (reply.status < 200 || reply.status > 299) {
      const detail = reply.text.slice(0, 200).trim();
      lastError = new Error(
        `TypeSafe request failed with status ${reply.status}${detail === '' ? '' : `: ${detail}`}`,
      );
      if (reply.status === 429 || reply.status >= 500) {
        waitMs = retryAfterMs(reply.retryAfter) ?? waitMs;
        continue;
      }
      throw lastError;
    }
    let data: unknown;
    try {
      data = JSON.parse(reply.text);
    } catch {
      throw new Error('TypeSafe response is not JSON');
    }
    const answer = parseAnswer(data, questions);
    if (cachePath !== undefined && cacheDir !== undefined) {
      mkdirSync(cacheDir, { recursive: true });
      // write then rename, so a concurrent reader never sees half a file
      const partial = `${cachePath}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
      writeFileSync(partial, JSON.stringify({ model, ...answer }));
      renameSync(partial, cachePath);
    }
    return { ...answer, cached: false };
  }
  throw lastError ?? new Error('TypeSafe request failed');
}

function overlapWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, ' ')
    .split(/\s+/)
    .filter((word) => word.length >= 3 && !STOPWORDS.has(word));
}

/**
 * Whole turns within `maxChars`. Under budget: every turn. Over: turns by descending overlap
 * with the question's content words (ties: earlier first), each kept if it still fits, then
 * returned in their original order. If not even one whole turn fits, the best one is cut.
 */
export function fitTurnsToBudget(
  turns: readonly string[],
  question: string,
  maxChars: number,
): string[] {
  const total = turns.reduce((sum, turn) => sum + turn.length, 0);
  if (total <= maxChars) return [...turns];
  const wanted = new Set(overlapWords(question));
  const overlap = turns.map(
    (turn) => new Set(overlapWords(turn).filter((word) => wanted.has(word))).size,
  );
  const order = turns
    .map((_turn, index) => index)
    .sort((a, b) => overlap[b]! - overlap[a]! || a - b);
  const kept = new Set<number>();
  let used = 0;
  for (const index of order) {
    const length = turns[index]!.length;
    if (used + length > maxChars) continue;
    kept.add(index);
    used += length;
  }
  if (kept.size === 0) return [turns[order[0]!]!.slice(0, maxChars)];
  return turns.filter((_turn, index) => kept.has(index));
}

export interface RerankStats {
  candidates: number;
  /** Candidates answered by a live request. */
  calls: number;
  /** Candidates answered from the on-disk cache. */
  cached: number;
  /** Input tokens of the live requests (cached answers cost nothing). */
  inputTokens: number;
  /** Candidates whose state failed the external-LLM safety check: never sent. */
  blocked: number;
}

export interface RerankSession {
  date: string;
  turns: ReadonlyArray<{ role: string; content: string }>;
}

export interface RerankParams {
  question: string;
  questionDate: string;
  pool: number;
  sessionChars: number;
  /** 'user': user turns only, as plain text; 'all': every turn, prefixed with its role. */
  contextRoles: 'user' | 'all';
  sessionFor: (sessionId: string) => RerankSession | undefined;
  nouls: TypesafeNouls;
}

export const RERANK_EVIDENCE_WEIGHT = 0.7;
export const RERANK_RELEVANT_WEIGHT = 0.3;

export function rerankState(
  params: Pick<RerankParams, 'question' | 'questionDate' | 'sessionChars' | 'contextRoles'>,
  session: RerankSession,
): { question: string; question_date: string; session_date: string; session: string[] } {
  const turns = session.turns
    .filter(({ role, content }) =>
      content.trim() !== '' && (params.contextRoles === 'all' || role === 'user'),
    )
    .map(({ role, content }) =>
      params.contextRoles === 'all' ? `${role}: ${content}` : content,
    );
  return {
    question: params.question,
    question_date: params.questionDate,
    session_date: session.date,
    session: fitTurnsToBudget(turns, params.question, params.sessionChars),
  };
}

/**
 * Re-rank the first `pool` sessions of a lexical order by 0.7 x evidence + 0.3 x relevant
 * (ties keep the lexical order); sessions beyond the pool follow unchanged. A candidate whose
 * state fails the safety check is not sent and scores (n - i) / (n + 1) from its lexical
 * position i among the n candidates.
 */
export async function rerankSessionOrder(
  ordered: readonly string[],
  params: RerankParams,
): Promise<{ order: string[]; stats: RerankStats }> {
  const pool = ordered.slice(0, params.pool);
  const stats: RerankStats = {
    candidates: pool.length,
    calls: 0,
    cached: 0,
    inputTokens: 0,
    blocked: 0,
  };
  const scores = await Promise.all(
    pool.map(async (sessionId, index) => {
      const lexicalScore = (pool.length - index) / (pool.length + 1);
      const session = params.sessionFor(sessionId);
      const state = rerankState(params, session ?? { date: '', turns: [] });
      try {
        assertSafeForExternalLlm(JSON.stringify(state), 'TypeSafe rerank candidate');
      } catch {
        stats.blocked += 1;
        return lexicalScore;
      }
      const answer = await params.nouls(state, RERANK_QUESTIONS);
      if (answer.cached) stats.cached += 1;
      else {
        stats.calls += 1;
        stats.inputTokens += answer.inputTokens;
      }
      return (
        RERANK_EVIDENCE_WEIGHT * answer.nouls.evidence! +
        RERANK_RELEVANT_WEIGHT * answer.nouls.relevant!
      );
    }),
  );
  const reranked = pool
    .map((sessionId, index) => ({ sessionId, index, score: scores[index]! }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(({ sessionId }) => sessionId);
  return { order: [...reranked, ...ordered.slice(params.pool)], stats };
}

export function typesafeCostUsd(inputTokens: number): number {
  return (inputTokens * TYPESAFE_USD_PER_MILLION_INPUT_TOKENS) / 1_000_000;
}
