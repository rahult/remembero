import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  evaluateLongMemEvalAnswerInstance,
  summarizeLongMemEvalAnswers,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { parseArgs } from '../src/evals/run-longmemeval-answer.js';
import {
  RERANK_QUESTIONS,
  TYPESAFE_URL,
  createLimiter,
  fitTurnsToBudget,
  rerankSessionOrder,
  typesafeNouls,
  type TypesafeNouls,
} from '../src/evals/typesafe-rerank.js';

const KEY = 'ts-test-key-not-real';
const dirs: string[] = [];
const tempDir = (): string => {
  const dir = mkdtempSync(join(tmpdir(), 'typesafe-test-'));
  dirs.push(dir);
  return dir;
};
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

type Call = { url: string; init: RequestInit };
function stubFetch(
  responses: Array<() => Response>,
): { fetchFn: typeof fetch; calls: Call[] } {
  const calls: Call[] = [];
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    const next = responses.shift();
    if (next === undefined) throw new Error('stub fetch exhausted');
    return next();
  }) as typeof fetch;
  return { fetchFn, calls };
}

const ok = (relevant: number, evidence: number, inputTokens = 100) => () =>
  new Response(
    JSON.stringify({
      model: 'jev-1.13.0',
      answers: {
        relevant: { type: 'noul', noul: relevant },
        evidence: { type: 'noul', noul: evidence },
      },
      usage: { input_tokens: inputTokens, output_tokens: 48 },
    }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  );

describe('typesafeNouls client', () => {
  it('posts the state, model and typed questions with bearer auth', async () => {
    const { fetchFn, calls } = stubFetch([ok(0.9, 0.4, 312)]);
    const result = await typesafeNouls(
      { question: 'q', session: ['a'] },
      RERANK_QUESTIONS,
      { apiKey: KEY, fetchFn, cacheDir: tempDir() },
    );
    expect(result).toEqual({
      nouls: { relevant: 0.9, evidence: 0.4 },
      inputTokens: 312,
      cached: false,
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe(TYPESAFE_URL);
    expect(calls[0]!.init.method).toBe('POST');
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers.Authorization).toBe(`Bearer ${KEY}`);
    expect(headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(String(calls[0]!.init.body))).toEqual({
      state: { question: 'q', session: ['a'] },
      model: 'jev-1.13.0',
      questions: {
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
      },
    });
  });

  it('answers a repeat from the on-disk cache without a request', async () => {
    const cacheDir = tempDir();
    const first = stubFetch([ok(0.2, 0.7)]);
    await typesafeNouls({ s: 1 }, RERANK_QUESTIONS, {
      apiKey: KEY,
      fetchFn: first.fetchFn,
      cacheDir,
    });
    const second = stubFetch([]);
    const again = await typesafeNouls({ s: 1 }, RERANK_QUESTIONS, {
      apiKey: KEY,
      fetchFn: second.fetchFn,
      cacheDir,
    });
    expect(second.calls).toHaveLength(0);
    expect(again).toEqual({
      nouls: { relevant: 0.2, evidence: 0.7 },
      inputTokens: 100,
      cached: true,
    });
    // a different model is a different key
    const third = stubFetch([ok(0.1, 0.1)]);
    await typesafeNouls({ s: 1 }, RERANK_QUESTIONS, {
      apiKey: KEY,
      fetchFn: third.fetchFn,
      cacheDir,
      model: 'jev-other',
    });
    expect(third.calls).toHaveLength(1);
  });

  it('retries a 429 after its retry-after, then succeeds', async () => {
    const sleeps: number[] = [];
    const { fetchFn, calls } = stubFetch([
      () => new Response('slow down', { status: 429, headers: { 'retry-after': '2' } }),
      () => new Response('oops', { status: 503 }),
      ok(0.5, 0.5),
    ]);
    const result = await typesafeNouls({ s: 2 }, RERANK_QUESTIONS, {
      apiKey: KEY,
      fetchFn,
      cacheDir: null,
      sleep: async (ms) => {
        sleeps.push(ms);
      },
    });
    expect(calls).toHaveLength(3);
    expect(sleeps[0]).toBe(2_000);
    expect(sleeps[1]).toBeGreaterThan(0);
    expect(result.nouls.evidence).toBe(0.5);
  });

  it('gives up after five attempts and never names the key', async () => {
    const { fetchFn, calls } = stubFetch(
      Array.from({ length: 5 }, () => () => new Response('busy', { status: 500 })),
    );
    const failure = typesafeNouls({ s: 3 }, RERANK_QUESTIONS, {
      apiKey: KEY,
      fetchFn,
      cacheDir: null,
      sleep: async () => {},
    });
    await expect(failure).rejects.toThrow(/TypeSafe request failed with status 500/);
    await failure.catch((error: Error) => expect(error.message).not.toContain(KEY));
    expect(calls).toHaveLength(5);
  });

  it('does not retry a 400', async () => {
    const { fetchFn, calls } = stubFetch([() => new Response('bad', { status: 400 })]);
    await expect(
      typesafeNouls({ s: 4 }, RERANK_QUESTIONS, {
        apiKey: KEY,
        fetchFn,
        cacheDir: null,
        sleep: async () => {},
      }),
    ).rejects.toThrow(/status 400/);
    expect(calls).toHaveLength(1);
  });

  it('the limiter caps work in flight', async () => {
    const limit = createLimiter(2);
    let active = 0;
    let peak = 0;
    await Promise.all(
      Array.from({ length: 6 }, () =>
        limit(async () => {
          active += 1;
          peak = Math.max(peak, active);
          await new Promise((resolve) => setTimeout(resolve, 5));
          active -= 1;
        }),
      ),
    );
    expect(peak).toBe(2);
  });
});

describe('fitTurnsToBudget', () => {
  it('keeps every turn under budget and prefers question overlap when cutting', () => {
    const turns = ['x'.repeat(40), 'my bicycle is a Trek', 'y'.repeat(40)];
    expect(fitTurnsToBudget(turns, 'Which bicycle?', 1_000)).toEqual(turns);
    expect(fitTurnsToBudget(turns, 'Which bicycle?', 50)).toEqual([
      'my bicycle is a Trek',
    ]);
    // whole turns only, original order kept
    expect(fitTurnsToBudget(['aa bicycle', 'b'.repeat(30), 'cc bicycle'], 'bicycle', 25)).toEqual([
      'aa bicycle',
      'cc bicycle',
    ]);
  });
});

describe('rerankSessionOrder', () => {
  const sessions: Record<string, { date: string; turns: Array<{ role: string; content: string }> }> = {
    a: { date: 'd1', turns: [{ role: 'user', content: 'alpha' }] },
    b: { date: 'd2', turns: [{ role: 'user', content: 'beta' }, { role: 'assistant', content: 'reply b' }] },
    c: { date: 'd3', turns: [{ role: 'user', content: 'gamma' }] },
    d: { date: 'd4', turns: [{ role: 'user', content: 'delta' }] },
  };
  const scores: Record<string, [number, number]> = {
    alpha: [0.1, 0.1],
    beta: [0.2, 0.9],
    gamma: [0.9, 0.2],
  };
  const nouls = (): { fn: TypesafeNouls; states: Array<Record<string, unknown>> } => {
    const states: Array<Record<string, unknown>> = [];
    const fn: TypesafeNouls = async (state) => {
      const s = state as { session: string[] };
      states.push(state as Record<string, unknown>);
      const first = s.session[0]!.replace(/^user: /, '');
      const [relevant, evidence] = scores[first]!;
      return { nouls: { relevant, evidence }, inputTokens: 10, cached: first === 'gamma' };
    };
    return { fn, states };
  };

  it('orders the pool by 0.7 evidence + 0.3 relevant and keeps the rest after it', async () => {
    const { fn, states } = nouls();
    const result = await rerankSessionOrder(['a', 'b', 'c', 'd'], {
      question: 'which?',
      questionDate: 'qd',
      pool: 3,
      sessionChars: 8_000,
      contextRoles: 'user',
      sessionFor: (id) => sessions[id],
      nouls: fn,
    });
    // b = 0.69, c = 0.41, a = 0.10
    expect(result.order).toEqual(['b', 'c', 'a', 'd']);
    expect(result.stats).toEqual({ candidates: 3, calls: 2, cached: 1, inputTokens: 20, blocked: 0 });
    expect(states[1]).toEqual({
      question: 'which?',
      question_date: 'qd',
      session_date: 'd2',
      session: ['beta'],
    });
  });

  it('shows assistant turns only when the reader would see them', async () => {
    const { fn, states } = nouls();
    await rerankSessionOrder(['b'], {
      question: 'which?',
      questionDate: 'qd',
      pool: 3,
      sessionChars: 8_000,
      contextRoles: 'all',
      sessionFor: (id) => sessions[id],
      nouls: fn,
    });
    expect(states[0]!.session).toEqual(['user: beta', 'assistant: reply b']);
  });

  it('ties keep the lexical order', async () => {
    const fn: TypesafeNouls = async () => ({ nouls: { relevant: 0.5, evidence: 0.5 }, inputTokens: 1, cached: false });
    const result = await rerankSessionOrder(['c', 'a', 'b'], {
      question: 'q',
      questionDate: 'qd',
      pool: 30,
      sessionChars: 8_000,
      contextRoles: 'user',
      sessionFor: (id) => sessions[id],
      nouls: fn,
    });
    expect(result.order).toEqual(['c', 'a', 'b']);
  });

  it('never sends a sensitive candidate and scores it from its lexical rank', async () => {
    const { fn, states } = nouls();
    const secret = {
      date: 'd9',
      turns: [{ role: 'user', content: 'my api key is sk-abcdefghijklmnopqrstuvwxyz0123456789' }],
    };
    const result = await rerankSessionOrder(['s', 'a'], {
      question: 'which?',
      questionDate: 'qd',
      pool: 2,
      sessionChars: 8_000,
      contextRoles: 'user',
      sessionFor: (id) => (id === 's' ? secret : sessions[id]),
      nouls: fn,
    });
    expect(states.map((s) => (s as { session: string[] }).session[0])).toEqual(['alpha']);
    expect(result.stats.blocked).toBe(1);
    expect(result.order).toEqual(['s', 'a']);
  });
});

class ScriptedCompletionClient implements LongMemEvalCompletionClient {
  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}
  async completeWithUsage(_messages: ChatMessage[]): Promise<LlmCompletion> {
    const content = this.outputs.shift();
    if (content === undefined) throw new Error('scripted completion exhausted');
    return {
      content,
      model: this.model,
      usage: {
        promptTokens: 1,
        completionTokens: 1,
        totalTokens: 2,
        cachedPromptTokens: 0,
        reasoningTokens: 0,
        costUsd: 0,
      },
    };
  }
}
const forbidden = (role: string): LongMemEvalCompletionClient => ({
  model: role,
  completeWithUsage: async () => {
    throw new Error(`${role} must not be called`);
  },
});

function fixture(questionType: string): LongMemEvalInstance {
  // lexically, "strong" beats "evidence"; the reranker knows better
  return {
    question_id: 'rr-1',
    question_type: questionType,
    question: 'Which marathon did I run?',
    answer: 'Berlin',
    question_date: '2024/01/05 (Fri) 09:00',
    haystack_session_ids: ['strong', 'evidence', 'noise'],
    haystack_dates: [
      '2024/01/01 (Mon) 09:00',
      '2024/01/02 (Tue) 09:00',
      '2024/01/03 (Wed) 09:00',
    ],
    haystack_sessions: [
      [{ role: 'user', content: 'marathon marathon marathon run run' }],
      [{ role: 'user', content: 'I finished the Berlin marathon.', has_answer: true }],
      [{ role: 'user', content: 'Compare credit card rewards for travel.' }],
    ],
    answer_session_ids: ['evidence'],
  };
}

describe('re-ranking inside the answer harness', () => {
  const byContent: TypesafeNouls = async (state) => {
    const text = (state as { session: string[] }).session.join(' ');
    const evidence = text.includes('Berlin') ? 0.95 : 0.05;
    return { nouls: { relevant: 0.5, evidence }, inputTokens: 50, cached: false };
  };

  for (const retrievalUnit of ['session', 'turn'] as const) {
    it(`promotes the evidence session into top-k (${retrievalUnit} unit)`, async () => {
      const plain = await evaluateLongMemEvalAnswerInstance(
        fixture('single-session-user'),
        forbidden('reader'),
        forbidden('judge'),
        { topK: 1, contextBytes: 4_096, retrievalOnly: true, retrievalUnit },
      );
      expect(plain.retrievedSessionIds).toEqual(['strong']);
      expect(plain.rerank).toBeUndefined();
      const reranked = await evaluateLongMemEvalAnswerInstance(
        fixture('single-session-user'),
        forbidden('reader'),
        forbidden('judge'),
        {
          topK: 1,
          contextBytes: 4_096,
          retrievalOnly: true,
          retrievalUnit,
          rerank: { nouls: byContent, pool: 30, sessionChars: 8_000 },
        },
      );
      expect(reranked.retrievedSessionIds).toEqual(['evidence']);
      expect(reranked.rerank).toEqual({
        candidates: 2,
        calls: 2,
        cached: 0,
        inputTokens: 100,
        blocked: 0,
      });
      const summary = summarizeLongMemEvalAnswers([reranked]);
      expect(summary.rerankUsage).toEqual({
        candidates: 2,
        calls: 2,
        cached: 0,
        blocked: 0,
        inputTokens: 100,
        costUsd: (100 * 0.042) / 1_000_000,
      });
    });
  }

  it('--rerank none leaves the observation byte-identical to a run without the option', async () => {
    // timings differ run to run; everything else must match byte for byte
    const strip = (o: Record<string, unknown>) =>
      JSON.parse(JSON.stringify(o, (key, value) => (/Ms$/.test(key) ? 0 : value))) as Record<string, unknown>;
    for (const retrievalUnit of ['session', 'turn'] as const) {
      const run = async (withNone: boolean) =>
        strip(
          (await evaluateLongMemEvalAnswerInstance(
            fixture('multi-session'),
            new ScriptedCompletionClient('reader', ['Berlin']),
            new ScriptedCompletionClient('judge', ['yes']),
            {
              topK: 2,
              multiSessionTopK: 2,
              contextBytes: 4_096,
              retrievalUnit,
              ...(withNone ? { rerank: undefined } : {}),
            },
          )) as unknown as Record<string, unknown>,
        );
      expect(JSON.stringify(await run(true))).toBe(JSON.stringify(await run(false)));
    }
  });

  it('the runner parses the flags with their defaults', () => {
    const plain = parseArgs([]);
    expect(plain.rerank).toBe('none');
    expect(plain.rerankPool).toBe(30);
    expect(plain.rerankSessionChars).toBe(8_000);
    expect(plain.rerankModel).toBe('jev-1.13.0');
    expect(plain.rerankKeyEnv).toBe('TYPESAFE_AI_API_KEY');
    expect(plain.rerankConcurrency).toBe(16);
    const set = parseArgs([
      '--rerank',
      'typesafe',
      '--rerank-pool',
      '20',
      '--rerank-session-chars',
      '4000',
      '--rerank-model',
      'jev-2',
      '--rerank-key-env',
      'MY_TS_KEY',
      '--rerank-concurrency',
      '8',
    ]);
    expect(set).toMatchObject({
      rerank: 'typesafe',
      rerankPool: 20,
      rerankSessionChars: 4_000,
      rerankModel: 'jev-2',
      rerankKeyEnv: 'MY_TS_KEY',
      rerankConcurrency: 8,
    });
    expect(() => parseArgs(['--rerank', 'cohere'])).toThrow(/--rerank must be none or typesafe/);
    expect(() => parseArgs(['--rerank', 'typesafe', '--memory-system', 'builtin:bm25'])).toThrow(/--rerank/);
  });
});
