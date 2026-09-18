import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  buildLongMemEvalAnswerContext,
  evaluateLongMemEvalAnswerInstance,
  finalAnswerLine as harnessFinalAnswerLine,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import { finalAnswerLine as productFinalAnswerLine } from '../src/knowledge/answer-line.js';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { questionKindFromText } from '../src/knowledge/question-kind.js';
import type { QuestionKind } from '../src/knowledge/question-kind.js';
import {
  buildReadingPrompt,
  readingContextRoles,
  readingQuestionDate,
  readsInNotes,
  retrieveSessions,
  sessionReadingText,
  type RetrievableSession,
  type SessionRetrievalOptions,
} from '../src/knowledge/session-retrieval.js';
import type { TypesafeNouls } from '../src/evals/typesafe-rerank.js';

const ASKED_AT = new Date('2024-03-01T09:00:00.000Z');

const SESSIONS: RetrievableSession[] = [
  {
    id: 'museums',
    date: '2024-01-12T09:00:00.000Z',
    turns: [
      {
        role: 'user',
        text: 'I visited three museums in Lisbon last week and spent 42 euros on tickets.',
      },
      {
        role: 'assistant',
        text: 'Lisbon has a wonderful museum scene; the Gulbenkian is a favourite.',
      },
      { role: 'user', text: 'The tile museum was the best of the three.' },
    ],
  },
  {
    id: 'camera',
    date: '2024-02-03T09:00:00.000Z',
    turns: [
      { role: 'user', text: 'I bought a second-hand camera for 210 euros.' },
      { role: 'assistant', text: 'A good price for a used body.' },
    ],
  },
  {
    id: 'bikes',
    date: '2024-02-27T09:00:00.000Z',
    turns: [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
      { role: 'assistant', text: 'Two bikes is a tidy stable.' },
    ],
  },
];

function instanceFor(
  question: string,
  askedAt: Date,
  sessions: readonly RetrievableSession[],
): LongMemEvalInstance {
  return {
    question_id: 'parity-1',
    question_type: 'single-session-user',
    question,
    answer: 'GOLD_ANSWER_NOT_IN_HISTORY',
    question_date: readingQuestionDate(askedAt),
    haystack_session_ids: sessions.map(({ id }) => id),
    haystack_dates: sessions.map(({ date }) => date),
    haystack_sessions: sessions.map(({ turns }) =>
      turns.map(({ role, text }) => ({ role, content: text })),
    ),
    answer_session_ids: [sessions[0]!.id],
  };
}

/** The harness prompt for the same inputs: the builder the benchmark runs measure. */
function harnessPrompt(
  question: string,
  askedAt: Date,
  chosen: readonly RetrievableSession[],
  options: SessionRetrievalOptions,
): { system: string; user: string } {
  const { kind } = options;
  const notes = readsInNotes(kind);
  const context = buildLongMemEvalAnswerContext(
    instanceFor(question, askedAt, chosen),
    chosen.map((session) => ({
      opId: session.id,
      ts: session.date,
      text: sessionReadingText(session, readingContextRoles(kind)),
    })),
    options.contextBytes,
    [],
    notes ? 'notes' : 'direct',
    undefined,
    options.dateDistances,
    options.computedNotes,
    false,
    false,
    undefined,
    undefined,
    kind.preference,
  );
  return {
    system: context.messages[0]!.content,
    user: context.messages[1]!.content,
  };
}

function kindOf(overrides: Partial<QuestionKind> = {}): QuestionKind {
  return {
    aggregation: false,
    temporal: false,
    update: false,
    preference: false,
    assistantRecall: false,
    ...overrides,
  };
}

describe('the shared session retrieval module builds the harness prompt', () => {
  const cases: Array<{
    name: string;
    question: string;
    options: SessionRetrievalOptions;
  }> = [
    {
      name: 'a plain lookup with both reader blocks on',
      question: 'Which museum did I like best in Lisbon?',
      options: {
        kind: kindOf(),
        topK: 4,
        contextBytes: 24_576,
        dateDistances: true,
        computedNotes: true,
      },
    },
    {
      name: 'an aggregation question (Notes-then-Answer)',
      question: 'How many museums did I visit in total?',
      options: {
        kind: kindOf({ aggregation: true }),
        topK: 12,
        contextBytes: 24_576,
        dateDistances: true,
        computedNotes: true,
      },
    },
    {
      name: 'a temporal question',
      question: 'How many days passed between the museums and the camera?',
      options: {
        kind: kindOf({ temporal: true }),
        topK: 10,
        contextBytes: 16_384,
        dateDistances: true,
        computedNotes: true,
      },
    },
    {
      name: 'a preference question (personalised prompt)',
      question: 'Can you recommend a museum for my next trip?',
      options: {
        kind: kindOf({ preference: true }),
        topK: 4,
        contextBytes: 8_192,
        dateDistances: false,
        computedNotes: false,
      },
    },
    {
      name: 'an assistant-recall question (assistant turns shown)',
      question: 'What did you tell me about the Gulbenkian?',
      options: {
        kind: kindOf({ assistantRecall: true }),
        topK: 4,
        contextBytes: 24_576,
        dateDistances: true,
        computedNotes: true,
      },
    },
    {
      name: 'both reader blocks off and a tight budget',
      question: 'Which museum did I like best in Lisbon?',
      options: {
        kind: kindOf(),
        topK: 4,
        contextBytes: 4_096,
        dateDistances: false,
        computedNotes: false,
      },
    },
  ];

  for (const { name, question, options } of cases) {
    it(`matches the harness prompt byte for byte: ${name}`, () => {
      const product = buildReadingPrompt(question, ASKED_AT, SESSIONS, options);
      const harness = harnessPrompt(question, ASKED_AT, SESSIONS, options);
      expect(product.system).toBe(harness.system);
      expect(product.user).toBe(harness.user);
    });
  }

  it('reads the question kind from the question text, as the product does', () => {
    const question = 'How many museums did I visit in total?';
    const kind = questionKindFromText(question);
    expect(kind.aggregation).toBe(true);
    const options: SessionRetrievalOptions = {
      kind,
      topK: 12,
      contextBytes: 24_576,
      dateDistances: true,
      computedNotes: true,
    };
    const product = buildReadingPrompt(question, ASKED_AT, SESSIONS, options);
    expect(product.user).toBe(
      harnessPrompt(question, ASKED_AT, SESSIONS, options).user,
    );
    expect(product.system).toContain('Notes:');
  });
});

/** The dataset spelling of each fixture session's date; `SESSIONS` carries the same instants. */
const DATASET_DATES = [
  '2024/01/12 (Fri) 09:00',
  '2024/02/03 (Sat) 09:00',
  '2024/02/27 (Tue) 09:00',
];

class ScriptedClient implements LongMemEvalCompletionClient {
  readonly calls: ChatMessage[][] = [];

  constructor(
    readonly model: string,
    private readonly reply: string,
  ) {}

  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    this.calls.push(structuredClone(messages));
    return {
      content: this.reply,
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

/**
 * The parity that matters: the harness's whole path — its own store, its own turn documents, its
 * own search and its own context builder — against the product's `retrieveSessions` and
 * `buildReadingPrompt`. Nothing here is built from the module's helpers, so the two sides can
 * genuinely disagree, on the chosen sessions and on the prompt alike.
 */
describe('the product and the harness retrieve and read alike, end to end', () => {
  const questions = [
    'How many museums did I visit in total?',
    'Which museum did I like best in Lisbon?',
    'What did you tell me about the Gulbenkian?',
  ];

  for (const question of questions) {
    it(`agrees with the harness run for: ${question}`, async () => {
      const reader = new ScriptedClient('reader', 'Notes: one\nAnswer: the tile museum');
      const judge = new ScriptedClient('judge', 'yes');
      const observation = await evaluateLongMemEvalAnswerInstance(
        {
          ...instanceFor(question, ASKED_AT, SESSIONS),
          question_type: 'multi-session',
          haystack_dates: DATASET_DATES,
        },
        reader,
        judge,
        {
          classify: 'text',
          formation: 'raw',
          retrievalUnit: 'turn',
          topK: 4,
          multiSessionTopK: 12,
          temporalTopK: 10,
          contextBytes: 24_576,
          dateDistances: true,
          computedNotes: true,
          readingStrategy: 'notes',
        },
      );
      expect(observation.status).toBe('judged');

      const kind = questionKindFromText(question);
      const options: SessionRetrievalOptions = {
        kind,
        topK: 4,
        aggregationTopK: 12,
        temporalTopK: 10,
        contextBytes: 24_576,
        dateDistances: true,
        computedNotes: true,
      };
      const retrieved = await retrieveSessions(
        question,
        ASKED_AT,
        SESSIONS,
        options,
      );
      // the same sessions, in the same order, from two independent indexes
      expect(retrieved.chosen).toEqual(observation.retrievedSessionIds);
      const byId = new Map(SESSIONS.map((session) => [session.id, session]));
      const product = buildReadingPrompt(
        question,
        ASKED_AT,
        retrieved.chosen.map((id) => byId.get(id)!),
        options,
      );
      const sent = reader.calls[0]!;
      expect(product.system).toBe(sent[0]!.content);
      expect(product.user).toBe(sent[1]!.content);
    });
  }
});

describe('retrieveSessions', () => {
  const baseOptions: SessionRetrievalOptions = {
    kind: kindOf(),
    topK: 4,
    contextBytes: 24_576,
    dateDistances: true,
    computedNotes: true,
  };

  it('ranks the session that answers the question first and cuts to the depth', async () => {
    const result = await retrieveSessions(
      'How many museums did I visit in total?',
      ASKED_AT,
      SESSIONS,
      { ...baseOptions, kind: kindOf({ aggregation: true }), topK: 12 },
    );
    expect(result.ranked[0]).toBe('museums');
    expect(result.chosen[0]).toBe('museums');
    expect(result.chosen.length).toBeLessThanOrEqual(12);
    expect(result.rerank).toBeUndefined();
  });

  it('cuts to the depth the options give', async () => {
    const result = await retrieveSessions(
      'How many euros did I spend on the camera and the museums?',
      ASKED_AT,
      SESSIONS,
      { ...baseOptions, topK: 1 },
    );
    expect(result.chosen).toHaveLength(1);
    expect(result.ranked.length).toBeGreaterThan(1);
  });

  it('a re-rank pool never reorders the head a plain run would have seen', async () => {
    // the harness aggregates the results a plain run would have fetched and only appends what
    // the deeper re-rank search found; a module that summed the whole deep list instead would
    // hand the pool a different order, and the benchmark's numbers would stop describing it
    const question = 'How much coffee did I drink on my trips?';
    const many: RetrievableSession[] = Array.from({ length: 26 }, (_, i) => ({
      id: `s${String(i).padStart(2, '0')}`,
      date: `2024-01-${String((i % 28) + 1).padStart(2, '0')}T09:00:00.000Z`,
      // the early sessions each hold one strong turn; the late ones hold many weak ones, which
      // only a deeper search sees and which would climb if the whole list were summed
      turns:
        i < 13
          ? [{ role: 'user' as const, text: 'I drank coffee on my trips to Porto.' }]
          : Array.from({ length: 6 }, () => ({
              role: 'user' as const,
              text: 'The trips were long.',
            })),
    }));
    const constant: TypesafeNouls = async () => ({
      nouls: { relevant: 0.5, evidence: 0.5 },
      inputTokens: 1,
      cached: false,
    });
    const plain = await retrieveSessions(question, ASKED_AT, many, baseOptions);
    const pooled = await retrieveSessions(question, ASKED_AT, many, {
      ...baseOptions,
      rerank: { client: constant },
    });
    expect(plain.ranked.length).toBeGreaterThan(4);
    // the deeper search really does find more than the plain one
    expect(pooled.ranked.length).toBeGreaterThan(plain.ranked.length);
    expect(pooled.ranked.slice(0, plain.ranked.length)).toEqual(plain.ranked);
  });

  it('orders the sessions inside a time range ahead of those outside it', async () => {
    const question = 'What did I buy in February?';
    const plain = await retrieveSessions(question, ASKED_AT, SESSIONS, {
      ...baseOptions,
      kind: kindOf({ temporal: true }),
      topK: 10,
    });
    const ranged = await retrieveSessions(question, ASKED_AT, SESSIONS, {
      ...baseOptions,
      kind: kindOf({ temporal: true }),
      topK: 10,
      timeRange: { start: '2024-02-01', end: '2024-02-29' },
    });
    expect(new Set(ranged.chosen)).toEqual(new Set(plain.chosen));
    const february = new Set(['camera', 'bikes']);
    const outside = ranged.chosen.findIndex((id) => !february.has(id));
    const inside = ranged.chosen.findIndex((id) => february.has(id));
    if (outside >= 0 && inside >= 0) expect(inside).toBeLessThan(outside);
  });

  it('reads nothing for a question the index cannot be searched for', async () => {
    // a product question, unlike a benchmark one, can carry no content word at all or run long
    for (const question of ['why?', 'and the of a to']) {
      await expect(
        retrieveSessions(question, ASKED_AT, SESSIONS, baseOptions),
      ).resolves.toEqual({ ranked: [], chosen: [] });
    }
    const tooLong = Array.from({ length: 400 }, (_, i) => `museum${i}`).join(' ');
    await expect(
      retrieveSessions(tooLong, ASKED_AT, SESSIONS, baseOptions),
    ).resolves.toEqual({ ranked: [], chosen: [] });
  });

  it('re-ranks the pool with the TypeSafe client and never touches the network', async () => {
    const question = 'How many euros did I spend on the museums and the camera?';
    const plain = await retrieveSessions(question, ASKED_AT, SESSIONS, {
      ...baseOptions,
      kind: kindOf({ aggregation: true }),
      topK: 12,
    });
    expect(plain.ranked.length).toBeGreaterThan(1);
    // a stub that scores every candidate above the one before it, reversing the lexical order
    const seen: string[] = [];
    const client: TypesafeNouls = async (state) => {
      const session = state as { session: string[]; session_date: string };
      seen.push(session.session_date);
      const score = 0.1 * seen.length;
      return {
        nouls: { relevant: score, evidence: score },
        inputTokens: 11,
        cached: false,
      };
    };
    const result = await retrieveSessions(question, ASKED_AT, SESSIONS, {
      ...baseOptions,
      kind: kindOf({ aggregation: true }),
      topK: 12,
      rerank: { client, pool: 30, sessionChars: 4_000 },
    });
    expect(seen).toHaveLength(plain.ranked.length);
    expect(result.ranked).toEqual([...plain.ranked].reverse());
    expect(result.chosen[0]).toBe(plain.ranked.at(-1));
    expect(result.rerank).toMatchObject({
      candidates: plain.ranked.length,
      calls: seen.length,
      cached: 0,
      inputTokens: 11 * seen.length,
      blocked: 0,
    });
  });
});

/**
 * `finalAnswerLine` exists twice: the product's copy in `src/knowledge/answer-line.ts`
 * and the harness's in `src/evals/longmemeval-answer.ts`. They cannot be one function
 * yet — `src/evals` imports `src/llm`, so `src/llm` importing `src/evals` would be a
 * cycle — and until now the two were eye-checked only. This is the contract that makes
 * a benchmark score describe product behaviour: the same reply must become the same
 * answer on both sides, so the two must be the same function, character for character.
 */
function finalAnswerLineSource(path: string): string {
  const text = readFileSync(resolve(path), 'utf8');
  const start = text.indexOf('export function finalAnswerLine(');
  expect(start, `${path} declares finalAnswerLine`).toBeGreaterThan(-1);
  // the declaration through the closing brace in column 1
  const end = text.indexOf('\n}\n', start);
  expect(end, `${path} closes finalAnswerLine`).toBeGreaterThan(start);
  return text.slice(start, end + 2);
}

describe('finalAnswerLine is one function kept in two files', () => {
  it('is byte-identical in the product and in the harness', () => {
    const product = finalAnswerLineSource('src/knowledge/answer-line.ts');
    const harness = finalAnswerLineSource('src/evals/longmemeval-answer.ts');
    expect(product).toBe(harness);
    // and it is a real implementation, not an empty shell the comparison would pass on
    expect(product).toContain("lastIndexOf('Answer:')");
  });

  it('turns the same replies into the same answer on both sides', () => {
    for (const reply of [
      'Answer: Two bikes.',
      'Notes:\n- 2024-02-27: two bikes\nAnswer: Two bikes.',
      'Answer: 42; notes: I subtracted the sold one',
      'Answer:',
      '   Answer:    ',
      'The user said "Answer: 42 bikes" earlier, so Answer: 42 bikes.',
      'I do not know.',
      '',
      '## Answer: Two bikes.',
    ]) {
      expect(productFinalAnswerLine(reply), JSON.stringify(reply)).toBe(
        harnessFinalAnswerLine(reply),
      );
    }
  });
});
