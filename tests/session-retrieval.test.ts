import { describe, expect, it } from 'vitest';
import { buildLongMemEvalAnswerContext } from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { questionKindFromText } from '../src/knowledge/question-kind.js';
import type { QuestionKind } from '../src/knowledge/question-kind.js';
import {
  buildReadingPrompt,
  readingQuestionDate,
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
  const notes = kind.aggregation || kind.temporal || kind.update;
  const context = buildLongMemEvalAnswerContext(
    instanceFor(question, askedAt, chosen),
    chosen.map((session) => ({
      opId: session.id,
      ts: session.date,
      text: sessionReadingText(session, kind.assistantRecall),
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
