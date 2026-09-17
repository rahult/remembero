import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  buildLongMemEvalAnswerContext,
  buildLongMemEvalJudgePrompt,
  evaluateLongMemEvalAnswerInstance,
  longMemEvalAnswerRun,
  parseLongMemEvalJudgeLabel,
  summarizeLongMemEvalAnswers,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { parseArgs } from '../src/evals/run-longmemeval-answer.js';

class ScriptedCompletionClient implements LongMemEvalCompletionClient {
  readonly calls: Array<{ messages: ChatMessage[]; maxTokens?: number }> = [];

  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}

  async completeWithUsage(
    messages: ChatMessage[],
    options: { maxTokens?: number } = {},
  ): Promise<LlmCompletion> {
    this.calls.push({
      messages: structuredClone(messages),
      maxTokens: options.maxTokens,
    });
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

function instance(
  overrides: Partial<LongMemEvalInstance> = {},
): LongMemEvalInstance {
  return {
    question_id: 'question-1',
    question_type: 'single-session-user',
    question: 'What degree did I graduate with?',
    answer: 'Business Administration',
    question_date: '2024/01/03 (Wed) 09:00',
    haystack_session_ids: ['noise', 'evidence'],
    haystack_dates: ['2024/01/01 (Mon) 09:00', '2024/01/02 (Tue) 09:00'],
    haystack_sessions: [
      [
        { role: 'user', content: 'Compare credit card rewards for travel.' },
        { role: 'assistant', content: 'Long generic reward-card explanation.' },
      ],
      [
        {
          role: 'user',
          content: 'My degree was in Business Administration.',
          has_answer: true,
        },
        { role: 'assistant', content: 'Long generic graduation explanation.' },
      ],
    ],
    answer_session_ids: ['evidence'],
    ...overrides,
  };
}

describe('LongMemEval end-to-end answer evaluation', () => {
  it('builds bounded answer context without leaking labels or a separate gold answer', () => {
    const test = instance({ answer: 'GOLD_REFERENCE_NOT_IN_HISTORY' });
    const context = buildLongMemEvalAnswerContext(
      test,
      [
        {
          opId: 'evidence',
          ts: '2024-01-02T09:00:00.000Z',
          text: 'user: My degree was in Business Administration.',
        },
        {
          opId: 'blocked',
          ts: '2024-01-01T09:00:00.000Z',
          text: '[sensitive source omitted]',
          redacted: true,
        },
      ],
      4_096,
    );
    const prompt = context.messages.map(({ content }) => content).join('\n');
    expect(prompt).toContain('Business Administration');
    expect(prompt).not.toContain('GOLD_REFERENCE_NOT_IN_HISTORY');
    expect(prompt).not.toContain('has_answer');
    expect(context.contextSessionIds).toEqual(['evidence']);
    expect(context.redactedRetrievedSessions).toBe(1);

    const preference = buildLongMemEvalAnswerContext(
      instance({ question_type: 'single-session-preference' }),
      [
        {
          opId: 'evidence',
          ts: '2024-01-02T09:00:00.000Z',
          text: 'user: I like quiet hotels.',
        },
      ],
      4_096,
    );
    expect(preference.messages[0]?.content).toContain('general knowledge');
    expect(preference.messages[0]?.content).toContain(
      'do not invent facts about the user',
    );
  });

  it('date distances: each session header states how long before the question date it happened', () => {
    const test = instance({ question_date: '2024/03/01 (Fri) 09:00' });
    const context = buildLongMemEvalAnswerContext(
      test,
      [
        {
          opId: 'a',
          ts: '2023-12-29T09:00:00.000Z',
          text: 'user: I bought a kit.',
        },
        {
          opId: 'b',
          ts: '2024-02-29T09:00:00.000Z',
          text: 'user: I bought another.',
        },
      ],
      4_096,
      [],
      'direct',
      undefined,
      true,
    );
    const prompt = context.messages[1]?.content ?? '';
    expect(prompt).toContain(
      'Session date: 2023-12-29 (63 days, about 9 weeks or 2 months, before the question date 2024-03-01)',
    );
    expect(prompt).toContain(
      'Session date: 2024-02-29 (1 day before the question date 2024-03-01)',
    );
    // off by default
    const plain = buildLongMemEvalAnswerContext(test, [
      {
        opId: 'a',
        ts: '2023-12-29T09:00:00.000Z',
        text: 'user: I bought a kit.',
      },
    ]);
    expect(plain.messages[1]?.content).not.toContain(
      'before the question date',
    );
  });

  it('runs durable formation, real local retrieval, answer generation, and judging', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096 },
    );
    expect(observation).toMatchObject({
      status: 'judged',
      correct: true,
      retrievedSessionIds: ['evidence'],
      contextSessionIds: ['evidence'],
      hypothesis: 'Business Administration',
      redactedRetrievedSessions: 0,
      retrievalRoute: 'local' as const,
      embeddingModel: null,
      embeddingCalls: 0,
      embeddingUsage: null,
      semanticPreparationCalls: 0,
      semanticPreparationUsage: null,
    });
    expect(observation.retrieval?.recallAtK).toBe(1);
    expect(observation.context?.recallAtK).toBe(1);
    expect(reader.calls[0]?.maxTokens).toBe(4_096);
    expect(reader.calls[0]?.messages.at(-1)?.content).not.toContain(
      'has_answer',
    );
    expect(reader.calls[0]?.messages.at(-1)?.content).not.toContain(
      'generic graduation',
    );
    expect(observation.contextRoles).toBe('user');
    expect(judge.calls[0]?.maxTokens).toBe(16);
    const summary = summarizeLongMemEvalAnswers([observation]);
    expect(summary).toMatchObject({
      questions: 1,
      judgedQuestions: 1,
      correct: 1,
      accuracy: 1,
      retrievalRecallAtK: 1,
      contextRecallAtK: 1,
    });
    expect(summary.readerUsage).toMatchObject({
      calls: 1,
      totalTokens: 12,
      costUsd: 0.001,
    });
    expect(summary.judgeUsage).toMatchObject({
      calls: 1,
      totalTokens: 12,
      costUsd: 0.001,
    });
  });

  it('extracted formation runs the product transcript extraction per session and keys facts to the session', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    // one call per session, in haystack order: the noise session yields nothing
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096, formation: 'extracted', extractor },
    );
    expect(observation).toMatchObject({
      status: 'judged',
      correct: true,
      retrievedSessionIds: ['evidence'],
      contextSessionIds: ['evidence'],
    });
    expect(observation.extraction).toMatchObject({
      calls: 2,
      sessionsWithFacts: 1,
      facts: 1,
      errors: 0,
    });
    expect(observation.extraction?.usage).toMatchObject({ totalTokens: 24 });
    // the extractor sees the transcript prompt and USER:/ASSISTANT: turns
    expect(extractor.calls[1]?.messages[0]?.content).toContain('transcript');
    expect(extractor.calls[1]?.messages.at(-1)?.content).toContain(
      'USER: My degree was in Business Administration.',
    );
    // the reader still reads the raw session text of the retrieved session
    expect(reader.calls[0]?.messages.at(-1)?.content).toContain(
      'Business Administration',
    );
    const summary = summarizeLongMemEvalAnswers([observation]);
    expect(summary.extractionUsage).toMatchObject({
      calls: 2,
      totalTokens: 24,
    });
    const run = longMemEvalAnswerRun([observation], 'reader', 'judge', {
      generatedAt: '2026-09-09T00:00:00.000Z',
      formation: 'extracted',
      extractionModel: 'dialect',
    });
    expect(run).toMatchObject({
      formation: 'extracted-transcript-facts',
      extractionModel: 'dialect',
    });
  });

  it('compacts long assistant turns before extraction while keeping user turns whole', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const long = 'x'.repeat(2_000);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        haystack_sessions: [
          [
            {
              role: 'user',
              content: 'Compare credit card rewards for travel.',
            },
            { role: 'assistant', content: long },
          ],
          [
            {
              role: 'user',
              content: 'My degree was in Business Administration.',
              has_answer: true,
            },
            { role: 'assistant', content: long },
          ],
        ],
      }),
      reader,
      judge,
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'extracted',
        extractor,
        extractionAssistantCharacters: 300,
      },
    );
    expect(observation.status).toBe('judged');
    const sent = extractor.calls[1]?.messages.at(-1)?.content ?? '';
    expect(sent).toContain('USER: My degree was in Business Administration.');
    expect(sent).toContain('ASSISTANT: ' + 'x'.repeat(300) + ' […]');
    expect(sent.length).toBeLessThan(600);
  });

  it("shows the retrieved session's remembered facts, dated, to the reader", async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    await evaluateLongMemEvalAnswerInstance(instance(), reader, judge, {
      topK: 1,
      contextBytes: 4_096,
      formation: 'hybrid',
      extractor,
    });
    const prompt = reader.calls[0]?.messages.at(-1)?.content ?? '';
    expect(prompt).toMatch(/[Rr]emembered facts/);
    expect(prompt).toContain('degree(user, business_administration).');
    // the placeholder fact is plumbing, never shown
    expect(prompt).not.toContain('longmem_session');
  });

  it('engine recall: the memory system queries its own facts and the reader sees the rows', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    // extraction for each session, then the query the writer authors over the remembered facts
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
      'q(Degree) :- degree(user, Degree).',
    ]);
    const result = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        extractor,
        engineRecall: { llm: extractor },
      },
    );
    const prompt = reader.calls[0]?.messages.at(-1)?.content ?? '';
    expect(prompt).toContain('Memory engine result');
    expect(prompt).toContain('q(Degree) :- degree(user, Degree).');
    expect(prompt).toContain('Degree = business_administration');
    // the writer saw its training dialect and the store's predicates, not the plumbing
    const authoring = extractor.calls.at(-1)?.messages[0]?.content ?? '';
    expect(authoring).toContain('degree(A1, A2)');
    expect(authoring).toContain('Rule shape');
    expect(authoring).not.toContain('longmem_session');
    expect(result.engineRecall?.status).toBe('answered');
    expect(result.engineRecall?.rows).toBe(1);
  });

  it('reserved hybrid retrieval keeps top-k for raw sessions and adds matched facts as a dated block', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    // the noise session wins raw lexical retrieval on "degree"/"graduate"; only the
    // extracted fact points at the evidence session
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        haystack_sessions: [
          [
            {
              role: 'user',
              content:
                'Which degree should my nephew graduate with? Compare degree options.',
            },
            { role: 'assistant', content: 'Long generic degree comparison.' },
          ],
          [
            {
              role: 'user',
              content: 'My major was Business Administration.',
              has_answer: true,
            },
            {
              role: 'assistant',
              content: 'Long generic graduation explanation.',
            },
          ],
        ],
      }),
      reader,
      judge,
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        hybridRetrieval: 'reserved',
        extractor,
      },
    );
    expect(observation.retrievedSessionIds).toEqual(['noise', 'evidence']);
    const prompt = reader.calls[0]?.messages.at(-1)?.content ?? '';
    expect(prompt).toMatch(/[Rr]emembered facts/);
    expect(prompt).toContain('degree(user, business_administration).');
    expect(prompt).toMatch(
      /[Rr]emembered facts[\s\S]*- 2024-01-0\d[^\n]*degree\(user, business_administration\)\./,
    );
    expect(observation.retrieval?.recallAtK).toBe(1);
  });

  it('routes by question type: hybrid only for the listed types, raw (no extraction) otherwise', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', []);
    // single-session-user is not in the hybrid set: no extraction call at all
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        hybridQuestionTypes: new Set(['knowledge-update', 'multi-session']),
        extractor,
      },
    );
    expect(observation.status).toBe('judged');
    expect(extractor.calls).toHaveLength(0);
    expect(observation.extraction).toBeUndefined();
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
  });

  it('reserved facts honour a minimum score and are framed as supplementary', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const run = async (minimumScore: number) => {
      const r = new ScriptedCompletionClient('reader', [
        'Business Administration',
      ]);
      const j = new ScriptedCompletionClient('judge', ['yes']);
      const x = new ScriptedCompletionClient('dialect', [
        '% nothing',
        'degree(user, business_administration).',
      ]);
      await evaluateLongMemEvalAnswerInstance(instance(), r, j, {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        hybridRetrieval: 'reserved',
        reservedMinimumScore: minimumScore,
        extractor: x,
      });
      return r.calls[0]?.messages.at(-1)?.content ?? '';
    };
    void reader;
    void judge;
    void extractor;
    const shown = await run(1);
    expect(shown).toContain('degree(user, business_administration).');
    expect(shown).toMatch(/supplementary|may be unrelated/i);
    const hidden = await run(100_000);
    expect(hidden).not.toContain('degree(user, business_administration).');
  });

  it('entity retrieval reaches a session through a shared relation even when its text shares no word with the question', async () => {
    const run = async (entityRetrieval: boolean) => {
      const reader = new ScriptedCompletionClient('reader', ['Two kits']);
      const judge = new ScriptedCompletionClient('judge', ['yes']);
      // k1's fact carries a "kit" constant (a lexical seed); k2's fact shares only the
      // relation-and-subject; the noise session's text is full of the question's words
      const extractor = new ScriptedCompletionClient('dialect', [
        'finished(user, spitfire_kit).',
        'finished(user, revell_f15).',
        '% nothing',
      ]);
      const observation = await evaluateLongMemEvalAnswerInstance(
        instance({
          question_type: 'multi-session',
          question: 'How many kits have I completed?',
          answer: 'Two',
          haystack_session_ids: ['k1', 'k2', 'noise'],
          haystack_dates: [
            '2024/01/01 (Mon) 09:00',
            '2024/01/02 (Tue) 09:00',
            '2024/01/03 (Wed) 09:00',
          ],
          haystack_sessions: [
            [
              {
                role: 'user',
                content: 'The Spitfire kit is done at last.',
                has_answer: true,
              },
              { role: 'assistant', content: 'Nice.' },
            ],
            [
              {
                role: 'user',
                content: 'Glued the Revell F-15 canopy tonight, all done.',
                has_answer: true,
              },
              { role: 'assistant', content: 'Nice.' },
            ],
            [
              {
                role: 'user',
                content:
                  'How many kits are on sale? Kits completed by others look great.',
              },
              { role: 'assistant', content: 'Soon.' },
            ],
          ],
          answer_session_ids: ['k1', 'k2'],
        }),
        reader,
        judge,
        {
          topK: 3,
          multiSessionTopK: 3,
          contextBytes: 4_096,
          formation: 'hybrid',
          entityRetrieval,
          extractor,
        },
      );
      return observation.retrievedSessionIds;
    };
    expect(await run(false)).not.toContain('k2');
    expect(await run(true)).toEqual(expect.arrayContaining(['k1', 'k2']));
  });

  it('replays cached extractions instead of calling the extractor again', async () => {
    const { mkdtempSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const cacheDir = mkdtempSync(join(tmpdir(), 'lme-cache-'));
    const first = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const a = await evaluateLongMemEvalAnswerInstance(
      instance(),
      new ScriptedCompletionClient('reader', ['Business Administration']),
      new ScriptedCompletionClient('judge', ['yes']),
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        extractor: first,
        extractionCacheDir: cacheDir,
      },
    );
    expect(a.extraction?.calls).toBe(2);
    // second run: the scripted extractor has nothing left, yet the facts come back from cache
    const second = new ScriptedCompletionClient('dialect', []);
    const b = await evaluateLongMemEvalAnswerInstance(
      instance(),
      new ScriptedCompletionClient('reader', ['Business Administration']),
      new ScriptedCompletionClient('judge', ['yes']),
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        extractor: second,
        extractionCacheDir: cacheDir,
      },
    );
    expect(second.calls).toHaveLength(0);
    expect(b.extraction).toMatchObject({ calls: 0, facts: 1, cached: 2 });
    expect(b.retrievedSessionIds).toEqual(['evidence']);
  });

  it('keyed hybrid prepends extracted facts to the session key so the fact words retrieve the session', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        haystack_sessions: [
          [
            {
              role: 'user',
              content:
                'Which university should my nephew pick? Compare options please.',
            },
            {
              role: 'assistant',
              content: 'Long generic university comparison.',
            },
          ],
          [
            {
              role: 'user',
              content: 'I finished my studies in Business Administration.',
              has_answer: true,
            },
            {
              role: 'assistant',
              content: 'Long generic graduation explanation.',
            },
          ],
        ],
      }),
      reader,
      judge,
      {
        topK: 1,
        contextBytes: 4_096,
        formation: 'hybrid',
        hybridRetrieval: 'keyed',
        extractor,
      },
    );
    // neither raw text says "degree"; the evidence session's fact does, and the fact is
    // part of the session's key rather than a competing document, so no fact list is shown
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    const prompt = reader.calls[0]?.messages.at(-1)?.content ?? '';
    expect(prompt).toContain('Business Administration');
    expect(prompt).not.toContain('Remembered facts (stated in this session)');
    expect(observation.extraction?.facts).toBe(1);
  });

  it('turn-level retrieval scores user turns, aggregates to sessions and returns whole sessions', async () => {
    const reader = new ScriptedCompletionClient('reader', ['Two']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        question: 'How many marathons have I run?',
        haystack_session_ids: ['one-turn', 'two-turns'],
        haystack_dates: ['2024/01/01 (Mon) 09:00', '2024/01/02 (Tue) 09:00'],
        haystack_sessions: [
          [
            { role: 'user', content: 'I finished a marathon in Boston.' },
            { role: 'assistant', content: 'Congratulations.' },
            {
              role: 'user',
              content: 'Unrelated: what is a good pasta recipe?',
            },
            { role: 'assistant', content: 'Try carbonara.' },
          ],
          [
            {
              role: 'user',
              content: 'I ran a marathon in Berlin.',
              has_answer: true,
            },
            { role: 'assistant', content: 'Nice.' },
            {
              role: 'user',
              content: 'And another marathon in Tokyo last spring.',
              has_answer: true,
            },
            { role: 'assistant', content: 'Great.' },
          ],
        ],
        answer_session_ids: ['two-turns'],
      }),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096, formation: 'raw', retrievalUnit: 'turn' },
    );
    expect(observation.retrievedSessionIds).toEqual(['two-turns']);
    const prompt = reader.calls[0]?.messages.at(-1)?.content ?? '';
    // the whole session comes back, not just the matching turn
    expect(prompt).toContain('Berlin');
    expect(prompt).toContain('Tokyo');
  });

  it('time-aware retrieval prefers sessions inside the range a frontier model reads off the question', async () => {
    const run = async (withRange: boolean) => {
      const reader = new ScriptedCompletionClient('reader', ['Two weeks ago']);
      const judge = new ScriptedCompletionClient('judge', ['yes']);
      const ranger = new ScriptedCompletionClient('ranger', [
        '{"start":"2024-01-15","end":"2024-03-01"}',
      ]);
      const observation = await evaluateLongMemEvalAnswerInstance(
        instance({
          question_type: 'temporal-reasoning',
          question: 'How many weeks ago did I see the dentist?',
          question_date: '2024/03/01 (Fri) 09:00',
          haystack_session_ids: ['old', 'recent'],
          haystack_dates: ['2023/06/01 (Thu) 09:00', '2024/02/15 (Thu) 09:00'],
          haystack_sessions: [
            [
              {
                role: 'user',
                content:
                  'My dentist moved offices; the dentist is now downtown, and the dentist visit was fine.',
              },
              { role: 'assistant', content: 'Good to hear.' },
            ],
            [
              {
                role: 'user',
                content: 'Saw the dentist today for a filling.',
                has_answer: true,
              },
              { role: 'assistant', content: 'Hope it went well.' },
            ],
          ],
          answer_session_ids: ['recent'],
        }),
        reader,
        judge,
        {
          topK: 1,
          temporalTopK: 1,
          contextBytes: 4_096,
          formation: 'raw',
          ...(withRange ? { temporalRangeExtractor: ranger } : {}),
        },
      );
      return {
        ids: observation.retrievedSessionIds,
        calls: ranger.calls.length,
        range: observation.temporalRange,
      };
    };
    const plain = await run(false);
    expect(plain.ids).toEqual(['old']);
    const ranged = await run(true);
    expect(ranged.ids).toEqual(['recent']);
    expect(ranged.calls).toBe(1);
    expect(ranged.range).toMatchObject({
      start: '2024-01-15',
      end: '2024-03-01',
    });
  });

  it('time-aware retrieval leaves the ranking alone when the model reports no time cue', async () => {
    const reader = new ScriptedCompletionClient('reader', ['x']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const ranger = new ScriptedCompletionClient('ranger', ['{"none":true}']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'temporal-reasoning' }),
      reader,
      judge,
      {
        topK: 1,
        temporalTopK: 1,
        contextBytes: 4_096,
        formation: 'raw',
        temporalRangeExtractor: ranger,
      },
    );
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.temporalRange).toBeNull();
  });

  it('structured reading asks for dated notes first and judges only the final answer line', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Notes:\n1. 2024-01-02: degree in Business Administration\n\nAnswer: Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'multi-session' }),
      reader,
      judge,
      {
        topK: 1,
        multiSessionTopK: 1,
        contextBytes: 4_096,
        formation: 'raw',
        readingStrategy: 'notes',
      },
    );
    const system = reader.calls[0]?.messages[0]?.content ?? '';
    expect(system).toMatch(/list every relevant item[\s\S]*Answer:/i);
    expect(observation.hypothesis).toBe('Business Administration');
    expect(judge.calls[0]?.messages[0]?.content).toContain(
      'Model response: Business Administration',
    );
    expect(judge.calls[0]?.messages[0]?.content).not.toContain('Notes:');
  });

  it('structured reading applies only to the listed question types', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    await evaluateLongMemEvalAnswerInstance(instance(), reader, judge, {
      topK: 1,
      contextBytes: 4_096,
      formation: 'raw',
      readingStrategy: 'notes',
    });
    // single-session-user is not an aggregation type: plain reading
    expect(reader.calls[0]?.messages[0]?.content ?? '').not.toMatch(
      /list every relevant item/i,
    );
  });

  it('two-call reading enumerates from the history, then answers from the enumeration alone', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      '- 2024-01-02: degree in Business Administration',
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'multi-session' }),
      reader,
      judge,
      {
        topK: 1,
        multiSessionTopK: 1,
        contextBytes: 4_096,
        formation: 'raw',
        readingStrategy: 'two-call',
      },
    );
    expect(reader.calls).toHaveLength(2);
    const first = reader.calls[0]?.messages.at(-1)?.content ?? '';
    const second = reader.calls[1]?.messages.at(-1)?.content ?? '';
    expect(first).toContain('History chats:');
    expect(second).toContain('2024-01-02: degree in Business Administration');
    expect(second).not.toContain('History chats:');
    expect(observation.hypothesis).toBe('Business Administration');
    expect(observation.readerUsage?.totalTokens).toBe(24);
  });

  it('routes aggregation question types to a separate reader and leaves the rest with the default', async () => {
    const base = new ScriptedCompletionClient('luna', [
      'Business Administration',
    ]);
    const strong = new ScriptedCompletionClient('strong', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes', 'yes']);
    await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'multi-session' }),
      base,
      judge,
      {
        topK: 1,
        multiSessionTopK: 1,
        contextBytes: 4_096,
        formation: 'raw',
        aggregationReader: strong,
      },
    );
    expect(strong.calls).toHaveLength(1);
    expect(base.calls).toHaveLength(0);
    await evaluateLongMemEvalAnswerInstance(instance(), base, judge, {
      topK: 1,
      contextBytes: 4_096,
      formation: 'raw',
      aggregationReader: strong,
    });
    expect(strong.calls).toHaveLength(1);
    expect(base.calls).toHaveLength(1);
  });

  it('counts top-k in distinct sessions when a session yields several matching facts', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      // grounded in the noise session's words, and sharing the question's word "degree"
      'degree_interest(user, travel).\ndegree_plan(user, rewards).\ndegree_notes(user, credit).',
      'degree(user, business_administration).',
    ]);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      { topK: 2, contextBytes: 4_096, formation: 'extracted', extractor },
    );
    // three noise facts would fill top-2 on their own; de-duplicated by session, both sessions fit
    expect(observation.retrievedSessionIds).toEqual(
      expect.arrayContaining(['evidence', 'noise']),
    );
    expect(observation.retrievedSessionIds).toHaveLength(2);
  });

  it('hybrid formation stores extracted facts under their own operation id next to the raw session fact', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const extractor = new ScriptedCompletionClient('dialect', [
      '% nothing',
      'degree(user, business_administration).',
    ]);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096, formation: 'hybrid', extractor },
    );
    expect(observation.extraction).toMatchObject({
      calls: 2,
      facts: 1,
      errors: 0,
    });
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
  });

  it('hybrid formation keeps the raw session fact so sessions without extracted facts stay retrievable', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    // the extractor fails on one session and finds nothing in the other: retrieval must still work
    const extractor = new ScriptedCompletionClient('dialect', ['% nothing']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096, formation: 'hybrid', extractor },
    );
    expect(observation.status).toBe('judged');
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.extraction).toMatchObject({
      calls: 2,
      facts: 0,
      errors: 1,
    });
    expect(observation.extraction?.errorKinds).toEqual({
      'scripted completion exhausted': 1,
    });
  });

  it('keeps repeated dataset session IDs from colliding in the durable journal', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        haystack_session_ids: ['shared', 'shared'],
        answer_session_ids: ['shared'],
      }),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096 },
    );
    expect(observation.status).toBe('judged');
    expect(observation.error).toBeUndefined();
  });

  it('retains assistant turns for assistant-memory questions', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'graduation explanation',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'single-session-assistant' }),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096 },
    );
    expect(observation.contextRoles).toBe('all');
    expect(reader.calls[0]?.messages.at(-1)?.content).toContain(
      'Long generic graduation explanation.',
    );
  });

  it('uses semantic retrieval for low-confidence multi-session questions', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({ question_type: 'multi-session' }),
      reader,
      judge,
      {
        topK: 1,
        multiSessionTopK: 1,
        contextBytes: 4_096,
        semanticQuestionTypes: new Set(['multi-session']),
        prepareSemantic: true,
        embeddings: {
          model: 'embedding',
          async embed(inputs) {
            return {
              model: this.model,
              vectors: inputs.map((input, index) =>
                index === 0 || input.includes('Business Administration')
                  ? [1, 0]
                  : [0, 1],
              ),
              usage: {
                promptTokens: inputs.length,
                totalTokens: inputs.length,
                costUsd: 0.001,
              },
            };
          },
        },
      },
    );
    expect(observation).toMatchObject({
      status: 'judged',
      correct: true,
      retrievalRoute: 'semantic',
      embeddingModel: 'embedding',
      embeddingCalls: 1,
      semanticPreparationCalls: 1,
      retrievedSessionIds: ['evidence'],
    });
    expect(observation.semanticPreparationUsage?.costUsd).toBe(0.001);
    expect(observation.userTurnMs).toBeLessThanOrEqual(observation.totalMs);
  });

  it('keeps high-confidence multi-session matches local', async () => {
    const reader = new ScriptedCompletionClient('reader', [
      'Business Administration',
    ]);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance({
        question_type: 'multi-session',
        question: 'My university degree was Business Administration.',
        haystack_sessions: [
          [
            {
              role: 'user',
              content: 'Compare credit card rewards for travel.',
            },
          ],
          [
            {
              role: 'user',
              content: 'My university degree was Business Administration.',
              has_answer: true,
            },
          ],
        ],
      }),
      reader,
      judge,
      {
        topK: 1,
        multiSessionTopK: 1,
        contextBytes: 4_096,
        semanticQuestionTypes: new Set(['multi-session']),
        embeddings: {
          model: 'embedding',
          async embed() {
            throw new Error(
              'high-confidence local result must not call embeddings',
            );
          },
        },
      },
    );
    expect(observation).toMatchObject({
      status: 'judged',
      correct: true,
      retrievalRoute: 'local',
      embeddingCalls: 0,
    });
  });

  it('uses task-specific judge contracts and rejects ambiguous labels', () => {
    expect(buildLongMemEvalJudgePrompt(instance(), 'answer')).toContain(
      'partial',
    );
    expect(
      buildLongMemEvalJudgePrompt(
        instance({ question_type: 'single-session-preference' }),
        'answer',
      ),
    ).toContain('personalization rubric');
    expect(
      buildLongMemEvalJudgePrompt(
        instance({ question_type: 'knowledge-update' }),
        'answer',
      ),
    ).toContain('updated correct answer');
    expect(
      buildLongMemEvalJudgePrompt(
        instance({ question_type: 'temporal-reasoning' }),
        'answer',
      ),
    ).toContain('one-unit error');
    expect(
      buildLongMemEvalJudgePrompt(
        instance({ question_id: 'question-1_abs' }),
        'answer',
      ),
    ).toContain('unanswerable');
    expect(parseLongMemEvalJudgeLabel('Yes.')).toBe(true);
    expect(parseLongMemEvalJudgeLabel('no')).toBe(false);
    expect(() => parseLongMemEvalJudgeLabel('yes, probably')).toThrow(
      /yes or no only/i,
    );
  });

  it('preserves pinned run metadata and counts errors against overall accuracy', () => {
    const failed = {
      questionId: 'failed',
      questionType: 'single-session-user',
      abstention: false,
      status: 'error' as const,
      correct: null,
      retrievedSessionIds: [],
      contextSessionIds: [],
      contextRoles: 'user' as const,
      redactedRetrievedSessions: 0,
      retrieval: null,
      context: null,
      hypothesis: null,
      judgeResponse: null,
      readerUsage: null,
      judgeUsage: null,
      formationMs: 1,
      semanticPreparationMs: 0,
      retrievalMs: 0,
      readerMs: 0,
      judgeMs: 0,
      totalMs: 1,
      userTurnMs: 0,
      error: 'provider unavailable',
    };
    const run = longMemEvalAnswerRun([failed], 'reader', 'judge', {
      generatedAt: '2026-08-20T00:00:00.000Z',
    });
    expect(run).toMatchObject({
      schemaVersion: 'remembero.longmemeval-answer.v1',
      formation: 'durable-raw-session-facts',
      semanticPreparation: 'cold',
      topK: 4,
      multiSessionTopK: 5,
      temporalTopK: 5,
      summary: { questions: 1, errors: 1, correct: 0, accuracy: 0 },
      dataset: { commit: expect.any(String), sha256: expect.any(String) },
    });
  });
});

describe('computed notes in the answer context', () => {
  it('appends the deterministic block when enabled and nothing otherwise', () => {
    const instance = {
      question_id: 'q1',
      question_type: 'temporal-reasoning',
      question: "How many days passed between the 'Walk for Hunger' event and the 'Coastal Cleanup' event?",
      question_date: '2023/03/14 (Tue) 21:24',
      answer: '14',
      haystack_session_ids: ['s1', 's2'],
      haystack_dates: [],
      haystack_sessions: [],
      answer_session_ids: ['s1', 's2'],
    } as unknown as LongMemEvalInstance;
    const sources = [
      { opId: 's1', ts: '2023-02-22T10:00:00Z', text: 'USER: I did the Walk for Hunger 5K yesterday.' },
      { opId: 's2', ts: '2023-03-08T09:00:00Z', text: 'USER: The Coastal Cleanup yesterday was muddy.' },
    ];
    const withNotes = buildLongMemEvalAnswerContext(instance, sources, 8192, [], 'direct', undefined, true, true);
    expect(withNotes.messages.at(-1)?.content).toContain('Computed from the history');
    expect(withNotes.messages.at(-1)?.content).toMatch(/14 days/);
    const without = buildLongMemEvalAnswerContext(instance, sources, 8192, [], 'direct', undefined, true, false);
    expect(without.messages.at(-1)?.content).not.toContain('Computed from the history');
  });
});

describe('focused budget', () => {
  it('gives sessions that mention the question more of the context than sessions that do not', () => {
    const instance = {
      question_id: 'q2',
      question_type: 'multi-session',
      question: 'How many movie festivals have I attended?',
      question_date: '2023/05/30 (Tue) 20:53',
      answer: '4',
      haystack_session_ids: ['a', 'b'],
      haystack_dates: [],
      haystack_sessions: [],
      answer_session_ids: ['a'],
    } as unknown as LongMemEvalInstance;
    const filler = 'USER: I reorganised my pantry and labelled every jar today. '.repeat(200);
    const relevant = `USER: I went to the Austin Film Festival and the AFI film festival this spring. ${'The festival screenings were long. '.repeat(200)}`;
    const sources = [
      { opId: 'a', ts: '2023-05-20T10:00:00Z', text: relevant },
      { opId: 'b', ts: '2023-05-21T10:00:00Z', text: filler },
    ];
    const even = buildLongMemEvalAnswerContext(instance, sources, 6144, [], 'direct', undefined, false, false, false);
    const focused = buildLongMemEvalAnswerContext(instance, sources, 6144, [], 'direct', undefined, false, false, true);
    const section = (ctx: { messages: Array<{ content: string }> }, id: string) => {
      const text = ctx.messages.at(-1)!.content;
      const start = text.indexOf(id === 'a' ? 'Retrieved session 1' : 'Retrieved session 2');
      const end = text.indexOf('### Retrieved session', start + 5);
      return text.slice(start, end === -1 ? undefined : end);
    };
    expect(section(focused, 'a').length).toBeGreaterThan(section(even, 'a').length);
    expect(section(focused, 'b').length).toBeLessThan(section(even, 'b').length);
  });
});

describe('structured evidence in the answer context', () => {
  it('places dated facts before the chats when enabled', () => {
    const instance = {
      question_id: 'q3',
      question_type: 'multi-session',
      question: 'How many online courses have I completed in total?',
      question_date: '2023/05/30 (Tue) 16:30',
      answer: '5',
      haystack_session_ids: ['a', 'b'],
      haystack_dates: [],
      haystack_sessions: [],
      answer_session_ids: ['a', 'b'],
    } as unknown as LongMemEvalInstance;
    const sources = [
      { opId: 'a', ts: '2023-05-23T09:00:00Z', text: 'USER: Just wrapped up my third Coursera course on deep learning.', facts: ['completed 3 courses on Coursera'] },
      { opId: 'b', ts: '2023-05-30T09:00:00Z', text: 'USER: I finished two courses on edX this week.', facts: ['completed 2 courses on edX'] },
    ];
    const ctx = buildLongMemEvalAnswerContext(instance, sources, 8192, [], 'direct', undefined, false, false, false, true);
    const text = ctx.messages.at(-1)!.content;
    expect(text.indexOf('Dated facts')).toBeGreaterThanOrEqual(0);
    expect(text.indexOf('Dated facts')).toBeLessThan(text.indexOf('History chats'));
    expect(text).toMatch(/2023-05-23.*completed 3 courses on Coursera/);
  });
});

class ForbiddenCompletionClient implements LongMemEvalCompletionClient {
  constructor(readonly model: string) {}

  async completeWithUsage(): Promise<LlmCompletion> {
    throw new Error(`${this.model} must not be called in retrieval-only mode`);
  }
}

/** An evidence session whose answer turn sits beyond a 4 KB window of question-word text. */
function cutEvidenceInstance(): LongMemEvalInstance {
  return instance({
    question_id: 'cut-1',
    question_type: 'multi-session',
    question: 'Which bicycle model did I choose?',
    answer: 'Trek Domane',
    haystack_sessions: [
      [
        { role: 'user', content: 'Compare credit card rewards for travel.' },
        { role: 'assistant', content: 'Long generic reward-card explanation.' },
      ],
      [
        {
          role: 'user',
          content: 'I am shopping for a bicycle model and need to choose soon.',
        },
        { role: 'assistant', content: 'Happy to help.' },
        { role: 'user', content: 'lorem ipsum dolor sit amet '.repeat(260) },
        {
          role: 'user',
          content: 'In the end I went with the Trek Domane.',
          has_answer: true,
        },
      ],
    ],
  });
}

describe('evidence coverage and retrieval-only mode', () => {
  it('counts an evidence session in context whose answer turn the byte budget cut', async () => {
    const reader = new ScriptedCompletionClient('reader', ['I do not know']);
    const judge = new ScriptedCompletionClient('judge', ['no']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      cutEvidenceInstance(),
      reader,
      judge,
      { multiSessionTopK: 1, contextBytes: 4_096 },
    );
    expect(observation.status).toBe('judged');
    expect(observation.contextSessionIds).toEqual(['evidence']);
    expect(reader.calls[0]?.messages.at(-1)?.content).not.toContain(
      'Trek Domane',
    );
    expect(observation.evidenceCoverage).toEqual({
      sessionsInContext: 1,
      sessionsTotal: 1,
      turnsInContext: 0,
      turnsTotal: 1,
    });
  });

  it('counts the answer turn once the budget leaves room for it', async () => {
    const observation = await evaluateLongMemEvalAnswerInstance(
      cutEvidenceInstance(),
      new ForbiddenCompletionClient('reader'),
      new ForbiddenCompletionClient('judge'),
      { multiSessionTopK: 1, contextBytes: 16_384, retrievalOnly: true },
    );
    expect(observation.evidenceCoverage).toEqual({
      sessionsInContext: 1,
      sessionsTotal: 1,
      turnsInContext: 1,
      turnsTotal: 1,
    });
  });

  it('retrieval-only mode builds the context and never calls the reader or the judge', async () => {
    const observation = await evaluateLongMemEvalAnswerInstance(
      cutEvidenceInstance(),
      new ForbiddenCompletionClient('reader'),
      new ForbiddenCompletionClient('judge'),
      { multiSessionTopK: 1, contextBytes: 4_096, retrievalOnly: true },
    );
    expect(observation).toMatchObject({
      status: 'retrieval-only',
      correct: null,
      hypothesis: null,
      judgeResponse: null,
      readerUsage: null,
      judgeUsage: null,
      readerMs: 0,
      judgeMs: 0,
      retrievedSessionIds: ['evidence'],
      contextSessionIds: ['evidence'],
      evidenceCoverage: {
        sessionsInContext: 1,
        sessionsTotal: 1,
        turnsInContext: 0,
        turnsTotal: 1,
      },
    });
    expect(observation.error).toBeUndefined();
    expect(observation.context?.recallAtK).toBe(1);
    const run = longMemEvalAnswerRun([observation], 'none', 'none', {
      generatedAt: '2026-09-17T00:00:00.000Z',
    });
    // a retrieval-only observation is not an error
    expect(run.summary).toMatchObject({ questions: 1, errors: 0 });
  });

  it('retrieval-only mode still runs the temporal-range model, which is part of retrieval', async () => {
    const ranger = new ScriptedCompletionClient('ranger', ['{"none":true}']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      { ...cutEvidenceInstance(), question_type: 'temporal-reasoning' },
      new ForbiddenCompletionClient('reader'),
      new ForbiddenCompletionClient('judge'),
      {
        temporalTopK: 1,
        contextBytes: 4_096,
        retrievalOnly: true,
        temporalRangeExtractor: ranger,
      },
    );
    expect(observation.status).toBe('retrieval-only');
    expect(ranger.calls).toHaveLength(1);
    expect(observation.temporalRange).toBeNull();
  });

  it('summarizes evidence completeness overall and per type, leaving abstention out', () => {
    const base = {
      questionType: 'multi-session',
      abstention: false,
      status: 'retrieval-only' as const,
      correct: null,
      retrievedSessionIds: [],
      contextSessionIds: [],
      contextRoles: 'user' as const,
      redactedRetrievedSessions: 0,
      retrievalRoute: 'local' as const,
      embeddingModel: null,
      embeddingCalls: 0,
      embeddingUsage: null,
      semanticPreparationCalls: 0,
      semanticPreparationUsage: null,
      retrieval: null,
      context: null,
      hypothesis: null,
      judgeResponse: null,
      readerUsage: null,
      judgeUsage: null,
      formationMs: 1,
      semanticPreparationMs: 0,
      retrievalMs: 0,
      readerMs: 0,
      judgeMs: 0,
      totalMs: 1,
      userTurnMs: 0,
    };
    const complete = {
      ...base,
      questionId: 'complete',
      evidenceCoverage: {
        sessionsInContext: 2,
        sessionsTotal: 2,
        turnsInContext: 2,
        turnsTotal: 2,
      },
    };
    const cut = {
      ...base,
      questionId: 'cut',
      questionType: 'temporal-reasoning',
      evidenceCoverage: {
        sessionsInContext: 1,
        sessionsTotal: 1,
        turnsInContext: 1,
        turnsTotal: 4,
      },
    };
    const abstention = {
      ...base,
      questionId: 'missing_abs',
      abstention: true,
      evidenceCoverage: {
        sessionsInContext: 0,
        sessionsTotal: 1,
        turnsInContext: 0,
        turnsTotal: 1,
      },
    };
    const summary = summarizeLongMemEvalAnswers([complete, cut, abstention]);
    expect(summary.evidenceSessionsCompleteRate).toBe(1);
    expect(summary.evidenceTurnsCompleteRate).toBe(0.5);
    expect(summary.meanEvidenceTurnCoverage).toBeCloseTo((1 + 0.25) / 2);
    expect(summary.errors).toBe(0);
    const run = longMemEvalAnswerRun(
      [complete, cut, abstention],
      'none',
      'none',
      { generatedAt: '2026-09-17T00:00:00.000Z' },
    );
    expect(run.byQuestionType['multi-session']).toMatchObject({
      evidenceSessionsCompleteRate: 1,
      evidenceTurnsCompleteRate: 1,
      meanEvidenceTurnCoverage: 1,
    });
    expect(run.byQuestionType['temporal-reasoning']).toMatchObject({
      evidenceSessionsCompleteRate: 1,
      evidenceTurnsCompleteRate: 0,
      meanEvidenceTurnCoverage: 0.25,
    });
    // stored runs from before the field existed summarize without it
    const legacy = summarizeLongMemEvalAnswers([
      { ...complete, evidenceCoverage: undefined },
    ] as never);
    expect(legacy.evidenceSessionsCompleteRate).toBe(0);
  });
});

describe('turn-level retrieval routed by question type', () => {
  // session unit: the short, dense session wins; turn unit: the session with many
  // matching turns wins
  const fixture = (questionType: string) =>
    instance({
      question_type: questionType,
      question: 'Which marathon did I run?',
      haystack_session_ids: ['dense', 'spread'],
      haystack_dates: ['2024/01/01 (Mon) 09:00', '2024/01/02 (Tue) 09:00'],
      haystack_sessions: [
        [{ role: 'user', content: 'marathon marathon marathon marathon run' }],
        [
          {
            role: 'user',
            content:
              'I ran the Berlin marathon after weeks of long training runs, early mornings, careful nutrition planning and lots of stretching.',
            has_answer: true,
          },
          {
            role: 'assistant',
            content:
              'A marathon in Berlin is fast and flat; many runners set personal records there because of the weather and the course.',
          },
          {
            role: 'user',
            content:
              'The marathon expo was crowded, the shuttle buses were late, and the hotel breakfast was cold but the city was lovely.',
          },
          {
            role: 'assistant',
            content:
              'Marathon weekends are busy everywhere; booking accommodation early and arriving a day ahead usually helps a lot.',
          },
        ],
      ],
      answer_session_ids: ['spread'],
    });

  const run = async (
    questionType: string,
    options: {
      retrievalUnit?: 'session' | 'turn';
      turnUnitQuestionTypes?: ReadonlySet<string>;
    },
  ) => {
    const reader = new ScriptedCompletionClient('reader', ['Berlin']);
    const judge = new ScriptedCompletionClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      fixture(questionType),
      reader,
      judge,
      { topK: 1, contextBytes: 4_096, formation: 'raw', ...options },
    );
    return {
      retrieved: observation.retrievedSessionIds,
      prompt: reader.calls[0]?.messages.at(-1)?.content ?? '',
    };
  };

  it('the fixture separates the two units', async () => {
    const session = await run('multi-session', { retrievalUnit: 'session' });
    const turn = await run('multi-session', { retrievalUnit: 'turn' });
    expect(session.retrieved).not.toEqual(turn.retrieved);
  });

  it('a type outside the set retrieves as the session unit; a type inside as the turn unit', async () => {
    const only = new Set(['multi-session']);
    for (const [questionType, plainUnit] of [
      ['temporal-reasoning', 'session'],
      ['multi-session', 'turn'],
    ] as const) {
      const routed = await run(questionType, {
        retrievalUnit: 'turn',
        turnUnitQuestionTypes: only,
      });
      const plain = await run(questionType, { retrievalUnit: plainUnit });
      expect(routed).toEqual(plain);
    }
  });

  it('the runner parses the flag and refuses it without --retrieval-unit turn', () => {
    const args = parseArgs([
      '--retrieval-unit',
      'turn',
      '--turn-unit-question-types',
      'multi-session,knowledge-update',
    ]);
    expect([...(args.turnUnitQuestionTypes ?? [])]).toEqual([
      'multi-session',
      'knowledge-update',
    ]);
    expect(parseArgs([]).turnUnitQuestionTypes).toBeUndefined();
    expect(() =>
      parseArgs(['--turn-unit-question-types', 'multi-session']),
    ).toThrow(/--turn-unit-question-types needs --retrieval-unit turn/);
    expect(() =>
      parseArgs([
        '--retrieval-unit',
        'turn',
        '--turn-unit-question-types',
        'multi-sesion',
      ]),
    ).toThrow(/unknown question type "multi-sesion"/);
  });
});
