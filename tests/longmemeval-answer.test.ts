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
            { role: 'assistant', content: 'Long generic university comparison.' },
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
