import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  DEFAULT_QUESTION_KIND_THRESHOLD,
  QUESTION_KIND_FLAGS,
  questionKindFromLabel,
  questionKindFromText,
  type QuestionKind,
} from '../src/knowledge/question-kind.js';
import {
  QUESTION_KIND_QUESTIONS,
  typesafeQuestionKind,
} from '../src/evals/typesafe-question-kind.js';
import {
  formatQuestionKindAgreement,
  questionKindAgreement,
} from '../src/evals/question-kind-agreement.js';
import { typesafeNouls, type TypesafeNouls } from '../src/evals/typesafe-rerank.js';
import {
  evaluateLongMemEvalAnswerInstance,
  summarizeLongMemEvalAnswers,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { parseArgs } from '../src/evals/run-longmemeval-answer.js';

describe('questionKindFromText', () => {
  const only = (question: string): Array<keyof QuestionKind> => {
    const kind = questionKindFromText(question);
    return QUESTION_KIND_FLAGS.filter((flag) => kind[flag]);
  };

  it('reads a count over several statements as aggregation', () => {
    expect(only('How many different museums did I visit?')).toEqual(['aggregation']);
    expect(only('How much did I spend on bike parts in total?')).toEqual(['aggregation']);
    expect(only('What percentage of packed shoes did I wear?')).toEqual(['aggregation']);
    expect(only('Which grocery store did I spend the most money at?')).toEqual([
      'aggregation',
    ]);
    // counting time units is arithmetic on two dates, not a count over a set
    expect(questionKindFromText('How many days passed between the two trips?')).toMatchObject(
      { aggregation: false, temporal: true },
    );
    // unless the question adds the durations up
    expect(
      questionKindFromText('How many hours in total did I spend driving to the three cities?'),
    ).toMatchObject({ aggregation: true });
    expect(questionKindFromText('What degree did I graduate with?').aggregation).toBe(false);
  });

  it('reads time from the question with the existing detector', () => {
    expect(questionKindFromText('When did I book the Airbnb?').temporal).toBe(true);
    expect(
      questionKindFromText('What is the order of the three trips, from earliest to latest?')
        .temporal,
    ).toBe(true);
    expect(questionKindFromText('What breed is my dog?').temporal).toBe(false);
  });

  it('reads a value that has moved on as an update', () => {
    expect(questionKindFromText('How many bikes do I currently own?').update).toBe(true);
    expect(questionKindFromText('How often do I see my therapist?').update).toBe(true);
    expect(questionKindFromText('How long have I been using my Fitbit Charge 3?').update).toBe(
      true,
    );
    expect(questionKindFromText('How many issues have I finished reading?').update).toBe(true);
    expect(questionKindFromText('Where did Rachel move to?').update).toBe(true);
    expect(questionKindFromText('What color did I repaint my bedroom walls?').update).toBe(
      false,
    );
  });

  it('reads a request for advice as a preference', () => {
    expect(only('Can you recommend a show or movie for me to watch tonight?')).toEqual([
      'preference',
    ]);
    expect(questionKindFromText('Any tips for keeping my kitchen clean?').preference).toBe(
      true,
    );
    expect(
      questionKindFromText("I'm trying to decide whether to buy a NAS device. What do you think?")
        .preference,
    ).toBe(true);
    expect(questionKindFromText('Where did I buy my new tennis racket?').preference).toBe(
      false,
    );
  });

  it('separates what the assistant said from what the user said', () => {
    expect(only('Can you remind me what the 7th job in the list you provided was?')).toEqual([
      'assistantRecall',
    ]);
    expect(
      questionKindFromText('In our previous conversation about Netflix, which show did you name?')
        .assistantRecall,
    ).toBe(true);
    // "I mentioned" is the user, not the assistant
    expect(
      questionKindFromText('I mentioned a restaurant in Rome. Where is it?').assistantRecall,
    ).toBe(false);
    // a question about the assistant's own reply is not also a count or an update
    expect(only('Can you remind me how many mummies you said the party will face?')).toEqual([
      'assistantRecall',
    ]);
  });

  it('is a drop-in for the label routing it replaces', () => {
    expect(questionKindFromLabel('multi-session')).toEqual({
      aggregation: true,
      temporal: false,
      update: false,
      preference: false,
      assistantRecall: false,
    });
    expect(questionKindFromLabel('single-session-user')).toEqual({
      aggregation: false,
      temporal: false,
      update: false,
      preference: false,
      assistantRecall: false,
    });
  });
});

describe('questionKindAgreement', () => {
  const instances = [
    { question_type: 'multi-session', question: 'How many museums did I visit?' },
    { question_type: 'multi-session', question: 'What degree did I graduate with?' },
    { question_type: 'single-session-user', question: 'How many shirts did I pack?' },
    { question_type: 'temporal-reasoning', question: 'When did I book the Airbnb?' },
  ];

  it('counts precision, recall and the confusion per flag', () => {
    const report = questionKindAgreement(
      instances,
      instances.map(({ question }) => questionKindFromText(question)),
    );
    expect(report.total).toBe(4);
    expect(report.flags.aggregation).toMatchObject({
      label: 'multi-session',
      truePositives: 1,
      falsePositives: 1,
      falseNegatives: 1,
      trueNegatives: 1,
      precision: 0.5,
      recall: 0.5,
      misses: ['What degree did I graduate with?'],
      falseAlarms: [
        { questionType: 'single-session-user', question: 'How many shirts did I pack?' },
      ],
    });
    expect(report.flags.temporal).toMatchObject({ truePositives: 1, falsePositives: 0 });
    // the notes reading fires for any of aggregation, temporal or update
    expect(report.notesReading).toMatchObject({
      truePositives: 2,
      falsePositives: 1,
      falseNegatives: 1,
    });
    expect(formatQuestionKindAgreement(report, 'text', false)).toContain(
      '| aggregation | multi-session |',
    );
  });

  it('needs one kind per instance', () => {
    expect(() => questionKindAgreement(instances, [])).toThrow(/one classified kind/);
  });

  const dataset = '.cache/longmemeval/longmemeval_s_cleaned.json';
  it.skipIf(!existsSync(dataset))('holds its agreement over the 500', () => {
    const rows = JSON.parse(readFileSync(dataset, 'utf8')) as Array<{
      question: string;
      question_type: string;
    }>;
    const report = questionKindAgreement(
      rows,
      rows.map(({ question }) => questionKindFromText(question)),
    );
    expect(report.flags.preference!.precision).toBeGreaterThanOrEqual(0.95);
    expect(report.flags.preference!.recall).toBeGreaterThanOrEqual(0.95);
    expect(report.flags.assistantRecall!.precision).toBeGreaterThanOrEqual(0.95);
    expect(report.flags.assistantRecall!.recall).toBeGreaterThanOrEqual(0.95);
    expect(report.notesReading.precision).toBeGreaterThanOrEqual(0.88);
    expect(report.notesReading.recall).toBeGreaterThanOrEqual(0.85);
  });
});

/** A stub client: the same probability for every flag named, 0 for the rest. */
const nouls = (set: ReadonlyArray<keyof QuestionKind>, probability = 0.9): TypesafeNouls =>
  async (_state, questions) => ({
    nouls: Object.fromEntries(
      Object.keys(questions).map((name) => [
        name,
        set.includes(name as keyof QuestionKind) ? probability : 0.02,
      ]),
    ),
    inputTokens: 40,
    cached: false,
  });

describe('typesafeQuestionKind', () => {
  it('carries the five nouls in one request and thresholds them at 0.5', async () => {
    const states: unknown[] = [];
    const seen: Array<Record<string, unknown>> = [];
    const client: TypesafeNouls = async (state, questions) => {
      states.push(state);
      seen.push(questions);
      return {
        nouls: {
          aggregation: 0.51,
          temporal: 0.49,
          update: 0.5,
          preference: 0.1,
          assistantRecall: 0.99,
        },
        inputTokens: 33,
        cached: false,
      };
    };
    const result = await typesafeQuestionKind('How many bikes do I own now?', client);
    expect(states).toEqual([{ question: 'How many bikes do I own now?' }]);
    expect(Object.keys(seen[0]!)).toEqual([
      'aggregation',
      'temporal',
      'update',
      'preference',
      'assistantRecall',
    ]);
    expect(QUESTION_KIND_QUESTIONS.assistantRecall.instructions).toBe(
      'The question asks what the assistant said, recommended or explained in the past, not what the user said.',
    );
    expect(result.kind).toEqual({
      aggregation: true,
      temporal: false,
      update: true,
      preference: false,
      assistantRecall: true,
    });
    expect(result).toMatchObject({ inputTokens: 33, cached: false });
    expect(DEFAULT_QUESTION_KIND_THRESHOLD).toBe(0.5);
  });

  it('takes the threshold from its options', async () => {
    const client = nouls(['temporal'], 0.6);
    expect((await typesafeQuestionKind('q', client, { threshold: 0.7 })).kind.temporal).toBe(
      false,
    );
    expect((await typesafeQuestionKind('q', client, { threshold: 0.55 })).kind.temporal).toBe(
      true,
    );
    await expect(typesafeQuestionKind('q', client, { threshold: 2 })).rejects.toThrow(
      /threshold must be from 0 to 1/,
    );
  });

  it('never sends a question that fails the external-LLM safety check', async () => {
    let called = false;
    const client: TypesafeNouls = async () => {
      called = true;
      throw new Error('unreachable');
    };
    await expect(
      typesafeQuestionKind(
        'Which key did I use, sk-abcdefghijklmnopqrstuvwxyz0123456789?',
        client,
      ),
    ).rejects.toThrow();
    expect(called).toBe(false);
  });

  it('goes through the shared client, its retries and its disk cache', async () => {
    const calls: string[] = [];
    const fetchFn = (async (url: string | URL) => {
      calls.push(String(url));
      return new Response(
        JSON.stringify({
          answers: Object.fromEntries(
            QUESTION_KIND_FLAGS.map((flag) => [
              flag,
              { type: 'noul', noul: flag === 'temporal' ? 0.8 : 0.1 },
            ]),
          ),
          usage: { input_tokens: 21 },
        }),
        { status: 200 },
      );
    }) as typeof fetch;
    const client: TypesafeNouls = (state, questions) =>
      typesafeNouls(state, questions, {
        apiKey: 'qk-test-key-not-real',
        fetchFn,
        cacheDir: null,
      });
    const result = await typesafeQuestionKind('When did I book it?', client);
    expect(calls).toHaveLength(1);
    expect(result.kind.temporal).toBe(true);
    expect(result.inputTokens).toBe(21);
  });
});

class Recorder implements LongMemEvalCompletionClient {
  readonly prompts: ChatMessage[][] = [];
  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    this.prompts.push(messages);
    const content = this.outputs.shift() ?? 'Answer: Berlin';
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

const QUESTION = 'Which marathon did I run?';

function fixture(questionType = 'single-session-user'): LongMemEvalInstance {
  const dates = [
    '2024/01/01 (Mon) 09:00',
    '2024/01/02 (Tue) 09:00',
    '2024/01/03 (Wed) 09:00',
    '2024/01/04 (Thu) 09:00',
  ];
  return {
    question_id: 'qk-1',
    question_type: questionType,
    question: QUESTION,
    answer: 'Berlin',
    question_date: '2024/01/10 (Wed) 09:00',
    haystack_session_ids: ['s1', 's2', 's3', 's4'],
    haystack_dates: dates,
    haystack_sessions: [
      [
        { role: 'user', content: 'I ran the Berlin marathon.', has_answer: true },
        { role: 'assistant', content: 'Nice, the Berlin marathon is fast.' },
      ],
      [{ role: 'user', content: 'The marathon training plan is going well.' }],
      [{ role: 'user', content: 'My marathon shoes need replacing.' }],
      [{ role: 'user', content: 'I signed up for another marathon.' }],
    ],
    answer_session_ids: ['s1'],
  };
}

const forbidden = (role: string): LongMemEvalCompletionClient => ({
  model: role,
  completeWithUsage: async () => {
    throw new Error(`${role} must not be called`);
  },
});

describe('the question kind drives the five answering decisions', () => {
  const depths = { topK: 1, multiSessionTopK: 3, temporalTopK: 2, contextBytes: 4_096 };

  it('routes retrieval depth by the aggregation and temporal flags', async () => {
    const depthFor = async (set: ReadonlyArray<keyof QuestionKind>) =>
      (
        await evaluateLongMemEvalAnswerInstance(
          fixture(),
          forbidden('reader'),
          forbidden('judge'),
          {
            ...depths,
            retrievalOnly: true,
            classify: 'typesafe',
            classifyNouls: nouls(set),
          },
        )
      ).retrievedSessionIds.length;
    expect(await depthFor([])).toBe(1);
    expect(await depthFor(['aggregation'])).toBe(3);
    expect(await depthFor(['temporal'])).toBe(2);
    // a preference or a recall question is not deeper
    expect(await depthFor(['preference'])).toBe(1);
    expect(await depthFor(['assistantRecall'])).toBe(1);
  });

  it('shows assistant turns to the reader only for the assistantRecall flag', async () => {
    const rolesFor = async (set: ReadonlyArray<keyof QuestionKind>) =>
      (
        await evaluateLongMemEvalAnswerInstance(
          fixture(),
          forbidden('reader'),
          forbidden('judge'),
          {
            ...depths,
            retrievalOnly: true,
            classify: 'typesafe',
            classifyNouls: nouls(set),
          },
        )
      ).contextRoles;
    expect(await rolesFor(['assistantRecall'])).toBe('all');
    expect(await rolesFor(['aggregation', 'temporal', 'update', 'preference'])).toBe('user');
  });

  it('runs the time-range model only for the temporal flag', async () => {
    const rangeCalls = async (set: ReadonlyArray<keyof QuestionKind>) => {
      const range = new Recorder('range', ['none', 'none']);
      const observation = await evaluateLongMemEvalAnswerInstance(
        fixture(),
        forbidden('reader'),
        forbidden('judge'),
        {
          ...depths,
          retrievalOnly: true,
          temporalRangeExtractor: range,
          classify: 'typesafe',
          classifyNouls: nouls(set),
        },
      );
      return { calls: range.prompts.length, range: observation.temporalRange };
    };
    expect(await rangeCalls(['temporal'])).toMatchObject({ calls: 1 });
    expect(await rangeCalls(['aggregation', 'update'])).toEqual({
      calls: 0,
      range: undefined,
    });
  });

  it('reads with notes for aggregation, temporal or update, and nothing else', async () => {
    const systemFor = async (set: ReadonlyArray<keyof QuestionKind>) => {
      const reader = new Recorder('reader', ['Notes:\n- a\nAnswer: Berlin']);
      await evaluateLongMemEvalAnswerInstance(
        fixture(),
        reader,
        new Recorder('judge', ['yes']),
        {
          ...depths,
          readingStrategy: 'notes',
          classify: 'typesafe',
          classifyNouls: nouls(set),
        },
      );
      return reader.prompts[0]![0]!.content;
    };
    for (const flag of ['aggregation', 'temporal', 'update'] as const) {
      expect(await systemFor([flag])).toContain('under "Notes:"');
    }
    expect(await systemFor([])).toContain('Answer only from the supplied history.');
    expect(await systemFor([])).not.toContain('under "Notes:"');
    expect(await systemFor(['assistantRecall'])).not.toContain('under "Notes:"');
  });

  it('personalizes the system prompt only for the preference flag', async () => {
    const systemFor = async (set: ReadonlyArray<keyof QuestionKind>) => {
      const reader = new Recorder('reader', ['Berlin']);
      await evaluateLongMemEvalAnswerInstance(
        fixture(),
        reader,
        new Recorder('judge', ['yes']),
        { ...depths, classify: 'typesafe', classifyNouls: nouls(set) },
      );
      return reader.prompts[0]![0]!.content;
    };
    expect(await systemFor(['preference'])).toContain('personalize the answer');
    expect(await systemFor(['aggregation'])).not.toContain('personalize the answer');
  });

  it('records the flags, their source and the classification tokens', async () => {
    const observation = await evaluateLongMemEvalAnswerInstance(
      fixture('multi-session'),
      forbidden('reader'),
      forbidden('judge'),
      {
        ...depths,
        retrievalOnly: true,
        classify: 'typesafe',
        classifyNouls: nouls(['temporal', 'update']),
      },
    );
    expect(observation.questionKind).toEqual({
      aggregation: false,
      temporal: true,
      update: true,
      preference: false,
      assistantRecall: false,
      source: 'typesafe',
      inputTokens: 40,
      cached: false,
    });
    expect(summarizeLongMemEvalAnswers([observation]).classifyUsage).toEqual({
      questions: 1,
      calls: 1,
      cached: 0,
      fallbacks: 0,
      inputTokens: 40,
      costUsd: (40 * 0.042) / 1_000_000,
    });
  });

  it('falls back to the text rules when the request fails, and records that', async () => {
    const failing: TypesafeNouls = async () => {
      throw new Error('TypeSafe request failed with status 500');
    };
    const observation = await evaluateLongMemEvalAnswerInstance(
      { ...fixture(), question: 'How many marathons did I run in total?' },
      forbidden('reader'),
      forbidden('judge'),
      {
        ...depths,
        retrievalOnly: true,
        classify: 'typesafe',
        classifyNouls: failing,
      },
    );
    expect(observation.questionKind).toMatchObject({
      source: 'typesafe-fallback',
      aggregation: true,
    });
    // the text rules decided, so the depth is the aggregation depth
    expect(observation.retrievedSessionIds).toHaveLength(3);
    expect(summarizeLongMemEvalAnswers([observation]).classifyUsage).toMatchObject({
      questions: 1,
      calls: 0,
      fallbacks: 1,
      inputTokens: 0,
    });
    // a missing client is the caller's mistake, not a classification failure
    await expect(
      evaluateLongMemEvalAnswerInstance(fixture(), forbidden('reader'), forbidden('judge'), {
        ...depths,
        retrievalOnly: true,
        classify: 'typesafe',
      }),
    ).rejects.toThrow(/needs a TypeSafe client/);
  });

  it("classify 'text' reads the kind off the question and implies the turn-unit rule", async () => {
    const observation = await evaluateLongMemEvalAnswerInstance(
      { ...fixture('single-session-assistant'), question: 'When did I run the marathon?' },
      forbidden('reader'),
      forbidden('judge'),
      { ...depths, retrievalOnly: true, retrievalUnit: 'turn', classify: 'text' },
    );
    expect(observation.questionKind).toMatchObject({ source: 'text', temporal: true });
    // the label would have shown the assistant's turns; the text does not
    expect(observation.contextRoles).toBe('user');
    expect(observation.retrievedSessionIds).toHaveLength(2);
  });
});

describe("classify 'label'", () => {
  // timings differ run to run; everything else must match byte for byte
  const strip = (observation: unknown) =>
    JSON.stringify(observation, (key, value) => (/Ms$/.test(key) ? 0 : value));

  it('leaves a run byte-identical to one that never heard of --classify', async () => {
    for (const questionType of [
      'single-session-user',
      'single-session-assistant',
      'single-session-preference',
      'multi-session',
      'knowledge-update',
      'temporal-reasoning',
    ]) {
      const run = async (withLabel: boolean) => {
        const reader = new Recorder('reader', ['Notes:\n- a\nAnswer: Berlin']);
        const observation = await evaluateLongMemEvalAnswerInstance(
          fixture(questionType),
          reader,
          new Recorder('judge', ['yes']),
          {
            topK: 1,
            multiSessionTopK: 3,
            temporalTopK: 2,
            contextBytes: 4_096,
            retrievalUnit: 'turn',
            readingStrategy: 'notes',
            ...(withLabel ? { classify: 'label' as const } : {}),
          },
        );
        return { observation: strip(observation), prompts: JSON.stringify(reader.prompts) };
      };
      const withLabel = await run(true);
      const without = await run(false);
      expect(withLabel.observation).toBe(without.observation);
      expect(withLabel.prompts).toBe(without.prompts);
      // and nothing extra is recorded on the label route
      expect(JSON.parse(withLabel.observation).questionKind).toBeUndefined();
    }
  });

  it('keeps --turn-unit-unless-temporal working on its own', async () => {
    const observation = await evaluateLongMemEvalAnswerInstance(
      { ...fixture('multi-session'), question: 'When did I run the marathon?' },
      forbidden('reader'),
      forbidden('judge'),
      {
        topK: 1,
        multiSessionTopK: 3,
        contextBytes: 4_096,
        retrievalOnly: true,
        retrievalUnit: 'turn',
        turnUnitRule: 'unless-temporal',
      },
    );
    // the label still sets the depth; only the unit came from the text
    expect(observation.retrievedSessionIds).toHaveLength(3);
    expect(observation.questionKind).toBeUndefined();
  });
});

describe('the runner flags', () => {
  it('defaults to the label and parses the classification knobs', () => {
    const plain = parseArgs([]);
    expect(plain.classify).toBe('label');
    expect(plain.classifyModel).toBe('jev-1.13.0');
    expect(plain.classifyKeyEnv).toBe('TYPESAFE_AI_API_KEY');
    expect(plain.classifyConcurrency).toBe(16);
    expect(plain.classifyThreshold).toBe(0.5);
    expect(
      parseArgs([
        '--classify',
        'typesafe',
        '--classify-model',
        'jev-2',
        '--classify-key-env',
        'MY_TS_KEY',
        '--classify-concurrency',
        '4',
        '--classify-threshold',
        '0.7',
      ]),
    ).toMatchObject({
      classify: 'typesafe',
      classifyModel: 'jev-2',
      classifyKeyEnv: 'MY_TS_KEY',
      classifyConcurrency: 4,
      classifyThreshold: 0.7,
    });
    expect(parseArgs(['--classify', 'text']).classify).toBe('text');
    expect(() => parseArgs(['--classify', 'labels'])).toThrow(
      /--classify must be label, text or typesafe/,
    );
    expect(() => parseArgs(['--classify-threshold', '3'])).toThrow(/probability from 0 to 1/);
    expect(() => parseArgs(['--classify-key-env', 'not a name'])).toThrow(
      /environment variable name/,
    );
    expect(() =>
      parseArgs([
        '--classify',
        'text',
        '--retrieval-unit',
        'turn',
        '--turn-unit-question-types',
        'multi-session',
      ]),
    ).toThrow(/--turn-unit-question-types/);
  });
});
