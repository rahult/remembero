import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  classifyMiss,
  classifyObservation,
  dropTypesFromThinking,
  recommendMissShare,
  recommendTypeWeights,
  reviewReader,
} from '../src/training/reader-review.js';

const base = {
  type: 'single-session-user',
  question: 'what colour is my car?',
  expected: 'Red.',
  hypothesis: 'Blue.',
  reply: 'Blue.',
};

describe('classifyMiss', () => {
  it('format: a thinking type whose reply has no Answer line', () => {
    const row = {
      ...base,
      type: 'multi-session',
      reply: 'Notes:\n- 2023-05-01: bought a car',
      hypothesis: 'Notes:\n- 2023-05-01: bought a car',
    };
    expect(classifyMiss({ ...row, thinking: true })).toBe('format');
    // the same reply from a direct student is not a format miss
    expect(classifyMiss({ ...row, thinking: false })).toBe('other');
  });

  it('format: an Answer line whose final text starts with an asterisk', () => {
    expect(
      classifyMiss({
        ...base,
        type: 'temporal-reasoning',
        reply: 'Notes:\n- x\n**Answer:** 3 weeks',
        hypothesis: '** 3 weeks',
        thinking: true,
      }),
    ).toBe('format');
  });

  it('format: a thinking-type completion cut at the token limit', () => {
    expect(
      classifyMiss({
        ...base,
        type: 'knowledge-update',
        reply: 'Notes:\n- x\nAnswer: Blue',
        hypothesis: 'Blue',
        thinking: true,
        cutAtLimit: true,
      }),
    ).toBe('format');
  });

  it('format applies to abstention rows under a thinking student, not to single-session types', () => {
    const noAnswer = { reply: 'Notes: nothing', hypothesis: 'Notes: nothing' };
    expect(
      classifyMiss({
        ...base,
        ...noAnswer,
        type: 'abstention',
        thinking: true,
      }),
    ).toBe('format');
    expect(
      classifyMiss({
        ...base,
        ...noAnswer,
        type: 'single-session-user',
        thinking: true,
      }),
    ).toBe('other');
  });

  it('false-abstain: the student abstains, the teacher answered', () => {
    expect(
      classifyMiss({
        ...base,
        hypothesis: 'Your history does not mention your car.',
        reply: 'Your history does not mention your car.',
      }),
    ).toBe('false-abstain');
  });

  it('missed-abstain: the teacher abstained, the student answered', () => {
    expect(
      classifyMiss({
        ...base,
        expected: 'I do not know; your history does not say.',
      }),
    ).toBe('missed-abstain');
  });

  it('count: a how-many question where both answers carry a number', () => {
    expect(
      classifyMiss({
        ...base,
        type: 'multi-session',
        question: 'How many marathons did I run?',
        expected: 'Three marathons.',
        hypothesis: '2',
        reply: '2',
      }),
    ).toBe('count');
    // no number in the hypothesis: other
    expect(
      classifyMiss({
        ...base,
        type: 'multi-session',
        question: 'How many marathons did I run?',
        expected: 'Three marathons.',
        hypothesis: 'Several.',
        reply: 'Several.',
      }),
    ).toBe('other');
  });

  it('date-arithmetic: a temporal row where both carry a digit, month or weekday', () => {
    expect(
      classifyMiss({
        ...base,
        type: 'temporal-reasoning',
        question: 'when did I sign up?',
        expected: 'In March.',
        hypothesis: 'On a Tuesday.',
        reply: 'On a Tuesday.',
      }),
    ).toBe('date-arithmetic');
    expect(
      classifyMiss({
        ...base,
        type: 'temporal-reasoning',
        question: 'when did I sign up?',
        expected: '10 days ago',
        hypothesis: 'Recently.',
        reply: 'Recently.',
      }),
    ).toBe('other');
    // the same answers on a non-temporal type: other
    expect(
      classifyMiss({
        ...base,
        question: 'when did I sign up?',
        expected: 'In March.',
        hypothesis: 'On a Tuesday.',
      }),
    ).toBe('other');
  });

  it('checks in order: format before abstention before count before date', () => {
    // a thinking temporal how-many row with no Answer line that abstains: format
    expect(
      classifyMiss({
        type: 'temporal-reasoning',
        question: 'how many days ago did I go?',
        expected: '3 days',
        hypothesis: 'I do not know',
        reply: 'I do not know',
        thinking: true,
      }),
    ).toBe('format');
    // abstention before count
    expect(
      classifyMiss({
        type: 'multi-session',
        question: 'how many days ago did I go?',
        expected: '3 days',
        hypothesis: 'no information about 3 trips',
        reply: 'no information about 3 trips',
      }),
    ).toBe('false-abstain');
    // count before date-arithmetic
    expect(
      classifyMiss({
        type: 'temporal-reasoning',
        question: 'how many days ago did I go?',
        expected: '3 days',
        hypothesis: '5 days',
        reply: '5 days',
      }),
    ).toBe('count');
  });
});

describe('classifyObservation', () => {
  const notes = { readingStrategy: 'notes', readerMaxTokens: 4000 };
  const direct = { readingStrategy: 'direct', readerMaxTokens: 300 };
  it('format from the token cap or an asterisk under notes', () => {
    expect(
      classifyObservation(
        {
          questionType: 'multi-session',
          abstention: false,
          hypothesis: 'Three',
          readerUsage: { completionTokens: 4000 },
        },
        notes,
      ),
    ).toBe('format');
    expect(
      classifyObservation(
        {
          questionType: 'temporal-reasoning',
          abstention: false,
          hypothesis: '** 3 weeks',
          readerUsage: { completionTokens: 40 },
        },
        notes,
      ),
    ).toBe('format');
    // direct reading: the cap is not a thinking format miss
    expect(
      classifyObservation(
        {
          questionType: 'multi-session',
          abstention: false,
          hypothesis: 'Three',
          readerUsage: { completionTokens: 300 },
        },
        direct,
      ),
    ).toBe('unclassified');
  });

  it('abstention classes from the abstention flag; the rest is unclassified', () => {
    expect(
      classifyObservation(
        {
          questionType: 'multi-session',
          abstention: false,
          hypothesis: 'Your history does not mention it.',
        },
        direct,
      ),
    ).toBe('false-abstain');
    expect(
      classifyObservation(
        { questionType: 'multi-session', abstention: true, hypothesis: 'Two' },
        direct,
      ),
    ).toBe('missed-abstain');
    expect(
      classifyObservation(
        { questionType: 'multi-session', abstention: false, hypothesis: 'Two' },
        direct,
      ),
    ).toBe('unclassified');
  });
});

describe('recommendations', () => {
  it('weights: misses over total misses, floored at 5%, renormalised', () => {
    const { spec, weights } = recommendTypeWeights({
      'multi-session': { asked: 100, misses: 59 },
      'temporal-reasoning': { asked: 100, misses: 40 },
      'single-session-user': { asked: 100, misses: 1 },
      'knowledge-update': { asked: 0, misses: 0 },
    });
    // 0.59, 0.40, max(0.01, 0.05) = 0.05; sum 1.04
    expect(weights['multi-session']).toBeCloseTo(0.59 / 1.04, 10);
    expect(weights['temporal-reasoning']).toBeCloseTo(0.4 / 1.04, 10);
    expect(weights['single-session-user']).toBeCloseTo(0.05 / 1.04, 10);
    expect(weights).not.toHaveProperty('knowledge-update');
    expect(spec).toBe(
      'multi-session=57,temporal-reasoning=38,single-session-user=5',
    );
  });

  it('weights with no misses at all are equal', () => {
    const { spec } = recommendTypeWeights({
      'multi-session': { asked: 10, misses: 0 },
      abstention: { asked: 10, misses: 0 },
    });
    expect(spec).toBe('abstention=50,multi-session=50');
  });

  it('miss share: 0.25, raised to 0.35 above a 40% miss rate', () => {
    expect(recommendMissShare(100, 40)).toBe(0.25);
    expect(recommendMissShare(100, 41)).toBe(0.35);
    expect(recommendMissShare(0, 0)).toBe(0.25);
  });

  it('drops a type whose thinking run lost more than 5 questions', () => {
    expect(
      dropTypesFromThinking(
        {
          'multi-session': { questions: 133, correct: 78 },
          'temporal-reasoning': { questions: 133, correct: 94 },
        },
        {
          'multi-session': { questions: 133, correct: 84 },
          'temporal-reasoning': { questions: 133, correct: 99 },
        },
      ),
    ).toEqual(['multi-session']);
  });
});

function jsonl(path: string, rows: unknown[]): void {
  writeFileSync(path, rows.map((row) => `${JSON.stringify(row)}\n`).join(''));
}

function runJson(
  path: string,
  correct: Record<string, number>,
  observations: unknown[],
  settings: Record<string, unknown>,
): void {
  writeFileSync(
    path,
    JSON.stringify({
      settings: {
        dateDistances: true,
        computedNotes: true,
        focusedBudget: false,
        structuredEvidence: false,
        fullSessions: null,
        abstractBytes: null,
        readerMaxTokens: 300,
        ...settings,
      },
      contextBytes: 24576,
      byQuestionType: Object.fromEntries(
        Object.entries(correct).map(([type, n]) => [
          type,
          { questions: 20, correct: n, accuracy: n / 20 },
        ]),
      ),
      observations,
    }),
  );
}

describe('reviewReader', () => {
  it('writes accuracy deltas, class counts and the recommendation', () => {
    const dir = mkdtempSync(join(tmpdir(), 'review-'));
    const run = join(dir, 'run.json');
    const baseline = join(dir, 'baseline.json');
    runJson(
      run,
      { 'multi-session': 10, 'temporal-reasoning': 15 },
      [
        {
          questionId: 'a',
          questionType: 'multi-session',
          abstention: false,
          status: 'judged',
          correct: false,
          hypothesis: 'Two',
          readerUsage: { completionTokens: 4000 },
        },
        {
          questionId: 'b',
          questionType: 'temporal-reasoning',
          abstention: true,
          status: 'judged',
          correct: false,
          hypothesis: '3 weeks',
          readerUsage: { completionTokens: 50 },
        },
        {
          questionId: 'c',
          questionType: 'temporal-reasoning',
          abstention: false,
          status: 'judged',
          correct: true,
          hypothesis: '3 weeks',
          readerUsage: { completionTokens: 50 },
        },
        {
          questionId: 'd',
          questionType: 'multi-session',
          abstention: false,
          status: 'judged',
          correct: false,
          hypothesis: 'Four',
          readerUsage: { completionTokens: 50 },
        },
      ],
      { readingStrategy: 'notes', readerMaxTokens: 4000 },
    );
    runJson(baseline, { 'multi-session': 16, 'temporal-reasoning': 14 }, [], {
      readingStrategy: 'direct',
    });
    const mined = join(dir, 'mined');
    mkdirSync(mined);
    writeFileSync(
      join(mined, 'manifest.json'),
      JSON.stringify({
        contract: { id: 'dd+notes+think@24576', thinking: true },
        studentContract: { id: 'dd+notes@24576', thinking: false },
      }),
    );
    const result = (
      file: string,
      index: number,
      type: string,
      correct: boolean,
      extra: Record<string, string> = {},
    ) => ({
      file,
      index,
      type,
      question: 'what did I do?',
      expected: 'Ran.',
      hypothesis: 'Walked.',
      reply: 'Walked.',
      correct,
      ...extra,
    });
    jsonl(join(mined, 'results.jsonl'), [
      result('conversations.jsonl', 0, 'multi-session', false, {
        question: 'how many runs?',
        expected: '3',
        hypothesis: '2',
        reply: '2',
      }),
      result('conversations.jsonl', 1, 'multi-session', true),
      result('conversations.jsonl', 2, 'temporal-reasoning', false, {
        hypothesis: 'I do not know',
        reply: 'I do not know',
      }),
      result('conversations.jsonl', 3, 'temporal-reasoning', true),
      result('conversations.jsonl', 4, 'abstention', true),
      result('heldout.jsonl', 0, 'multi-session', false),
    ]);
    const review = reviewReader({
      run,
      baseline,
      mined,
      now: new Date('2026-09-16T00:00:00Z'),
    });
    expect(review.generatedAt).toBe('2026-09-16T00:00:00.000Z');
    expect(review.inputs).toEqual({
      run: { path: run, contract: 'dd+notes+think@24576' },
      baseline: { path: baseline, contract: 'dd+notes@24576' },
      mined: {
        path: mined,
        contract: 'dd+notes+think@24576',
        studentContract: 'dd+notes@24576',
      },
    });
    expect(review.accuracy['multi-session']).toEqual({
      questions: 20,
      correct: 10,
      accuracy: 0.5,
      baseline: { questions: 20, correct: 16, accuracy: 0.8 },
      delta: { correct: -6, accuracy: 0.5 - 0.8 },
    });
    expect(review.accuracy['temporal-reasoning']!.delta!.correct).toBe(1);
    expect(review.longMemEval.run).toEqual({
      misses: 3,
      classes: {
        format: 1,
        'false-abstain': 0,
        'missed-abstain': 1,
        unclassified: 1,
      },
      byType: {
        'multi-session': { format: 1, unclassified: 1 },
        'temporal-reasoning': { 'missed-abstain': 1 },
      },
      unavailable: ['count', 'date-arithmetic', 'other'],
    });
    expect(review.mined.classes).toEqual({
      format: 0,
      'false-abstain': 1,
      'missed-abstain': 0,
      count: 1,
      'date-arithmetic': 0,
      other: 1,
    });
    expect(review.mined.byType).toEqual({
      'multi-session': { asked: 3, correct: 1, misses: 2 },
      'temporal-reasoning': { asked: 2, correct: 1, misses: 1 },
      abstention: { asked: 1, correct: 1, misses: 0 },
    });
    // conversations only: 2 misses over 5 asked = 0.40, not above
    expect(review.mined.missRate).toBe(0.4);
    expect(review.recommend.missShare).toBe(0.25);
    expect(review.recommend.dropTypesFromThinking).toEqual(['multi-session']);
    // 2/3, 1/3, max(0, 0.05): sum 1.05
    expect(review.recommend.typeWeights).toBe(
      'multi-session=63,temporal-reasoning=32,abstention=5',
    );
    expect(review.recommend.typeWeightsRaw.abstention).toBeCloseTo(
      0.05 / 1.05,
      10,
    );
  });

  it('without a baseline: no deltas and nothing dropped', () => {
    const dir = mkdtempSync(join(tmpdir(), 'review-'));
    const run = join(dir, 'run.json');
    runJson(run, { 'multi-session': 10 }, [], { readingStrategy: 'direct' });
    const mined = join(dir, 'mined');
    mkdirSync(mined);
    writeFileSync(
      join(mined, 'manifest.json'),
      JSON.stringify({
        contract: { id: 'x@1' },
        studentContract: { id: 'y@1' },
      }),
    );
    jsonl(join(mined, 'results.jsonl'), []);
    const review = reviewReader({ run, mined });
    expect(review.inputs.baseline).toBeNull();
    expect(review.accuracy['multi-session']).toEqual({
      questions: 20,
      correct: 10,
      accuracy: 0.5,
    });
    expect(review.recommend.dropTypesFromThinking).toEqual([]);
    expect(review.longMemEval.baseline).toBeNull();
  });
});
