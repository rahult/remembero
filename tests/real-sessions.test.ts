import { describe, expect, it } from 'vitest';
import {
  compareFactSets,
  selectTrainingSessions,
  toRealConversation,
} from '../src/training/real-sessions.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';

const instance = (
  id: string,
  sessionIds: string[],
  evidence: string[],
  texts: string[],
): LongMemEvalInstance => ({
  question_id: id,
  question_type: 'single-session-user',
  question: 'q',
  answer: 'a',
  question_date: '2024/01/03 (Wed) 09:00',
  haystack_session_ids: sessionIds,
  haystack_dates: sessionIds.map(() => '2024/01/01 (Mon) 09:00'),
  haystack_sessions: texts.map((t) => [
    { role: 'user', content: t },
    { role: 'assistant', content: 'ok' },
  ]),
  answer_session_ids: evidence,
});

describe('selectTrainingSessions: real sessions that cannot leak into the dev evaluation', () => {
  it('keeps only sessions that appear in no dev haystack, are evidence for nothing, and repeat no dev text', () => {
    const split = (id: string) => (id.startsWith('dev') ? 'dev' : 'test');
    const instances = [
      instance(
        'dev-1',
        ['s1', 's2'],
        ['s1'],
        ['dev fact one', 'shared filler'],
      ),
      instance(
        'test-1',
        ['s2', 's3', 's4', 's5'],
        ['s4'],
        ['shared filler', 'test filler A', 'test evidence', 'shared filler'],
      ),
    ];
    const chosen = selectTrainingSessions(instances, split);
    // s2 is in a dev haystack, s4 is evidence, s5 repeats dev text 'shared filler'
    expect(chosen.map((s) => s.id)).toEqual(['s3']);
    expect(chosen[0]?.session[0]?.content).toBe('test filler A');
  });
});

describe('compareFactSets: loose recall and precision between a reference and a model extraction', () => {
  it('matches facts by shared non-self constants, ignoring predicate names', () => {
    const reference = [
      'degree(user, business_administration).',
      'bought(user, kayak).',
      "visited(user, 'Big Sur').",
    ];
    const model = [
      'major(user, business_administration).',
      "trip(user, 'Big Sur', 3).",
      'likes(user, coffee).',
    ];
    const score = compareFactSets(reference, model, 'user');
    expect(score).toMatchObject({
      referenceFacts: 3,
      modelFacts: 3,
      matchedReference: 2,
      matchedModel: 2,
    });
    expect(score.recall).toBeCloseTo(2 / 3);
    expect(score.precision).toBeCloseTo(2 / 3);
  });

  it('treats an empty reference with an empty model output as agreement, not division by zero', () => {
    const score = compareFactSets([], [], 'user');
    expect(score.recall).toBe(1);
    expect(score.precision).toBe(1);
  });
});

describe('toRealConversation: the transcript prompt over an empty store', () => {
  it('exports USER/ASSISTANT turns with the facts as the answer, or the nothing sentinel', () => {
    const session = [
      { role: 'user' as const, content: 'By the way, I bought a kayak.' },
      { role: 'assistant' as const, content: 'x'.repeat(2000) },
    ];
    const withFacts = toRealConversation(
      session,
      ['bought(user, kayak).'],
      'user',
    );
    expect(withFacts.messages[0]?.content).toContain('transcript');
    expect(withFacts.messages[1]?.content).toContain(
      'USER: By the way, I bought a kayak.',
    );
    // assistant turns are compacted the way the evaluation compacts them
    expect(withFacts.messages[1]?.content.length).toBeLessThan(1_000);
    expect(withFacts.messages[2]?.content).toBe('bought(user, kayak).');
    const nothing = toRealConversation(session, [], 'user');
    expect(nothing.messages[2]?.content).toBe('% nothing');
  });
});
