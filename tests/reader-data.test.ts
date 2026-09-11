import { describe, expect, it } from 'vitest';
import {
  generateReaderExamples,
  templatedQuestion,
  toReaderConversation,
  type LabelledSession,
} from '../src/training/reader-data.js';
import { createRng } from '../src/training/rng.js';

const session = (
  id: string,
  date: string,
  facts: string[],
  text: string,
): LabelledSession => ({ id, date, facts, transcript: `user: ${text}` });

const SESSIONS: LabelledSession[] = [
  session(
    's1',
    '2023-01-10',
    ['lives_in(user, austin).'],
    'I live in Austin these days.',
  ),
  session(
    's2',
    '2023-02-14',
    ['bought(user, spitfire_kit).'],
    'Picked up a Spitfire kit.',
  ),
  session(
    's3',
    '2023-03-20',
    ['bought(user, zero_kit).'],
    'Bought a Zero kit today.',
  ),
  session(
    's4',
    '2023-04-02',
    ['bought(user, mustang_kit).'],
    'Got a Mustang kit.',
  ),
  session(
    's5',
    '2023-05-02',
    ['lives_in(user, denver).'],
    'We moved to Denver last week.',
  ),
  session(
    's6',
    '2023-05-09',
    ['drinks(user, oat_milk_latte).'],
    'My usual is an oat milk latte.',
  ),
  session(
    's7',
    '2023-06-01',
    ['plays(user, chess).'],
    'Chess club again tonight.',
  ),
  session(
    's8',
    '2023-06-11',
    ['owns(user, corgi).'],
    'The corgi chewed a shoe.',
  ),
];

describe('reader training data', () => {
  const examples = generateReaderExamples(SESSIONS, createRng(1), {
    perType: 3,
    questionDate: '2023-07-01',
  });

  it('builds a count question whose gold is the number of sessions sharing the predicate', () => {
    const count = examples.find((e) => e.type === 'multi-session-count')!;
    expect(count.evidenceSessionIds.sort()).toEqual(['s2', 's3', 's4']);
    expect(count.gold).toMatch(/^3\b/);
    expect(count.gold).toContain('spitfire kit');
  });

  it('builds a knowledge-update question whose gold is the later value', () => {
    const update = examples.find((e) => e.type === 'knowledge-update')!;
    expect(update.evidenceSessionIds.sort()).toEqual(['s1', 's5']);
    expect(update.gold).toMatch(/^denver\b/);
    expect(update.gold).toContain('austin');
  });

  it('builds a temporal question whose gold is computed from the session date', () => {
    const temporal = examples.find((e) => e.type === 'temporal-reasoning')!;
    // one dated fact; the question date is 2023-07-01
    expect(temporal.gold).toMatch(/\d+ days? \(about \d+ weeks?/);
  });

  it('builds an abstention question about a predicate absent from the assembled history', () => {
    const abstention = examples.find((e) => e.type === 'abstention')!;
    expect(abstention.gold).toMatch(/does not say/i);
    const conversation = toReaderConversation(abstention);
    const prompt = conversation.messages[1].content;
    for (const id of abstention.evidenceSessionIds)
      expect(prompt).not.toContain(id);
  });

  it('renders the context in the evaluation format with distractor sessions and date distances', () => {
    const count = examples.find((e) => e.type === 'multi-session-count')!;
    const conversation = toReaderConversation(count);
    expect(conversation.messages[0].role).toBe('system');
    const prompt = conversation.messages[1].content;
    expect(prompt).toContain('History chats:');
    expect(prompt).toContain('before the question date 2023-07-01');
    expect(prompt).toContain(
      'Remembered facts (stated in this session): bought(user, spitfire_kit).',
    );
    expect(prompt).toContain('Question: ');
    expect(count.distractorSessionIds.length).toBeGreaterThan(0);
    expect(conversation.messages[2].content).toBe(count.gold);
  });

  it('phrases templated questions from the predicate without leaking the answer', () => {
    expect(
      templatedQuestion('single-session-user', 'lives_in', 'austin'),
    ).not.toContain('austin');
    expect(templatedQuestion('multi-session-count', 'bought', null)).toMatch(
      /how many/i,
    );
  });
});
