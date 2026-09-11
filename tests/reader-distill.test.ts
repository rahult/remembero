import { describe, expect, it } from 'vitest';
import {
  assembleHaystack,
  parseQuestionReply,
  pickType,
  questionWriterPrompt,
  acceptDistilled,
  type DistillType,
} from '../src/training/reader-distill.js';
import type { LabelledSession } from '../src/training/reader-data.js';
import { createRng } from '../src/training/rng.js';

const sessions: LabelledSession[] = Array.from({ length: 30 }, (_, i) => ({
  id: `s${i}`,
  date: `2023-${String(1 + (i % 9)).padStart(2, '0')}-${String(1 + (i % 27)).padStart(2, '0')}`,
  facts: i % 3 === 0 ? [`likes(user, thing_${i}).`] : [],
  transcript: `user: message ${i}\nassistant: reply ${i}`,
}));

describe('reader distillation data', () => {
  it('assembles a dated haystack of 5 to 15 sessions with a question date after the latest', () => {
    const rng = createRng(4);
    for (let i = 0; i < 20; i += 1) {
      const h = assembleHaystack(sessions, rng);
      expect(h.sessions.length).toBeGreaterThanOrEqual(5);
      expect(h.sessions.length).toBeLessThanOrEqual(15);
      const latest = h.sessions
        .map((s) => s.date)
        .sort()
        .at(-1)!;
      expect(h.questionDate > latest).toBe(true);
      // date order
      for (let k = 1; k < h.sessions.length; k += 1)
        expect(h.sessions[k - 1].date <= h.sessions[k].date).toBe(true);
    }
  });

  it('draws question types with abstention and preference present', () => {
    const rng = createRng(9);
    const counts = new Map<DistillType, number>();
    for (let i = 0; i < 2000; i += 1)
      counts.set(pickType(rng), (counts.get(pickType(rng)) ?? 0) + 1);
    for (const t of [
      'multi-session',
      'temporal-reasoning',
      'knowledge-update',
      'single-session-user',
      'single-session-assistant',
      'single-session-preference',
      'abstention',
    ] as DistillType[]) {
      expect(counts.get(t) ?? 0).toBeGreaterThan(50);
    }
  });

  it('asks the question writer for the type with the sessions numbered and dated, and parses its JSON', () => {
    const h = assembleHaystack(sessions, createRng(1));
    const prompt = questionWriterPrompt(h, 'multi-session');
    expect(prompt).toContain('Session 1');
    expect(prompt).toContain(h.sessions[0].date);
    expect(prompt).toContain('several sessions');
    const parsed = parseQuestionReply(
      'Here you go:\n{"question": "How many books have I finished this year?", "evidence": [1, 3], "note": "x"}',
    );
    expect(parsed).toEqual({
      question: 'How many books have I finished this year?',
      evidence: [1, 3],
    });
    expect(parseQuestionReply('no json here')).toBeUndefined();
  });

  it('keeps an abstention example only when the reader abstained, and drops an abstaining answer otherwise', () => {
    expect(
      acceptDistilled('abstention', 'The history does not mention that.'),
    ).toBe(true);
    expect(acceptDistilled('abstention', 'You bought three kits.')).toBe(false);
    expect(acceptDistilled('multi-session', 'I do not know.')).toBe(false);
    expect(
      acceptDistilled(
        'multi-session',
        'You mentioned 3 kits: a Zero, a Spitfire and a Mustang.',
      ),
    ).toBe(true);
    expect(acceptDistilled('single-session-user', '')).toBe(false);
  });
});
