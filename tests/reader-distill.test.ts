import { describe, expect, it } from 'vitest';
import {
  assembleHaystack,
  parseTypeWeights,
  predicateGroups,
  parseQuestionReply,
  pickType,
  questionWriterPrompt,
  acceptDistilled,
  distillRunIdentity,
  distillSeeds,
  splitDistillPools,
  type DistillType,
} from '../src/training/reader-distill.js';
import { assertRunIdentity } from '../src/training/reader-think.js';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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

  it('seeds a multi-session haystack with sessions sharing a predicate and an update haystack with a changed value', () => {
    const pool: LabelledSession[] = [
      {
        id: 'a',
        date: '2023-01-01',
        facts: ['bought(user, kit_a).'],
        transcript: 'user: a',
      },
      {
        id: 'b',
        date: '2023-02-01',
        facts: ['bought(user, kit_b).'],
        transcript: 'user: b',
      },
      {
        id: 'c',
        date: '2023-03-01',
        facts: ['bought(user, kit_c).'],
        transcript: 'user: c',
      },
      {
        id: 'd',
        date: '2023-04-01',
        facts: ['lives_in(user, austin).'],
        transcript: 'user: d',
      },
      {
        id: 'e',
        date: '2023-05-01',
        facts: ['lives_in(user, denver).'],
        transcript: 'user: e',
      },
      ...Array.from({ length: 10 }, (_, i) => ({
        id: `f${i}`,
        date: `2023-06-${String(i + 1).padStart(2, '0')}`,
        facts: [],
        transcript: `user: filler ${i}`,
      })),
    ];
    const groups = predicateGroups(pool);
    expect(groups.get('bought')!.length).toBe(3);
    const multi = assembleHaystack(pool, createRng(2), {
      seed: 'multi-session',
      groups,
    });
    expect(
      multi.sessions.filter((s) => ['a', 'b', 'c'].includes(s.id)).length,
    ).toBe(3);
    const update = assembleHaystack(pool, createRng(3), {
      seed: 'knowledge-update',
      groups,
    });
    expect(
      update.sessions.filter((s) => ['d', 'e'].includes(s.id)).length,
    ).toBe(2);
    expect(update.sessions.length).toBeGreaterThanOrEqual(5);
  });

  it('parses type weights from a flag', () => {
    const w = parseTypeWeights(
      'multi-session=40,temporal-reasoning=35,abstention=10',
    );
    expect(w.find(([t]) => t === 'multi-session')![1]).toBe(40);
    expect(w.length).toBe(3);
    const rng = createRng(1);
    for (let i = 0; i < 50; i += 1)
      expect(['multi-session', 'temporal-reasoning', 'abstention']).toContain(
        pickType(rng, w),
      );
  });
});

describe('distill seeds and session pools', () => {
  const pool = Array.from({ length: 40 }, (_, i) => ({
    id: `session-${String(i).padStart(2, '0')}`,
  }));
  const pools = (argv: string[]) => {
    const { splitSeed } = distillSeeds(argv);
    const { train, held } = splitDistillPools(pool, splitSeed, 30);
    return { train: train.map((s) => s.id), held: held.map((s) => s.id) };
  };

  it('defaults the split seed to --seed and reads --split-seed when given', () => {
    expect(distillSeeds([])).toEqual({ seed: 7, splitSeed: 7 });
    expect(distillSeeds(['--seed', '17'])).toEqual({ seed: 17, splitSeed: 17 });
    expect(distillSeeds(['--seed', '17', '--split-seed', '7'])).toEqual({
      seed: 17,
      splitSeed: 7,
    });
    expect(() => distillSeeds(['--split-seed', 'x'])).toThrow(/--split-seed/);
  });

  it('keeps the same train and held-out pools when --seed changes under a fixed split seed', () => {
    const v6 = pools(['--seed', '7']);
    const fresh17 = pools(['--seed', '17', '--split-seed', '7']);
    const fresh19 = pools(['--seed', '19', '--split-seed', '7']);
    expect(fresh17).toEqual(v6);
    expect(fresh19).toEqual(v6);
    expect(v6.train).toHaveLength(30);
    expect(v6.held).toHaveLength(10);
    expect(v6.train.filter((id) => v6.held.includes(id))).toEqual([]);
    expect(pools(['--seed', '17'])).not.toEqual(v6);
  });

  it('orders as shuffle-by-id then filter, so a split seed reproduces an earlier seed run', () => {
    const shuffled = createRng(7)
      .shuffle([...pool].reverse().sort((a, b) => a.id.localeCompare(b.id)))
      .filter((s) => !s.id.endsWith('3'));
    const { train, held } = splitDistillPools(
      [...pool].reverse(),
      7,
      20,
      (s) => !s.id.endsWith('3'),
    );
    expect(train).toEqual(shuffled.slice(0, 20));
    expect(held).toEqual(shuffled.slice(20));
  });
});

describe('distill run identity', () => {
  const identity = distillRunIdentity({
    contract: 'dd+notes+think@24576',
    teacher: 'glm-5.3-flash:cloud',
    seed: 17,
    splitSeed: 7,
    typeWeights: 'multi-session=40,temporal-reasoning=35',
    trainCount: 3000,
    labels: 'data/real/labels-glmflash8.jsonl',
  });

  it('pins contract, teacher, seed, split seed, type weights, train count and labels', () => {
    expect(Object.keys(identity).sort()).toEqual(
      [
        'contract',
        'labels',
        'seed',
        'splitSeed',
        'teacher',
        'trainCount',
        'typeWeights',
      ].sort(),
    );
    const out = mkdtempSync(join(tmpdir(), 'distill-run-'));
    assertRunIdentity(out, identity);
    expect(() => assertRunIdentity(out, { ...identity })).not.toThrow();
  });

  it('refuses a resume begun with any different pinned field', () => {
    const out = mkdtempSync(join(tmpdir(), 'distill-run-'));
    assertRunIdentity(out, identity);
    const changed: Record<string, unknown> = {
      contract: 'dd+notes@24576',
      teacher: 'other',
      seed: 19,
      splitSeed: 17,
      typeWeights: 'default',
      trainCount: 2000,
      labels: 'other.jsonl',
    };
    for (const [key, value] of Object.entries(changed))
      expect(() =>
        assertRunIdentity(out, { ...identity, [key]: value }),
      ).toThrow(new RegExp(key));
  });
});
