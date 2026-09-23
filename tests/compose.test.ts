import { describe, expect, it } from 'vitest';
import { addDays, formatDate, fromDayNumber, toDayNumber, ymd } from '../src/compose/dates.js';
import { WorldProgram } from '../src/compose/program.js';
import { generateQuestions, FAMILIES } from '../src/compose/questions.js';
import { paginate, renderWorld } from '../src/compose/render.js';
import { declines, mentions, numbersIn, scoreCompose } from '../src/compose/score.js';
import { generateWorld, personName } from '../src/compose/world.js';

describe('dates', () => {
  it('round-trips day numbers across leap years', () => {
    for (const d of [ymd(2020, 2, 29), ymd(2021, 12, 31), ymd(2024, 3, 1)]) expect(fromDayNumber(toDayNumber(d))).toBe(d);
    expect(addDays(ymd(2024, 2, 28), 2)).toBe(ymd(2024, 3, 1));
  });

  it('writes the four document styles', () => {
    expect(formatDate(ymd(2025, 3, 14), 'long')).toBe('14 March 2025');
    expect(formatDate(ymd(2025, 3, 14), 'slash')).toBe('14/03/2025');
  });
});

describe('generateWorld', () => {
  it('is the same world for the same seed', () => {
    expect(generateWorld(5, 'test')).toEqual(generateWorld(5, 'test'));
  });

  it('draws names from disjoint pools per split', () => {
    const names = (split: 'train' | 'test') => new Set(generateWorld(3, split).people.map((p) => p.last));
    const train = names('train');
    expect([...names('test')].some((n) => train.has(n))).toBe(false);
  });

  it('never gives one person two roles at once', () => {
    const world = generateWorld(11, 'test');
    for (const a of world.roleTerms) {
      for (const b of world.roleTerms) {
        if (a === b || a.person !== b.person) continue;
        expect(a.to <= b.from || b.to <= a.from).toBe(true);
      }
    }
  });
});

describe('WorldProgram', () => {
  const world = generateWorld(101, 'test');
  const program = new WorldProgram(world, [ymd(2023, 6, 1)]);

  it('finds exactly one holder of a role on a covered date, or none in a gap', () => {
    const holders = program.column(`holds_role_on(cro, P, ${ymd(2023, 6, 1)})`, 'P');
    expect(holders.length).toBeLessThanOrEqual(1);
  });

  it('reads the latest value in force, not an earlier one', () => {
    const amended = world.contractValues.filter((v) => v.amendment > 0);
    if (amended.length === 0) return;
    const last = amended[amended.length - 1]!;
    const later = new WorldProgram(world, [addDays(last.effective, 1)]);
    const values = world.contractValues.filter((v) => v.contract === last.contract && v.effective <= addDays(last.effective, 1));
    const newest = values.sort((a, b) => b.effective - a.effective)[0]!;
    expect(later.column(`value_on(${last.contract}, V, ${addDays(last.effective, 1)})`, 'V')).toEqual([newest.value]);
  });
});

describe('renderWorld and paginate', () => {
  const world = generateWorld(101, 'test');
  const pages = paginate(renderWorld(world));

  it('puts every appointment, register row and incident on some page', () => {
    const all = new Set(pages.flatMap((p) => p.keys));
    for (const t of world.roleTerms) expect(all.has(`role:${t.role}:${t.person}:from`)).toBe(true);
    for (const c of world.contracts) expect(all.has(`value:${c.id}:0`)).toBe(true);
    for (const i of world.incidents) expect(all.has(`incident:${i.id}`)).toBe(true);
  });

  it('writes every appointed person by full name somewhere', () => {
    const text = pages.map((p) => p.text).join('\n');
    for (const t of world.roleTerms) expect(text).toContain(personName(world.people.find((p) => p.id === t.person)!));
  });
});

describe('generateQuestions', () => {
  const world = generateWorld(101, 'test');
  const questions = generateQuestions(world);
  const keysOnPages = new Set(paginate(renderWorld(world)).flatMap((p) => p.keys));

  it('covers every family', () => {
    for (const family of FAMILIES) expect(questions.some((q) => q.family === family)).toBe(true);
  });

  it('points every evidence key at a rendered page', () => {
    for (const q of questions) for (const key of q.evidence) expect(keysOnPages.has(key)).toBe(true);
  });

  it('gives unanswerable questions no evidence', () => {
    for (const q of questions.filter((x) => x.family === 'unanswerable')) expect(q.evidence).toEqual([]);
  });

  it('is deterministic', () => {
    expect(generateQuestions(world)).toEqual(questions);
  });
});

describe('scoreCompose', () => {
  const entity = { kind: 'entity' as const, items: [['Amara Osei', 'Osei']], distractors: ['Tobias Brennan', 'Brennan'], distractorNumbers: [], display: 'Amara Osei' };

  it('accepts any spelling of the gold and rejects a distractor', () => {
    expect(scoreCompose('It was Dr Osei.', entity)).toBe('correct');
    expect(scoreCompose('Tobias Brennan', entity)).toBe('wrong');
  });

  it('scores the candidate the answer asserts first, not the ones it explains away', () => {
    expect(scoreCompose('Osei, who succeeded Tobias Brennan in March.', entity)).toBe('correct');
    expect(scoreCompose('Brennan; Osei only took over later.', entity)).toBe('wrong');
  });

  it('calls an answer that names nobody partial, not wrong', () => {
    expect(scoreCompose('The Chief Risk Officer.', entity)).toBe('partial');
    expect(scoreCompose('A. Osei', { ...entity, items: [['Amara Osei']], partial: ['Osei'] })).toBe('partial');
  });

  it('counts a decline as declined, never as wrong', () => {
    expect(scoreCompose('I do not know from the supplied history.', entity)).toBe('declined');
  });

  it('matches money in any format and rejects a superseded value', () => {
    const number = { kind: 'number' as const, items: [], number: 1_410_000, distractors: [], distractorNumbers: [1_250_000], display: '1410000' };
    expect(scoreCompose('AUD 1,410,000', number)).toBe('correct');
    expect(scoreCompose('$1.41 million', number)).toBe('correct');
    expect(scoreCompose('AUD 1,410,000, up from AUD 1,250,000 at signing', number)).toBe('correct');
    expect(scoreCompose('AUD 1,250,000', number)).toBe('wrong');
    expect(scoreCompose('AUD 990,000', number)).toBe('wrong');
  });

  it('reads yes and no', () => {
    const decision = { kind: 'decision' as const, items: [], allowed: false, distractors: [], distractorNumbers: [], display: 'no' };
    expect(scoreCompose('No — the value exceeded the $250,000 limit.', decision)).toBe('correct');
    expect(scoreCompose('Yes, it was within authority.', decision)).toBe('wrong');
  });

  it('requires every item of a set', () => {
    const set = { kind: 'set' as const, items: [['Halstead'], ['Kestrel']], distractors: ['Northgate'], distractorNumbers: [], display: '' };
    expect(scoreCompose('Halstead Logistics and Kestrel Freight', set)).toBe('correct');
    expect(scoreCompose('Halstead Logistics and Kestrel Freight, while Northgate held a valid certificate.', set)).toBe('correct');
    expect(scoreCompose('Halstead Logistics', set)).toBe('partial');
    expect(scoreCompose('Halstead and Northgate', set)).toBe('wrong');
  });

  it('marks an unknown question correct only when the answer declines', () => {
    const unknown = { kind: 'unknown' as const, items: [], distractors: [], distractorNumbers: [], display: 'unknown' };
    expect(scoreCompose('The records do not say.', unknown)).toBe('correct');
    expect(scoreCompose('AUD 400,000', unknown)).toBe('wrong');
  });

  it('parses numbers and phrases in isolation', () => {
    expect(numbersIn('2 incidents, AUD 1,410,000 and $1.5m')).toEqual([2, 1_410_000, 1_500_000]);
    expect(mentions('Osei approved it', 'Osei')).toBe(true);
    expect(mentions('Oseiman approved it', 'Osei')).toBe(false);
    expect(declines('Unknown.')).toBe(true);
  });
});
