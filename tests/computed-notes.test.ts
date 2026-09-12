import { describe, expect, it } from 'vitest';
import {
  buildComputedNotes,
  extractQuantities,
  resolveTemporalExpressions,
} from '../src/evals/computed-notes.js';

const session = '2023-02-22T10:00:00Z';

describe('resolveTemporalExpressions', () => {
  it('anchors relative expressions to the session date', () => {
    const events = resolveTemporalExpressions(
      'USER: I did the Walk for Hunger 5K yesterday. Two weeks ago I started training. Today I rested.',
      session,
    );
    const byExpr = Object.fromEntries(events.map((e) => [e.expression.toLowerCase(), e.iso]));
    expect(byExpr['yesterday']).toBe('2023-02-21');
    expect(byExpr['two weeks ago']).toBe('2023-02-08');
    expect(byExpr['today']).toBe('2023-02-22');
  });

  it('reads absolute dates and assumes the session year when none is given', () => {
    const events = resolveTemporalExpressions(
      'USER: The Coastal Cleanup is on March 7th and I replaced my spark plugs on Feb 14, 2023. Trip on 2022-01-15.',
      session,
    );
    const isos = events.map((e) => e.iso);
    expect(isos).toContain('2023-03-07');
    expect(isos).toContain('2023-02-14');
    expect(isos).toContain('2022-01-15');
    expect(events.find((e) => e.iso === '2023-03-07')?.assumedYear).toBe(true);
  });

  it('resolves weekdays to the previous occurrence', () => {
    // 2023-02-22 is a Wednesday
    const events = resolveTemporalExpressions('USER: I went hiking last Saturday and to the dentist on Monday.', session);
    const byExpr = Object.fromEntries(events.map((e) => [e.expression.toLowerCase(), e.iso]));
    expect(byExpr['last saturday']).toBe('2023-02-18');
    expect(byExpr['on monday']).toBe('2023-02-20');
  });

  it('ignores assistant turns', () => {
    const events = resolveTemporalExpressions('ASSISTANT: Yesterday would be a fine day.\n\nUSER: I agree.', session);
    expect(events).toEqual([]);
  });
});

describe('extractQuantities', () => {
  it('reads numbers with units and durations', () => {
    const q = extractQuantities('USER: Outer Banks is about 4 hours away. My best time was 4 hours 22 minutes; my target was 4 hours 10 minutes. I packed 5 pairs of shoes and wore two pairs.');
    const units = q.map((x) => `${x.value} ${x.unit}`);
    expect(units).toContain('4 hour');
    expect(units).toContain('262 minute');
    expect(units).toContain('250 minute');
    expect(units).toContain('5 pair');
    expect(units).toContain('2 pair');
  });
});

describe('buildComputedNotes', () => {
  it('computes the gap between two dated events named in the question', () => {
    const notes = buildComputedNotes(
      "How many days had passed between the 'Walk for Hunger' event and the 'Coastal Cleanup' event?",
      '2023/03/14 (Tue) 21:24',
      [
        { ts: '2023-02-22T10:00:00Z', text: 'USER: I did the Walk for Hunger 5K yesterday, it was great.' },
        { ts: '2023-03-08T09:00:00Z', text: 'USER: The Coastal Cleanup yesterday was muddy but fun.' },
      ],
    );
    expect(notes).toContain('2023-02-21');
    expect(notes).toContain('2023-03-07');
    expect(notes).toMatch(/14 days/);
    expect(notes).toMatch(/21 days.*before the question|21 days ago/);
  });

  it('totals quantities that share a unit and differences durations', () => {
    const notes = buildComputedNotes(
      'How many hours in total did I spend driving to my three road trip destinations combined?',
      '2023/05/30 (Tue) 19:37',
      [
        { ts: '2023-05-10T10:00:00Z', text: 'USER: Outer Banks is about 4 hours of driving from my place.' },
        { ts: '2023-05-12T10:00:00Z', text: 'USER: Tybee Island is 5 hours of driving.' },
        { ts: '2023-05-20T10:00:00Z', text: 'USER: Asheville was a 6 hours drive for the road trip.' },
      ],
    );
    expect(notes).toMatch(/sum of.*15 hours/);
    const marathon = buildComputedNotes(
      'How many minutes did I exceed my target time by in the marathon?',
      '2023/05/30 (Tue) 22:30',
      [
        { ts: '2023-05-25T10:00:00Z', text: 'USER: My marathon time was 4 hours 22 minutes.' },
        { ts: '2023-05-20T10:00:00Z', text: 'USER: My marathon target time is 4 hours 10 minutes.' },
      ],
    );
    expect(marathon).toMatch(/difference.*12 minutes/);
  });

  it('is empty when the history has nothing datable or countable near the question', () => {
    expect(buildComputedNotes('What is my favourite colour?', '2023/05/30 (Tue) 19:37', [{ ts: '2023-05-10T10:00:00Z', text: 'USER: I like blue best.' }])).toBe('');
  });
});

describe('lessons from the first paired run', () => {
  it('keeps every paragraph of an assistant turn out of the user quantities', () => {
    const text = 'ASSISTANT: Some options:\n\n**Tustin**: homes in the $250,000-$350,000 range.\n\nUSER: I saw a house on 2/15 and loved it.';
    expect(extractQuantities(text)).toEqual([]);
    const events = resolveTemporalExpressions(text, '2022-03-02T10:00:00Z');
    expect(events.map((e) => e.iso)).toContain('2022-02-15');
  });

  it('offers a sum only when the question asks for a total', () => {
    const sources = [
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I raised $1,000 for charity at the bake sale.' },
      { ts: '2023-04-01T10:00:00Z', text: 'USER: I raised $2,750 for charity at the gala.' },
    ];
    expect(buildComputedNotes('How much money did I raise for charity in total?', '2023/05/30 (Tue) 19:37', sources)).toMatch(/sum of the 2 dollar figures above: 3750/);
    expect(buildComputedNotes('Which charity event did I enjoy more?', '2023/05/30 (Tue) 19:37', sources)).not.toMatch(/sum of/);
  });

  it('states the order of dated events for order questions', () => {
    const notes = buildComputedNotes(
      'Which event did I participate in first, the charity gala or the charity bake sale?',
      '2023/05/30 (Tue) 19:37',
      [
        { ts: '2023-03-28T10:00:00Z', text: 'USER: I am attending the charity gala tonight.' },
        { ts: '2023-03-20T10:00:00Z', text: 'USER: The charity bake sale was yesterday.' },
      ],
    );
    expect(notes).toMatch(/Order of the dated events.*2023-03-19.*→ 2023-03-28/);
  });
});

describe('lessons from the full paired run', () => {
  it('lists the gap between the events the question names first', () => {
    const notes = buildComputedNotes(
      "How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication'?",
      '2023/01/20 (Fri) 10:00',
      [
        { ts: '2023-01-05T10:00:00Z', text: 'USER: I renewed my gym membership yesterday.' },
        { ts: '2023-01-11T10:00:00Z', text: "USER: I attended the workshop on Effective Communication yesterday." },
        { ts: '2023-01-17T10:00:00Z', text: 'USER: The team meeting I was preparing for is today.' },
      ],
    );
    const firstGap = notes.split('Gaps between dated events')[1]!.split('\n')[1]!;
    expect(firstGap).toMatch(/2023-01-10.*2023-01-17.*7 days/);
  });

  it('leads with months for long distances', () => {
    const notes = buildComputedNotes('How many months ago did I attend the film festival?', '2021/10/02 (Sat) 10:00', [{ ts: '2021-06-01T10:00:00Z', text: 'USER: I saw Coda at the film festival today.' }]);
    expect(notes).toMatch(/about 4 months \(18 weeks, 123 days\) before the question date/);
  });
});

describe('lessons from the second paired run', () => {
  it('does not treat a date that names a thing as an event date', () => {
    const events = resolveTemporalExpressions('USER: I finally read the March 15th issue of The New Yorker today.', '2023-03-20T10:00:00Z');
    expect(events.map((e) => e.iso)).toEqual(['2023-03-20']);
  });

  it('says which side of an order question the history never dates', () => {
    const notes = buildComputedNotes(
      'Which task did I complete first, fixing the fence or purchasing three cows from Peter?',
      '2023/05/30 (Tue) 19:37',
      [
        { ts: '2023-05-21T10:00:00Z', text: 'USER: I fixed the broken fence on the east side three weeks ago.' },
        { ts: '2023-05-22T10:00:00Z', text: 'USER: The fence held up well yesterday in the storm.' },
      ],
    );
    expect(notes).toMatch(/Coverage: .*none match "purchasing cow peter"/);
  });
});

describe('roster for counting questions', () => {
  it('lists every user sentence naming the subject with its session date, once each', () => {
    const notes = buildComputedNotes(
      'How many movie festivals have I attended?',
      '2023/05/30 (Tue) 20:53',
      [
        { ts: '2023-03-10T10:00:00Z', text: 'USER: I went to the AFI film festival last year and loved the Q&A.\n\nASSISTANT: Festivals are great.' },
        { ts: '2023-04-02T10:00:00Z', text: 'USER: Sundance was cold but the film festival programme was superb.' },
        { ts: '2023-05-20T10:00:00Z', text: 'USER: Just back from the Austin Film Festival short film challenge. I went to the AFI film festival last year and loved the Q&A.' },
      ],
    );
    expect(notes).toContain('Sentences in the history that name the question\'s subject');
    expect(notes).toMatch(/2023-03-10: "I went to the AFI film festival/);
    expect(notes).toMatch(/2023-04-02: "Sundance was cold/);
    expect(notes).toMatch(/2023-05-20: "Just back from the Austin Film Festival/);
    const roster = notes.split('Sentences in the history')[1]!.split('Dated events')[0]!;
    expect((roster.match(/AFI film festival last year/g) ?? []).length).toBe(1);
  });
});
