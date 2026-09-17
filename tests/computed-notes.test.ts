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

describe('the question itself, seasons and anchored offsets', () => {
  it('resolves the question\'s own time reference against the question date and points at the matching event', () => {
    const notes = buildComputedNotes(
      'I received a piece of jewelry last Saturday from whom?',
      '2023/05/24 (Wed) 10:00',
      [
        { ts: '2023-05-21T10:00:00Z', text: 'USER: My aunt gave me a necklace yesterday, it was her mother\'s.' },
        { ts: '2023-04-02T10:00:00Z', text: 'USER: I received a crystal chandelier from my aunt last week.' },
      ],
    );
    expect(notes).toMatch(/The question's "last Saturday".*2023-05-20/);
    expect(notes).toMatch(/closest dated event.*2023-05-20/);
  });

  it('resolves seasons to an approximate date', () => {
    const events = resolveTemporalExpressions('USER: I went to Europe last summer and loved Lisbon.', '2023-05-01T10:00:00Z');
    const summer = events.find((e) => e.expression.toLowerCase() === 'last summer');
    expect(summer?.iso).toBe('2022-07-15');
    expect(summer?.approximate).toBe(true);
  });

  it('resolves offsets from holidays and from dates', () => {
    const events = resolveTemporalExpressions('USER: I went to the Holiday Market a week before Black Friday. I bought the phone two days after Christmas.', '2023-12-10T10:00:00Z');
    const byExpr = Object.fromEntries(events.map((e) => [e.expression.toLowerCase(), e.iso]));
    expect(byExpr['a week before black friday']).toBe('2023-11-17');
    expect(byExpr['two days after christmas']).toBe('2023-12-27');
  });
});

describe('rule 5: bugs from the v7 temporal misses', () => {
  it('quotes the best-matching sentence for a date, not the last one said that day', () => {
    // 2023-06-03 is a Saturday; both sentences are dated that day
    const notes = buildComputedNotes('I received a gift last Saturday from whom?', '2023/06/07 (Wed) 10:00', [
      {
        ts: '2023-06-03T10:00:00Z',
        text: 'USER: By the way, I also got a lovely wool scarf from my uncle today.\n\nUSER: By the way, I repainted the bookshelf today, so it is ready for the books.',
      },
    ]);
    expect(notes).toMatch(/closest dated event[^\n]*2023-06-03 \("[^"]*wool scarf/);
  });

  it('moves an assumed-year date back a year when a past-tense sentence would put it after the session', () => {
    const past = resolveTemporalExpressions('USER: I have been busy, like the River Run I did on May 15th, and I would love to do more.', '2022-02-19T10:00:00Z');
    expect(past.map((e) => e.iso)).toEqual(['2021-05-15']);
    expect(past[0]?.yearRolledBack).toBe(true);
    const already = resolveTemporalExpressions("USER: I'd like to learn about the charities I've already supported, like the River Run on May 15th.", '2022-02-19T10:00:00Z');
    expect(already.map((e) => e.iso)).toEqual(['2021-05-15']);
    const slash = resolveTemporalExpressions('USER: We have been busy since we started on 6/20.', '2023-05-29T10:00:00Z');
    expect(slash.map((e) => e.iso)).toEqual(['2022-06-20']);
  });

  it('keeps an assumed-year date in the session year when the sentence is not past tense', () => {
    const plain = resolveTemporalExpressions('USER: The River Run is on May 15th and I trained hard yesterday.', '2022-02-19T10:00:00Z');
    expect(plain.map((e) => e.iso)).toContain('2022-05-15');
    const tickets = resolveTemporalExpressions('USER: I bought tickets for the concert on May 15th.', '2022-02-19T10:00:00Z');
    expect(tickets.map((e) => e.iso)).toEqual(['2022-05-15']);
    expect(tickets[0]?.yearRolledBack).toBeUndefined();
  });
});
