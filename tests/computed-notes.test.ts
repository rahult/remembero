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

describe('rule 3: which sentences count as relevant', () => {
  it('keeps short capitalised names and reads met as meet', () => {
    const notes = buildComputedNotes('When did I meet Jo?', '2023/05/28 (Sun) 10:00', [
      { ts: '2023-05-28T10:00:00Z', text: 'USER: A few months ago, I met a lot of people at a fair, including a potter named Jo.' },
    ]);
    expect(notes).toMatch(/2023-02-2\d: "[^"]*named Jo/);
    const sf = buildComputedNotes('When was I in SF?', '2023/05/28 (Sun) 10:00', [{ ts: '2023-05-28T10:00:00Z', text: 'USER: I was in SF exactly two weeks ago.' }]);
    expect(sf).toContain('2023-05-14');
  });

  it('matches whole words, so a question word inside a longer word does not pull a sentence in', () => {
    expect(buildComputedNotes('What did I do with my old cart?', '2023/05/28 (Sun) 10:00', [{ ts: '2023-05-28T10:00:00Z', text: 'USER: I watched a cartoon yesterday.' }])).toBe('');
  });

  it('keeps a dated sentence that refers back to a named thing with "the <noun>" in the same turn', () => {
    const turn = 'USER: I need a sleeve for my new tablet, Lenovo Yoga, and a strap for my camera. By the way, I ordered the tablet on April 2nd.';
    const notes = buildComputedNotes('When did I get the Lenovo Yoga?', '2023/05/28 (Sun) 10:00', [{ ts: '2023-05-10T10:00:00Z', text: turn }]);
    expect(notes).toMatch(/2023-04-02: "[^"]*ordered the tablet/);
    // another turn does not carry the link
    const apart = buildComputedNotes('When did I get the Lenovo Yoga?', '2023/05/28 (Sun) 10:00', [
      { ts: '2023-05-10T10:00:00Z', text: 'USER: I need a sleeve for my new tablet, Lenovo Yoga.\n\nASSISTANT: Sure.\n\nUSER: By the way, I ordered the tablet on April 2nd.' },
    ]);
    expect(apart).not.toContain('2023-04-02');
  });

  it('only says the history never mentions a side when no user turn names it', () => {
    const notes = buildComputedNotes('Which did I buy first, the desk lamp or the Ergo chair?', '2023/05/30 (Tue) 19:37', [
      { ts: '2023-05-21T10:00:00Z', text: 'USER: I bought the desk lamp three weeks ago.\n\nUSER: I love my Ergo chair so much.' },
    ]);
    expect(notes).not.toMatch(/none match "ergo chair"/);
    expect(notes).toMatch(/"ergo chair" is named in the history, but no sentence naming it carries a date/);
  });
});

describe('rule 2: dates for events stated without one', () => {
  it('dates a past event with no date expression by its session, and says so', () => {
    const notes = buildComputedNotes('When did I cancel my gym membership?', '2023/03/18 (Sat) 10:00', [
      { ts: '2023-02-01T10:00:00Z', text: "USER: By the way, I'm glad I cancelled my gym membership at FitWorks, it was too pricey." },
    ]);
    expect(notes).toMatch(/- 2023-02-01: "[^"]*cancelled my gym membership[^"]*" \[said 2023-02-01; no date stated/);
    const plan = buildComputedNotes('When did I cancel my gym membership?', '2023/03/18 (Sat) 10:00', [
      { ts: '2023-02-01T10:00:00Z', text: "USER: I'm planning to cancel my gym membership soon." },
    ]);
    expect(plan).toBe('');
  });

  it('turns "for N units now" into an approximate start date, only in the extended reading', () => {
    const text = "USER: By the way, I've been getting into pottery for about four months now. I went to Spain for two weeks.";
    const events = resolveTemporalExpressions(text, '2023-05-01T10:00:00Z', { extended: true });
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ iso: '2023-01-01', start: true, approximate: true, expression: 'for about four months now' });
    expect(resolveTemporalExpressions(text, '2023-05-01T10:00:00Z')).toEqual([]);
    const past = resolveTemporalExpressions("USER: I've been living in Canada for the past three years.", '2023-05-27T10:00:00Z', { extended: true });
    expect(past.map((e) => e.iso)).toEqual(['2020-05-27']);
  });

  it('gives the age at an event from a stated age and a start date', () => {
    const notes = buildComputedNotes('How old was I when I moved to Canada?', '2023/05/27 (Sat) 10:00', [
      {
        ts: '2023-05-27T10:00:00Z',
        text: "USER: I've been living in Canada for the past three years on a work permit.\n\nUSER: I'm a 40-year-old nurse, and I want to apply for residency.",
      },
    ]);
    expect(notes).toMatch(/^Age: [^\n]*40 on 2023-05-27[^\n]*about 37\b/m);
  });

  it('reads a month without a day as a month, in the year the tense points to', () => {
    const at = '2023-03-17T10:00:00Z';
    const one = (text: string) => resolveTemporalExpressions(`USER: ${text}`, at, { extended: true });
    expect(one('My cousin Dana adopted a puppy in January.')).toMatchObject([{ iso: '2023-01-01', monthOnly: true, kind: 'month' }]);
    expect(one('We visited Rome in November.').map((e) => e.iso)).toEqual(['2022-11-01']);
    expect(one("I'm flying to Lisbon in June.").map((e) => e.iso)).toEqual(['2023-06-01']);
    expect(one('I may go to the lake in the spring.')).toEqual([]);
    expect(one('I went there on January 10th.').map((e) => e.iso)).toEqual(['2023-01-10']);
    const notes = buildComputedNotes('When did Dana adopt the puppy?', '2023/03/17 (Fri) 10:00', [{ ts: at, text: 'USER: My cousin Dana adopted a puppy in January.' }]);
    expect(notes).toMatch(/- 2023-01 \(month only\): /);
  });

  it('counts "N units before <event>" from that event, never from the session', () => {
    const at = '2023-03-15T10:00:00Z';
    const chained = resolveTemporalExpressions('USER: I got my new phone on March 3rd. I bought a phone case a week before I got my new phone.', at);
    const caseEvent = chained.find((e) => e.expression.startsWith('a week before'));
    expect(caseEvent?.iso).toBe('2023-02-24');
    expect(caseEvent?.anchor?.iso).toBe('2023-03-03');
    // nothing to count from: no date at all, rather than "a week ago"
    expect(resolveTemporalExpressions('USER: I bought a charger a week before I started my trip.', at)).toEqual([]);
    // still "ago" when nothing follows
    expect(resolveTemporalExpressions('USER: I fixed it two days before, luckily.', at).map((e) => e.iso)).toEqual(['2023-03-13']);
  });

  it('counts "N units in advance" from the event the sentence books for', () => {
    const events = resolveTemporalExpressions(
      "USER: I stayed at a cabin for my sister's graduation and had to book two months in advance.\n\nUSER: I went to my sister's graduation exactly one month ago.",
      '2023-06-01T10:00:00Z',
    );
    const booking = events.find((e) => e.expression === 'two months in advance');
    expect(booking).toMatchObject({ iso: '2023-03-01', approximate: true });
    expect(booking?.anchor?.iso).toBe('2023-05-01');
  });

  it('looks for the anchor in the other sessions when the chained sentence\'s own session has none', () => {
    const notes = buildComputedNotes('When did I book the cabin?', '2023/06/01 (Thu) 20:00', [
      { ts: '2023-06-01T18:00:00Z', text: "USER: I stayed at a cabin for my sister's graduation and had to book two months in advance." },
      { ts: '2023-06-01T09:00:00Z', text: "USER: I went to my sister's graduation exactly one month ago." },
    ]);
    expect(notes).toMatch(/- 2023-03-01: "[^"]*book two months in advance\.?" \[[^\]]*counted from 2023-05-01/);
  });
});

describe('rule 1: an ordering verdict for which-came-first questions', () => {
  const at = '2023-06-10T10:00:00Z';
  const verdictOf = (notes: string) => notes.split('\n').find((l) => l.startsWith('Which came first:')) ?? '';

  it('ties each alternative to the clause holding its date and states the order first', () => {
    const notes = buildComputedNotes('Which project did I start first, the sailboat model or the lighthouse puzzle?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: "USER: By the way, I'm currently painting a sailboat model, and in the evenings I'm also assembling a lighthouse puzzle, which I started about a month ago." },
      { ts: at, text: 'USER: I started the sailboat model about three weeks ago on a rainy Sunday.' },
    ]);
    expect(notes.split('\n')[1]).toMatch(/^Which came first:/);
    expect(verdictOf(notes)).toMatch(/"lighthouse puzzle" ≈2023-05-10 [^\n]*is earlier than "sailboat model" 2023-05-20[^\n]*→ lighthouse puzzle came first/);
  });

  it('calls the order too close only when the two rough date ranges overlap', () => {
    const apart = buildComputedNotes('Which happened first, losing my umbrella or getting my raincoat?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: 'USER: I lost my umbrella at the station about three weeks ago.\n\nUSER: I got my raincoat about a month ago.' },
    ]);
    expect(verdictOf(apart)).toMatch(/→ getting my raincoat came first/);
    const close = buildComputedNotes('Which happened first, losing my umbrella or getting my raincoat?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: 'USER: I lost my umbrella at the station about two months ago.\n\nUSER: I got my raincoat nine weeks ago.' },
    ]);
    expect(verdictOf(close)).toMatch(/too close to order/);
    expect(verdictOf(close)).not.toMatch(/→ .* came first/);
  });

  it('follows "the tablet" to the name set beside it and prefers the date whose verb the question asks about', () => {
    const notes = buildComputedNotes('Which device did I get first, the Lenovo Yoga or the Pixel 7?', '2023/05/01 (Mon) 20:00', [
      {
        ts: '2023-05-01T10:00:00Z',
        text: 'USER: I need cases for my new tablet, Lenovo Yoga, and my new phone, Pixel 7. By the way, I ordered the tablet on April 2nd, and it finally arrived on April 20th after a delay from the expected date of April 9th.\n\nUSER: I got the Pixel 7 at the mall on April 12th.',
      },
    ]);
    expect(verdictOf(notes)).toMatch(/"Pixel 7" 2023-04-12 [^\n]*is earlier than "Lenovo Yoga" 2023-04-20 [^\n]*→ Pixel 7 came first/);
    expect(verdictOf(notes)).toMatch(/2023-04-02/);  // the other date is shown, not hidden
  });

  it('works with short names and "met"', () => {
    const notes = buildComputedNotes('Who did I meet first, Priya or Jo?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: 'USER: I met Priya at a book club about two weeks ago.\n\nUSER: A few months ago, I met a lot of people at a fair, including a potter named Jo.' },
    ]);
    expect(verdictOf(notes)).toMatch(/→ Jo came first/);
  });

  it('reads "before or after" questions', () => {
    const notes = buildComputedNotes('Did I repaint the fence before or after the garden party?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: 'USER: I repainted the fence last Saturday.\n\nUSER: The garden party was two weeks ago.' },
    ]);
    expect(verdictOf(notes)).toMatch(/→ garden party came first/);
  });

  it('gives no verdict from a clause that names both alternatives', () => {
    const notes = buildComputedNotes('Which did I buy first, the kettle or the toaster?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: 'USER: I bought the kettle and the toaster about a month ago.\n\nUSER: I bought the toaster two weeks ago.' },
    ]);
    expect(verdictOf(notes)).toBe('');
  });

  it('quotes the clause that holds the date when a sentence is long', () => {
    const notes = buildComputedNotes('When did I start the lighthouse puzzle?', '2023/06/10 (Sat) 20:00', [
      { ts: at, text: "USER: I've been thinking a lot about how hobbies relate to my own week at work and at home, and I started assembling a lighthouse puzzle today, which got me thinking about patience." },
    ]);
    expect(notes).toMatch(/- 2023-06-10: "…?I started assembling a lighthouse puzzle today/);
  });
});

describe("rule 4: the question's own gap comes first", () => {
  const first = (notes: string) => notes.split('\n')[1] ?? '';

  it('states the gap between the two events the question names, first, where the character limit cannot cut it', () => {
    const notes = buildComputedNotes(
      'How many days did it take me to finish the quilt after buying the fabric?',
      '2022/04/25 (Mon) 10:00',
      [
        {
          ts: '2022-04-25T10:00:00Z',
          text: "USER: By the way, I started writing poems again about two months ago, after a long break.\n\nUSER: I went to a talk on quilting history last week, and the speaker was great.\n\nUSER: Since I bought the fabric on 4/2, I've been sewing every night.\n\nUSER: I finished the quilt on 4/20, and it looks lovely.",
        },
      ],
      { maxChars: 420 },
    );
    expect(first(notes)).toMatch(/^Gap the question asks for: "buying the fabric" 2022-04-02 [^\n]*to "finish the quilt" 2022-04-20 [^\n]*= 18 days \(19 counting both/);
  });

  it('gives the gap in the unit the question asks for', () => {
    const notes = buildComputedNotes('How many weeks passed between the day I adopted my cat and the day I took her to the vet?', '2023/04/01 (Sat) 10:00', [
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I adopted my cat today, she is tiny.' },
      { ts: '2023-03-19T10:00:00Z', text: 'USER: I took my cat to the vet today for her shots.' },
    ]);
    expect(first(notes)).toMatch(/= 2\.6 weeks \(18 days\)/);
    const ago = buildComputedNotes('How many months ago did I book the cabin?', '2023/06/01 (Thu) 20:00', [
      { ts: '2023-06-01T18:00:00Z', text: "USER: I stayed at a cabin for my sister's graduation and had to book two months in advance." },
      { ts: '2023-06-01T09:00:00Z', text: "USER: I went to my sister's graduation exactly one month ago." },
    ]);
    expect(first(ago)).toMatch(/^Gap the question asks for: "book the cabin" ≈2023-03-01 [^\n]*to the question date 2023-06-01 = about 3 months \(3\.0 months, 92 days\)/);
  });

  it('uses an event told without a date, and prefers the sentence with the side\'s own words', () => {
    const notes = buildComputedNotes('How many days passed between the day I cancelled my gym membership and the day I joined the climbing club?', '2023/03/18 (Sat) 10:00', [
      { ts: '2023-02-01T10:00:00Z', text: "USER: I'm glad I cancelled my gym membership at FitWorks.\n\nUSER: I want to try the new climbing gym I discovered last month." },
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I joined the climbing club today.' },
    ]);
    expect(first(notes)).toMatch(/^Gap the question asks for: "cancelled my gym membership" on or before 2023-02-01 \(no date stated\) [^\n]*to "joined the climbing club" 2023-03-01 [^\n]*= 28 days/);
  });

  it('reads "how long had I been X when Y" from the start of X', () => {
    const notes = buildComputedNotes('How long had I been doing pottery when I went to the pottery fair?', '2023/05/01 (Mon) 10:00', [
      { ts: '2023-05-01T10:00:00Z', text: "USER: I've been doing pottery for about four months now.\n\nUSER: I went to a pottery fair two weeks ago." },
    ]);
    expect(first(notes)).toMatch(/^Gap the question asks for: "doing pottery" ≈2023-01-01 [^\n]*to "went to the pottery fair" 2023-04-17 [^\n]*= about 3 months \(3\.5 months, 15\.1 weeks, 106 days\)/);
  });

  it('quotes the words that tie a long sentence to its side, and gives a long "how long" in months too', () => {
    const notes = buildComputedNotes('How long had I been knitting when I joined the yarn club?', '2023/05/21 (Sun) 10:00', [
      {
        ts: '2023-05-21T10:00:00Z',
        text: "USER: By the way, I've been getting into knitting for about three months now, and it's been really relaxing so far.\n\nUSER: By the way, I finally joined the local yarn club at the community center downtown, and paid the fee a month ago.",
      },
    ]);
    expect(first(notes)).toMatch(/"joined the yarn club" ≈2023-04-21 \("[^"]*yarn club[^"]*a month ago/);
    expect(first(notes)).toMatch(/= about 2 months \(1\.9 months, 8\.4 weeks, 59 days\)/);
  });

  it('says nothing when a side matches two sentences with different dates equally well', () => {
    const notes = buildComputedNotes('How many days passed between my dentist visit and my eye exam?', '2023/03/18 (Sat) 10:00', [
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I had a dentist visit today.\n\nUSER: My eye exam was yesterday.' },
      { ts: '2023-03-10T10:00:00Z', text: 'USER: Another dentist visit today, sadly.' },
    ]);
    expect(notes).not.toMatch(/Gap the question asks for/);
  });
});

describe('safeguards found by rebuilding the block for every LongMemEval question', () => {
  const at = '2023-03-10T10:00:00Z';

  it('does not stem "care" into "car"', () => {
    const notes = buildComputedNotes('Which vehicle did I take care of first, the bike or the car?', '2023/03/10 (Fri) 10:00', [
      { ts: at, text: "USER: I'm glad I got my bike repaired back in mid-February.\n\nUSER: I got my flu shot on March 1st, which reminded me how important it is to take care of my health." },
    ]);
    expect(notes).not.toMatch(/"car" [^\n]*flu shot/);
  });

  it('reads "how many days ago did I X when I Y" as the gap from X to Y', () => {
    const notes = buildComputedNotes('How many days ago did I launch my blog when I signed my first sponsor?', '2023/03/25 (Sat) 10:00', [
      { ts: '2023-02-10T10:00:00Z', text: 'USER: I launched my blog today, finally!' },
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I signed my first sponsor today.' },
    ]);
    expect(notes.split('\n')[1]).toMatch(/^Gap the question asks for: "launch my blog" 2023-02-10 [^\n]*to "signed my first sponsor" 2023-03-01 [^\n]*= 19 days/);
  });

  it('counts a capitalised question word only where the sentence capitalises it too', () => {
    const notes = buildComputedNotes('How many days passed between my visit to the Museum of Modern Art and the Ancient Coins exhibit?', '2023/02/01 (Wed) 10:00', [
      { ts: '2023-01-15T10:00:00Z', text: 'USER: I visit a museum in the city and see a great show on modern art today.\n\nUSER: I attended the Ancient Coins exhibit today.' },
      { ts: '2023-01-08T10:00:00Z', text: 'USER: I just got back from a tour at the Museum of Modern Art.' },
    ]);
    expect(notes).not.toMatch(/Gap the question asks for/);
  });

  it('drops a side whose best sentence matches the other side as well, and a gap whose order contradicts the question', () => {
    const mixed = buildComputedNotes("How many days before my sister's birthday party did I order her gift?", '2022/05/15 (Sun) 10:00', [
      { ts: '2022-05-15T10:00:00Z', text: "USER: I recently ordered a photo album gift for my sister's birthday.\n\nUSER: My sister's birthday party was on the 22nd of April." },
    ]);
    expect(mixed).not.toMatch(/Gap the question asks for/);
    const backwards = buildComputedNotes('How many days before the concert did I buy the tickets?', '2022/05/15 (Sun) 10:00', [
      { ts: '2022-05-15T10:00:00Z', text: 'USER: The concert was on April 2nd.\n\nUSER: I bought the tickets on April 9th.' },
    ]);
    expect(backwards).not.toMatch(/Gap the question asks for/);
  });

  it('gives no two-way verdict for a question about three events or their ranks', () => {
    const three = buildComputedNotes('Which three events happened in the order from first to last: the day I painted the shed, the day I planted the roses, and the day I fixed the gate?', '2023/03/10 (Fri) 10:00', [
      { ts: at, text: 'USER: I planted the roses two weeks ago.\n\nUSER: I fixed the gate three weeks ago.\n\nUSER: I painted the shed a week ago.' },
    ]);
    expect(three).not.toMatch(/Which came first/);
    const ranks = buildComputedNotes('Who graduated first, second and third among Ana, Ben and Cy?', '2023/03/10 (Fri) 10:00', [
      { ts: at, text: 'USER: Ana graduated two weeks ago.\n\nUSER: Ben and Cy graduated a week ago.' },
    ]);
    expect(ranks).not.toMatch(/Which came first/);
  });

  it('says which way an undated end can move the gap', () => {
    const notes = buildComputedNotes('How many days passed between the day I cancelled my gym membership and the day I joined the climbing club?', '2023/03/18 (Sat) 10:00', [
      { ts: '2023-02-01T10:00:00Z', text: "USER: I'm glad I cancelled my gym membership at FitWorks." },
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I joined the climbing club today.' },
    ]);
    expect(notes.split('\n')[1]).toMatch(/the earlier end was told without a date[^\n]*so the real gap may be longer/);
  });
});

describe('more safeguards from the full rebuild', () => {
  const ext = (text: string, at: string) => resolveTemporalExpressions(`USER: ${text}`, at, { extended: true });

  it('reads the tense of a month-only date from the words before it', () => {
    const at = '2022-03-09T10:00:00Z';
    expect(ext("I'm tracking races in California, as I'll be heading there for an event in August.", at).map((e) => e.iso)).toEqual(['2022-08-01']);
    expect(ext("I'm planning a trip to Tokyo in mid-April and I need ideas.", '2023-05-25T10:00:00Z').map((e) => e.iso)).toEqual(['2024-04-01']);
    expect(ext("Since I've been getting fresh produce since late July, I'm looking forward to more.", at).map((e) => e.iso)).toEqual(['2021-07-01']);
  });

  it('takes a start date only from a present-perfect sentence that is not a question', () => {
    const at = '2023-03-10T10:00:00Z';
    expect(ext('I had been training for six months before my first marathon.', at)).toEqual([]);
    expect(ext('Can you help me track my costs for the past few months?', at)).toEqual([]);
    expect(ext("I've had my cat, Luna, for about 9 months now.", at).map((e) => e.iso)).toEqual(['2022-06-10']);
  });

  it('takes the question\'s event verb from after "I", not from an adjective', () => {
    const notes = buildComputedNotes('How many weeks passed between the time I sold homemade baked goods and the time I joined the choir?', '2023/03/10 (Fri) 10:00', [
      { ts: '2023-02-26T10:00:00Z', text: 'USER: I\'m thinking of offering a bundle of my seasonal baked goods at the market.' },
    ]);
    expect(notes).not.toMatch(/no date stated/);
  });

  it('writes "1 day"', () => {
    const notes = buildComputedNotes('How many days passed between the day I adopted my cat and the day I took her to the vet?', '2023/04/01 (Sat) 10:00', [
      { ts: '2023-03-01T10:00:00Z', text: 'USER: I adopted my cat today, she is tiny.' },
      { ts: '2023-03-02T10:00:00Z', text: 'USER: I took my cat to the vet today for her shots.' },
    ]);
    expect(notes.split('\n')[1]).toMatch(/= 1 day \(2 counting both/);
  });
});
