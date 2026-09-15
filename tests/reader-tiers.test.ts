import { describe, expect, it } from 'vitest';
import { buildLongMemEvalAnswerContext } from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import { READER_CONTRACT_V5 } from '../src/evals/reader-contract.js';
import { readerMessages, type Haystack } from '../src/training/reader-distill.js';

const instance = (question: string) =>
  ({
    question_id: 'tiers',
    question_type: 'multi-session',
    question,
    question_date: '2023/07/10 (Mon) 09:00',
    answer: '',
    haystack_session_ids: [],
    haystack_dates: [],
    haystack_sessions: [],
    answer_session_ids: [],
  }) as unknown as LongMemEvalInstance;

const QUESTION = 'Which guitar did I buy before the concert?';

/** Eight sessions, ranked s1..s8, dated out of rank order, each long enough to be windowed. */
const sources = Array.from({ length: 8 }, (_, index) => {
  const n = index + 1;
  const filler = `I reorganised shelf ${n} in the pantry and labelled the jars. `.repeat(60);
  return {
    opId: `s${n}`,
    ts: `2023-0${(n % 6) + 1}-1${n}T09:00:00.000Z`,
    text:
      `USER: I bought a Fender guitar in session ${n} on 3 May. The weather was grey. ${filler}My guitar teacher booked a concert for 20 May.\n` +
      `ASSISTANT: A guitar concert sounds wonderful, and the weather should clear.\n` +
      `USER: I also walked the dog.`,
    facts: n === 2 ? ['owns(user, fender_guitar).'] : [],
  };
});

function userContent(ctx: { messages: Array<{ content: string }> }): string {
  return ctx.messages.at(-1)!.content;
}

/** Every abstract section in the prompt: header line through the body line, with its newline. */
function abstractSections(text: string): string[] {
  const lines = text.split('\n');
  const sections: string[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (!/^### Retrieved session \d+ \(abstract\)$/.test(lines[i]!)) continue;
    const block: string[] = [];
    for (let j = i; j < lines.length && lines[j] !== ''; j += 1) block.push(lines[j]!);
    sections.push(`${block.join('\n')}\n`);
  }
  return sections;
}

describe('context tiers', () => {
  it('renders the top-ranked sessions in full and the rest as abstracts within their byte cap', () => {
    const tiered = buildLongMemEvalAnswerContext(
      instance(QUESTION), sources, 24576, [], 'direct', undefined,
      true, true, false, false, { fullSessions: 3, abstractBytes: 320 },
    );
    const text = userContent(tiered);
    const full = text.match(/^### Retrieved session \d+$/gm) ?? [];
    expect(full.sort()).toEqual(['### Retrieved session 1', '### Retrieved session 2', '### Retrieved session 3']);
    const abstracts = abstractSections(text);
    expect(abstracts).toHaveLength(5);
    for (const section of abstracts) expect(Buffer.byteLength(section, 'utf8')).toBeLessThanOrEqual(320);
    expect(abstracts.map((s) => s.split('\n')[0]).sort()).toEqual(
      [4, 5, 6, 7, 8].map((n) => `### Retrieved session ${n} (abstract)`),
    );
    // the full sessions get more than the even split's share once the abstracts are paid for
    const even = buildLongMemEvalAnswerContext(instance(QUESTION), sources, 24576, [], 'direct', undefined, true, true);
    expect(text.split('shelf 1 ').length).toBeGreaterThan(userContent(even).split('shelf 1 ').length);
    // total history stays within the budget
    const history = text.slice(text.indexOf('History chats:'), text.indexOf('\n### Computed') === -1 ? text.indexOf('Current date:') : text.indexOf('\n### Computed'));
    expect(Buffer.byteLength(history, 'utf8')).toBeLessThanOrEqual(24576 + 8 * 64);
    expect(tiered.contextSessionIds).toEqual(sources.map((s) => s.opId));
  });

  it('an abstract keeps only the user sentences that name the question, never assistant text', () => {
    const ranked = [
      { opId: 'top', ts: '2023-05-01T09:00:00.000Z', text: 'USER: Nothing here.' },
      {
        opId: 'second',
        ts: '2023-05-02T09:00:00.000Z',
        text:
          'user: I bought a Fender guitar last week. The weather is nice today! My guitar teacher is Sam.\n' +
          'assistant: Guitar lessons with a concert teacher are great.\n' +
          'user: We will go to the concert together?',
      },
    ];
    const text = userContent(
      buildLongMemEvalAnswerContext(instance(QUESTION), ranked, 24576, [], 'direct', undefined,
        false, false, false, false, { fullSessions: 1, abstractBytes: 320 }),
    );
    const [section] = abstractSections(text);
    expect(section).toBe(
      '### Retrieved session 2 (abstract)\n' +
        'Session date: 2023-05-02T09:00:00.000Z\n' +
        'I bought a Fender guitar last week. My guitar teacher is Sam. We will go to the concert together?\n',
    );
  });

  it('cuts an abstract at a sentence boundary, and falls back to the first user sentence when none match', () => {
    const long = Array.from({ length: 12 }, (_, i) => `Guitar note number ${i} is here.`).join(' ');
    const ranked = [
      { opId: 'top', ts: '2023-05-01T09:00:00.000Z', text: 'USER: Nothing here.' },
      { opId: 'cut', ts: '2023-05-02T09:00:00.000Z', text: `USER: ${long}` },
      { opId: 'none', ts: '2023-05-03T09:00:00.000Z', text: 'ASSISTANT: The guitar is fine.\nUSER: I walked the dog. Then I slept.' },
    ];
    const text = userContent(
      buildLongMemEvalAnswerContext(instance(QUESTION), ranked, 24576, [], 'direct', undefined,
        false, false, false, false, { fullSessions: 1, abstractBytes: 160 }),
    );
    const [cut, none] = abstractSections(text);
    expect(Buffer.byteLength(cut!, 'utf8')).toBeLessThanOrEqual(160);
    expect(cut!.trimEnd().split('\n').at(-1)).toMatch(/^Guitar note number 0 is here\.( Guitar note number \d+ is here\.)*$/);
    expect(none!.trimEnd().split('\n').at(-1)).toBe('I walked the dog.');
  });

  it('is byte-identical to the even split when every session fits in the full tier', () => {
    const three = sources.slice(0, 3);
    const even = buildLongMemEvalAnswerContext(instance(QUESTION), three, 6144, [], 'direct', undefined, true, true);
    for (const fullSessions of [3, 5]) {
      const tiered = buildLongMemEvalAnswerContext(instance(QUESTION), three, 6144, [], 'direct', undefined,
        true, true, false, false, { fullSessions, abstractBytes: 320 });
      expect(tiered).toEqual(even);
    }
  });

  it('leaves the computed-notes block byte-identical between even split and tiers', () => {
    const q = 'How many days passed between buying the guitar and the concert?';
    const even = userContent(buildLongMemEvalAnswerContext(instance(q), sources, 24576, [], 'direct', undefined, true, true));
    const tiered = userContent(buildLongMemEvalAnswerContext(instance(q), sources, 24576, [], 'direct', undefined,
      true, true, false, false, { fullSessions: 2, abstractBytes: 320 }));
    const notes = (text: string) => text.slice(text.lastIndexOf('\n', text.indexOf('Computed from the history')));
    expect(tiered).toContain('Computed from the history');
    expect(notes(tiered)).toBe(notes(even));
  });

  it('refuses tiers combined with the focused budget', () => {
    expect(() =>
      buildLongMemEvalAnswerContext(instance(QUESTION), sources, 24576, [], 'direct', undefined,
        true, true, true, false, { fullSessions: 3, abstractBytes: 320 }),
    ).toThrow(/focused budget/);
  });

  it('readerMessages refuses a tiered contract until distillation orders sessions by rank', () => {
    const haystack: Haystack = {
      questionDate: '2023-07-10',
      sessions: sources.map((s) => ({ id: s.opId, date: s.ts.slice(0, 10), facts: s.facts, transcript: s.text })),
    };
    const contract = { ...READER_CONTRACT_V5, fullSessions: 3, abstractBytes: 320 };
    expect(() => readerMessages(haystack, QUESTION, 'multi-session', contract)).toThrow(
      'tiered contracts need rank-ordered haystacks; see plan Task 5',
    );
    expect(() => readerMessages(haystack, QUESTION, 'multi-session', READER_CONTRACT_V5)).not.toThrow();
  });

  const abstractBody = (question: string, userText: string, abstractBytes = 320, facts: string[] = []) => {
    const ranked = [
      { opId: 'top', ts: '2023-05-01T09:00:00.000Z', text: 'USER: Nothing here.' },
      { opId: 'abs', ts: '2023-05-02T09:00:00.000Z', text: userText, facts },
    ];
    const text = userContent(
      buildLongMemEvalAnswerContext(instance(question), ranked, 24576, [], 'direct', undefined,
        false, false, false, false, { fullSessions: 1, abstractBytes }),
    );
    const [section] = abstractSections(text);
    return section!;
  };
  const lastLine = (section: string) => section.trimEnd().split('\n').at(-1);

  it('a sentence that matches only filler words loses to one that matches a content word', () => {
    const section = abstractBody(
      'How many hours have I spent playing the piano since January?',
      'USER: I have so many things to do this week and need a break from work since Monday. My piano practice took two hours on Sunday.',
      150,
    );
    expect(lastLine(section)).toBe('My piano practice took two hours on Sunday.');
  });

  it('skips a top-ranked sentence that does not fit in favour of a lower one that does', () => {
    const long = `The piano recital in the grand concert hall ran late with a long encore and ${'several '.repeat(20)}more piano pieces.`;
    const section = abstractBody(
      'When was the piano recital at the concert hall?',
      `USER: ${long} I booked the piano recital.`,
      160,
    );
    expect(lastLine(section)).toBe('I booked the piano recital.');
  });

  it('matches whole canonical words, not substrings', () => {
    const hit = abstractBody('Where did I learn about germs?', 'USER: We toured Germany. I read about germs in class.');
    expect(lastLine(hit)).toBe('I read about germs in class.');
    // "germ" is inside "Germany" but is not its word: no match, so the first user sentence
    const none = abstractBody('How do germs spread?', 'USER: It was cold. We toured Germany.');
    expect(lastLine(none)).toBe('It was cold.');
  });

  it('does not split a sentence after an abbreviation', () => {
    const section = abstractBody(
      'Where did I see the cherry blossoms?',
      'USER: It rained. I saw the cherry blossoms in Washington D.C. last week. Dr. Lee said the blossoms peak early.',
    );
    expect(lastLine(section)).toBe(
      'I saw the cherry blossoms in Washington D.C. last week. Dr. Lee said the blossoms peak early.',
    );
  });

  it('still splits after ordinary short words, and after the word that follows a title', () => {
    expect(lastLine(abstractBody('Who is my piano teacher?', 'USER: I love it. My piano teacher said so.'))).toBe(
      'My piano teacher said so.',
    );
    expect(lastLine(abstractBody('What did I ask about pickup?', 'USER: Please pick me up. Then the pickup is done.'))).toBe(
      'Then the pickup is done.',
    );
    expect(lastLine(abstractBody('Who is Chen?', 'USER: The weather is ok. Dr. Chen called. She was kind.'))).toBe(
      'Dr. Chen called.',
    );
    expect(lastLine(abstractBody('Where in Washington did I drive?', 'USER: It rained. We drove to Washington D.C. last week.'))).toBe(
      'We drove to Washington D.C. last week.',
    );
    expect(lastLine(abstractBody('Which recipes use basil?', 'USER: Herbs, e.g. basil and mint, grow fast. Mr. Park sells basil. So do I.'))).toBe(
      'Herbs, e.g. basil and mint, grow fast. Mr. Park sells basil.',
    );
  });

  it('abstracts carry no facts line; full sections keep theirs', () => {
    const section = abstractBody('Which guitar did I buy?', 'USER: I bought a guitar.', 320, ['owns(user, guitar).']);
    expect(section).toBe('### Retrieved session 2 (abstract)\nSession date: 2023-05-02T09:00:00.000Z\nI bought a guitar.\n');
    const text = userContent(
      buildLongMemEvalAnswerContext(instance('Which guitar did I buy?'),
        [{ opId: 'f', ts: '2023-05-02T09:00:00.000Z', text: 'USER: I bought a guitar.', facts: ['owns(user, guitar).'] }],
        24576, [], 'direct', undefined, false, false, false, false, { fullSessions: 1, abstractBytes: 320 }),
    );
    expect(text).toContain('Remembered facts (stated in this session): owns(user, guitar).');
  });

  it('refuses abstracts that would leave the full sessions less than 256 bytes each', () => {
    const many = Array.from({ length: 15 }, (_, i) => ({
      opId: `m${i}`,
      ts: '2023-05-02T09:00:00.000Z',
      text: `USER: ${Array.from({ length: 80 }, (_, j) => `My guitar note ${i}-${j} is here.`).join(' ')}`,
    }));
    expect(() =>
      buildLongMemEvalAnswerContext(instance(QUESTION), many, 4096, [], 'direct', undefined,
        false, false, false, false, { fullSessions: 2, abstractBytes: 2048 }),
    ).toThrow(/abstract sections take \d+ bytes, more than the 3584 bytes/);
  });
});
