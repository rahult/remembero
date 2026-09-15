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
    if (!/^### Retrieved session \d+ \(abstract: lines matching the question\)$/.test(lines[i]!)) continue;
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
      [4, 5, 6, 7, 8].map((n) => `### Retrieved session ${n} (abstract: lines matching the question)`),
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
      '### Retrieved session 2 (abstract: lines matching the question)\n' +
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

  it('readerMessages with a tiered contract renders what the builder renders', () => {
    const haystack: Haystack = {
      questionDate: '2023-07-10',
      sessions: sources.map((s) => ({ id: s.opId, date: s.ts.slice(0, 10), facts: s.facts, transcript: s.text })),
    };
    const contract = { ...READER_CONTRACT_V5, fullSessions: 3, abstractBytes: 320 };
    const distilled = readerMessages(haystack, QUESTION, 'multi-session', contract);
    const distillInstance = {
      ...instance(QUESTION),
      question_id: 'distill-multi-session',
      question_date: '2023/07/10 (Sat) 09:00',
    } as LongMemEvalInstance;
    const harness = buildLongMemEvalAnswerContext(
      distillInstance,
      haystack.sessions.map((s) => ({ opId: s.id, ts: `${s.date}T09:00:00.000Z`, text: s.transcript, facts: s.facts })),
      contract.contextBytes, [], 'direct', undefined,
      contract.dateDistances, contract.computedNotes, contract.focusedBudget, contract.structuredEvidence,
      { fullSessions: 3, abstractBytes: 320 },
    );
    expect(distilled).toEqual(harness.messages);
    expect(abstractSections(distilled[1]!.content)).toHaveLength(5);
  });
});
