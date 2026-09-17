import { describe, expect, it } from 'vitest';
import {
  READER_CONTRACT_V5,
  THINKING_TYPES,
  assertDistillReading,
  readingFor,
} from '../src/evals/reader-contract.js';
import { assertRerenderContract } from '../src/training/reader-rerender.js';
import {
  acceptDistilled,
  completionAnswer,
  readerMessages,
  type Haystack,
} from '../src/training/reader-distill.js';

const haystack: Haystack = {
  questionDate: '2023-07-10',
  sessions: [
    { id: 's1', date: '2023-05-15', facts: [], transcript: 'USER: I saw John Mulaney on 3 May.\nASSISTANT: Nice.' },
    { id: 's2', date: '2023-06-01', facts: [], transcript: 'USER: I ran 12 km on 26 May.\nASSISTANT: Great run.' },
  ],
};

const THINKING = { ...READER_CONTRACT_V5, thinking: true };
const NOTES_START = 'Answer only from the supplied history. Work in two steps.';
const DIRECT_SYSTEM =
  'Answer only from the supplied history. If it does not support an answer, say that you do not know. Be concise and do not invent details.';

describe('thinking step', () => {
  it('thinks on the question types that combine or compute', () => {
    expect([...THINKING_TYPES].sort()).toEqual(['knowledge-update', 'multi-session', 'temporal-reasoning']);
    expect(readingFor(THINKING, 'temporal-reasoning')).toBe('notes');
    expect(readingFor(THINKING, 'single-session-user')).toBe('direct');
    expect(readingFor(READER_CONTRACT_V5, 'multi-session')).toBe('direct');
  });

  it('renders the notes prompt for thinking types and abstention, the direct prompt otherwise', () => {
    const temporal = readerMessages(haystack, 'How long ago did I see John Mulaney?', 'temporal-reasoning', THINKING);
    expect(temporal[0]!.content.startsWith(NOTES_START)).toBe(true);
    const single = readerMessages(haystack, 'How far did I run?', 'single-session-user', THINKING);
    expect(single[0]!.content).toBe(DIRECT_SYSTEM);
    // LongMemEval abstention questions keep an aggregation type, so the harness reads them with notes
    const abstention = readerMessages(haystack, 'What is my dog called?', 'abstention', THINKING);
    expect(abstention[0]!.content.startsWith(NOTES_START)).toBe(true);
    const directAbstention = readerMessages(haystack, 'What is my dog called?', 'abstention', READER_CONTRACT_V5);
    expect(directAbstention[0]!.content).toBe(DIRECT_SYSTEM);
    const unthinking = readerMessages(haystack, 'How long ago did I see John Mulaney?', 'temporal-reasoning', READER_CONTRACT_V5);
    expect(unthinking[0]!.content).toBe(DIRECT_SYSTEM);
  });

  it('accepts a thinking reply only with a final answer line', () => {
    expect(acceptDistilled('multi-session', 'Notes:\n- x\nAnswer: 3', true)).toBe(true);
    expect(acceptDistilled('multi-session', 'Notes:\n- x', true)).toBe(false);
    expect(acceptDistilled('multi-session', 'Notes:\n- x\nAnswer:   ', true)).toBe(false);
    expect(acceptDistilled('abstention', 'Notes:\n- nothing\nAnswer: I do not know', true)).toBe(true);
    // an abstention row is notes-rendered under thinking: it needs the answer line, and only that line is tested
    expect(acceptDistilled('abstention', 'I do not know.', true)).toBe(false);
    expect(acceptDistilled('abstention', 'Notes:\n- the history does not mention a dog\nAnswer: Rex', true)).toBe(false);
    expect(acceptDistilled('abstention', 'I do not know.')).toBe(true);
    // the notes may say what the history does not mention; only the answer line is tested
    expect(
      acceptDistilled('temporal-reasoning', 'Notes:\n- the history does not mention the exact day\n- 3 May\nAnswer: 68 days', true),
    ).toBe(true);
    expect(acceptDistilled('knowledge-update', 'Notes:\n- x\nAnswer: I do not know', true)).toBe(false);
    // without thinking, a thinking type is still judged on the whole reply
    expect(acceptDistilled('multi-session', 'Three.')).toBe(true);
    expect(acceptDistilled('single-session-user', '12 km', true)).toBe(true);
  });

  it('returns the judged text', () => {
    const reply = 'Notes:\n- 3 May\nAnswer: 68 days';
    expect(completionAnswer(reply, true, 'temporal-reasoning')).toBe('68 days');
    expect(completionAnswer(reply, false, 'temporal-reasoning')).toBe(reply);
    expect(completionAnswer('  12 km  ', true, 'single-session-user')).toBe('12 km');
    expect(completionAnswer('Notes:\n- nothing\nAnswer: I do not know', true, 'abstention')).toBe('I do not know');
    expect(completionAnswer('Notes:\n- nothing\nAnswer: I do not know', false, 'abstention')).toBe(
      'Notes:\n- nothing\nAnswer: I do not know',
    );
  });

  it('rerender refuses a thinking contract', () => {
    expect(() => assertRerenderContract(THINKING)).toThrow(
      'rerender keeps stored direct answers; use the think command for a thinking contract',
    );
    expect(() => assertRerenderContract(READER_CONTRACT_V5)).not.toThrow();
  });

  it('the distill command reads only direct and notes', () => {
    expect(() => assertDistillReading(['--reading', 'notes'])).not.toThrow();
    expect(() => assertDistillReading(['--reading', 'direct'])).not.toThrow();
    expect(() => assertDistillReading([])).not.toThrow();
    expect(() => assertDistillReading(['--reading', 'two-call'])).toThrow(
      '--reading must be direct or notes, got two-call',
    );
  });
});
