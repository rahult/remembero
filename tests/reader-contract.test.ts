import { describe, expect, it } from 'vitest';
import {
  READER_CONTRACT_V5,
  contractFromEnv,
  contractFromFlags,
  contractId,
  contractBuilderArgs,
  contractRunnerFlags,
  distillManifestContract,
} from '../src/evals/reader-contract.js';
import { buildLongMemEvalAnswerContext } from '../src/evals/longmemeval-answer.js';
import { readerMessages, type Haystack } from '../src/training/reader-distill.js';

const haystack: Haystack = {
  questionDate: '2023-07-10',
  sessions: [
    { id: 's1', date: '2023-05-15', facts: ['likes(user, stand_up_comedy).'], transcript: 'USER: I saw John Mulaney on 3 May, it was 56 days before my exam.\nASSISTANT: Nice.' },
    { id: 's2', date: '2023-06-01', facts: [], transcript: 'USER: I ran 12 km on 26 May.\nASSISTANT: Great run.' },
  ],
};

describe('reader contract', () => {
  it('names itself stably', () => {
    expect(contractId(READER_CONTRACT_V5)).toBe('dd+notes@24576');
    expect(contractId({ ...READER_CONTRACT_V5, structuredEvidence: true, focusedBudget: true })).toBe('dd+notes+focus+evidence@24576');
  });

  it('reads the same flags the evaluation runner takes', () => {
    const c = contractFromFlags(['--date-distances', '--computed-notes', '--structured-evidence']);
    expect(c).toEqual({ ...READER_CONTRACT_V5, structuredEvidence: true });
    expect(contractRunnerFlags(c)).toEqual(['--date-distances', '--computed-notes', '--structured-evidence', '--context-bytes', '24576']);
  });

  it('names, reads and emits the thinking step', () => {
    expect(READER_CONTRACT_V5.thinking).toBe(false);
    expect(contractId(READER_CONTRACT_V5)).toBe('dd+notes@24576');
    const argv = ['--date-distances', '--computed-notes', '--reading', 'notes', '--context-bytes', '24576'];
    const c = contractFromFlags(argv);
    expect(c).toEqual({ ...READER_CONTRACT_V5, thinking: true });
    expect(contractId(c)).toBe('dd+notes+think@24576');
    expect(contractRunnerFlags(c)).toEqual(['--date-distances', '--computed-notes', '--reading', 'notes', '--context-bytes', '24576']);
    expect(contractFromFlags(contractRunnerFlags(c))).toEqual(c);
    expect(distillManifestContract(argv).id).toBe('dd+notes+think@24576');
    expect(contractFromFlags(['--reading', 'two-call']).thinking).toBe(false);
    expect(contractFromFlags(['--reading', 'direct']).thinking).toBe(false);
    expect(contractFromFlags(['--reading', 'notes'], { ...READER_CONTRACT_V5, thinking: true }).thinking).toBe(true);
    expect(contractFromFlags([], { ...READER_CONTRACT_V5, thinking: true }).thinking).toBe(false);
    const tiered = contractFromFlags(['--date-distances', '--computed-notes', '--reading', 'notes', '--full-sessions', '3']);
    expect(contractId(tiered)).toBe('dd+notes+think+full3@24576');
    expect(contractRunnerFlags(tiered)).toEqual([
      '--date-distances', '--computed-notes', '--full-sessions', '3', '--abstract-bytes', '320', '--reading', 'notes', '--context-bytes', '24576',
    ]);
    expect(contractFromFlags(contractRunnerFlags(tiered))).toEqual(tiered);
  });

  it('refuses a --context-bytes that is not a positive integer', () => {
    expect(() => contractFromFlags(['--context-bytes', 'abc'])).toThrow(
      '--context-bytes must be a positive integer, got abc',
    );
    expect(contractFromFlags(['--context-bytes', '8192']).contextBytes).toBe(8192);
  });

  it('renders the distillation prompt byte-for-byte as the harness renders it', () => {
    const contract = { ...READER_CONTRACT_V5, structuredEvidence: true };
    const distilled = readerMessages(haystack, 'How long ago did I see John Mulaney?', 'temporal-reasoning', contract);
    const instance = {
      question_id: 'distill-temporal-reasoning', question_type: 'temporal-reasoning',
      question: 'How long ago did I see John Mulaney?', question_date: '2023/07/10 (Sat) 09:00',
      answer: '', haystack_session_ids: [], haystack_dates: [], haystack_sessions: [], answer_session_ids: [],
    } as never;
    const harness = buildLongMemEvalAnswerContext(
      instance,
      haystack.sessions.map((s) => ({ opId: s.id, ts: `${s.date}T09:00:00.000Z`, text: s.transcript, facts: s.facts })),
      contract.contextBytes, [], 'direct', undefined,
      contract.dateDistances, contract.computedNotes, contract.focusedBudget, contract.structuredEvidence,
    );
    expect(distilled).toEqual(harness.messages);
    expect(distilled[1]!.content).toContain('Computed');
  });

  it('renders a thinking prompt byte-for-byte as the harness renders it with notes', () => {
    const contract = { ...READER_CONTRACT_V5, thinking: true };
    const sessions = haystack.sessions.map((s) => ({ opId: s.id, ts: `${s.date}T09:00:00.000Z`, text: s.transcript, facts: s.facts }));
    const base = {
      question: 'How long ago did I see John Mulaney?', question_date: '2023/07/10 (Sat) 09:00',
      answer: '', haystack_session_ids: [], haystack_dates: [], haystack_sessions: [], answer_session_ids: [],
    };
    const distilled = readerMessages(haystack, base.question, 'temporal-reasoning', contract);
    const harness = buildLongMemEvalAnswerContext(
      { ...base, question_id: 'distill-temporal-reasoning', question_type: 'temporal-reasoning' } as never,
      sessions, contract.contextBytes, [], 'notes', undefined, ...contractBuilderArgs(contract),
    );
    expect(distilled).toEqual(harness.messages);
    const abstention = readerMessages(haystack, base.question, 'abstention', contract);
    const harnessAbstention = buildLongMemEvalAnswerContext(
      { ...base, question_id: 'distill-abstention', question_type: 'multi-session' } as never,
      sessions, contract.contextBytes, [], 'notes', undefined, ...contractBuilderArgs(contract),
    );
    expect(abstention).toEqual(harnessAbstention.messages);
  });

  it('the distill manifest records the contract it rendered with', () => {
    const entry = distillManifestContract(['--computed-notes', '--date-distances']);
    expect(entry.id).toBe('dd+notes@24576');
    expect(entry.runnerFlags).toEqual(['--date-distances', '--computed-notes', '--context-bytes', '24576']);
  });

  it('names, reads and emits context tiers', () => {
    expect(READER_CONTRACT_V5.fullSessions).toBeNull();
    expect(READER_CONTRACT_V5.abstractBytes).toBe(320);
    const c = contractFromFlags(['--date-distances', '--computed-notes', '--full-sessions', '3']);
    expect(c).toEqual({ ...READER_CONTRACT_V5, fullSessions: 3 });
    expect(contractId(c)).toBe('dd+notes+full3@24576');
    expect(contractRunnerFlags(c)).toEqual([
      '--date-distances', '--computed-notes', '--full-sessions', '3', '--abstract-bytes', '320', '--context-bytes', '24576',
    ]);
    expect(contractFromFlags(contractRunnerFlags(c))).toEqual(c);
    expect(contractBuilderArgs(c)).toEqual([true, true, false, false, { fullSessions: 3, abstractBytes: 320 }]);
    expect(contractBuilderArgs(READER_CONTRACT_V5)).toEqual([true, true, false, false, undefined]);
    const a = contractFromFlags(['--full-sessions', '5', '--abstract-bytes', '200']);
    expect(contractId(a)).toBe('plain+full5a200@24576');
    expect(contractFromFlags(contractRunnerFlags(a))).toEqual(a);
    const entry = distillManifestContract(['--date-distances', '--computed-notes', '--full-sessions', '2']);
    expect(entry.id).toBe('dd+notes+full2@24576');
    expect(entry.runnerFlags).toContain('--full-sessions');
  });

  it('refuses malformed tier flags and tiers combined with the focused budget', () => {
    expect(() => contractFromFlags(['--full-sessions', '0'])).toThrow('--full-sessions must be a positive integer, got 0');
    expect(() => contractFromFlags(['--full-sessions', 'x'])).toThrow('--full-sessions must be a positive integer, got x');
    expect(() => contractFromFlags(['--full-sessions', '3', '--abstract-bytes', '100'])).toThrow(
      '--abstract-bytes must be an integer from 120 to 2048, got 100',
    );
    expect(() => contractFromFlags(['--full-sessions', '3', '--abstract-bytes', '4096'])).toThrow(/--abstract-bytes/);
    expect(() => contractFromFlags(['--abstract-bytes', '200'])).toThrow('--abstract-bytes needs --full-sessions');
    expect(() => contractFromFlags(['--full-sessions', '3', '--focused-budget'])).toThrow(
      '--full-sessions cannot be combined with --focused-budget',
    );
  });

  it('reads the one environment variable the old distill commands set', () => {
    expect(contractFromEnv({} as NodeJS.ProcessEnv)).toEqual({ ...READER_CONTRACT_V5, computedNotes: false });
  });
});
