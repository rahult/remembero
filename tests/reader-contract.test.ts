import { describe, expect, it } from 'vitest';
import {
  READER_CONTRACT_V5,
  contractFromFlags,
  contractId,
  contractRunnerFlags,
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
    expect(contractRunnerFlags(c)).toEqual(['--date-distances', '--computed-notes', '--structured-evidence']);
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
});
