import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { buildLongMemEvalJudgePrompt } from '../src/evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';
import {
  READER_CONTRACT_V5,
  contractId,
} from '../src/evals/reader-contract.js';
import type { ChatMessage } from '../src/llm/client.js';
import type { LabelledSession } from '../src/training/reader-data.js';
import {
  readerMessages,
  toDistilledConversation,
} from '../src/training/reader-distill.js';
import {
  mineCounts,
  mineFile,
  mineJudgePrompt,
  readDistilledFile,
  studentContractFromFlags,
  teacherContractFromManifest,
} from '../src/training/reader-mine.js';
import {
  haystackFromMeta,
  type DistilledMetaRow,
} from '../src/training/reader-rerender.js';
import type { CompletionClient } from '../src/training/reader-think.js';

const THINKING = { ...READER_CONTRACT_V5, thinking: true };
const NOTES_START = 'Answer only from the supplied history. Work in two steps.';
const DIRECT_START =
  'Answer only from the supplied history. If it does not support an answer';

const pool = new Map<string, LabelledSession>([
  [
    's1',
    {
      id: 's1',
      date: '2023-05-15',
      facts: [],
      transcript:
        'USER: I went to the marathon three weeks ago and ran 21 kilometres.\nASSISTANT: That is a strong half marathon time.',
    },
  ],
  [
    's2',
    {
      id: 's2',
      date: '2023-06-20',
      facts: [],
      transcript:
        'USER: Yesterday I signed up for the second marathon of the year.\nASSISTANT: Two marathons is a real year of running.',
    },
  ],
]);

const multi: DistilledMetaRow = {
  type: 'multi-session',
  question: 'how many marathons have I mentioned running this year?',
  questionDate: '2023-07-05',
  sessionIds: ['s2', 's1'],
  evidence: [1, 2],
  answer: 'Notes:\n- 2023-06-19: second marathon\nAnswer: Two',
};
const temporal: DistilledMetaRow = {
  type: 'temporal-reasoning',
  question: 'how long ago did I run the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [1],
  answer: 'Notes:\n- 2023-04-24: ran the marathon\nAnswer: Ten weeks ago',
};
const abstention: DistilledMetaRow = {
  type: 'abstention',
  question: 'what shoes did I wear for the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [],
  answer: 'Notes:\n- no shoes mentioned\nAnswer: I do not know',
};
const single: DistilledMetaRow = {
  type: 'single-session-user',
  question: 'how far did I run at the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [1],
  answer: '21 kilometres.',
};
const rows = [multi, temporal, abstention, single];

interface Call {
  messages: ChatMessage[];
  maxTokens: number | undefined;
}

/** A client answering from the first entry whose key the last user message contains. */
function stub(replies: Array<[string, string]>): CompletionClient & {
  calls: Call[];
} {
  const calls: Call[] = [];
  return {
    calls,
    async completeWithUsage(messages, options = {}) {
      calls.push({ messages, maxTokens: options.maxTokens });
      const last = messages.filter((m) => m.role === 'user').at(-1)!.content;
      const hit = replies.find(([key]) => last.includes(key));
      if (hit === undefined) throw new Error('stub has no reply');
      return { content: hit[1] };
    },
  };
}

/** The client's replies carrying a completion-token count, as the real client's usage does. */
function withUsage<C extends CompletionClient>(
  client: C,
  completionTokens: number,
): C {
  return {
    ...client,
    async completeWithUsage(messages, options) {
      const completion = await client.completeWithUsage(messages, options);
      return { ...completion, usage: { completionTokens } };
    },
  };
}

const throwing: CompletionClient = {
  async completeWithUsage() {
    throw new Error('must not be called');
  },
};

const lines = (path: string) =>
  readFileSync(path, 'utf8').split('\n').filter(Boolean);

/** A distilled directory as `distill` writes it under the thinking contract. */
function distilledDir(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mine-from-'));
  const conversations = rows.map((row) => {
    const built = haystackFromMeta(row, pool);
    if ('missing' in built) throw new Error('fixture');
    return JSON.stringify(
      toDistilledConversation({
        ...row,
        messages: readerMessages(
          built.haystack,
          row.question,
          row.type,
          THINKING,
        ),
      }),
    );
  });
  writeFileSync(
    join(dir, 'conversations.jsonl'),
    `${conversations.join('\n')}\n`,
  );
  writeFileSync(
    join(dir, 'conversations.jsonl.meta.jsonl'),
    `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`,
  );
  return dir;
}

// the v4 student reads directly; its judged text is its whole reply
const studentReplies: Array<[string, string]> = [
  [multi.question, 'Three.'],
  [temporal.question, 'Ten weeks ago.'],
  [abstention.question, 'You wore red shoes.'],
  [single.question, '21 kilometres.'],
];

/** The judge says yes when the model response is one of `right`. */
function judgeOf(right: string[]): CompletionClient & { calls: Call[] } {
  const calls: Call[] = [];
  return {
    calls,
    async completeWithUsage(messages, options = {}) {
      calls.push({ messages, maxTokens: options.maxTokens });
      const prompt = messages.at(-1)!.content;
      const response = prompt.slice(prompt.indexOf('Model response: '));
      return {
        content: right.some((r) => response.includes(`Model response: ${r}`))
          ? 'Yes.'
          : 'no',
      };
    },
  };
}

describe('student contract flags', () => {
  it('reads only the --student- prefixed contract flags', () => {
    const contract = studentContractFromFlags([
      'mine',
      '--computed-notes',
      '--reading',
      'notes',
      '--student-date-distances',
      '--student-structured-evidence',
      '--student-context-bytes',
      '16384',
      '--student-model',
      'reader-v4',
    ]);
    expect(contract).toMatchObject({
      dateDistances: true,
      computedNotes: false,
      structuredEvidence: true,
      thinking: false,
      contextBytes: 16384,
    });
    expect(contract.id).toBe(
      contractId({
        ...READER_CONTRACT_V5,
        computedNotes: false,
        structuredEvidence: true,
        contextBytes: 16384,
      }),
    );
    expect(contract.runnerFlags).toContain('--structured-evidence');
    expect(
      studentContractFromFlags([
        '--student-reading',
        'notes',
        '--student-computed-notes',
      ]).thinking,
    ).toBe(true);
  });

  it('refuses a student reading that is not direct or notes', () => {
    expect(() =>
      studentContractFromFlags(['--student-reading', 'two-call']),
    ).toThrow(/direct or notes/);
  });

  it('refuses a tiered student contract', () => {
    expect(() =>
      studentContractFromFlags(['--student-full-sessions', '4']),
    ).toThrow(/full-sessions/);
  });
});

describe('teacher contract', () => {
  it('comes from the distilled manifest, and a manifest without one is refused', () => {
    const contract = { ...THINKING, id: 'dd+notes+think@24576' };
    expect(teacherContractFromManifest({ contract })).toEqual(contract);
    expect(() => teacherContractFromManifest({ labels: 'x' })).toThrow(
      /contract/,
    );
    expect(() =>
      teacherContractFromManifest({ contract: { thinking: true } }),
    ).toThrow(/contract/);
  });
});

describe('judge prompt', () => {
  it('is the LongMemEval judge prompt for the row type, abstention through its own branch', () => {
    const instance = (row: DistilledMetaRow, id: string) =>
      ({
        question_id: id,
        question_type: row.type,
        question: row.question,
        answer: 'EXPECTED',
      }) as unknown as LongMemEvalInstance;
    expect(mineJudgePrompt(temporal, 'EXPECTED', 'HYP')).toBe(
      buildLongMemEvalJudgePrompt(instance(temporal, 'mine'), 'HYP'),
    );
    expect(mineJudgePrompt(temporal, 'EXPECTED', 'HYP')).toContain(
      'one-unit error',
    );
    expect(mineJudgePrompt(abstention, 'EXPECTED', 'HYP')).toBe(
      buildLongMemEvalJudgePrompt(instance(abstention, 'mine_abs'), 'HYP'),
    );
    expect(mineJudgePrompt(abstention, 'EXPECTED', 'HYP')).toContain(
      'unanswerable',
    );
    expect(mineJudgePrompt(multi, 'EXPECTED', 'HYP')).not.toContain(
      'unanswerable',
    );
  });
});

describe('reading a distilled file', () => {
  it('returns the meta rows with their conversation lines, byte for byte', () => {
    const dir = distilledDir();
    const read = readDistilledFile(dir, 'conversations.jsonl');
    expect(read.rows).toEqual(rows);
    expect(read.lines).toEqual(lines(join(dir, 'conversations.jsonl')));
  });

  it('refuses twins that are not line-aligned', () => {
    const dir = distilledDir();
    writeFileSync(
      join(dir, 'conversations.jsonl.meta.jsonl'),
      `${[temporal, multi, abstention, single].map((r) => JSON.stringify(r)).join('\n')}\n`,
    );
    expect(() => readDistilledFile(dir, 'conversations.jsonl')).toThrow(
      /aligned/,
    );
  });
});

describe('mining a file', () => {
  it('keeps a missed row as the teacher row unchanged, results for every judged row', async () => {
    const from = distilledDir();
    const out = mkdtempSync(join(tmpdir(), 'mine-out-'));
    const { rows: meta, lines: source } = readDistilledFile(
      from,
      'conversations.jsonl',
    );
    const student = stub(studentReplies);
    const judge = judgeOf(['Ten weeks ago.', '21 kilometres.']);
    await mineFile({
      file: 'conversations.jsonl',
      rows: meta,
      lines: source,
      pool,
      teacherThinking: true,
      studentContract: READER_CONTRACT_V5,
      clients: { student, judge },
      outDir: out,
      concurrency: 2,
      studentMaxTokens: 1024,
    });

    // the student read under its own direct contract, not the teacher's notes prompt
    expect(student.calls).toHaveLength(4);
    for (const call of student.calls) {
      expect(call.maxTokens).toBe(1024);
      expect(call.messages[0]!.content.startsWith(DIRECT_START)).toBe(true);
    }
    const multiCall = student.calls.find((c) =>
      c.messages.at(-1)!.content.includes(multi.question),
    )!;
    const built = haystackFromMeta(multi, pool);
    if ('missing' in built) throw new Error('fixture');
    expect(multiCall.messages).toEqual(
      readerMessages(
        built.haystack,
        multi.question,
        multi.type,
        READER_CONTRACT_V5,
      ),
    );

    // judge compares the student's judged text with the teacher's final line
    expect(judge.calls).toHaveLength(4);
    expect(judge.calls.every((c) => c.maxTokens === 16)).toBe(true);
    expect(judge.calls.map((c) => c.messages.at(-1)!.content)).toEqual(
      expect.arrayContaining([
        mineJudgePrompt(multi, 'Two', 'Three.'),
        mineJudgePrompt(abstention, 'I do not know', 'You wore red shoes.'),
        mineJudgePrompt(single, '21 kilometres.', '21 kilometres.'),
      ]),
    );

    const misses = lines(join(out, 'misses.jsonl'));
    const missMeta = lines(join(out, 'misses.jsonl.meta.jsonl')).map(
      (l) =>
        JSON.parse(l) as DistilledMetaRow & { file: string; index: number },
    );
    expect(misses).toHaveLength(2);
    expect(missMeta).toHaveLength(2);
    misses.forEach((line, i) => {
      const { file, index, ...row } = missMeta[i]!;
      expect(file).toBe('conversations.jsonl');
      // byte for byte the teacher's line, rendered for the teacher (notes prompt)
      expect(line).toBe(source[index]);
      expect(row).toEqual(rows[index]);
      const conversation = JSON.parse(line) as {
        messages: Array<{ content: string }>;
      };
      expect(conversation.messages[0]!.content.startsWith(NOTES_START)).toBe(
        true,
      );
    });
    expect(missMeta.map((m) => m.type).sort()).toEqual([
      'abstention',
      'multi-session',
    ]);

    const results = lines(join(out, 'results.jsonl')).map(
      (l) => JSON.parse(l) as Record<string, unknown>,
    );
    expect(results).toHaveLength(4);
    expect(results).toContainEqual({
      file: 'conversations.jsonl',
      index: 0,
      type: 'multi-session',
      question: multi.question,
      expected: 'Two',
      hypothesis: 'Three.',
      reply: 'Three.',
      completionTokens: null,
      maxTokens: 1024,
      correct: false,
    });
    expect(results).toContainEqual(
      expect.objectContaining({ index: 3, correct: true }),
    );

    expect(mineCounts(join(out, 'progress.jsonl'))).toEqual({
      byType: {
        'multi-session': { asked: 1, correct: 0, misses: 1 },
        'temporal-reasoning': { asked: 1, correct: 1, misses: 0 },
        abstention: { asked: 1, correct: 0, misses: 1 },
        'single-session-user': { asked: 1, correct: 1, misses: 0 },
      },
      errors: 0,
      missing: 0,
    });
  });

  it("judges a thinking student's final answer line and keeps its whole reply", async () => {
    const out = mkdtempSync(join(tmpdir(), 'mine-think-'));
    const reply = 'Notes:\n- 2023-06-19: second marathon\nAnswer: Two';
    const student = withUsage(stub([[multi.question, reply]]), 37);
    await mineFile({
      file: 'heldout.jsonl',
      rows: [multi],
      lines: ['{"teacher":1}'],
      pool,
      teacherThinking: true,
      studentContract: THINKING,
      clients: { student, judge: judgeOf(['Two']) },
      outDir: out,
      studentMaxTokens: 2048,
    });
    expect(student.calls[0]!.messages[0]!.content.startsWith(NOTES_START)).toBe(
      true,
    );
    const [result] = lines(join(out, 'results.jsonl')).map(
      (l) => JSON.parse(l) as Record<string, unknown>,
    );
    expect(result).toMatchObject({
      file: 'heldout.jsonl',
      hypothesis: 'Two',
      reply,
      completionTokens: 37,
      maxTokens: 2048,
      correct: true,
    });
  });

  it('resumes: judged rows are not asked again, an errored row is retried with its stored student reply', async () => {
    const from = distilledDir();
    const out = mkdtempSync(join(tmpdir(), 'mine-resume-'));
    const { rows: meta, lines: source } = readDistilledFile(
      from,
      'conversations.jsonl',
    );
    const base = {
      file: 'conversations.jsonl',
      rows: meta,
      lines: source,
      pool,
      teacherThinking: true,
      studentContract: READER_CONTRACT_V5,
      outDir: out,
      studentMaxTokens: 1024,
    };
    // the judge fails on the abstention row only
    const flakyJudge: CompletionClient = {
      async completeWithUsage(messages, options) {
        if (messages.at(-1)!.content.includes('unanswerable'))
          return { content: 'Maybe' };
        return judgeOf(['Ten weeks ago.', '21 kilometres.']).completeWithUsage(
          messages,
          options,
        );
      },
    };
    await mineFile({
      ...base,
      clients: {
        student: withUsage(stub(studentReplies), 55),
        judge: flakyJudge,
      },
    });
    expect(mineCounts(join(out, 'progress.jsonl'))).toMatchObject({
      errors: 1,
    });
    expect(lines(join(out, 'misses.jsonl'))).toHaveLength(1);
    expect(lines(join(out, 'results.jsonl'))).toHaveLength(3);

    const judge = judgeOf([]);
    await mineFile({ ...base, clients: { student: throwing, judge } });
    expect(judge.calls).toHaveLength(1);
    expect(judge.calls[0]!.messages.at(-1)!.content).toBe(
      mineJudgePrompt(abstention, 'I do not know', 'You wore red shoes.'),
    );
    expect(lines(join(out, 'misses.jsonl'))).toHaveLength(2);
    expect(lines(join(out, 'misses.jsonl.meta.jsonl'))).toHaveLength(2);
    expect(lines(join(out, 'results.jsonl'))).toHaveLength(4);
    // the retried row keeps the completion tokens of its stored reply
    expect(
      lines(join(out, 'results.jsonl'))
        .map((l) => JSON.parse(l) as Record<string, unknown>)
        .find((r) => r.type === 'abstention'),
    ).toMatchObject({ completionTokens: 55, maxTokens: 1024 });
    expect(mineCounts(join(out, 'progress.jsonl'))).toMatchObject({
      errors: 0,
      byType: { abstention: { asked: 1, correct: 0, misses: 1 } },
    });

    // a finished file asks nobody anything
    await mineFile({
      ...base,
      clients: { student: throwing, judge: throwing },
    });
    expect(lines(join(out, 'results.jsonl'))).toHaveLength(4);
  });

  it('counts a row whose sessions left the pool as missing, not asked', async () => {
    const out = mkdtempSync(join(tmpdir(), 'mine-missing-'));
    await mineFile({
      file: 'conversations.jsonl',
      rows: [{ ...multi, sessionIds: ['s1', 'gone'] }],
      lines: ['{}'],
      pool,
      teacherThinking: true,
      studentContract: READER_CONTRACT_V5,
      clients: { student: throwing, judge: throwing },
      outDir: out,
      studentMaxTokens: 1024,
    });
    expect(mineCounts(join(out, 'progress.jsonl'))).toEqual({
      byType: {},
      errors: 0,
      missing: 1,
    });
  });
});
