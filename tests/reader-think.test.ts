import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { READER_CONTRACT_V5 } from '../src/evals/reader-contract.js';
import type { ChatMessage } from '../src/llm/client.js';
import type { LabelledSession } from '../src/training/reader-data.js';
import {
  rerenderRow,
  type DistilledMetaRow,
} from '../src/training/reader-rerender.js';
import {
  agreementPrompt,
  sourceLabels,
  thinkCounts,
  thinkFile,
  thinkRow,
  type CompletionClient,
} from '../src/training/reader-think.js';

const THINKING = { ...READER_CONTRACT_V5, thinking: true };
const NOTES_START = 'Answer only from the supplied history. Work in two steps.';

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
  answer: 'Two.',
};
const temporal: DistilledMetaRow = {
  type: 'temporal-reasoning',
  question: 'how long ago did I run the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [1],
  answer: 'About ten weeks ago.',
};
const abstention: DistilledMetaRow = {
  type: 'abstention',
  question: 'what shoes did I wear for the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [],
  answer: 'I do not know.',
};
const single: DistilledMetaRow = {
  type: 'single-session-user',
  question: 'how far did I run at the marathon?',
  questionDate: '2023-07-01',
  sessionIds: ['s1', 's2'],
  evidence: [1],
  answer: '21 kilometres.',
};

interface Call {
  messages: ChatMessage[];
  maxTokens: number | undefined;
}

/** A client answering from the first entry whose key the last user message contains. */
function stub(
  replies: Array<[string, string]>,
  calls: Call[] = [],
): CompletionClient & { calls: Call[] } {
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

const throwing: CompletionClient = {
  async completeWithUsage() {
    throw new Error('must not be called');
  },
};

describe('agreement prompt', () => {
  it('asks whether the two answers state the same result, yes or no', () => {
    const prompt = agreementPrompt('how many marathons?', 'Two.', '2');
    expect(prompt).toContain('how many marathons?');
    expect(prompt).toContain('Two.');
    expect(prompt).toContain('2');
    expect(prompt).toMatch(/same result/);
    expect(prompt).toMatch(/unavailable/);
    expect(prompt.endsWith('Answer yes or no only.')).toBe(true);
  });
});

describe('thinking regeneration of one row', () => {
  it('keeps a thinking-type row when the judge agrees, the thinking reply as the assistant turn', async () => {
    const reply =
      'Notes:\n- 2023-06-19: signed up for the second marathon\nAnswer: Two';
    const teacher = stub([[multi.question, reply]]);
    const judge = stub([['Answer yes or no only.', 'Yes']]);
    const result = await thinkRow(multi, pool, THINKING, { teacher, judge });
    expect(result.outcome).toBe('kept');
    if (result.outcome !== 'kept') return;
    expect(teacher.calls[0]!.messages[0]!.content.startsWith(NOTES_START)).toBe(
      true,
    );
    const last = result.conversation.messages.at(-1)!;
    expect(last).toEqual({ role: 'assistant', content: reply });
    expect(result.meta).toEqual({
      ...multi,
      answer: reply,
      storedAnswer: 'Two.',
    });
    expect(judge.calls).toHaveLength(1);
    expect(judge.calls[0]!.maxTokens).toBe(16);
    expect(judge.calls[0]!.messages.at(-1)!.content).toBe(
      agreementPrompt(multi.question, 'Two.', 'Two'),
    );
  });

  it('moves a row to disagreements when the judge says no, with both answers', async () => {
    const reply =
      'Notes:\n- 2023-04-24: ran the marathon\nAnswer: Nine weeks ago';
    const result = await thinkRow(temporal, pool, THINKING, {
      teacher: stub([[temporal.question, reply]]),
      judge: stub([['Answer yes or no only.', 'no']]),
    });
    expect(result.outcome).toBe('disagreed');
    if (result.outcome !== 'disagreed') return;
    expect(result.disagreement).toMatchObject({
      type: 'temporal-reasoning',
      question: temporal.question,
      storedAnswer: temporal.answer,
      reply,
      finalLine: 'Nine weeks ago',
    });
  });

  it('counts a reply without an answer line as a format rejection and never asks the judge', async () => {
    const judge = stub([]);
    const result = await thinkRow(multi, pool, THINKING, {
      teacher: stub([[multi.question, 'Notes:\n- two marathons']]),
      judge,
    });
    expect(result.outcome).toBe('rejectedFormat');
    expect(judge.calls).toHaveLength(0);
  });

  it('keeps an abstention row whose final line abstains, without a judge call', async () => {
    const reply = 'Notes:\n- no shoes mentioned\nAnswer: I do not know';
    const teacher = stub([[abstention.question, reply]]);
    const result = await thinkRow(abstention, pool, THINKING, {
      teacher,
      judge: throwing,
    });
    expect(result.outcome).toBe('kept');
    expect(teacher.calls[0]!.messages[0]!.content.startsWith(NOTES_START)).toBe(
      true,
    );
  });

  it('rejects an abstention row whose final line answers', async () => {
    const result = await thinkRow(abstention, pool, THINKING, {
      teacher: stub([[abstention.question, 'Notes:\n- s1\nAnswer: Red shoes']]),
      judge: throwing,
    });
    expect(result.outcome).toBe('rejectedFormat');
  });

  it('re-renders a single-session row byte-identical to rerenderRow, with no model calls', async () => {
    const result = await thinkRow(single, pool, THINKING, {
      teacher: throwing,
      judge: throwing,
    });
    expect(result.outcome).toBe('rendered');
    if (result.outcome !== 'rendered') return;
    expect(JSON.stringify(result.conversation)).toBe(
      JSON.stringify(
        rerenderRow(single, pool, { ...THINKING, thinking: false }),
      ),
    );
    expect(result.meta).toEqual(single);
  });

  it('reports a row whose sessions left the pool as missing', async () => {
    const result = await thinkRow(
      { ...multi, sessionIds: ['s1', 'gone'] },
      pool,
      THINKING,
      { teacher: throwing, judge: throwing },
    );
    expect(result).toEqual({ outcome: 'missing', missing: ['gone'] });
  });

  it('records a judge reply that is not yes or no as an error', async () => {
    const result = await thinkRow(multi, pool, THINKING, {
      teacher: stub([[multi.question, 'Notes:\n- x\nAnswer: Two']]),
      judge: stub([['Answer yes or no only.', 'Probably']]),
    });
    expect(result.outcome).toBe('error');
  });
});

describe('thinking regeneration of a file', () => {
  const rows = [multi, single, temporal, abstention];
  const replies: Array<[string, string]> = [
    [multi.question, 'Notes:\n- two sign-ups\nAnswer: Two'],
    [temporal.question, 'Notes:\n- 2023-04-24\nAnswer: Nine weeks ago'],
    [abstention.question, 'Notes:\n- nothing\nAnswer: I do not know'],
  ];
  const judge = () =>
    stub([
      [`B: Two`, 'yes'],
      ['B: Nine weeks ago', 'no'],
    ]);
  const lines = (path: string) =>
    readFileSync(path, 'utf8').split('\n').filter(Boolean);

  it('writes kept and rendered rows line-aligned with their meta, disagreements apart, progress per index', async () => {
    const out = mkdtempSync(join(tmpdir(), 'think-'));
    await thinkFile({
      file: 'conversations.jsonl',
      rows,
      pool,
      contract: THINKING,
      clients: { teacher: stub(replies), judge: judge() },
      outDir: out,
      concurrency: 2,
    });
    const conversations = lines(join(out, 'conversations.jsonl'));
    const meta = lines(join(out, 'conversations.jsonl.meta.jsonl')).map(
      (l) => JSON.parse(l) as DistilledMetaRow & { storedAnswer?: string },
    );
    expect(conversations).toHaveLength(3);
    expect(meta).toHaveLength(3);
    conversations.forEach((line, i) => {
      const conversation = JSON.parse(line) as {
        messages: Array<{ content: string }>;
      };
      expect(conversation.messages.at(-1)!.content).toBe(meta[i]!.answer);
    });
    expect(meta.map((m) => m.type).sort()).toEqual([
      'abstention',
      'multi-session',
      'single-session-user',
    ]);
    const disagreements = lines(join(out, 'disagreements.jsonl')).map(
      (l) => JSON.parse(l) as Record<string, unknown>,
    );
    expect(disagreements).toEqual([
      expect.objectContaining({
        file: 'conversations.jsonl',
        index: 2,
        storedAnswer: temporal.answer,
      }),
    ]);
    expect(thinkCounts(join(out, 'progress.jsonl'))).toEqual({
      'multi-session': { kept: 1 },
      'single-session-user': { rendered: 1 },
      'temporal-reasoning': { disagreed: 1 },
      abstention: { kept: 1 },
    });
  });

  it('resumes: done indices make no calls and write nothing twice; errored indices are retried', async () => {
    const out = mkdtempSync(join(tmpdir(), 'think-resume-'));
    const flaky = stub(replies.filter(([q]) => q !== abstention.question));
    await thinkFile({
      file: 'heldout.jsonl',
      rows,
      pool,
      contract: THINKING,
      clients: { teacher: flaky, judge: judge() },
      outDir: out,
    });
    expect(thinkCounts(join(out, 'progress.jsonl')).abstention).toEqual({
      errors: 1,
    });
    const retried = stub(replies);
    await thinkFile({
      file: 'heldout.jsonl',
      rows,
      pool,
      contract: THINKING,
      clients: { teacher: retried, judge: throwing },
      outDir: out,
    });
    expect(retried.calls).toHaveLength(1);
    expect(retried.calls[0]!.messages.at(-1)!.content).toContain(
      abstention.question,
    );
    expect(lines(join(out, 'heldout.jsonl'))).toHaveLength(3);
    expect(lines(join(out, 'heldout.jsonl.meta.jsonl'))).toHaveLength(3);
    expect(lines(join(out, 'disagreements.jsonl'))).toHaveLength(1);
    expect(thinkCounts(join(out, 'progress.jsonl')).abstention).toEqual({
      kept: 1,
    });
    // a third run over a finished file asks nobody anything
    await thinkFile({
      file: 'heldout.jsonl',
      rows,
      pool,
      contract: THINKING,
      clients: { teacher: throwing, judge: throwing },
      outDir: out,
    });
    expect(lines(join(out, 'heldout.jsonl'))).toHaveLength(3);
  });
});

describe('source labels', () => {
  it('reads the labels file from the manifest, or from its parts when they agree', () => {
    expect(sourceLabels({ labels: 'a.jsonl' })).toBe('a.jsonl');
    expect(
      sourceLabels({
        parts: { v3: { labels: 'a.jsonl' }, v4: { labels: 'a.jsonl' } },
      }),
    ).toBe('a.jsonl');
    expect(
      sourceLabels({
        parts: { v3: { labels: 'a.jsonl' }, v4: { labels: 'b.jsonl' } },
      }),
    ).toBeUndefined();
    expect(sourceLabels({})).toBeUndefined();
  });
});
