import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { READER_CONTRACT_V5 } from '../src/evals/reader-contract.js';
import type { LabelledSession } from '../src/training/reader-data.js';
import {
  haystackFromMeta,
  rerenderMetaFile,
  rerenderRow,
  type DistilledMetaRow,
} from '../src/training/reader-rerender.js';

const pool = new Map<string, LabelledSession>([
  [
    's1',
    {
      id: 's1',
      date: '2023-05-15',
      facts: ['attended(user, ces).'],
      transcript:
        'USER: I went to the marathon three weeks ago and ran 21 kilometres.\nASSISTANT: That is a strong half marathon time.',
    },
  ],
  [
    's2',
    {
      id: 's2',
      date: '2023-06-20',
      facts: ['runs(user, marathon).'],
      transcript:
        'USER: Yesterday I signed up for the second marathon of the year.\nASSISTANT: Two marathons is a real year of running.',
    },
  ],
]);

const rows: DistilledMetaRow[] = [
  {
    type: 'temporal-reasoning',
    question: 'how long ago did I run the marathon?',
    questionDate: '2023-07-01',
    sessionIds: ['s1', 's2'],
    evidence: [1],
    answer: 'You ran it on 2023-04-24, about ten weeks ago.',
  },
  {
    type: 'multi-session',
    question: 'how many marathons have I mentioned running this year?',
    questionDate: '2023-07-05',
    sessionIds: ['s2', 's1'],
    evidence: [1, 2],
    answer: 'Two.',
  },
];

describe('re-rendering distilled reader data through a contract', () => {
  it('keeps the meta answer as the assistant turn and puts the computed notes in the user turn', () => {
    for (const row of rows) {
      const conversation = rerenderRow(row, pool, READER_CONTRACT_V5)!;
      expect(conversation).toBeDefined();
      const last = conversation.messages.at(-1)!;
      expect(last.role).toBe('assistant');
      expect(last.content).toBe(row.answer);
      const user = conversation.messages.filter((m) => m.role === 'user').at(-1)!;
      expect(user.content).toContain('Computed');
      expect(user.content).toContain(row.question);
    }
  });

  it('rebuilds the haystack in the recorded session order, not date order', () => {
    const built = haystackFromMeta(rows[1], pool);
    expect('haystack' in built).toBe(true);
    if (!('haystack' in built)) return;
    expect(built.haystack.sessions.map((s) => s.id)).toEqual(['s2', 's1']);
    expect(built.haystack.questionDate).toBe('2023-07-05');
  });

  it('skips and counts a row whose sessions are not in the pool', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rerender-'));
    const metaPath = join(dir, 'conversations.jsonl.meta.jsonl');
    const orphan: DistilledMetaRow = {
      ...rows[0],
      sessionIds: ['s1', 'gone'],
    };
    writeFileSync(
      metaPath,
      `${[...rows, orphan].map((r) => JSON.stringify(r)).join('\n')}\n`,
    );
    const outPath = join(dir, 'out', 'conversations.jsonl');
    const result = rerenderMetaFile(metaPath, outPath, pool, READER_CONTRACT_V5);
    expect(result).toMatchObject({ rows: 2, skipped: 1 });
    expect(result.missingExamples[0]).toEqual({ index: 2, missing: ['gone'] });
    const written = readFileSync(outPath, 'utf8').split('\n').filter(Boolean);
    expect(written).toHaveLength(2);
    expect(
      written.map(
        (line) =>
          (
            JSON.parse(line) as { messages: Array<{ content: string }> }
          ).messages.at(-1)!.content,
      ),
    ).toEqual([rows[0].answer, rows[1].answer]);
  });
});
