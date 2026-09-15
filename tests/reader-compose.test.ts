import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { composeTraining } from '../src/training/reader-compose.js';

const TYPES = ['multi-session', 'temporal-reasoning', 'abstention'];

function metaRow(tag: string, i: number) {
  return {
    type: TYPES[i % TYPES.length],
    question: `question ${tag} ${i}?`,
    questionDate: '2023-07-01',
    sessionIds: ['s1'],
    evidence: [1],
    answer: `Notes:\n- ${tag} ${i}\nAnswer: ${tag}-${i}`,
  };
}

function line(row: { question: string; answer: string }): string {
  return JSON.stringify({
    messages: [
      { role: 'system', content: 'Answer only from the supplied history.' },
      { role: 'user', content: row.question },
      { role: 'assistant', content: row.answer },
    ],
  });
}

function writeTwins(
  dir: string,
  file: string,
  rows: Array<Record<string, unknown> & { question: string; answer: string }>,
): void {
  writeFileSync(join(dir, file), rows.map((r) => `${line(r)}\n`).join(''));
  writeFileSync(
    join(dir, `${file}.meta.jsonl`),
    rows.map((r) => `${JSON.stringify(r)}\n`).join(''),
  );
}

function contract(id: string) {
  return { id, thinking: id.includes('+think'), contextBytes: 24576 };
}

interface Fixture {
  root: string;
  base: string;
  misses: string;
  other: string;
}

/** Base: 10 rows + 4 heldout. Misses: 3 conversations rows (one written twice) + 1 heldout row. */
function fixture(
  ids: { base?: string; misses?: string; other?: string } = {},
): Fixture {
  const root = mkdtempSync(join(tmpdir(), 'compose-'));
  const base = join(root, 'base');
  const misses = join(root, 'misses');
  const other = join(root, 'other');
  for (const dir of [base, misses, other]) mkdirSync(dir);
  writeTwins(
    base,
    'conversations.jsonl',
    Array.from({ length: 10 }, (_, i) => metaRow('base', i)),
  );
  writeTwins(
    base,
    'heldout.jsonl',
    Array.from({ length: 4 }, (_, i) => metaRow('base-heldout', i)),
  );
  writeFileSync(
    join(base, 'manifest.json'),
    JSON.stringify({
      contract: contract(ids.base ?? 'dd+notes+think@24576'),
      train: 999,
    }),
  );
  const miss = (file: string, index: number) => ({
    ...metaRow(`miss-${file}`, index),
    file,
    index,
  });
  writeTwins(misses, 'misses.jsonl', [
    miss('conversations.jsonl', 0),
    miss('heldout.jsonl', 0),
    miss('conversations.jsonl', 4),
    // a second copy of index 0 (a resumed mine): the first copy is the one kept
    {
      ...miss('conversations.jsonl', 0),
      question: 'second copy?',
      answer: 'Answer: second copy',
    },
    miss('conversations.jsonl', 7),
  ]);
  writeFileSync(
    join(misses, 'manifest.json'),
    JSON.stringify({
      contract: contract(ids.misses ?? 'dd+notes+think@24576'),
      studentContract: { id: 'dd+notes@24576' },
      misses: 500,
    }),
  );
  writeTwins(
    other,
    'heldout.jsonl',
    Array.from({ length: 2 }, (_, i) => metaRow('other-heldout', i)),
  );
  writeFileSync(
    join(other, 'manifest.json'),
    JSON.stringify({ contract: contract(ids.other ?? 'dd+notes+think@24576') }),
  );
  return { root, base, misses, other };
}

function readLines(path: string): string[] {
  return readFileSync(path, 'utf8').split('\n').filter(Boolean);
}

describe('composeTraining', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps round(rows × (1 − share)) sampled base rows and fills the rest with misses', () => {
    const f = fixture();
    const out = join(f.root, 'out');
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {});
    const manifest = composeTraining({
      base: f.base,
      misses: f.misses,
      out,
      share: 0.25,
      rows: 8,
      seed: 3,
      now: new Date('2026-09-16T00:00:00Z'),
    });
    const lines = readLines(join(out, 'conversations.jsonl'));
    const meta = readLines(join(out, 'conversations.jsonl.meta.jsonl')).map(
      (l) => JSON.parse(l) as Record<string, unknown>,
    );
    expect(lines).toHaveLength(8);
    // 2 / 8 is the share asked for: no note
    expect(stderr).not.toHaveBeenCalled();
    expect(meta).toHaveLength(8);
    // twins aligned
    lines.forEach((l, i) => {
      const messages = (
        JSON.parse(l) as { messages: Array<{ content: string }> }
      ).messages;
      expect(messages.at(-1)!.content).toBe(meta[i]!.answer);
    });
    const baseRows = meta.filter((m) => m.source === 'base');
    const missRows = meta.filter((m) => m.source === 'miss');
    expect(baseRows).toHaveLength(6);
    expect(missRows).toHaveLength(2);
    // sampled without replacement
    expect(new Set(baseRows.map((m) => m.question)).size).toBe(6);
    // misses: never a heldout-file miss, and two distinct of the three deduplicated
    expect(missRows.every((m) => m.file === 'conversations.jsonl')).toBe(true);
    expect(new Set(missRows.map((m) => m.index)).size).toBe(2);
    expect(manifest).toMatchObject({
      share: 0.25,
      rows: 8,
      seed: 3,
      train: { base: 6, miss: 2 },
      heldout: 4,
      contract: contract('dd+notes+think@24576'),
      generatedAt: '2026-09-16T00:00:00.000Z',
      sources: {
        base: { path: f.base, contract: 'dd+notes+think@24576', rows: 10 },
        misses: {
          path: f.misses,
          contract: 'dd+notes+think@24576',
          rows: 5,
          heldoutDropped: 1,
          duplicatesDropped: 1,
          usable: 3,
        },
        heldout: { path: f.base, contract: 'dd+notes+think@24576' },
      },
    });
    const byType = (rows: Array<Record<string, unknown>>) => {
      const counts: Record<string, number> = {};
      for (const row of rows)
        counts[row.type as string] = (counts[row.type as string] ?? 0) + 1;
      return counts;
    };
    expect(manifest.byType).toEqual({
      base: byType(baseRows),
      miss: byType(missRows),
    });
    const written = JSON.parse(
      readFileSync(join(out, 'manifest.json'), 'utf8'),
    );
    expect(written).toEqual(JSON.parse(JSON.stringify(manifest)));
    // the base heldout copied byte for byte
    expect(readFileSync(join(out, 'heldout.jsonl'), 'utf8')).toBe(
      readFileSync(join(f.base, 'heldout.jsonl'), 'utf8'),
    );
    expect(readFileSync(join(out, 'heldout.jsonl.meta.jsonl'), 'utf8')).toBe(
      readFileSync(join(f.base, 'heldout.jsonl.meta.jsonl'), 'utf8'),
    );
  });

  it('keeps every base row when the base is short and cycles the misses to fill', () => {
    const f = fixture();
    const out = join(f.root, 'out');
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {});
    const manifest = composeTraining({
      base: f.base,
      misses: f.misses,
      out,
      share: 0.25,
      rows: 20,
      seed: 3,
    });
    expect(manifest.realisedShare).toBe(0.5);
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]![0])).toMatch(/0\.5.*0\.25/);
    const meta = readLines(join(out, 'conversations.jsonl.meta.jsonl')).map(
      (l) => JSON.parse(l) as Record<string, unknown>,
    );
    expect(meta).toHaveLength(20);
    expect(manifest.train).toEqual({ base: 10, miss: 10 });
    const uses = new Map<unknown, number>();
    const missRows = meta.filter((row) => row.source === 'miss');
    expect(missRows.every((m) => m.file === 'conversations.jsonl')).toBe(true);
    expect(missRows.some((m) => m.question === 'second copy?')).toBe(false);
    for (const m of missRows) uses.set(m.index, (uses.get(m.index) ?? 0) + 1);
    expect([...uses.keys()].sort()).toEqual([0, 4, 7]);
    expect([...uses.values()].sort()).toEqual([3, 3, 4]);
  });

  it('is reproducible from the seed', () => {
    const f = fixture();
    const run = (name: string, seed: number) => {
      const out = join(f.root, name);
      composeTraining({
        base: f.base,
        misses: f.misses,
        out,
        share: 0.25,
        rows: 8,
        seed,
      });
      return readFileSync(join(out, 'conversations.jsonl'), 'utf8');
    };
    expect(run('a', 11)).toBe(run('b', 11));
    expect(run('c', 11)).not.toBe(run('d', 12));
  });

  it('copies the heldout from --heldout when given', () => {
    const f = fixture();
    const out = join(f.root, 'out');
    const manifest = composeTraining({
      base: f.base,
      misses: f.misses,
      heldout: f.other,
      out,
      share: 0.25,
      rows: 8,
      seed: 1,
    });
    expect(readFileSync(join(out, 'heldout.jsonl'), 'utf8')).toBe(
      readFileSync(join(f.other, 'heldout.jsonl'), 'utf8'),
    );
    expect(manifest.heldout).toBe(2);
    expect(manifest.sources.heldout.path).toBe(f.other);
  });

  it('refuses sources with different contracts, naming each id', () => {
    const f = fixture({ misses: 'dd+notes@24576' });
    expect(() =>
      composeTraining({
        base: f.base,
        misses: f.misses,
        out: join(f.root, 'out'),
        share: 0.25,
        rows: 8,
        seed: 1,
      }),
    ).toThrow(
      /base .*dd\+notes\+think@24576.*misses .*dd\+notes@24576.*heldout .*dd\+notes\+think@24576/,
    );
    const g = fixture({ other: 'plain@24576' });
    expect(() =>
      composeTraining({
        base: g.base,
        misses: g.misses,
        heldout: g.other,
        out: join(g.root, 'out'),
        share: 0.25,
        rows: 8,
        seed: 1,
      }),
    ).toThrow(/heldout .*plain@24576/);
  });

  it('refuses a source without a contract and a fresh directory that already holds data', () => {
    const f = fixture();
    writeFileSync(
      join(f.misses, 'manifest.json'),
      JSON.stringify({ misses: 3 }),
    );
    expect(() =>
      composeTraining({
        base: f.base,
        misses: f.misses,
        out: join(f.root, 'out'),
        share: 0.25,
        rows: 8,
        seed: 1,
      }),
    ).toThrow(/misses .* no contract/);
    const g = fixture();
    const out = join(g.root, 'out');
    composeTraining({
      base: g.base,
      misses: g.misses,
      out,
      share: 0.25,
      rows: 8,
      seed: 1,
    });
    expect(() =>
      composeTraining({
        base: g.base,
        misses: g.misses,
        out,
        share: 0.25,
        rows: 8,
        seed: 1,
      }),
    ).toThrow(/already/);
  });
});
