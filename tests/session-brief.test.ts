import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildSessionBrief } from '../src/autocapture/session-brief.js';
import { MemoryStore } from '../src/store/store.js';

let storeRoot: string;
let store: MemoryStore;

beforeEach(() => {
  storeRoot = mkdtempSync(join(tmpdir(), 'rembero-brief-'));
  store = new MemoryStore(storeRoot);
});

describe('journal append caching', () => {
  it('keeps interleaved writers from two store instances consistent', () => {
    const second = new MemoryStore(storeRoot);
    store.assert('personal', 'a(1).');
    second.assert('personal', 'b(2).');
    store.assert('personal', 'c(3).');
    second.assert('personal', 'd(4).');

    const replayed = new MemoryStore(storeRoot);
    expect(replayed.load('personal').length).toBe(4);
    const history = readFileSync(join(storeRoot, 'journal.log'), 'utf8')
      .split('\n')
      .filter((line) => line.trim() !== '')
      .map((line) => JSON.parse(line) as { op: string; added?: string[] })
      .filter((entry) => entry.op === 'assert')
      .map((entry) => entry.added?.[0]);
    expect(history).toEqual(['a(1).', 'b(2).', 'c(3).', 'd(4).']);
  });
});

describe('session-start memory brief', () => {
  it('returns an empty brief for an empty namespace', () => {
    expect(buildSessionBrief(store, 'personal')).toBe('');
  });

  it('summarizes stored predicates with bounded samples and tool guidance', () => {
    store.assert('personal', 'works_at(rahul, acme). works_at(mira, acme).');
    store.assert(
      'personal',
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.'
    );
    const brief = buildSessionBrief(store, 'personal');

    expect(brief).toContain("namespace 'personal'");
    expect(brief).toContain('works_at(rahul, acme).');
    expect(brief).toContain('1 rule');
    expect(brief).toContain('remember');
    expect(Buffer.byteLength(brief, 'utf8')).toBeLessThanOrEqual(4096);
  });

  it('stays within the byte budget and reports elided predicates on large stores', () => {
    const facts: string[] = [];
    for (let index = 0; index < 200; index += 1) {
      facts.push(`predicate_number_${index}(subject_${index}, some_longer_atom_value_${index}).`);
    }
    store.assert('personal', facts.join(' '));
    const brief = buildSessionBrief(store, 'personal', { maxBytes: 2048 });

    expect(Buffer.byteLength(brief, 'utf8')).toBeLessThanOrEqual(2048);
    expect(brief).toMatch(/\d+ more predicate/);
  });
});
