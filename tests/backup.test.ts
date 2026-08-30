import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { backupKnowledge, restoreKnowledge } from '../src/knowledge/backup.js';
import { MemoryStore } from '../src/store/store.js';

let sourceRoot: string;
let store: MemoryStore;
let backupPath: string;

beforeEach(() => {
  sourceRoot = mkdtempSync(join(tmpdir(), 'rembero-backup-'));
  store = new MemoryStore(sourceRoot);
  store.assert('personal', 'works_at(rahul, acme). city(rahul, sydney).', {
    sourceText: 'Rahul works at Acme in Sydney.',
  });
  store.assert('family', 'parent(alice, bob).');
  backupPath = join(mkdtempSync(join(tmpdir(), 'rembero-backup-out-')), 'backup.json');
});

describe('backup and restore', () => {
  it('writes a verified backup and restores it into an empty store', () => {
    const summary = backupKnowledge(store, backupPath);
    expect(summary.namespaceCount).toBe(2);
    expect(summary.clauseCount).toBe(3);

    const target = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-restore-')));
    const restored = restoreKnowledge(target, backupPath);
    expect(restored.namespaces.sort()).toEqual(['family', 'personal']);
    expect(restored.clausesAdded).toBe(3);
    expect(target.load('personal').length).toBe(2);
    expect(target.load('family').length).toBe(1);

    const again = restoreKnowledge(target, backupPath);
    expect(again.clausesAdded).toBe(0);
    expect(target.load('personal').length).toBe(2);
  });

  it('refuses to restore into a namespace that already has knowledge', () => {
    backupKnowledge(store, backupPath);
    const target = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-restore2-')));
    target.assert('personal', 'existing(fact).');
    expect(() => restoreKnowledge(target, backupPath)).toThrow(/non-empty namespace/i);
  });

  it('refuses a tampered backup file', () => {
    backupKnowledge(store, backupPath);
    const tampered = readFileSync(backupPath, 'utf8').replace('acme', 'evil');
    writeFileSync(backupPath, tampered, 'utf8');
    const target = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-restore3-')));
    expect(() => restoreKnowledge(target, backupPath)).toThrow();
  });
});
