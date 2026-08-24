import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { OperationConflictError } from '../src/store/store.js';
import { createSqliteMemoryStore, type SqliteMemoryStore } from '../src/store/sqlite-store.js';
import { serializeClause } from '../src/engine/index.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

describe.skipIf(nodeMajor < 22)('SQLite-backed MemoryStore adapter', () => {
  let DatabaseSync: typeof import('node:sqlite').DatabaseSync;

  beforeAll(async () => {
    ({ DatabaseSync } = await import('node:sqlite'));
  });

  function databasePath(label: string): string {
    return join(mkdtempSync(join(tmpdir(), `rembero-sqlite-store-${label}-`)), 'memory.db');
  }

  function closeStore(database: InstanceType<typeof DatabaseSync>, store: SqliteMemoryStore): void {
    store.dispose();
    database.close();
  }

  it('reopens durable memory from SQLite without relying on a sidecar projection', () => {
    const path = databasePath('durable');
    const firstDatabase = new DatabaseSync(path);
    const firstStore = createSqliteMemoryStore(firstDatabase);

    firstStore.assert('default', 'works_at(mira, acme).', { opId: 'seed' });
    firstStore.replace('default', ['works_at(mira, _)'], 'works_at(mira, initech).', {
      opId: 'replace',
    });

    closeStore(firstDatabase, firstStore);

    const reopenedDatabase = new DatabaseSync(path);
    const reopenedStore = createSqliteMemoryStore(reopenedDatabase);
    try {
      expect(reopenedStore.load('default').map(serializeClause)).toEqual([
        'works_at(mira, initech).',
      ]);
      expect(reopenedStore.recordedSnapshot(['default'], 2).clauses.map(serializeClause)).toEqual([
        'works_at(mira, initech).',
      ]);
      expect(reopenedStore.history('works_at(mira, _)').events.map((event) => event.action)).toEqual([
        'asserted',
        'retracted',
        'asserted',
      ]);
    } finally {
      closeStore(reopenedDatabase, reopenedStore);
    }
  });

  it('commits and rolls back memory atomically with caller-owned SQLite transactions', () => {
    const path = databasePath('transaction');
    const database = new DatabaseSync(path);
    const store = createSqliteMemoryStore(database);
    try {
      database.exec('CREATE TABLE app_events(id INTEGER PRIMARY KEY, value TEXT NOT NULL)');

      database.exec('BEGIN');
      database.prepare('INSERT INTO app_events(value) VALUES (?)').run('committed');
      store.assert('default', 'status(mira, active).', { opId: 'txn-commit' });
      database.exec('COMMIT');

      expect(store.load('default').map(serializeClause)).toEqual(['status(mira, active).']);
      expect(
        database.prepare('SELECT value FROM app_events ORDER BY id').all()
      ).toEqual([{ value: 'committed' }]);

      database.exec('BEGIN');
      database.prepare('INSERT INTO app_events(value) VALUES (?)').run('rolled-back');
      store.assert('default', 'status(mira, paused).', { opId: 'txn-rollback' });
      database.exec('ROLLBACK');

      expect(store.load('default').map(serializeClause)).toEqual(['status(mira, active).']);
      expect(
        database.prepare('SELECT value FROM app_events ORDER BY id').all()
      ).toEqual([{ value: 'committed' }]);
    } finally {
      closeStore(database, store);
    }
  });

  it('preserves retry-safe opIds plus tentative/history behavior through the SQLite authority layer', () => {
    const path = databasePath('behavior');
    const database = new DatabaseSync(path);
    const store = createSqliteMemoryStore(database);
    try {
      const first = store.assertTentative('personal', 'status(mira, active).', {
        opId: 'tentative',
      });
      expect(first.added).toHaveLength(1);
      expect(store.load('personal').map(serializeClause)).toEqual([
        "rembero_tentative('status(mira, active).').",
      ]);

      const accepted = store.resolveTentative(
        'personal',
        'status(mira, active).',
        'accept',
        { opId: 'accept-status' }
      );
      expect(accepted.added.map(serializeClause)).toEqual(['status(mira, active).']);
      expect(
        store.resolveTentative('personal', 'status(mira, active).', 'accept', {
          opId: 'accept-status',
        })
      ).toEqual(accepted);
      expect(() =>
        store.resolveTentative('personal', 'status(mira, active).', 'reject', {
          opId: 'accept-status',
        })
      ).toThrow(OperationConflictError);

      expect(store.history('status(mira, active)', { namespaces: ['personal'] })).toMatchObject({
        events: [expect.objectContaining({ action: 'asserted', trustAction: 'accept' })],
      });
      expect(store.sourcesFor(['personal']).get('status(mira, active).')).toMatchObject([
        { opId: 'accept-status', trustAction: 'accept' },
      ]);
    } finally {
      closeStore(database, store);
    }
  });

  it('refreshes reads across connections when the SQLite revision changes', () => {
    const path = databasePath('connections');
    const firstDatabase = new DatabaseSync(path);
    const secondDatabase = new DatabaseSync(path);
    const firstStore = createSqliteMemoryStore(firstDatabase);
    const secondStore = createSqliteMemoryStore(secondDatabase);

    try {
      firstStore.assert('default', 'works_at(mira, acme).', { opId: 'first' });
      expect(secondStore.load('default').map(serializeClause)).toEqual([
        'works_at(mira, acme).',
      ]);

      secondStore.assert('default', 'works_at(rahul, acme).', { opId: 'second' });
      expect(firstStore.load('default').map(serializeClause).sort()).toEqual([
        'works_at(mira, acme).',
        'works_at(rahul, acme).',
      ]);

      firstDatabase.exec('BEGIN');
      firstStore.assert('default', 'works_at(chen, initech).', { opId: 'third' });
      expect(secondStore.load('default').map(serializeClause).sort()).toEqual([
        'works_at(mira, acme).',
        'works_at(rahul, acme).',
      ]);
      firstDatabase.exec('COMMIT');

      expect(secondStore.load('default').map(serializeClause).sort()).toEqual([
        'works_at(chen, initech).',
        'works_at(mira, acme).',
        'works_at(rahul, acme).',
      ]);
    } finally {
      closeStore(firstDatabase, firstStore);
      closeStore(secondDatabase, secondStore);
    }
  });

  it('initializes shared schema without changing caller busy_timeout on either connection', () => {
    const path = databasePath('init-shared');
    const firstDatabase = new DatabaseSync(path);
    const secondDatabase = new DatabaseSync(path);
    firstDatabase.exec('PRAGMA busy_timeout = 111');
    secondDatabase.exec('PRAGMA busy_timeout = 222');

    const firstStore = createSqliteMemoryStore(firstDatabase);
    const secondStore = createSqliteMemoryStore(secondDatabase);

    try {
      expect(firstDatabase.prepare('PRAGMA busy_timeout').get()).toEqual({ timeout: 111 });
      expect(secondDatabase.prepare('PRAGMA busy_timeout').get()).toEqual({ timeout: 222 });

      firstStore.assert('default', 'status(mira, active).', { opId: 'init-shared-first' });
      expect(secondStore.load('default').map(serializeClause)).toEqual([
        'status(mira, active).',
      ]);

      const tables = secondDatabase
        .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
        .all();
      expect(tables).toEqual([
        { name: 'rembero_memory_store_files' },
        { name: 'rembero_memory_store_meta' },
      ]);
    } finally {
      closeStore(firstDatabase, firstStore);
      closeStore(secondDatabase, secondStore);
    }
  });
});
