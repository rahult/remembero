import { randomUUID } from 'node:crypto';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, sep } from 'node:path';
import type { DatabaseSync } from 'node:sqlite';
import { MemoryStore } from './store.js';

const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NAMESPACE_FILE_RE = /^[a-z0-9_-]+\.dl$/;
const JOURNAL_SEGMENT_FILE_RE = /^journal-\d{12}-\d{12}-[a-f0-9]{64}\.jsonl$/;
const JOURNAL_CHECKPOINT_FILE_RE = /^checkpoint-\d{12}-[a-f0-9]{64}\.json$/;
const EXCLUDED_EXACT_PATHS = new Set([
  '.pending-mutation.before',
  '.pending-mutation.json',
  '.pending-mutation.next',
]);
const MUTATION_METHODS = new Set<keyof MemoryStore>([
  'note',
  'assert',
  'retract',
  'applyRuleChange',
  'applyMemoryChange',
  'supersede',
  'replace',
  'assertTentative',
  'importClauses',
  'resolveTentative',
  'compactJournal',
  'reserveAutoCapture',
  'finishAutoCapture',
  'recordAutoCaptureSkip',
  'recordAutoCaptureEmergency',
  'pruneAutoCaptureFacts',
]);

export interface SqliteMemoryStoreOptions {
  filesTableName?: string;
  metaTableName?: string;
  tempRootParent?: string;
}

export type SqliteMemoryStore = MemoryStore & { dispose(): void };

interface SqliteFileRow {
  path: string;
  content: Buffer;
}

interface SqliteTableColumn {
  name: string;
  type: string;
  notNull: number;
  pk: number;
}

const META_TABLE_SCHEMA: readonly SqliteTableColumn[] = [
  { name: 'key', type: 'TEXT', notNull: 0, pk: 1 },
  { name: 'int_value', type: 'INTEGER', notNull: 0, pk: 0 },
  { name: 'text_value', type: 'TEXT', notNull: 0, pk: 0 },
];

const FILES_TABLE_SCHEMA: readonly SqliteTableColumn[] = [
  { name: 'path', type: 'TEXT', notNull: 0, pk: 1 },
  { name: 'content', type: 'BLOB', notNull: 1, pk: 0 },
];

function quoteIdentifier(value: string, label: string): string {
  if (!IDENTIFIER_RE.test(value)) {
    throw new Error(`${label} must match ${IDENTIFIER_RE.source}`);
  }
  return `"${value}"`;
}

function normalizeRelativePath(path: string): string {
  const normalized = path.replaceAll('\\', '/');
  if (
    normalized.length === 0 ||
    normalized.startsWith('/') ||
    normalized.split('/').some((segment) => segment.length === 0 || segment === '.' || segment === '..')
  ) {
    throw new Error(`refusing invalid memory store path '${path}'`);
  }
  return normalized;
}

function isExcludedProjectionPath(path: string): boolean {
  return (
    path.endsWith('.lock') ||
    EXCLUDED_EXACT_PATHS.has(path) ||
    path.startsWith('.pending-mutation.json.tmp-') ||
    path === '.semantic-embeddings' ||
    path.startsWith('.semantic-embeddings/')
  );
}

function validateAuthoritativePath(path: string): string {
  const normalized = normalizeRelativePath(path);
  if (isExcludedProjectionPath(normalized)) {
    throw new Error(`excluded path '${normalized}' cannot be authoritative`);
  }
  if (
    normalized === 'journal.log' ||
    normalized === 'capture-errors.log' ||
    NAMESPACE_FILE_RE.test(normalized)
  ) {
    return normalized;
  }

  const [directory, file] = normalized.split('/');
  if (file === undefined) {
    throw new Error(`unexpected memory store artifact '${normalized}'`);
  }
  if (directory === '.journal-segments' && JOURNAL_SEGMENT_FILE_RE.test(file)) {
    return normalized;
  }
  if (directory === '.journal-checkpoints' && JOURNAL_CHECKPOINT_FILE_RE.test(file)) {
    return normalized;
  }
  throw new Error(`unexpected memory store artifact '${normalized}'`);
}

function coerceBlob(value: unknown, path: string): Buffer {
  if (Buffer.isBuffer(value)) return value;
  if (value instanceof Uint8Array) return Buffer.from(value);
  if (typeof value === 'string') return Buffer.from(value, 'utf8');
  throw new Error(`SQLite memory file '${path}' has unsupported content`);
}

export function createSqliteMemoryStore(
  database: DatabaseSync,
  options: SqliteMemoryStoreOptions = {}
): SqliteMemoryStore {
  const metaTableName = options.metaTableName ?? 'rembero_memory_store_meta';
  const filesTableName = options.filesTableName ?? 'rembero_memory_store_files';
  const metaTable = quoteIdentifier(
    metaTableName,
    'SQLite memory meta table name'
  );
  const filesTable = quoteIdentifier(
    filesTableName,
    'SQLite memory files table name'
  );
  const projectionParent = options.tempRootParent ?? tmpdir();
  mkdirSync(projectionParent, { recursive: true, mode: 0o700 });
  const projectionRoot = mkdtempSync(join(projectionParent, 'rembero-sqlite-store-'));

  let disposed = false;
  let lastRevision = -1;
  let store = new MemoryStore(projectionRoot);

  function assertNotDisposed(): void {
    if (disposed) throw new Error('SQLite memory store has been disposed');
  }

  function schemaEntry(tableName: string): { type: string } | undefined {
    const row = database
      .prepare(
        "SELECT type FROM sqlite_schema WHERE name = ? LIMIT 1"
      )
      .get(tableName) as { type?: string } | undefined;
    return row?.type === undefined ? undefined : { type: row.type };
  }

  function readTableSchema(tableName: string): SqliteTableColumn[] {
    const rows = database
      .prepare(
        "SELECT name, type, \"notnull\" AS nn, pk FROM pragma_table_info(?) ORDER BY cid"
      )
      .all(tableName) as Array<Record<string, unknown>>;
    return rows.map((row) => ({
      name: typeof row.name === 'string' ? row.name : '',
      type: typeof row.type === 'string' ? row.type : '',
      notNull: typeof row.nn === 'number' ? row.nn : Number.NaN,
      pk: typeof row.pk === 'number' ? row.pk : Number.NaN,
    }));
  }

  function validateTableSchema(
    tableName: string,
    expectedColumns: readonly SqliteTableColumn[]
  ): void {
    const entry = schemaEntry(tableName);
    if (entry === undefined) {
      throw new Error(`SQLite memory store table '${tableName}' is missing`);
    }
    if (entry.type !== 'table') {
      throw new Error(`SQLite memory store artifact '${tableName}' must be a table`);
    }
    const actual = readTableSchema(tableName);
    if (JSON.stringify(actual) !== JSON.stringify(expectedColumns)) {
      throw new Error(`SQLite memory store table '${tableName}' has an unexpected schema`);
    }
  }

  function hasRequiredMetaRows(): boolean {
    const rows = database
      .prepare(
        `SELECT key, int_value FROM ${metaTable}
         WHERE key IN ('schema_version', 'revision')
         ORDER BY key`
      )
      .all() as Array<{ key?: string; int_value?: number }>;
    const schemaVersion = rows.find((row) => row.key === 'schema_version')?.int_value;
    if (schemaVersion !== undefined && schemaVersion !== 1) {
      throw new Error(`unsupported SQLite memory store schema version ${schemaVersion}`);
    }
    return schemaVersion === 1 && rows.some((row) => row.key === 'revision');
  }

  function schemaReady(): boolean {
    const metaEntry = schemaEntry(metaTableName);
    const filesEntry = schemaEntry(filesTableName);
    if (metaEntry === undefined || filesEntry === undefined) {
      if (metaEntry !== undefined || filesEntry !== undefined) {
        throw new Error('SQLite memory store schema is partially initialized');
      }
      return false;
    }
    validateTableSchema(metaTableName, META_TABLE_SCHEMA);
    validateTableSchema(filesTableName, FILES_TABLE_SCHEMA);
    return hasRequiredMetaRows();
  }

  function ensureSchema(): void {
    assertNotDisposed();
    if (schemaReady()) return;
    const savepoint = `rembero_sqlite_schema_${randomUUID().replaceAll('-', '_')}`;
    database.exec(`SAVEPOINT ${savepoint}`);
    try {
      database.exec(
        `CREATE TABLE IF NOT EXISTS ${metaTable} (
          key TEXT PRIMARY KEY,
          int_value INTEGER,
          text_value TEXT
        )`
      );
      database.exec(
        `CREATE TABLE IF NOT EXISTS ${filesTable} (
          path TEXT PRIMARY KEY,
          content BLOB NOT NULL
        )`
      );
      database.exec(
        `INSERT OR IGNORE INTO ${metaTable}(key, int_value, text_value) VALUES
          ('schema_version', 1, NULL),
          ('revision', 0, NULL)`
      );
      database.exec(`RELEASE SAVEPOINT ${savepoint}`);
    } catch (error) {
      const cleanupFailures: string[] = [];
      try {
        database.exec(`ROLLBACK TO SAVEPOINT ${savepoint}`);
      } catch (rollbackError) {
        cleanupFailures.push(`schema rollback failed: ${String(rollbackError)}`);
      }
      try {
        database.exec(`RELEASE SAVEPOINT ${savepoint}`);
      } catch (releaseError) {
        cleanupFailures.push(`schema release failed: ${String(releaseError)}`);
      }
      if (error instanceof Error && cleanupFailures.length > 0) {
        error.message = `${error.message}; ${cleanupFailures.join('; ')}`;
        throw error;
      }
      if (cleanupFailures.length > 0) {
        throw new Error(`${String(error)}; ${cleanupFailures.join('; ')}`);
      }
      throw error;
    }
    if (!schemaReady()) {
      throw new Error('SQLite memory store schema initialization did not complete');
    }
  }

  function readRevision(): number {
    const row = database
      .prepare(`SELECT int_value FROM ${metaTable} WHERE key = 'revision'`)
      .get() as { int_value?: number } | undefined;
    const revision = row?.int_value;
    if (typeof revision !== 'number' || !Number.isSafeInteger(revision)) {
      throw new Error('SQLite memory store revision row is missing or invalid');
    }
    return revision;
  }

  function clearProjection(): void {
    rmSync(projectionRoot, { recursive: true, force: true });
    mkdirSync(projectionRoot, { recursive: true, mode: 0o700 });
  }

  function writeProjectionFile(path: string, content: Buffer): void {
    const absolute = join(projectionRoot, path);
    mkdirSync(dirname(absolute), { recursive: true, mode: 0o700 });
    writeFileSync(absolute, content, { mode: 0o600 });
  }

  function hydrateProjection(): void {
    clearProjection();
    const rows = database
      .prepare(`SELECT path, content FROM ${filesTable} ORDER BY path`)
      .all() as Array<{ path: unknown; content: unknown }>;
    for (const row of rows) {
      if (typeof row.path !== 'string') {
        throw new Error('SQLite memory store path row is invalid');
      }
      const path = validateAuthoritativePath(row.path);
      writeProjectionFile(path, coerceBlob(row.content, path));
    }
    store = new MemoryStore(projectionRoot);
    lastRevision = readRevision();
  }

  function ensureFreshProjection(): void {
    ensureSchema();
    const revision = readRevision();
    if (revision !== lastRevision) hydrateProjection();
  }

  function walkProjection(directory: string, output: SqliteFileRow[]): void {
    const entries = readdirSync(directory, { withFileTypes: true });
    for (const entry of entries) {
      const absolute = join(directory, entry.name);
      const relativePath = normalizeRelativePath(
        relative(projectionRoot, absolute).split(sep).join('/')
      );
      if (entry.isSymbolicLink()) {
        throw new Error(`refusing symbolic-link memory store artifact ${absolute}`);
      }
      if (entry.isDirectory()) {
        if (isExcludedProjectionPath(relativePath)) continue;
        walkProjection(absolute, output);
        continue;
      }
      if (!entry.isFile()) {
        throw new Error(`refusing unsupported memory store artifact ${absolute}`);
      }
      if (isExcludedProjectionPath(relativePath)) continue;
      const authoritativePath = validateAuthoritativePath(relativePath);
      const size = statSync(absolute).size;
      output.push({
        path: authoritativePath,
        content: size === 0 ? Buffer.alloc(0) : readFileSync(absolute),
      });
    }
  }

  function snapshotProjection(): SqliteFileRow[] {
    const rows: SqliteFileRow[] = [];
    walkProjection(projectionRoot, rows);
    rows.sort((left, right) => left.path.localeCompare(right.path));
    return rows;
  }

  function persistProjection(): void {
    const files = snapshotProjection();
    database.prepare(`DELETE FROM ${filesTable}`).run();
    const insert = database.prepare(
      `INSERT INTO ${filesTable}(path, content) VALUES (?, ?)`
    );
    for (const file of files) {
      insert.run(file.path, file.content);
    }
    database
      .prepare(
        `UPDATE ${metaTable}
         SET int_value = int_value + 1
         WHERE key = 'revision'`
      )
      .run();
  }

  function annotatePrimaryError(
    error: unknown,
    cleanupFailures: string[]
  ): never {
    if (cleanupFailures.length === 0) {
      throw error;
    }
    const suffix = cleanupFailures.join('; ');
    if (error instanceof Error) {
      error.message = `${error.message}; ${suffix}`;
      throw error;
    }
    throw new Error(`${String(error)}; ${suffix}`);
  }

  function rollbackSavepoint(name: string, label: string): string[] {
    const failures: string[] = [];
    try {
      database.exec(`ROLLBACK TO SAVEPOINT ${name}`);
    } catch (error) {
      failures.push(`${label} rollback failed: ${String(error)}`);
    }
    try {
      database.exec(`RELEASE SAVEPOINT ${name}`);
    } catch (error) {
      failures.push(`${label} release failed: ${String(error)}`);
    }
    return failures;
  }

  function withMutation<T>(operation: () => T): T {
    ensureSchema();
    const savepoint = `rembero_sqlite_store_${randomUUID().replaceAll('-', '_')}`;
    database.exec(`SAVEPOINT ${savepoint}`);
    try {
      database
        .prepare(
          `UPDATE ${metaTable}
           SET int_value = int_value
           WHERE key = 'revision'`
        )
        .run();
      hydrateProjection();
      const result = operation();
      persistProjection();
      database.exec(`RELEASE SAVEPOINT ${savepoint}`);
      lastRevision = readRevision();
      return result;
    } catch (error) {
      const cleanupFailures = rollbackSavepoint(savepoint, 'mutation');
      try {
        hydrateProjection();
      } catch (currentError) {
        cleanupFailures.push(`projection recovery failed: ${String(currentError)}`);
      }
      annotatePrimaryError(error, cleanupFailures);
    }
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    rmSync(projectionRoot, { recursive: true, force: true });
  }

  function currentStoreValue(property: PropertyKey): unknown {
    return (store as unknown as Record<PropertyKey, unknown>)[property];
  }

  function currentStoreMethod(property: string): (...args: unknown[]) => unknown {
    const value = currentStoreValue(property);
    if (typeof value !== 'function') {
      throw new Error(`memory store property '${property}' is not callable`);
    }
    return (value as (...args: unknown[]) => unknown).bind(store);
  }

  ensureSchema();
  hydrateProjection();

  const proxyTarget = Object.create(MemoryStore.prototype) as SqliteMemoryStore;

  return new Proxy(proxyTarget, {
    get(_target, property) {
      if (property === 'dispose') return dispose;
      if (typeof property !== 'string') {
        const value = currentStoreValue(property);
        return typeof value === 'function' ? (value as Function).bind(store) : value;
      }
      assertNotDisposed();
      if (property === 'createOperationId' || property === 'semanticEmbeddingCacheRoot') {
        const value = currentStoreValue(property);
        return typeof value === 'function' ? (value as Function).bind(store) : value;
      }
      if (MUTATION_METHODS.has(property as keyof MemoryStore)) {
        return (...args: unknown[]) => withMutation(() => currentStoreMethod(property)(...args));
      }
      return (...args: unknown[]) => {
        ensureFreshProjection();
        return currentStoreMethod(property)(...args);
      };
    },
    getPrototypeOf() {
      return MemoryStore.prototype;
    },
  }) as SqliteMemoryStore;
}
