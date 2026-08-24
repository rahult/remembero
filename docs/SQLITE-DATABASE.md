# Use Remembero as SQLite plus governed memory

`openRememberoDatabase(...)` opens an ordinary SQLite file and returns the real
`node:sqlite` `DatabaseSync` connection. The connection keeps its normal SQL, prepared
statement, transaction, function, backup, and inspection APIs, and adds:

- `datalogSql`, `datalogQuery`, `datalogExplain`, and `datalogPlan` over ordinary tables;
- `memory`, a `MemoryStore`-compatible governed-memory API whose current clauses,
  provenance journal, checkpoints, trust transitions, and recorded history are durable in
  the same SQLite file.

SQLite remains the database engine and transaction authority. Remembero does not
reimplement SQLite or introduce a new database file format.

## Install and open

The database API needs Node.js 22.13 or newer. The native extension currently supports
macOS and Linux and must be compiled once after installing:

```bash
npm install remembero
npx remembero sqlite-build
```

Open a new or existing database:

```ts
import {
  explainKnowledge,
  openRememberoDatabase,
} from 'remembero';

const db = await openRememberoDatabase('app.db');
db.exec(`
  CREATE TABLE IF NOT EXISTS works_at(
    person TEXT NOT NULL,
    company TEXT NOT NULL
  )
`);

const insert = db.prepare(
  'INSERT INTO works_at(person, company) VALUES (?, ?)'
);
insert.run('mira', 'acme');

const colleagues = db.datalogQuery(`
  colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
`);

db.memory.assert('default', 'prefers(mira, tea).', {
  opId: 'preference-2026-08-24',
});
const explanation = explainKnowledge(
  db.memory.clausesFor('*'),
  'prefers(mira, Choice)',
  db.memory.sourcesFor('*')
);

db.close();
```

No relational-data migration is required. An existing `.db` file remains readable by
stock SQLite after Remembero opens it.

## One transaction for application data and memory

Memory mutations use nested SQLite savepoints. A caller-owned transaction therefore
commits or rolls back application rows and governed memory together:

```ts
db.exec('BEGIN IMMEDIATE');
try {
  db.prepare('UPDATE accounts SET plan = ? WHERE id = ?').run('pro', 42);
  db.memory.assert('audit', 'plan_changed(42, pro).', {
    opId: 'account-42-plan-pro',
  });
  db.exec('COMMIT');
} catch (error) {
  db.exec('ROLLBACK');
  throw error;
}
```

## Use the same authority through MCP

The SQLite-backed store implements the same `MemoryStore` contract as the portable
file-backed store. Pass it to the MCP server when an embedded application wants every
Remembero tool to read and mutate the same database:

```ts
import { createServer, openRememberoDatabase } from 'remembero';

const db = await openRememberoDatabase('app.db');
const mcp = createServer({ store: db.memory, llm });
```

Keep `db` open for the lifetime of the MCP server, then close the server before closing
the database. This is explicit application wiring; the standalone `remembero mcp`
command retains its portable file-backed default.

The internal `rembero_memory_store_meta` and `rembero_memory_store_files` tables are
implementation details. Do not edit them directly. Multiple connections observe changes
through a revision gate; normal SQLite locking and the connection's configured busy
timeout still govern write contention.

## Exact boundary

- The returned object is a native `DatabaseSync`, not a partial SQL wrapper.
- Further dynamic extension loading is disabled after Remembero initializes the connection.
- Positive and recursive rules use the native extension. Negation, arithmetic,
  aggregation, raw conjunctions, and multiple derived predicates use the documented
  bounded portable evaluator over a SQLite read snapshot.
- Portable evaluation intentionally rejects `NULL`, BLOB, non-finite numeric values, and
  unsafe integers in referenced relations; it uses Datalog equality rather than SQLite
  affinity or collation.
- The semantic embedding cache is derived and is not stored in the SQLite authority. It
  may be rebuilt after reopening without changing facts, proofs, history, or provenance.
- This API is a Node/local database integration. It is not a replacement SQLite binary,
  C ABI, wire protocol, or compatibility claim for arbitrary third-party SQLite drivers.

Use `openDatalogDatabase(...)` when an integration needs only the legacy Datalog wrapper
and must not create the governed-memory tables.
