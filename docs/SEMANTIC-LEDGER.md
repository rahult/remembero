# Immutable semantic ledger

Remembero's semantic ledger is a generic, SQLite-backed version graph for content-addressed
objects, typed dependencies, contracts, evidence, compatibility assessments, and promotion
decisions. It can version Remembero knowledge, but it is not tied to the memory store: any local
application can use the ledger directly, and a larger software platform can compose the same
public API without becoming part of Remembero.

The ledger uses ordinary public tables with a configurable prefix. It never reads or writes the
internal `rembero_memory_store_*` tables.

## Standalone use

The ledger needs a `node:sqlite` `DatabaseSync` connection. It does not require the native Datalog
extension, a `MemoryStore`, an LLM, DuckDB, or an external service.

```ts
import { DatabaseSync } from 'node:sqlite';
import { createSemanticLedger } from 'remembero';

const db = new DatabaseSync('versions.db');
const ledger = createSemanticLedger(db);

const app = ledger.putObject({
  kind: 'component',
  value: { name: 'notes', schemaVersion: 1 },
});
const runtime = ledger.putObject({
  kind: 'runtime',
  value: { name: 'node', version: '24' },
});

const version = ledger.createVersion({
  label: 'notes@1.0.0',
  members: [
    { key: 'app', objectDigest: app.digest },
    { key: 'runtime', objectDigest: runtime.digest },
  ],
  edges: [{ kind: 'requires', from: 'app', to: 'runtime' }],
});

ledger.setRef({
  name: 'main',
  versionDigest: version.digest,
  operationId: 'initialize-main',
});
```

Object digests cover both the object kind and canonical JSON value. Version digests cover the
ordered parent list, keyed members, typed edges, contracts, and metadata. Creation timestamps and
human labels do not change the exact state digest. Repeating the same content is idempotent.

## Version graph

A version contains:

- one or more keyed object members;
- zero or more immutable parent versions;
- typed directed edges whose endpoints are member keys;
- first-class contracts connecting consumer and provider members; and
- canonical metadata owned by the caller.

Edge and contract kinds are deliberately open but validated identifiers. Applications may use
vocabularies such as `requires`, `consumes`, `produces`, `emits`, `reads`, `writes`, or
`evaluated-by` without Remembero assigning platform-specific meaning to them.

`traverseGraph(...)` performs a bounded upstream, downstream, or bidirectional walk.
`diffVersions(...)` reports member, edge, and contract changes, then compares the latest evidence
for each evaluator and returns numeric metric deltas.

## Evidence and compatibility

`recordEvidence(...)` appends a content-addressed execution or evaluation envelope. Evidence has
an explicit status (`passed`, `failed`, `blocked`, `error`, or `observed`), optional baseline,
canonical payload, and finite numeric metrics. Missing measurements should be `null`; callers
must not turn a zero denominator or blocked provider into a passing score.

`recordCompatibility(...)` records a vector of named checks. Each dimension is one of:

- `pass`;
- `fail`;
- `review`;
- `blocked`; or
- `not_applicable`.

`promote(...)` always records an immutable accepted or rejected decision. `fail` and `blocked`
dimensions reject promotion. A `review` dimension rejects promotion unless that exact dimension
is explicitly accepted in the promotion request. Accepted promotion moves the selected ref in
the same SQLite transaction.

## Refs and history

Version objects never move. A ref such as `main`, `current`, or `experiment/model-x` is a mutable
pointer, and every movement creates an immutable ref event with a caller-supplied operation ID.
Operation IDs make retries idempotent and reject reuse for another movement. Optional expected
current digests provide compare-and-set protection against stale promotion.

Use ordinary refs for experiments and `promote(...)` for policy-gated refs. `resolveVersion(...)`
accepts an exact digest, immutable human label, or current ref.

## Capture Remembero knowledge

`captureKnowledgeVersion(...)` is the optional bridge from governed memory into the generic
ledger:

```ts
import {
  captureKnowledgeVersion,
  createSemanticLedger,
  diffKnowledgeVersions,
  openRememberoDatabase,
} from 'remembero';

const db = await openRememberoDatabase('remembero.db');
const ledger = createSemanticLedger(db);

db.memory.assert('default', 'status(mira, active).', { opId: 'baseline' });
const v1 = captureKnowledgeVersion(ledger, db.memory, {
  namespaces: ['default'],
  label: 'knowledge@1',
});

db.memory.assert('default', 'employee(mira).', { opId: 'employee' });
const v2 = captureKnowledgeVersion(ledger, db.memory, {
  namespaces: ['default'],
  parents: [v1.version.digest],
  label: 'knowledge@2',
});

const impact = diffKnowledgeVersions(
  ledger,
  db.memory,
  v1.version.digest,
  v2.version.digest,
  { query: 'employee(mira)' }
);
```

The captured object contains the exact recorded sequence, journal length, selected namespaces,
canonical clauses, durable sources, and knowledge-program digest. The generic version can also
contain application, schema, runtime, model, or policy objects supplied by the caller.

## Remembero product review adapter

`captureRememberoVersion(...)` composes the knowledge snapshot with stable Remembero members:

```text
application
documents
evaluation-suite
integrity-policy
knowledge
model
rules
runtime
```

It adds typed edges for document production, rule consumption, evaluation, integrity checks, and
runtime/model requirements. The adapter does not mutate memory. `reviewRememberoCandidate(...)`
records deterministic knowledge-diff and document-showcase evidence, then records a compatibility
vector covering schema, lineage, integrity behavior, evaluation quality, provider status, cost,
latency, and human policy review. `promoteRememberoReview(...)` delegates the final ref mutation
to the generic `promote(...)` gate.

The CLI exposes this adapter as `remembero version capture|list|inspect|diff|review|history|promote`.
The MCP server exposes the corresponding semantic-version tools, and the local web console's
**Versions** workspace is a human review surface over the same APIs. The dark-factory worker,
queue, coding harness, and autonomous promotion loop are intentionally not part of this layer.

Remembero continues to operate normally without this bridge. The portable file store, SQLite
memory store, CLI, MCP server, and reasoning engine do not depend on semantic-ledger tables.

## Authority and operational boundary

- SQLite remains the transaction and file-format authority.
- Immutable tables reject normal `UPDATE` and `DELETE` statements with guard triggers.
- An owner with unrestricted write access to the SQLite file can still drop triggers or rewrite
  bytes; this is application-level immutability plus digest verification, not protection from the
  database owner.
- Refs are the only mutable ledger records; their complete movement history is append-only.
- The ledger owns no deployment, source-control, process execution, or agent authority.
- Deterministic Remembero history can be replayed by recorded sequence. Non-deterministic model or
  OCR runs are evidence to compare or rerun, not promises of byte-identical replay.
- DuckDB is intentionally absent from the authority path. A future analytics adapter may read or
  export evidence for large comparisons, but promotion must continue to reference immutable
  SQLite evidence digests.

Downstream systems should add their domain objects and policies through these generic contracts
rather than adding their concepts to Remembero's core schema.
