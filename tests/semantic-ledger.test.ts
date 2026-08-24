import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  MemoryStore,
  captureKnowledgeVersion,
  createSemanticLedger,
  diffKnowledgeVersions,
} from '../src/index.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

describe.skipIf(nodeMajor < 22)('generic semantic ledger', () => {
  let DatabaseSync: typeof import('node:sqlite').DatabaseSync;

  beforeAll(async () => {
    ({ DatabaseSync } = await import('node:sqlite'));
  });

  function memoryStore(label: string): MemoryStore {
    return new MemoryStore(mkdtempSync(join(tmpdir(), `remembero-ledger-${label}-`)));
  }

  it('stores content-addressed objects and immutable typed version graphs', () => {
    const database = new DatabaseSync(':memory:');
    try {
      const ledger = createSemanticLedger(database);
      const application = ledger.putObject({
        kind: 'component',
        value: { name: 'memory-app', capabilities: ['state.read'] },
        createdAt: '2026-08-24T00:00:00.000Z',
      });
      const sameApplication = ledger.putObject({
        kind: 'component',
        value: { capabilities: ['state.read'], name: 'memory-app' },
        createdAt: '2026-08-24T01:00:00.000Z',
      });
      const runtime = ledger.putObject({
        kind: 'runtime',
        value: { name: 'node', version: '24' },
      });
      const contract = ledger.putObject({
        kind: 'contract',
        value: { requires: ['state.read'] },
      });

      expect(sameApplication.digest).toBe(application.digest);
      expect(sameApplication.createdAt).toBe('2026-08-24T00:00:00.000Z');

      const version = ledger.createVersion({
        label: 'memory-app@1.0.0',
        members: [
          { key: 'app', objectDigest: application.digest },
          { key: 'runtime', objectDigest: runtime.digest },
        ],
        edges: [{ kind: 'requires', from: 'app', to: 'runtime' }],
        contracts: [
          {
            key: 'runtime-contract',
            kind: 'capability',
            objectDigest: contract.digest,
            consumer: 'app',
            provider: 'runtime',
          },
        ],
        metadata: { product: 'remembero' },
      });

      expect(ledger.resolveVersion('memory-app@1.0.0')).toMatchObject({
        digest: version.digest,
        labels: ['memory-app@1.0.0'],
        members: [
          { key: 'app', objectKind: 'component' },
          { key: 'runtime', objectKind: 'runtime' },
        ],
        edges: [{ kind: 'requires', from: 'app', to: 'runtime' }],
        contracts: [{ key: 'runtime-contract', kind: 'capability' }],
      });
      expect(ledger.traverseGraph(version.digest, 'app')).toMatchObject({
        nodes: [
          { key: 'app', depth: 0 },
          { key: 'runtime', depth: 1 },
        ],
        edges: [{ kind: 'requires' }],
      });
      expect(() =>
        database
          .prepare('UPDATE remembero_semantic_objects SET kind = ? WHERE digest = ?')
          .run('changed', application.digest)
      ).toThrow(/immutable/);
    } finally {
      database.close();
    }
  });

  it('compares evidence and records rejected and accepted promotion decisions', () => {
    const database = new DatabaseSync(':memory:');
    try {
      const ledger = createSemanticLedger(database);
      const baselineObject = ledger.putObject({ kind: 'component', value: { version: 1 } });
      const candidateObject = ledger.putObject({ kind: 'component', value: { version: 2 } });
      const baseline = ledger.createVersion({
        members: [{ key: 'component', objectDigest: baselineObject.digest }],
      });
      const candidate = ledger.createVersion({
        parents: [baseline.digest],
        members: [{ key: 'component', objectDigest: candidateObject.digest }],
      });
      ledger.setRef({
        name: 'main',
        versionDigest: baseline.digest,
        operationId: 'initialize-main',
      });
      const baselineEvidence = ledger.recordEvidence({
        versionDigest: baseline.digest,
        kind: 'evaluation',
        status: 'passed',
        evaluator: 'document-suite',
        metrics: { accuracy: 1, cost_usd: 0.03, latency_ms: 20 },
      });
      const candidateEvidence = ledger.recordEvidence({
        versionDigest: candidate.digest,
        baselineVersionDigest: baseline.digest,
        kind: 'evaluation',
        status: 'passed',
        evaluator: 'document-suite',
        metrics: { accuracy: 1, cost_usd: 0.07, latency_ms: 24 },
      });
      const rejectedAssessment = ledger.recordCompatibility({
        baselineVersionDigest: baseline.digest,
        candidateVersionDigest: candidate.digest,
        checks: [
          {
            dimension: 'behavior',
            status: 'pass',
            summary: 'All deterministic answers and proofs remain stable.',
            evidenceDigests: [candidateEvidence.digest],
          },
          {
            dimension: 'cost',
            status: 'fail',
            summary: 'Provider cost increased above policy.',
            evidenceDigests: [baselineEvidence.digest, candidateEvidence.digest],
          },
        ],
      });
      const rejected = ledger.promote({
        ref: 'main',
        candidateVersionDigest: candidate.digest,
        assessmentDigest: rejectedAssessment.digest,
        operationId: 'reject-expensive-candidate',
      });

      expect(rejected).toMatchObject({
        outcome: 'rejected',
        blockingDimensions: ['cost'],
      });
      expect(ledger.getRef('main')?.versionDigest).toBe(baseline.digest);

      const blockedEvidence = ledger.recordEvidence({
        versionDigest: candidate.digest,
        baselineVersionDigest: baseline.digest,
        kind: 'ocr-quality',
        status: 'blocked',
        evaluator: 'live-provider',
        metrics: { completed_documents: 0, quality_score: null },
        payload: { reason: 'provider quota', attemptedDocuments: 4 },
      });
      const blockedAssessment = ledger.recordCompatibility({
        baselineVersionDigest: baseline.digest,
        candidateVersionDigest: candidate.digest,
        checks: [
          {
            dimension: 'ocr-quality',
            status: 'blocked',
            summary: 'No document completed, so no quality score exists.',
            evidenceDigests: [blockedEvidence.digest],
          },
        ],
      });
      expect(
        ledger.promote({
          ref: 'main',
          candidateVersionDigest: candidate.digest,
          assessmentDigest: blockedAssessment.digest,
          operationId: 'reject-blocked-quality',
        })
      ).toMatchObject({ outcome: 'rejected', blockingDimensions: ['ocr-quality'] });
      expect(ledger.getRef('main')?.versionDigest).toBe(baseline.digest);

      const acceptedAssessment = ledger.recordCompatibility({
        baselineVersionDigest: baseline.digest,
        candidateVersionDigest: candidate.digest,
        checks: [
          {
            dimension: 'behavior',
            status: 'pass',
            summary: 'All deterministic answers and proofs remain stable.',
            evidenceDigests: [candidateEvidence.digest],
          },
          {
            dimension: 'capability',
            status: 'review',
            summary: 'A human must accept the expanded runtime authority.',
          },
        ],
      });
      const accepted = ledger.promote({
        ref: 'main',
        candidateVersionDigest: candidate.digest,
        assessmentDigest: acceptedAssessment.digest,
        acceptedReviewDimensions: ['capability'],
        operationId: 'accept-reviewed-candidate',
      });

      expect(accepted).toMatchObject({
        outcome: 'accepted',
        acceptedReviewDimensions: ['capability'],
        blockingDimensions: [],
      });
      expect(ledger.getRef('main')?.versionDigest).toBe(candidate.digest);
      expect(ledger.refHistory('main')).toMatchObject([
        { operationId: 'initialize-main', toVersionDigest: baseline.digest },
        {
          operationId: 'promotion:accept-reviewed-candidate',
          fromVersionDigest: baseline.digest,
          toVersionDigest: candidate.digest,
          promotionDigest: accepted.digest,
        },
      ]);
      expect(
        ledger.promote({
          ref: 'main',
          candidateVersionDigest: candidate.digest,
          assessmentDigest: acceptedAssessment.digest,
          acceptedReviewDimensions: ['capability'],
          operationId: 'accept-reviewed-candidate',
        })
      ).toEqual(accepted);

      const diff = ledger.diffVersions(baseline.digest, candidate.digest);
      expect(diff).toMatchObject({
        changed: true,
        members: {
          changed: [{ key: 'component' }],
        },
        evidence: [
          {
            key: 'evaluation:document-suite',
            before: { status: 'passed' },
            after: { status: 'passed' },
            metricDelta: { accuracy: 0, cost_usd: 0.04000000000000001, latency_ms: 4 },
          },
          {
            key: 'ocr-quality:live-provider',
            after: { status: 'blocked', metrics: { quality_score: null } },
            metricDelta: { completed_documents: null, quality_score: null },
          },
        ],
        compatibility: { digest: acceptedAssessment.digest },
      });
    } finally {
      database.close();
    }
  });

  it('captures and semantically diffs exact Remembero knowledge versions', () => {
    const database = new DatabaseSync(':memory:');
    try {
      const ledger = createSemanticLedger(database);
      const store = memoryStore('knowledge');
      store.assert('default', 'employee(mira).', { opId: 'baseline' });
      const baseline = captureKnowledgeVersion(ledger, store, {
        namespaces: ['default'],
        label: 'knowledge@1',
      });

      store.assert(
        'default',
        'badge(mira). eligible(X) :- employee(X), badge(X).',
        { opId: 'candidate' }
      );
      const candidate = captureKnowledgeVersion(ledger, store, {
        namespaces: ['default'],
        parents: [baseline.version.digest],
        label: 'knowledge@2',
      });

      expect(baseline.recordedSnapshot).toMatchObject({ sequence: 1, journalEntries: 1 });
      expect(candidate.recordedSnapshot).toMatchObject({ sequence: 2, journalEntries: 2 });
      expect(ledger.diffVersions(baseline.version.digest, candidate.version.digest)).toMatchObject({
        members: { changed: [{ key: 'knowledge' }] },
      });
      expect(
        diffKnowledgeVersions(
          ledger,
          store,
          baseline.version.digest,
          candidate.version.digest,
          { query: 'eligible(mira)' }
        )
      ).toMatchObject({
        clauses: {
          added: [
            { kind: 'fact', clause: 'badge(mira).' },
            { kind: 'rule', clause: 'eligible(X) :- employee(X), badge(X).' },
          ],
        },
        queryImpact: {
          before: { rows: [] },
          after: { rows: [{ bindings: {} }] },
          added: [{ bindings: {} }],
        },
      });
      const repeated = captureKnowledgeVersion(ledger, store, {
        namespaces: ['default'],
        parents: [baseline.version.digest],
      });
      expect(repeated.knowledgeObject.digest).toBe(candidate.knowledgeObject.digest);
      expect(repeated.version.digest).toBe(candidate.version.digest);
    } finally {
      database.close();
    }
  });

  it('persists independently, supports caller transactions, and detects stale ref moves', () => {
    const path = join(
      mkdtempSync(join(tmpdir(), 'remembero-semantic-ledger-persistence-')),
      'ledger.db'
    );
    const firstDatabase = new DatabaseSync(path);
    const firstLedger = createSemanticLedger(firstDatabase, { tablePrefix: 'standalone_ledger' });
    const object = firstLedger.putObject({ kind: 'artifact', value: { name: 'portable' } });
    const version = firstLedger.createVersion({
      members: [{ key: 'artifact', objectDigest: object.digest }],
    });
    firstLedger.setRef({
      name: 'current',
      versionDigest: version.digest,
      operationId: 'current-v1',
    });

    firstDatabase.exec('BEGIN');
    firstLedger.putObject({ kind: 'artifact', value: { name: 'rolled-back' } });
    firstDatabase.exec('ROLLBACK');
    firstDatabase.close();

    const reopenedDatabase = new DatabaseSync(path);
    try {
      const reopened = createSemanticLedger(reopenedDatabase, {
        tablePrefix: 'standalone_ledger',
      });
      expect(reopened.resolveVersion('current').digest).toBe(version.digest);
      expect(() =>
        reopened.setRef({
          name: 'current',
          versionDigest: version.digest,
          operationId: 'stale-move',
          expectedCurrentVersionDigest: null,
        })
      ).toThrow(/no longer matches/);
      expect(() =>
        reopened.setRef({
          name: 'other',
          versionDigest: version.digest,
          operationId: 'current-v1',
        })
      ).toThrow(/reused/);
      const objectCount = reopenedDatabase
        .prepare('SELECT COUNT(*) AS count FROM standalone_ledger_objects')
        .get() as { count: number };
      expect(objectCount.count).toBe(1);
    } finally {
      reopenedDatabase.close();
    }
  });
});
