import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  MemoryStore,
  captureRememberoVersion,
  createSemanticLedger,
  promoteRememberoReview,
  reviewRememberoCandidate,
} from '../src/index.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

describe.skipIf(nodeMajor < 22)('Remembero semantic review workflow', () => {
  let DatabaseSync: typeof import('node:sqlite').DatabaseSync;

  beforeAll(async () => {
    ({ DatabaseSync } = await import('node:sqlite'));
  });

  it('captures the complete Remembero version graph and promotes a reviewed candidate', () => {
    const database = new DatabaseSync(':memory:');
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'remembero-review-')));
    try {
      store.assert('default', 'status(atlas, active).', {
        opId: 'baseline-status',
        sourceText: 'Atlas is active.',
      });
      const ledger = createSemanticLedger(database);
      const baseline = captureRememberoVersion({
        ledger,
        store,
        namespaces: ['default'],
        label: 'remembero@baseline',
      });
      ledger.setRef({
        name: 'main',
        versionDigest: baseline.version.digest,
        operationId: 'test-initialize-main',
      });

      store.assert('default', 'owner(atlas, rahul).', {
        opId: 'candidate-owner',
        sourceText: 'Rahul owns Atlas.',
      });
      const candidate = captureRememberoVersion({
        ledger,
        store,
        namespaces: ['default'],
        parents: [baseline.version.digest],
        label: 'remembero@candidate',
      });
      expect(candidate.version.members.map((member) => member.key)).toEqual([
        'application',
        'documents',
        'evaluation-suite',
        'integrity-policy',
        'knowledge',
        'model',
        'rules',
        'runtime',
      ]);
      expect(candidate.version.edges).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ kind: 'produces', from: 'documents', to: 'knowledge' }),
          expect.objectContaining({ kind: 'evaluated-by', from: 'knowledge', to: 'evaluation-suite' }),
          expect.objectContaining({ kind: 'requires', from: 'runtime', to: 'model' }),
        ])
      );

      const review = reviewRememberoCandidate({
        ledger,
        store,
        baselineVersionDigest: baseline.version.digest,
        candidateVersionDigest: candidate.version.digest,
      });
      expect(review.evidence).toEqual(expect.arrayContaining([
        expect.objectContaining({
          kind: 'knowledge-diff',
          status: 'passed',
          evaluator: 'remembero.knowledge-diff',
        }),
        expect.objectContaining({
          kind: 'document-evaluation',
          status: 'passed',
          evaluator: 'remembero.document-showcase',
        }),
      ]));
      expect(review.assessment.checks).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ dimension: 'evaluation-quality', status: 'pass' }),
          expect.objectContaining({ dimension: 'policy-review', status: 'review' }),
        ])
      );

      const rejected = promoteRememberoReview({
        ledger,
        ref: 'main',
        candidateVersionDigest: candidate.version.digest,
        assessmentDigest: review.assessment.digest,
        operationId: 'test-reject-unreviewed',
        expectedCurrentVersionDigest: baseline.version.digest,
      });
      expect(rejected).toMatchObject({ outcome: 'rejected' });
      expect(ledger.getRef('main')?.versionDigest).toBe(baseline.version.digest);

      const acceptedReviewDimensions = review.assessment.checks
        .filter((check) => check.status === 'review')
        .map((check) => check.dimension);
      const accepted = promoteRememberoReview({
        ledger,
        ref: 'main',
        candidateVersionDigest: candidate.version.digest,
        assessmentDigest: review.assessment.digest,
        operationId: 'test-accept-reviewed',
        expectedCurrentVersionDigest: baseline.version.digest,
        acceptedReviewDimensions,
        reason: 'Reviewed the exact semantic diff and deterministic evidence.',
      });
      expect(accepted).toMatchObject({ outcome: 'accepted', blockingDimensions: [] });
      expect(ledger.getRef('main')?.versionDigest).toBe(candidate.version.digest);
    } finally {
      database.close();
    }
  });

  it('keeps blocked provider evidence from becoming a quality pass', () => {
    const database = new DatabaseSync(':memory:');
    try {
      const ledger = createSemanticLedger(database);
      const object = ledger.putObject({ kind: 'test', value: { version: 1 } });
      const version = ledger.createVersion({ members: [{ key: 'test', objectDigest: object.digest }] });
      ledger.setRef({ name: 'main', versionDigest: version.digest, operationId: 'blocked-main' });
      const evidence = ledger.recordEvidence({
        versionDigest: version.digest,
        kind: 'live-provider',
        status: 'blocked',
        evaluator: 'live-ocr',
        metrics: { completedDocuments: 0, qualityPercent: null },
        payload: { reason: 'provider unavailable' },
      });
      const assessment = ledger.recordCompatibility({
        baselineVersionDigest: version.digest,
        candidateVersionDigest: version.digest,
        checks: [{
          dimension: 'evaluation-quality',
          status: 'blocked',
          summary: 'No documents completed; no quality score exists.',
          evidenceDigests: [evidence.digest],
        }],
      });
      const decision = ledger.promote({
        ref: 'main',
        candidateVersionDigest: version.digest,
        assessmentDigest: assessment.digest,
        operationId: 'blocked-quality-promotion',
      });
      expect(decision).toMatchObject({ outcome: 'rejected', blockingDimensions: ['evaluation-quality'] });
    } finally {
      database.close();
    }
  });
});
