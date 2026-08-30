import type { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { evaluateDocumentShowcases } from '../evals/document-showcase.js';
import { diffKnowledgeVersions } from './knowledge-version.js';
import type { MemoryStore } from '../store/store.js';
import type { SemanticLedger, SemanticVersionDiff, SemanticEvidence, SemanticCompatibilityAssessment, SemanticPromotionDecision } from './semantic-ledger.js';

export interface RememberoReviewSummary {
  baselineVersionDigest?: string;
  candidateVersionDigest: string;
  diff: SemanticVersionDiff;
  evidence: SemanticEvidence[];
  assessment: SemanticCompatibilityAssessment;
  knowledgeDiff?: ReturnType<typeof diffKnowledgeVersions>;
}

export interface ReviewCandidateOptions {
  ledger: SemanticLedger;
  store?: MemoryStore;
  baselineVersionDigest?: string;
  candidateVersionDigest: string;
  query?: string;
  includeDocumentEvaluation?: boolean;
  metadata?: unknown;
}

export interface PromoteReviewOptions {
  ledger: SemanticLedger;
  ref: string;
  candidateVersionDigest: string;
  assessmentDigest: string;
  operationId: string;
  expectedCurrentVersionDigest?: string | null;
  acceptedReviewDimensions?: string[];
  reason?: string;
}

export interface OpenSemanticLedgerResult {
  database: DatabaseSync;
  ledger: SemanticLedger;
}

/** Open the SQLite authority used by the semantic review surface. */
export async function openSemanticLedger(path: string): Promise<OpenSemanticLedgerResult> {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  const { DatabaseSync } = await import('node:sqlite');
  const database = new DatabaseSync(path);
  const { createSemanticLedger } = await import('./semantic-ledger.js');
  return { database, ledger: createSemanticLedger(database) };
}

function isMissingSqliteBuiltin(error: unknown): boolean {
  return (
    (error as NodeJS.ErrnoException)?.code === 'ERR_UNKNOWN_BUILTIN_MODULE' ||
    /No such built-in module: node:sqlite/.test(
      error instanceof Error ? error.message : String(error)
    )
  );
}

/**
 * Open the semantic ledger when this Node runtime ships `node:sqlite`
 * (Node 22+), or return undefined so memory surfaces degrade gracefully:
 * the MCP server and web console run without the semantic version authority
 * rather than refusing to start on Node 20.
 */
export async function openSemanticLedgerIfSupported(
  path: string
): Promise<OpenSemanticLedgerResult | undefined> {
  try {
    return await openSemanticLedger(path);
  } catch (error) {
    if (isMissingSqliteBuiltin(error)) return undefined;
    throw error;
  }
}

function metricPercent(metric: { percent?: unknown; total?: unknown }): number | null {
  return typeof metric.total === 'number' && metric.total > 0 && typeof metric.percent === 'number'
    ? metric.percent
    : null;
}

/**
 * Evaluate a candidate with deterministic built-in evidence and produce the
 * compatibility vector consumed by the ledger promotion gate.
 */
export function reviewRememberoCandidate(
  options: ReviewCandidateOptions
): RememberoReviewSummary {
  const diff = options.ledger.diffVersions(
    options.baselineVersionDigest ?? options.candidateVersionDigest,
    options.candidateVersionDigest
  );
  const evidence: SemanticEvidence[] = [];
  const knowledgeDiff = options.store !== undefined && options.baselineVersionDigest !== undefined
    ? diffKnowledgeVersions(
        options.ledger,
        options.store,
        options.baselineVersionDigest,
        options.candidateVersionDigest,
        options.query === undefined ? {} : { query: options.query }
      )
    : undefined;
  if (knowledgeDiff !== undefined) {
    const integrityPassed = knowledgeDiff.integrity.after.status !== 'violations';
    evidence.push(options.ledger.recordEvidence({
      versionDigest: options.candidateVersionDigest,
      baselineVersionDigest: options.baselineVersionDigest,
      kind: 'knowledge-diff',
      status: integrityPassed ? 'passed' : 'failed',
      evaluator: 'remembero.knowledge-diff',
      payload: {
        changed: knowledgeDiff.changed,
        clauses: knowledgeDiff.clauses,
        integrityDelta: knowledgeDiff.integrity,
        ...(knowledgeDiff.queryImpact === undefined ? {} : { queryImpact: knowledgeDiff.queryImpact }),
      },
      metrics: {
        addedClauses: knowledgeDiff.clauses.added.length,
        removedClauses: knowledgeDiff.clauses.removed.length,
        candidateIntegrityViolations: knowledgeDiff.integrity.after.violationCount,
      },
    }));
  }
  const documentEvaluation = options.includeDocumentEvaluation === false
    ? undefined
    : evaluateDocumentShowcases();
  if (documentEvaluation !== undefined) {
    const aggregate = documentEvaluation.aggregate;
    evidence.push(options.ledger.recordEvidence({
      versionDigest: options.candidateVersionDigest,
      ...(options.baselineVersionDigest === undefined
        ? {}
        : { baselineVersionDigest: options.baselineVersionDigest }),
      kind: 'document-evaluation',
      status: aggregate.status === 'pass' ? 'passed' : 'failed',
      evaluator: 'remembero.document-showcase',
      payload: aggregate,
      metrics: {
        parseCoveragePercent: metricPercent(aggregate.metrics.parseCoverage),
        answerAccuracyPercent: metricPercent(aggregate.metrics.answerAccuracy),
        sourceRecallPercent: metricPercent(aggregate.metrics.sourceRecall),
        proofGroundingPercent: metricPercent(aggregate.metrics.proofGrounding),
        abstentionCorrectnessPercent: metricPercent(aggregate.metrics.abstentionCorrectness),
        idempotencyPercent: metricPercent(aggregate.metrics.idempotency),
        averageQuestionMs: aggregate.latencyMs.averageQuestionMs,
        totalMs: aggregate.latencyMs.totalMs,
      },
    }));
  }
  const evidenceDigests = evidence.map((entry) => entry.digest);
  const qualityPassed = documentEvaluation?.aggregate.status === 'pass';
  const qualityStatus = documentEvaluation === undefined
    ? 'not_applicable' as const
    : qualityPassed
      ? 'pass' as const
      : 'fail' as const;
  const checks = [
    {
      dimension: 'knowledge-schema',
      status: diff.members.changed.length === 0 ? 'pass' as const : 'review' as const,
      summary: diff.members.changed.length === 0
        ? 'Knowledge member shape is unchanged.'
        : `${diff.members.changed.length} semantic member(s) changed.`,
    },
    {
      dimension: 'document-source-lineage',
      status: diff.members.added.some((entry) => entry.key === 'documents') ||
        diff.members.changed.some((entry) => entry.key === 'documents')
        ? 'review' as const
        : 'pass' as const,
      summary: 'Document/source objects are bound to the candidate version.',
    },
    {
      dimension: 'rule-integrity-behavior',
      status: knowledgeDiff !== undefined && knowledgeDiff.integrity.after.status === 'violations'
        ? 'fail' as const
        : diff.members.added.some((entry) => entry.key === 'rules' || entry.key === 'integrity-policy') ||
        diff.members.changed.some((entry) => entry.key === 'rules' || entry.key === 'integrity-policy')
        ? 'review' as const
        : 'pass' as const,
      summary: knowledgeDiff !== undefined && knowledgeDiff.integrity.after.status === 'violations'
        ? 'The candidate introduces an integrity violation.'
        : 'Rule and integrity changes require explicit review when present.',
    },
    {
      dimension: 'proof-query-regression',
      status: qualityStatus,
      summary: documentEvaluation === undefined
        ? 'Document evaluation was not requested for this review.'
        : qualityPassed
        ? 'The deterministic document proof suite passed.'
        : 'The deterministic document proof suite failed.',
    },
    {
      dimension: 'evaluation-quality',
      status: qualityStatus,
      summary: documentEvaluation === undefined
        ? 'Document evaluation was not requested for this review.'
        : qualityPassed
        ? 'The evaluation suite completed with passing metrics.'
        : 'The evaluation suite did not produce a passing result.',
      evidenceDigests,
    },
    {
      dimension: 'model-provider',
      status: diff.members.changed.some((entry) => entry.key === 'model') ? 'review' as const : 'not_applicable' as const,
      summary: 'Provider changes are reviewable but do not become authority automatically.',
    },
    {
      dimension: 'latency-cost',
      status: 'not_applicable' as const,
      summary: 'No cost or latency threshold is configured for this local deterministic review.',
    },
    {
      dimension: 'policy-review',
      status: 'review' as const,
      summary: 'A human must review the exact semantic diff before promotion.',
    },
  ];
  const assessment = options.ledger.recordCompatibility({
    baselineVersionDigest: options.baselineVersionDigest,
    candidateVersionDigest: options.candidateVersionDigest,
    checks,
    metadata: options.metadata ?? { adapter: 'remembero.review/v1' },
  });
  return {
    baselineVersionDigest: options.baselineVersionDigest,
    candidateVersionDigest: options.candidateVersionDigest,
    diff,
    evidence,
    assessment,
    ...(knowledgeDiff === undefined ? {} : { knowledgeDiff }),
  };
}

export function promoteRememberoReview(
  options: PromoteReviewOptions
): SemanticPromotionDecision {
  return options.ledger.promote({
    ref: options.ref,
    candidateVersionDigest: options.candidateVersionDigest,
    assessmentDigest: options.assessmentDigest,
    operationId: options.operationId,
    expectedCurrentVersionDigest: options.expectedCurrentVersionDigest,
    acceptedReviewDimensions: options.acceptedReviewDimensions,
    reason: options.reason,
  });
}
