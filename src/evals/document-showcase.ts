import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import type { MemoryStore } from '../store/store.js';
import { MemoryStore as Store } from '../store/store.js';
import {
  DOCUMENT_SHOWCASE_FIXTURES,
  type DocumentShowcaseFixtureQuestion,
} from '../document/showcase-fixture.js';
import {
  askDocumentQuestion,
  documentNamespaceFor,
  documentQuestionIds,
  documentFixtureIds,
  materializeDocumentShowcase,
  seedDocumentShowcase,
  type DocumentEvidenceItem,
  type DocumentQuestionStatus,
} from '../document/showcase.js';

export interface DocumentEvaluationMetric {
  passed: number;
  total: number;
  percent: number;
}

export interface DocumentEvaluationLatency {
  parseMs: number;
  averageQuestionMs: number;
  maxQuestionMs: number;
  totalMs: number;
}

export interface DocumentEvaluationCheck {
  questionId: string;
  label: string;
  question: string;
  expectedStatus: DocumentQuestionStatus;
  actualStatus: DocumentQuestionStatus;
  statusPass: boolean;
  answerPass: boolean;
  sourceRecallPass: boolean;
  proofGroundingPass: boolean;
  abstentionPass: boolean;
  latencyMs: number;
}

export interface DocumentEvaluationSummary {
  documentId: string;
  title: string;
  namespace: string;
  status: 'pass' | 'fail';
  checks: DocumentEvaluationCheck[];
  metrics: {
    parseCoverage: DocumentEvaluationMetric;
    answerAccuracy: DocumentEvaluationMetric;
    statusAccuracy: DocumentEvaluationMetric;
    sourceRecall: DocumentEvaluationMetric;
    proofGrounding: DocumentEvaluationMetric;
    abstentionCorrectness: DocumentEvaluationMetric;
    idempotency: DocumentEvaluationMetric;
  };
  latencyMs: DocumentEvaluationLatency;
}

export interface DocumentEvaluationAggregate {
  documentCount: number;
  questionCount: number;
  status: 'pass' | 'fail';
  metrics: DocumentEvaluationSummary['metrics'];
  latencyMs: {
    totalParseMs: number;
    averageQuestionMs: number;
    maxQuestionMs: number;
    totalMs: number;
  };
}

export interface DocumentEvaluationReport {
  generatedAt: string;
  documents: DocumentEvaluationSummary[];
  aggregate: DocumentEvaluationAggregate;
}

function metric(passed: number, total: number): DocumentEvaluationMetric {
  return {
    passed,
    total,
    percent: total === 0 ? 100 : Math.round((passed / total) * 100),
  };
}

function tempStore(label: string): { root: string; store: MemoryStore } {
  const root = mkdtempSync(join(tmpdir(), `rembero-document-eval-${label}-`));
  return { root, store: new Store(root) };
}

function fixtureForDocument(documentId: string) {
  const fixture = DOCUMENT_SHOWCASE_FIXTURES.find((entry) => entry.id === documentId);
  if (fixture === undefined) {
    throw new Error(`unknown document fixture ${documentId}`);
  }
  return fixture;
}

function sortedRegionIds(items: DocumentEvidenceItem[]): string[] {
  return Array.from(
    new Set(items.flatMap((item) => (item.regionId === undefined ? [] : [item.regionId])))
  ).sort();
}

function arraysEqual(left: string[], right: string[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

function unsupportedEvidencePass(
  fixture: (typeof DOCUMENT_SHOWCASE_FIXTURES)[number],
  fixtureQuestion: DocumentShowcaseFixtureQuestion,
  actual: ReturnType<typeof askDocumentQuestion>
): boolean {
  const expectedRegionIds = fixtureQuestion.relatedEvidenceIds
    .filter((id) => fixture.pages.some((page) => page.regions.some((region) => region.id === id)))
    .sort();
  const expectedClaimClauses = fixtureQuestion.relatedEvidenceIds
    .map((id) => fixture.claims.find((claim) => claim.id === id)?.clause)
    .filter((value): value is string => value !== undefined)
    .sort();
  const actualRegionIds = actual.relatedEvidence
    .flatMap((item) => (item.kind === 'raw_region' && item.regionId !== undefined ? [item.regionId] : []))
    .sort();
  const actualClaimClauses = actual.relatedEvidence
    .flatMap((item) => (item.kind !== 'raw_region' && item.clause !== undefined ? [item.clause] : []))
    .sort();
  return (
    actual.steps.length === 0 &&
    actual.sources.length === 0 &&
    arraysEqual(actualRegionIds, expectedRegionIds) &&
    arraysEqual(actualClaimClauses, expectedClaimClauses)
  );
}

function proofGroundingPass(
  documentId: string,
  actual: ReturnType<typeof askDocumentQuestion>
): boolean {
  const namespace = documentNamespaceFor(documentId);
  if (actual.status === 'unsupported') {
    return actual.steps.length === 0 && actual.sources.length === 0;
  }
  const conclusion = actual.steps.at(-1);
  if (conclusion?.kind !== 'conclusion') return false;
  if (actual.sources.length === 0) return false;
  for (const source of actual.sources) {
    if (source.kind !== 'accepted_fact' && source.kind !== 'rule') return false;
    if (source.namespace !== namespace) return false;
  }
  for (const step of actual.steps.slice(0, -1)) {
    if (step.kind !== 'accepted_fact' && step.kind !== 'rule') return false;
  }
  return true;
}

function evaluateOneDocument(documentId: string): DocumentEvaluationSummary {
  const fixture = fixtureForDocument(documentId);
  const temporary = tempStore(documentId);
  try {
    const { store } = temporary;
    const materialized = materializeDocumentShowcase(documentId);
    const parseStart = performance.now();
    const firstSeed = seedDocumentShowcase(store, documentId);
    const parseMs = performance.now() - parseStart;
    const secondSeed = seedDocumentShowcase(store, documentId);
    const expectedOperations =
      materialized.parse.acceptedClaimCount + materialized.document.rules.length;

    const checks = fixture.questions.map((question) => {
    const questionStart = performance.now();
    const actual = askDocumentQuestion(store, question.id, documentId);
    const latencyMs = performance.now() - questionStart;
    const expectedStatus = question.expectedStatus;
    const statusPass = actual.status === expectedStatus;
    const answerPass =
      expectedStatus === 'answered'
        ? actual.answer === question.supportedAnswer
        : actual.answer === question.unsupportedAnswer;
    const sourceRecallPass =
      expectedStatus === 'answered'
        ? arraysEqual(
            sortedRegionIds(actual.sources),
            [...(question.expectedSourceRegionIds ?? [])].sort()
          )
        : actual.sources.length === 0;
    const abstentionPass =
      expectedStatus === 'unsupported'
        ? unsupportedEvidencePass(fixture, question, actual)
        : true;
    return {
      questionId: question.id,
      label: question.label,
      question: question.question,
      expectedStatus,
      actualStatus: actual.status,
      statusPass,
      answerPass,
      sourceRecallPass,
      proofGroundingPass: proofGroundingPass(documentId, actual),
      abstentionPass,
      latencyMs: Number(latencyMs.toFixed(3)),
    };
    });

    const questionCount = checks.length;
    const unsupportedChecks = checks.filter((check) => check.expectedStatus === 'unsupported');
    const latencyTotal = checks.reduce((sum, check) => sum + check.latencyMs, 0);
    const maxQuestionMs = checks.reduce((max, check) => Math.max(max, check.latencyMs), 0);
    const parseCoveragePass =
      materialized.parse.pageCoveragePercent === 100 &&
      materialized.parse.acceptedClaimCoveragePercent === 100 &&
      firstSeed.parse.pageCount === materialized.parse.pageCount &&
      firstSeed.parse.regionCount === materialized.parse.regionCount &&
      firstSeed.parse.acceptedClaimCount === materialized.parse.acceptedClaimCount &&
      firstSeed.parse.proposedClaimCount === materialized.parse.proposedClaimCount;
    const idempotencyPass =
      firstSeed.added === expectedOperations &&
      firstSeed.duplicates === 0 &&
      secondSeed.added === 0 &&
      secondSeed.duplicates === expectedOperations &&
      documentQuestionIds(documentId).length === fixture.questions.length;

    const metrics = {
    parseCoverage: metric(parseCoveragePass ? 1 : 0, 1),
    answerAccuracy: metric(checks.filter((check) => check.answerPass).length, questionCount),
    statusAccuracy: metric(checks.filter((check) => check.statusPass).length, questionCount),
    sourceRecall: metric(checks.filter((check) => check.sourceRecallPass).length, questionCount),
    proofGrounding: metric(
      checks.filter((check) => check.proofGroundingPass).length,
      questionCount
    ),
    abstentionCorrectness: metric(
      unsupportedChecks.filter((check) => check.abstentionPass).length,
      unsupportedChecks.length
    ),
    idempotency: metric(idempotencyPass ? 1 : 0, 1),
    };

    const status =
      Object.values(metrics).every((entry) => entry.percent === 100) ? 'pass' : 'fail';

    return {
      documentId,
      title: fixture.title,
      namespace: documentNamespaceFor(documentId),
      status,
      checks,
      metrics,
      latencyMs: {
        parseMs: Number(parseMs.toFixed(3)),
        averageQuestionMs: Number((latencyTotal / Math.max(questionCount, 1)).toFixed(3)),
        maxQuestionMs: Number(maxQuestionMs.toFixed(3)),
        totalMs: Number((parseMs + latencyTotal).toFixed(3)),
      },
    };
  } finally {
    rmSync(temporary.root, { recursive: true, force: true });
  }
}

export function evaluateDocumentShowcases(): DocumentEvaluationReport {
  const documents = documentFixtureIds().map(evaluateOneDocument);
  const questionCount = documents.reduce((sum, document) => sum + document.checks.length, 0);
  const totalQuestionLatencyMs = documents.reduce(
    (sum, document) =>
      sum + document.checks.reduce((checkSum, check) => checkSum + check.latencyMs, 0),
    0
  );
  const unsupportedQuestionCount = documents.reduce(
    (sum, document) =>
      sum + document.checks.filter((check) => check.expectedStatus === 'unsupported').length,
    0
  );
  const aggregate = {
    documentCount: documents.length,
    questionCount,
    status: documents.every((document) => document.status === 'pass') ? 'pass' : 'fail',
    metrics: {
      parseCoverage: metric(
        documents.filter((document) => document.metrics.parseCoverage.passed === 1).length,
        documents.length
      ),
      answerAccuracy: metric(
        documents.reduce((sum, document) => sum + document.metrics.answerAccuracy.passed, 0),
        questionCount
      ),
      statusAccuracy: metric(
        documents.reduce((sum, document) => sum + document.metrics.statusAccuracy.passed, 0),
        questionCount
      ),
      sourceRecall: metric(
        documents.reduce((sum, document) => sum + document.metrics.sourceRecall.passed, 0),
        questionCount
      ),
      proofGrounding: metric(
        documents.reduce((sum, document) => sum + document.metrics.proofGrounding.passed, 0),
        questionCount
      ),
      abstentionCorrectness: metric(
        documents.reduce((sum, document) => sum + document.metrics.abstentionCorrectness.passed, 0),
        unsupportedQuestionCount
      ),
      idempotency: metric(
        documents.filter((document) => document.metrics.idempotency.passed === 1).length,
        documents.length
      ),
    },
    latencyMs: {
      totalParseMs: Number(
        documents.reduce((sum, document) => sum + document.latencyMs.parseMs, 0).toFixed(3)
      ),
      averageQuestionMs: Number(
        (totalQuestionLatencyMs / Math.max(questionCount, 1)).toFixed(3)
      ),
      maxQuestionMs: Number(
        documents.reduce((max, document) => Math.max(max, document.latencyMs.maxQuestionMs), 0).toFixed(3)
      ),
      totalMs: Number(
        documents.reduce((sum, document) => sum + document.latencyMs.totalMs, 0).toFixed(3)
      ),
    },
  } satisfies DocumentEvaluationAggregate;

  return {
    generatedAt: new Date().toISOString(),
    documents,
    aggregate,
  };
}

function formatMetric(label: string, value: DocumentEvaluationMetric): string {
  return `${label}: ${value.percent}% (${value.passed}/${value.total})`;
}

export function formatDocumentEvaluationReport(report: DocumentEvaluationReport): string {
  const lines: string[] = [
    `Document showcase evaluation · ${report.generatedAt}`,
    '',
  ];

  for (const document of report.documents) {
    lines.push(`${document.title} [${document.status}]`);
    lines.push(
      [
        formatMetric('parse', document.metrics.parseCoverage),
        formatMetric('status', document.metrics.statusAccuracy),
        formatMetric('answer', document.metrics.answerAccuracy),
        formatMetric('recall', document.metrics.sourceRecall),
        formatMetric('proof', document.metrics.proofGrounding),
        formatMetric('abstention', document.metrics.abstentionCorrectness),
        formatMetric('idempotency', document.metrics.idempotency),
      ].join(' · ')
    );
    lines.push(
      `latency: parse ${document.latencyMs.parseMs}ms · avg question ${document.latencyMs.averageQuestionMs}ms · max question ${document.latencyMs.maxQuestionMs}ms`
    );
    for (const check of document.checks) {
      lines.push(
        `  - ${check.label}: ${check.actualStatus} · answer=${check.answerPass ? 'pass' : 'fail'} · recall=${check.sourceRecallPass ? 'pass' : 'fail'} · proof=${check.proofGroundingPass ? 'pass' : 'fail'} · abstention=${check.abstentionPass ? 'pass' : 'fail'} · ${check.latencyMs}ms`
      );
    }
    lines.push('');
  }

  lines.push('Aggregate');
  lines.push(
    [
      formatMetric('parse', report.aggregate.metrics.parseCoverage),
      formatMetric('status', report.aggregate.metrics.statusAccuracy),
      formatMetric('answer', report.aggregate.metrics.answerAccuracy),
      formatMetric('recall', report.aggregate.metrics.sourceRecall),
      formatMetric('proof', report.aggregate.metrics.proofGrounding),
      formatMetric('abstention', report.aggregate.metrics.abstentionCorrectness),
      formatMetric('idempotency', report.aggregate.metrics.idempotency),
    ].join(' · ')
  );
  lines.push(
    `latency: total parse ${report.aggregate.latencyMs.totalParseMs}ms · avg question ${report.aggregate.latencyMs.averageQuestionMs}ms · max question ${report.aggregate.latencyMs.maxQuestionMs}ms · total ${report.aggregate.latencyMs.totalMs}ms`
  );

  return lines.join('\n');
}
