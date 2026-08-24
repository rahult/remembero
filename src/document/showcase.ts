import { createHash } from 'node:crypto';
import type { ExplainKnowledgeResult, SourcedQueryProof } from '../knowledge/graph.js';
import { explainQueryTool } from '../mcp/tools.js';
import type { MemoryStore } from '../store/store.js';
import {
  DOCUMENT_SHOWCASE_DEFAULT_FIXTURE_ID,
  DOCUMENT_SHOWCASE_FIXTURE,
  DOCUMENT_SHOWCASE_FIXTURES,
  type DocumentShowcaseFixture,
  type DocumentShowcaseFixtureClaim,
  type DocumentShowcaseFixtureQuestion,
  type DocumentShowcaseFixtureRegion,
  type DocumentShowcaseFixtureRule,
  type DocumentShowcaseFixtureSource,
} from './showcase-fixture.js';

export const DOCUMENT_SHOWCASE_ROOT_NAMESPACE = 'documents';
export const DOCUMENT_SHOWCASE_DEFAULT_ID = DOCUMENT_SHOWCASE_DEFAULT_FIXTURE_ID;
export const DOCUMENT_SHOWCASE_NAMESPACE = documentNamespaceFor(DOCUMENT_SHOWCASE_DEFAULT_ID);
const DOCUMENT_SHOWCASE_ASSERTED_AT = new Date('2026-08-20T00:00:00.000Z');

export type DocumentQuestionStatus = 'answered' | 'unsupported';
export type DocumentEvidenceKind =
  | 'raw_region'
  | 'proposed_claim'
  | 'accepted_fact'
  | 'rule'
  | 'conclusion';

export interface DocumentBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DocumentRegion {
  id: string;
  pageId: string;
  pageNumber: number;
  order: number;
  label: string;
  kind: 'title' | 'heading' | 'paragraph' | 'status' | 'table_row';
  text: string;
  bbox: DocumentBoundingBox;
}

export interface DocumentPage {
  id: string;
  pageNumber: number;
  width: number;
  height: number;
  label: string;
  imageUrl: string;
  imageSha256: string;
  regions: DocumentRegion[];
}

export interface DocumentEvidenceAnchor {
  pageId: string;
  pageNumber: number;
  regionId: string;
  anchorLabel: string;
  sourceText: string;
}

export interface DocumentClaim {
  id: string;
  kind: 'accepted' | 'proposed';
  clause: string;
  summary: string;
  reviewLabel: string;
  operationId?: string;
  evidence: DocumentEvidenceAnchor;
}

export interface DocumentRule {
  id: string;
  clause: string;
  summary: string;
  reviewLabel: 'Reviewed rule';
  operationId: string;
  sourceText: string;
}

export interface DocumentQuestion {
  id: string;
  label: string;
  question: string;
}

export interface DocumentShowcase {
  id: string;
  namespace: string;
  kindLabel: string;
  fixtureVersion: string;
  fixtureDigest: string;
  fileName: string;
  title: string;
  parserMode: 'source_text_reviewed';
  parserLabel: string;
  adapterLabel: string;
  source: DocumentShowcaseFixtureSource;
  pages: DocumentPage[];
  claims: DocumentClaim[];
  rules: DocumentRule[];
}

export interface DocumentCatalogEntry {
  id: string;
  namespace: string;
  kindLabel: string;
  fileName: string;
  title: string;
  parserMode: 'source_text_reviewed';
  parserLabel: string;
  publisher: string;
  sourceUrl: string;
  fixtureVersion: string;
  pageCount: number;
  regionCount: number;
  acceptedClaimCount: number;
  proposedClaimCount: number;
  questionCount: number;
  supportedQuestionCount: number;
  unsupportedQuestionCount: number;
  defaultQuestionId: string;
}

export interface DocumentParseSummary {
  status: 'ready';
  fixtureVersion: string;
  fixtureDigest: string;
  pageCount: number;
  regionCount: number;
  acceptedClaimCount: number;
  proposedClaimCount: number;
  acceptedClaimCoveragePercent: number;
  pageCoveragePercent: number;
  seededCount: number;
  duplicateCount: number;
}

export interface DocumentEvidenceItem {
  id: string;
  kind: DocumentEvidenceKind;
  label: string;
  detail: string;
  clause?: string;
  operationId?: string;
  namespace?: string;
  pageId?: string;
  pageNumber?: number;
  regionId?: string;
  anchorLabel?: string;
  badge: string;
}

export interface DocumentProofStep {
  id: string;
  kind: 'accepted_fact' | 'rule' | 'conclusion';
  label: string;
  detail: string;
  clause?: string;
  sourceIds: string[];
  badge: string;
}

export interface DocumentQuestionResult {
  questionId: string;
  status: DocumentQuestionStatus;
  question: string;
  query: string;
  answer: string;
  bindings: Array<Record<string, string>>;
  steps: DocumentProofStep[];
  sources: DocumentEvidenceItem[];
  relatedEvidence: DocumentEvidenceItem[];
}

export interface DocumentSnapshotResponse {
  documents: DocumentCatalogEntry[];
  document: DocumentShowcase;
  parse: DocumentParseSummary;
  questions: DocumentQuestion[];
  defaultQuestionId: string;
  proof: DocumentQuestionResult;
}

export interface DocumentSeedResult {
  seeded: boolean;
  added: number;
  duplicates: number;
  document: DocumentShowcase;
  parse: DocumentParseSummary;
}

interface DocumentQuestionDefinition extends DocumentQuestion {
  query: string;
  supportedAnswer?: string;
  unsupportedAnswer: string;
  relatedEvidenceIds: string[];
}

interface MaterializedFixture {
  fixture: DocumentShowcaseFixture;
  document: DocumentShowcase;
  catalog: DocumentCatalogEntry;
  questions: DocumentQuestionDefinition[];
  defaultQuestionId: string;
  parseBase: Omit<DocumentParseSummary, 'seededCount' | 'duplicateCount'>;
}

interface MaterializedIndexes {
  regionsById: Map<string, DocumentRegion>;
  claimsById: Map<string, DocumentClaim>;
  claimsByOperationId: Map<string, DocumentClaim>;
}

function fixtureDigest(fixture: DocumentShowcaseFixture): string {
  return createHash('sha256').update(JSON.stringify(fixture)).digest('hex');
}

function acceptedClaimOperationId(fixtureVersion: string, claimId: string): string {
  return `document-demo-${fixtureVersion}-${claimId}`;
}

function reviewedRuleOperationId(fixtureVersion: string, ruleId: string): string {
  return `document-demo-${fixtureVersion}-${ruleId}`;
}

function buildClaim(
  fixtureVersion: string,
  claim: DocumentShowcaseFixtureClaim,
  region: DocumentShowcaseFixtureRegion
): DocumentClaim {
  if (claim.pageId !== region.pageId || claim.pageNumber !== region.pageNumber) {
    throw new Error(`claim ${claim.id} page anchor does not match region ${region.id}`);
  }
  if (claim.sourceText.trim() === '' || claim.sourceText.trim() !== region.text.trim()) {
    throw new Error(`claim ${claim.id} source text does not match region ${region.id}`);
  }
  return {
    id: claim.id,
    kind: claim.kind,
    clause: claim.clause,
    summary: claim.summary,
    reviewLabel: claim.kind === 'accepted' ? 'Accepted' : 'Proposed only',
    ...(claim.kind === 'accepted'
      ? { operationId: acceptedClaimOperationId(fixtureVersion, claim.id) }
      : {}),
    evidence: {
      pageId: region.pageId,
      pageNumber: region.pageNumber,
      regionId: region.id,
      anchorLabel: region.label,
      sourceText: claim.sourceText,
    },
  };
}

function buildRule(
  fixtureVersion: string,
  rule: DocumentShowcaseFixtureRule
): DocumentRule {
  return {
    id: rule.id,
    clause: rule.clause,
    summary: rule.summary,
    reviewLabel: 'Reviewed rule',
    operationId: reviewedRuleOperationId(fixtureVersion, rule.id),
    sourceText: rule.sourceText,
  };
}

function questionDefinition(
  question: DocumentShowcaseFixtureQuestion
): DocumentQuestionDefinition {
  return {
    id: question.id,
    label: question.label,
    question: question.question,
    query: question.query,
    supportedAnswer: question.supportedAnswer,
    unsupportedAnswer: question.unsupportedAnswer,
    relatedEvidenceIds: [...question.relatedEvidenceIds],
  };
}

function validateBoundingBox(page: DocumentPage, region: DocumentRegion): void {
  const { x, y, width, height } = region.bbox;
  if (
    x < 0 ||
    y < 0 ||
    width <= 0 ||
    height <= 0 ||
    x + width > page.width ||
    y + height > page.height
  ) {
    throw new Error(`region ${region.id} has out-of-bounds geometry`);
  }
}

function validateMaterializedFixture(materialized: MaterializedFixture): MaterializedFixture {
  const { fixture, document, questions, defaultQuestionId } = materialized;
  const pageIds = new Set<string>();
  const pageNumbers = new Set<number>();
  const regionIds = new Set<string>();
  const claimIds = new Set<string>();
  const questionIds = new Set<string>();
  const ruleIds = new Set(document.rules.map((rule) => rule.id));

  for (const page of document.pages) {
    if (pageIds.has(page.id)) throw new Error(`duplicate page id ${page.id}`);
    if (pageNumbers.has(page.pageNumber)) throw new Error(`duplicate page number ${page.pageNumber}`);
    if (!Number.isSafeInteger(page.pageNumber) || page.pageNumber < 1) {
      throw new Error(`page ${page.id} must use a positive PDF page number`);
    }
    if (page.width <= 0 || page.height <= 0) {
      throw new Error(`page ${page.id} must have positive dimensions`);
    }
    pageIds.add(page.id);
    pageNumbers.add(page.pageNumber);

    const orders = new Set<number>();
    for (const region of page.regions) {
      if (regionIds.has(region.id)) throw new Error(`duplicate region id ${region.id}`);
      if (orders.has(region.order)) {
        throw new Error(`duplicate region order ${region.order} on ${page.id}`);
      }
      if (region.pageId !== page.id) throw new Error(`region ${region.id} has mismatched pageId`);
      if (region.pageNumber !== page.pageNumber) {
        throw new Error(`region ${region.id} has mismatched pageNumber`);
      }
      validateBoundingBox(page, region);
      regionIds.add(region.id);
      orders.add(region.order);
    }
  }

  for (const claim of document.claims) {
    if (claimIds.has(claim.id)) throw new Error(`duplicate claim id ${claim.id}`);
    claimIds.add(claim.id);
    if (!pageIds.has(claim.evidence.pageId)) {
      throw new Error(`claim ${claim.id} references unknown page ${claim.evidence.pageId}`);
    }
    if (!regionIds.has(claim.evidence.regionId)) {
      throw new Error(`claim ${claim.id} references unknown region ${claim.evidence.regionId}`);
    }
    if (
      claim.kind === 'accepted' &&
      (claim.operationId === undefined || claim.operationId.length === 0)
    ) {
      throw new Error(`accepted claim ${claim.id} requires a stable operationId`);
    }
  }

  for (const rule of document.rules) {
    if (rule.operationId.length === 0) {
      throw new Error(`rule ${rule.id} requires a stable operationId`);
    }
  }

  for (const question of questions) {
    if (questionIds.has(question.id)) throw new Error(`duplicate question id ${question.id}`);
    questionIds.add(question.id);
  }

  if (!questionIds.has(defaultQuestionId)) {
    throw new Error(`default question ${defaultQuestionId} is not defined`);
  }

  const acceptedClaims = document.claims.filter((claim) => claim.kind === 'accepted');
  if (acceptedClaims.length === 0) {
    throw new Error('fixture requires at least one accepted claim');
  }

  for (const question of fixture.questions) {
    for (const relatedId of question.relatedEvidenceIds) {
      if (!claimIds.has(relatedId) && !regionIds.has(relatedId)) {
        throw new Error(`question ${question.id} references unknown related evidence ${relatedId}`);
      }
    }
    for (const claimId of question.expectedAcceptedClaimIds ?? []) {
      if (!claimIds.has(claimId)) {
        throw new Error(`question ${question.id} references unknown accepted claim ${claimId}`);
      }
    }
    for (const ruleId of question.expectedRuleIds ?? []) {
      if (!ruleIds.has(ruleId)) {
        throw new Error(`question ${question.id} references unknown rule ${ruleId}`);
      }
    }
    for (const regionId of question.expectedSourceRegionIds ?? []) {
      if (!regionIds.has(regionId)) {
        throw new Error(`question ${question.id} references unknown source region ${regionId}`);
      }
    }
    if (question.expectedStatus === 'answered' && question.supportedAnswer === undefined) {
      throw new Error(`answered question ${question.id} requires a supportedAnswer`);
    }
  }

  return materialized;
}

export function documentNamespaceFor(documentId: string): string {
  return `${DOCUMENT_SHOWCASE_ROOT_NAMESPACE}-${documentId}`;
}

function materializeFixture(fixture: DocumentShowcaseFixture): MaterializedFixture {
  const digest = fixtureDigest(fixture);
  const regionById = new Map(
    fixture.pages.flatMap((page) => page.regions.map((region) => [region.id, region] as const))
  );
  const pages: DocumentPage[] = fixture.pages.map((page) => ({
    id: page.id,
    pageNumber: page.pageNumber,
    width: page.width,
    height: page.height,
    label: page.label,
    imageUrl: page.imageUrl,
    imageSha256: page.imageSha256,
    regions: page.regions.map((region) => ({
      id: region.id,
      pageId: region.pageId,
      pageNumber: region.pageNumber,
      order: region.order,
      label: region.label,
      kind: region.kind,
      text: region.text,
      bbox: { ...region.bbox },
    })),
  }));
  const claims = fixture.claims.map((claim) => {
    const region = regionById.get(claim.regionId);
    if (region === undefined) {
      throw new Error(`claim ${claim.id} references unknown region ${claim.regionId}`);
    }
    return buildClaim(fixture.fixtureVersion, claim, region);
  });
  const rules = fixture.rules.map((rule) => buildRule(fixture.fixtureVersion, rule));
  const document: DocumentShowcase = {
    id: fixture.id,
    namespace: documentNamespaceFor(fixture.id),
    kindLabel: fixture.kindLabel,
    fixtureVersion: fixture.fixtureVersion,
    fixtureDigest: digest,
    fileName: fixture.fileName,
    title: fixture.title,
    parserMode: fixture.parserMode,
    parserLabel: fixture.parserLabel,
    adapterLabel: fixture.adapterLabel,
    source: { ...fixture.source, selectedPageNumbers: [...fixture.source.selectedPageNumbers] },
    pages,
    claims,
    rules,
  };
  const pageCount = document.pages.length;
  const regionCount = document.pages.reduce((sum, page) => sum + page.regions.length, 0);
  const acceptedClaimCount = document.claims.filter((claim) => claim.kind === 'accepted').length;
  const proposedClaimCount = document.claims.length - acceptedClaimCount;
  const supportedQuestionCount = fixture.questions.filter(
    (question) => question.expectedStatus === 'answered'
  ).length;
  return validateMaterializedFixture({
    fixture,
    document,
    questions: fixture.questions.map(questionDefinition),
    defaultQuestionId: fixture.defaultQuestionId,
    catalog: {
      id: fixture.id,
      namespace: document.namespace,
      kindLabel: fixture.kindLabel,
      fileName: fixture.fileName,
      title: fixture.title,
      parserMode: fixture.parserMode,
      parserLabel: fixture.parserLabel,
      publisher: fixture.source.publisher,
      sourceUrl: fixture.source.url,
      fixtureVersion: fixture.fixtureVersion,
      pageCount,
      regionCount,
      acceptedClaimCount,
      proposedClaimCount,
      questionCount: fixture.questions.length,
      supportedQuestionCount,
      unsupportedQuestionCount: fixture.questions.length - supportedQuestionCount,
      defaultQuestionId: fixture.defaultQuestionId,
    },
    parseBase: {
      status: 'ready',
      fixtureVersion: document.fixtureVersion,
      fixtureDigest: document.fixtureDigest,
      pageCount,
      regionCount,
      acceptedClaimCount,
      proposedClaimCount,
      acceptedClaimCoveragePercent: 100,
      pageCoveragePercent: 100,
    },
  });
}

const MATERIALIZED_FIXTURES = new Map(
  DOCUMENT_SHOWCASE_FIXTURES.map((fixture) => [fixture.id, materializeFixture(fixture)])
);

function materializedFixtureFor(documentId = DOCUMENT_SHOWCASE_DEFAULT_ID): MaterializedFixture {
  const fixture = MATERIALIZED_FIXTURES.get(documentId);
  if (fixture === undefined) {
    throw new Error(`unknown document fixture ${documentId}`);
  }
  return fixture;
}

function fixtureIndexes(document: DocumentShowcase): MaterializedIndexes {
  return {
    regionsById: new Map(
      document.pages.flatMap((page) => page.regions.map((region) => [region.id, region] as const))
    ),
    claimsById: new Map(document.claims.map((claim) => [claim.id, claim])),
    claimsByOperationId: new Map(
      document.claims
        .filter(
          (claim): claim is DocumentClaim & { operationId: string } =>
            claim.kind === 'accepted' && typeof claim.operationId === 'string'
        )
        .map((claim) => [claim.operationId, claim])
    ),
  };
}

function questionById(questions: DocumentQuestionDefinition[], questionId: string) {
  const question = questions.find((entry) => entry.id === questionId);
  if (question === undefined) {
    throw new Error(`unknown document question ${questionId}`);
  }
  return question;
}

function collectProof(
  proof: SourcedQueryProof,
  leafOperationIds: string[],
  seenLeafOperations: Set<string>,
  ruleNumbers: number[],
  seenRuleNumbers: Set<number>
): void {
  if ('aggregated' in proof) {
    for (const contributor of proof.contributors) {
      for (const child of contributor.proofs) {
        collectProof(child, leafOperationIds, seenLeafOperations, ruleNumbers, seenRuleNumbers);
      }
    }
    return;
  }
  if ('negated' in proof) return;
  if (proof.rule === undefined) {
    for (const source of [...(proof.sources ?? []), ...(proof.sourceAlternatives ?? [])]) {
      if (!seenLeafOperations.has(source.opId)) {
        seenLeafOperations.add(source.opId);
        leafOperationIds.push(source.opId);
      }
    }
  } else if (!seenRuleNumbers.has(proof.rule)) {
    seenRuleNumbers.add(proof.rule);
    ruleNumbers.push(proof.rule);
  }
  for (const child of proof.because ?? []) {
    collectProof(child, leafOperationIds, seenLeafOperations, ruleNumbers, seenRuleNumbers);
  }
  for (const contributor of proof.aggregate?.contributors ?? []) {
    for (const child of contributor.proofs) {
      collectProof(child, leafOperationIds, seenLeafOperations, ruleNumbers, seenRuleNumbers);
    }
  }
}

function acceptedFactSource(
  claim: DocumentClaim,
  namespace: string
): DocumentEvidenceItem {
  return {
    id: `source-${claim.id}`,
    kind: 'accepted_fact',
    label: claim.clause.replace(/\.$/, ''),
    detail: claim.evidence.sourceText,
    clause: claim.clause,
    operationId: claim.operationId,
    namespace,
    pageId: claim.evidence.pageId,
    pageNumber: claim.evidence.pageNumber,
    regionId: claim.evidence.regionId,
    anchorLabel: claim.evidence.anchorLabel,
    badge: claim.reviewLabel,
  };
}

function reviewedRuleSource(
  rule: DocumentRule,
  namespace: string
): DocumentEvidenceItem {
  return {
    id: `source-${rule.id}`,
    kind: 'rule',
    label: rule.clause,
    detail: rule.summary,
    clause: rule.clause,
    operationId: rule.operationId,
    namespace,
    badge: rule.reviewLabel,
  };
}

function buildAnsweredProof(
  explanation: ExplainKnowledgeResult,
  document: DocumentShowcase,
  question: DocumentQuestionDefinition,
  answer: string
): Pick<DocumentQuestionResult, 'steps' | 'sources'> {
  const indexes = fixtureIndexes(document);
  const leafOperationIds: string[] = [];
  const seenLeafOperations = new Set<string>();
  const ruleNumbers: number[] = [];
  const seenRuleNumbers = new Set<number>();
  for (const row of explanation.rows) {
    for (const proof of row.proofs) {
      collectProof(proof, leafOperationIds, seenLeafOperations, ruleNumbers, seenRuleNumbers);
    }
  }

  const ruleCatalog = new Map(explanation.rules.map((rule) => [rule.number, rule.clause]));
  const sources: DocumentEvidenceItem[] = [];
  const steps: DocumentProofStep[] = [];

  for (const operationId of leafOperationIds) {
    const claim = indexes.claimsByOperationId.get(operationId);
    if (claim === undefined) {
      throw new Error(`document proof source ${operationId} is not grounded in the fixture`);
    }
    const source = acceptedFactSource(claim, document.namespace);
    sources.push(source);
    steps.push({
      id: `step-${claim.id}`,
      kind: 'accepted_fact',
      label: source.label,
      detail: `${claim.evidence.anchorLabel} · ${claim.evidence.sourceText}`,
      clause: claim.clause,
      sourceIds: [source.id],
      badge: claim.reviewLabel,
    });
  }

  for (const ruleNumber of ruleNumbers) {
    const clause = ruleCatalog.get(ruleNumber);
    if (clause === undefined) {
      throw new Error(`document proof rule ${ruleNumber} was not present in the explanation`);
    }
    const rule = document.rules.find((entry) => entry.clause === clause);
    if (rule === undefined) {
      throw new Error(`document proof rule ${ruleNumber} is not grounded in reviewed rules`);
    }
    const source = reviewedRuleSource(rule, document.namespace);
    sources.push(source);
    steps.push({
      id: `step-${rule.id}`,
      kind: 'rule',
      label: rule.clause,
      detail: rule.summary,
      clause: rule.clause,
      sourceIds: [source.id],
      badge: rule.reviewLabel,
    });
  }

  steps.push({
    id: `step-conclusion-${question.id}`,
    kind: 'conclusion',
    label: 'Conclusion',
    detail: answer,
    clause: question.query,
    sourceIds: sources.map((source) => source.id),
    badge: 'Supported',
  });

  return { steps, sources };
}

function buildRelatedEvidence(
  document: DocumentShowcase,
  question: DocumentQuestionDefinition
): DocumentEvidenceItem[] {
  const indexes = fixtureIndexes(document);
  const related: DocumentEvidenceItem[] = [];
  for (const id of question.relatedEvidenceIds) {
    const claim = indexes.claimsById.get(id);
    if (claim !== undefined) {
      related.push({
        id: `related-${claim.id}`,
        kind: claim.kind === 'accepted' ? 'accepted_fact' : 'proposed_claim',
        label: claim.clause.replace(/\.$/, ''),
        detail: claim.summary,
        clause: claim.clause,
        ...(claim.operationId === undefined ? {} : { operationId: claim.operationId }),
        namespace: document.namespace,
        pageId: claim.evidence.pageId,
        pageNumber: claim.evidence.pageNumber,
        regionId: claim.evidence.regionId,
        anchorLabel: claim.evidence.anchorLabel,
        badge: claim.reviewLabel,
      });
      continue;
    }
    const region = indexes.regionsById.get(id);
    if (region !== undefined) {
      related.push({
        id: `related-${region.id}`,
        kind: 'raw_region',
        label: `${region.label} · ${region.kind.replace('_', ' ')}`,
        detail: region.text,
        pageId: region.pageId,
        pageNumber: region.pageNumber,
        regionId: region.id,
        anchorLabel: region.label,
        badge: 'Source region',
      });
    }
  }
  return related;
}

function parseSummary(
  materialized: MaterializedFixture,
  seededCount: number,
  duplicateCount: number
): DocumentParseSummary {
  return {
    ...materialized.parseBase,
    seededCount,
    duplicateCount,
  };
}

export function documentFixtureIds(): string[] {
  return DOCUMENT_SHOWCASE_FIXTURES.map((fixture) => fixture.id);
}

export function materializeDocumentCatalog(): DocumentCatalogEntry[] {
  return documentFixtureIds().map((documentId) => materializedFixtureFor(documentId).catalog);
}

export function materializeDocumentShowcase(documentId = DOCUMENT_SHOWCASE_DEFAULT_ID): {
  document: DocumentShowcase;
  questions: DocumentQuestion[];
  defaultQuestionId: string;
  parse: DocumentParseSummary;
} {
  const materialized = materializedFixtureFor(documentId);
  return {
    document: materialized.document,
    questions: materialized.questions.map(({ id, label, question }) => ({
      id,
      label,
      question,
    })),
    defaultQuestionId: materialized.defaultQuestionId,
    parse: parseSummary(materialized, 0, 0),
  };
}

export function seedDocumentShowcase(
  store: MemoryStore,
  documentId = DOCUMENT_SHOWCASE_DEFAULT_ID
): DocumentSeedResult {
  const materialized = materializedFixtureFor(documentId);
  const namespace = materialized.document.namespace;
  const expectedOperations =
    materialized.document.claims.filter((claim) => claim.kind === 'accepted').length +
    materialized.document.rules.length;
  const before = store.load(namespace).length;
  for (const claim of materialized.document.claims) {
    if (claim.kind !== 'accepted') continue;
    store.assert(namespace, claim.clause, {
      opId: claim.operationId,
      sourceText: claim.evidence.sourceText,
      at: DOCUMENT_SHOWCASE_ASSERTED_AT,
    });
  }
  for (const rule of materialized.document.rules) {
    store.assert(namespace, rule.clause, {
      opId: rule.operationId,
      sourceText: rule.sourceText,
      at: DOCUMENT_SHOWCASE_ASSERTED_AT,
    });
  }
  const added = store.load(namespace).length - before;
  const duplicates = expectedOperations - added;
  return {
    seeded: added > 0,
    added,
    duplicates,
    document: materialized.document,
    parse: parseSummary(materialized, added, duplicates),
  };
}

export function seedAllDocumentShowcases(store: MemoryStore): DocumentSeedResult[] {
  return documentFixtureIds().map((documentId) => seedDocumentShowcase(store, documentId));
}

export function askDocumentQuestion(
  store: MemoryStore,
  questionId: string,
  documentId = DOCUMENT_SHOWCASE_DEFAULT_ID
): DocumentQuestionResult {
  const materialized = materializedFixtureFor(documentId);
  const question = questionById(materialized.questions, questionId);
  const namespace = materialized.document.namespace;
  const explanation = explainQueryTool(
    { store },
    { query: question.query, namespaces: [namespace], proofLimit: 1 }
  );
  const bindings = explanation.rows.map((row) => row.bindings);
  if (bindings.length === 0) {
    return {
      questionId: question.id,
      status: 'unsupported',
      question: question.question,
      query: question.query,
      answer: question.unsupportedAnswer,
      bindings: [],
      steps: [],
      sources: [],
      relatedEvidence: buildRelatedEvidence(materialized.document, question),
    };
  }
  const answer =
    question.supportedAnswer ??
    (bindings.length === 1 ? 'One supported result is available.' : `${bindings.length} supported results.`);
  const proof = buildAnsweredProof(explanation, materialized.document, question, answer);
  return {
    questionId: question.id,
    status: 'answered',
    question: question.question,
    query: question.query,
    answer,
    bindings,
    steps: proof.steps,
    sources: proof.sources,
    relatedEvidence: [],
  };
}

export function documentQuestionIds(documentId = DOCUMENT_SHOWCASE_DEFAULT_ID): string[] {
  return materializedFixtureFor(documentId).questions.map((question) => question.id);
}

export function documentSnapshotResponse(
  store: MemoryStore,
  documentId = DOCUMENT_SHOWCASE_DEFAULT_ID
): DocumentSnapshotResponse {
  const { document, questions, defaultQuestionId, parse } = materializeDocumentShowcase(documentId);
  const expectedOperationIds = new Set([
    ...document.claims.flatMap((claim) => (claim.operationId === undefined ? [] : [claim.operationId])),
    ...document.rules.map((rule) => rule.operationId),
  ]);
  const existingOperationIds = new Set(
    [...store.sourcesFor([document.namespace]).values()]
      .flat()
      .map((source) => source.opId)
      .filter((opId) => expectedOperationIds.has(opId))
  );
  return {
    documents: materializeDocumentCatalog(),
    document,
    parse: {
      ...parse,
      seededCount: 0,
      duplicateCount: existingOperationIds.size,
    },
    questions,
    defaultQuestionId,
    proof: askDocumentQuestion(store, defaultQuestionId, documentId),
  };
}

export {
  DOCUMENT_SHOWCASE_FIXTURE,
  DOCUMENT_SHOWCASE_FIXTURES,
};
