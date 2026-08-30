import { randomUUID } from 'node:crypto';
import {
  isIntegrityConstraint,
  parseProgram,
  serializeTerm,
  type Term,
} from '../engine/index.js';
import {
  askDocumentQuestion,
  DOCUMENT_SHOWCASE_DEFAULT_ID,
  documentFixtureIds,
  documentQuestionIds,
  documentSnapshotResponse,
  seedAllDocumentShowcases,
  seedDocumentShowcase,
} from '../document/showcase.js';
import { LIVE_OCR_EVIDENCE } from '../document/live-ocr-evidence.js';
import { PRODUCT_SHIP_EVIDENCE } from '../document/product-ship-evidence.js';
import {
  createDocumentMemorgExport,
  verifyDocumentMemorgExport,
} from '../document/memorg.js';
import {
  evaluateDocumentShowcases,
  type DocumentEvaluationReport,
} from '../evals/document-showcase.js';
import type { LlmClient } from '../llm/client.js';
import type {
  ExplainKnowledgeResult,
  SourcedQueryProof,
} from '../knowledge/graph.js';
import type { KnowledgeSearchClauseKind } from '../knowledge/search.js';
import { assertBoundedInput } from '../safety.js';
import type { MemorySource, MemoryStore } from '../store/store.js';
import {
  captureRememberoVersion,
  type RememberoVersionCapture,
} from '../ledger/remembero-version.js';
import {
  promoteRememberoReview,
  reviewRememberoCandidate,
} from '../ledger/remembero-review.js';
import type { SemanticLedger, SemanticVersion } from '../ledger/semantic-ledger.js';
import {
  browseKnowledgeGraphTool,
  explainQueryTool,
  exportKnowledgeBundleTool,
  knowledgeHealthTool,
  recallExplainTool,
  searchKnowledgeTool,
  whyNotTool,
} from '../mcp/tools.js';

export const WEB_DEMO_NAMESPACE = 'personal';
let cachedDocumentEvaluations: DocumentEvaluationReport | undefined;

function documentEvaluationSnapshot(): DocumentEvaluationReport {
  cachedDocumentEvaluations ??= evaluateDocumentShowcases();
  return cachedDocumentEvaluations;
}

export interface GuidedQuestion {
  id: string;
  question: string;
  query: string;
}

export const GUIDED_QUESTIONS: readonly GuidedQuestion[] = [
  {
    id: 'collaborators',
    question: 'Who is collaborating on Atlas?',
    query: 'collaborator(Person, atlas)',
  },
  {
    id: 'owner',
    question: 'Who owns Atlas?',
    query: 'project_owner(atlas, Owner)',
  },
  {
    id: 'follow-up',
    question: 'What follow-up do I owe Maya?',
    query: 'needs_follow_up(maya, Project)',
  },
  {
    id: 'risk',
    question: 'What is blocking Atlas?',
    query: 'project_risk(atlas, Risk)',
  },
  {
    id: 'gift',
    question: 'What gift does Maya want?',
    query: 'prefers_gift(maya, Gift)',
  },
] as const;

const DEMO_BATCHES = [
  {
    opId: 'web-demo-directory-v1',
    at: '2026-08-15T09:00:00.000Z',
    sourceText: 'Personal directory: Rahul works with Maya Patel, who leads design at Northstar.',
    clauses: `
      person(rahul).
      person(maya).
      works_at(maya, northstar).
    `,
  },
  {
    opId: 'web-demo-atlas-session-v1',
    at: '2026-08-17T09:00:00.000Z',
    sourceText:
      'Atlas planning session: Rahul owns Atlas, Maya contributes, Atlas relates to Northstar, and vendor security review is blocking progress. Rahul promised Maya an update.',
    clauses: `
      project(atlas).
      project_owner(atlas, rahul).
      project_contributor(atlas, maya).
      initiative_related(atlas, northstar).
      met_at(rahul, maya, systems_roundtable).
      prefers_meeting(maya, morning).
      status(atlas, blocked).
      blocker(atlas, vendor_security_review).
      promised_update(rahul, maya, atlas).
    `,
  },
  {
    opId: 'web-demo-rules-v1',
    at: '2026-08-17T09:05:00.000Z',
    sourceText:
      'Reviewed personal operating rules for collaboration, follow-up, and project risk.',
    clauses: `
      collaborator(Person, Project) :- project_owner(Project, Owner), project_contributor(Project, Person), Owner != Person.
      needs_follow_up(Person, Project) :- promised_update(rahul, Person, Project), status(Project, blocked).
      project_risk(Project, Risk) :- blocker(Project, Risk), status(Project, blocked).
    `,
  },
] as const;

export class WebServiceError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status = 400
  ) {
    super(message);
    this.name = 'WebServiceError';
  }
}

export interface RemberoWebServiceOptions {
  store: MemoryStore;
  ledger?: SemanticLedger;
  llm?: LlmClient;
  llmConfigured?: boolean;
  namespace?: string;
}

interface EvidenceSummary {
  claims: string[];
  rules: Array<{ number: number; clause: string }>;
  sources: MemorySource[];
}

function atom(value: string | number): Term {
  return typeof value === 'number'
    ? { type: 'num', value }
    : { type: 'atom', value };
}

function claimText(predicate: string, values: Array<string | number>): string {
  return `${predicate}(${values.map((value) => serializeTerm(atom(value))).join(', ')})`;
}

function collectProof(
  proof: SourcedQueryProof,
  claims: Set<string>,
  ruleNumbers: Set<number>,
  sources: Map<string, MemorySource>
): void {
  if ('aggregated' in proof) {
    for (const contributor of proof.contributors) {
      for (const child of contributor.proofs) {
        collectProof(child, claims, ruleNumbers, sources);
      }
    }
    return;
  }
  if ('negated' in proof) return;
  if (proof.rule === undefined) {
    claims.add(claimText(proof.predicate, proof.values));
  }
  if (proof.rule !== undefined) ruleNumbers.add(proof.rule);
  for (const source of [...(proof.sources ?? []), ...(proof.sourceAlternatives ?? [])]) {
    sources.set(JSON.stringify(source), source);
  }
  for (const child of proof.because ?? []) {
    collectProof(child, claims, ruleNumbers, sources);
  }
  for (const contributor of proof.aggregate?.contributors ?? []) {
    for (const child of contributor.proofs) {
      collectProof(child, claims, ruleNumbers, sources);
    }
  }
}

function summarizeEvidence(explanation: ExplainKnowledgeResult): EvidenceSummary {
  const claims = new Set<string>();
  const ruleNumbers = new Set<number>();
  const sources = new Map<string, MemorySource>();
  for (const row of explanation.rows) {
    for (const proof of row.proofs) {
      collectProof(proof, claims, ruleNumbers, sources);
    }
    for (const alternative of row.alternativeProofs ?? []) {
      for (const proof of alternative) {
        collectProof(proof, claims, ruleNumbers, sources);
      }
    }
  }
  const ruleCatalog = new Map(
    explanation.rules.map((rule) => [rule.number, rule.clause])
  );
  return {
    claims: [...claims],
    rules: [...ruleNumbers]
      .sort((left, right) => left - right)
      .map((number) => ({ number, clause: ruleCatalog.get(number) ?? 'Unknown rule' })),
    sources: [...sources.values()].sort(
      (left, right) =>
        left.ts.localeCompare(right.ts) || left.opId.localeCompare(right.opId)
    ),
  };
}

function titleValue(value: string): string {
  return value
    .replace(/^'(.*)'$/, '$1')
    .split('_')
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(' ');
}

function friendlyAnswer(id: string, bindings: Record<string, string>[]): string {
  const first = bindings[0] ?? {};
  if (id === 'collaborators' && first.Person !== undefined) {
    return `${titleValue(first.Person)} is collaborating on Atlas.`;
  }
  if (id === 'owner' && first.Owner !== undefined) {
    return `${titleValue(first.Owner)} owns Atlas.`;
  }
  if (id === 'follow-up' && first.Project !== undefined) {
    return `You owe Maya an update on ${titleValue(first.Project)}.`;
  }
  if (id === 'risk' && first.Risk !== undefined) {
    return `${titleValue(first.Risk)} is blocking Atlas.`;
  }
  return bindings.length === 1
    ? Object.entries(first)
        .map(([key, value]) => `${key}: ${titleValue(value)}`)
        .join(' · ')
    : `${bindings.length} supported results.`;
}

function recentSources(
  bundle: ReturnType<typeof exportKnowledgeBundleTool>
): Array<{
  opId: string;
  ts: string;
  text?: string;
  clauses: string[];
}> {
  const byOperation = new Map<
    string,
    { opId: string; ts: string; text?: string; clauses: Set<string> }
  >();
  for (const namespace of bundle.namespaces) {
    for (const entry of namespace.clauses) {
      for (const source of entry.sources) {
        const key = `${source.namespace}\0${source.opId}`;
        const current = byOperation.get(key) ?? {
          opId: source.opId,
          ts: source.ts,
          ...(source.text === undefined ? {} : { text: source.text }),
          clauses: new Set<string>(),
        };
        current.clauses.add(entry.clause);
        byOperation.set(key, current);
      }
    }
  }
  return [...byOperation.values()]
    .sort((left, right) => right.ts.localeCompare(left.ts) || left.opId.localeCompare(right.opId))
    .map(({ clauses, ...source }) => ({ ...source, clauses: [...clauses].sort() }));
}

export class RemberoWebService {
  readonly namespace: string;
  readonly llmConfigured: boolean;

  constructor(private readonly options: RemberoWebServiceOptions) {
    this.namespace = options.namespace ?? WEB_DEMO_NAMESPACE;
    this.llmConfigured = options.llmConfigured ?? options.llm !== undefined;
  }

  seedDemo(): { seeded: boolean; added: number } {
    const before = this.options.store.load(this.namespace).length;
    for (const batch of DEMO_BATCHES) {
      this.options.store.assert(this.namespace, batch.clauses, {
        opId: batch.opId,
        sourceText: batch.sourceText,
        at: new Date(batch.at),
      });
    }
    const added = this.options.store.load(this.namespace).length - before;
    return { seeded: added > 0, added };
  }

  bootstrap() {
    const namespaces = [this.namespace];
    const bundle = exportKnowledgeBundleTool(
      { store: this.options.store },
      { namespaces }
    );
    const health = knowledgeHealthTool(
      { store: this.options.store },
      { namespaces }
    );
    let facts = 0;
    let rules = 0;
    let constraints = 0;
    const entries = bundle.namespaces.flatMap((namespace) => namespace.clauses);
    const ruleItems: Array<{
      id: string;
      clause: string;
      summary: string;
      status: 'stable';
      sourceLabel: string;
    }> = [];
    for (const entry of entries) {
      const clause = parseProgram(entry.clause)[0];
      if (isIntegrityConstraint(clause)) constraints++;
      else if (clause.body.length === 0) facts++;
      else {
        rules++;
        ruleItems.push({
          id: `rule-${rules}`,
          clause: entry.clause,
          summary: `Derives ${clause.head.predicate.replaceAll('_', ' ')}`,
          status: 'stable',
          sourceLabel: entry.sources[0]?.text ?? 'Reviewed personal rule',
        });
      }
    }
    const recent = recentSources(bundle).slice(0, 8);
    const graph = facts === 0
      ? undefined
      : browseKnowledgeGraphTool(
          { store: this.options.store },
          {
            focus: 'atlas',
            depth: 2,
            maxClaims: 50,
            namespaces,
          }
        );
    const highlights = facts === 0
      ? []
      : searchKnowledgeTool(
          { store: this.options.store },
          { text: 'Atlas Maya', namespaces, limit: 4, kinds: ['fact', 'rule'] }
        ).results;
    return {
      namespace: this.namespace,
      llmConfigured: this.llmConfigured,
      empty: health.clauseCount === 0,
      profile: {
        workspaceLabel: 'Remembero',
        personaLabel: 'Personal',
        storageLabel: 'Local-first',
      },
      counts: {
        facts,
        rules,
        constraints,
        sourcedPercent: health.provenance.sourceCoveragePercent,
      },
      memoryPulse: {
        factCount: facts,
        ruleCount: rules,
        sourceCoveragePercent: health.provenance.sourceCoveragePercent,
        healthTone: health.status,
        healthLabel:
          health.status === 'healthy'
            ? 'Healthy'
            : health.status === 'review'
              ? 'Review needed'
              : 'Violations',
        findingCount: health.findings.length,
      },
      health,
      bundle,
      recent,
      recentMemory: recent.map((source, index) => ({
        id: source.opId,
        title:
          source.opId === 'web-demo-atlas-session-v1'
            ? 'Atlas'
            : source.opId === 'web-demo-directory-v1'
              ? 'Maya Patel'
              : source.opId === 'web-demo-rules-v1'
                ? 'Personal rules'
                : 'Personal memory',
        detail: source.text ?? `${source.clauses.length} sourced clauses`,
        dateLabel: new Date(source.ts).toLocaleDateString('en-AU', {
          day: 'numeric',
          month: 'short',
          year: 'numeric',
          timeZone: 'UTC',
        }),
        clause: source.clauses[0],
        sourceLabel: this.namespace,
        rank: index + 1,
      })),
      guidedQuestions: GUIDED_QUESTIONS,
      askPresets: GUIDED_QUESTIONS.map(({ id, question }) => ({
        id,
        label: question,
        question,
      })),
      knowledgeHighlights: highlights,
      ...(graph === undefined ? {} : { graph }),
      rules: ruleItems,
      healthFindings: health.findings.map((finding) => finding.message),
    };
  }

  async ask(input: { question: string; presetId?: string }) {
    assertBoundedInput(input.question, 'web recall question');
    const question = input.question.trim();
    if (question.length === 0) {
      throw new WebServiceError('empty_question', 'Enter a question to ask memory.');
    }
    const preset = input.presetId === undefined
      ? GUIDED_QUESTIONS.find((candidate) => candidate.question === question)
      : GUIDED_QUESTIONS.find((candidate) => candidate.id === input.presetId);
    if (preset !== undefined) return this.askGuided(preset);
    if (!this.llmConfigured || this.options.llm === undefined) {
      throw new WebServiceError(
        'model_not_configured',
        'Custom questions need LLM_API_KEY. Guided questions remain fully local.'
      );
    }
    const result = await recallExplainTool(
      { store: this.options.store, llm: this.options.llm },
      {
        question,
        namespaces: [this.namespace],
        answerMode: 'evidence',
        proofLimit: 2,
        relatedKnowledge: { limit: 4, kinds: ['fact', 'rule'] },
      }
    );
    return {
      ...result,
      mode: 'model-assisted' as const,
      evidence: result.explanation === undefined
        ? { claims: [], rules: [], sources: [] }
        : summarizeEvidence(result.explanation),
    };
  }

  private askGuided(preset: GuidedQuestion) {
    const namespaces = [this.namespace];
    const explanation = explainQueryTool(
      { store: this.options.store },
      { query: preset.query, namespaces, proofLimit: 2 }
    );
    const bindings = explanation.rows.map((row) => row.bindings);
    if (bindings.length > 0) {
      return {
        mode: 'guided-local' as const,
        status: 'answered' as const,
        question: preset.question,
        query: preset.query,
        answer: friendlyAnswer(preset.id, bindings),
        bindings,
        explanation,
        evidence: summarizeEvidence(explanation),
      };
    }
    const whyNot = whyNotTool(
      { store: this.options.store },
      { query: preset.query, namespaces, proofLimit: 2 }
    );
    const relatedKnowledge = searchKnowledgeTool(
      { store: this.options.store },
      {
        text: preset.question,
        namespaces,
        limit: 4,
        kinds: ['fact', 'rule'],
      }
    );
    return {
      mode: 'guided-local' as const,
      status: 'no_match' as const,
      question: preset.question,
      query: preset.query,
      answer: `We don't have a supported answer to “${preset.question}”`,
      bindings: [],
      explanation,
      evidence: { claims: [], rules: [], sources: [] },
      whyNot,
      relatedKnowledge,
    };
  }

  search(input: { text: string; kinds?: KnowledgeSearchClauseKind[] }) {
    assertBoundedInput(input.text, 'web knowledge search');
    return searchKnowledgeTool(
      { store: this.options.store },
      {
        text: input.text,
        namespaces: [this.namespace],
        limit: 20,
        ...(input.kinds === undefined ? {} : { kinds: input.kinds }),
      }
    );
  }

  graph(input: { focus: string }) {
    assertBoundedInput(input.focus, 'web graph focus');
    if (input.focus.trim().length === 0) {
      throw new WebServiceError('empty_graph_focus', 'Choose an entity to explore.');
    }
    return browseKnowledgeGraphTool(
      { store: this.options.store },
      {
        focus: input.focus.trim(),
        depth: 2,
        maxClaims: 50,
        namespaces: [this.namespace],
      }
    );
  }

  addMemory(input: {
    subject: string;
    predicate: string;
    object: string;
    sourceText: string;
  }) {
    for (const [label, value] of Object.entries(input)) {
      assertBoundedInput(value, `web memory ${label}`);
      if (value.trim().length === 0) {
        throw new WebServiceError('empty_memory_field', `${label} is required.`);
      }
    }
    const predicate = input.predicate.trim();
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(predicate) || predicate.startsWith('rembero_')) {
      throw new WebServiceError(
        'invalid_predicate',
        'Relationship must use lowercase letters, numbers, and underscores.'
      );
    }
    const clause = `${predicate}(${serializeTerm(atom(input.subject.trim()))}, ${serializeTerm(
      atom(input.object.trim())
    )}).`;
    const result = this.options.store.assert(this.namespace, clause, {
      opId: `web-memory-${randomUUID()}`,
      sourceText: input.sourceText.trim(),
    });
    return {
      status: 'saved' as const,
      clause,
      added: result.added.length,
      duplicate: result.duplicates > 0,
    };
  }

  private documentSnapshot(documentId = DOCUMENT_SHOWCASE_DEFAULT_ID) {
    const snapshot = documentSnapshotResponse(this.options.store, documentId);
    const evaluations = documentEvaluationSnapshot();
    const evaluationById = new Map(
      evaluations.documents.map((entry) => [entry.documentId, entry])
    );
    const memorg = createDocumentMemorgExport();
    const memorgVerification = verifyDocumentMemorgExport(memorg);
    return {
      ...snapshot,
      documents: snapshot.documents.map((document) => ({
        ...document,
        evaluation: evaluationById.get(document.id),
      })),
      evaluation: evaluationById.get(snapshot.document.id),
      corpusEvaluation: evaluations.aggregate,
      liveOcrEvidence: LIVE_OCR_EVIDENCE,
      shipEvidence: PRODUCT_SHIP_EVIDENCE,
      memorgExport: {
        format: memorg.format,
        version: memorg.version,
        targetVersion: memorg.target.version,
        sha256: memorg.sha256,
        itemCount: memorgVerification.itemCount,
        downloadUrl: '/documents/document-intelligence.memorg.json',
      },
    };
  }

  private requireLedger(): SemanticLedger {
    if (this.options.ledger === undefined) {
      throw new WebServiceError(
        'versioning_unavailable',
        'Semantic version review is unavailable for this workspace.',
        503
      );
    }
    return this.options.ledger;
  }

  private versionStatus(version: SemanticVersion, currentRefs: Map<string, string>): string {
    if ([...currentRefs.values()].includes(version.digest)) return 'promoted';
    const parent = version.parents[0];
    if (parent === undefined) return 'baseline';
    const diff = this.requireLedger().diffVersions(parent, version.digest);
    const checks = diff.compatibility?.checks ?? [];
    if (checks.some((check) => check.status === 'fail' || check.status === 'blocked')) {
      return 'blocked';
    }
    if (checks.some((check) => check.status === 'review')) return 'review';
    return 'candidate';
  }

  ensureSemanticBaseline(): Record<string, unknown> {
    const ledger = this.requireLedger();
    const current = ledger.getRef('main');
    if (current !== undefined) return { created: false, versionDigest: current.versionDigest };
    const capture = captureRememberoVersion({
      ledger,
      store: this.options.store,
      label: 'remembero@0.55.0',
      metadata: { purpose: 'web-baseline' },
    });
    ledger.setRef({
      name: 'main',
      versionDigest: capture.version.digest,
      operationId: 'remembero-web-main-baseline',
      reason: 'Initialize the human review baseline',
    });
    return { created: true, versionDigest: capture.version.digest };
  }

  versionWorkspace() {
    const ledger = this.requireLedger();
    const refs = ledger.listRefs();
    const currentRefs = new Map(refs.map((ref) => [ref.name, ref.versionDigest]));
    const versions = ledger.listVersions(100).map((version) => {
      const parent = version.parents[0];
      const diff = parent === undefined ? undefined : ledger.diffVersions(parent, version.digest);
      return {
        digest: version.digest,
        labels: version.labels,
        parents: version.parents,
        createdAt: version.createdAt,
        status: this.versionStatus(version, currentRefs),
        memberKeys: version.members.map((member) => member.key),
        edgeCount: version.edges.length,
        contractCount: version.contracts.length,
        compatibility: diff?.compatibility,
        changed: diff?.changed ?? false,
      };
    });
    return { refs, versions };
  }

  captureSemanticVersion(input: { label?: string; ref?: string }): Record<string, unknown> {
    const ledger = this.requireLedger();
    const ref = input.ref ?? 'main';
    const baseline = ledger.getRef(ref);
    const capture: RememberoVersionCapture = captureRememberoVersion({
      ledger,
      store: this.options.store,
      ...(baseline === undefined ? {} : { parents: [baseline.versionDigest] }),
      label: input.label ?? `remembero@candidate-${Date.now()}`,
      metadata: { purpose: 'web-candidate', baselineRef: ref },
    });
    return {
      version: capture.version,
      baselineVersionDigest: baseline?.versionDigest,
      documents: capture.documents,
      recordedSnapshot: {
        sequence: capture.recordedSnapshot.sequence,
        journalEntries: capture.recordedSnapshot.journalEntries,
        namespaces: capture.recordedSnapshot.namespaces,
      },
    };
  }

  reviewSemanticVersion(input: { candidateVersionDigest: string; includeDocumentEvaluation?: boolean }) {
    const ledger = this.requireLedger();
    const candidate = ledger.getVersion(input.candidateVersionDigest);
    const baseline = candidate.parents[0];
    return reviewRememberoCandidate({
      ledger,
      store: this.options.store,
      candidateVersionDigest: candidate.digest,
      baselineVersionDigest: baseline,
      includeDocumentEvaluation: input.includeDocumentEvaluation,
    });
  }

  promoteSemanticVersion(input: {
    ref: string;
    candidateVersionDigest: string;
    assessmentDigest: string;
    operationId: string;
    acceptedReviewDimensions?: string[];
    reason?: string;
  }) {
    const ledger = this.requireLedger();
    const current = ledger.getRef(input.ref);
    return promoteRememberoReview({
      ledger,
      ref: input.ref,
      candidateVersionDigest: input.candidateVersionDigest,
      assessmentDigest: input.assessmentDigest,
      operationId: input.operationId,
      expectedCurrentVersionDigest: current?.versionDigest,
      acceptedReviewDimensions: input.acceptedReviewDimensions,
      reason: input.reason,
    });
  }

  semanticRefHistory(ref: string) {
    return this.requireLedger().refHistory(ref);
  }

  private allowlistedDocumentId(documentId?: string): string {
    const selected = documentId ?? DOCUMENT_SHOWCASE_DEFAULT_ID;
    if (!documentFixtureIds().includes(selected)) {
      throw new WebServiceError(
        'invalid_document_id',
        'Document ID is not allowlisted.'
      );
    }
    return selected;
  }

  documentShowcase(input?: { documentId?: string }) {
    return this.documentSnapshot(this.allowlistedDocumentId(input?.documentId));
  }

  documentMemorg() {
    return createDocumentMemorgExport();
  }

  parseDocument(input?: { documentId?: string }) {
    const documentId = this.allowlistedDocumentId(input?.documentId);
    const seeded = seedDocumentShowcase(this.options.store, documentId);
    return {
      ...this.documentSnapshot(documentId),
      parse: seeded.parse,
    };
  }

  parseAllDocuments() {
    return seedAllDocumentShowcases(this.options.store);
  }

  askDocument(input: { questionId: string; documentId?: string }) {
    const documentId = this.allowlistedDocumentId(input.documentId);
    const questionId = input.questionId.trim();
    if (!documentQuestionIds(documentId).includes(questionId)) {
      throw new WebServiceError(
        'invalid_document_question',
        'Document question ID is not allowlisted.'
      );
    }
    return askDocumentQuestion(this.options.store, questionId, documentId);
  }
}

/** Canonical product-name alias. The original export remains for compatibility. */
export { RemberoWebService as RememberoWebService };
export type RememberoWebServiceOptions = RemberoWebServiceOptions;
