export type NavigationView = 'ask' | 'documents' | 'knowledge' | 'graph' | 'rules' | 'versions';
export type SearchKind = 'fact' | 'rule' | 'constraint';
export type HealthTone = 'healthy' | 'review' | 'violations';
export type DocumentEvidenceKind =
  | 'raw_region'
  | 'proposed_claim'
  | 'accepted_fact'
  | 'rule'
  | 'conclusion';
export type DocumentQuestionStatus = 'answered' | 'unsupported';

export interface ProfileSummary {
  workspaceLabel: string;
  personaLabel: string;
  storageLabel: string;
}

export interface AskPreset {
  id: string;
  label: string;
  question: string;
}

export interface MemoryPulse {
  factCount: number;
  ruleCount: number;
  sourceCoveragePercent: number;
  healthTone: HealthTone;
  healthLabel: string;
  findingCount: number;
}

export interface SourceItem {
  id: string;
  label: string;
  detail: string;
  dateLabel: string;
  namespace: string;
}

export interface ProofClaim {
  id: string;
  clause: string;
  supportingSourceIds: string[];
}

export interface RecentMemoryItem {
  id: string;
  title: string;
  detail: string;
  dateLabel: string;
  clause?: string;
  sourceLabel?: string;
}

export interface KnowledgeResultItem {
  id: string;
  rank: number;
  kind: SearchKind;
  clause: string;
  score: number;
  reasonSummary: string;
  sourcePreview: string;
}

export interface RuleListItem {
  id: string;
  clause: string;
  summary: string;
  status: 'stable' | 'review';
  sourceLabel: string;
}

export interface GraphNodeView {
  id: string;
  label: string;
  aliases: string[];
  emphasis: boolean;
}

export interface GraphLinkView {
  id: string;
  from: string;
  to: string;
  label: string;
}

export interface GraphRelationship {
  id: string;
  clause: string;
  label: string;
  left: string;
  right?: string;
}

export interface GraphData {
  focus: string | null;
  nodes: GraphNodeView[];
  links: GraphLinkView[];
  relationships: GraphRelationship[];
}

export interface BootstrapResponse {
  profile: ProfileSummary;
  memoryPulse: MemoryPulse;
  askPresets: AskPreset[];
  recentMemory: RecentMemoryItem[];
  knowledgeHighlights: KnowledgeResultItem[];
  graph: GraphData;
  rules: RuleListItem[];
  healthFindings: string[];
}

export interface SemanticRefView {
  name: string;
  versionDigest: string;
  updatedAt: string;
}

export interface SemanticVersionView {
  digest: string;
  labels: string[];
  parents: string[];
  createdAt: string;
  status: 'baseline' | 'candidate' | 'review' | 'blocked' | 'promoted';
  memberKeys: string[];
  edgeCount: number;
  contractCount: number;
  changed: boolean;
  compatibility?: {
    digest: string;
    checks: Array<{ dimension: string; status: string; summary: string }>;
  };
}

export interface SemanticVersionWorkspace {
  refs: SemanticRefView[];
  versions: SemanticVersionView[];
}

export interface SemanticVersionReview {
  candidateVersionDigest: string;
  baselineVersionDigest?: string;
  diff: Record<string, unknown>;
  evidence: Array<{ digest: string; kind: string; status: string; evaluator?: string; metrics: Record<string, number | null> }>;
  assessment: {
    digest: string;
    checks: Array<{ dimension: string; status: string; summary: string }>;
  };
}

export interface AskResponse {
  question: string;
  query: string;
  answer: string;
  status: string;
  claims: ProofClaim[];
  sources: SourceItem[];
  relatedKnowledge: KnowledgeResultItem[];
  graph: GraphData;
}

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

export interface DocumentClaim {
  id: string;
  kind: 'accepted' | 'proposed';
  clause: string;
  summary: string;
  pageId: string;
  pageNumber: number;
  regionId: string;
  sourceText: string;
  opId?: string;
  reviewLabel: string;
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

export interface DocumentSource {
  publisher: string;
  url: string;
  retrievedAt: string;
  sha256: string;
  pdfPageCount: number;
  selectedPageNumbers: number[];
  rightsNote: string;
}

export interface DocumentQuestion {
  id: string;
  label: string;
  question: string;
}

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

export interface DocumentCorpusEvaluation {
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

export interface LiveOcrEvidenceMetric {
  passed: number;
  total: number;
  percent: number;
}

export interface LiveOcrEvidenceSnapshot {
  generatedAt: string;
  source: string;
  model: string;
  mode: string;
  status: 'pass' | 'fail' | 'blocked';
  documentCount: number;
  completedDocuments: number;
  errorDocuments?: number;
  operationalMessage?: string;
  requiredFieldRecall: LiveOcrEvidenceMetric;
  readingOrderRecall: LiveOcrEvidenceMetric;
  readingOrderOrder: LiveOcrEvidenceMetric;
  groundingCoordinateCoverage: LiveOcrEvidenceMetric;
  tableDetection: LiveOcrEvidenceMetric;
  normalizedSimilarityPercent?: number;
  totalLatencyMs: number;
  averageDocumentLatencyMs: number;
  maximumDocumentLatencyMs: number;
  authorityBoundary: string;
}

export interface DocumentMemorgExportSummary {
  format: string;
  version: number;
  targetVersion: string;
  sha256: string;
  itemCount: number;
  downloadUrl: string;
}

export interface ProductShipModelEvidence {
  model: string;
  role: 'default' | 'frontier' | 'economy';
  recallAccuracyPercent: number;
  recallTokens: number;
  recallCostUsd: number;
  recallDurationMs: number;
  extractionAccuracyPercent: number;
  extractionTokens: number;
  extractionCostUsd: number;
  extractionDurationMs: number;
}

export interface ProductShipEvidence {
  generatedAt: string;
  decision: string;
  defaultModel: string;
  testFiles: number;
  passingTests: number;
  skippedTests: number;
  deterministicDocumentQuestions: number;
  deterministicDocumentAccuracyPercent: number;
  deterministicModelCalls: number;
  deterministicTokens: number;
  deterministicProviderCostUsd: number;
  models: ProductShipModelEvidence[];
  liveOcrStatus: string;
  boundary: string;
}

export interface DocumentListItem {
  id: string;
  fileName: string;
  title: string;
  kindLabel: string;
  pageCount: number;
  questionCount: number;
  acceptedClaimCount: number;
  proposedClaimCount: number;
  supportedQuestionCount: number;
  unsupportedQuestionCount: number;
  publisher?: string;
  sourceUrl?: string;
  evaluation?: DocumentEvaluationSummary;
}

export interface DocumentShowcase {
  id: string;
  namespace: string;
  kindLabel: string;
  fileName: string;
  title: string;
  parserLabel: string;
  parserMode: string;
  source: DocumentSource;
  pages: DocumentPage[];
  claims: DocumentClaim[];
}

export interface DocumentParseState {
  status: string;
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
  pageNumber?: number;
  regionId?: string;
  anchorLabel?: string;
  badge?: string;
}

export interface DocumentProofStep {
  id: string;
  kind: 'accepted_fact' | 'rule' | 'conclusion';
  clause?: string;
  label: string;
  detail: string;
  sourceIds: string[];
  pageNumber?: number;
  regionId?: string;
  anchorLabel?: string;
  badge: string;
}

export interface DocumentProofResult {
  questionId: string;
  question: string;
  status: DocumentQuestionStatus;
  query: string;
  answer: string;
  bindings: Array<Record<string, string>>;
  steps: DocumentProofStep[];
  sources: DocumentEvidenceItem[];
  relatedEvidence: DocumentEvidenceItem[];
}

export interface DocumentShowcaseResponse {
  documents: DocumentListItem[];
  document: DocumentShowcase;
  parse: DocumentParseState;
  questions: DocumentQuestion[];
  defaultQuestionId: string;
  proof: DocumentProofResult;
  evaluation?: DocumentEvaluationSummary;
  corpusEvaluation?: DocumentCorpusEvaluation;
  liveOcrEvidence?: LiveOcrEvidenceSnapshot;
  memorgExport?: DocumentMemorgExportSummary;
  shipEvidence?: ProductShipEvidence;
}

export interface SearchResponse {
  text: string;
  kinds: SearchKind[];
  status: 'matches' | 'no_match';
  results: KnowledgeResultItem[];
}

export interface GraphResponse {
  focus: string | null;
  graph: GraphData;
}

export interface MemoryMutationResponse {
  ok: boolean;
  message: string;
}

interface MemorySourceRecord {
  namespace: string;
  opId: string;
  ts: string;
  text?: string;
}

const DEFAULT_PRESETS: AskPreset[] = [
  {
    id: 'owners',
    label: 'Who owns Atlas?',
    question: 'Who owns Atlas?',
  },
  {
    id: 'northstar',
    label: 'What is Northstar?',
    question: 'What is Northstar?',
  },
  {
    id: 'contributors',
    label: 'Who worked with Maya?',
    question: 'Who worked with Maya?',
  },
  {
    id: 'rules',
    label: 'Show rules about projects',
    question: 'Show rules about projects',
  },
];

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function requestJson<TPayload>(
  path: string,
  init?: RequestInit
): Promise<TPayload> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body === undefined ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const payload = JSON.parse(text) as { message?: unknown };
      if (typeof payload.message === 'string') message = payload.message;
    } catch {
      // Preserve a non-JSON server response verbatim.
    }
    throw new ApiError(message || `Request failed for ${path}`, response.status);
  }
  return (await response.json()) as TPayload;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function titleCase(value: string): string {
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatPredicate(value: string): string {
  return titleCase(value.replaceAll('/', ' '));
}

export function formatClause(predicate: string, values: Array<string | number>): string {
  return `${predicate}(${values.map((value) => String(value)).join(', ')})`;
}

function formatDateLabel(value: unknown): string {
  if (typeof value !== 'string' || value.length === 0) return 'Recent';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString('en-AU', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

function entitySignature(valueType: unknown, value: unknown): string {
  const kind = valueType === 'number' ? 'number' : 'atom';
  return `${kind}:${String(value)}`;
}

function normalizeSourceItem(record: unknown, index: number): SourceItem {
  const source = isRecord(record) ? record : {};
  const namespace = asString(source.namespace, 'memory');
  const detail = asString(source.text, `${namespace}/${asString(source.opId, 'source')}`);
  const opId = asString(source.opId, String(index));
  const knownLabels: Record<string, string> = {
    'web-demo-atlas-session-v1': 'Atlas planning session',
    'web-demo-directory-v1': 'Personal directory',
    'web-demo-rules-v1': 'Reviewed personal rules',
  };
  return {
    id: `${namespace}:${opId || index}`,
    label:
      knownLabels[opId] ?? (detail.length > 72 ? `${detail.slice(0, 69)}...` : detail),
    detail,
    dateLabel: formatDateLabel(source.ts),
    namespace,
  };
}

function normalizeKnowledgeItem(record: unknown, index: number): KnowledgeResultItem {
  const item = isRecord(record) ? record : {};
  const reasons = asArray<Record<string, unknown>>(item.reasons);
  const sources = asArray<Record<string, unknown>>(item.sources);
  const reasonSummary = reasons
    .map((reason) => titleCase(asString(reason.kind).replaceAll('_', ' ')))
    .filter(Boolean)
    .slice(0, 2)
    .join(' • ');
  return {
    id: asString(item.id, `knowledge-${index}`),
    rank: asNumber(item.rank, index + 1),
    kind: (asString(item.kind, 'fact') as SearchKind),
    clause: asString(item.clause, 'No clause provided.'),
    score: asNumber(item.score),
    reasonSummary: reasonSummary || 'Local lexical match',
    sourcePreview: asString(sources[0]?.text, asString(sources[0]?.namespace, 'Local memory')),
  };
}

function normalizeRecentMemory(record: unknown, index: number): RecentMemoryItem {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `recent-${index}`),
    title: asString(item.title, asString(item.entity, 'Memory')),
    detail: asString(item.detail, asString(item.summary, 'Stored in local-first memory.')),
    dateLabel: asString(item.dateLabel, formatDateLabel(item.ts)),
    ...(typeof item.clause === 'string' ? { clause: item.clause } : {}),
    ...(typeof item.sourceLabel === 'string' ? { sourceLabel: item.sourceLabel } : {}),
  };
}

function normalizeRuleItem(record: unknown, index: number): RuleListItem {
  const item = isRecord(record) ? record : {};
  const clause = asString(item.clause, asString(item.rule, ''));
  const findingCount = asNumber(item.findingCount);
  const sources = asArray<Record<string, unknown>>(item.sources);
  return {
    id: asString(item.id, `rule-${index}`),
    clause: clause || 'No rule clause available.',
    summary:
      asString(item.summary) ||
      asString(item.message) ||
      (findingCount > 0 ? `${findingCount} audit note(s)` : 'Deterministic rule'),
    status:
      asString(item.status) === 'review' || findingCount > 0 ? 'review' : 'stable',
    sourceLabel: asString(
      item.sourceLabel,
      asString(sources[0]?.namespace, 'Knowledge rule')
    ),
  };
}

function graphPayload(payload: unknown): Record<string, unknown> {
  if (!isRecord(payload)) return {};
  if (isRecord(payload.graph)) return payload.graph;
  return payload;
}

function normalizeGraph(payload: unknown, preferredFocus?: string | null): GraphData {
  const root = graphPayload(payload);
  const selection = isRecord(root.selection) ? root.selection : {};
  const nodes = asArray<Record<string, unknown>>(root.nodes);
  const entityNodes = nodes.filter((node) => node.kind === 'entity');
  const claimNodes = nodes.filter((node) => node.kind === 'claim');
  const entityBySignature = new Map<
    string,
    { id: string; label: string; aliases: string[] }
  >();

  for (const node of entityNodes) {
    const label = String(node.value ?? '');
    entityBySignature.set(entitySignature(node.valueType, node.value), {
      id: asString(node.id, label),
      label,
      aliases: asArray<Record<string, unknown>>(node.aliases).map((alias) =>
        asString(alias.alias)
      ),
    });
  }

  const fallbackFocus = asString(selection.resolvedFocus, asString(selection.focus));
  const focusCandidate = preferredFocus ?? fallbackFocus;
  const focus = focusCandidate || entityNodes[0]?.value?.toString() || null;
  const relationships: GraphRelationship[] = [];
  const linkMap = new Map<string, GraphLinkView>();

  for (const [index, claim] of claimNodes.entries()) {
    const predicate = asString(claim.predicate, 'relates_to');
    const values = asArray<string | number>(claim.values);
    const entityValues = values
      .map((value) =>
        entityBySignature.get(entitySignature(typeof value === 'number' ? 'number' : 'atom', value))
      )
      .filter((value): value is { id: string; label: string; aliases: string[] } => value !== undefined);
    const clause = formatClause(predicate, values);
    const left = entityValues[0]?.label ?? String(values[0] ?? focus ?? 'memory');
    const right = entityValues[1]?.label;
    relationships.push({
      id: asString(claim.id, `relationship-${index}`),
      clause,
      label: formatPredicate(predicate),
      left,
      ...(right === undefined ? {} : { right }),
    });
    if (entityValues.length >= 2) {
      const [first, second] = entityValues;
      const key = `${first.id}:${second.id}:${predicate}`;
      if (!linkMap.has(key)) {
        linkMap.set(key, {
          id: key,
          from: first.id,
          to: second.id,
          label: predicate.replaceAll('_', ' '),
        });
      }
    }
  }

  const normalizedNodes = [...entityBySignature.values()]
    .sort((left, right) => {
      if (left.label === focus) return -1;
      if (right.label === focus) return 1;
      return left.label.localeCompare(right.label);
    })
    .map((node) => ({
      id: node.id,
      label: node.label,
      aliases: node.aliases,
      emphasis: node.label === focus,
    }));

  return {
    focus,
    nodes: normalizedNodes,
    links: [...linkMap.values()],
    relationships,
  };
}

function collectGraphSources(payload: unknown): SourceItem[] {
  const root = graphPayload(payload);
  const nodes = asArray<Record<string, unknown>>(root.nodes);
  const seen = new Set<string>();
  const sources: SourceItem[] = [];
  for (const node of nodes) {
    const rawSources = asArray<MemorySourceRecord>(node.sources);
    for (const source of rawSources) {
      const item = normalizeSourceItem(source, sources.length);
      if (seen.has(item.id)) continue;
      seen.add(item.id);
      sources.push(item);
    }
  }
  return sources;
}

function normalizeBootstrap(payload: unknown): BootstrapResponse {
  const root = isRecord(payload) ? payload : {};
  const health = isRecord(root.health) ? root.health : {};
  const healthStatus = asString(root.memoryPulse && isRecord(root.memoryPulse) ? root.memoryPulse.healthTone : health.status, 'healthy') as HealthTone;
  const topology = isRecord(health.rules) && isRecord(health.rules.topology)
    ? health.rules.topology
    : {};
  const ruleNodes =
    asArray(root.rules).length > 0
      ? asArray(root.rules)
      : asArray(topology.rules);
  const graph = normalizeGraph(root.graph, null);
  const knowledgeHighlights = asArray(root.knowledgeHighlights).map(normalizeKnowledgeItem);
  const recentMemory =
    asArray(root.recentMemory).length > 0
      ? asArray(root.recentMemory).map(normalizeRecentMemory)
      : collectGraphSources(root.graph).slice(0, 3).map((source, index) => ({
          id: `source-memory-${index}`,
          title: source.namespace === 'memory' ? 'Memory source' : titleCase(source.namespace),
          detail: source.detail,
          dateLabel: source.dateLabel,
          sourceLabel: source.namespace,
        }));
  const healthFindings = asArray<Record<string, unknown>>(health.findings)
    .slice(0, 4)
    .map((finding) => asString(finding.message))
    .filter(Boolean);
  return {
    profile: {
      workspaceLabel: asString(root.workspaceLabel, 'Remembero'),
      personaLabel:
        asString(isRecord(root.profile) ? root.profile.personaLabel : undefined) ||
        asString(isRecord(root.profile) ? root.profile.name : undefined) ||
        'Personal',
      storageLabel:
        asString(isRecord(root.profile) ? root.profile.storageLabel : undefined) ||
        'Local-first',
    },
    memoryPulse: {
      factCount:
        asNumber(isRecord(root.memoryPulse) ? root.memoryPulse.factCount : undefined) ||
        asNumber(health.clauseCount),
      ruleCount:
        asNumber(isRecord(root.memoryPulse) ? root.memoryPulse.ruleCount : undefined) ||
        asNumber(topology.ruleCount) ||
        ruleNodes.length,
      sourceCoveragePercent:
        asNumber(
          isRecord(root.memoryPulse) ? root.memoryPulse.sourceCoveragePercent : undefined
        ) || asNumber(isRecord(health.provenance) ? health.provenance.sourceCoveragePercent : undefined, 100),
      healthTone: healthStatus,
      healthLabel:
        asString(isRecord(root.memoryPulse) ? root.memoryPulse.healthLabel : undefined) ||
        titleCase(healthStatus),
      findingCount: healthFindings.length,
    },
    askPresets:
      asArray(root.askPresets).length > 0
        ? asArray<Record<string, unknown>>(root.askPresets).map((preset, index) => ({
            id: asString(preset.id, `preset-${index}`),
            label: asString(preset.label, asString(preset.question, `Preset ${index + 1}`)),
            question: asString(preset.question, ''),
          }))
        : DEFAULT_PRESETS,
    recentMemory,
    knowledgeHighlights,
    graph,
    rules: ruleNodes.map(normalizeRuleItem),
    healthFindings,
  };
}

function normalizeAsk(payload: unknown, question: string): AskResponse {
  const root = isRecord(payload) ? payload : {};
  const explanation = isRecord(root.explanation) ? root.explanation : {};
  const evidence = isRecord(root.evidence) ? root.evidence : {};
  const graph = normalizeGraph(explanation.graph ?? root.graph, null);
  const sources = collectGraphSources(explanation.graph ?? root.graph);
  const related = isRecord(root.relatedKnowledge) ? root.relatedKnowledge : {};
  const evidenceClaims = asArray<string>(evidence.claims);
  const claims = evidenceClaims.length > 0
    ? evidenceClaims.map((clause, index) => ({
        id: `evidence-claim-${index}`,
        clause,
        supportingSourceIds: [],
      }))
    : asArray<Record<string, unknown>>(graphPayload(explanation.graph).nodes)
        .filter((node) => node.kind === 'claim' && node.derived !== true)
        .map((node, index) => {
          const values = asArray<string | number>(node.values);
          return {
            id: asString(node.id, `claim-${index}`),
            clause: formatClause(asString(node.predicate, 'claim'), values),
            supportingSourceIds: asArray(node.sources).map((_, sourceIndex) =>
              `${asString(node.id, `claim-${index}`)}:${sourceIndex}`
            ),
          };
        });
  return {
    question,
    query: asString(root.query, question),
    answer: asString(root.answer, 'No answer returned.'),
    status: asString(root.status, 'answered'),
    claims,
    sources,
    relatedKnowledge: asArray(related.results).map(normalizeKnowledgeItem),
    graph,
  };
}

function normalizeDocumentBBox(value: unknown): DocumentBoundingBox {
  const root = isRecord(value) ? value : {};
  return {
    x: asNumber(root.x),
    y: asNumber(root.y),
    width: asNumber(root.width),
    height: asNumber(root.height),
  };
}

function normalizeDocumentRegion(record: unknown, index: number): DocumentRegion {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `region-${index}`),
    pageId: asString(item.pageId),
    pageNumber: asNumber(item.pageNumber, 1),
    order: asNumber(item.order, index + 1),
    label: asString(item.label, `R${index}`),
    kind: asString(item.kind, 'paragraph') as DocumentRegion['kind'],
    text: asString(item.text),
    bbox: normalizeDocumentBBox(item.bbox),
  };
}

function normalizeDocumentClaim(record: unknown, index: number): DocumentClaim {
  const item = isRecord(record) ? record : {};
  const evidence = isRecord(item.evidence) ? item.evidence : {};
  return {
    id: asString(item.id, `claim-${index}`),
    kind: asString(item.kind, 'proposed') as DocumentClaim['kind'],
    clause: asString(item.clause),
    summary: asString(item.summary),
    pageId: asString(evidence.pageId),
    pageNumber: asNumber(evidence.pageNumber, 1),
    regionId: asString(evidence.regionId),
    sourceText: asString(evidence.sourceText),
    reviewLabel: asString(item.reviewLabel, 'Reviewed'),
    ...(typeof item.operationId === 'string' ? { opId: item.operationId } : {}),
  };
}

function normalizeDocumentPage(record: unknown, index: number): DocumentPage {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `page-${index + 1}`),
    pageNumber: asNumber(item.pageNumber, index + 1),
    width: asNumber(item.width, 816),
    height: asNumber(item.height, 1056),
    label: asString(item.label, `Page ${index + 1}`),
    imageUrl: asString(item.imageUrl),
    imageSha256: asString(item.imageSha256),
    regions: asArray(item.regions).map(normalizeDocumentRegion),
  };
}

function normalizeDocumentQuestion(record: unknown, index: number): DocumentQuestion {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `question-${index}`),
    label: asString(item.label, `Question ${index + 1}`),
    question: asString(item.question),
  };
}

function normalizeDocumentListItem(record: unknown, index: number): DocumentListItem {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `document-${index}`),
    fileName: asString(item.fileName, 'Document.pdf'),
    title: asString(item.title, `Document ${index + 1}`),
    kindLabel: asString(item.kindLabel, 'Document'),
    pageCount: asNumber(item.pageCount),
    questionCount: asNumber(item.questionCount),
    acceptedClaimCount: asNumber(item.acceptedClaimCount),
    proposedClaimCount: asNumber(item.proposedClaimCount),
    supportedQuestionCount: asNumber(item.supportedQuestionCount),
    unsupportedQuestionCount: asNumber(item.unsupportedQuestionCount),
    ...(typeof item.publisher === 'string' ? { publisher: item.publisher } : {}),
    ...(typeof item.sourceUrl === 'string' ? { sourceUrl: item.sourceUrl } : {}),
    ...(item.evaluation === undefined
      ? {}
      : { evaluation: normalizeDocumentEvaluation(item.evaluation) }),
  };
}

function normalizeDocumentParse(payload: unknown): DocumentParseState {
  const root = isRecord(payload) ? payload : {};
  return {
    status: asString(root.status, 'ready'),
    fixtureDigest: asString(root.fixtureDigest),
    pageCount: asNumber(root.pageCount),
    regionCount: asNumber(root.regionCount),
    acceptedClaimCount: asNumber(root.acceptedClaimCount),
    proposedClaimCount: asNumber(root.proposedClaimCount),
    acceptedClaimCoveragePercent: asNumber(root.acceptedClaimCoveragePercent),
    pageCoveragePercent: asNumber(root.pageCoveragePercent),
    seededCount: asNumber(root.seededCount),
    duplicateCount: asNumber(root.duplicateCount),
  };
}

function normalizeDocumentEvidenceItem(record: unknown, index: number): DocumentEvidenceItem {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `document-evidence-${index}`),
    kind: asString(item.kind, 'raw_region') as DocumentEvidenceKind,
    label: asString(item.label, 'Document evidence'),
    detail: asString(item.detail),
    ...(typeof item.clause === 'string' ? { clause: item.clause } : {}),
    ...(typeof item.pageNumber === 'number' ? { pageNumber: item.pageNumber } : {}),
    ...(typeof item.regionId === 'string' ? { regionId: item.regionId } : {}),
    ...(typeof item.anchorLabel === 'string' ? { anchorLabel: item.anchorLabel } : {}),
    ...(typeof item.badge === 'string' ? { badge: item.badge } : {}),
  };
}

function normalizeDocumentProofStep(record: unknown, index: number): DocumentProofStep {
  const item = isRecord(record) ? record : {};
  return {
    id: asString(item.id, `document-step-${index}`),
    kind: asString(item.kind, 'accepted_fact') as DocumentProofStep['kind'],
    label: asString(item.label, 'Trace step'),
    detail: asString(item.detail),
    sourceIds: asArray<string>(item.sourceIds).filter((value) => typeof value === 'string'),
    badge: asString(item.badge, 'Trace'),
    ...(typeof item.clause === 'string' ? { clause: item.clause } : {}),
    ...(typeof item.pageNumber === 'number' ? { pageNumber: item.pageNumber } : {}),
    ...(typeof item.regionId === 'string' ? { regionId: item.regionId } : {}),
    ...(typeof item.anchorLabel === 'string' ? { anchorLabel: item.anchorLabel } : {}),
  };
}

function normalizeDocumentProof(payload: unknown): DocumentProofResult {
  const root = isRecord(payload) ? payload : {};
  const sources = asArray(root.sources).map(normalizeDocumentEvidenceItem);
  const sourcesById = new Map(sources.map((source) => [source.id, source]));
  const steps = asArray(root.steps).map(normalizeDocumentProofStep).map((step) => {
    const groundedSource = step.sourceIds
      .map((sourceId) => sourcesById.get(sourceId))
      .find((source) => source?.regionId !== undefined);
    if (groundedSource === undefined) return step;
    return {
      ...step,
      ...(groundedSource.pageNumber === undefined
        ? {}
        : { pageNumber: groundedSource.pageNumber }),
      ...(groundedSource.regionId === undefined ? {} : { regionId: groundedSource.regionId }),
      ...(groundedSource.anchorLabel === undefined
        ? {}
        : { anchorLabel: groundedSource.anchorLabel }),
    };
  });
  return {
    questionId: asString(root.questionId),
    question: asString(root.question),
    status: asString(root.status, 'unsupported') as DocumentQuestionStatus,
    query: asString(root.query),
    answer: asString(root.answer, 'No answer returned.'),
    bindings: asArray<Record<string, string>>(root.bindings).map((binding) =>
      Object.fromEntries(
        Object.entries(isRecord(binding) ? binding : {}).map(([key, value]) => [
          key,
          asString(value),
        ])
      )
    ),
    steps,
    sources,
    relatedEvidence: asArray(root.relatedEvidence).map(normalizeDocumentEvidenceItem),
  };
}

function normalizeDocument(payload: unknown): DocumentShowcase {
  const root = isRecord(payload) ? payload : {};
  const source = isRecord(root.source) ? root.source : {};
  return {
    id: asString(root.id, 'document'),
    namespace: asString(root.namespace, 'documents'),
    kindLabel: asString(root.kindLabel, 'Document'),
    fileName: asString(root.fileName, 'Document.pdf'),
    title: asString(root.title, 'Document'),
    parserLabel: asString(root.parserLabel, 'Reviewed OCR evidence'),
    parserMode: asString(root.parserMode, 'source_text_reviewed'),
    source: {
      publisher: asString(source.publisher, 'Unknown publisher'),
      url: asString(source.url),
      retrievedAt: asString(source.retrievedAt),
      sha256: asString(source.sha256),
      pdfPageCount: asNumber(source.pdfPageCount),
      selectedPageNumbers: asArray(source.selectedPageNumbers).map((page) => asNumber(page)),
      rightsNote: asString(source.rightsNote),
    },
    pages: asArray(root.pages).map(normalizeDocumentPage),
    claims: asArray(root.claims).map(normalizeDocumentClaim),
  };
}

function normalizeDocumentMetric(payload: unknown): DocumentEvaluationMetric {
  const root = isRecord(payload) ? payload : {};
  return {
    passed: asNumber(root.passed),
    total: asNumber(root.total),
    percent: asNumber(root.percent),
  };
}

function normalizeDocumentLatency(payload: unknown): DocumentEvaluationLatency {
  const root = isRecord(payload) ? payload : {};
  return {
    parseMs: asNumber(root.parseMs),
    averageQuestionMs: asNumber(root.averageQuestionMs),
    maxQuestionMs: asNumber(root.maxQuestionMs),
    totalMs: asNumber(root.totalMs),
  };
}

function normalizeDocumentEvaluationCheck(record: unknown, index: number): DocumentEvaluationCheck {
  const item = isRecord(record) ? record : {};
  return {
    questionId: asString(item.questionId, `check-${index}`),
    label: asString(item.label, `Check ${index + 1}`),
    question: asString(item.question),
    expectedStatus: asString(item.expectedStatus, 'unsupported') as DocumentQuestionStatus,
    actualStatus: asString(item.actualStatus, 'unsupported') as DocumentQuestionStatus,
    statusPass: asBoolean(item.statusPass),
    answerPass: asBoolean(item.answerPass),
    sourceRecallPass: asBoolean(item.sourceRecallPass),
    proofGroundingPass: asBoolean(item.proofGroundingPass),
    abstentionPass: asBoolean(item.abstentionPass),
    latencyMs: asNumber(item.latencyMs),
  };
}

function normalizeDocumentEvaluation(payload: unknown): DocumentEvaluationSummary {
  const root = isRecord(payload) ? payload : {};
  return {
    documentId: asString(root.documentId, 'document'),
    title: asString(root.title, 'Document'),
    namespace: asString(root.namespace, 'documents'),
    status: asString(root.status, 'fail') as 'pass' | 'fail',
    checks: asArray(root.checks).map(normalizeDocumentEvaluationCheck),
    metrics: {
      parseCoverage: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.parseCoverage : {}),
      answerAccuracy: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.answerAccuracy : {}),
      statusAccuracy: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.statusAccuracy : {}),
      sourceRecall: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.sourceRecall : {}),
      proofGrounding: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.proofGrounding : {}),
      abstentionCorrectness: normalizeDocumentMetric(
        isRecord(root.metrics) ? root.metrics.abstentionCorrectness : {}
      ),
      idempotency: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.idempotency : {}),
    },
    latencyMs: normalizeDocumentLatency(root.latencyMs),
  };
}

function normalizeDocumentCorpusEvaluation(payload: unknown): DocumentCorpusEvaluation {
  const root = isRecord(payload) ? payload : {};
  return {
    documentCount: asNumber(root.documentCount),
    questionCount: asNumber(root.questionCount),
    status: asString(root.status, 'fail') as 'pass' | 'fail',
    metrics: {
      parseCoverage: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.parseCoverage : {}),
      answerAccuracy: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.answerAccuracy : {}),
      statusAccuracy: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.statusAccuracy : {}),
      sourceRecall: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.sourceRecall : {}),
      proofGrounding: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.proofGrounding : {}),
      abstentionCorrectness: normalizeDocumentMetric(
        isRecord(root.metrics) ? root.metrics.abstentionCorrectness : {}
      ),
      idempotency: normalizeDocumentMetric(isRecord(root.metrics) ? root.metrics.idempotency : {}),
    },
    latencyMs: {
      totalParseMs: asNumber(isRecord(root.latencyMs) ? root.latencyMs.totalParseMs : 0),
      averageQuestionMs: asNumber(isRecord(root.latencyMs) ? root.latencyMs.averageQuestionMs : 0),
      maxQuestionMs: asNumber(isRecord(root.latencyMs) ? root.latencyMs.maxQuestionMs : 0),
      totalMs: asNumber(isRecord(root.latencyMs) ? root.latencyMs.totalMs : 0),
    },
  };
}

function normalizeLiveOcrEvidence(payload: unknown): LiveOcrEvidenceSnapshot {
  const root = isRecord(payload) ? payload : {};
  return {
    generatedAt: asString(root.generatedAt),
    source: asString(root.source),
    model: asString(root.model),
    mode: asString(root.mode),
    status: asString(root.status, 'fail') as 'pass' | 'fail' | 'blocked',
    documentCount: asNumber(root.documentCount),
    completedDocuments: asNumber(root.completedDocuments),
    requiredFieldRecall: normalizeDocumentMetric(root.requiredFieldRecall),
    readingOrderRecall: normalizeDocumentMetric(root.readingOrderRecall),
    readingOrderOrder: normalizeDocumentMetric(root.readingOrderOrder),
    groundingCoordinateCoverage: normalizeDocumentMetric(root.groundingCoordinateCoverage),
    tableDetection: normalizeDocumentMetric(root.tableDetection),
    ...(typeof root.errorDocuments === 'number' ? { errorDocuments: root.errorDocuments } : {}),
    ...(typeof root.operationalMessage === 'string'
      ? { operationalMessage: root.operationalMessage }
      : {}),
    ...(typeof root.normalizedSimilarityPercent === 'number'
      ? { normalizedSimilarityPercent: root.normalizedSimilarityPercent }
      : {}),
    totalLatencyMs: asNumber(root.totalLatencyMs),
    averageDocumentLatencyMs: asNumber(root.averageDocumentLatencyMs),
    maximumDocumentLatencyMs: asNumber(root.maximumDocumentLatencyMs),
    authorityBoundary: asString(root.authorityBoundary),
  };
}

function normalizeDocumentMemorgExport(payload: unknown): DocumentMemorgExportSummary {
  const root = isRecord(payload) ? payload : {};
  return {
    format: asString(root.format),
    version: asNumber(root.version),
    targetVersion: asString(root.targetVersion),
    sha256: asString(root.sha256),
    itemCount: asNumber(root.itemCount),
    downloadUrl: asString(root.downloadUrl, '/documents/document-intelligence.memorg.json'),
  };
}

function normalizeProductShipEvidence(payload: unknown): ProductShipEvidence {
  const root = isRecord(payload) ? payload : {};
  return {
    generatedAt: asString(root.generatedAt),
    decision: asString(root.decision),
    defaultModel: asString(root.defaultModel),
    testFiles: asNumber(root.testFiles),
    passingTests: asNumber(root.passingTests),
    skippedTests: asNumber(root.skippedTests),
    deterministicDocumentQuestions: asNumber(root.deterministicDocumentQuestions),
    deterministicDocumentAccuracyPercent: asNumber(root.deterministicDocumentAccuracyPercent),
    deterministicModelCalls: asNumber(root.deterministicModelCalls),
    deterministicTokens: asNumber(root.deterministicTokens),
    deterministicProviderCostUsd: asNumber(root.deterministicProviderCostUsd),
    models: asArray(root.models).map((value) => {
      const model = isRecord(value) ? value : {};
      return {
        model: asString(model.model),
        role: asString(model.role, 'frontier') as ProductShipModelEvidence['role'],
        recallAccuracyPercent: asNumber(model.recallAccuracyPercent),
        recallTokens: asNumber(model.recallTokens),
        recallCostUsd: asNumber(model.recallCostUsd),
        recallDurationMs: asNumber(model.recallDurationMs),
        extractionAccuracyPercent: asNumber(model.extractionAccuracyPercent),
        extractionTokens: asNumber(model.extractionTokens),
        extractionCostUsd: asNumber(model.extractionCostUsd),
        extractionDurationMs: asNumber(model.extractionDurationMs),
      };
    }),
    liveOcrStatus: asString(root.liveOcrStatus),
    boundary: asString(root.boundary),
  };
}

function normalizeDocumentSnapshot(payload: unknown): DocumentShowcaseResponse {
  const root = isRecord(payload) ? payload : {};
  const document = normalizeDocument(root.document);
  const questions = asArray(root.questions).map(normalizeDocumentQuestion);
  return {
    documents:
      asArray(root.documents).length > 0
        ? asArray(root.documents).map(normalizeDocumentListItem)
        : [
            {
              id: document.id,
              fileName: document.fileName,
              title: document.title,
              kindLabel: document.kindLabel,
              pageCount: document.pages.length,
              questionCount: questions.length,
              acceptedClaimCount: document.claims.filter((claim) => claim.kind === 'accepted').length,
              proposedClaimCount: document.claims.filter((claim) => claim.kind === 'proposed').length,
              supportedQuestionCount: questions.length,
              unsupportedQuestionCount: 0,
            },
          ],
    document,
    parse: normalizeDocumentParse(root.parse),
    questions,
    defaultQuestionId: asString(root.defaultQuestionId),
    proof: normalizeDocumentProof(root.proof),
    ...(root.evaluation === undefined ? {} : { evaluation: normalizeDocumentEvaluation(root.evaluation) }),
    ...(root.corpusEvaluation === undefined
      ? {}
      : { corpusEvaluation: normalizeDocumentCorpusEvaluation(root.corpusEvaluation) }),
    ...(root.liveOcrEvidence === undefined
      ? {}
      : { liveOcrEvidence: normalizeLiveOcrEvidence(root.liveOcrEvidence) }),
    ...(root.memorgExport === undefined
      ? {}
      : { memorgExport: normalizeDocumentMemorgExport(root.memorgExport) }),
    ...(root.shipEvidence === undefined
      ? {}
      : { shipEvidence: normalizeProductShipEvidence(root.shipEvidence) }),
  };
}

function normalizeSearch(
  payload: unknown,
  text: string,
  kinds: SearchKind[]
): SearchResponse {
  const root = isRecord(payload) ? payload : {};
  return {
    text: asString(root.text, text),
    kinds,
    status: asString(root.status, 'no_match') as 'matches' | 'no_match',
    results: asArray(root.results).map(normalizeKnowledgeItem),
  };
}

export async function getBootstrap(): Promise<BootstrapResponse> {
  const payload = await requestJson<unknown>('/api/bootstrap');
  return normalizeBootstrap(payload);
}

function normalizeVersionWorkspace(payload: unknown): SemanticVersionWorkspace {
  const root = isRecord(payload) ? payload : {};
  return {
    refs: asArray(root.refs).map((value) => {
      const record = isRecord(value) ? value : {};
      return {
        name: asString(record.name),
        versionDigest: asString(record.versionDigest),
        updatedAt: asString(record.updatedAt),
      };
    }),
    versions: asArray(root.versions).map((value) => {
      const record = isRecord(value) ? value : {};
      const compatibility = isRecord(record.compatibility) ? record.compatibility : undefined;
      return {
        digest: asString(record.digest),
        labels: asArray<string>(record.labels),
        parents: asArray<string>(record.parents),
        createdAt: asString(record.createdAt),
        status: asString(record.status, 'candidate') as SemanticVersionView['status'],
        memberKeys: asArray<string>(record.memberKeys),
        edgeCount: asNumber(record.edgeCount),
        contractCount: asNumber(record.contractCount),
        changed: asBoolean(record.changed),
        ...(compatibility === undefined
          ? {}
          : {
              compatibility: {
                digest: asString(compatibility.digest),
                checks: asArray(compatibility.checks).map((check) => {
                  const item = isRecord(check) ? check : {};
                  return {
                    dimension: asString(item.dimension),
                    status: asString(item.status),
                    summary: asString(item.summary),
                  };
                }),
              },
            }),
      };
    }),
  };
}

export async function getVersionWorkspace(): Promise<SemanticVersionWorkspace> {
  return normalizeVersionWorkspace(await requestJson<unknown>('/api/versions'));
}

export async function captureSemanticVersion(input: {
  label?: string;
  ref?: string;
}): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>('/api/versions/capture', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export async function reviewSemanticVersion(input: {
  candidateVersionDigest: string;
  includeDocumentEvaluation?: boolean;
}): Promise<SemanticVersionReview> {
  return requestJson<SemanticVersionReview>('/api/versions/review', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export async function promoteSemanticVersion(input: {
  ref: string;
  candidateVersionDigest: string;
  assessmentDigest: string;
  operationId: string;
  acceptedReviewDimensions?: string[];
  reason?: string;
}): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>('/api/versions/promote', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export async function askMemory(input: {
  question: string;
  presetId?: string;
}): Promise<AskResponse> {
  const payload = await requestJson<unknown>('/api/ask', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  return normalizeAsk(payload, input.question);
}

export async function createMemory(input: {
  subject: string;
  predicate: string;
  object: string;
  sourceText: string;
}): Promise<MemoryMutationResponse> {
  const payload = await requestJson<unknown>('/api/memory', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  const root = isRecord(payload) ? payload : {};
  return {
    ok: true,
    message:
      asString(root.message) ||
      `${input.subject} ${input.predicate} ${input.object} stored in local memory.`,
  };
}

export async function searchKnowledge(input: {
  text: string;
  kinds?: SearchKind[];
}): Promise<SearchResponse> {
  const payload = await requestJson<unknown>('/api/search', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  return normalizeSearch(payload, input.text, input.kinds ?? []);
}

export async function getGraph(focus?: string): Promise<GraphResponse> {
  const query = focus ? `?focus=${encodeURIComponent(focus)}` : '';
  const payload = await requestJson<unknown>(`/api/graph${query}`);
  const graph = normalizeGraph(payload, focus ?? null);
  return {
    focus: graph.focus,
    graph,
  };
}

export async function seedDemo(): Promise<MemoryMutationResponse> {
  const payload = await requestJson<unknown>('/api/seed', { method: 'POST' });
  const root = isRecord(payload) ? payload : {};
  return {
    ok: true,
    message: asString(root.message, 'Demo knowledge seeded.'),
  };
}

export async function getDocument(documentId?: string): Promise<DocumentShowcaseResponse> {
  const query = documentId ? `?documentId=${encodeURIComponent(documentId)}` : '';
  const payload = await requestJson<unknown>(`/api/document${query}`);
  return normalizeDocumentSnapshot(payload);
}

export async function parseDocument(input: { documentId: string }): Promise<DocumentShowcaseResponse> {
  const payload = await requestJson<unknown>('/api/document/parse', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  return normalizeDocumentSnapshot(payload);
}

export async function askDocumentQuestion(input: {
  documentId: string;
  questionId: string;
}): Promise<DocumentProofResult> {
  const payload = await requestJson<unknown>('/api/document/ask', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  return normalizeDocumentProof(payload);
}
