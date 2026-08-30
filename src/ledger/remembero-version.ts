import { createHash } from 'node:crypto';
import { DEFAULT_MODEL } from '../llm/client.js';
import { isIntegrityConstraint, serializeClause } from '../engine/index.js';
import { createDocumentMemorgExport } from '../document/memorg.js';
import type { MemoryStore, RecordedKnowledgeSnapshot } from '../store/store.js';
import {
  captureKnowledgeSnapshot,
  type CaptureKnowledgeVersionOptions,
  type CapturedKnowledgeSnapshot,
} from './knowledge-version.js';
import type {
  SemanticLedger,
  SemanticObject,
  SemanticVersion,
  SemanticVersionContractInput,
  SemanticVersionEdgeInput,
  SemanticVersionMemberInput,
} from './semantic-ledger.js';

export const REMEMBERO_VERSION_ADAPTER = 'remembero.semantic-version/v1';

export interface RememberoDocumentVersionInput {
  id: string;
  digest: string;
  namespace?: string;
  fixtureVersion?: string;
  sourceUrl?: string;
}

export interface RememberoVersionMetadata {
  provider?: string;
  model?: string;
  runtime?: string;
  evaluationSuite?: string;
  integrityPolicy?: string;
  [key: string]: unknown;
}

export interface CaptureRememberoVersionOptions
  extends Pick<CaptureKnowledgeVersionOptions, 'namespaces' | 'recordedSequence' | 'parents' | 'label' | 'createdAt'> {
  ledger: SemanticLedger;
  store: MemoryStore;
  documents?: RememberoDocumentVersionInput[];
  model?: { provider: string; model: string; configuration?: unknown };
  runtime?: { packageVersion: string; nodeVersion?: string; platform?: string };
  evaluationSuite?: { id: string; version: string; digest?: string };
  integrityPolicy?: { id: string; version: string; constraints?: string[] };
  metadata?: RememberoVersionMetadata;
}

export interface RememberoVersionCapture {
  version: SemanticVersion;
  knowledge: CapturedKnowledgeSnapshot;
  members: Record<string, SemanticObject>;
  documents: RememberoDocumentVersionInput[];
  recordedSnapshot: RecordedKnowledgeSnapshot;
}

function sha256(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function defaultDocuments(): RememberoDocumentVersionInput[] {
  const exportArtifact = createDocumentMemorgExport();
  return [
    {
      id: 'document-intelligence-corpus',
      digest: exportArtifact.sha256,
      fixtureVersion: String(exportArtifact.version),
    },
  ];
}

function constraintsFromSnapshot(snapshot: RecordedKnowledgeSnapshot): string[] {
  return snapshot.clauses
    .filter((clause) => isIntegrityConstraint(clause))
    .map((clause) => serializeClause(clause))
    .sort();
}

function object(
  ledger: SemanticLedger,
  kind: string,
  value: unknown,
  createdAt?: Date | string
): SemanticObject {
  return ledger.putObject({ kind, value, createdAt });
}

/**
 * Capture a complete Remembero semantic version without mutating memory.
 * The memory journal remains the source of knowledge truth; the semantic ledger
 * records the exact snapshot and the surrounding application/runtime objects.
 */
export function captureRememberoVersion(
  options: CaptureRememberoVersionOptions
): RememberoVersionCapture {
  const documents = options.documents ?? defaultDocuments();
  const model = options.model ?? {
    provider: process.env.LLM_API_KEY === undefined ? 'unconfigured' : 'openrouter',
    model: process.env.LLM_MODEL ?? DEFAULT_MODEL,
  };
  const runtime = options.runtime ?? {
    packageVersion: '0.55.0',
    nodeVersion: process.version,
    platform: process.platform,
  };
  const evaluationSuite = options.evaluationSuite ?? {
    id: 'document-showcase',
    version: '1',
    digest: sha256({ id: 'document-showcase', version: '1', documents }),
  };

  const captured = captureKnowledgeSnapshot(options.ledger, options.store, {
    namespaces: options.namespaces,
    recordedSequence: options.recordedSequence,
    createdAt: options.createdAt,
  });
  const constraints = options.integrityPolicy?.constraints ??
    constraintsFromSnapshot(captured.recordedSnapshot);
  const stateDigest = captured.knowledgeObject.value;
  const rules = object(options.ledger, 'remembero.rules', {
    programDigest:
      typeof stateDigest === 'object' && stateDigest !== null && !Array.isArray(stateDigest)
        ? stateDigest.stateDigest
        : undefined,
    namespaces:
      typeof stateDigest === 'object' && stateDigest !== null && !Array.isArray(stateDigest)
        ? stateDigest.namespaces
        : [],
  }, options.createdAt);
  const integrityPolicy = object(options.ledger, 'remembero.integrity-policy', {
    id: options.integrityPolicy?.id ?? 'remembero-integrity',
    version: options.integrityPolicy?.version ?? '1',
    constraints,
    digest: sha256(constraints),
  }, options.createdAt);
  const documentObject = object(options.ledger, 'remembero.documents', {
    documents: documents.map((document) => ({ ...document })).sort((a, b) => a.id.localeCompare(b.id)),
  }, options.createdAt);
  const modelObject = object(options.ledger, 'remembero.model', model, options.createdAt);
  const runtimeObject = object(options.ledger, 'remembero.runtime', runtime, options.createdAt);
  const evaluationObject = object(
    options.ledger,
    'remembero.evaluation-suite',
    evaluationSuite,
    options.createdAt
  );
  const applicationObject = object(options.ledger, 'remembero.application', {
    adapter: REMEMBERO_VERSION_ADAPTER,
    package: 'remembero',
  }, options.createdAt);

  const members: SemanticVersionMemberInput[] = [
    { key: 'application', objectDigest: applicationObject.digest },
    { key: 'documents', objectDigest: documentObject.digest },
    { key: 'evaluation-suite', objectDigest: evaluationObject.digest },
    { key: 'integrity-policy', objectDigest: integrityPolicy.digest },
    { key: 'knowledge', objectDigest: captured.knowledgeObject.digest },
    { key: 'model', objectDigest: modelObject.digest },
    { key: 'rules', objectDigest: rules.digest },
    { key: 'runtime', objectDigest: runtimeObject.digest },
  ];
  const edges: SemanticVersionEdgeInput[] = [
    { kind: 'contains', from: 'application', to: 'knowledge' },
    { kind: 'produces', from: 'documents', to: 'knowledge' },
    { kind: 'consumes', from: 'rules', to: 'knowledge' },
    { kind: 'evaluated-by', from: 'knowledge', to: 'evaluation-suite' },
    { kind: 'evaluates', from: 'integrity-policy', to: 'knowledge' },
    { kind: 'requires', from: 'runtime', to: 'model' },
  ];
  const contracts: SemanticVersionContractInput[] = [
    {
      key: 'knowledge-evaluation',
      kind: 'evaluation-contract',
      objectDigest: evaluationObject.digest,
      consumer: 'evaluation-suite',
      provider: 'knowledge',
    },
    {
      key: 'rule-knowledge',
      kind: 'rule-contract',
      objectDigest: rules.digest,
      consumer: 'rules',
      provider: 'knowledge',
    },
  ];
  const finalVersion = options.ledger.createVersion({
    parents: options.parents,
    members,
    edges,
    contracts,
    label: options.label,
    createdAt: options.createdAt,
    metadata: {
      adapter: REMEMBERO_VERSION_ADAPTER,
      remembero: options.metadata ?? {},
      recorded: {
        sequence: captured.recordedSnapshot.sequence,
        journalEntries: captured.recordedSnapshot.journalEntries,
        namespaces: captured.recordedSnapshot.namespaces,
      },
    },
  });

  return {
    version: finalVersion,
    knowledge: captured,
    members: {
      application: applicationObject,
      documents: documentObject,
      'evaluation-suite': evaluationObject,
      'integrity-policy': integrityPolicy,
      knowledge: captured.knowledgeObject,
      model: modelObject,
      rules,
      runtime: runtimeObject,
    },
    documents,
    recordedSnapshot: captured.recordedSnapshot,
  };
}
