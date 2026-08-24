import {
  canonicalKey,
  serializeClause,
  type Clause,
} from '../engine/index.js';
import {
  diffRecordedKnowledge,
  type RecordedKnowledgeDiffOptions,
  type RecordedKnowledgeDiffResult,
} from '../knowledge/recorded-diff.js';
import {
  knowledgeProgramDigest,
  type MemorySource,
  type MemoryStore,
  type RecordedKnowledgeSnapshot,
} from '../store/store.js';
import {
  type CreateSemanticVersionInput,
  type SemanticLedger,
  type SemanticObject,
  type SemanticVersion,
  type SemanticVersionContractInput,
  type SemanticVersionEdgeInput,
  type SemanticVersionMemberInput,
} from './semantic-ledger.js';

export const REMEMBERO_KNOWLEDGE_SNAPSHOT_FORMAT = 'remembero-knowledge-snapshot';
export const REMEMBERO_KNOWLEDGE_SNAPSHOT_VERSION = 1;
export const REMEMBERO_KNOWLEDGE_OBJECT_KIND = 'remembero.knowledge-snapshot';
export const DEFAULT_KNOWLEDGE_MEMBER_KEY = 'knowledge';

export interface RememberoKnowledgeSnapshotClause {
  namespace: string;
  clause: string;
  sources: MemorySource[];
}

export interface RememberoKnowledgeSnapshotValue {
  format: typeof REMEMBERO_KNOWLEDGE_SNAPSHOT_FORMAT;
  version: typeof REMEMBERO_KNOWLEDGE_SNAPSHOT_VERSION;
  recorded: {
    sequence: number;
    journalEntries: number;
  };
  namespaces: string[];
  stateDigest: string;
  clauses: RememberoKnowledgeSnapshotClause[];
}

export interface CaptureKnowledgeVersionOptions {
  namespaces?: string[] | '*';
  recordedSequence?: number;
  parents?: string[];
  memberKey?: string;
  members?: SemanticVersionMemberInput[];
  edges?: SemanticVersionEdgeInput[];
  contracts?: SemanticVersionContractInput[];
  metadata?: unknown;
  label?: string;
  createdAt?: Date | string;
}

export interface CapturedKnowledgeVersion {
  version: SemanticVersion;
  knowledgeObject: SemanticObject;
  recordedSnapshot: RecordedKnowledgeSnapshot;
}

export interface DiffKnowledgeVersionsOptions extends RecordedKnowledgeDiffOptions {
  memberKey?: string;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function jsonSource(source: MemorySource): MemorySource {
  return JSON.parse(JSON.stringify(source)) as MemorySource;
}

function clausesByNamespace(
  snapshot: RecordedKnowledgeSnapshot
): Map<string, Clause[]> {
  const output = new Map(snapshot.namespaces.map((namespace) => [namespace, [] as Clause[]]));
  for (const clause of snapshot.clauses) {
    const sources = snapshot.sources.get(canonicalKey(clause)) ?? [];
    const namespaces = [...new Set(sources.map((source) => source.namespace))];
    if (namespaces.length === 0) {
      throw new Error(
        `recorded knowledge clause '${serializeClause(clause)}' has no durable namespace source`
      );
    }
    for (const namespace of namespaces) {
      const clauses = output.get(namespace);
      if (clauses === undefined) {
        throw new Error(`recorded knowledge source references unselected namespace '${namespace}'`);
      }
      clauses.push(clause);
    }
  }
  return output;
}

function snapshotValue(
  snapshot: RecordedKnowledgeSnapshot
): RememberoKnowledgeSnapshotValue {
  const byNamespace = clausesByNamespace(snapshot);
  const clauses: RememberoKnowledgeSnapshotClause[] = [];
  for (const namespace of snapshot.namespaces) {
    for (const clause of byNamespace.get(namespace) ?? []) {
      const key = canonicalKey(clause);
      clauses.push({
        namespace,
        clause: serializeClause(clause),
        sources: (snapshot.sources.get(key) ?? [])
          .filter((source) => source.namespace === namespace)
          .map(jsonSource)
          .sort(
            (left, right) =>
              compareText(left.opId, right.opId) || compareText(left.ts, right.ts)
          ),
      });
    }
  }
  clauses.sort(
    (left, right) =>
      compareText(left.namespace, right.namespace) || compareText(left.clause, right.clause)
  );
  return {
    format: REMEMBERO_KNOWLEDGE_SNAPSHOT_FORMAT,
    version: REMEMBERO_KNOWLEDGE_SNAPSHOT_VERSION,
    recorded: {
      sequence: snapshot.sequence,
      journalEntries: snapshot.journalEntries,
    },
    namespaces: [...snapshot.namespaces],
    stateDigest: knowledgeProgramDigest(snapshot.namespaces, byNamespace),
    clauses,
  };
}

function knowledgeSnapshotValue(
  ledger: SemanticLedger,
  versionDigest: string,
  memberKey: string
): RememberoKnowledgeSnapshotValue {
  const version = ledger.getVersion(versionDigest);
  const member = version.members.find((entry) => entry.key === memberKey);
  if (member === undefined) {
    throw new Error(`semantic version '${versionDigest}' has no '${memberKey}' member`);
  }
  const object = ledger.getObject(member.objectDigest);
  if (object.kind !== REMEMBERO_KNOWLEDGE_OBJECT_KIND) {
    throw new Error(`semantic member '${memberKey}' is not a Remembero knowledge snapshot`);
  }
  if (
    typeof object.value !== 'object' ||
    object.value === null ||
    Array.isArray(object.value)
  ) {
    throw new Error('Remembero knowledge snapshot object is invalid');
  }
  const value = object.value as unknown as RememberoKnowledgeSnapshotValue;
  if (
    value.format !== REMEMBERO_KNOWLEDGE_SNAPSHOT_FORMAT ||
    value.version !== REMEMBERO_KNOWLEDGE_SNAPSHOT_VERSION ||
    !Array.isArray(value.namespaces) ||
    typeof value.recorded?.sequence !== 'number' ||
    typeof value.recorded?.journalEntries !== 'number' ||
    typeof value.stateDigest !== 'string' ||
    !Array.isArray(value.clauses)
  ) {
    throw new Error('Remembero knowledge snapshot object has an unsupported shape');
  }
  return value;
}

/**
 * Capture Remembero's exact recorded head (or an explicit sequence) as one generic
 * semantic-version member. The ledger remains independently usable for non-knowledge objects.
 */
export function captureKnowledgeVersion(
  ledger: SemanticLedger,
  store: MemoryStore,
  options: CaptureKnowledgeVersionOptions = {}
): CapturedKnowledgeVersion {
  const namespaces = options.namespaces ?? '*';
  const recordedSnapshot = options.recordedSequence === undefined
    ? store.recordedHead(namespaces)
    : store.recordedSnapshot(namespaces, options.recordedSequence);
  const value = snapshotValue(recordedSnapshot);
  const knowledgeObject = ledger.putObject({
    kind: REMEMBERO_KNOWLEDGE_OBJECT_KIND,
    value,
    createdAt: options.createdAt,
  });
  const memberKey = options.memberKey ?? DEFAULT_KNOWLEDGE_MEMBER_KEY;
  const members: SemanticVersionMemberInput[] = [
    { key: memberKey, objectDigest: knowledgeObject.digest },
    ...(options.members ?? []),
  ];
  const versionInput: CreateSemanticVersionInput = {
    parents: options.parents,
    members,
    edges: options.edges,
    contracts: options.contracts,
    metadata: {
      remembero: {
        knowledgeMemberKey: memberKey,
        stateDigest: value.stateDigest,
        recordedSequence: value.recorded.sequence,
        journalEntries: value.recorded.journalEntries,
        namespaces: value.namespaces,
      },
      integration: options.metadata ?? {},
    },
    label: options.label,
    createdAt: options.createdAt,
  };
  return {
    version: ledger.createVersion(versionInput),
    knowledgeObject,
    recordedSnapshot,
  };
}

/** Compare the exact Remembero knowledge states referenced by two generic versions. */
export function diffKnowledgeVersions(
  ledger: SemanticLedger,
  store: MemoryStore,
  fromVersionDigest: string,
  toVersionDigest: string,
  options: DiffKnowledgeVersionsOptions = {}
): RecordedKnowledgeDiffResult {
  const memberKey = options.memberKey ?? DEFAULT_KNOWLEDGE_MEMBER_KEY;
  const before = knowledgeSnapshotValue(ledger, fromVersionDigest, memberKey);
  const after = knowledgeSnapshotValue(ledger, toVersionDigest, memberKey);
  const namespaces = options.namespaces ?? [
    ...new Set([...before.namespaces, ...after.namespaces]),
  ].sort(compareText);
  const { memberKey: _memberKey, ...diffOptions } = options;
  return diffRecordedKnowledge(
    store,
    before.recorded.sequence,
    after.recorded.sequence,
    { ...diffOptions, namespaces }
  );
}
