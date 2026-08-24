import { createHash, randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

export const SEMANTIC_LEDGER_SCHEMA_VERSION = 1;
export const MAX_SEMANTIC_OBJECT_BYTES = 16 * 1024 * 1024;
export const MAX_SEMANTIC_VERSION_PARENTS = 64;
export const MAX_SEMANTIC_VERSION_MEMBERS = 4_096;
export const MAX_SEMANTIC_VERSION_EDGES = 32_768;
export const MAX_SEMANTIC_VERSION_CONTRACTS = 4_096;
export const MAX_SEMANTIC_GRAPH_DEPTH = 16;
export const MAX_SEMANTIC_GRAPH_NODES = 4_096;

const DIGEST_RE = /^[a-f0-9]{64}$/;
const TABLE_PREFIX_RE = /^[a-z][a-z0-9_]{0,47}$/;
const KIND_RE = /^[a-z][a-z0-9._:-]{0,127}$/;
const KEY_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$/;
const REF_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$/;
const LABEL_RE = /^[A-Za-z0-9@][A-Za-z0-9@._/+:-]{0,255}$/;
const OPERATION_ID_RE = /^[^\s\0]+$/u;

export type SemanticJsonValue =
  | null
  | boolean
  | number
  | string
  | SemanticJsonValue[]
  | { [key: string]: SemanticJsonValue };

export type SemanticEvidenceStatus =
  | 'passed'
  | 'failed'
  | 'blocked'
  | 'error'
  | 'observed';

export type SemanticCompatibilityStatus =
  | 'pass'
  | 'fail'
  | 'review'
  | 'blocked'
  | 'not_applicable';

export type SemanticGraphDirection = 'upstream' | 'downstream' | 'both';

export interface SemanticLedgerOptions {
  /** Prefix for public ledger tables. Useful when several ledgers share one SQLite file. */
  tablePrefix?: string;
}

export interface PutSemanticObjectInput {
  kind: string;
  value: unknown;
  createdAt?: Date | string;
}

export interface SemanticObject {
  digest: string;
  kind: string;
  value: SemanticJsonValue;
  bytes: number;
  createdAt: string;
}

export interface SemanticVersionMemberInput {
  key: string;
  objectDigest: string;
}

export interface SemanticVersionMember extends SemanticVersionMemberInput {
  objectKind: string;
}

export interface SemanticVersionEdgeInput {
  kind: string;
  from: string;
  to: string;
  metadata?: unknown;
}

export interface SemanticVersionEdge {
  digest: string;
  kind: string;
  from: string;
  to: string;
  metadata: SemanticJsonValue;
}

export interface SemanticVersionContractInput {
  key: string;
  kind: string;
  objectDigest: string;
  consumer: string;
  provider: string;
}

export interface SemanticVersionContract extends SemanticVersionContractInput {
  objectKind: string;
}

export interface CreateSemanticVersionInput {
  parents?: string[];
  members: SemanticVersionMemberInput[];
  edges?: SemanticVersionEdgeInput[];
  contracts?: SemanticVersionContractInput[];
  metadata?: unknown;
  label?: string;
  createdAt?: Date | string;
}

export interface SemanticVersion {
  digest: string;
  parents: string[];
  members: SemanticVersionMember[];
  edges: SemanticVersionEdge[];
  contracts: SemanticVersionContract[];
  metadata: SemanticJsonValue;
  labels: string[];
  createdAt: string;
}

export type SemanticMetrics = Record<string, number | null>;

export interface RecordSemanticEvidenceInput {
  versionDigest: string;
  baselineVersionDigest?: string;
  kind: string;
  status: SemanticEvidenceStatus;
  evaluator?: string;
  payload?: unknown;
  metrics?: SemanticMetrics;
  createdAt?: Date | string;
}

export interface SemanticEvidence {
  sequence: number;
  digest: string;
  versionDigest: string;
  baselineVersionDigest?: string;
  kind: string;
  status: SemanticEvidenceStatus;
  evaluator?: string;
  payload: SemanticJsonValue;
  metrics: SemanticMetrics;
  createdAt: string;
}

export interface SemanticCompatibilityCheckInput {
  dimension: string;
  status: SemanticCompatibilityStatus;
  summary: string;
  evidenceDigests?: string[];
  details?: unknown;
}

export interface SemanticCompatibilityCheck {
  dimension: string;
  status: SemanticCompatibilityStatus;
  summary: string;
  evidenceDigests: string[];
  details: SemanticJsonValue;
}

export interface RecordSemanticCompatibilityInput {
  baselineVersionDigest?: string;
  candidateVersionDigest: string;
  checks: SemanticCompatibilityCheckInput[];
  metadata?: unknown;
  createdAt?: Date | string;
}

export interface SemanticCompatibilityAssessment {
  sequence: number;
  digest: string;
  baselineVersionDigest?: string;
  candidateVersionDigest: string;
  checks: SemanticCompatibilityCheck[];
  metadata: SemanticJsonValue;
  createdAt: string;
}

export interface SetSemanticRefInput {
  name: string;
  versionDigest: string;
  operationId: string;
  expectedCurrentVersionDigest?: string | null;
  reason?: string;
  createdAt?: Date | string;
}

export interface SemanticRef {
  name: string;
  versionDigest: string;
  updatedAt: string;
}

export interface SemanticRefEvent {
  sequence: number;
  digest: string;
  operationId: string;
  name: string;
  fromVersionDigest?: string;
  toVersionDigest: string;
  promotionDigest?: string;
  reason?: string;
  createdAt: string;
}

export interface PromoteSemanticVersionInput {
  ref: string;
  candidateVersionDigest: string;
  assessmentDigest: string;
  operationId: string;
  expectedCurrentVersionDigest?: string | null;
  acceptedReviewDimensions?: string[];
  reason?: string;
  createdAt?: Date | string;
}

export interface SemanticPromotionDecision {
  sequence: number;
  digest: string;
  operationId: string;
  ref: string;
  fromVersionDigest?: string;
  candidateVersionDigest: string;
  assessmentDigest: string;
  outcome: 'accepted' | 'rejected';
  acceptedReviewDimensions: string[];
  blockingDimensions: string[];
  reason?: string;
  createdAt: string;
}

export interface SemanticVersionDiffEntry<T> {
  key: string;
  before?: T;
  after?: T;
}

export interface SemanticEvidenceComparison {
  key: string;
  before?: SemanticEvidence;
  after?: SemanticEvidence;
  metricDelta: SemanticMetrics;
}

export interface SemanticVersionDiff {
  changed: boolean;
  from: SemanticVersion;
  to: SemanticVersion;
  members: {
    added: SemanticVersionMember[];
    removed: SemanticVersionMember[];
    changed: SemanticVersionDiffEntry<SemanticVersionMember>[];
  };
  edges: {
    added: SemanticVersionEdge[];
    removed: SemanticVersionEdge[];
  };
  contracts: {
    added: SemanticVersionContract[];
    removed: SemanticVersionContract[];
    changed: SemanticVersionDiffEntry<SemanticVersionContract>[];
  };
  evidence: SemanticEvidenceComparison[];
  compatibility?: SemanticCompatibilityAssessment;
}

export interface TraverseSemanticGraphOptions {
  direction?: SemanticGraphDirection;
  maxDepth?: number;
}

export interface SemanticGraphTraversal {
  versionDigest: string;
  focus: string;
  direction: SemanticGraphDirection;
  maxDepth: number;
  nodes: Array<SemanticVersionMember & { depth: number }>;
  edges: SemanticVersionEdge[];
}

interface LedgerTables {
  meta: string;
  objects: string;
  versions: string;
  parents: string;
  members: string;
  edges: string;
  contracts: string;
  labels: string;
  evidence: string;
  assessments: string;
  checks: string;
  promotions: string;
  refs: string;
  refEvents: string;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function quoted(identifier: string): string {
  return `"${identifier}"`;
}

function tableNames(prefix: string): LedgerTables {
  return {
    meta: `${prefix}_meta`,
    objects: `${prefix}_objects`,
    versions: `${prefix}_versions`,
    parents: `${prefix}_version_parents`,
    members: `${prefix}_version_members`,
    edges: `${prefix}_version_edges`,
    contracts: `${prefix}_version_contracts`,
    labels: `${prefix}_version_labels`,
    evidence: `${prefix}_evidence`,
    assessments: `${prefix}_assessments`,
    checks: `${prefix}_assessment_checks`,
    promotions: `${prefix}_promotions`,
    refs: `${prefix}_refs`,
    refEvents: `${prefix}_ref_events`,
  };
}

function normalizeJson(
  value: unknown,
  label: string,
  ancestors = new Set<object>()
): SemanticJsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${label} contains a non-finite number`);
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) {
    if (ancestors.has(value)) throw new Error(`${label} contains a cycle`);
    ancestors.add(value);
    const normalized = value.map((entry, index) =>
      normalizeJson(entry, `${label}[${index}]`, ancestors)
    );
    ancestors.delete(value);
    return normalized;
  }
  if (typeof value === 'object') {
    const object = value as Record<string, unknown>;
    const prototype = Object.getPrototypeOf(object);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(`${label} must contain JSON objects only`);
    }
    if (ancestors.has(object)) throw new Error(`${label} contains a cycle`);
    ancestors.add(object);
    const normalized: Record<string, SemanticJsonValue> = {};
    for (const key of Object.keys(object).sort(compareText)) {
      normalized[key] = normalizeJson(object[key], `${label}.${key}`, ancestors);
    }
    ancestors.delete(object);
    return normalized;
  }
  throw new Error(`${label} contains a non-JSON value`);
}

function canonicalJson(value: unknown, label: string): string {
  return JSON.stringify(normalizeJson(value, label));
}

function parseCanonicalJson(value: unknown, label: string): SemanticJsonValue {
  if (typeof value !== 'string') throw new Error(`${label} is not stored as text`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
  if (canonicalJson(parsed, label) !== value) throw new Error(`${label} is not canonical JSON`);
  return parsed as SemanticJsonValue;
}

function digest(kind: string, canonicalPayload: string): string {
  return createHash('sha256')
    .update(kind)
    .update('\0')
    .update(canonicalPayload)
    .digest('hex');
}

function validateDigest(value: string, label: string): string {
  if (!DIGEST_RE.test(value)) throw new Error(`${label} must be a lowercase SHA-256 digest`);
  return value;
}

function validateKind(value: string, label: string): string {
  if (!KIND_RE.test(value)) throw new Error(`${label} must match ${KIND_RE.source}`);
  return value;
}

function validateKey(value: string, label: string): string {
  if (!KEY_RE.test(value)) throw new Error(`${label} must match ${KEY_RE.source}`);
  return value;
}

function validateRef(value: string, label: string): string {
  if (!REF_RE.test(value)) throw new Error(`${label} must match ${REF_RE.source}`);
  return value;
}

function validateLabel(value: string, label: string): string {
  if (!LABEL_RE.test(value)) throw new Error(`${label} must match ${LABEL_RE.source}`);
  return value;
}

function validateOperationId(value: string): string {
  if (!OPERATION_ID_RE.test(value) || Buffer.byteLength(value, 'utf8') > 256) {
    throw new Error('semantic ledger operationId must be 1 to 256 non-whitespace bytes');
  }
  return value;
}

function normalizeInstant(value: Date | string | undefined): string {
  const date = value === undefined ? new Date() : value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) throw new Error('semantic ledger timestamp must be valid');
  return date.toISOString();
}

function optionalText(value: string | undefined, label: string, maximum = 4_096): string | undefined {
  if (value === undefined) return undefined;
  const trimmed = value.trim();
  if (trimmed.length === 0 || Buffer.byteLength(trimmed, 'utf8') > maximum) {
    throw new Error(`${label} must contain 1 to ${maximum} UTF-8 bytes`);
  }
  return trimmed;
}

function uniqueValues(values: string[], label: string): string[] {
  const unique = new Set(values);
  if (unique.size !== values.length) throw new Error(`${label} contains duplicates`);
  return values;
}

function metricsJson(metrics: SemanticMetrics | undefined): string {
  const normalized: SemanticMetrics = {};
  for (const key of Object.keys(metrics ?? {}).sort(compareText)) {
    validateKey(key, `metric '${key}'`);
    const value = metrics?.[key];
    if (value !== null && (typeof value !== 'number' || !Number.isFinite(value))) {
      throw new Error(`metric '${key}' must be a finite number or null`);
    }
    normalized[key] = value ?? null;
  }
  return canonicalJson(normalized, 'semantic evidence metrics');
}

function parsedMetrics(value: unknown): SemanticMetrics {
  const parsed = parseCanonicalJson(value, 'semantic evidence metrics');
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error('semantic evidence metrics must be an object');
  }
  const output: SemanticMetrics = {};
  for (const [key, metric] of Object.entries(parsed)) {
    if (metric !== null && typeof metric !== 'number') {
      throw new Error(`stored semantic evidence metric '${key}' is invalid`);
    }
    output[key] = metric;
  }
  return output;
}

function recordValue(row: unknown, label: string): Record<string, unknown> {
  if (typeof row !== 'object' || row === null) throw new Error(`${label} is missing`);
  return row as Record<string, unknown>;
}

export class SemanticLedger {
  private readonly tables: LedgerTables;
  private readonly prefix: string;

  constructor(
    private readonly database: DatabaseSync,
    options: SemanticLedgerOptions = {}
  ) {
    this.prefix = options.tablePrefix ?? 'remembero_semantic';
    if (!TABLE_PREFIX_RE.test(this.prefix)) {
      throw new Error(`semantic ledger tablePrefix must match ${TABLE_PREFIX_RE.source}`);
    }
    this.tables = tableNames(this.prefix);
    this.ensureSchema();
  }

  putObject(input: PutSemanticObjectInput): SemanticObject {
    const kind = validateKind(input.kind, 'semantic object kind');
    const payload = canonicalJson(input.value, 'semantic object value');
    const bytes = Buffer.byteLength(payload, 'utf8');
    if (bytes > MAX_SEMANTIC_OBJECT_BYTES) {
      throw new Error(`semantic object exceeds ${MAX_SEMANTIC_OBJECT_BYTES} bytes`);
    }
    const objectDigest = digest(`object:${kind}`, payload);
    const createdAt = normalizeInstant(input.createdAt);
    this.database
      .prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.objects)}
         (digest, kind, canonical_json, byte_length, created_at)
         VALUES (?, ?, ?, ?, ?)`
      )
      .run(objectDigest, kind, payload, bytes, createdAt);
    return this.getObject(objectDigest);
  }

  getObject(objectDigest: string): SemanticObject {
    validateDigest(objectDigest, 'semantic object digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT digest, kind, canonical_json, byte_length, created_at
           FROM ${quoted(this.tables.objects)} WHERE digest = ?`
        )
        .get(objectDigest),
      `semantic object '${objectDigest}'`
    );
    if (typeof row.kind !== 'string' || typeof row.canonical_json !== 'string') {
      throw new Error(`semantic object '${objectDigest}' is corrupt`);
    }
    const value = parseCanonicalJson(row.canonical_json, `semantic object '${objectDigest}'`);
    const expected = digest(`object:${row.kind}`, row.canonical_json);
    const bytes = Buffer.byteLength(row.canonical_json, 'utf8');
    if (
      expected !== objectDigest ||
      row.byte_length !== bytes ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic object '${objectDigest}' failed integrity validation`);
    }
    return {
      digest: objectDigest,
      kind: row.kind,
      value,
      bytes,
      createdAt: row.created_at,
    };
  }

  createVersion(input: CreateSemanticVersionInput): SemanticVersion {
    const parents = uniqueValues(
      (input.parents ?? []).map((value, index) =>
        validateDigest(value, `semantic version parent ${index + 1}`)
      ),
      'semantic version parents'
    );
    if (parents.length > MAX_SEMANTIC_VERSION_PARENTS) {
      throw new Error(`semantic version exceeds ${MAX_SEMANTIC_VERSION_PARENTS} parents`);
    }
    const members = [...input.members]
      .map((member, index) => ({
        key: validateKey(member.key, `semantic member ${index + 1} key`),
        objectDigest: validateDigest(
          member.objectDigest,
          `semantic member '${member.key}' objectDigest`
        ),
      }))
      .sort((left, right) => compareText(left.key, right.key));
    uniqueValues(members.map((member) => member.key), 'semantic version member keys');
    if (members.length === 0 || members.length > MAX_SEMANTIC_VERSION_MEMBERS) {
      throw new Error(
        `semantic version members must contain 1 to ${MAX_SEMANTIC_VERSION_MEMBERS} entries`
      );
    }
    const memberKeys = new Set(members.map((member) => member.key));
    const edges = (input.edges ?? [])
      .map((edge, index) => {
        const kind = validateKind(edge.kind, `semantic edge ${index + 1} kind`);
        const from = validateKey(edge.from, `semantic edge ${index + 1} from`);
        const to = validateKey(edge.to, `semantic edge ${index + 1} to`);
        if (!memberKeys.has(from) || !memberKeys.has(to)) {
          throw new Error(`semantic edge '${from}' -> '${to}' must reference version members`);
        }
        const metadataJson = canonicalJson(edge.metadata ?? {}, `semantic edge ${index + 1} metadata`);
        const edgeDigest = digest(
          'edge',
          canonicalJson({ kind, from, to, metadata: JSON.parse(metadataJson) }, 'semantic edge')
        );
        return {
          digest: edgeDigest,
          kind,
          from,
          to,
          metadata: JSON.parse(metadataJson) as SemanticJsonValue,
        };
      })
      .sort((left, right) => compareText(left.digest, right.digest));
    uniqueValues(edges.map((edge) => edge.digest), 'semantic version edges');
    if (edges.length > MAX_SEMANTIC_VERSION_EDGES) {
      throw new Error(`semantic version exceeds ${MAX_SEMANTIC_VERSION_EDGES} edges`);
    }
    const contracts = (input.contracts ?? [])
      .map((contract, index) => {
        const key = validateKey(contract.key, `semantic contract ${index + 1} key`);
        const consumer = validateKey(contract.consumer, `semantic contract '${key}' consumer`);
        const provider = validateKey(contract.provider, `semantic contract '${key}' provider`);
        if (!memberKeys.has(consumer) || !memberKeys.has(provider)) {
          throw new Error(`semantic contract '${key}' must reference version members`);
        }
        return {
          key,
          kind: validateKind(contract.kind, `semantic contract '${key}' kind`),
          objectDigest: validateDigest(
            contract.objectDigest,
            `semantic contract '${key}' objectDigest`
          ),
          consumer,
          provider,
        };
      })
      .sort((left, right) => compareText(left.key, right.key));
    uniqueValues(contracts.map((contract) => contract.key), 'semantic version contract keys');
    if (contracts.length > MAX_SEMANTIC_VERSION_CONTRACTS) {
      throw new Error(`semantic version exceeds ${MAX_SEMANTIC_VERSION_CONTRACTS} contracts`);
    }
    const metadata = normalizeJson(input.metadata ?? {}, 'semantic version metadata');
    const manifest = {
      format: 'remembero-semantic-version',
      version: 1,
      parents,
      members,
      edges,
      contracts,
      metadata,
    };
    const manifestJson = canonicalJson(manifest, 'semantic version manifest');
    const versionDigest = digest('version', manifestJson);
    const createdAt = normalizeInstant(input.createdAt);

    return this.withSavepoint('create_version', () => {
      for (const parent of parents) this.requireVersion(parent);
      for (const member of members) this.getObject(member.objectDigest);
      for (const contract of contracts) this.getObject(contract.objectDigest);
      this.database
        .prepare(
          `INSERT OR IGNORE INTO ${quoted(this.tables.versions)}
           (digest, manifest_json, metadata_json, created_at) VALUES (?, ?, ?, ?)`
        )
        .run(versionDigest, manifestJson, canonicalJson(metadata, 'semantic version metadata'), createdAt);
      const insertParent = this.database.prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.parents)}
         (version_digest, position, parent_digest) VALUES (?, ?, ?)`
      );
      parents.forEach((parent, index) => insertParent.run(versionDigest, index, parent));
      const insertMember = this.database.prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.members)}
         (version_digest, member_key, object_digest) VALUES (?, ?, ?)`
      );
      members.forEach((member) =>
        insertMember.run(versionDigest, member.key, member.objectDigest)
      );
      const insertEdge = this.database.prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.edges)}
         (version_digest, edge_digest, kind, from_key, to_key, metadata_json)
         VALUES (?, ?, ?, ?, ?, ?)`
      );
      edges.forEach((edge) =>
        insertEdge.run(
          versionDigest,
          edge.digest,
          edge.kind,
          edge.from,
          edge.to,
          canonicalJson(edge.metadata, 'semantic edge metadata')
        )
      );
      const insertContract = this.database.prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.contracts)}
         (version_digest, contract_key, kind, object_digest, consumer_key, provider_key)
         VALUES (?, ?, ?, ?, ?, ?)`
      );
      contracts.forEach((contract) =>
        insertContract.run(
          versionDigest,
          contract.key,
          contract.kind,
          contract.objectDigest,
          contract.consumer,
          contract.provider
        )
      );
      if (input.label !== undefined) {
        this.labelVersion(input.label, versionDigest, input.createdAt);
      }
      return this.getVersion(versionDigest);
    });
  }

  getVersion(versionDigest: string): SemanticVersion {
    validateDigest(versionDigest, 'semantic version digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT manifest_json, metadata_json, created_at
           FROM ${quoted(this.tables.versions)} WHERE digest = ?`
        )
        .get(versionDigest),
      `semantic version '${versionDigest}'`
    );
    const parents = (
      this.database
        .prepare(
          `SELECT parent_digest FROM ${quoted(this.tables.parents)}
           WHERE version_digest = ? ORDER BY position`
        )
        .all(versionDigest) as Array<{ parent_digest: string }>
    ).map((entry) => entry.parent_digest);
    const members = (
      this.database
        .prepare(
          `SELECT m.member_key, m.object_digest, o.kind AS object_kind
           FROM ${quoted(this.tables.members)} m
           JOIN ${quoted(this.tables.objects)} o ON o.digest = m.object_digest
           WHERE m.version_digest = ? ORDER BY m.member_key`
        )
        .all(versionDigest) as Array<{
          member_key: string;
          object_digest: string;
          object_kind: string;
        }>
    ).map((entry) => ({
      key: entry.member_key,
      objectDigest: entry.object_digest,
      objectKind: entry.object_kind,
    }));
    const edges = (
      this.database
        .prepare(
          `SELECT edge_digest, kind, from_key, to_key, metadata_json
           FROM ${quoted(this.tables.edges)} WHERE version_digest = ? ORDER BY edge_digest`
        )
        .all(versionDigest) as Array<{
          edge_digest: string;
          kind: string;
          from_key: string;
          to_key: string;
          metadata_json: string;
        }>
    ).map((entry) => ({
      digest: entry.edge_digest,
      kind: entry.kind,
      from: entry.from_key,
      to: entry.to_key,
      metadata: parseCanonicalJson(entry.metadata_json, `semantic edge '${entry.edge_digest}'`),
    }));
    const contracts = (
      this.database
        .prepare(
          `SELECT c.contract_key, c.kind, c.object_digest, c.consumer_key, c.provider_key,
                  o.kind AS object_kind
           FROM ${quoted(this.tables.contracts)} c
           JOIN ${quoted(this.tables.objects)} o ON o.digest = c.object_digest
           WHERE c.version_digest = ? ORDER BY c.contract_key`
        )
        .all(versionDigest) as Array<{
          contract_key: string;
          kind: string;
          object_digest: string;
          consumer_key: string;
          provider_key: string;
          object_kind: string;
        }>
    ).map((entry) => ({
      key: entry.contract_key,
      kind: entry.kind,
      objectDigest: entry.object_digest,
      consumer: entry.consumer_key,
      provider: entry.provider_key,
      objectKind: entry.object_kind,
    }));
    const metadata = parseCanonicalJson(row.metadata_json, `semantic version '${versionDigest}' metadata`);
    const normalizedManifest = canonicalJson(
      {
        format: 'remembero-semantic-version',
        version: 1,
        parents,
        members: members.map(({ key, objectDigest }) => ({ key, objectDigest })),
        edges,
        contracts: contracts.map(({ objectKind: _objectKind, ...contract }) => contract),
        metadata,
      },
      `semantic version '${versionDigest}' manifest`
    );
    if (
      row.manifest_json !== normalizedManifest ||
      digest('version', normalizedManifest) !== versionDigest ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic version '${versionDigest}' failed integrity validation`);
    }
    const labels = (
      this.database
        .prepare(
          `SELECT label FROM ${quoted(this.tables.labels)}
           WHERE version_digest = ? ORDER BY label`
        )
        .all(versionDigest) as Array<{ label: string }>
    ).map((entry) => entry.label);
    return {
      digest: versionDigest,
      parents,
      members,
      edges,
      contracts,
      metadata,
      labels,
      createdAt: row.created_at,
    };
  }

  labelVersion(label: string, versionDigest: string, createdAt?: Date | string): void {
    validateLabel(label, 'semantic version label');
    this.requireVersion(versionDigest);
    const existing = this.database
      .prepare(
        `SELECT version_digest FROM ${quoted(this.tables.labels)} WHERE label = ?`
      )
      .get(label) as { version_digest?: string } | undefined;
    if (existing?.version_digest !== undefined && existing.version_digest !== versionDigest) {
      throw new Error(`semantic version label '${label}' already identifies another version`);
    }
    this.database
      .prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.labels)}
         (label, version_digest, created_at) VALUES (?, ?, ?)`
      )
      .run(label, versionDigest, normalizeInstant(createdAt));
  }

  resolveVersion(reference: string): SemanticVersion {
    if (DIGEST_RE.test(reference)) return this.getVersion(reference);
    if (!REF_RE.test(reference) && !LABEL_RE.test(reference)) {
      throw new Error('semantic version reference is invalid');
    }
    const row = this.database
      .prepare(
        `SELECT version_digest FROM ${quoted(this.tables.refs)} WHERE name = ?
         UNION ALL
         SELECT version_digest FROM ${quoted(this.tables.labels)} WHERE label = ?
         LIMIT 1`
      )
      .get(reference, reference) as { version_digest?: string } | undefined;
    if (typeof row?.version_digest !== 'string') {
      throw new Error(`unknown semantic version reference '${reference}'`);
    }
    return this.getVersion(row.version_digest);
  }

  recordEvidence(input: RecordSemanticEvidenceInput): SemanticEvidence {
    const versionDigest = this.requireVersion(input.versionDigest).digest;
    const baselineVersionDigest = input.baselineVersionDigest === undefined
      ? undefined
      : this.requireVersion(input.baselineVersionDigest).digest;
    const kind = validateKind(input.kind, 'semantic evidence kind');
    const evaluator = input.evaluator === undefined
      ? undefined
      : validateKind(input.evaluator, 'semantic evidence evaluator');
    const statusValues: SemanticEvidenceStatus[] = [
      'passed',
      'failed',
      'blocked',
      'error',
      'observed',
    ];
    if (!statusValues.includes(input.status)) throw new Error('semantic evidence status is invalid');
    const payloadJson = canonicalJson(input.payload ?? {}, 'semantic evidence payload');
    const normalizedMetrics = metricsJson(input.metrics);
    const evidenceManifest = canonicalJson(
      {
        versionDigest,
        baselineVersionDigest: baselineVersionDigest ?? null,
        kind,
        status: input.status,
        evaluator: evaluator ?? null,
        payload: JSON.parse(payloadJson),
        metrics: JSON.parse(normalizedMetrics),
      },
      'semantic evidence manifest'
    );
    const evidenceDigest = digest('evidence', evidenceManifest);
    this.database
      .prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.evidence)}
         (digest, version_digest, baseline_version_digest, kind, status, evaluator,
          payload_json, metrics_json, manifest_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        evidenceDigest,
        versionDigest,
        baselineVersionDigest ?? null,
        kind,
        input.status,
        evaluator ?? null,
        payloadJson,
        normalizedMetrics,
        evidenceManifest,
        normalizeInstant(input.createdAt)
      );
    return this.getEvidence(evidenceDigest);
  }

  getEvidence(evidenceDigest: string): SemanticEvidence {
    validateDigest(evidenceDigest, 'semantic evidence digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT sequence, version_digest, baseline_version_digest, kind, status, evaluator,
                  payload_json, metrics_json, manifest_json, created_at
           FROM ${quoted(this.tables.evidence)} WHERE digest = ?`
        )
        .get(evidenceDigest),
      `semantic evidence '${evidenceDigest}'`
    );
    const payload = parseCanonicalJson(row.payload_json, `semantic evidence '${evidenceDigest}' payload`);
    const metrics = parsedMetrics(row.metrics_json);
    const manifest = canonicalJson(
      {
        versionDigest: row.version_digest,
        baselineVersionDigest: row.baseline_version_digest ?? null,
        kind: row.kind,
        status: row.status,
        evaluator: row.evaluator ?? null,
        payload,
        metrics,
      },
      `semantic evidence '${evidenceDigest}' manifest`
    );
    if (
      row.manifest_json !== manifest ||
      digest('evidence', manifest) !== evidenceDigest ||
      typeof row.sequence !== 'number' ||
      typeof row.version_digest !== 'string' ||
      typeof row.kind !== 'string' ||
      typeof row.status !== 'string' ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic evidence '${evidenceDigest}' failed integrity validation`);
    }
    return {
      sequence: row.sequence,
      digest: evidenceDigest,
      versionDigest: row.version_digest,
      ...(typeof row.baseline_version_digest === 'string'
        ? { baselineVersionDigest: row.baseline_version_digest }
        : {}),
      kind: row.kind,
      status: row.status as SemanticEvidenceStatus,
      ...(typeof row.evaluator === 'string' ? { evaluator: row.evaluator } : {}),
      payload,
      metrics,
      createdAt: row.created_at,
    };
  }

  evidenceForVersion(versionDigest: string): SemanticEvidence[] {
    this.requireVersion(versionDigest);
    const rows = this.database
      .prepare(
        `SELECT digest FROM ${quoted(this.tables.evidence)}
         WHERE version_digest = ? ORDER BY sequence`
      )
      .all(versionDigest) as Array<{ digest: string }>;
    return rows.map((row) => this.getEvidence(row.digest));
  }

  recordCompatibility(
    input: RecordSemanticCompatibilityInput
  ): SemanticCompatibilityAssessment {
    const candidateVersionDigest = this.requireVersion(input.candidateVersionDigest).digest;
    const baselineVersionDigest = input.baselineVersionDigest === undefined
      ? undefined
      : this.requireVersion(input.baselineVersionDigest).digest;
    if (input.checks.length === 0 || input.checks.length > 256) {
      throw new Error('semantic compatibility requires 1 to 256 checks');
    }
    const allowedStatuses: SemanticCompatibilityStatus[] = [
      'pass',
      'fail',
      'review',
      'blocked',
      'not_applicable',
    ];
    const checks = input.checks
      .map((check, index): SemanticCompatibilityCheck => {
        const dimension = validateKind(check.dimension, `compatibility check ${index + 1} dimension`);
        if (!allowedStatuses.includes(check.status)) {
          throw new Error(`compatibility check '${dimension}' has an invalid status`);
        }
        const evidenceDigests = uniqueValues(
          (check.evidenceDigests ?? [])
            .map((value) => validateDigest(value, `compatibility check '${dimension}' evidence`))
            .sort(compareText),
          `compatibility check '${dimension}' evidence`
        );
        for (const evidenceDigest of evidenceDigests) {
          const evidence = this.getEvidence(evidenceDigest);
          if (
            evidence.versionDigest !== candidateVersionDigest &&
            evidence.versionDigest !== baselineVersionDigest
          ) {
            throw new Error(
              `compatibility check '${dimension}' evidence belongs to another version`
            );
          }
        }
        return {
          dimension,
          status: check.status,
          summary: optionalText(check.summary, `compatibility check '${dimension}' summary`)! ,
          evidenceDigests,
          details: normalizeJson(
            check.details ?? {},
            `compatibility check '${dimension}' details`
          ),
        };
      })
      .sort((left, right) => compareText(left.dimension, right.dimension));
    uniqueValues(checks.map((check) => check.dimension), 'semantic compatibility dimensions');
    const metadata = normalizeJson(input.metadata ?? {}, 'semantic compatibility metadata');
    const manifest = canonicalJson(
      {
        baselineVersionDigest: baselineVersionDigest ?? null,
        candidateVersionDigest,
        checks,
        metadata,
      },
      'semantic compatibility manifest'
    );
    const assessmentDigest = digest('compatibility', manifest);
    return this.withSavepoint('compatibility', () => {
      this.database
        .prepare(
          `INSERT OR IGNORE INTO ${quoted(this.tables.assessments)}
           (digest, baseline_version_digest, candidate_version_digest, metadata_json,
            manifest_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)`
        )
        .run(
          assessmentDigest,
          baselineVersionDigest ?? null,
          candidateVersionDigest,
          canonicalJson(metadata, 'semantic compatibility metadata'),
          manifest,
          normalizeInstant(input.createdAt)
        );
      const insert = this.database.prepare(
        `INSERT OR IGNORE INTO ${quoted(this.tables.checks)}
         (assessment_digest, dimension, status, summary, evidence_json, details_json)
         VALUES (?, ?, ?, ?, ?, ?)`
      );
      checks.forEach((check) =>
        insert.run(
          assessmentDigest,
          check.dimension,
          check.status,
          check.summary,
          canonicalJson(check.evidenceDigests, 'compatibility evidence digests'),
          canonicalJson(check.details, 'compatibility details')
        )
      );
      return this.getCompatibility(assessmentDigest);
    });
  }

  getCompatibility(assessmentDigest: string): SemanticCompatibilityAssessment {
    validateDigest(assessmentDigest, 'semantic compatibility digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT sequence, baseline_version_digest, candidate_version_digest,
                  metadata_json, manifest_json, created_at
           FROM ${quoted(this.tables.assessments)} WHERE digest = ?`
        )
        .get(assessmentDigest),
      `semantic compatibility '${assessmentDigest}'`
    );
    const checks = (
      this.database
        .prepare(
          `SELECT dimension, status, summary, evidence_json, details_json
           FROM ${quoted(this.tables.checks)}
           WHERE assessment_digest = ? ORDER BY dimension`
        )
        .all(assessmentDigest) as Array<{
          dimension: string;
          status: SemanticCompatibilityStatus;
          summary: string;
          evidence_json: string;
          details_json: string;
        }>
    ).map((check): SemanticCompatibilityCheck => ({
      dimension: check.dimension,
      status: check.status,
      summary: check.summary,
      evidenceDigests: parseCanonicalJson(
        check.evidence_json,
        `compatibility check '${check.dimension}' evidence`
      ) as string[],
      details: parseCanonicalJson(
        check.details_json,
        `compatibility check '${check.dimension}' details`
      ),
    }));
    const metadata = parseCanonicalJson(row.metadata_json, 'semantic compatibility metadata');
    const manifest = canonicalJson(
      {
        baselineVersionDigest: row.baseline_version_digest ?? null,
        candidateVersionDigest: row.candidate_version_digest,
        checks,
        metadata,
      },
      'semantic compatibility manifest'
    );
    if (
      row.manifest_json !== manifest ||
      digest('compatibility', manifest) !== assessmentDigest ||
      typeof row.sequence !== 'number' ||
      typeof row.candidate_version_digest !== 'string' ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic compatibility '${assessmentDigest}' failed integrity validation`);
    }
    return {
      sequence: row.sequence,
      digest: assessmentDigest,
      ...(typeof row.baseline_version_digest === 'string'
        ? { baselineVersionDigest: row.baseline_version_digest }
        : {}),
      candidateVersionDigest: row.candidate_version_digest,
      checks,
      metadata,
      createdAt: row.created_at,
    };
  }

  setRef(input: SetSemanticRefInput): SemanticRefEvent {
    return this.withSavepoint('set_ref', () =>
      this.setRefInternal({ ...input, promotionDigest: undefined })
    );
  }

  getRef(name: string): SemanticRef | undefined {
    validateRef(name, 'semantic ref name');
    const row = this.database
      .prepare(
        `SELECT version_digest, updated_at FROM ${quoted(this.tables.refs)} WHERE name = ?`
      )
      .get(name) as { version_digest?: string; updated_at?: string } | undefined;
    if (row === undefined) return undefined;
    if (typeof row.version_digest !== 'string' || typeof row.updated_at !== 'string') {
      throw new Error(`semantic ref '${name}' is corrupt`);
    }
    return { name, versionDigest: row.version_digest, updatedAt: row.updated_at };
  }

  refHistory(name: string): SemanticRefEvent[] {
    validateRef(name, 'semantic ref name');
    const rows = this.database
      .prepare(
        `SELECT event_digest FROM ${quoted(this.tables.refEvents)}
         WHERE ref_name = ? ORDER BY sequence`
      )
      .all(name) as Array<{ event_digest: string }>;
    return rows.map((row) => this.getRefEvent(row.event_digest));
  }

  promote(input: PromoteSemanticVersionInput): SemanticPromotionDecision {
    return this.withSavepoint('promote', () => {
      validateOperationId(input.operationId);
      const prior = this.database
        .prepare(
          `SELECT digest FROM ${quoted(this.tables.promotions)} WHERE operation_id = ?`
        )
        .get(input.operationId) as { digest?: string } | undefined;
      if (typeof prior?.digest === 'string') {
        const decision = this.getPromotion(prior.digest);
        if (
          decision.ref !== input.ref ||
          decision.candidateVersionDigest !== input.candidateVersionDigest ||
          decision.assessmentDigest !== input.assessmentDigest
        ) {
          throw new Error(`semantic promotion operation '${input.operationId}' was reused`);
        }
        return decision;
      }
      const ref = validateRef(input.ref, 'semantic promotion ref');
      const candidate = this.requireVersion(input.candidateVersionDigest).digest;
      const assessment = this.getCompatibility(input.assessmentDigest);
      if (assessment.candidateVersionDigest !== candidate) {
        throw new Error('semantic promotion assessment targets another candidate version');
      }
      const current = this.getRef(ref)?.versionDigest;
      if (
        input.expectedCurrentVersionDigest !== undefined &&
        (input.expectedCurrentVersionDigest ?? undefined) !== current
      ) {
        throw new Error(`semantic ref '${ref}' no longer matches the expected version`);
      }
      if (assessment.baselineVersionDigest !== current) {
        throw new Error('semantic promotion assessment baseline does not match the current ref');
      }
      const acceptedReviewDimensions = uniqueValues(
        (input.acceptedReviewDimensions ?? [])
          .map((dimension) => validateKind(dimension, 'accepted review dimension'))
          .sort(compareText),
        'accepted review dimensions'
      );
      const acceptedReviews = new Set(acceptedReviewDimensions);
      const blockingDimensions = assessment.checks
        .filter(
          (check) =>
            check.status === 'fail' ||
            check.status === 'blocked' ||
            (check.status === 'review' && !acceptedReviews.has(check.dimension))
        )
        .map((check) => check.dimension);
      for (const dimension of acceptedReviewDimensions) {
        if (!assessment.checks.some((check) => check.dimension === dimension && check.status === 'review')) {
          throw new Error(`accepted review dimension '${dimension}' is not under review`);
        }
      }
      const outcome = blockingDimensions.length === 0 ? 'accepted' : 'rejected';
      const reason = optionalText(input.reason, 'semantic promotion reason');
      const decisionManifest = canonicalJson(
        {
          ref,
          fromVersionDigest: current ?? null,
          candidateVersionDigest: candidate,
          assessmentDigest: assessment.digest,
          outcome,
          acceptedReviewDimensions,
          blockingDimensions,
          reason: reason ?? null,
        },
        'semantic promotion decision'
      );
      const decisionDigest = digest('promotion', decisionManifest);
      const createdAt = normalizeInstant(input.createdAt);
      this.database
        .prepare(
          `INSERT INTO ${quoted(this.tables.promotions)}
           (digest, operation_id, ref_name, from_version_digest, candidate_version_digest,
            assessment_digest, outcome, accepted_reviews_json, blocking_dimensions_json,
            reason, manifest_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          decisionDigest,
          input.operationId,
          ref,
          current ?? null,
          candidate,
          assessment.digest,
          outcome,
          canonicalJson(acceptedReviewDimensions, 'accepted review dimensions'),
          canonicalJson(blockingDimensions, 'blocking dimensions'),
          reason ?? null,
          decisionManifest,
          createdAt
        );
      if (outcome === 'accepted') {
        this.setRefInternal({
          name: ref,
          versionDigest: candidate,
          operationId: `promotion:${input.operationId}`,
          expectedCurrentVersionDigest: current ?? null,
          reason,
          createdAt,
          promotionDigest: decisionDigest,
        });
      }
      return this.getPromotion(decisionDigest);
    });
  }

  getPromotion(decisionDigest: string): SemanticPromotionDecision {
    validateDigest(decisionDigest, 'semantic promotion digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT sequence, operation_id, ref_name, from_version_digest,
                  candidate_version_digest, assessment_digest, outcome,
                  accepted_reviews_json, blocking_dimensions_json, reason,
                  manifest_json, created_at
           FROM ${quoted(this.tables.promotions)} WHERE digest = ?`
        )
        .get(decisionDigest),
      `semantic promotion '${decisionDigest}'`
    );
    const acceptedReviewDimensions = parseCanonicalJson(
      row.accepted_reviews_json,
      'semantic promotion accepted reviews'
    ) as string[];
    const blockingDimensions = parseCanonicalJson(
      row.blocking_dimensions_json,
      'semantic promotion blocking dimensions'
    ) as string[];
    const manifest = canonicalJson(
      {
        ref: row.ref_name,
        fromVersionDigest: row.from_version_digest ?? null,
        candidateVersionDigest: row.candidate_version_digest,
        assessmentDigest: row.assessment_digest,
        outcome: row.outcome,
        acceptedReviewDimensions,
        blockingDimensions,
        reason: row.reason ?? null,
      },
      'semantic promotion decision'
    );
    if (
      row.manifest_json !== manifest ||
      digest('promotion', manifest) !== decisionDigest ||
      typeof row.sequence !== 'number' ||
      typeof row.operation_id !== 'string' ||
      typeof row.ref_name !== 'string' ||
      typeof row.candidate_version_digest !== 'string' ||
      typeof row.assessment_digest !== 'string' ||
      (row.outcome !== 'accepted' && row.outcome !== 'rejected') ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic promotion '${decisionDigest}' failed integrity validation`);
    }
    return {
      sequence: row.sequence,
      digest: decisionDigest,
      operationId: row.operation_id,
      ref: row.ref_name,
      ...(typeof row.from_version_digest === 'string'
        ? { fromVersionDigest: row.from_version_digest }
        : {}),
      candidateVersionDigest: row.candidate_version_digest,
      assessmentDigest: row.assessment_digest,
      outcome: row.outcome,
      acceptedReviewDimensions,
      blockingDimensions,
      ...(typeof row.reason === 'string' ? { reason: row.reason } : {}),
      createdAt: row.created_at,
    };
  }

  diffVersions(fromDigest: string, toDigest: string): SemanticVersionDiff {
    const from = this.getVersion(fromDigest);
    const to = this.getVersion(toDigest);
    const memberDiff = this.keyedDiff(from.members, to.members, (entry) => entry.key);
    const contractDiff = this.keyedDiff(from.contracts, to.contracts, (entry) => entry.key);
    const fromEdges = new Map(from.edges.map((edge) => [edge.digest, edge]));
    const toEdges = new Map(to.edges.map((edge) => [edge.digest, edge]));
    const evidence = this.compareEvidence(from.digest, to.digest);
    const compatibility = this.latestCompatibility(from.digest, to.digest);
    const edges = {
      added: to.edges.filter((edge) => !fromEdges.has(edge.digest)),
      removed: from.edges.filter((edge) => !toEdges.has(edge.digest)),
    };
    const changed =
      memberDiff.added.length > 0 ||
      memberDiff.removed.length > 0 ||
      memberDiff.changed.length > 0 ||
      edges.added.length > 0 ||
      edges.removed.length > 0 ||
      contractDiff.added.length > 0 ||
      contractDiff.removed.length > 0 ||
      contractDiff.changed.length > 0 ||
      evidence.some(
        (entry) =>
          entry.before?.digest !== entry.after?.digest ||
          Object.values(entry.metricDelta).some((value) => value !== 0 && value !== null)
      );
    return {
      changed,
      from,
      to,
      members: memberDiff,
      edges,
      contracts: contractDiff,
      evidence,
      ...(compatibility === undefined ? {} : { compatibility }),
    };
  }

  traverseGraph(
    versionDigest: string,
    focus: string,
    options: TraverseSemanticGraphOptions = {}
  ): SemanticGraphTraversal {
    const version = this.getVersion(versionDigest);
    const focusKey = validateKey(focus, 'semantic graph focus');
    const members = new Map(version.members.map((member) => [member.key, member]));
    if (!members.has(focusKey)) throw new Error(`semantic graph focus '${focusKey}' is not a member`);
    const direction = options.direction ?? 'both';
    if (!['upstream', 'downstream', 'both'].includes(direction)) {
      throw new Error('semantic graph direction is invalid');
    }
    const maxDepth = options.maxDepth ?? 4;
    if (!Number.isSafeInteger(maxDepth) || maxDepth < 0 || maxDepth > MAX_SEMANTIC_GRAPH_DEPTH) {
      throw new Error(`semantic graph maxDepth must be 0 to ${MAX_SEMANTIC_GRAPH_DEPTH}`);
    }
    const depths = new Map<string, number>([[focusKey, 0]]);
    const pending = [focusKey];
    while (pending.length > 0) {
      const current = pending.shift()!;
      const depth = depths.get(current)!;
      if (depth >= maxDepth) continue;
      for (const edge of version.edges) {
        const neighbors: string[] = [];
        if ((direction === 'downstream' || direction === 'both') && edge.from === current) {
          neighbors.push(edge.to);
        }
        if ((direction === 'upstream' || direction === 'both') && edge.to === current) {
          neighbors.push(edge.from);
        }
        for (const neighbor of neighbors) {
          if (depths.has(neighbor)) continue;
          if (depths.size >= MAX_SEMANTIC_GRAPH_NODES) {
            throw new Error(`semantic graph exceeds ${MAX_SEMANTIC_GRAPH_NODES} nodes`);
          }
          depths.set(neighbor, depth + 1);
          pending.push(neighbor);
        }
      }
    }
    const selected = new Set(depths.keys());
    const nodes = [...depths]
      .map(([key, depth]) => ({ ...members.get(key)!, depth }))
      .sort((left, right) => left.depth - right.depth || compareText(left.key, right.key));
    const edges = version.edges.filter(
      (edge) => selected.has(edge.from) && selected.has(edge.to)
    );
    return { versionDigest, focus: focusKey, direction, maxDepth, nodes, edges };
  }

  private compareEvidence(fromDigest: string, toDigest: string): SemanticEvidenceComparison[] {
    const latest = (versionDigest: string): Map<string, SemanticEvidence> => {
      const entries = new Map<string, SemanticEvidence>();
      for (const evidence of this.evidenceForVersion(versionDigest)) {
        entries.set(`${evidence.kind}\0${evidence.evaluator ?? ''}`, evidence);
      }
      return entries;
    };
    const before = latest(fromDigest);
    const after = latest(toDigest);
    const keys = [...new Set([...before.keys(), ...after.keys()])].sort(compareText);
    return keys.map((key) => {
      const left = before.get(key);
      const right = after.get(key);
      const metricKeys = [...new Set([
        ...Object.keys(left?.metrics ?? {}),
        ...Object.keys(right?.metrics ?? {}),
      ])].sort(compareText);
      const metricDelta: SemanticMetrics = {};
      for (const metric of metricKeys) {
        const leftValue = left?.metrics[metric];
        const rightValue = right?.metrics[metric];
        metricDelta[metric] =
          typeof leftValue === 'number' && typeof rightValue === 'number'
            ? rightValue - leftValue
            : null;
      }
      return {
        key: key.replace('\0', ':'),
        ...(left === undefined ? {} : { before: left }),
        ...(right === undefined ? {} : { after: right }),
        metricDelta,
      };
    });
  }

  private latestCompatibility(
    baselineVersionDigest: string,
    candidateVersionDigest: string
  ): SemanticCompatibilityAssessment | undefined {
    const row = this.database
      .prepare(
        `SELECT digest FROM ${quoted(this.tables.assessments)}
         WHERE baseline_version_digest = ? AND candidate_version_digest = ?
         ORDER BY sequence DESC LIMIT 1`
      )
      .get(baselineVersionDigest, candidateVersionDigest) as { digest?: string } | undefined;
    return typeof row?.digest === 'string' ? this.getCompatibility(row.digest) : undefined;
  }

  private keyedDiff<T>(
    before: T[],
    after: T[],
    keyOf: (value: T) => string
  ): { added: T[]; removed: T[]; changed: SemanticVersionDiffEntry<T>[] } {
    const left = new Map(before.map((entry) => [keyOf(entry), entry]));
    const right = new Map(after.map((entry) => [keyOf(entry), entry]));
    const added = after.filter((entry) => !left.has(keyOf(entry)));
    const removed = before.filter((entry) => !right.has(keyOf(entry)));
    const changed = [...left.keys()]
      .filter(
        (key) =>
          right.has(key) &&
          canonicalJson(left.get(key), `semantic diff '${key}' before`) !==
            canonicalJson(right.get(key), `semantic diff '${key}' after`)
      )
      .sort(compareText)
      .map((key) => ({ key, before: left.get(key), after: right.get(key) }));
    return { added, removed, changed };
  }

  private setRefInternal(
    input: SetSemanticRefInput & { promotionDigest?: string }
  ): SemanticRefEvent {
    const name = validateRef(input.name, 'semantic ref name');
    const versionDigest = this.requireVersion(input.versionDigest).digest;
    const operationId = validateOperationId(input.operationId);
    const existingEvent = this.database
      .prepare(
        `SELECT event_digest FROM ${quoted(this.tables.refEvents)} WHERE operation_id = ?`
      )
      .get(operationId) as { event_digest?: string } | undefined;
    if (typeof existingEvent?.event_digest === 'string') {
      const event = this.getRefEvent(existingEvent.event_digest);
      if (event.name !== name || event.toVersionDigest !== versionDigest) {
        throw new Error(`semantic ref operation '${operationId}' was reused`);
      }
      return event;
    }
    const current = this.getRef(name)?.versionDigest;
    if (
      input.expectedCurrentVersionDigest !== undefined &&
      (input.expectedCurrentVersionDigest ?? undefined) !== current
    ) {
      throw new Error(`semantic ref '${name}' no longer matches the expected version`);
    }
    const reason = optionalText(input.reason, 'semantic ref reason');
    const createdAt = normalizeInstant(input.createdAt);
    const eventManifest = canonicalJson(
      {
        operationId,
        name,
        fromVersionDigest: current ?? null,
        toVersionDigest: versionDigest,
        promotionDigest: input.promotionDigest ?? null,
        reason: reason ?? null,
      },
      'semantic ref event'
    );
    const eventDigest = digest('ref-event', eventManifest);
    if (current === undefined) {
      this.database
        .prepare(
          `INSERT INTO ${quoted(this.tables.refs)}
           (name, version_digest, updated_at) VALUES (?, ?, ?)`
        )
        .run(name, versionDigest, createdAt);
    } else {
      const result = this.database
        .prepare(
          `UPDATE ${quoted(this.tables.refs)} SET version_digest = ?, updated_at = ?
           WHERE name = ? AND version_digest = ?`
        )
        .run(versionDigest, createdAt, name, current);
      if (result.changes !== 1) throw new Error(`semantic ref '${name}' changed concurrently`);
    }
    this.database
      .prepare(
        `INSERT INTO ${quoted(this.tables.refEvents)}
         (event_digest, operation_id, ref_name, from_version_digest, to_version_digest,
          promotion_digest, reason, manifest_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        eventDigest,
        operationId,
        name,
        current ?? null,
        versionDigest,
        input.promotionDigest ?? null,
        reason ?? null,
        eventManifest,
        createdAt
      );
    return this.getRefEvent(eventDigest);
  }

  private getRefEvent(eventDigest: string): SemanticRefEvent {
    validateDigest(eventDigest, 'semantic ref event digest');
    const row = recordValue(
      this.database
        .prepare(
          `SELECT sequence, operation_id, ref_name, from_version_digest, to_version_digest,
                  promotion_digest, reason, manifest_json, created_at
           FROM ${quoted(this.tables.refEvents)} WHERE event_digest = ?`
        )
        .get(eventDigest),
      `semantic ref event '${eventDigest}'`
    );
    const manifest = canonicalJson(
      {
        operationId: row.operation_id,
        name: row.ref_name,
        fromVersionDigest: row.from_version_digest ?? null,
        toVersionDigest: row.to_version_digest,
        promotionDigest: row.promotion_digest ?? null,
        reason: row.reason ?? null,
      },
      'semantic ref event'
    );
    if (
      row.manifest_json !== manifest ||
      digest('ref-event', manifest) !== eventDigest ||
      typeof row.sequence !== 'number' ||
      typeof row.operation_id !== 'string' ||
      typeof row.ref_name !== 'string' ||
      typeof row.to_version_digest !== 'string' ||
      typeof row.created_at !== 'string'
    ) {
      throw new Error(`semantic ref event '${eventDigest}' failed integrity validation`);
    }
    return {
      sequence: row.sequence,
      digest: eventDigest,
      operationId: row.operation_id,
      name: row.ref_name,
      ...(typeof row.from_version_digest === 'string'
        ? { fromVersionDigest: row.from_version_digest }
        : {}),
      toVersionDigest: row.to_version_digest,
      ...(typeof row.promotion_digest === 'string'
        ? { promotionDigest: row.promotion_digest }
        : {}),
      ...(typeof row.reason === 'string' ? { reason: row.reason } : {}),
      createdAt: row.created_at,
    };
  }

  private requireVersion(versionDigest: string): SemanticVersion {
    return this.getVersion(validateDigest(versionDigest, 'semantic version digest'));
  }

  private withSavepoint<T>(label: string, operation: () => T): T {
    const savepoint = `${this.prefix}_${label}_${randomUUID().replaceAll('-', '_')}`;
    this.database.exec(`SAVEPOINT ${savepoint}`);
    try {
      const result = operation();
      this.database.exec(`RELEASE SAVEPOINT ${savepoint}`);
      return result;
    } catch (error) {
      const failures: string[] = [];
      try {
        this.database.exec(`ROLLBACK TO SAVEPOINT ${savepoint}`);
      } catch (rollbackError) {
        failures.push(`rollback failed: ${String(rollbackError)}`);
      }
      try {
        this.database.exec(`RELEASE SAVEPOINT ${savepoint}`);
      } catch (releaseError) {
        failures.push(`release failed: ${String(releaseError)}`);
      }
      if (failures.length > 0 && error instanceof Error) {
        error.message = `${error.message}; ${failures.join('; ')}`;
      }
      throw error;
    }
  }

  private ensureSchema(): void {
    const t = Object.fromEntries(
      Object.entries(this.tables).map(([key, value]) => [key, quoted(value)])
    ) as Record<keyof LedgerTables, string>;
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS ${t.meta}(
        key TEXT PRIMARY KEY,
        int_value INTEGER,
        text_value TEXT
      );
      CREATE TABLE IF NOT EXISTS ${t.objects}(
        digest TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        canonical_json TEXT NOT NULL,
        byte_length INTEGER NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.versions}(
        digest TEXT PRIMARY KEY,
        manifest_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.parents}(
        version_digest TEXT NOT NULL,
        position INTEGER NOT NULL,
        parent_digest TEXT NOT NULL,
        PRIMARY KEY(version_digest, position),
        UNIQUE(version_digest, parent_digest)
      );
      CREATE TABLE IF NOT EXISTS ${t.members}(
        version_digest TEXT NOT NULL,
        member_key TEXT NOT NULL,
        object_digest TEXT NOT NULL,
        PRIMARY KEY(version_digest, member_key)
      );
      CREATE TABLE IF NOT EXISTS ${t.edges}(
        version_digest TEXT NOT NULL,
        edge_digest TEXT NOT NULL,
        kind TEXT NOT NULL,
        from_key TEXT NOT NULL,
        to_key TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        PRIMARY KEY(version_digest, edge_digest)
      );
      CREATE TABLE IF NOT EXISTS ${t.contracts}(
        version_digest TEXT NOT NULL,
        contract_key TEXT NOT NULL,
        kind TEXT NOT NULL,
        object_digest TEXT NOT NULL,
        consumer_key TEXT NOT NULL,
        provider_key TEXT NOT NULL,
        PRIMARY KEY(version_digest, contract_key)
      );
      CREATE TABLE IF NOT EXISTS ${t.labels}(
        label TEXT PRIMARY KEY,
        version_digest TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.evidence}(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        digest TEXT NOT NULL UNIQUE,
        version_digest TEXT NOT NULL,
        baseline_version_digest TEXT,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        evaluator TEXT,
        payload_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.assessments}(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        digest TEXT NOT NULL UNIQUE,
        baseline_version_digest TEXT,
        candidate_version_digest TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.checks}(
        assessment_digest TEXT NOT NULL,
        dimension TEXT NOT NULL,
        status TEXT NOT NULL,
        summary TEXT NOT NULL,
        evidence_json TEXT NOT NULL,
        details_json TEXT NOT NULL,
        PRIMARY KEY(assessment_digest, dimension)
      );
      CREATE TABLE IF NOT EXISTS ${t.promotions}(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        digest TEXT NOT NULL UNIQUE,
        operation_id TEXT NOT NULL UNIQUE,
        ref_name TEXT NOT NULL,
        from_version_digest TEXT,
        candidate_version_digest TEXT NOT NULL,
        assessment_digest TEXT NOT NULL,
        outcome TEXT NOT NULL,
        accepted_reviews_json TEXT NOT NULL,
        blocking_dimensions_json TEXT NOT NULL,
        reason TEXT,
        manifest_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.refs}(
        name TEXT PRIMARY KEY,
        version_digest TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS ${t.refEvents}(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_digest TEXT NOT NULL UNIQUE,
        operation_id TEXT NOT NULL UNIQUE,
        ref_name TEXT NOT NULL,
        from_version_digest TEXT,
        to_version_digest TEXT NOT NULL,
        promotion_digest TEXT,
        reason TEXT,
        manifest_json TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      INSERT OR IGNORE INTO ${t.meta}(key, int_value, text_value)
      VALUES ('schema_version', ${SEMANTIC_LEDGER_SCHEMA_VERSION}, NULL);
    `);
    const row = this.database
      .prepare(`SELECT int_value FROM ${t.meta} WHERE key = 'schema_version'`)
      .get() as { int_value?: number } | undefined;
    if (row?.int_value !== SEMANTIC_LEDGER_SCHEMA_VERSION) {
      throw new Error(`unsupported semantic ledger schema version ${String(row?.int_value)}`);
    }
    const immutableTables = [
      this.tables.objects,
      this.tables.versions,
      this.tables.parents,
      this.tables.members,
      this.tables.edges,
      this.tables.contracts,
      this.tables.labels,
      this.tables.evidence,
      this.tables.assessments,
      this.tables.checks,
      this.tables.promotions,
      this.tables.refEvents,
    ];
    for (const table of immutableTables) {
      const updateTrigger = `${table}_immutable_update`;
      const deleteTrigger = `${table}_immutable_delete`;
      this.database.exec(`
        CREATE TRIGGER IF NOT EXISTS ${quoted(updateTrigger)}
        BEFORE UPDATE ON ${quoted(table)}
        BEGIN SELECT RAISE(ABORT, 'semantic ledger records are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ${quoted(deleteTrigger)}
        BEFORE DELETE ON ${quoted(table)}
        BEGIN SELECT RAISE(ABORT, 'semantic ledger records are immutable'); END;
      `);
    }
  }
}

export function createSemanticLedger(
  database: DatabaseSync,
  options: SemanticLedgerOptions = {}
): SemanticLedger {
  return new SemanticLedger(database, options);
}
