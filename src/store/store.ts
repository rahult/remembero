import {
  appendFileSync,
  closeSync,
  existsSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';
import { redactSensitiveText } from '../safety.js';
import {
  type Clause,
  type Goal,
  type Literal,
  ParseError,
  canonicalKey,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  literalMatches,
  parseProgram,
  parseQuery,
  serializeClause,
  serializeGoal,
} from '../engine/index.js';
import {
  enforceIntegrityCandidate,
  type IntegrityEnforcementOptions,
} from '../knowledge/enforcement.js';
import {
  enforceKnowledgeCheckCandidate,
  type KnowledgeCheckEnforcementOptions,
} from '../knowledge/check-enforcement.js';
import {
  isEntityMetadataPredicate,
  type EntityRewrite,
} from '../knowledge/identity.js';
import {
  decodeTentativeDeclaration,
  assertTrustMetadataSafety,
  isTrustMetadataPredicate,
  TrustMetadataError,
  wrapTentativeFacts,
  type TentativeResolutionAction,
} from '../knowledge/trust.js';

const NAMESPACE_RE = /^[a-z0-9_-]+$/;
const HEADER = '% rembero memory — one Datalog clause per line; edit by hand if you like.\n';
const MAX_JOURNAL_BYTES = 16 * 1024 * 1024;
const MAX_CAPTURE_ERROR_BYTES = 1024 * 1024;
const MAX_JOURNAL_ENTRIES = 100_000;
export const MAX_JOURNAL_SEGMENTS = 64;
export const MAX_TOTAL_JOURNAL_ENTRIES = 1_000_000;
export const MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024;
export const MAX_RECORDED_SNAPSHOT_BATCH = 64;
const JOURNAL_SEGMENT_RE = /^journal-(\d{12})-(\d{12})-([a-f0-9]{64})\.jsonl$/;
const JOURNAL_CHECKPOINT_RE = /^checkpoint-(\d{12})-([a-f0-9]{64})\.json$/;
const MAX_PENDING_MUTATION_BYTES = 256 * 1024;
export const MAX_HISTORY_EVENTS = 1_000;
export const MAX_OPERATION_ID_BYTES = 256;
const MAX_HISTORY_SOURCE_BYTES = 4_096;
const LOCK_WAIT_MS = 2_000;
const LOCK_STALE_MS = 30_000;
const LOCK_RETRY_MS = 10;
const MAX_REVIEW_DAYS = 3_650;
const RECORDED_AUDIT_OPERATIONS = new Set([
  'remember',
  'auto_capture',
  'auto_capture_pruned',
]);
const sleepCell = new Int32Array(new SharedArrayBuffer(4));

export interface AssertResult {
  added: Clause[];
  duplicates: number;
  opId: string;
}

export interface MutationContext {
  opId?: string;
  sourceText?: string;
  origin?: 'manual' | 'claude-stop';
  captureId?: string;
  at?: Date;
  /** Optional atomic reject-on-write policy for this mutation. */
  integrity?: IntegrityEnforcementOptions;
  /** Optional atomic portable knowledge regression and coverage guard. */
  checks?: KnowledgeCheckEnforcementOptions;
}

export type IdempotentMutationOperation =
  | 'assert'
  | 'retract'
  | 'supersede'
  | 'resolve_tentative'
  | 'rule_change'
  | 'memory_change'
  | 'checkpoint';

export class OperationConflictError extends Error {
  readonly code = 'operation_conflict';

  constructor(
    readonly operation: IdempotentMutationOperation,
    readonly namespace: string,
    readonly opId: string
  ) {
    super(`${operation} operation '${opId}' was already used for another mutation`);
    this.name = 'OperationConflictError';
  }

  toJSON(): Record<string, string> {
    return {
      error: this.code,
      message: this.message,
      operation: this.operation,
      namespace: this.namespace,
      opId: this.opId,
    };
  }
}

export type ValidTimeMode = 'delete' | 'archive_until';
export const MAX_SUPERSEDE_PATTERNS = 64;

export interface SupersedeResult {
  added: Clause[];
  duplicates: number;
  retracted: number;
  archived: Clause[];
  opId: string;
}

export const MAX_RULE_CHANGE_RULES = 64;
export const MAX_MEMORY_CHANGE_CLAUSES = 128;

export interface RuleChangeCandidateView {
  namespaces: string[];
  clauses: Clause[];
  clausesByNamespace: Map<string, Clause[]>;
}

export interface RuleChangeMutationRequest {
  namespaces: string[];
  expectedBaselineDigest: string;
  proposalDigest: string;
  add: string | Clause[];
  remove: string | Clause[];
  validateCandidate?: (candidate: RuleChangeCandidateView) => void;
}

export interface RuleChangeMutationResult {
  added: Clause[];
  removed: Clause[];
  namespace: string;
  namespaces: string[];
  baselineDigest: string;
  proposalDigest: string;
  opId: string;
  sequence: number;
}

export interface MemoryChangeTemporalArchive {
  from: string;
  to: string;
  validUntil: string;
}

export interface MemoryChangeMutationRequest {
  namespaces: string[];
  expectedBaselineDigest: string;
  proposalDigest: string;
  add: string | Clause[];
  remove: string | Clause[];
  temporalArchives?: MemoryChangeTemporalArchive[];
  validateCandidate?: (candidate: RuleChangeCandidateView) => void;
}

export interface MemoryChangeMutationResult {
  added: Clause[];
  removed: Clause[];
  archived: Clause[];
  namespace: string;
  namespaces: string[];
  baselineDigest: string;
  proposalDigest: string;
  opId: string;
  sequence: number;
}

export class RuleChangeStaleError extends Error {
  readonly code = 'rule_change_stale';

  constructor(
    readonly expectedDigest: string,
    readonly actualDigest: string
  ) {
    super('rule change proposal baseline no longer matches current knowledge');
    this.name = 'RuleChangeStaleError';
  }

  toJSON(): Record<string, string> {
    return {
      error: this.code,
      message: this.message,
      expectedDigest: this.expectedDigest,
      actualDigest: this.actualDigest,
    };
  }
}

export class MemoryChangeStaleError extends Error {
  readonly code = 'memory_change_stale';

  constructor(
    readonly expectedDigest: string,
    readonly actualDigest: string
  ) {
    super('memory change proposal baseline no longer matches current knowledge');
    this.name = 'MemoryChangeStaleError';
  }

  toJSON(): Record<string, string> {
    return {
      error: this.code,
      message: this.message,
      expectedDigest: this.expectedDigest,
      actualDigest: this.actualDigest,
    };
  }
}

export type MemoryHistoryAction = 'asserted' | 'retracted' | 'superseded';

export interface MemoryHistoryEvent {
  /** One-based journal line. Append order, not timestamp order, is authoritative. */
  sequence: number;
  /** Stable position within a multi-fact journal operation. */
  position: number;
  namespace: string;
  ts: string;
  opId: string;
  action: MemoryHistoryAction;
  clause: string;
  current: boolean;
  sourceText?: string;
  sourceRedacted?: boolean;
  sourceTruncated?: boolean;
  origin?: 'manual' | 'claude-stop';
  previousSourceOpId?: string;
  archivedAs?: string;
  validUntil?: string;
  trustAction?: TentativeResolutionAction;
}

export interface MemoryHistory {
  pattern: string;
  namespaces: string[];
  events: MemoryHistoryEvent[];
}

export interface MemoryHistoryOptions {
  namespaces?: string[] | '*';
  limit?: number;
}

export interface RecordedSnapshotMetadata {
  /** Global journal position. Zero means before the first journal entry. */
  sequence: number;
  /** Number of entries in the journal when the snapshot was read. */
  journalEntries: number;
  namespaces: string[];
}

export interface RecordedKnowledgeSnapshot extends RecordedSnapshotMetadata {
  clauses: Clause[];
  sources: Map<string, MemorySource[]>;
}

export interface CurrentKnowledgeSnapshot {
  namespaces: string[];
  clauses: Clause[];
  clausesByNamespace: Map<string, Clause[]>;
  sources: Map<string, MemorySource[]>;
}

export interface JournalCheckpointNamespace {
  namespace: string;
  clauses: string[];
  sources: Array<{ key: string; values: MemorySource[] }>;
}

export interface JournalCheckpointArtifact {
  version: 1;
  sequence: number;
  createdAt: string;
  opId: string;
  atProvided: boolean;
  segment: {
    file: string;
    startSequence: number;
    endSequence: number;
    entries: number;
    bytes: number;
    sha256: string;
  };
  namespaces: JournalCheckpointNamespace[];
  stateDigest: string;
}

export interface JournalCompactionOptions {
  opId?: string;
  at?: Date;
  dryRun?: boolean;
}

export interface JournalCompactionResult {
  rotated: boolean;
  sequence: number;
  activeEntries: number;
  segmentCount: number;
  checkpoint?: JournalCheckpointArtifact;
}

export class IncompleteHistoryError extends Error {
  readonly code = 'incomplete_recorded_history';

  constructor(readonly namespaces: string[]) {
    super(
      `recorded history does not reconcile with current knowledge in: ${namespaces.join(', ')}`
    );
    this.name = 'IncompleteHistoryError';
  }

  toJSON(): Record<string, unknown> {
    return {
      error: this.code,
      message: this.message,
      namespaces: this.namespaces,
    };
  }
}

export interface TemporalMemorySource {
  kind: 'superseded';
  previousClause: string;
  validUntil: string;
}

export interface MemorySource {
  namespace: string;
  opId: string;
  ts: string;
  /** Present only on read-only counterfactual assumptions; never written to the journal. */
  hypothetical?: true;
  text?: string;
  redacted?: boolean;
  temporal?: TemporalMemorySource;
  /** Present when an opt-in identity or trust view projected a stored clause. */
  projectedFrom?: string;
  identityRewrites?: EntityRewrite[];
  trust?: 'tentative';
  trustAction?: TentativeResolutionAction;
}

export type AutoCaptureStatus = 'started' | 'captured' | 'empty' | 'failed' | 'skipped';

export interface AutoCaptureReservationRequest {
  captureId?: string;
  fingerprint: string;
  sessionId: string;
  tailBytes: number;
  dailyCap: number;
  at?: Date;
}

export interface AutoCaptureReservation {
  captureId: string;
  reserved: boolean;
  reason?: 'duplicate' | 'daily_cap';
}

export interface AutoCaptureBatch {
  captureId: string;
  namespace: string;
  ts: string;
  status: AutoCaptureStatus;
  sessionId?: string;
  reason?: string;
  added?: number;
  duplicates?: number;
}

export interface AutoCaptureFact {
  id: string;
  captureId: string;
  opId: string;
  namespace: string;
  ts: string;
  clause: string;
  current: boolean;
}

export interface AutoCaptureReview {
  captures: AutoCaptureBatch[];
  facts: AutoCaptureFact[];
}

export interface AutoCaptureReviewOptions {
  days?: number;
  namespace?: string;
  now?: Date;
}

export interface PruneAutoCaptureOptions {
  now?: Date;
  integrity?: IntegrityEnforcementOptions;
  checks?: KnowledgeCheckEnforcementOptions;
}

interface CachedNamespace {
  clauses: Clause[];
  keys: Set<string>;
  /** mtime+size of the file this cache was read from; '' when the file did not exist. */
  fileStamp: string;
}

interface JournalEntry {
  ts: string;
  op: string;
  namespace: string;
  [key: string]: unknown;
}

interface JournalFileContents {
  text: string;
  bytes: number;
  sha256: string;
  entries: JournalEntry[];
}

interface JournalSegmentDescriptor extends JournalFileContents {
  file: string;
  path: string;
  startSequence: number;
  endSequence: number;
}

interface PendingMutation {
  version: 1;
  namespace: string;
  hadPrevious: boolean;
  journalEntry: JournalEntry;
}

function fileStamp(path: string): string {
  try {
    const stat = statSync(path);
    return `${stat.mtimeMs}:${stat.size}`;
  } catch {
    return '';
  }
}

function sanitizeJournalDetails(details: Record<string, unknown>): Record<string, unknown> {
  const sanitized = { ...details };
  let redacted = false;
  for (const key of ['text', 'sourceText']) {
    const value = sanitized[key];
    if (typeof value === 'string') {
      const result = redactSensitiveText(value);
      sanitized[key] = result.text;
      redacted ||= result.redacted;
    }
  }
  return redacted ? { ...sanitized, sourceRedacted: true } : sanitized;
}

function validDate(value: Date, label: string): Date {
  if (!(value instanceof Date) || Number.isNaN(value.getTime())) {
    throw new Error(`${label} must be a valid Date`);
  }
  return value;
}

export function knowledgeProgramDigest(
  namespaces: readonly string[],
  clausesByNamespace: ReadonlyMap<string, readonly Clause[]>
): string {
  return createHash('sha256')
    .update(
      JSON.stringify(
        namespaces.map((namespace) => [
          namespace,
          (clausesByNamespace.get(namespace) ?? []).map(canonicalKey),
        ])
      )
    )
    .digest('hex');
}

function ruleChangeRules(
  value: string | Clause[],
  label: string
): Clause[] {
  const clauses = typeof value === 'string' ? parseProgram(value) : value;
  if (clauses.length > MAX_RULE_CHANGE_RULES) {
    throw new Error(`${label} exceeds ${MAX_RULE_CHANGE_RULES} rules`);
  }
  const keys = new Set<string>();
  for (const clause of clauses) {
    if (clause.body.length === 0 || isIntegrityConstraint(clause)) {
      throw new Error(`${label} must contain ordinary or aggregate rules only`);
    }
    if (
      isEntityMetadataPredicate(clause.head.predicate) ||
      isTrustMetadataPredicate(clause.head.predicate)
    ) {
      throw new Error(`${label} may not define reserved metadata`);
    }
    const key = canonicalKey(clause);
    if (keys.has(key)) throw new Error(`${label} contains duplicate rules`);
    keys.add(key);
  }
  return clauses;
}

function reviewedMemoryClauses(
  value: string | Clause[],
  label: string
): Clause[] {
  const clauses = typeof value === 'string' ? parseProgram(value) : value;
  if (clauses.length > MAX_MEMORY_CHANGE_CLAUSES) {
    throw new Error(`${label} exceeds ${MAX_MEMORY_CHANGE_CLAUSES} clauses`);
  }
  const keys = new Set<string>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) {
      throw new Error(`${label} may not change integrity policy`);
    }
    if (
      isEntityMetadataPredicate(clause.head.predicate) ||
      isTrustMetadataPredicate(clause.head.predicate)
    ) {
      throw new Error(`${label} may not change reserved metadata`);
    }
    if (
      clause.body.length === 0 &&
      clause.head.args.some(
        (term) =>
          (term.type !== 'atom' && term.type !== 'num') ||
          (term.type === 'num' && !Number.isFinite(term.value))
      )
    ) {
      throw new Error(`${label} facts must be finite and ground`);
    }
    const key = canonicalKey(clause);
    if (keys.has(key)) throw new Error(`${label} contains duplicate clauses`);
    keys.add(key);
  }
  return clauses;
}

function validateSha256(value: string, label: string): string {
  if (!/^[a-f0-9]{64}$/.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest`);
  }
  return value;
}

function validateOperationId(opId: string): string {
  if (opId.length === 0) throw new Error('operation id must not be empty');
  if (Buffer.byteLength(opId, 'utf8') > MAX_OPERATION_ID_BYTES) {
    throw new Error(`operation id exceeds ${MAX_OPERATION_ID_BYTES} bytes`);
  }
  return opId;
}

function assertIsoTimestamp(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string') throw new Error(`${label} must be an ISO timestamp`);
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString() !== value) {
    throw new Error(`${label} must be a canonical ISO timestamp`);
  }
}

function parseFactPattern(pattern: string, label: string): Literal {
  let goals: Goal[];
  try {
    goals = parseQuery(pattern);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ParseError(`${label}: ${message}`);
  }
  if (goals.length !== 1 || isComparison(goals[0]) || isNegation(goals[0])) {
    throw new ParseError(`${label} must be a single literal, e.g. works_at(rahul, _)`);
  }
  return goals[0] as Literal;
}

type RetractionTarget =
  | { serialized: string; identity: string; literal: Literal }
  | { serialized: string; identity: string; clauseKey: string };

function parseRetractionTarget(pattern: string, label: string): RetractionTarget {
  let literalError: unknown;
  try {
    const literal = parseFactPattern(pattern, label);
    return {
      serialized: serializeGoal(literal),
      identity: canonicalKey({ head: literal, body: [] }),
      literal,
    };
  } catch (error) {
    literalError = error;
  }
  let clauses: Clause[];
  try {
    clauses = parseProgram(pattern);
  } catch {
    throw literalError;
  }
  if (clauses.length !== 1 || clauses[0].body.length === 0) {
    throw literalError;
  }
  return {
    serialized: serializeClause(clauses[0]),
    identity: canonicalKey(clauses[0]),
    clauseKey: canonicalKey(clauses[0]),
  };
}

function parseJournalClause(value: unknown, label: string): Clause {
  if (typeof value !== 'string') throw new Error(`${label} must be a serialized clause`);
  const clauses = parseProgram(value);
  if (clauses.length !== 1) throw new Error(`${label} must contain exactly one clause`);
  return clauses[0];
}

function parseJournalClauseList(value: unknown, label: string): Clause[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value.map((serialized, index) =>
    parseJournalClause(serialized, `${label}[${index}]`)
  );
}

function archiveUntilClause(clause: Clause, validUntil: string): Clause {
  if (clause.body.length !== 0) throw new Error('valid-time supersession accepts ground facts only');
  if (clause.head.predicate.endsWith('_until')) {
    throw new Error(`refusing to archive temporal predicate '${clause.head.predicate}' again`);
  }
  return {
    head: {
      predicate: `${clause.head.predicate}_until`,
      args: [...clause.head.args, { type: 'atom', value: validUntil }],
    },
    body: [],
  };
}

function truncateUtf8(value: string, maxBytes: number): { text: string; truncated: boolean } {
  if (Buffer.byteLength(value, 'utf8') <= maxBytes) return { text: value, truncated: false };
  const suffix = '…';
  const budget = maxBytes - Buffer.byteLength(suffix, 'utf8');
  const bytes = Buffer.from(value, 'utf8');
  let end = Math.max(0, budget);
  while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
  return { text: `${bytes.subarray(0, end).toString('utf8')}${suffix}`, truncated: true };
}

function historySourceFields(entry: JournalEntry): Pick<
  MemoryHistoryEvent,
  'sourceText' | 'sourceRedacted' | 'sourceTruncated' | 'origin'
> {
  const fields: Pick<
    MemoryHistoryEvent,
    'sourceText' | 'sourceRedacted' | 'sourceTruncated' | 'origin'
  > = {};
  if (typeof entry.sourceText === 'string') {
    const redacted = redactSensitiveText(entry.sourceText);
    const bounded = truncateUtf8(redacted.text, MAX_HISTORY_SOURCE_BYTES);
    fields.sourceText = bounded.text;
    if (redacted.redacted || entry.sourceRedacted === true) fields.sourceRedacted = true;
    if (bounded.truncated) fields.sourceTruncated = true;
  }
  if (entry.origin === 'manual' || entry.origin === 'claude-stop') fields.origin = entry.origin;
  return fields;
}

function journalMemorySource(
  entry: JournalEntry,
  temporal?: TemporalMemorySource
): MemorySource {
  if (typeof entry.opId !== 'string') throw new Error('journal source has no opId');
  return {
    namespace: entry.namespace,
    opId: entry.opId,
    ts: entry.ts,
    ...(typeof entry.sourceText === 'string' ? { text: entry.sourceText } : {}),
    ...(entry.sourceRedacted === true ? { redacted: true } : {}),
    ...(temporal === undefined ? {} : { temporal }),
    ...(entry.trustAction === 'accept' || entry.trustAction === 'reject'
      ? { trustAction: entry.trustAction }
      : {}),
  };
}

function memoryChangeTemporalSources(
  entry: JournalEntry,
  label: string
): Map<string, TemporalMemorySource> {
  if (!Array.isArray(entry.temporalArchives)) {
    throw new Error(`${label} temporalArchives must be an array`);
  }
  const removedKeys = new Set(
    parseJournalClauseList(entry.removedClauses, `${label} removedClauses`).map(
      canonicalKey
    )
  );
  const addedKeys = new Set(
    parseJournalClauseList(entry.addedClauses, `${label} addedClauses`).map(
      canonicalKey
    )
  );
  const result = new Map<string, TemporalMemorySource>();
  for (const [index, value] of entry.temporalArchives.entries()) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new Error(`${label} temporalArchives[${index}] must be an object`);
    }
    const record = value as Record<string, unknown>;
    const from = parseJournalClause(
      record.from,
      `${label} temporalArchives[${index}].from`
    );
    const to = parseJournalClause(
      record.to,
      `${label} temporalArchives[${index}].to`
    );
    assertIsoTimestamp(
      record.validUntil,
      `${label} temporalArchives[${index}].validUntil`
    );
    if (
      !removedKeys.has(canonicalKey(from)) ||
      !addedKeys.has(canonicalKey(to)) ||
      canonicalKey(archiveUntilClause(from, record.validUntil as string)) !==
        canonicalKey(to)
    ) {
      throw new Error(`${label} temporalArchives[${index}] is inconsistent`);
    }
    const key = canonicalKey(to);
    if (result.has(key)) throw new Error(`${label} has duplicate temporal archives`);
    result.set(key, {
      kind: 'superseded',
      previousClause: serializeClause(from),
      validUntil: record.validUntil as string,
    });
  }
  return result;
}

function utcDay(value: Date): string {
  return value.toISOString().slice(0, 10);
}

function autoCaptureFactId(captureId: string, namespace: string, clause: string): string {
  const digest = createHash('sha256')
    .update(`${captureId}\0${namespace}\0${clause}`, 'utf8')
    .digest('hex')
    .slice(0, 16);
  return `${captureId}:${digest}`;
}

export function defaultRoot(): string {
  return join(process.env.REMBERO_HOME ?? join(homedir(), '.rembero'), 'memory');
}

export class MemoryStore {
  private cache = new Map<string, CachedNamespace>();
  private heldLocks = new Set<string>();

  constructor(private root: string = defaultRoot()) {
    this.withMutationLock(() =>
      this.withLock('journal', () => this.recoverPendingMutationUnlocked())
    );
  }

  createOperationId(): string {
    return randomUUID();
  }

  semanticEmbeddingCacheRoot(): string {
    return join(this.root, '.semantic-embeddings');
  }

  listJournalCheckpoints(): JournalCheckpointArtifact[] {
    return this.withLock('journal', () => {
      const segments = this.journalSegmentsUnlocked();
      const checkpoints = this.journalCheckpointsUnlocked();
      this.assertCheckpointReplayUnlocked(checkpoints, segments);
      return checkpoints.map((checkpoint) => structuredClone(checkpoint));
    });
  }

  compactJournal(
    options: JournalCompactionOptions = {}
  ): JournalCompactionResult {
    const explicitOpId = options.opId !== undefined;
    const opId = validateOperationId(options.opId ?? this.createOperationId());
    const at = validDate(options.at ?? new Date(), 'checkpoint timestamp');
    return this.withMutationLock(() =>
      this.withLock('journal', () => {
        this.recoverPendingMutationUnlocked();
        let segments = this.journalSegmentsUnlocked();
        let checkpoints = this.journalCheckpointsUnlocked();
        this.assertCheckpointReplayUnlocked(checkpoints, segments);
        const checkpointBySegment = new Map(
          checkpoints.map((checkpoint) => [checkpoint.segment.file, checkpoint])
        );

        for (const segment of segments) {
          if (checkpointBySegment.has(segment.file)) continue;
          const recovered = this.checkpointArtifact(
            segment,
            `recovered-${segment.sha256.slice(0, 32)}`,
            new Date(segment.entries.at(-1)?.ts ?? at),
            false
          );
          if (!options.dryRun) this.writeCheckpointUnlocked(recovered);
          checkpointBySegment.set(segment.file, recovered);
        }
        checkpoints = [...checkpointBySegment.values()].sort(
          (left, right) => left.sequence - right.sequence
        );

        const prior = explicitOpId
          ? checkpoints.find((checkpoint) => checkpoint.opId === opId)
          : undefined;
        if (prior !== undefined) {
          if (
            prior.atProvided !== (options.at !== undefined) ||
            (options.at !== undefined && prior.createdAt !== at.toISOString())
          ) {
            throw new OperationConflictError('checkpoint', '*', opId);
          }
          return {
            rotated: true,
            sequence: prior.sequence,
            activeEntries: 0,
            segmentCount:
              checkpoints.findIndex(
                (checkpoint) => checkpoint.opId === prior.opId
              ) + 1,
            checkpoint: structuredClone(prior),
          };
        }

        const active = this.activeJournalUnlocked();
        const previousEnd = segments.at(-1)?.endSequence ?? 0;
        if (active.entries.length === 0) {
          return {
            rotated: false,
            sequence: previousEnd,
            activeEntries: 0,
            segmentCount: segments.length,
          };
        }
        if (segments.length >= MAX_JOURNAL_SEGMENTS) {
          throw new Error(`journal segments would exceed ${MAX_JOURNAL_SEGMENTS}`);
        }

        const startSequence = previousEnd + 1;
        const endSequence = previousEnd + active.entries.length;
        const file = `journal-${String(startSequence).padStart(12, '0')}-${String(
          endSequence
        ).padStart(12, '0')}-${active.sha256}.jsonl`;
        const segment: JournalSegmentDescriptor = {
          ...active,
          file,
          path: join(this.journalSegmentsPath(), file),
          startSequence,
          endSequence,
        };
        const checkpoint = this.checkpointArtifact(
          segment,
          opId,
          at,
          options.at !== undefined
        );
        const result: JournalCompactionResult = {
          rotated: true,
          sequence: endSequence,
          activeEntries: 0,
          segmentCount: segments.length + 1,
          checkpoint,
        };
        if (options.dryRun) return result;

        this.ensurePrivateDirectory(this.journalSegmentsPath(), 'journal segments');
        if (existsSync(segment.path)) {
          throw new Error(`journal segment '${file}' already exists`);
        }
        renameSync(this.journalPath(), segment.path);
        try {
          this.writeCheckpointUnlocked(checkpoint);
        } catch (error) {
          // The immutable segment remains authoritative and readable. A later
          // compaction repairs its missing checkpoint deterministically.
          throw error;
        }
        segments = [...segments, segment];
        return { ...result, segmentCount: segments.length };
      })
    );
  }

  private withLock<T>(name: string, operation: () => T): T {
    if (this.heldLocks.has(name)) return operation();
    mkdirSync(this.root, { recursive: true, mode: 0o700 });
    const lockPath = join(this.root, `.${name}.lock`);
    const deadline = Date.now() + LOCK_WAIT_MS;
    let descriptor: number | undefined;
    let ownedDevice: number | undefined;
    let ownedInode: number | undefined;
    while (descriptor === undefined) {
      try {
        const acquired = openSync(lockPath, 'wx', 0o600);
        try {
          writeFileSync(
            acquired,
            `${JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() })}\n`,
            'utf8'
          );
          const owned = fstatSync(acquired);
          ownedDevice = owned.dev;
          ownedInode = owned.ino;
          descriptor = acquired;
        } catch (error) {
          closeSync(acquired);
          try {
            unlinkSync(lockPath);
          } catch {
            // Preserve the acquisition failure.
          }
          throw error;
        }
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== 'EEXIST') throw error;
        try {
          const lock = lstatSync(lockPath);
          if (lock.isSymbolicLink()) {
            throw new Error(`refusing symbolic-link lock file ${lockPath}`);
          }
          let ownerAlive = false;
          try {
            const owner = JSON.parse(readFileSync(lockPath, 'utf8')) as { pid?: unknown };
            if (Number.isSafeInteger(owner.pid) && (owner.pid as number) > 0) {
              try {
                process.kill(owner.pid as number, 0);
                ownerAlive = true;
              } catch (ownerError) {
                ownerAlive = (ownerError as NodeJS.ErrnoException).code === 'EPERM';
              }
            }
          } catch {
            // A crashed writer can leave an empty or partial lock; age still gates cleanup.
          }
          if (Date.now() - lock.mtimeMs > LOCK_STALE_MS && !ownerAlive) {
            unlinkSync(lockPath);
            continue;
          }
        } catch (statError) {
          if ((statError as NodeJS.ErrnoException).code === 'ENOENT') continue;
          throw statError;
        }
        if (Date.now() >= deadline) {
          throw new Error(`timed out waiting for memory lock '${name}'`);
        }
        Atomics.wait(sleepCell, 0, 0, LOCK_RETRY_MS);
      }
    }
    try {
      this.heldLocks.add(name);
      return operation();
    } finally {
      this.heldLocks.delete(name);
      closeSync(descriptor);
      try {
        const current = lstatSync(lockPath);
        if (current.dev === ownedDevice && current.ino === ownedInode) {
          unlinkSync(lockPath);
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
    }
  }

  private withNamespaceLock<T>(namespace: string, operation: () => T): T {
    this.filePath(namespace);
    return this.withLock(`namespace-${namespace}`, operation);
  }

  /** Every supported .dl writer participates so an enforcing snapshot cannot race. */
  private withMutationLock<T>(operation: () => T): T {
    return this.withLock('mutation', operation);
  }

  private integrityNamespaces(
    targetNamespace: string,
    options: IntegrityEnforcementOptions
  ): string[] {
    const requested = options.namespaces ?? [targetNamespace];
    const names = requested === '*'
      ? [...new Set([...this.listNamespaces(), targetNamespace])].sort()
      : [...new Set(requested)];
    if (names.length === 0 || names.length > 32) {
      throw new Error('integrity enforcement namespace list must contain 1 to 32 entries');
    }
    for (const namespace of names) this.filePath(namespace);
    if (!names.includes(targetNamespace)) {
      throw new Error(
        `integrity enforcement namespaces must include target '${targetNamespace}'`
      );
    }
    return names;
  }

  private checkEnforcementNamespaces(
    targetNamespace: string,
    options: KnowledgeCheckEnforcementOptions
  ): string[] {
    const requested = options.namespaces ?? [targetNamespace];
    const names = requested === '*'
      ? [...new Set([...this.listNamespaces(), targetNamespace])].sort()
      : [...new Set(requested)];
    if (names.length === 0 || names.length > 32) {
      throw new Error('knowledge check namespace list must contain 1 to 32 entries');
    }
    for (const namespace of names) this.filePath(namespace);
    if (!names.includes(targetNamespace)) {
      throw new Error(
        `knowledge check namespaces must include target '${targetNamespace}'`
      );
    }
    return names;
  }

  private enforceMutation(
    namespace: string,
    currentClauses: Clause[],
    candidateClauses: Clause[],
    addedClauses: Clause[],
    context: MutationContext,
    at: Date,
    temporalByClause: Map<string, TemporalMemorySource> = new Map()
  ): void {
    if (context.integrity === undefined && context.checks === undefined) return;
    const buildView = (names: string[]) => {
      const baselineClauses = names.flatMap((name) =>
        name === namespace ? currentClauses : this.load(name)
      );
      const candidateView = names.flatMap((name) =>
        name === namespace ? candidateClauses : this.load(name)
      );
      const baselineSources = this.sourcesFor(names);
      const candidateKeys = new Set(candidateClauses.map(canonicalKey));
      const candidateSources = new Map<string, MemorySource[]>();
      const namespaceOrder = new Map(names.map((name, index) => [name, index]));
      for (const [key, sources] of baselineSources) {
        const retained = sources.filter(
          (source) => source.namespace !== namespace || candidateKeys.has(key)
        );
        if (retained.length > 0) candidateSources.set(key, retained);
      }
      const sanitizedSource = context.sourceText === undefined
        ? {}
        : sanitizeJournalDetails({ sourceText: context.sourceText });
      for (const clause of addedClauses) {
        const key = canonicalKey(clause);
        const sources = candidateSources.get(key) ?? [];
        sources.push({
          namespace,
          opId: context.opId ?? '',
          ts: at.toISOString(),
          ...(typeof sanitizedSource.sourceText !== 'string'
            ? {}
            : { text: sanitizedSource.sourceText }),
          ...(sanitizedSource.sourceRedacted === true ? { redacted: true } : {}),
          ...(temporalByClause.get(key) === undefined
            ? {}
            : { temporal: temporalByClause.get(key) }),
        });
        sources.sort(
          (left, right) =>
            (namespaceOrder.get(left.namespace) ?? Number.MAX_SAFE_INTEGER) -
              (namespaceOrder.get(right.namespace) ?? Number.MAX_SAFE_INTEGER) ||
            left.opId.localeCompare(right.opId)
        );
        candidateSources.set(key, sources);
      }
      return { baselineClauses, candidateView, baselineSources, candidateSources };
    };
    if (context.integrity !== undefined) {
      const view = buildView(this.integrityNamespaces(namespace, context.integrity));
      enforceIntegrityCandidate(
        view.baselineClauses,
        view.candidateView,
        view.baselineSources,
        view.candidateSources,
        context.integrity
      );
    }
    if (context.checks !== undefined) {
      const view = buildView(
        this.checkEnforcementNamespaces(namespace, context.checks)
      );
      enforceKnowledgeCheckCandidate(
        view.baselineClauses,
        view.candidateView,
        view.baselineSources,
        view.candidateSources,
        context.checks
      );
    }
  }

  private journalPath(): string {
    return join(this.root, 'journal.log');
  }

  private journalSegmentsPath(): string {
    return join(this.root, '.journal-segments');
  }

  private journalCheckpointsPath(): string {
    return join(this.root, '.journal-checkpoints');
  }

  private ensurePrivateDirectory(path: string, label: string): void {
    if (!existsSync(path)) {
      mkdirSync(path, { recursive: true, mode: 0o700 });
      return;
    }
    const stat = lstatSync(path);
    if (stat.isSymbolicLink() || !stat.isDirectory()) {
      throw new Error(`refusing invalid ${label} directory ${path}`);
    }
  }

  private captureErrorPath(): string {
    return join(this.root, 'capture-errors.log');
  }

  private createJournalEntry(
    namespace: string,
    op: string,
    details: Record<string, unknown>,
    at: Date
  ): JournalEntry {
    this.filePath(namespace);
    return {
      ts: validDate(at, 'journal timestamp').toISOString(),
      op,
      namespace,
      ...sanitizeJournalDetails(details),
    };
  }

  private appendJournalUnlocked(entry: JournalEntry): void {
    const line = `${JSON.stringify(entry)}\n`;
    const path = this.journalPath();
    const active = this.activeJournalUnlocked();
    const current = active.text;
    if (active.entries.length >= MAX_JOURNAL_ENTRIES) {
      throw new Error(`journal.log would exceed ${MAX_JOURNAL_ENTRIES} entries`);
    }
    const currentBytes = Buffer.byteLength(current, 'utf8');
    const nextBytes = currentBytes + Buffer.byteLength(line, 'utf8');
    if (nextBytes > MAX_JOURNAL_BYTES) {
      throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
    }
    const tmp = `${path}.tmp-${process.pid}-${randomUUID()}`;
    try {
      writeFileSync(tmp, `${current}${line}`, { encoding: 'utf8', mode: 0o600, flag: 'wx' });
      renameSync(tmp, path);
    } catch (error) {
      this.unlinkIfPresent(tmp);
      throw error;
    }
  }

  private readJournalFileUnlocked(
    path: string,
    label: string
  ): JournalFileContents {
    const stat = lstatSync(path);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`refusing invalid ${label} ${path}`);
    }
    if (stat.size > MAX_JOURNAL_BYTES) {
      throw new Error(`${label} exceeds ${MAX_JOURNAL_BYTES} bytes`);
    }
    const text = readFileSync(path, 'utf8');
    const entries: JournalEntry[] = [];
    for (const [index, line] of text.split('\n').entries()) {
      if (line.trim() === '') continue;
      let entry: unknown;
      try {
        entry = JSON.parse(line);
      } catch {
        throw new Error(`failed to read ${label} line ${index + 1}`);
      }
      if (
        typeof entry !== 'object' ||
        entry === null ||
        Array.isArray(entry) ||
        typeof (entry as Record<string, unknown>).ts !== 'string' ||
        typeof (entry as Record<string, unknown>).op !== 'string' ||
        typeof (entry as Record<string, unknown>).namespace !== 'string'
      ) {
        throw new Error(`failed to read ${label} line ${index + 1}`);
      }
      entries.push(entry as JournalEntry);
      if (entries.length > MAX_JOURNAL_ENTRIES) {
        throw new Error(`${label} exceeds ${MAX_JOURNAL_ENTRIES} entries`);
      }
    }
    return {
      text,
      bytes: Buffer.byteLength(text, 'utf8'),
      sha256: createHash('sha256').update(text).digest('hex'),
      entries,
    };
  }

  private journalSegmentsUnlocked(): JournalSegmentDescriptor[] {
    const directory = this.journalSegmentsPath();
    if (!existsSync(directory)) return [];
    this.ensurePrivateDirectory(directory, 'journal segments');
    const directoryEntries = readdirSync(directory);
    const unexpected = directoryEntries.find(
      (name) => !JOURNAL_SEGMENT_RE.test(name) && !name.includes('.tmp-')
    );
    if (unexpected !== undefined) {
      throw new Error(`unexpected journal segment artifact '${unexpected}'`);
    }
    const names = directoryEntries.filter((name) => JOURNAL_SEGMENT_RE.test(name)).sort();
    if (names.length > MAX_JOURNAL_SEGMENTS) {
      throw new Error(`journal segments exceed ${MAX_JOURNAL_SEGMENTS}`);
    }
    const segments = names.map((file): JournalSegmentDescriptor => {
      const match = file.match(JOURNAL_SEGMENT_RE)!;
      const startSequence = Number(match[1]);
      const endSequence = Number(match[2]);
      const expectedDigest = match[3];
      if (
        !Number.isSafeInteger(startSequence) ||
        !Number.isSafeInteger(endSequence) ||
        startSequence < 1 ||
        endSequence < startSequence
      ) {
        throw new Error(`journal segment '${file}' has an invalid sequence range`);
      }
      const path = join(directory, file);
      const contents = this.readJournalFileUnlocked(path, `journal segment ${file}`);
      if (contents.sha256 !== expectedDigest) {
        throw new Error(`journal segment '${file}' failed SHA-256 validation`);
      }
      if (contents.entries.length !== endSequence - startSequence + 1) {
        throw new Error(`journal segment '${file}' has an inconsistent entry count`);
      }
      return { file, path, startSequence, endSequence, ...contents };
    });
    let expectedStart = 1;
    for (const segment of segments) {
      if (segment.startSequence !== expectedStart) {
        throw new Error(
          `journal segment chain is incomplete at sequence ${expectedStart}`
        );
      }
      expectedStart = segment.endSequence + 1;
    }
    return segments;
  }

  private activeJournalUnlocked(): JournalFileContents {
    const path = this.journalPath();
    try {
      return this.readJournalFileUnlocked(path, 'journal.log');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return {
          text: '',
          bytes: 0,
          sha256: createHash('sha256').update('').digest('hex'),
          entries: [],
        };
      }
      throw error;
    }
  }

  private readJournalUnlocked(): JournalEntry[] {
    const segments = this.journalSegmentsUnlocked();
    const active = this.activeJournalUnlocked();
    const entries = [
      ...segments.flatMap((segment) => segment.entries),
      ...active.entries,
    ];
    if (entries.length > MAX_TOTAL_JOURNAL_ENTRIES) {
      throw new Error(
        `journal history exceeds ${MAX_TOTAL_JOURNAL_ENTRIES} entries`
      );
    }
    return entries;
  }

  private checkpointFileName(sequence: number, digest: string): string {
    return `checkpoint-${String(sequence).padStart(12, '0')}-${digest}.json`;
  }

  private checkpointNamespacesAt(sequence: number): JournalCheckpointNamespace[] {
    return this.listNamespaces().map((namespace) => {
      const snapshot = this.recordedSnapshot([namespace], sequence);
      return {
        namespace,
        clauses: snapshot.clauses.map(serializeClause),
        sources: [...snapshot.sources]
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, values]) => ({
            key,
            values: values.map((source) => ({ ...source })),
          })),
      };
    });
  }

  private assertCheckpointReplayUnlocked(
    checkpoints: JournalCheckpointArtifact[],
    segments: JournalSegmentDescriptor[]
  ): void {
    const byFile = new Map(segments.map((segment) => [segment.file, segment]));
    for (const checkpoint of checkpoints) {
      const segment = byFile.get(checkpoint.segment.file);
      if (
        segment === undefined ||
        segment.sha256 !== checkpoint.segment.sha256 ||
        segment.startSequence !== checkpoint.segment.startSequence ||
        segment.endSequence !== checkpoint.segment.endSequence
      ) {
        throw new Error(
          `journal checkpoint at sequence ${checkpoint.sequence} has no matching segment`
        );
      }
      const replayed = this.checkpointNamespacesAt(checkpoint.sequence);
      const replayDigest = createHash('sha256')
        .update(JSON.stringify(replayed))
        .digest('hex');
      if (
        replayDigest !== checkpoint.stateDigest ||
        JSON.stringify(replayed) !== JSON.stringify(checkpoint.namespaces)
      ) {
        throw new Error(
          `journal checkpoint at sequence ${checkpoint.sequence} does not match replay`
        );
      }
    }
  }

  private checkpointArtifact(
    segment: JournalSegmentDescriptor,
    opId: string,
    at: Date,
    atProvided: boolean
  ): JournalCheckpointArtifact {
    const namespaces = this.checkpointNamespacesAt(segment.endSequence);
    return {
      version: 1,
      sequence: segment.endSequence,
      createdAt: at.toISOString(),
      opId,
      atProvided,
      segment: {
        file: segment.file,
        startSequence: segment.startSequence,
        endSequence: segment.endSequence,
        entries: segment.entries.length,
        bytes: segment.bytes,
        sha256: segment.sha256,
      },
      namespaces,
      stateDigest: createHash('sha256')
        .update(JSON.stringify(namespaces))
        .digest('hex'),
    };
  }

  private readCheckpointUnlocked(path: string): JournalCheckpointArtifact {
    const stat = lstatSync(path);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`refusing invalid journal checkpoint ${path}`);
    }
    if (stat.size > MAX_CHECKPOINT_BYTES) {
      throw new Error(`journal checkpoint exceeds ${MAX_CHECKPOINT_BYTES} bytes`);
    }
    let value: unknown;
    try {
      value = JSON.parse(readFileSync(path, 'utf8'));
    } catch {
      throw new Error(`failed to read journal checkpoint ${path}`);
    }
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new Error(`journal checkpoint ${path} has an invalid shape`);
    }
    const checkpoint = value as JournalCheckpointArtifact;
    if (
      checkpoint.version !== 1 ||
      !Number.isSafeInteger(checkpoint.sequence) ||
      checkpoint.sequence < 1 ||
      typeof checkpoint.createdAt !== 'string' ||
      typeof checkpoint.opId !== 'string' ||
      typeof checkpoint.atProvided !== 'boolean' ||
      typeof checkpoint.segment !== 'object' ||
      checkpoint.segment === null ||
      !Array.isArray(checkpoint.namespaces) ||
      typeof checkpoint.stateDigest !== 'string'
    ) {
      throw new Error(`journal checkpoint ${path} has an invalid shape`);
    }
    assertIsoTimestamp(checkpoint.createdAt, 'journal checkpoint createdAt');
    validateOperationId(checkpoint.opId);
    const segment = checkpoint.segment;
    if (
      typeof segment.file !== 'string' ||
      !Number.isSafeInteger(segment.startSequence) ||
      !Number.isSafeInteger(segment.endSequence) ||
      !Number.isSafeInteger(segment.entries) ||
      !Number.isSafeInteger(segment.bytes) ||
      typeof segment.sha256 !== 'string' ||
      segment.endSequence !== checkpoint.sequence ||
      segment.entries !== segment.endSequence - segment.startSequence + 1 ||
      segment.bytes < 0 ||
      !/^[a-f0-9]{64}$/.test(segment.sha256)
    ) {
      throw new Error(`journal checkpoint ${path} has invalid segment metadata`);
    }
    const fileName = basename(path);
    const fileMatch = fileName.match(JOURNAL_CHECKPOINT_RE);
    if (
      fileMatch === null ||
      Number(fileMatch[1]) !== checkpoint.sequence ||
      fileMatch[2] !== segment.sha256
    ) {
      throw new Error(`journal checkpoint ${path} has inconsistent identity`);
    }
    for (const namespace of checkpoint.namespaces) {
      if (
        typeof namespace !== 'object' ||
        namespace === null ||
        typeof namespace.namespace !== 'string' ||
        !Array.isArray(namespace.clauses) ||
        !namespace.clauses.every((clause) => typeof clause === 'string') ||
        !Array.isArray(namespace.sources)
      ) {
        throw new Error(`journal checkpoint ${path} has invalid namespace state`);
      }
      this.filePath(namespace.namespace);
      for (const clause of namespace.clauses) {
        const parsed = parseProgram(clause);
        if (parsed.length !== 1) {
          throw new Error(`journal checkpoint ${path} has invalid clause state`);
        }
      }
      for (const source of namespace.sources) {
        if (
          typeof source !== 'object' ||
          source === null ||
          typeof source.key !== 'string' ||
          !Array.isArray(source.values)
        ) {
          throw new Error(`journal checkpoint ${path} has invalid source state`);
        }
      }
    }
    const stateDigest = createHash('sha256')
      .update(JSON.stringify(checkpoint.namespaces))
      .digest('hex');
    if (stateDigest !== checkpoint.stateDigest) {
      throw new Error(`journal checkpoint ${path} failed state digest validation`);
    }
    return checkpoint;
  }

  private journalCheckpointsUnlocked(): JournalCheckpointArtifact[] {
    const directory = this.journalCheckpointsPath();
    if (!existsSync(directory)) return [];
    this.ensurePrivateDirectory(directory, 'journal checkpoints');
    const directoryEntries = readdirSync(directory);
    const unexpected = directoryEntries.find(
      (name) => !JOURNAL_CHECKPOINT_RE.test(name) && !name.includes('.tmp-')
    );
    if (unexpected !== undefined) {
      throw new Error(`unexpected journal checkpoint artifact '${unexpected}'`);
    }
    const checkpoints = directoryEntries
      .filter((name) => JOURNAL_CHECKPOINT_RE.test(name))
      .sort()
      .map((name) => this.readCheckpointUnlocked(join(directory, name)));
    let previousSequence = 0;
    for (const checkpoint of checkpoints) {
      if (checkpoint.sequence <= previousSequence) {
        throw new Error('journal checkpoints are not strictly increasing');
      }
      previousSequence = checkpoint.sequence;
    }
    return checkpoints;
  }

  private writeCheckpointUnlocked(checkpoint: JournalCheckpointArtifact): void {
    const directory = this.journalCheckpointsPath();
    this.ensurePrivateDirectory(directory, 'journal checkpoints');
    const file = this.checkpointFileName(
      checkpoint.sequence,
      checkpoint.segment.sha256
    );
    const target = join(directory, file);
    const text = `${JSON.stringify(checkpoint, null, 2)}\n`;
    if (Buffer.byteLength(text, 'utf8') > MAX_CHECKPOINT_BYTES) {
      throw new Error(`journal checkpoint exceeds ${MAX_CHECKPOINT_BYTES} bytes`);
    }
    if (existsSync(target)) {
      const existing = this.readCheckpointUnlocked(target);
      if (JSON.stringify(existing) !== JSON.stringify(checkpoint)) {
        throw new Error(`journal checkpoint '${file}' already exists with other content`);
      }
      return;
    }
    const tmp = `${target}.tmp-${process.pid}-${randomUUID()}`;
    try {
      writeFileSync(tmp, text, { encoding: 'utf8', mode: 0o600, flag: 'wx' });
      renameSync(tmp, target);
    } catch (error) {
      this.unlinkIfPresent(tmp);
      throw error;
    }
  }

  private readCaptureErrorsUnlocked(): JournalEntry[] {
    const path = this.captureErrorPath();
    let text: string;
    try {
      const stat = statSync(path);
      if (stat.size > MAX_CAPTURE_ERROR_BYTES) {
        throw new Error(`capture-errors.log exceeds ${MAX_CAPTURE_ERROR_BYTES} bytes`);
      }
      text = readFileSync(path, 'utf8');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw error;
    }
    const entries: JournalEntry[] = [];
    for (const [index, line] of text.split('\n').entries()) {
      if (line.trim() === '') continue;
      let entry: unknown;
      try {
        entry = JSON.parse(line);
      } catch {
        throw new Error(`failed to read capture-errors.log line ${index + 1}`);
      }
      if (
        typeof entry !== 'object' ||
        entry === null ||
        Array.isArray(entry) ||
        typeof (entry as Record<string, unknown>).ts !== 'string' ||
        (entry as Record<string, unknown>).op !== 'auto_capture' ||
        typeof (entry as Record<string, unknown>).namespace !== 'string'
      ) {
        throw new Error(`failed to read capture-errors.log line ${index + 1}`);
      }
      entries.push(entry as JournalEntry);
    }
    return entries;
  }

  private filePath(namespace: string): string {
    if (!NAMESPACE_RE.test(namespace)) {
      throw new Error(
        `invalid namespace '${namespace}': use lowercase letters, digits, '_' or '-'`
      );
    }
    return join(this.root, `${namespace}.dl`);
  }

  private loadCached(namespace: string): CachedNamespace {
    const path = this.filePath(namespace);
    const stamp = fileStamp(path);
    const cached = this.cache.get(namespace);
    // another process may have written the file since we cached it
    if (cached && cached.fileStamp === stamp) return cached;
    let text: string;
    try {
      text = readFileSync(path, 'utf8');
    } catch {
      const empty = { clauses: [], keys: new Set<string>(), fileStamp: stamp };
      this.cache.set(namespace, empty);
      return empty;
    }
    let clauses: Clause[];
    try {
      clauses = parseProgram(text);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      throw new ParseError(`failed to load ${path}: ${message}`);
    }
    const entry = { clauses, keys: new Set(clauses.map(canonicalKey)), fileStamp: stamp };
    this.cache.set(namespace, entry);
    return entry;
  }

  private namespaceBody(entry: CachedNamespace): string {
    const facts = entry.clauses.filter((c) => c.body.length === 0);
    const rules = entry.clauses.filter((c) => c.body.length > 0);
    const body = [...facts, ...rules].map(serializeClause).join('\n');
    return `${HEADER}${body}\n`;
  }

  private pendingMutationPath(): string {
    return join(this.root, '.pending-mutation.json');
  }

  private pendingNextPath(): string {
    return join(this.root, '.pending-mutation.next');
  }

  private pendingBackupPath(): string {
    return join(this.root, '.pending-mutation.before');
  }

  private unlinkIfPresent(path: string): void {
    try {
      unlinkSync(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  private readPendingMutationUnlocked(): PendingMutation | undefined {
    const path = this.pendingMutationPath();
    let text: string;
    try {
      const stat = lstatSync(path);
      if (stat.isSymbolicLink()) {
        throw new Error(`refusing symbolic-link pending mutation ${path}`);
      }
      if (stat.size > MAX_PENDING_MUTATION_BYTES) {
        throw new Error(`pending mutation exceeds ${MAX_PENDING_MUTATION_BYTES} bytes`);
      }
      text = readFileSync(path, 'utf8');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw error;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error('failed to read pending mutation');
    }
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed) ||
      (parsed as Record<string, unknown>).version !== 1 ||
      typeof (parsed as Record<string, unknown>).namespace !== 'string' ||
      typeof (parsed as Record<string, unknown>).hadPrevious !== 'boolean' ||
      typeof (parsed as Record<string, unknown>).journalEntry !== 'object' ||
      (parsed as Record<string, unknown>).journalEntry === null ||
      Array.isArray((parsed as Record<string, unknown>).journalEntry)
    ) {
      throw new Error('pending mutation has an invalid shape');
    }
    const pending = parsed as PendingMutation;
    this.filePath(pending.namespace);
    if (
      pending.journalEntry.namespace !== pending.namespace ||
      typeof pending.journalEntry.ts !== 'string' ||
      typeof pending.journalEntry.op !== 'string' ||
      typeof pending.journalEntry.opId !== 'string'
    ) {
      throw new Error('pending mutation has an invalid journal entry');
    }
    return pending;
  }

  private recoverPendingMutationUnlocked(): void {
    const pending = this.readPendingMutationUnlocked();
    if (pending === undefined) {
      if (existsSync(this.pendingBackupPath())) {
        throw new Error('orphaned pending mutation backup requires manual recovery');
      }
      // The marker is published before the namespace changes. Without it, an
      // interrupted preparation is safe to discard and must not block writers.
      this.unlinkIfPresent(this.pendingNextPath());
      const markerPrefix = '.pending-mutation.json.tmp-';
      for (const name of readdirSync(this.root)) {
        if (name.startsWith(markerPrefix)) this.unlinkIfPresent(join(this.root, name));
      }
      return;
    }
    const committed = this.readJournalUnlocked().some(
      (entry) => JSON.stringify(entry) === JSON.stringify(pending.journalEntry)
    );

    if (committed) {
      this.completePendingMutationUnlocked(pending);
    } else {
      this.rollbackPendingMutationUnlocked(pending);
    }
  }

  private completePendingMutationUnlocked(pending: PendingMutation): void {
    const target = this.filePath(pending.namespace);
    if (!existsSync(target)) {
      throw new Error('committed pending mutation has no namespace file');
    }
    this.unlinkIfPresent(this.pendingBackupPath());
    this.unlinkIfPresent(this.pendingNextPath());
    this.unlinkIfPresent(this.pendingMutationPath());
    this.cache.delete(pending.namespace);
  }

  private rollbackPendingMutationUnlocked(pending: PendingMutation): void {
    const target = this.filePath(pending.namespace);
    const backup = this.pendingBackupPath();
    if (existsSync(backup)) {
      this.unlinkIfPresent(target);
      renameSync(backup, target);
    } else if (pending.hadPrevious) {
      if (!existsSync(target)) {
        throw new Error('pending mutation lost its previous namespace file');
      }
    } else {
      this.unlinkIfPresent(target);
    }
    this.unlinkIfPresent(this.pendingNextPath());
    this.unlinkIfPresent(this.pendingMutationPath());
    this.cache.delete(pending.namespace);
  }

  private commitMutation(
    namespace: string,
    entry: CachedNamespace,
    journalEntry: JournalEntry
  ): void {
    this.recoverPendingMutationUnlocked();
    const target = this.filePath(namespace);
    const next = this.pendingNextPath();
    const backup = this.pendingBackupPath();
    const pendingPath = this.pendingMutationPath();
    if (existsSync(next) || existsSync(backup) || existsSync(pendingPath)) {
      throw new Error('refusing to overwrite unresolved pending mutation files');
    }
    if (existsSync(target) && lstatSync(target).isSymbolicLink()) {
      throw new Error(`refusing symbolic-link namespace file ${target}`);
    }
    const hadPrevious = existsSync(target);
    writeFileSync(next, this.namespaceBody(entry), {
      encoding: 'utf8',
      mode: 0o600,
      flag: 'wx',
    });
    const pending: PendingMutation = {
      version: 1,
      namespace,
      hadPrevious,
      journalEntry,
    };
    const markerTmp = `${pendingPath}.tmp-${process.pid}-${randomUUID()}`;
    try {
      writeFileSync(markerTmp, `${JSON.stringify(pending)}\n`, {
        encoding: 'utf8',
        mode: 0o600,
        flag: 'wx',
      });
      renameSync(markerTmp, pendingPath);
    } catch (error) {
      this.unlinkIfPresent(markerTmp);
      this.unlinkIfPresent(next);
      throw error;
    }

    let journalCommitted = false;
    try {
      if (hadPrevious) renameSync(target, backup);
      renameSync(next, target);
      this.appendJournalUnlocked(journalEntry);
      journalCommitted = true;
      this.completePendingMutationUnlocked(pending);
    } catch (error) {
      try {
        if (journalCommitted) this.recoverPendingMutationUnlocked();
        else this.rollbackPendingMutationUnlocked(pending);
      } catch (recoveryError) {
        const primary = error instanceof Error ? error.message : String(error);
        const recovery = recoveryError instanceof Error
          ? recoveryError.message
          : String(recoveryError);
        throw new Error(`${primary}; pending mutation recovery failed: ${recovery}`);
      }
      throw error;
    }
    entry.fileStamp = fileStamp(target);
    this.cache.set(namespace, entry);
  }

  load(namespace: string): Clause[] {
    return this.loadCached(namespace).clauses;
  }

  /** Append an entry to the append-only operation journal ("why does it think that?"). */
  note(
    namespace: string,
    op: string,
    details: Record<string, unknown> = {},
    at = new Date()
  ): void {
    const entry = this.createJournalEntry(namespace, op, details, at);
    this.withLock('journal', () => this.appendJournalUnlocked(entry));
  }

  assert(
    namespace: string,
    clauses: string | Clause[],
    context: MutationContext = {}
  ): AssertResult {
    return this.assertInternal(namespace, clauses, context, false);
  }

  private assertInternal(
    namespace: string,
    clauses: string | Clause[],
    context: MutationContext,
    allowTrustMetadata: boolean
  ): AssertResult {
    const explicitOpId = context.opId !== undefined;
    const opId = validateOperationId(context.opId ?? this.createOperationId());
    const at = validDate(context.at ?? new Date(), 'assert timestamp');
    const effectiveContext = { ...context, opId };
    const parsed = typeof clauses === 'string' ? parseProgram(clauses) : clauses;
    assertTrustMetadataSafety(parsed, allowTrustMetadata);
    const requested = parsed.map(serializeClause);
    const requestKeys = parsed.map(canonicalKey);
    return this.withMutationLock(() =>
      this.withNamespaceLock(namespace, () => {
        return this.withLock('journal', () => {
          const priorOperation = explicitOpId
            ? this.readJournalUnlocked().find(
                (journalEntry) =>
                  journalEntry.op === 'assert' &&
                  journalEntry.namespace === namespace &&
                  journalEntry.opId === opId
              )
            : undefined;
          if (priorOperation !== undefined) {
            const added = parseJournalClauseList(
              priorOperation.added,
              `assert operation '${opId}' added`
            );
            if (
              typeof priorOperation.duplicates !== 'number' ||
              !Number.isSafeInteger(priorOperation.duplicates) ||
              priorOperation.duplicates < 0
            ) {
              throw new Error(`assert operation '${opId}' has an invalid duplicate count`);
            }
            const durableRequested = priorOperation.requested === undefined &&
                priorOperation.duplicates === 0
              ? added.map(serializeClause)
              : priorOperation.requested;
            if (
              !Array.isArray(durableRequested) ||
              !durableRequested.every((value) => typeof value === 'string')
            ) {
              throw new Error(`assert operation '${opId}' has invalid durable parameters`);
            }
            let durableRequestKeys: string[];
            try {
              durableRequestKeys = (durableRequested as string[]).map((value, index) =>
                canonicalKey(
                  parseJournalClause(
                    value,
                    `assert operation '${opId}' requested[${index}]`
                  )
                )
              );
            } catch {
              throw new Error(`assert operation '${opId}' has invalid durable parameters`);
            }
            if (
              priorOperation.requestKeys !== undefined &&
              (!Array.isArray(priorOperation.requestKeys) ||
                !priorOperation.requestKeys.every((value) => typeof value === 'string') ||
                JSON.stringify(priorOperation.requestKeys) !==
                  JSON.stringify(durableRequestKeys))
            ) {
              throw new Error(`assert operation '${opId}' has invalid durable parameters`);
            }
            if (JSON.stringify(durableRequestKeys) !== JSON.stringify(requestKeys)) {
              throw new OperationConflictError('assert', namespace, opId);
            }
            return { added, duplicates: priorOperation.duplicates, opId };
          }

          const loaded = this.loadCached(namespace);
          const entry: CachedNamespace = {
            clauses: [...loaded.clauses],
            keys: new Set(loaded.keys),
            fileStamp: loaded.fileStamp,
          };
          const added: Clause[] = [];
          let duplicates = 0;
          for (const clause of parsed) {
            const key = canonicalKey(clause);
            if (entry.keys.has(key)) {
              duplicates++;
            } else {
              entry.keys.add(key);
              entry.clauses.push(clause);
              added.push(clause);
            }
          }
          if (added.length > 0) {
            this.enforceMutation(
              namespace,
              loaded.clauses,
              entry.clauses,
              added,
              effectiveContext,
              at
            );
          }
          if (added.length > 0 || explicitOpId) {
            const journalEntry = this.createJournalEntry(
              namespace,
              'assert',
              {
                opId,
                requested,
                requestKeys,
                added: added.map(serializeClause),
                duplicates,
                ...(context.sourceText === undefined
                  ? {}
                  : { sourceText: context.sourceText }),
                ...(context.origin === undefined ? {} : { origin: context.origin }),
                ...(context.captureId === undefined ? {} : { captureId: context.captureId }),
              },
              at
            );
            if (added.length === 0) this.appendJournalUnlocked(journalEntry);
            else {
              const lineBytes = Buffer.byteLength(`${JSON.stringify(journalEntry)}\n`, 'utf8');
              const path = this.journalPath();
              const currentBytes = existsSync(path) ? statSync(path).size : 0;
              if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
                throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
              }
              this.commitMutation(namespace, entry, journalEntry);
            }
          }
          return { added, duplicates, opId };
        });
      })
    );
  }

  retract(
    namespace: string,
    pattern: string,
    context: MutationContext = {}
  ): { removed: number; opId: string } {
    const explicitOpId = context.opId !== undefined;
    const opId = validateOperationId(context.opId ?? this.createOperationId());
    const at = validDate(context.at ?? new Date(), 'retract timestamp');
    const effectiveContext = { ...context, opId };
    const target = parseRetractionTarget(pattern, 'forget pattern');
    const targetPredicate = 'literal' in target
      ? target.literal.predicate
      : parseProgram(target.serialized)[0]?.head.predicate;
    if (
      targetPredicate !== undefined &&
      isTrustMetadataPredicate(targetPredicate)
    ) {
      throw new TrustMetadataError(
        'raw retraction may not mutate trust metadata; use resolveTentative'
      );
    }
    return this.withMutationLock(() =>
      this.withNamespaceLock(namespace, () => {
        return this.withLock('journal', () => {
          const priorOperation = explicitOpId
            ? this.readJournalUnlocked().find(
                (journalEntry) =>
                  journalEntry.op === 'retract' &&
                  journalEntry.namespace === namespace &&
                  journalEntry.opId === opId
              )
            : undefined;
          if (priorOperation !== undefined) {
            if (typeof priorOperation.pattern !== 'string') {
              throw new Error(`retract operation '${opId}' has invalid durable parameters`);
            }
            let priorTarget: RetractionTarget;
            try {
              priorTarget = parseRetractionTarget(
                priorOperation.pattern,
                `retract operation '${opId}' pattern`
              );
            } catch {
              throw new Error(`retract operation '${opId}' has invalid durable parameters`);
            }
            if (priorTarget.identity !== target.identity) {
              throw new OperationConflictError('retract', namespace, opId);
            }
            if (
              typeof priorOperation.removed !== 'number' ||
              !Number.isSafeInteger(priorOperation.removed) ||
              priorOperation.removed < 0
            ) {
              throw new Error(`retract operation '${opId}' has an invalid removed count`);
            }
            if (priorOperation.removedClauses !== undefined) {
              const removedClauses = parseJournalClauseList(
                priorOperation.removedClauses,
                `retract operation '${opId}' removedClauses`
              );
              if (removedClauses.length !== priorOperation.removed) {
                throw new Error(`retract operation '${opId}' has invalid removed facts`);
              }
            }
            return { removed: priorOperation.removed, opId };
          }

          const loaded = this.loadCached(namespace);
          const entry: CachedNamespace = {
            clauses: [...loaded.clauses],
            keys: new Set(loaded.keys),
            fileStamp: loaded.fileStamp,
          };
          const keep = 'clauseKey' in target
            ? entry.clauses.filter((clause) => canonicalKey(clause) !== target.clauseKey)
            : entry.clauses.filter(
                (clause) =>
                  clause.body.length > 0 || !literalMatches(target.literal, clause.head)
              );
          const keptKeys = new Set(keep.map(canonicalKey));
          const removedClauses = entry.clauses.filter(
            (clause) => !keptKeys.has(canonicalKey(clause))
          );
          const removed = removedClauses.length;
          if (removed > 0) {
            this.enforceMutation(
              namespace,
              loaded.clauses,
              keep,
              [],
              effectiveContext,
              at
            );
          }
          if (removed > 0 || explicitOpId) {
            const journalEntry = this.createJournalEntry(
              namespace,
              'retract',
              {
                opId,
                pattern: target.serialized,
                removed,
                removedClauses: removedClauses.map(serializeClause),
                ...(context.sourceText === undefined
                  ? {}
                  : { sourceText: context.sourceText }),
                ...(context.origin === undefined ? {} : { origin: context.origin }),
                ...(context.captureId === undefined ? {} : { captureId: context.captureId }),
              },
              at
            );
            if (removed === 0) this.appendJournalUnlocked(journalEntry);
            else {
              const lineBytes = Buffer.byteLength(`${JSON.stringify(journalEntry)}\n`, 'utf8');
              const path = this.journalPath();
              const currentBytes = existsSync(path) ? statSync(path).size : 0;
              if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
                throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
              }
              entry.clauses = keep;
              entry.keys = new Set(keep.map(canonicalKey));
              this.commitMutation(namespace, entry, journalEntry);
            }
          }
          return { removed, opId };
        });
      })
    );
  }

  /**
   * Atomically apply one reviewed, digest-bound exact rule change.
   * The caller-supplied validator runs while the global mutation lock is held.
   */
  applyRuleChange(
    namespace: string,
    request: RuleChangeMutationRequest,
    context: MutationContext
  ): RuleChangeMutationResult {
    if (context.opId === undefined) {
      throw new Error('rule change application requires an explicit operation id');
    }
    if (context.integrity?.mode !== 'no_new_violations') {
      throw new Error('rule change application requires no_new_violations enforcement');
    }
    const opId = validateOperationId(context.opId);
    const at = validDate(context.at ?? new Date(), 'rule change timestamp');
    const expectedBaselineDigest = validateSha256(
      request.expectedBaselineDigest,
      'rule change baseline digest'
    );
    const proposalDigest = validateSha256(
      request.proposalDigest,
      'rule change proposal digest'
    );
    const additions = ruleChangeRules(request.add, 'rule change additions');
    const removals = ruleChangeRules(request.remove, 'rule change removals');
    if (additions.length === 0 && removals.length === 0) {
      throw new Error('rule change application requires at least one rule change');
    }
    const namespaces = [...new Set(request.namespaces)];
    if (namespaces.length === 0 || namespaces.length > 32) {
      throw new Error('rule change namespace list must contain 1 to 32 entries');
    }
    if (!namespaces.includes(namespace)) {
      throw new Error(`rule change namespaces must include target '${namespace}'`);
    }
    for (const name of namespaces) this.filePath(name);
    const integrityNames = this.integrityNamespaces(namespace, context.integrity);
    if (JSON.stringify(integrityNames) !== JSON.stringify(namespaces)) {
      throw new Error('rule change integrity namespaces must match proposal namespaces');
    }
    const serializedAdditions = additions.map(serializeClause);
    const serializedRemovals = removals.map(serializeClause);
    const effectiveContext = { ...context, opId };

    return this.withMutationLock(() =>
      this.withNamespaceLock(namespace, () =>
        this.withLock('journal', () => {
          const journal = this.readJournalUnlocked();
          const priorIndex = journal.findIndex(
            (entry) =>
              entry.op === 'rule_change' &&
              entry.namespace === namespace &&
              entry.opId === opId
          );
          const priorOperation = priorIndex < 0 ? undefined : journal[priorIndex];
          if (priorOperation !== undefined) {
            if (
              priorOperation.proposalDigest !== proposalDigest ||
              priorOperation.baselineDigest !== expectedBaselineDigest ||
              JSON.stringify(priorOperation.namespaces) !== JSON.stringify(namespaces) ||
              JSON.stringify(priorOperation.addedRules) !==
                JSON.stringify(serializedAdditions) ||
              JSON.stringify(priorOperation.removedRules) !==
                JSON.stringify(serializedRemovals)
            ) {
              throw new OperationConflictError('rule_change', namespace, opId);
            }
            return {
              added: parseJournalClauseList(
                priorOperation.addedRules,
                `rule change operation '${opId}' addedRules`
              ),
              removed: parseJournalClauseList(
                priorOperation.removedRules,
                `rule change operation '${opId}' removedRules`
              ),
              namespace,
              namespaces,
              baselineDigest: expectedBaselineDigest,
              proposalDigest,
              opId,
              sequence: priorIndex + 1,
            };
          }

          const clausesByNamespace = new Map<string, Clause[]>();
          for (const name of namespaces) {
            clausesByNamespace.set(name, [...this.loadCached(name).clauses]);
          }
          const actualDigest = knowledgeProgramDigest(
            namespaces,
            clausesByNamespace
          );
          if (actualDigest !== expectedBaselineDigest) {
            throw new RuleChangeStaleError(expectedBaselineDigest, actualDigest);
          }
          const loaded = this.loadCached(namespace);
          const removalKeys = new Set(removals.map(canonicalKey));
          const existingKeys = new Set(loaded.clauses.map(canonicalKey));
          if ([...removalKeys].some((key) => !existingKeys.has(key))) {
            throw new Error('reviewed rule removal is absent from the target namespace');
          }
          const retained = loaded.clauses.filter(
            (clause) => !removalKeys.has(canonicalKey(clause))
          );
          const retainedKeys = new Set(retained.map(canonicalKey));
          if (additions.some((clause) => retainedKeys.has(canonicalKey(clause)))) {
            throw new Error('reviewed rule addition already exists in the target namespace');
          }
          const candidateTarget = [...retained, ...additions];
          const candidateByNamespace = new Map(clausesByNamespace);
          candidateByNamespace.set(namespace, candidateTarget);
          const candidateView: RuleChangeCandidateView = {
            namespaces,
            clauses: namespaces.flatMap(
              (name) => candidateByNamespace.get(name) ?? []
            ),
            clausesByNamespace: candidateByNamespace,
          };
          request.validateCandidate?.(candidateView);
          this.enforceMutation(
            namespace,
            loaded.clauses,
            candidateTarget,
            additions,
            effectiveContext,
            at
          );

          const journalEntry = this.createJournalEntry(
            namespace,
            'rule_change',
            {
              opId,
              namespaces,
              baselineDigest: expectedBaselineDigest,
              proposalDigest,
              addedRules: serializedAdditions,
              removedRules: serializedRemovals,
              ...(context.sourceText === undefined
                ? {}
                : { sourceText: context.sourceText }),
              ...(context.origin === undefined ? {} : { origin: context.origin }),
            },
            at
          );
          const lineBytes = Buffer.byteLength(
            `${JSON.stringify(journalEntry)}\n`,
            'utf8'
          );
          const journalPath = this.journalPath();
          const currentBytes = existsSync(journalPath)
            ? statSync(journalPath).size
            : 0;
          if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
            throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
          }
          const entry: CachedNamespace = {
            clauses: candidateTarget,
            keys: new Set(candidateTarget.map(canonicalKey)),
            fileStamp: loaded.fileStamp,
          };
          this.commitMutation(namespace, entry, journalEntry);
          return {
            added: additions,
            removed: removals,
            namespace,
            namespaces,
            baselineDigest: expectedBaselineDigest,
            proposalDigest,
            opId,
            sequence: journal.length + 1,
          };
        })
      )
    );
  }

  /** Atomically apply one reviewed, digest-bound accepted-memory proposal. */
  applyMemoryChange(
    namespace: string,
    request: MemoryChangeMutationRequest,
    context: MutationContext
  ): MemoryChangeMutationResult {
    if (context.opId === undefined) {
      throw new Error('memory change application requires an explicit operation id');
    }
    if (context.integrity?.mode !== 'no_new_violations') {
      throw new Error('memory change application requires no_new_violations enforcement');
    }
    const opId = validateOperationId(context.opId);
    const at = validDate(context.at ?? new Date(), 'memory change timestamp');
    const expectedBaselineDigest = validateSha256(
      request.expectedBaselineDigest,
      'memory change baseline digest'
    );
    const proposalDigest = validateSha256(
      request.proposalDigest,
      'memory change proposal digest'
    );
    const additions = reviewedMemoryClauses(
      request.add,
      'memory change additions'
    );
    const removals = reviewedMemoryClauses(
      request.remove,
      'memory change removals'
    );
    if (additions.length === 0 && removals.length === 0) {
      throw new Error('memory change application requires at least one clause change');
    }
    const namespaces = [...new Set(request.namespaces)];
    if (namespaces.length === 0 || namespaces.length > 32) {
      throw new Error('memory change namespace list must contain 1 to 32 entries');
    }
    if (!namespaces.includes(namespace)) {
      throw new Error(`memory change namespaces must include target '${namespace}'`);
    }
    for (const name of namespaces) this.filePath(name);
    const integrityNames = this.integrityNamespaces(namespace, context.integrity);
    if (JSON.stringify(integrityNames) !== JSON.stringify(namespaces)) {
      throw new Error('memory change integrity namespaces must match proposal namespaces');
    }
    const serializedAdditions = additions.map(serializeClause);
    const serializedRemovals = removals.map(serializeClause);
    const additionKeys = new Set(additions.map(canonicalKey));
    const removalKeys = new Set(removals.map(canonicalKey));
    const temporalArchives = request.temporalArchives ?? [];
    const temporalByClause = new Map<string, TemporalMemorySource>();
    const archived: Clause[] = [];
    for (const [index, archive] of temporalArchives.entries()) {
      const from = parseJournalClause(
        archive.from,
        `memory change temporalArchives[${index}].from`
      );
      const to = parseJournalClause(
        archive.to,
        `memory change temporalArchives[${index}].to`
      );
      assertIsoTimestamp(
        archive.validUntil,
        `memory change temporalArchives[${index}].validUntil`
      );
      if (
        !removalKeys.has(canonicalKey(from)) ||
        !additionKeys.has(canonicalKey(to)) ||
        canonicalKey(archiveUntilClause(from, archive.validUntil)) !==
          canonicalKey(to)
      ) {
        throw new Error(`memory change temporalArchives[${index}] is inconsistent`);
      }
      if (temporalByClause.has(canonicalKey(to))) {
        throw new Error('memory change contains duplicate temporal archives');
      }
      temporalByClause.set(canonicalKey(to), {
        kind: 'superseded',
        previousClause: serializeClause(from),
        validUntil: archive.validUntil,
      });
      archived.push(to);
    }
    const effectiveContext = { ...context, opId };

    return this.withMutationLock(() =>
      this.withNamespaceLock(namespace, () =>
        this.withLock('journal', () => {
          const journal = this.readJournalUnlocked();
          const priorIndex = journal.findIndex(
            (entry) =>
              entry.op === 'memory_change' &&
              entry.namespace === namespace &&
              entry.opId === opId
          );
          const prior = priorIndex < 0 ? undefined : journal[priorIndex];
          if (prior !== undefined) {
            if (
              prior.proposalDigest !== proposalDigest ||
              prior.baselineDigest !== expectedBaselineDigest ||
              JSON.stringify(prior.namespaces) !== JSON.stringify(namespaces) ||
              JSON.stringify(prior.addedClauses) !==
                JSON.stringify(serializedAdditions) ||
              JSON.stringify(prior.removedClauses) !==
                JSON.stringify(serializedRemovals) ||
              JSON.stringify(prior.temporalArchives ?? []) !==
                JSON.stringify(temporalArchives)
            ) {
              throw new OperationConflictError('memory_change', namespace, opId);
            }
            return {
              added: parseJournalClauseList(
                prior.addedClauses,
                `memory change operation '${opId}' addedClauses`
              ),
              removed: parseJournalClauseList(
                prior.removedClauses,
                `memory change operation '${opId}' removedClauses`
              ),
              archived: temporalArchives.map((value, index) =>
                parseJournalClause(
                  value.to,
                  `memory change operation '${opId}' temporalArchives[${index}].to`
                )
              ),
              namespace,
              namespaces,
              baselineDigest: expectedBaselineDigest,
              proposalDigest,
              opId,
              sequence: priorIndex + 1,
            };
          }

          const clausesByNamespace = new Map<string, Clause[]>();
          for (const name of namespaces) {
            clausesByNamespace.set(name, [...this.loadCached(name).clauses]);
          }
          const actualDigest = knowledgeProgramDigest(
            namespaces,
            clausesByNamespace
          );
          if (actualDigest !== expectedBaselineDigest) {
            throw new MemoryChangeStaleError(expectedBaselineDigest, actualDigest);
          }
          const loaded = this.loadCached(namespace);
          const existingKeys = new Set(loaded.clauses.map(canonicalKey));
          if ([...removalKeys].some((key) => !existingKeys.has(key))) {
            throw new Error('reviewed memory removal is absent from the target namespace');
          }
          const retained = loaded.clauses.filter(
            (clause) => !removalKeys.has(canonicalKey(clause))
          );
          const retainedKeys = new Set(retained.map(canonicalKey));
          if (additions.some((clause) => retainedKeys.has(canonicalKey(clause)))) {
            throw new Error('reviewed memory addition already exists in the target namespace');
          }
          const candidateTarget = [...retained, ...additions];
          const candidateByNamespace = new Map(clausesByNamespace);
          candidateByNamespace.set(namespace, candidateTarget);
          request.validateCandidate?.({
            namespaces,
            clauses: namespaces.flatMap(
              (name) => candidateByNamespace.get(name) ?? []
            ),
            clausesByNamespace: candidateByNamespace,
          });
          this.enforceMutation(
            namespace,
            loaded.clauses,
            candidateTarget,
            additions,
            effectiveContext,
            at,
            temporalByClause
          );

          const journalEntry = this.createJournalEntry(
            namespace,
            'memory_change',
            {
              opId,
              namespaces,
              baselineDigest: expectedBaselineDigest,
              proposalDigest,
              addedClauses: serializedAdditions,
              removedClauses: serializedRemovals,
              temporalArchives,
              ...(context.sourceText === undefined
                ? {}
                : { sourceText: context.sourceText }),
              ...(context.origin === undefined ? {} : { origin: context.origin }),
            },
            at
          );
          const lineBytes = Buffer.byteLength(
            `${JSON.stringify(journalEntry)}\n`,
            'utf8'
          );
          const journalPath = this.journalPath();
          const currentBytes = existsSync(journalPath)
            ? statSync(journalPath).size
            : 0;
          if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
            throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
          }
          const entry: CachedNamespace = {
            clauses: candidateTarget,
            keys: new Set(candidateTarget.map(canonicalKey)),
            fileStamp: loaded.fileStamp,
          };
          this.commitMutation(namespace, entry, journalEntry);
          return {
            added: additions,
            removed: removals,
            archived,
            namespace,
            namespaces,
            baselineDigest: expectedBaselineDigest,
            proposalDigest,
            opId,
            sequence: journal.length + 1,
          };
        })
      )
    );
  }

  private latestJournalSourcesUnlocked(namespace: string): Map<string, string> {
    const sources = new Map<string, { clause: Clause; opId: string }>();
    for (const [index, journalEntry] of this.readJournalUnlocked().entries()) {
      const label = `journal.log line ${index + 1}`;
      if (journalEntry.namespace !== namespace) continue;
      if (journalEntry.op === 'assert') {
        if (typeof journalEntry.opId !== 'string') throw new Error(`${label} has no opId`);
        for (const clause of parseJournalClauseList(journalEntry.added, `${label} added`)) {
          sources.set(canonicalKey(clause), { clause, opId: journalEntry.opId });
        }
        continue;
      }
      if (journalEntry.op === 'rule_change') {
        if (typeof journalEntry.opId !== 'string') throw new Error(`${label} has no opId`);
        for (const clause of parseJournalClauseList(
          journalEntry.removedRules,
          `${label} removedRules`
        )) {
          sources.delete(canonicalKey(clause));
        }
        for (const clause of parseJournalClauseList(
          journalEntry.addedRules,
          `${label} addedRules`
        )) {
          sources.set(canonicalKey(clause), {
            clause,
            opId: journalEntry.opId,
          });
        }
        continue;
      }
      if (journalEntry.op === 'memory_change') {
        if (typeof journalEntry.opId !== 'string') throw new Error(`${label} has no opId`);
        for (const clause of parseJournalClauseList(
          journalEntry.removedClauses,
          `${label} removedClauses`
        )) {
          sources.delete(canonicalKey(clause));
        }
        for (const clause of parseJournalClauseList(
          journalEntry.addedClauses,
          `${label} addedClauses`
        )) {
          sources.set(canonicalKey(clause), {
            clause,
            opId: journalEntry.opId,
          });
        }
        continue;
      }
      if (journalEntry.op === 'supersede') {
        if (typeof journalEntry.opId !== 'string') throw new Error(`${label} has no opId`);
        if (
          !Array.isArray(journalEntry.patterns) ||
          journalEntry.patterns.length === 0 ||
          journalEntry.patterns.length > 64 ||
          !journalEntry.patterns.every((value) => typeof value === 'string')
        ) {
          throw new Error(`${label} patterns must be a non-empty string array`);
        }
        journalEntry.patterns.forEach((value, patternIndex) =>
          parseFactPattern(value as string, `${label} patterns[${patternIndex}]`)
        );
        if (!Array.isArray(journalEntry.ended)) throw new Error(`${label} ended must be an array`);
        for (const [endedIndex, ended] of journalEntry.ended.entries()) {
          if (typeof ended !== 'object' || ended === null || Array.isArray(ended)) {
            throw new Error(`${label} ended[${endedIndex}] must be an object`);
          }
          const clause = parseJournalClause(
            (ended as Record<string, unknown>).clause,
            `${label} ended[${endedIndex}].clause`
          );
          sources.delete(canonicalKey(clause));
        }
        for (const clause of parseJournalClauseList(journalEntry.added, `${label} added`)) {
          sources.set(canonicalKey(clause), { clause, opId: journalEntry.opId });
        }
        continue;
      }
      if (journalEntry.op !== 'retract') continue;
      if (Array.isArray(journalEntry.removedClauses)) {
        for (const clause of parseJournalClauseList(
          journalEntry.removedClauses,
          `${label} removedClauses`
        )) {
          sources.delete(canonicalKey(clause));
        }
        continue;
      }
      if (typeof journalEntry.pattern !== 'string') throw new Error(`${label} has no pattern`);
      const target = parseRetractionTarget(journalEntry.pattern, `${label} pattern`);
      if ('clauseKey' in target) {
        sources.delete(target.clauseKey);
        continue;
      }
      for (const [key, candidate] of sources) {
        if (
          candidate.clause.body.length === 0 &&
          literalMatches(target.literal, candidate.clause.head)
        ) {
          sources.delete(key);
        }
      }
    }
    return new Map([...sources].map(([key, value]) => [key, value.opId]));
  }

  /**
   * Atomically end matching ground facts, retain them as ordinary *_until facts,
   * and add their replacements. Append order is the authoritative event order;
   * the timestamp is descriptive valid-time metadata.
   */
  supersede(
    namespace: string,
    patterns: string[],
    replacements: string | Clause[],
    context: MutationContext = {}
  ): SupersedeResult {
    return this.replaceFacts(namespace, patterns, replacements, true, context);
  }

  /** Atomically retract matching ground facts and add their replacements. */
  replace(
    namespace: string,
    patterns: string[],
    replacements: string | Clause[],
    context: MutationContext = {}
  ): SupersedeResult {
    return this.replaceFacts(namespace, patterns, replacements, false, context);
  }

  /** Store explicit ground facts outside the accepted reasoning view pending review. */
  assertTentative(
    namespace: string,
    clauses: string | Clause[],
    context: MutationContext = {}
  ): AssertResult {
    return this.assertInternal(
      namespace,
      wrapTentativeFacts(clauses),
      context,
      true
    );
  }

  /** Import portable clauses, including fully validated tentative declarations. */
  importClauses(
    namespace: string,
    clauses: string | Clause[],
    context: MutationContext = {}
  ): AssertResult {
    const parsed = typeof clauses === 'string' ? parseProgram(clauses) : clauses;
    assertTrustMetadataSafety(parsed, true);
    return this.assertInternal(namespace, parsed, context, true);
  }

  /** Atomically accept or reject exact tentative facts; every requested claim must exist. */
  resolveTentative(
    namespace: string,
    clauses: string | Clause[],
    action: TentativeResolutionAction,
    context: MutationContext = {}
  ): SupersedeResult {
    if (action !== 'accept' && action !== 'reject') {
      throw new TrustMetadataError("tentative action must be 'accept' or 'reject'");
    }
    const declarations = wrapTentativeFacts(clauses);
    if (new Set(declarations.map(canonicalKey)).size !== declarations.length) {
      throw new TrustMetadataError('tentative resolution contains duplicate claims');
    }
    const facts = declarations.map((declaration) => {
      const fact = decodeTentativeDeclaration(declaration);
      if (fact === undefined) throw new Error('expected tentative declaration');
      return fact;
    });
    const patterns = declarations.map((declaration) =>
      serializeGoal(declaration.head)
    );
    return this.replaceFacts(
      namespace,
      patterns,
      action === 'accept' ? facts : [],
      false,
      context,
      true,
      action,
      true
    );
  }

  private replaceFacts(
    namespace: string,
    patterns: string[],
    replacements: string | Clause[],
    archive: boolean,
    context: MutationContext,
    requireEveryPattern = false,
    trustAction?: TentativeResolutionAction,
    allowTrustMetadata = false
  ): SupersedeResult {
    if (patterns.length === 0) throw new Error('supersede requires at least one fact pattern');
    if (patterns.length > MAX_SUPERSEDE_PATTERNS) {
      throw new Error(`supersede accepts at most ${MAX_SUPERSEDE_PATTERNS} fact patterns`);
    }
    const parsedPatterns = patterns.map((pattern) => ({
      literal: parseFactPattern(pattern, 'supersede pattern'),
    }));
    if (
      !allowTrustMetadata &&
      parsedPatterns.some(({ literal }) =>
        isTrustMetadataPredicate(literal.predicate)
      )
    ) {
      throw new TrustMetadataError(
        'raw replacement may not mutate trust metadata; use resolveTentative'
      );
    }
    const requestedPatterns = parsedPatterns.map((pattern) => serializeGoal(pattern.literal));
    const parsedReplacements =
      typeof replacements === 'string' ? parseProgram(replacements) : replacements;
    assertTrustMetadataSafety(parsedReplacements, allowTrustMetadata);
    const explicitOpId = context.opId !== undefined;
    const opId = validateOperationId(context.opId ?? this.createOperationId());
    const at = validDate(
      context.at ?? new Date(),
      archive ? 'supersession timestamp' : 'replacement timestamp'
    );
    const validUntil = at.toISOString();
    const requestedReplacementClauses = parsedReplacements.map(serializeClause);
    const effectiveContext = { ...context, opId };

    return this.withMutationLock(() => this.withNamespaceLock(namespace, () => {
      const loaded = this.loadCached(namespace);
      const entry: CachedNamespace = {
        clauses: [...loaded.clauses],
        keys: new Set(loaded.keys),
        fileStamp: loaded.fileStamp,
      };

      return this.withLock('journal', () => {
        const priorOperation = explicitOpId
          ? this.readJournalUnlocked().find(
              (journalEntry) =>
                journalEntry.op === 'supersede' &&
                journalEntry.namespace === namespace &&
                journalEntry.opId === opId
            )
          : undefined;
        if (priorOperation !== undefined) {
          if (
            !Array.isArray(priorOperation.patterns) ||
            !priorOperation.patterns.every((value) => typeof value === 'string') ||
            !Array.isArray(priorOperation.replacementRequested) ||
            !priorOperation.replacementRequested.every((value) => typeof value === 'string') ||
            typeof priorOperation.ts !== 'string' ||
            (priorOperation.trustAction !== undefined &&
              priorOperation.trustAction !== 'accept' &&
              priorOperation.trustAction !== 'reject') ||
            (priorOperation.atProvided !== undefined &&
              typeof priorOperation.atProvided !== 'boolean')
          ) {
            throw new Error(`supersede operation '${opId}' has invalid durable parameters`);
          }
          if (
            JSON.stringify(priorOperation.patterns) !==
              JSON.stringify(requestedPatterns) ||
            JSON.stringify(priorOperation.replacementRequested) !==
              JSON.stringify(requestedReplacementClauses) ||
            (priorOperation.validTimeMode ?? 'archive_until') !==
              (archive ? 'archive_until' : 'delete') ||
            priorOperation.trustAction !== trustAction ||
            (priorOperation.atProvided === undefined && context.at === undefined) ||
            (typeof priorOperation.atProvided === 'boolean' &&
              priorOperation.atProvided !== (context.at !== undefined)) ||
            (context.at !== undefined && priorOperation.ts !== validUntil)
          ) {
            throw new OperationConflictError(
              trustAction === undefined ? 'supersede' : 'resolve_tentative',
              namespace,
              opId
            );
          }
          if (!Array.isArray(priorOperation.ended)) {
            throw new Error(`supersede operation '${opId}' has invalid ended facts`);
          }
          if (!Array.isArray(priorOperation.archived)) {
            throw new Error(`supersede operation '${opId}' has invalid archived facts`);
          }
          const archived = priorOperation.archived.map((value, index) => {
            if (typeof value !== 'object' || value === null || Array.isArray(value)) {
              throw new Error(`supersede operation '${opId}' archived[${index}] is invalid`);
            }
            return parseJournalClause(
              (value as Record<string, unknown>).to,
              `supersede operation '${opId}' archived[${index}].to`
            );
          });
          const added = parseJournalClauseList(
            priorOperation.replacementAdded,
            `supersede operation '${opId}' replacementAdded`
          );
          if (
            typeof priorOperation.duplicates !== 'number' ||
            !Number.isSafeInteger(priorOperation.duplicates) ||
            priorOperation.duplicates < 0
          ) {
            throw new Error(`supersede operation '${opId}' has an invalid duplicate count`);
          }
          return {
            added,
            duplicates: priorOperation.duplicates,
            retracted: priorOperation.ended.length,
            archived,
            opId,
          };
        }
        const previousSources = this.latestJournalSourcesUnlocked(namespace);
        const ended: Clause[] = [];
        const endedKeys = new Set<string>();
        for (const { literal } of parsedPatterns) {
          for (const clause of entry.clauses) {
            const key = canonicalKey(clause);
            if (
              clause.body.length === 0 &&
              !endedKeys.has(key) &&
              literalMatches(literal, clause.head)
            ) {
              if (archive) archiveUntilClause(clause, validUntil); // validate before changing state
              endedKeys.add(key);
              ended.push(clause);
            }
          }
        }
        if (requireEveryPattern && ended.length !== parsedPatterns.length) {
          throw new TrustMetadataError(
            `tentative resolution requires all ${parsedPatterns.length} requested claims to be current`
          );
        }

        const archives = archive
          ? ended.map((clause) => archiveUntilClause(clause, validUntil))
          : [];
        entry.clauses = entry.clauses.filter((clause) => !endedKeys.has(canonicalKey(clause)));
        entry.keys = new Set(entry.clauses.map(canonicalKey));

        const archivedAdded: Clause[] = [];
        for (const clause of archives) {
          const key = canonicalKey(clause);
          if (!entry.keys.has(key)) {
            entry.keys.add(key);
            entry.clauses.push(clause);
            archivedAdded.push(clause);
          }
        }

        const replacementAdded: Clause[] = [];
        let duplicates = 0;
        for (const clause of parsedReplacements) {
          const key = canonicalKey(clause);
          if (entry.keys.has(key)) {
            duplicates++;
          } else {
            entry.keys.add(key);
            entry.clauses.push(clause);
            replacementAdded.push(clause);
          }
        }

        const allAdded = [...archivedAdded, ...replacementAdded];
        if (ended.length > 0 || allAdded.length > 0 || explicitOpId) {
          const archivedAddedKeys = new Set(archivedAdded.map(canonicalKey));
          const proposedTemporalSources = new Map<string, TemporalMemorySource>();
          for (const [index, clause] of archives.entries()) {
            const key = canonicalKey(clause);
            if (!archivedAddedKeys.has(key)) continue;
            proposedTemporalSources.set(key, {
              kind: 'superseded',
              previousClause: serializeClause(ended[index]),
              validUntil,
            });
          }
          this.enforceMutation(
            namespace,
            loaded.clauses,
            entry.clauses,
            allAdded,
            effectiveContext,
            at,
            proposedTemporalSources
          );
          const journalEntry = this.createJournalEntry(
            namespace,
            'supersede',
            {
              opId,
              validTimeMode: archive ? 'archive_until' : 'delete',
              atProvided: context.at !== undefined,
              patterns: requestedPatterns,
              ended: ended.map((clause) => ({
                clause: serializeClause(clause),
                ...(previousSources.get(canonicalKey(clause)) === undefined
                  ? {}
                  : { sourceOpId: previousSources.get(canonicalKey(clause)) }),
              })),
              archived: archive
                ? ended.map((clause, index) => ({
                    from: serializeClause(clause),
                    to: serializeClause(archives[index]),
                    validUntil,
                  }))
                : [],
              added: allAdded.map(serializeClause),
              replacementRequested: requestedReplacementClauses,
              replacementAdded: replacementAdded.map(serializeClause),
              duplicates,
              ...(trustAction === undefined ? {} : { trustAction }),
              ...(context.sourceText === undefined ? {} : { sourceText: context.sourceText }),
              ...(context.origin === undefined ? {} : { origin: context.origin }),
              ...(context.captureId === undefined ? {} : { captureId: context.captureId }),
            },
            at
          );
          const lineBytes = Buffer.byteLength(`${JSON.stringify(journalEntry)}\n`, 'utf8');
          const path = this.journalPath();
          const currentBytes = existsSync(path) ? statSync(path).size : 0;
          if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
            throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
          }
          if (ended.length === 0 && allAdded.length === 0) {
            this.appendJournalUnlocked(journalEntry);
          } else {
            this.commitMutation(namespace, entry, journalEntry);
          }
        }

        return {
          added: replacementAdded,
          duplicates,
          retracted: ended.length,
          archived: archives,
          opId,
        };
      });
    }));
  }

  private retractFactIfSourcedBy(
    namespace: string,
    serialized: string,
    expectedSourceOpId: string,
    context: Required<Pick<MutationContext, 'opId' | 'captureId' | 'at'>> &
      Pick<MutationContext, 'integrity' | 'checks'>
  ): number {
    const clauses = parseProgram(serialized);
    if (clauses.length !== 1 || clauses[0].body.length !== 0) {
      throw new Error('auto-capture pruning accepts exactly one ground fact');
    }
    const target = clauses[0];
    const targetKey = canonicalKey(target);
    return this.withNamespaceLock(namespace, () => {
      const loaded = this.loadCached(namespace);
      if (!loaded.keys.has(targetKey)) return 0;
      const entry: CachedNamespace = {
        clauses: [...loaded.clauses],
        keys: new Set(loaded.keys),
        fileStamp: loaded.fileStamp,
      };

      return this.withLock('journal', () => {
        let latestSourceOpId: string | undefined;
        for (const journalEntry of this.readJournalUnlocked()) {
          if (
            journalEntry.op !== 'assert' ||
            journalEntry.namespace !== namespace ||
            typeof journalEntry.opId !== 'string' ||
            !Array.isArray(journalEntry.added)
          ) {
            continue;
          }
          const containsTarget = journalEntry.added.some((candidate) => {
            if (typeof candidate !== 'string') return false;
            const [clause] = parseProgram(candidate);
            return canonicalKey(clause) === targetKey;
          });
          if (containsTarget) latestSourceOpId = journalEntry.opId;
        }
        if (latestSourceOpId !== expectedSourceOpId) return 0;

        entry.clauses = entry.clauses.filter((clause) => canonicalKey(clause) !== targetKey);
        entry.keys.delete(targetKey);
        this.enforceMutation(
          namespace,
          loaded.clauses,
          entry.clauses,
          [],
          context,
          context.at
        );
        const journalEntry = this.createJournalEntry(
          namespace,
          'retract',
          {
            opId: context.opId,
            pattern: serializeClause(target),
            removed: 1,
            sourceText: 'Pruned from auto-capture review',
            origin: 'manual',
            captureId: context.captureId,
          },
          context.at
        );
        const lineBytes = Buffer.byteLength(`${JSON.stringify(journalEntry)}\n`, 'utf8');
        const path = this.journalPath();
        const currentBytes = existsSync(path) ? statSync(path).size : 0;
        if (currentBytes + lineBytes > MAX_JOURNAL_BYTES) {
          throw new Error(`journal.log would exceed ${MAX_JOURNAL_BYTES} bytes`);
        }
        this.commitMutation(namespace, entry, journalEntry);
        return 1;
      });
    });
  }

  reserveAutoCapture(
    namespace: string,
    request: AutoCaptureReservationRequest
  ): AutoCaptureReservation {
    this.filePath(namespace);
    if (!/^[a-f0-9]{64}$/.test(request.fingerprint)) {
      throw new Error('auto-capture fingerprint must be a SHA-256 hex digest');
    }
    if (!/^[a-zA-Z0-9._-]{1,256}$/.test(request.sessionId)) {
      throw new Error('auto-capture session id contains invalid characters');
    }
    if (!Number.isSafeInteger(request.tailBytes) || request.tailBytes < 0) {
      throw new Error('auto-capture tail byte count must be a non-negative integer');
    }
    if (!Number.isSafeInteger(request.dailyCap) || request.dailyCap < 1) {
      throw new Error('auto-capture daily cap must be a positive integer');
    }
    const at = validDate(request.at ?? new Date(), 'auto-capture timestamp');
    const captureId = request.captureId ?? this.createOperationId();
    if (!/^[a-zA-Z0-9._-]{1,256}$/.test(captureId)) {
      throw new Error('auto-capture id contains invalid characters');
    }

    return this.withLock('journal', () => {
      const entries = this.readJournalUnlocked();
      const duplicate = entries.some(
        (entry) =>
          entry.op === 'auto_capture' &&
          entry.namespace === namespace &&
          entry.status === 'started' &&
          entry.fingerprint === request.fingerprint
      );
      if (duplicate) {
        this.appendJournalUnlocked(
          this.createJournalEntry(
            namespace,
            'auto_capture',
            {
              captureId,
              status: 'skipped',
              reason: 'duplicate',
              source: 'claude-stop',
              sessionId: request.sessionId,
              fingerprint: request.fingerprint,
              tailBytes: request.tailBytes,
            },
            at
          )
        );
        return { captureId, reserved: false, reason: 'duplicate' };
      }

      const day = utcDay(at);
      const used = entries.filter(
        (entry) =>
          entry.op === 'auto_capture' &&
          entry.namespace === namespace &&
          entry.status === 'started' &&
          typeof entry.ts === 'string' &&
          entry.ts.slice(0, 10) === day
      ).length;
      if (used >= request.dailyCap) {
        this.appendJournalUnlocked(
          this.createJournalEntry(
            namespace,
            'auto_capture',
            {
              captureId,
              status: 'skipped',
              reason: 'daily_cap',
              source: 'claude-stop',
              sessionId: request.sessionId,
              fingerprint: request.fingerprint,
              tailBytes: request.tailBytes,
              dailyCap: request.dailyCap,
            },
            at
          )
        );
        return { captureId, reserved: false, reason: 'daily_cap' };
      }

      this.appendJournalUnlocked(
        this.createJournalEntry(
          namespace,
          'auto_capture',
          {
            captureId,
            status: 'started',
            source: 'claude-stop',
            sessionId: request.sessionId,
            fingerprint: request.fingerprint,
            tailBytes: request.tailBytes,
            dailyCap: request.dailyCap,
          },
          at
        )
      );
      return { captureId, reserved: true };
    });
  }

  finishAutoCapture(
    namespace: string,
    captureId: string,
    status: Exclude<AutoCaptureStatus, 'started' | 'skipped'>,
    details: { added?: number; duplicates?: number; reason?: string } = {},
    at = new Date()
  ): void {
    if (captureId.trim() === '' || captureId.length > 256) {
      throw new Error('auto-capture id must be between 1 and 256 characters');
    }
    if (details.reason !== undefined && !/^[a-z0-9_-]{1,64}$/.test(details.reason)) {
      throw new Error('auto-capture failure reason must be a short machine-readable code');
    }
    this.note(
      namespace,
      'auto_capture',
      {
        captureId,
        status,
        source: 'claude-stop',
        ...(details.added === undefined ? {} : { added: details.added }),
        ...(details.duplicates === undefined ? {} : { duplicates: details.duplicates }),
        ...(details.reason === undefined ? {} : { reason: details.reason }),
      },
      at
    );
  }

  recordAutoCaptureSkip(
    namespace: string,
    reason: string,
    options: { captureId?: string; at?: Date } = {}
  ): string {
    if (!/^[a-z0-9_-]{1,64}$/.test(reason)) {
      throw new Error('auto-capture skip reason must be a short machine-readable code');
    }
    const captureId = options.captureId ?? this.createOperationId();
    this.note(
      namespace,
      'auto_capture',
      { captureId, status: 'skipped', reason, source: 'claude-stop' },
      options.at
    );
    return captureId;
  }

  /**
   * Record a capture failure when the primary journal itself is unavailable.
   * This secondary bounded log uses an independent lock and is merged into review.
   */
  recordAutoCaptureEmergency(
    namespace: string,
    captureId: string,
    reason = 'journal_unavailable',
    at = new Date()
  ): void {
    if (!/^[a-zA-Z0-9._-]{1,256}$/.test(captureId)) {
      throw new Error('auto-capture id contains invalid characters');
    }
    if (!/^[a-z0-9_-]{1,64}$/.test(reason)) {
      throw new Error('auto-capture emergency reason must be a short machine-readable code');
    }
    const entry = this.createJournalEntry(
      namespace,
      'auto_capture',
      {
        captureId,
        status: 'failed',
        reason,
        source: 'claude-stop',
        fallback: true,
      },
      at
    );
    this.withLock('capture-errors', () => {
      const path = this.captureErrorPath();
      const line = `${JSON.stringify(entry)}\n`;
      const currentBytes = existsSync(path) ? statSync(path).size : 0;
      if (currentBytes + Buffer.byteLength(line, 'utf8') > MAX_CAPTURE_ERROR_BYTES) {
        throw new Error(`capture-errors.log would exceed ${MAX_CAPTURE_ERROR_BYTES} bytes`);
      }
      appendFileSync(path, line, { encoding: 'utf8', mode: 0o600 });
    });
  }

  reviewAutoCaptures(options: AutoCaptureReviewOptions = {}): AutoCaptureReview {
    const days = options.days ?? 7;
    if (!Number.isSafeInteger(days) || days < 1 || days > MAX_REVIEW_DAYS) {
      throw new Error(`review days must be an integer between 1 and ${MAX_REVIEW_DAYS}`);
    }
    if (options.namespace !== undefined) this.filePath(options.namespace);
    const now = validDate(options.now ?? new Date(), 'review timestamp');
    const since = now.getTime() - days * 24 * 60 * 60 * 1000;
    const entries = [
      ...this.withLock('journal', () => this.readJournalUnlocked()),
      ...this.withLock('capture-errors', () => this.readCaptureErrorsUnlocked()),
    ].sort((left, right) => left.ts.localeCompare(right.ts));
    const captures = new Map<string, AutoCaptureBatch>();
    const facts: AutoCaptureFact[] = [];

    for (const entry of entries) {
      if (options.namespace !== undefined && entry.namespace !== options.namespace) continue;
      const timestamp = Date.parse(entry.ts);
      if (!Number.isFinite(timestamp) || timestamp < since || timestamp > now.getTime()) continue;
      if (
        entry.op === 'auto_capture' &&
        typeof entry.captureId === 'string' &&
        (entry.status === 'started' ||
          entry.status === 'captured' ||
          entry.status === 'empty' ||
          entry.status === 'failed' ||
          entry.status === 'skipped')
      ) {
        const previous = captures.get(entry.captureId);
        captures.set(entry.captureId, {
          captureId: entry.captureId,
          namespace: entry.namespace,
          ts: previous?.ts ?? entry.ts,
          status: entry.status,
          ...(typeof entry.sessionId === 'string'
            ? { sessionId: entry.sessionId }
            : previous?.sessionId === undefined
              ? {}
              : { sessionId: previous.sessionId }),
          ...(typeof entry.reason === 'string' ? { reason: entry.reason } : {}),
          ...(typeof entry.added === 'number' ? { added: entry.added } : {}),
          ...(typeof entry.duplicates === 'number' ? { duplicates: entry.duplicates } : {}),
        });
      }
    }

    const sourceOperationsByNamespace = new Map<string, Map<string, string>>();
    const sourceOperations = (namespace: string): Map<string, string> => {
      const existing = sourceOperationsByNamespace.get(namespace);
      if (existing !== undefined) return existing;
      const operations = new Map<string, string>();
      for (const [key, sources] of this.sourcesFor([namespace])) {
        const source = sources.find((candidate) => candidate.namespace === namespace);
        if (source !== undefined) operations.set(key, source.opId);
      }
      sourceOperationsByNamespace.set(namespace, operations);
      return operations;
    };

    for (const entry of entries) {
      if (options.namespace !== undefined && entry.namespace !== options.namespace) continue;
      const timestamp = Date.parse(entry.ts);
      if (!Number.isFinite(timestamp) || timestamp < since || timestamp > now.getTime()) continue;
      if (
        entry.op !== 'assert' ||
        entry.origin !== 'claude-stop' ||
        typeof entry.captureId !== 'string' ||
        typeof entry.opId !== 'string' ||
        !Array.isArray(entry.added)
      ) {
        continue;
      }
      for (const serialized of entry.added) {
        if (typeof serialized !== 'string') continue;
        const [clause] = parseProgram(serialized);
        const canonical = serializeClause(clause);
        facts.push({
          id: autoCaptureFactId(entry.captureId, entry.namespace, canonical),
          captureId: entry.captureId,
          opId: entry.opId,
          namespace: entry.namespace,
          ts: entry.ts,
          clause: canonical,
          current: sourceOperations(entry.namespace).get(canonicalKey(clause)) === entry.opId,
        });
      }
    }

    const sortedCaptures = [...captures.values()].sort(
      (left, right) =>
        right.ts.localeCompare(left.ts) || left.captureId.localeCompare(right.captureId)
    );
    facts.sort(
      (left, right) =>
        right.ts.localeCompare(left.ts) ||
        left.namespace.localeCompare(right.namespace) ||
        left.clause.localeCompare(right.clause) ||
        left.id.localeCompare(right.id)
    );
    return { captures: sortedCaptures, facts };
  }

  pruneAutoCaptureFacts(
    selections: AutoCaptureFact[],
    options: PruneAutoCaptureOptions = {}
  ): { removed: number; opId: string } {
    const at = validDate(options.now ?? new Date(), 'prune timestamp');
    const opId = this.createOperationId();
    if (selections.length === 0) return { removed: 0, opId };

    return this.withMutationLock(() => {
    const entries = this.withLock('journal', () => this.readJournalUnlocked());
    const allowed = new Map<string, AutoCaptureFact>();
    for (const entry of entries) {
      if (
        entry.op !== 'assert' ||
        entry.origin !== 'claude-stop' ||
        typeof entry.captureId !== 'string' ||
        typeof entry.opId !== 'string' ||
        !Array.isArray(entry.added)
      ) {
        continue;
      }
      for (const serialized of entry.added) {
        if (typeof serialized !== 'string') continue;
        const [clause] = parseProgram(serialized);
        const canonical = serializeClause(clause);
        const id = autoCaptureFactId(entry.captureId, entry.namespace, canonical);
        allowed.set(`${id}\0${entry.opId}`, {
          id,
          captureId: entry.captureId,
          opId: entry.opId,
          namespace: entry.namespace,
          ts: entry.ts,
          clause: canonical,
          current: false,
        });
      }
    }

    const unique = new Map<string, AutoCaptureFact>();
    for (const selection of selections) {
      const key = `${selection.id}\0${selection.opId}`;
      const journalFact = allowed.get(key);
      if (
        journalFact === undefined ||
        journalFact.captureId !== selection.captureId ||
        journalFact.namespace !== selection.namespace ||
        journalFact.clause !== selection.clause
      ) {
        throw new Error(`auto-capture fact '${selection.id}' is not present in the journal`);
      }
      unique.set(key, journalFact);
    }

    let removed = 0;
    const byNamespace = new Map<string, AutoCaptureFact[]>();
    for (const selection of unique.values()) {
      const group = byNamespace.get(selection.namespace) ?? [];
      group.push(selection);
      byNamespace.set(selection.namespace, group);
    }
    if (options.integrity !== undefined || options.checks !== undefined) {
      if (byNamespace.size > 1) {
        throw new Error(
          'enforced auto-capture pruning accepts one namespace per operation'
        );
      }
      for (const [namespace, selected] of byNamespace) {
        const loaded = this.loadCached(namespace);
        const sources = this.sourcesFor([namespace]);
        const removableKeys = new Set(
          selected.flatMap((selection) => {
            const [clause] = parseProgram(selection.clause);
            const key = canonicalKey(clause);
            const currentSource = sources
              .get(key)
              ?.find((source) => source.namespace === namespace);
            return currentSource?.opId === selection.opId ? [key] : [];
          })
        );
        const candidate = loaded.clauses.filter(
          (clause) => !removableKeys.has(canonicalKey(clause))
        );
        this.enforceMutation(
          namespace,
          loaded.clauses,
          candidate,
          [],
          {
            opId,
            ...(options.integrity === undefined
              ? {}
              : { integrity: options.integrity }),
            ...(options.checks === undefined ? {} : { checks: options.checks }),
          },
          at
        );
      }
    }
    for (const [namespace, selected] of byNamespace) {
      let namespaceRemoved = 0;
      for (const selection of selected) {
        namespaceRemoved += this.retractFactIfSourcedBy(
          namespace,
          selection.clause,
          selection.opId,
          {
            opId,
            captureId: selection.captureId,
            at,
            integrity: undefined,
            checks: undefined,
          }
        );
      }
      removed += namespaceRemoved;
      this.note(
        namespace,
        'auto_capture_pruned',
        {
          opId,
          removed: namespaceRemoved,
          factIds: selected.map((selection) => selection.id).sort(),
          captureIds: [...new Set(selected.map((selection) => selection.captureId))].sort(),
        },
        at
      );
    }
    return { removed, opId };
    });
  }

  /** Replay the bounded append-only journal into a deterministic fact life story. */
  history(pattern: string, options: MemoryHistoryOptions = {}): MemoryHistory {
    const selector = parseFactPattern(pattern, 'history pattern');
    const requestedLimit = options.limit ?? MAX_HISTORY_EVENTS;
    if (
      !Number.isSafeInteger(requestedLimit) ||
      requestedLimit < 1 ||
      requestedLimit > MAX_HISTORY_EVENTS
    ) {
      throw new Error(`history limit must be an integer between 1 and ${MAX_HISTORY_EVENTS}`);
    }
    if (options.namespaces !== '*' && (options.namespaces?.length ?? 0) > 32) {
      throw new Error('history namespace list exceeds 32 entries');
    }

    return this.withLock('journal', () => {
      const journal = this.readJournalUnlocked();
      const names =
        options.namespaces === '*'
          ? [...new Set([
              ...this.listNamespaces(),
              ...journal.map((entry) => entry.namespace),
            ])].sort()
          : [...new Set(options.namespaces ?? ['default'])];
      for (const namespace of names) this.filePath(namespace);
      const selected = new Set(names);
      const state = new Map<
        string,
        { clause: Clause; opId: string; sequence: number; position: number }
      >();
      const lastTransition = new Map<string, string>();
      const events: MemoryHistoryEvent[] = [];
      const stateKey = (namespace: string, clause: Clause) =>
        `${namespace}\u0000${canonicalKey(clause)}`;
      const pushEvent = (event: MemoryHistoryEvent) => {
        events.push(event);
        if (events.length > requestedLimit) {
          throw new Error(`history result exceeds ${requestedLimit} events`);
        }
      };

      for (const [lineIndex, journalEntry] of journal.entries()) {
        const sequence = lineIndex + 1;
        const label = `journal.log line ${sequence}`;
        assertIsoTimestamp(journalEntry.ts, `${label} timestamp`);
        if (!NAMESPACE_RE.test(journalEntry.namespace)) {
          throw new Error(`${label} has an invalid namespace`);
        }
        const inScope = selected.has(journalEntry.namespace);
        if (!['assert', 'retract', 'supersede', 'memory_change'].includes(journalEntry.op)) continue;
        if (typeof journalEntry.opId !== 'string' || journalEntry.opId.length === 0) {
          throw new Error(`${label} has no opId`);
        }
        const source = historySourceFields(journalEntry);

        if (journalEntry.op === 'assert') {
          const added = parseJournalClauseList(journalEntry.added, `${label} added`);
          for (const [position, clause] of added.entries()) {
            if (inScope) {
              const key = stateKey(journalEntry.namespace, clause);
              const previousSourceOpId = lastTransition.get(key);
              state.set(key, {
                clause,
                opId: journalEntry.opId,
                sequence,
                position,
              });
              lastTransition.set(key, journalEntry.opId);
              if (clause.body.length === 0 && literalMatches(selector, clause.head)) {
                pushEvent({
                  sequence,
                  position,
                  namespace: journalEntry.namespace,
                  ts: journalEntry.ts,
                  opId: journalEntry.opId,
                  action: 'asserted',
                  clause: serializeClause(clause),
                  current: false,
                  ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                  ...source,
                });
              }
            }
          }
          continue;
        }

        if (journalEntry.op === 'memory_change') {
          const removedClauses = parseJournalClauseList(
            journalEntry.removedClauses,
            `${label} removedClauses`
          );
          const addedClauses = parseJournalClauseList(
            journalEntry.addedClauses,
            `${label} addedClauses`
          );
          const temporal = memoryChangeTemporalSources(journalEntry, label);
          const temporalByPrevious = new Map(
            [...temporal].map(([toKey, value]) => [
              canonicalKey(parseJournalClause(value.previousClause, `${label} previousClause`)),
              { toKey, value },
            ])
          );
          for (const [position, clause] of removedClauses.entries()) {
            if (!inScope) continue;
            const key = stateKey(journalEntry.namespace, clause);
            const previousSourceOpId = state.get(key)?.opId ?? lastTransition.get(key);
            state.delete(key);
            lastTransition.set(key, journalEntry.opId);
            if (clause.body.length === 0 && literalMatches(selector, clause.head)) {
              const archivedValue = temporalByPrevious.get(canonicalKey(clause));
              const archivedClause = archivedValue === undefined
                ? undefined
                : addedClauses.find(
                    (candidate) => canonicalKey(candidate) === archivedValue.toKey
                  );
              pushEvent({
                sequence,
                position,
                namespace: journalEntry.namespace,
                ts: journalEntry.ts,
                opId: journalEntry.opId,
                action: archivedClause === undefined ? 'retracted' : 'superseded',
                clause: serializeClause(clause),
                current: false,
                ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                ...(archivedClause === undefined
                  ? {}
                  : {
                      archivedAs: serializeClause(archivedClause),
                      validUntil: archivedValue!.value.validUntil,
                    }),
                ...source,
              });
            }
          }
          for (const [position, clause] of addedClauses.entries()) {
            if (!inScope) continue;
            const key = stateKey(journalEntry.namespace, clause);
            const previousSourceOpId = lastTransition.get(key);
            state.set(key, {
              clause,
              opId: journalEntry.opId,
              sequence,
              position: removedClauses.length + position,
            });
            lastTransition.set(key, journalEntry.opId);
            if (clause.body.length === 0 && literalMatches(selector, clause.head)) {
              pushEvent({
                sequence,
                position: removedClauses.length + position,
                namespace: journalEntry.namespace,
                ts: journalEntry.ts,
                opId: journalEntry.opId,
                action: 'asserted',
                clause: serializeClause(clause),
                current: false,
                ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                ...source,
              });
            }
          }
          continue;
        }

        if (journalEntry.op === 'retract') {
          if (typeof journalEntry.pattern !== 'string') throw new Error(`${label} has no pattern`);
          if (
            typeof journalEntry.removed !== 'number' ||
            !Number.isSafeInteger(journalEntry.removed) ||
            journalEntry.removed < 0
          ) {
            throw new Error(`${label} has an invalid removed count`);
          }
          const retractionTarget = parseRetractionTarget(
            journalEntry.pattern,
            `${label} pattern`
          );
          const retractionLiteral = 'literal' in retractionTarget
            ? retractionTarget.literal
            : undefined;
          const retractionRuleKey = 'clauseKey' in retractionTarget
            ? retractionTarget.clauseKey
            : undefined;
          let removedClauses: Clause[];
          if (Array.isArray(journalEntry.removedClauses)) {
            removedClauses = parseJournalClauseList(
              journalEntry.removedClauses,
              `${label} removedClauses`
            );
            if (removedClauses.length !== journalEntry.removed) {
              throw new Error(`${label} removedClauses does not match removed count`);
            }
            if (
              removedClauses.some((clause) =>
                retractionRuleKey === undefined
                  ? clause.body.length !== 0 ||
                    retractionLiteral === undefined ||
                    !literalMatches(retractionLiteral, clause.head)
                  : canonicalKey(clause) !== retractionRuleKey
              )
            ) {
              throw new Error(`${label} removedClauses do not match the retraction pattern`);
            }
          } else if (!inScope) {
            removedClauses = [];
          } else if (retractionRuleKey !== undefined) {
            removedClauses = [...state.values()]
              .filter((value) => canonicalKey(value.clause) === retractionRuleKey)
              .map((value) => value.clause);
          } else {
            removedClauses = [...state.values()]
              .filter(
                (value) =>
                  value.clause.body.length === 0 &&
                  retractionLiteral !== undefined &&
                  literalMatches(retractionLiteral, value.clause.head)
              )
              .map((value) => value.clause)
              .sort((left, right) => serializeClause(left).localeCompare(serializeClause(right)));
          }
          for (const [position, clause] of removedClauses.entries()) {
            if (!inScope) continue;
            const key = stateKey(journalEntry.namespace, clause);
            const previousSourceOpId = state.get(key)?.opId ?? lastTransition.get(key);
            state.delete(key);
            lastTransition.set(key, journalEntry.opId);
            if (clause.body.length === 0 && literalMatches(selector, clause.head)) {
              pushEvent({
                sequence,
                position,
                namespace: journalEntry.namespace,
                ts: journalEntry.ts,
                opId: journalEntry.opId,
                action: 'retracted',
                clause: serializeClause(clause),
                current: false,
                ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                ...source,
              });
            }
          }
          continue;
        }

        if (
          !Array.isArray(journalEntry.patterns) ||
          journalEntry.patterns.length === 0 ||
          journalEntry.patterns.length > 64 ||
          !journalEntry.patterns.every((value) => typeof value === 'string')
        ) {
          throw new Error(`${label} patterns must be a non-empty string array`);
        }
        const supersedePatterns = journalEntry.patterns.map((value, patternIndex) =>
          parseFactPattern(value as string, `${label} patterns[${patternIndex}]`)
        );
        if (!Array.isArray(journalEntry.ended)) throw new Error(`${label} ended must be an array`);
        if (!Array.isArray(journalEntry.archived)) {
          throw new Error(`${label} archived must be an array`);
        }
        const validTimeMode = journalEntry.validTimeMode ?? 'archive_until';
        if (validTimeMode !== 'delete' && validTimeMode !== 'archive_until') {
          throw new Error(`${label} has an invalid valid-time mode`);
        }
        if (validTimeMode === 'delete' && journalEntry.archived.length !== 0) {
          throw new Error(`${label} delete replacement cannot contain archives`);
        }
        const trustAction = journalEntry.trustAction;
        if (
          trustAction !== undefined &&
          trustAction !== 'accept' &&
          trustAction !== 'reject'
        ) {
          throw new Error(`${label} has an invalid trust action`);
        }
        const archives = new Map<
          string,
          { to: Clause; validUntil: string }
        >();
        for (const [archiveIndex, archive] of journalEntry.archived.entries()) {
          if (typeof archive !== 'object' || archive === null || Array.isArray(archive)) {
            throw new Error(`${label} archived[${archiveIndex}] must be an object`);
          }
          const record = archive as Record<string, unknown>;
          const from = parseJournalClause(record.from, `${label} archived[${archiveIndex}].from`);
          const to = parseJournalClause(record.to, `${label} archived[${archiveIndex}].to`);
          assertIsoTimestamp(record.validUntil, `${label} archived[${archiveIndex}].validUntil`);
          if (record.validUntil !== journalEntry.ts) {
            throw new Error(`${label} archived validUntil must match the event timestamp`);
          }
          const expected = archiveUntilClause(from, record.validUntil);
          if (canonicalKey(expected) !== canonicalKey(to)) {
            throw new Error(`${label} has an inconsistent archived clause`);
          }
          archives.set(canonicalKey(from), { to, validUntil: record.validUntil });
        }

        const ended: Array<{ clause: Clause; sourceOpId?: string }> = [];
        for (const [endedIndex, endedValue] of journalEntry.ended.entries()) {
          if (
            typeof endedValue !== 'object' ||
            endedValue === null ||
            Array.isArray(endedValue)
          ) {
            throw new Error(`${label} ended[${endedIndex}] must be an object`);
          }
          const record = endedValue as Record<string, unknown>;
          const clause = parseJournalClause(record.clause, `${label} ended[${endedIndex}].clause`);
          if (record.sourceOpId !== undefined && typeof record.sourceOpId !== 'string') {
            throw new Error(`${label} ended[${endedIndex}].sourceOpId must be a string`);
          }
          if (
            validTimeMode === 'archive_until' &&
            !archives.has(canonicalKey(clause))
          ) {
            throw new Error(`${label} ended fact has no archived counterpart`);
          }
          if (
            clause.body.length !== 0 ||
            !supersedePatterns.some((pattern) => literalMatches(pattern, clause.head))
          ) {
            throw new Error(`${label} ended fact does not match a supersede pattern`);
          }
          ended.push({
            clause,
            ...(record.sourceOpId === undefined ? {} : { sourceOpId: record.sourceOpId }),
          });
        }
        if (
          validTimeMode === 'archive_until' &&
          archives.size !== ended.length
        ) {
          throw new Error(`${label} archived facts do not match ended facts`);
        }

        for (const [position, endedFact] of ended.entries()) {
          const archive = archives.get(canonicalKey(endedFact.clause));
          if (validTimeMode === 'archive_until' && archive === undefined) {
            throw new Error(`${label} has an incomplete archive mapping`);
          }
          if (inScope) {
            const key = stateKey(journalEntry.namespace, endedFact.clause);
            const activeSourceOpId = state.get(key)?.opId;
            if (
              endedFact.sourceOpId !== undefined &&
              activeSourceOpId !== undefined &&
              endedFact.sourceOpId !== activeSourceOpId
            ) {
              throw new Error(`${label} ended fact has inconsistent source lineage`);
            }
            const previousSourceOpId =
              endedFact.sourceOpId ?? state.get(key)?.opId ?? lastTransition.get(key);
            state.delete(key);
            lastTransition.set(key, journalEntry.opId);
            if (
              endedFact.clause.body.length === 0 &&
              literalMatches(selector, endedFact.clause.head)
            ) {
              pushEvent({
                sequence,
                position,
                namespace: journalEntry.namespace,
                ts: journalEntry.ts,
                opId: journalEntry.opId,
                action: validTimeMode === 'archive_until' ? 'superseded' : 'retracted',
                clause: serializeClause(endedFact.clause),
                current: false,
                ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                ...(archive === undefined
                  ? {}
                  : {
                      archivedAs: serializeClause(archive.to),
                      validUntil: archive.validUntil,
                    }),
                ...(trustAction === undefined ? {} : { trustAction }),
                ...source,
              });
            }
          }
        }

        const added = parseJournalClauseList(journalEntry.added, `${label} added`);
        const replacementRequested = parseJournalClauseList(
          journalEntry.replacementRequested,
          `${label} replacementRequested`
        );
        const replacementAdded = parseJournalClauseList(
          journalEntry.replacementAdded,
          `${label} replacementAdded`
        );
        if (
          typeof journalEntry.duplicates !== 'number' ||
          !Number.isSafeInteger(journalEntry.duplicates) ||
          journalEntry.duplicates < 0 ||
          replacementAdded.length + journalEntry.duplicates !== replacementRequested.length
        ) {
          throw new Error(`${label} has inconsistent replacement counts`);
        }
        const allowedAdded = new Set([
          ...replacementAdded.map(canonicalKey),
          ...[...archives.values()].map((archive) => canonicalKey(archive.to)),
        ]);
        const addedKeys = new Set(added.map(canonicalKey));
        if (
          added.some((clause) => !allowedAdded.has(canonicalKey(clause))) ||
          replacementAdded.some((clause) => !addedKeys.has(canonicalKey(clause)))
        ) {
          throw new Error(`${label} has inconsistent added facts`);
        }
        for (const [addedIndex, clause] of added.entries()) {
          const position = ended.length + addedIndex;
          if (inScope) {
            const key = stateKey(journalEntry.namespace, clause);
            const previousSourceOpId = lastTransition.get(key);
            state.set(key, {
              clause,
              opId: journalEntry.opId,
              sequence,
              position,
            });
            lastTransition.set(key, journalEntry.opId);
            if (clause.body.length === 0 && literalMatches(selector, clause.head)) {
              pushEvent({
                sequence,
                position,
                namespace: journalEntry.namespace,
                ts: journalEntry.ts,
                opId: journalEntry.opId,
                action: 'asserted',
                clause: serializeClause(clause),
                current: false,
                ...(previousSourceOpId === undefined ? {} : { previousSourceOpId }),
                ...(trustAction === undefined ? {} : { trustAction }),
                ...source,
              });
            }
          }
        }
      }

      const currentKeys = new Set<string>();
      for (const namespace of names) {
        for (const clause of this.load(namespace)) {
          currentKeys.add(stateKey(namespace, clause));
        }
      }
      for (const event of events) {
        if (event.action !== 'asserted') continue;
        const clause = parseJournalClause(event.clause, 'history event clause');
        const key = stateKey(event.namespace, clause);
        const active = state.get(key);
        event.current =
          currentKeys.has(key) &&
          active?.sequence === event.sequence &&
          active.position === event.position &&
          active.opId === event.opId;
      }

      return {
        pattern: serializeGoal(selector),
        namespaces: names,
        events,
      };
    });
  }

  /** Capture the exact current recorded state under the same mutation/journal boundary. */
  recordedHead(namespaces: string[] | '*'): RecordedKnowledgeSnapshot {
    return this.withMutationLock(() =>
      this.withLock('journal', () => {
        const sequence = this.readJournalUnlocked().length;
        return this.recordedSnapshot(namespaces, sequence);
      })
    );
  }

  /**
   * Reconstruct a deterministic read-only knowledge view at a global journal position.
   * The current files must exactly reconcile with the complete journal before any
   * historical view is returned; otherwise the journal cannot prove completeness.
   */
  recordedSnapshot(
    namespaces: string[] | '*',
    sequence: number
  ): RecordedKnowledgeSnapshot {
    if (!Number.isSafeInteger(sequence) || sequence < 0) {
      throw new Error('recorded snapshot sequence must be a non-negative safe integer');
    }

    return this.withLock('journal', () => {
      const journal = this.readJournalUnlocked();
      if (sequence > journal.length) {
        throw new Error(
          `recorded snapshot sequence ${sequence} exceeds journal length ${journal.length}`
        );
      }
      const names = namespaces === '*'
        ? [...new Set([
            ...this.listNamespaces(),
            ...journal.map((entry) => entry.namespace),
          ])].sort()
        : [...new Set(namespaces)];
      if (names.length > 32) throw new Error('recorded snapshot namespace list exceeds 32 entries');
      for (const namespace of names) this.filePath(namespace);
      const selected = new Set(names);
      const namespaceOrder = new Map(names.map((name, index) => [name, index]));
      const state = new Map<
        string,
        { namespace: string; clause: Clause; source: MemorySource }
      >();
      let captured: typeof state | undefined;
      const stateKey = (namespace: string, clause: Clause) =>
        `${namespace}\u0000${canonicalKey(clause)}`;
      const capture = () => {
        captured = new Map(
          [...state].map(([key, value]) => [
            key,
            { ...value, source: { ...value.source } },
          ])
        );
      };
      if (sequence === 0) capture();

      for (const [lineIndex, entry] of journal.entries()) {
        const currentSequence = lineIndex + 1;
        const label = `journal.log line ${currentSequence}`;
        assertIsoTimestamp(entry.ts, `${label} timestamp`);
        if (!NAMESPACE_RE.test(entry.namespace)) {
          throw new Error(`${label} has an invalid namespace`);
        }
        const isMutation =
          entry.op === 'assert' ||
          entry.op === 'retract' ||
          entry.op === 'supersede' ||
          entry.op === 'rule_change' ||
          entry.op === 'memory_change';
        if (isMutation) {
          if (typeof entry.opId !== 'string' || entry.opId.length === 0) {
            throw new Error(`${label} has no opId`);
          }
        }
        if (
          selected.has(entry.namespace) &&
          !isMutation &&
          !RECORDED_AUDIT_OPERATIONS.has(entry.op)
        ) {
          throw new Error(`${label} has unsupported operation '${entry.op}'`);
        }
        if (!selected.has(entry.namespace)) {
          if (currentSequence === sequence) capture();
          continue;
        }

        if (entry.op === 'assert') {
          for (const clause of parseJournalClauseList(entry.added, `${label} added`)) {
            state.set(stateKey(entry.namespace, clause), {
              namespace: entry.namespace,
              clause,
              source: journalMemorySource(entry),
            });
          }
        } else if (entry.op === 'memory_change') {
          validateSha256(
            typeof entry.baselineDigest === 'string' ? entry.baselineDigest : '',
            `${label} baselineDigest`
          );
          validateSha256(
            typeof entry.proposalDigest === 'string' ? entry.proposalDigest : '',
            `${label} proposalDigest`
          );
          if (
            !Array.isArray(entry.namespaces) ||
            !entry.namespaces.every((value) => typeof value === 'string') ||
            !entry.namespaces.includes(entry.namespace)
          ) {
            throw new Error(`${label} memory change has invalid namespaces`);
          }
          const removed = parseJournalClauseList(
            entry.removedClauses,
            `${label} removedClauses`
          );
          const added = parseJournalClauseList(
            entry.addedClauses,
            `${label} addedClauses`
          );
          if (
            removed.length > MAX_MEMORY_CHANGE_CLAUSES ||
            added.length > MAX_MEMORY_CHANGE_CLAUSES
          ) {
            throw new Error(`${label} memory change exceeds clause limits`);
          }
          for (const clause of removed) {
            const key = stateKey(entry.namespace, clause);
            if (!state.has(key)) throw new Error(`${label} removed clause is not active`);
            state.delete(key);
          }
          const temporalByKey = memoryChangeTemporalSources(entry, label);
          for (const clause of added) {
            const key = stateKey(entry.namespace, clause);
            if (state.has(key)) throw new Error(`${label} added clause is already active`);
            state.set(key, {
              namespace: entry.namespace,
              clause,
              source: journalMemorySource(
                entry,
                temporalByKey.get(canonicalKey(clause))
              ),
            });
          }
        } else if (entry.op === 'rule_change') {
          validateSha256(
            typeof entry.baselineDigest === 'string' ? entry.baselineDigest : '',
            `${label} baselineDigest`
          );
          validateSha256(
            typeof entry.proposalDigest === 'string' ? entry.proposalDigest : '',
            `${label} proposalDigest`
          );
          if (
            !Array.isArray(entry.namespaces) ||
            !entry.namespaces.every((value) => typeof value === 'string') ||
            !entry.namespaces.includes(entry.namespace)
          ) {
            throw new Error(`${label} rule change has invalid namespaces`);
          }
          const removedRules = parseJournalClauseList(
            entry.removedRules,
            `${label} removedRules`
          );
          const addedRules = parseJournalClauseList(
            entry.addedRules,
            `${label} addedRules`
          );
          if (
            removedRules.length > MAX_RULE_CHANGE_RULES ||
            addedRules.length > MAX_RULE_CHANGE_RULES ||
            removedRules.some(
              (clause) => clause.body.length === 0 || isIntegrityConstraint(clause)
            ) ||
            addedRules.some(
              (clause) => clause.body.length === 0 || isIntegrityConstraint(clause)
            )
          ) {
            throw new Error(`${label} rule change contains a non-rule clause`);
          }
          for (const clause of removedRules) {
            const key = stateKey(entry.namespace, clause);
            if (!state.has(key)) {
              throw new Error(`${label} removed rule is not active`);
            }
            state.delete(key);
          }
          for (const clause of addedRules) {
            const key = stateKey(entry.namespace, clause);
            if (state.has(key)) {
              throw new Error(`${label} added rule is already active`);
            }
            state.set(key, {
              namespace: entry.namespace,
              clause,
              source: journalMemorySource(entry),
            });
          }
        } else if (entry.op === 'retract') {
          if (typeof entry.pattern !== 'string') throw new Error(`${label} has no pattern`);
          if (
            typeof entry.removed !== 'number' ||
            !Number.isSafeInteger(entry.removed) ||
            entry.removed < 0
          ) {
            throw new Error(`${label} has an invalid removed count`);
          }
          const target = parseRetractionTarget(entry.pattern, `${label} pattern`);
          let removed: Clause[];
          if (Array.isArray(entry.removedClauses)) {
            removed = parseJournalClauseList(entry.removedClauses, `${label} removedClauses`);
            if (removed.length !== entry.removed) {
              throw new Error(`${label} removedClauses does not match removed count`);
            }
            if (
              removed.some((clause) =>
                'clauseKey' in target
                  ? canonicalKey(clause) !== target.clauseKey
                  : clause.body.length !== 0 || !literalMatches(target.literal, clause.head)
              )
            ) {
              throw new Error(`${label} removedClauses do not match the retraction pattern`);
            }
          } else {
            removed = [...state.values()]
              .filter(({ namespace, clause }) => {
                if (namespace !== entry.namespace) return false;
                return 'clauseKey' in target
                  ? canonicalKey(clause) === target.clauseKey
                  : clause.body.length === 0 && literalMatches(target.literal, clause.head);
              })
              .map(({ clause }) => clause);
          }
          for (const clause of removed) state.delete(stateKey(entry.namespace, clause));
        } else if (entry.op === 'supersede') {
          if (
            !Array.isArray(entry.patterns) ||
            entry.patterns.length === 0 ||
            entry.patterns.length > 64 ||
            !entry.patterns.every((value) => typeof value === 'string')
          ) {
            throw new Error(`${label} patterns must be a non-empty string array`);
          }
          const patterns = entry.patterns.map((value, index) =>
            parseFactPattern(value as string, `${label} patterns[${index}]`)
          );
          const validTimeMode = entry.validTimeMode ?? 'archive_until';
          if (validTimeMode !== 'delete' && validTimeMode !== 'archive_until') {
            throw new Error(`${label} has an invalid valid-time mode`);
          }
          if (
            entry.trustAction !== undefined &&
            entry.trustAction !== 'accept' &&
            entry.trustAction !== 'reject'
          ) {
            throw new Error(`${label} has an invalid trust action`);
          }
          if (!Array.isArray(entry.ended)) throw new Error(`${label} ended must be an array`);
          const temporalByKey = new Map<string, TemporalMemorySource>();
          if (!Array.isArray(entry.archived)) throw new Error(`${label} archived must be an array`);
          if (validTimeMode === 'delete' && entry.archived.length !== 0) {
            throw new Error(`${label} delete replacement cannot contain archives`);
          }
          const archivedFrom = new Set<string>();
          for (const [archiveIndex, archived] of entry.archived.entries()) {
            if (typeof archived !== 'object' || archived === null || Array.isArray(archived)) {
              throw new Error(`${label} archived[${archiveIndex}] must be an object`);
            }
            const record = archived as Record<string, unknown>;
            const previous = parseJournalClause(
              record.from,
              `${label} archived[${archiveIndex}].from`
            );
            const archivedClause = parseJournalClause(
              record.to,
              `${label} archived[${archiveIndex}].to`
            );
            assertIsoTimestamp(
              record.validUntil,
              `${label} archived[${archiveIndex}].validUntil`
            );
            if (record.validUntil !== entry.ts) {
              throw new Error(`${label} archived validUntil must match the event timestamp`);
            }
            if (
              canonicalKey(archiveUntilClause(previous, record.validUntil)) !==
              canonicalKey(archivedClause)
            ) {
              throw new Error(`${label} has an inconsistent archived clause`);
            }
            temporalByKey.set(canonicalKey(archivedClause), {
              kind: 'superseded',
              previousClause: serializeClause(previous),
              validUntil: record.validUntil,
            });
            archivedFrom.add(canonicalKey(previous));
          }
          if (validTimeMode === 'archive_until' && archivedFrom.size !== entry.ended.length) {
            throw new Error(`${label} archived facts do not match ended facts`);
          }
          for (const [endedIndex, ended] of entry.ended.entries()) {
            if (typeof ended !== 'object' || ended === null || Array.isArray(ended)) {
              throw new Error(`${label} ended[${endedIndex}] must be an object`);
            }
            const record = ended as Record<string, unknown>;
            const clause = parseJournalClause(
              record.clause,
              `${label} ended[${endedIndex}].clause`
            );
            if (
              clause.body.length !== 0 ||
              !patterns.some((pattern) => literalMatches(pattern, clause.head))
            ) {
              throw new Error(`${label} ended fact does not match a supersede pattern`);
            }
            if (validTimeMode === 'archive_until' && !archivedFrom.has(canonicalKey(clause))) {
              throw new Error(`${label} ended fact has no archived counterpart`);
            }
            if (record.sourceOpId !== undefined && typeof record.sourceOpId !== 'string') {
              throw new Error(`${label} ended[${endedIndex}].sourceOpId must be a string`);
            }
            const active = state.get(stateKey(entry.namespace, clause));
            if (
              typeof record.sourceOpId === 'string' &&
              active !== undefined &&
              active.source.opId !== record.sourceOpId
            ) {
              throw new Error(`${label} ended fact has inconsistent source lineage`);
            }
            state.delete(stateKey(entry.namespace, clause));
          }
          const added = parseJournalClauseList(entry.added, `${label} added`);
          const requested = parseJournalClauseList(
            entry.replacementRequested,
            `${label} replacementRequested`
          );
          const replacementAdded = parseJournalClauseList(
            entry.replacementAdded,
            `${label} replacementAdded`
          );
          if (
            typeof entry.duplicates !== 'number' ||
            !Number.isSafeInteger(entry.duplicates) ||
            entry.duplicates < 0 ||
            replacementAdded.length + entry.duplicates !== requested.length
          ) {
            throw new Error(`${label} has inconsistent replacement counts`);
          }
          const allowedAdded = new Set([
            ...replacementAdded.map(canonicalKey),
            ...temporalByKey.keys(),
          ]);
          const addedKeys = new Set(added.map(canonicalKey));
          if (
            added.some((clause) => !allowedAdded.has(canonicalKey(clause))) ||
            replacementAdded.some((clause) => !addedKeys.has(canonicalKey(clause)))
          ) {
            throw new Error(`${label} has inconsistent added facts`);
          }
          for (const clause of added) {
            state.set(stateKey(entry.namespace, clause), {
              namespace: entry.namespace,
              clause,
              source: journalMemorySource(entry, temporalByKey.get(canonicalKey(clause))),
            });
          }
        }
        if (currentSequence === sequence) capture();
      }

      const drifted: string[] = [];
      for (const namespace of names) {
        const journalKeys = new Set(
          [...state.values()]
            .filter((value) => value.namespace === namespace)
            .map((value) => canonicalKey(value.clause))
        );
        const fileKeys = new Set(this.load(namespace).map(canonicalKey));
        if (
          journalKeys.size !== fileKeys.size ||
          [...journalKeys].some((key) => !fileKeys.has(key))
        ) {
          drifted.push(namespace);
        }
      }
      if (drifted.length > 0) throw new IncompleteHistoryError(drifted);

      const snapshot = captured ?? new Map();
      const values = [...snapshot.values()].sort(
        (left, right) =>
          (namespaceOrder.get(left.namespace) ?? Number.MAX_SAFE_INTEGER) -
            (namespaceOrder.get(right.namespace) ?? Number.MAX_SAFE_INTEGER) ||
          serializeClause(left.clause).localeCompare(serializeClause(right.clause))
      );
      const sources = new Map<string, MemorySource[]>();
      for (const value of values) {
        const key = canonicalKey(value.clause);
        const grouped = sources.get(key) ?? [];
        grouped.push(value.source);
        sources.set(key, grouped);
      }
      return {
        sequence,
        journalEntries: journal.length,
        namespaces: names,
        clauses: values.map((value) => value.clause),
        sources,
      };
    });
  }

  /** Capture several exact recorded positions under one coherent journal/file boundary. */
  recordedSnapshots(
    namespaces: string[] | '*',
    sequences: number[]
  ): RecordedKnowledgeSnapshot[] {
    if (sequences.length > MAX_RECORDED_SNAPSHOT_BATCH) {
      throw new Error(
        `recorded snapshot batch exceeds ${MAX_RECORDED_SNAPSHOT_BATCH} positions`
      );
    }
    return this.withMutationLock(() =>
      this.withLock('journal', () =>
        sequences.map((sequence) => this.recordedSnapshot(namespaces, sequence))
      )
    );
  }

  listNamespaces(): string[] {
    let files: string[];
    try {
      files = readdirSync(this.root);
    } catch {
      return [];
    }
    return files
      .filter((f) => f.endsWith('.dl'))
      .map((f) => f.slice(0, -3))
      .sort();
  }

  clausesFor(namespaces: string[] | '*'): Clause[] {
    const names = namespaces === '*' ? this.listNamespaces() : namespaces;
    return names.flatMap((ns) => this.load(ns));
  }

  /** One current clause/source view serialized against every supported store writer. */
  knowledgeSnapshot(namespaces: string[] | '*'): CurrentKnowledgeSnapshot {
    return this.withMutationLock(() => {
      this.withLock('journal', () => this.recoverPendingMutationUnlocked());
      const names = namespaces === '*' ? this.listNamespaces() : [...namespaces];
      if (names.length > 32) {
        throw new Error('knowledge snapshot namespace list exceeds 32 entries');
      }
      const clausesByNamespace = new Map<string, Clause[]>();
      for (const namespace of names) {
        this.filePath(namespace);
        clausesByNamespace.set(namespace, structuredClone(this.load(namespace)));
      }
      const clauses = names.flatMap(
        (namespace) => clausesByNamespace.get(namespace) ?? []
      );
      const sources = new Map(
        [...this.sourcesFor(names)].map(([key, values]) => [
          key,
          values.map((source) => structuredClone(source)),
        ])
      );
      return { namespaces: names, clauses, clausesByNamespace, sources };
    });
  }

  /** Latest durable assertion source for every currently stored clause. */
  sourcesFor(namespaces: string[] | '*'): Map<string, MemorySource[]> {
    const names = namespaces === '*' ? this.listNamespaces() : [...namespaces];
    const namespaceOrder = new Map(names.map((name, index) => [name, index]));
    const selected = new Set(names);
    const current = new Set<string>();
    for (const namespace of names) {
      for (const clause of this.load(namespace)) {
        current.add(`${namespace}\u0000${canonicalKey(clause)}`);
      }
    }

    const entries = this.withLock('journal', () => this.readJournalUnlocked());
    const latest = new Map<string, { key: string; source: MemorySource }>();
    for (const [index, entry] of entries.entries()) {
      if (
        entry.op !== 'assert' &&
        entry.op !== 'supersede' &&
        entry.op !== 'rule_change' &&
        entry.op !== 'memory_change'
      ) continue;
      if (!selected.has(entry.namespace)) continue;
      const label = `journal.log line ${index + 1}`;
      if (typeof entry.opId !== 'string') throw new Error(`${label} has no opId`);
      assertIsoTimestamp(entry.ts, `${label} timestamp`);
      const temporalByClause = new Map<string, TemporalMemorySource>();
      if (entry.op === 'supersede') {
        if (!Array.isArray(entry.archived)) throw new Error(`${label} archived must be an array`);
        for (const [archiveIndex, archived] of entry.archived.entries()) {
          if (typeof archived !== 'object' || archived === null || Array.isArray(archived)) {
            throw new Error(`${label} archived[${archiveIndex}] must be an object`);
          }
          const record = archived as Record<string, unknown>;
          const previous = parseJournalClause(
            record.from,
            `${label} archived[${archiveIndex}].from`
          );
          const archivedClause = parseJournalClause(
            record.to,
            `${label} archived[${archiveIndex}].to`
          );
          assertIsoTimestamp(
            record.validUntil,
            `${label} archived[${archiveIndex}].validUntil`
          );
          if (record.validUntil !== entry.ts) {
            throw new Error(`${label} archived validUntil must match the event timestamp`);
          }
          if (
            canonicalKey(archiveUntilClause(previous, record.validUntil)) !==
            canonicalKey(archivedClause)
          ) {
            throw new Error(`${label} has an inconsistent archived clause`);
          }
          temporalByClause.set(canonicalKey(archivedClause), {
            kind: 'superseded',
            previousClause: serializeClause(previous),
            validUntil: record.validUntil,
          });
        }
      } else if (entry.op === 'memory_change') {
        for (const [key, temporal] of memoryChangeTemporalSources(entry, label)) {
          temporalByClause.set(key, temporal);
        }
      }
      const addedField = entry.op === 'rule_change'
        ? entry.addedRules
        : entry.op === 'memory_change'
          ? entry.addedClauses
          : entry.added;
      const addedLabel = entry.op === 'rule_change'
        ? 'addedRules'
        : entry.op === 'memory_change'
          ? 'addedClauses'
          : 'added';
      for (const clause of parseJournalClauseList(
        addedField,
        `${label} ${addedLabel}`
      )) {
        const key = canonicalKey(clause);
        const currentKey = `${entry.namespace}\u0000${key}`;
        if (!current.has(currentKey)) continue;
        latest.set(currentKey, {
          key,
          source: journalMemorySource(entry, temporalByClause.get(key)),
        });
      }
    }

    const result = new Map<string, MemorySource[]>();
    for (const { key, source } of latest.values()) {
      const sources = result.get(key) ?? [];
      sources.push(source);
      result.set(key, sources);
    }
    for (const sources of result.values()) {
      sources.sort((left, right) =>
        (namespaceOrder.get(left.namespace) ?? Number.MAX_SAFE_INTEGER) -
          (namespaceOrder.get(right.namespace) ?? Number.MAX_SAFE_INTEGER) ||
        left.opId.localeCompare(right.opId)
      );
    }
    return result;
  }
}
