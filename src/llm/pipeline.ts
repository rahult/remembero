import {
  type Bindings,
  type AggregateOperator,
  type Clause,
  type Goal,
  type Literal,
  type QuerySpec,
  type ScalarExpression,
  type Term,
  EngineLimitError,
  evaluateQuerySpec,
  isComparison,
  isArithmeticExpression,
  isAggregateRule,
  isIntegrityConstraint,
  isNegation,
  parseProgram,
  parseQuery,
  parseQuerySpec,
  predKey,
  serializeClause,
  serializeGoal,
  serializeQuerySpec,
  serializeTerm,
} from '../engine/index.js';
import type {
  MemoryStore,
  MemorySource,
  RecordedSnapshotMetadata,
  ValidTimeMode,
} from '../store/store.js';
import type { ChatMessage, LlmClient } from './client.js';
import type { EmbeddingClient } from './embeddings.js';
import type { EmbeddingCache } from '../knowledge/semantic-search.js';
import type { SemanticLedger } from '../ledger/semantic-ledger.js';
import {
  explainKnowledge,
  type ExplainKnowledgeResult,
  type SourcedQueryProof,
} from '../knowledge/graph.js';
import {
  explainWhyNot,
  type ExplainWhyNotResult,
} from '../knowledge/why-not.js';
import type { IntegrityEnforcementOptions } from '../knowledge/enforcement.js';
import type { KnowledgeCheckEnforcementOptions } from '../knowledge/check-enforcement.js';
import {
  canonicalizeKnowledge,
  isEntityMetadataDeclaration,
  isEntityMetadataPredicate,
  literalKnowledge,
  type EntityIdentityMode,
} from '../knowledge/identity.js';
import type { ExplanationGraphSelector } from '../knowledge/graph-navigation.js';
import { assertTentativeFacts } from '../knowledge/trust-store.js';
import {
  isTentativeDeclaration,
  isTrustMetadataPredicate,
  type KnowledgeTrust,
  type TrustViewMode,
} from '../knowledge/trust.js';
import {
  searchKnowledge,
  type KnowledgeSearchClauseKind,
  type KnowledgeSearchResult,
} from '../knowledge/search.js';
import { assertBoundedOutput, assertSafeForExternalLlm } from '../safety.js';
import {
  NOTHING_SENTINEL,
  PHRASING_SYSTEM_PROMPT,
  UNANSWERABLE,
  answeredQueryReviewPrompt,
  buildSchemaSummary,
  extractionSystemPrompt,
  phrasingUserPrompt,
  queryGenSystemPrompt,
  transcriptExtractionSystemPrompt,
  type QueryPromptVariant,
} from './prompts.js';
import {
  type RecallSchemaDiagnostics,
  type RecallSchemaSelection,
  MAX_RECALL_SCHEMA_PREDICATES,
  RecallSchemaBudgetError,
  recallEditDistance,
  recallSchemaDiagnostics,
  recallWords,
  selectRecallSchema,
} from './schema.js';

export interface PipelineDeps {
  store: MemoryStore;
  llm: LlmClient;
  /** Optional SQLite semantic version authority for review/promotion tools. */
  semanticLedger?: SemanticLedger;
  /** Optional semantic-retrieval provider; structured reasoning never requires it. */
  embeddings?: EmbeddingClient;
  /** Optional process-local document-vector cache for semantic retrieval. */
  semanticCache?: EmbeddingCache;
  /** When set, natural-language operations may export only these namespaces to the LLM. */
  llmAllowedNamespaces?: ReadonlySet<string>;
  /** Default supersession policy for manual natural-language remember operations. */
  validTimeMode?: ValidTimeMode;
  /** Maximum predicate groups receiving detailed recall schema context. */
  recallSchemaPredicateLimit?: number;
  /** Internal/library override for the hard recall schema byte budget. */
  recallSchemaByteLimit?: number;
  /** Optional default atomic reject-on-write policy for memory mutations. */
  /** `false` explicitly disables an environment-derived server default. */
  integrityEnforcement?: IntegrityEnforcementOptions | false;
  /** Optional default portable regression and semantic coverage write guard. */
  knowledgeCheckEnforcement?: KnowledgeCheckEnforcementOptions | false;
  /** Optional default explicit entity projection for recall and schema reads. */
  entityIdentity?: EntityIdentityMode | false;
  /** Optional default trust projection; tentative claims remain excluded by default. */
  trustMode?: TrustViewMode | false;
  /** Optional default final rendering mode for successful recall. */
  recallAnswerMode?: RecallAnswerMode;
  /**
   * MCP tool surface selection. `core` registers only the daily-driver memory
   * tools; `full` (the default) registers the complete knowledge-engineering set.
   */
  toolProfile?: McpToolProfile;
  /** Namespace used when a tool call names none (default: 'default'). */
  defaultNamespace?: string;
}

export type McpToolProfile = 'core' | 'full';

export interface RememberResult {
  added: string[];
  duplicates: number;
  retracted: number;
  archived?: string[];
  opId?: string;
  trust?: Extract<KnowledgeTrust, 'tentative'>;
}

export interface RememberOptions {
  validTimeMode?: ValidTimeMode;
  /** Per-call enforcement override; omission uses the dependency default. */
  integrityEnforcement?: IntegrityEnforcementOptions | false;
  /** Per-call check guard override; omission uses the dependency default. */
  knowledgeCheckEnforcement?: KnowledgeCheckEnforcementOptions | false;
  /** Opt-in canonical read view for the extraction schema; stored writes stay literal. */
  entityIdentity?: EntityIdentityMode | false;
  /** Explicit caller authority; tentative facts remain outside accepted reasoning. */
  trust?: KnowledgeTrust;
  /** Controlled clock injection for library tests and deterministic integrations. */
  at?: Date;
}

export interface RememberTranscriptOptions {
  captureId: string;
  at?: Date;
}

export interface RecallResult {
  status: RecallStatus;
  answer: string;
  query: string | null;
  bindings: Record<string, string>[];
  explanation?: ExplainKnowledgeResult;
  whyNot?: ExplainWhyNotResult;
  whyNotUnavailable?: RecallWhyNotUnavailable;
  rowTrust?: KnowledgeTrust[];
  queryReviews?: RecallQueryReview[];
  pruning?: RecallPruningReport;
  recordedSnapshot?: RecordedSnapshotMetadata;
  trustMode?: TrustViewMode;
  answerMode?: RecallAnswerMode;
  /** Local discovery evidence for a non-answer; never changes recall authority. */
  relatedKnowledge?: KnowledgeSearchResult;
}

export interface RetrievalResult {
  status: RecallStatus;
  query: string | null;
  bindings: Record<string, string>[];
  explanation?: ExplainKnowledgeResult;
  whyNot?: ExplainWhyNotResult;
  whyNotUnavailable?: RecallWhyNotUnavailable;
  rowTrust?: KnowledgeTrust[];
  queryReviews?: RecallQueryReview[];
  pruning?: RecallPruningReport;
  recordedSnapshot?: RecordedSnapshotMetadata;
  trustMode?: TrustViewMode;
  /** Local discovery evidence for a non-answer; never changes recall authority. */
  relatedKnowledge?: KnowledgeSearchResult;
}

export interface RecallWhyNotUnavailable {
  reason: 'diagnostic_limit';
  message: string;
}

export type RecallQueryReviewReason =
  | 'competing_predicate'
  | 'missing_temporal_context';

export interface RecallQueryReview {
  originalQuery: string;
  reviewedQuery: string | null;
  reasons: RecallQueryReviewReason[];
  competingPredicates: string[];
  outcome: 'repeated' | 'corrected' | 'unanswerable';
}

export type RecallSchemaAttemptOutcome = 'answered' | 'empty' | 'unanswerable';

export interface RecallSchemaAttempt {
  detailedPredicates: number;
  advertisedPredicates: number;
  catalogComplete: boolean;
  schemaComplete: boolean;
  summaryBytes: number;
  outcome: RecallSchemaAttemptOutcome;
}

export interface RecallPruningReport extends RecallSchemaDiagnostics {
  initialSelectedPredicates: string[];
  attempts: RecallSchemaAttempt[];
}

export type RecallStatus =
  | 'answered'
  | 'no_match'
  | 'unanswerable'
  | 'schema_budget_exhausted';

export type RecallAnswerMode = 'natural' | 'deterministic' | 'evidence';

export interface RecallRelatedKnowledgeOptions {
  limit?: number;
  kinds?: KnowledgeSearchClauseKind[];
}

export interface RecallOptions {
  queryPromptVariant?: QueryPromptVariant;
  explain?: boolean;
  /** Total proof witnesses per returned row, including the primary witness. */
  proofLimit?: number;
  schemaPredicateLimit?: number;
  schemaByteLimit?: number;
  entityIdentity?: EntityIdentityMode | false;
  trustMode?: TrustViewMode;
  graphSelector?: ExplanationGraphSelector;
  /** Read from the deterministic global journal position instead of current files. */
  recordedSequence?: number;
  /** Natural LLM phrasing, exact local bindings, or compact local evidence. */
  answerMode?: RecallAnswerMode;
  /** Add deterministic lexical/provenance discovery when recall cannot answer. */
  relatedKnowledge?: boolean | RecallRelatedKnowledgeOptions;
}

/** Render successful recall bindings locally without granting an LLM phrasing authority. */
export function deterministicRecallAnswer(
  query: string,
  bindings: Record<string, string>[],
  rowTrust?: KnowledgeTrust[]
): string {
  if (rowTrust !== undefined && rowTrust.length !== bindings.length) {
    throw new Error('deterministic recall rowTrust must match binding row count');
  }
  if (bindings.length === 0) {
    const answer = `No stored result matches ${query}.`;
    assertBoundedOutput(answer, 'deterministic recall answer');
    return answer;
  }
  const renderRow = (binding: Record<string, string>, index: number): string => {
    const values = Object.entries(binding)
      .map(([name, value]) => `${name} = ${value}`)
      .join(', ');
    const trust = rowTrust?.[index] === 'tentative' ? '[tentative] ' : '';
    return `${trust}${values.length === 0 ? 'supported' : values}`;
  };
  let answer: string;
  if (bindings.length === 1) {
    const tentative = rowTrust?.[0] === 'tentative';
    if (Object.keys(bindings[0]).length === 0) {
      answer = `The query ${query} is ${tentative ? 'tentatively ' : ''}supported.`;
    } else {
      answer = `${tentative ? 'Tentative result' : 'Result'} for ${query}: ${renderRow(
        bindings[0],
        0
      ).replace(/^\[tentative\] /, '')}.`;
    }
  } else {
    answer = `Results for ${query}:\n${bindings
      .map((binding, index) => `${index + 1}. ${renderRow(binding, index)}`)
      .join('\n')}`;
  }
  assertBoundedOutput(answer, 'deterministic recall answer');
  return answer;
}

interface RecallEvidenceSummary {
  claims: Set<string>;
  rules: Set<number>;
  absences: Set<string>;
  aggregates: Set<string>;
  projections: Set<string>;
  sources: Map<string, string>;
}

function evidenceSummary(): RecallEvidenceSummary {
  return {
    claims: new Set(),
    rules: new Set(),
    absences: new Set(),
    aggregates: new Set(),
    projections: new Set(),
    sources: new Map(),
  };
}

function sourceLabel(source: MemorySource): string {
  const temporal = source.temporal === undefined
    ? ''
    : ` [valid until ${source.temporal.validUntil}; previously ${source.temporal.previousClause}]`;
  const trust = source.trust === 'tentative' ? ' [tentative]' : '';
  const text = source.text === undefined ? '' : ` ${JSON.stringify(source.text)}`;
  return `${source.namespace}/${source.opId}@${source.ts}${trust}${temporal}${text}`;
}

function evidenceValue(value: string | number): string {
  return serializeTerm(
    typeof value === 'number'
      ? { type: 'num', value }
      : { type: 'atom', value }
  );
}

function collectEvidence(
  proof: SourcedQueryProof,
  summary: RecallEvidenceSummary
): void {
  if ('aggregated' in proof) {
    summary.aggregates.add(`${proof.op}(${proof.input}) = ${proof.value}`);
    for (const contributor of proof.contributors) {
      for (const child of contributor.proofs) collectEvidence(child, summary);
    }
    return;
  }
  if ('negated' in proof) {
    summary.absences.add(
      `${proof.predicate}(${proof.pattern
        .map((value) => value === null ? '_' : evidenceValue(value))
        .join(', ')})`
    );
    return;
  }
  summary.claims.add(
    `${proof.predicate}(${proof.values.map(evidenceValue).join(', ')})`
  );
  if (proof.rule !== undefined) summary.rules.add(proof.rule);
  if (proof.projectedFrom !== undefined) {
    summary.projections.add(proof.projectedFrom);
  }
  for (const source of [
    ...(proof.sources ?? []),
    ...(proof.sourceAlternatives ?? []),
  ]) {
    const key = JSON.stringify(source);
    summary.sources.set(key, sourceLabel(source));
  }
  for (const child of proof.because ?? []) collectEvidence(child, summary);
  if (proof.aggregate !== undefined) {
    summary.aggregates.add(
      `${proof.aggregate.op}(${proof.aggregate.input}) = ${proof.aggregate.value}`
    );
    for (const contributor of proof.aggregate.contributors) {
      for (const child of contributor.proofs) collectEvidence(child, summary);
    }
  }
}

/** Render successful recall with compact local proof and provenance evidence. */
export function evidenceRecallAnswer(
  query: string,
  bindings: Record<string, string>[],
  explanation: ExplainKnowledgeResult,
  rowTrust?: KnowledgeTrust[]
): string {
  if (explanation.rows.length !== bindings.length) {
    throw new Error('evidence recall explanation rows must match binding rows');
  }
  if (rowTrust !== undefined && rowTrust.length !== bindings.length) {
    throw new Error('evidence recall rowTrust must match binding rows');
  }
  const ruleByNumber = new Map(
    explanation.rules.map((rule) => [rule.number, rule.clause])
  );
  const lines = [`Evidence for ${query}:`];
  for (const [index, binding] of bindings.entries()) {
    const values = Object.entries(binding)
      .map(([name, value]) => `${name} = ${value}`)
      .join(', ');
    const tentative = rowTrust?.[index] === 'tentative' ? '[tentative] ' : '';
    lines.push(`${index + 1}. ${tentative}${values || 'supported'}`);
    const summary = evidenceSummary();
    const row = explanation.rows[index];
    for (const proof of row.proofs) collectEvidence(proof, summary);
    for (const alternative of row.alternativeProofs ?? []) {
      for (const proof of alternative) collectEvidence(proof, summary);
    }
    if (summary.claims.size > 0) {
      lines.push(`   Claims: ${[...summary.claims].sort().join('; ')}`);
    }
    if (summary.rules.size > 0) {
      lines.push(
        `   Rules: ${[...summary.rules]
          .sort((left, right) => left - right)
          .map((number) => `#${number} ${ruleByNumber.get(number) ?? '(unknown rule)'}`)
          .join('; ')}`
      );
    }
    if (summary.absences.size > 0) {
      lines.push(`   Absent: ${[...summary.absences].sort().join('; ')}`);
    }
    if (summary.aggregates.size > 0) {
      lines.push(`   Aggregates: ${[...summary.aggregates].sort().join('; ')}`);
    }
    if (summary.projections.size > 0) {
      lines.push(`   Projected from: ${[...summary.projections].sort().join('; ')}`);
    }
    if (summary.sources.size > 0) {
      lines.push(`   Sources: ${[...summary.sources.values()].sort().join('; ')}`);
    }
  }
  const answer = lines.join('\n');
  assertBoundedOutput(answer, 'evidence recall answer');
  return answer;
}

function resolvedRecallAnswerMode(value: unknown): RecallAnswerMode {
  if (value === undefined || value === 'natural') return 'natural';
  if (value === 'deterministic' || value === 'evidence') return value;
  throw new Error("recall answer mode must be 'natural', 'deterministic', or 'evidence'");
}

function resolvedRelatedKnowledgeOptions(
  value: RecallOptions['relatedKnowledge']
): RecallRelatedKnowledgeOptions | undefined {
  if (value === undefined || value === false) return undefined;
  if (value === true) return {};
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('recall related knowledge must be a boolean or options object');
  }
  return value;
}

function stripFences(text: string): string {
  return text
    .trim()
    .replace(/^```[a-zA-Z]*\n?/, '')
    .replace(/\n?```$/, '')
    .trim();
}

function assertLlmNamespacesAllowed(
  deps: PipelineDeps,
  namespaces: string[] | '*'
): void {
  const allowed = deps.llmAllowedNamespaces;
  if (allowed === undefined) return;
  const selected = namespaces === '*' ? deps.store.listNamespaces() : namespaces;
  const denied = selected.find((namespace) => !allowed.has(namespace));
  if (denied !== undefined) {
    throw new Error(
      `namespace '${denied}' is local-only under REMBERO_LLM_ALLOWED_NAMESPACES`
    );
  }
}

/** Ask the LLM, validate its output; on failure, retry once with the error message. */
async function completeWithRetry<T>(
  llm: LlmClient,
  messages: ChatMessage[],
  validate: (response: string) => T
): Promise<T> {
  const response = stripFences(await llm.complete(messages));
  try {
    return validate(response);
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    const retryMessages: ChatMessage[] = [
      ...messages,
      { role: 'assistant', content: response },
      {
        role: 'user',
        content: `Your previous output failed validation.\nError: ${error}\nOutput corrected lines only.`,
      },
    ];
    return validate(stripFences(await llm.complete(retryMessages)));
  }
}

export interface RememberExtraction {
  clauses: Clause[];
  retractions: Goal[][];
}

/** Extract and validate natural-language memory changes without mutating the store. */
export async function extractRememberText(
  deps: PipelineDeps,
  text: string,
  namespace = 'default',
  options: RememberOptions = {}
): Promise<RememberExtraction | null> {
  const trust = options.trust ?? 'accepted';
  if (trust !== 'accepted' && trust !== 'tentative') {
    throw new Error("knowledge trust must be 'accepted' or 'tentative'");
  }
  assertLlmNamespacesAllowed(deps, [namespace]);
  assertSafeForExternalLlm(text, 'memory text');
  const literalClauses = deps.store.load(namespace);
  const configuredIdentity = options.entityIdentity ?? deps.entityIdentity;
  const entityIdentity = configuredIdentity === false ? undefined : configuredIdentity;
  const schemaClauses = entityIdentity === 'canonical'
    ? canonicalizeKnowledge(
        literalClauses,
        deps.store.sourcesFor([namespace])
      ).clauses
    : literalKnowledge(literalClauses).clauses;
  const schema = buildSchemaSummary(schemaClauses);
  assertSafeForExternalLlm(schema, 'memory schema');
  const messages: ChatMessage[] = [
    { role: 'system', content: extractionSystemPrompt(schema, trust) },
    { role: 'user', content: text },
  ];
  return completeWithRetry(
    deps.llm,
    messages,
    (response): RememberExtraction | null => {
      if (response === NOTHING_SENTINEL) return null;
      const retractionLines: string[] = [];
      const clauseLines: string[] = [];
      for (const line of response.split('\n')) {
        const retractMatch = line.trim().match(/^retract\s+(.*)$/);
        if (retractMatch) retractionLines.push(retractMatch[1].replace(/\.\s*$/, ''));
        else clauseLines.push(line);
      }
      // parse retraction patterns up front so a bad one triggers the retry loop
      const retractions = retractionLines.map((p) => parseQuery(p));
      if (
        retractions.some(
          (goals) =>
            goals.length !== 1 || isComparison(goals[0]) || isNegation(goals[0])
        )
      ) {
        throw new Error('each retract line must contain exactly one positive fact pattern');
      }
      if (
        retractions.some((goals) => {
          const goal = goals[0];
          return (
            goal !== undefined &&
            !isComparison(goal) &&
            !isNegation(goal) &&
            isTrustMetadataPredicate(goal.predicate)
          );
        })
      ) {
        throw new Error(
          'natural-language memory extraction may not retract trust metadata'
        );
      }
      if (
        retractions.some((goals) => {
          const goal = goals[0];
          return (
            goal !== undefined &&
            !isComparison(goal) &&
            !isNegation(goal) &&
            isEntityMetadataPredicate(goal.predicate)
          );
        })
      ) {
        throw new Error(
          'natural-language memory extraction may not retract entity identity metadata'
        );
      }
      const clauses = parseProgram(clauseLines.join('\n'));
      if (clauses.some(isIntegrityConstraint)) {
        throw new Error(
          'natural-language memory extraction may not create integrity constraints'
        );
      }
      if (clauses.some(isTentativeDeclaration)) {
        throw new Error(
          'natural-language memory extraction may not assign trust metadata; the caller must request tentative storage'
        );
      }
      if (clauses.some(isEntityMetadataDeclaration)) {
        throw new Error(
          'natural-language memory extraction may not create entity identity metadata'
        );
      }
      return { clauses, retractions };
    }
  );
}

export async function rememberText(
  deps: PipelineDeps,
  text: string,
  namespace = 'default',
  options: RememberOptions = {}
): Promise<RememberResult> {
  const validTimeMode = options.validTimeMode ?? deps.validTimeMode ?? 'delete';
  const trust = options.trust ?? 'accepted';
  if (validTimeMode !== 'delete' && validTimeMode !== 'archive_until') {
    throw new Error("valid-time mode must be 'delete' or 'archive_until'");
  }
  const extraction = await extractRememberText(deps, text, namespace, options);
  if (extraction === null) return { added: [], duplicates: 0, retracted: 0 };
  if (trust === 'tentative' && extraction.retractions.length > 0) {
    throw new Error('tentative memory is additive; it cannot retract accepted facts');
  }

  const opId = deps.store.createOperationId();
  const configuredIntegrity =
    options.integrityEnforcement ?? deps.integrityEnforcement;
  const integrity = configuredIntegrity === false ? undefined : configuredIntegrity;
  const configuredChecks =
    options.knowledgeCheckEnforcement ?? deps.knowledgeCheckEnforcement;
  const checks = configuredChecks === false ? undefined : configuredChecks;
  const context = {
    opId,
    sourceText: text,
    origin: 'manual' as const,
    at: options.at,
    ...(integrity === undefined ? {} : { integrity }),
    ...(checks === undefined ? {} : { checks }),
  };
  if (extraction.retractions.length > 0) {
    const patterns = extraction.retractions.map((goals) =>
      goals.map(serializeGoal).join(', ')
    );
    const result = validTimeMode === 'archive_until'
      ? deps.store.supersede(namespace, patterns, extraction.clauses, context)
      : deps.store.replace(namespace, patterns, extraction.clauses, context);
    return {
      added: result.added.map(serializeClause),
      duplicates: result.duplicates,
      retracted: result.retracted,
      ...(result.archived.length === 0
        ? {}
        : { archived: result.archived.map(serializeClause) }),
      opId,
    };
  }
  if (extraction.clauses.length === 0) {
    return { added: [], duplicates: 0, retracted: 0 };
  }
  if (trust === 'tentative') {
    const result = assertTentativeFacts(
      deps.store,
      namespace,
      extraction.clauses,
      context
    );
    return {
      added: result.added,
      duplicates: result.duplicates,
      retracted: 0,
      opId: result.opId,
      trust: 'tentative',
    };
  }
  if (integrity === undefined) {
    deps.store.note(namespace, 'remember', { opId, text }, options.at);
  }
  const { added, duplicates } = deps.store.assert(namespace, extraction.clauses, context);
  return { added: added.map(serializeClause), duplicates, retracted: 0, opId };
}

/**
 * Extract only additive, ground facts from an untrusted transcript tail.
 * The raw transcript is never persisted as per-fact provenance.
 */
export async function rememberTranscriptText(
  deps: PipelineDeps,
  transcript: string,
  namespace: string,
  options: RememberTranscriptOptions
): Promise<RememberResult> {
  assertLlmNamespacesAllowed(deps, [namespace]);
  assertSafeForExternalLlm(transcript, 'transcript');
  const literalClauses = deps.store.load(namespace);
  const schemaClauses = deps.entityIdentity === 'canonical'
    ? canonicalizeKnowledge(
        literalClauses,
        deps.store.sourcesFor([namespace])
      ).clauses
    : literalKnowledge(literalClauses).clauses;
  const schema = buildSchemaSummary(schemaClauses);
  assertSafeForExternalLlm(schema, 'memory schema');
  const messages: ChatMessage[] = [
    { role: 'system', content: transcriptExtractionSystemPrompt(schema) },
    { role: 'user', content: transcript },
  ];
  const clauses = await completeWithRetry(
    deps.llm,
    messages,
    (response): Clause[] | null => {
      if (response === NOTHING_SENTINEL) return null;
      if (response.split('\n').some((line) => /^\s*retract\b/i.test(line))) {
        throw new Error('auto-capture accepts additive ground facts only; retractions are forbidden');
      }
      const parsed = parseProgram(response);
      if (parsed.some(isTentativeDeclaration)) {
        throw new Error('auto-capture may not create trust metadata');
      }
      if (parsed.some(isEntityMetadataDeclaration)) {
        throw new Error('auto-capture may not create entity identity metadata');
      }
      if (parsed.some((clause) => clause.body.length > 0)) {
        throw new Error('auto-capture accepts additive ground facts only; rules are forbidden');
      }
      if (parsed.length > 12) {
        throw new Error('auto-capture accepts at most 12 additive ground facts');
      }
      return parsed;
    }
  );
  if (clauses === null || clauses.length === 0) {
    return { added: [], duplicates: 0, retracted: 0 };
  }

  const opId = deps.store.createOperationId();
  const { added, duplicates } = deps.store.assert(namespace, clauses, {
    opId,
    captureId: options.captureId,
    origin: 'claude-stop',
    sourceText: 'Auto-captured from a Claude Code Stop hook',
    at: options.at,
    ...(deps.integrityEnforcement === undefined || deps.integrityEnforcement === false
      ? {}
      : { integrity: deps.integrityEnforcement }),
    ...(deps.knowledgeCheckEnforcement === undefined ||
    deps.knowledgeCheckEnforcement === false
      ? {}
      : { checks: deps.knowledgeCheckEnforcement }),
  });
  return {
    added: added.map(serializeClause),
    duplicates,
    retracted: 0,
    opId,
  };
}

function visiblePredicateList(known: ReadonlySet<string>): string {
  const ordered = [...known].sort();
  const visible = ordered.slice(0, 64);
  return `${visible.join(', ') || '(none)'}${
    ordered.length > visible.length ? `, ... (${ordered.length - visible.length} more shown in schema)` : ''
  }`;
}

function validateQueryPredicates(
  goals: Goal[],
  known: ReadonlySet<string>,
  question: string
): void {
  const questionWords = new Set(recallWords(question));
  for (const goal of goals) {
    if (isComparison(goal)) continue;
    const literal = isNegation(goal) ? goal.not : goal;
    const key = `${literal.predicate}/${literal.args.length}`;
    if (known.has(key)) continue;
    if (isNegation(goal)) {
      const sameArity = [...known]
        .map((candidate) => candidate.match(/^(.*)\/(\d+)$/))
        .filter((match): match is RegExpMatchArray => match !== null)
        .filter((match) => Number(match[2]) === literal.args.length)
        .map((match) => match[1]);
      const lookalike = sameArity.find(
        (predicate) =>
          predicate !== literal.predicate &&
          recallEditDistance(predicate, literal.predicate) <= 1
      );
      if (lookalike !== undefined) {
        throw new Error(
          `unknown negated predicate ${key} resembles ${lookalike}/${literal.args.length}; correct the predicate name`
        );
      }
      const predicateWords = recallWords(literal.predicate);
      if (
        predicateWords.length === 0 ||
        !predicateWords.every((word) => questionWords.has(word))
      ) {
        throw new Error(
          `unknown negated predicate ${key} must be explicitly named by the question`
        );
      }
      continue;
    }
    if (!known.has(key)) {
      throw new Error(
        `unknown predicate ${key} — available in this schema: ${visiblePredicateList(known)}`
      );
    }
  }
}

const AGGREGATE_INTENT: Record<'count' | 'sum' | 'min' | 'max', RegExp> = {
  count: /\b(?:how many|number of|count)\b/i,
  sum: /\b(?:sum|total)\b/i,
  min: /\b(?:min(?:imum)?|smallest|least|lowest|earliest|youngest)\b/i,
  max: /\b(?:max(?:imum)?|largest|greatest|highest|latest|oldest|most)\b/i,
};
const DISTRIBUTIVE_AGGREGATE_INTENT = /\b(?:each|every|per|by)\b/i;

function directlyQueriesAggregateRelation(
  query: QuerySpec,
  requested: string | undefined,
  question: string,
  aggregatePredicates: ReadonlyMap<
    string,
    ReadonlyArray<{ op: AggregateOperator; outputPosition: number }>
  >
): boolean {
  if (query.kind !== 'relational' || requested === undefined) return false;
  const positive = query.goals.flatMap((goal, index) =>
    isComparison(goal) || isNegation(goal) ? [] : [{ goal, index }]
  );
  const candidates = positive.flatMap(({ goal, index }) =>
    (aggregatePredicates.get(predKey(goal)) ?? []).flatMap((signature) =>
      signature.op === requested &&
      goal.args[signature.outputPosition]?.type === 'var'
        ? [{ goal, index, signature }]
        : []
    )
  );
  if (candidates.length !== 1) return false;

  const { goal: aggregateGoal, index: aggregateIndex, signature } = candidates[0];
  const groupTerms = aggregateGoal.args.filter(
    (_term, position) => position !== signature.outputPosition
  );
  const groupVariables = new Set(
    groupTerms.flatMap((term) => (term.type === 'var' ? [term.name] : []))
  );
  const auxiliaryVariables = new Set<string>();
  for (const { goal, index } of positive) {
    if (index === aggregateIndex) continue;
    for (const term of goal.args) {
      if (term.type !== 'var') continue;
      if (!groupVariables.has(term.name)) return false;
      auxiliaryVariables.add(term.name);
    }
  }
  const distributive = DISTRIBUTIVE_AGGREGATE_INTENT.test(question);
  return groupTerms.every(
    (term) =>
      term.type === 'atom' ||
      term.type === 'num' ||
      (term.type === 'var' &&
        (distributive || auxiliaryVariables.has(term.name)))
  );
}

function validateQuerySpec(
  query: QuerySpec,
  known: ReadonlySet<string>,
  question: string,
  aggregatePredicates: ReadonlyMap<
    string,
    ReadonlyArray<{ op: AggregateOperator; outputPosition: number }>
  >,
  requireProjection: boolean
): void {
  validateQueryPredicates(query.goals, known, question);
  if (
    requireProjection &&
    query.kind === 'relational' &&
    query.project === undefined &&
    relationalVariableNames(query.goals).size > 1
  ) {
    throw new Error(
      'grounded relational queries with multiple variables must use select to declare answer columns'
    );
  }
  const requested = Object.entries(AGGREGATE_INTENT).find(([, pattern]) =>
    pattern.test(question)
  )?.[0];
  const directAggregateRelation = directlyQueriesAggregateRelation(
    query,
    requested,
    question,
    aggregatePredicates
  );
  if (
    query.kind === 'relational' &&
    requested !== undefined &&
    !directAggregateRelation
  ) {
    throw new Error(
      `question explicitly requests ${requested} aggregation; emit the scalar aggregate query form`
    );
  }
  if (query.kind === 'aggregate' && !AGGREGATE_INTENT[query.op].test(question)) {
    throw new Error(
      `${query.op} aggregation requires the question to explicitly request that aggregate`
    );
  }
}

function expressionVariableNames(
  expression: ScalarExpression,
  names: Set<string>
): void {
  if (!isArithmeticExpression(expression)) {
    if (expression.type === 'var') names.add(expression.name);
    return;
  }
  if (expression.kind === 'unary') {
    expressionVariableNames(expression.operand, names);
    return;
  }
  expressionVariableNames(expression.left, names);
  expressionVariableNames(expression.right, names);
}

function relationalVariableNames(goals: Goal[]): Set<string> {
  const names = new Set<string>();
  for (const goal of goals) {
    if (isComparison(goal)) {
      expressionVariableNames(goal.left, names);
      expressionVariableNames(goal.right, names);
      continue;
    }
    for (const term of (isNegation(goal) ? goal.not.args : goal.args)) {
      if (term.type === 'var') names.add(term.name);
    }
  }
  return names;
}

const UNANSWERABLE_RE = new RegExp(`^(\\?-)?\\s*${UNANSWERABLE}\\s*\\.?$`);

const MAX_QUERY_REVIEW_ROWS = 3;
const MAX_QUERY_REVIEW_COMPETITORS = 4;
const NAMED_LATER_STATE = /\b(?:before|prior\s+to)\b/i;

interface AnsweredQueryAmbiguity {
  reasons: RecallQueryReviewReason[];
  competingPredicates: string[];
}

function positiveLiterals(query: QuerySpec): Literal[] {
  return query.goals.filter(
    (goal): goal is Literal => !isComparison(goal) && !isNegation(goal)
  );
}

function sameGroundTerm(left: Term, right: Term): boolean {
  return (
    (left.type === 'atom' && right.type === 'atom' && left.value === right.value) ||
    (left.type === 'num' && right.type === 'num' && left.value === right.value)
  );
}

function literalAnchors(literal: Literal, arity = literal.args.length): Array<{
  position: number;
  term: Extract<Term, { type: 'atom' | 'num' }>;
}> {
  return literal.args
    .slice(0, arity)
    .map((term, position) => ({ term, position }))
    .filter(
      (entry): entry is {
        position: number;
        term: Extract<Term, { type: 'atom' | 'num' }>;
      } => entry.term.type === 'atom' || entry.term.type === 'num'
    );
}

function factMatchesAnchors(
  clause: Clause,
  anchors: ReadonlyArray<{ position: number; term: Term }>
): boolean {
  return anchors.every(({ position, term }) =>
    sameGroundTerm(clause.head.args[position], term)
  );
}

function factsByPredicate(clauses: Clause[]): ReadonlyMap<string, Clause[]> {
  const facts = new Map<string, Clause[]>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause) || clause.body.length > 0) continue;
    const key = `${clause.head.predicate}/${clause.head.args.length}`;
    const grouped = facts.get(key) ?? [];
    grouped.push(clause);
    facts.set(key, grouped);
  }
  return facts;
}

function predicateWordOverlap(predicate: string, questionWords: ReadonlySet<string>): number {
  return new Set(
    recallWords(predicate).filter((word) => questionWords.has(word))
  ).size;
}

function predicateParts(key: string): { predicate: string; arity: number } | undefined {
  const match = key.match(/^(.*)\/(\d+)$/);
  return match === null
    ? undefined
    : { predicate: match[1], arity: Number(match[2]) };
}

function directCompetitors(
  query: QuerySpec,
  selection: RecallSchemaSelection,
  facts: ReadonlyMap<string, Clause[]>,
  questionWords: ReadonlySet<string>
): string[] {
  const ordered = [...selection.availablePredicates];
  const rank = new Map(ordered.map((key, index) => [key, index]));
  const used = new Set(
    positiveLiterals(query).map(
      (literal) => `${literal.predicate}/${literal.args.length}`
    )
  );
  const competitors = new Set<string>();

  for (const literal of positiveLiterals(query)) {
    const chosenKey = `${literal.predicate}/${literal.args.length}`;
    const chosenRank = rank.get(chosenKey);
    const chosenOverlap = predicateWordOverlap(literal.predicate, questionWords);
    const anchors = literalAnchors(literal);
    if (chosenRank === undefined || chosenOverlap !== 0 || anchors.length === 0) continue;

    for (const candidateKey of ordered) {
      if (used.has(candidateKey)) continue;
      const candidate = predicateParts(candidateKey);
      if (candidate === undefined || candidate.arity !== literal.args.length) continue;
      const candidateRank = rank.get(candidateKey)!;
      const candidateOverlap = predicateWordOverlap(candidate.predicate, questionWords);
      if (candidateOverlap <= chosenOverlap && candidateRank >= chosenRank) continue;
      if (
        (facts.get(candidateKey) ?? []).some((clause) =>
          factMatchesAnchors(clause, anchors)
        )
      ) {
        competitors.add(candidateKey);
      }
    }
  }

  return [...competitors].sort(
    (left, right) => (rank.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (rank.get(right) ?? Number.MAX_SAFE_INTEGER) || left.localeCompare(right)
  );
}

function temporalCompetitors(
  query: QuerySpec,
  selection: RecallSchemaSelection,
  facts: ReadonlyMap<string, Clause[]>,
  question: string,
  questionWords: ReadonlySet<string>
): string[] {
  if (!NAMED_LATER_STATE.test(question)) return [];
  const literals = positiveLiterals(query);
  const usedPredicates = new Set(literals.map((literal) => literal.predicate));
  const competitors = new Set<string>();

  for (const historical of literals) {
    if (!historical.predicate.endsWith('_until') || historical.args.length < 2) continue;
    const basePredicate = historical.predicate.slice(0, -'_until'.length);
    const baseArity = historical.args.length - 1;
    const baseKey = `${basePredicate}/${baseArity}`;
    if (
      usedPredicates.has(basePredicate) ||
      !selection.availablePredicates.has(baseKey)
    ) {
      continue;
    }
    const anchors = literalAnchors(historical, baseArity);
    const anchorPositions = new Set(anchors.map(({ position }) => position));
    const namesCurrentState = (facts.get(baseKey) ?? []).some((clause) => {
      if (!factMatchesAnchors(clause, anchors)) return false;
      return clause.head.args.some((term, position) => {
        if (anchorPositions.has(position)) return false;
        const value = term.type === 'atom'
          ? term.value
          : term.type === 'num'
            ? String(term.value)
            : '';
        return recallWords(value).some((word) => questionWords.has(word));
      });
    });
    if (namesCurrentState) competitors.add(baseKey);
  }

  return [...competitors].sort();
}

function answeredQueryAmbiguity(
  query: QuerySpec,
  selection: RecallSchemaSelection,
  clauses: Clause[],
  question: string
): AnsweredQueryAmbiguity | undefined {
  const questionWords = new Set(recallWords(question));
  const literals = positiveLiterals(query);
  const mayHaveDirectCompetitor = literals.some(
    (literal) =>
      literalAnchors(literal).length > 0 &&
      predicateWordOverlap(literal.predicate, questionWords) === 0
  );
  const mayNeedTemporalContext =
    NAMED_LATER_STATE.test(question) &&
    literals.some((literal) => literal.predicate.endsWith('_until'));
  if (!mayHaveDirectCompetitor && !mayNeedTemporalContext) return undefined;
  const facts = factsByPredicate(clauses);
  const direct = mayHaveDirectCompetitor
    ? directCompetitors(query, selection, facts, questionWords)
    : [];
  const temporal = mayNeedTemporalContext
    ? temporalCompetitors(
        query,
        selection,
        facts,
        question,
        questionWords
      )
    : [];
  const reasons: RecallQueryReviewReason[] = [];
  if (direct.length > 0) reasons.push('competing_predicate');
  if (temporal.length > 0) reasons.push('missing_temporal_context');
  if (reasons.length === 0) return undefined;
  return {
    reasons,
    competingPredicates: [...new Set([...direct, ...temporal])].slice(
      0,
      MAX_QUERY_REVIEW_COMPETITORS
    ),
  };
}

function proofTrust(proof: SourcedQueryProof): KnowledgeTrust {
  return 'trust' in proof && proof.trust === 'tentative'
    ? 'tentative'
    : 'accepted';
}

function explanationRowTrust(
  explanation: ExplainKnowledgeResult
): KnowledgeTrust[] {
  return explanation.rows.map((row) =>
    row.proofs.some((proof) => proofTrust(proof) === 'tentative')
      ? 'tentative'
      : 'accepted'
  );
}

export async function retrieveQuestion(
  deps: PipelineDeps,
  question: string,
  namespaces: string[] | '*' = ['default'],
  options: RecallOptions = {}
): Promise<RetrievalResult> {
  assertLlmNamespacesAllowed(deps, namespaces);
  assertSafeForExternalLlm(question, 'recall question');
  const recorded = options.recordedSequence === undefined
    ? undefined
    : deps.store.recordedSnapshot(namespaces, options.recordedSequence);
  const current = recorded === undefined
    ? deps.store.knowledgeSnapshot(namespaces)
    : undefined;
  const literalClauses = recorded?.clauses ?? current!.clauses;
  const literalSources = recorded?.sources ?? current!.sources;
  const recordedSnapshot = recorded === undefined
    ? undefined
    : {
        sequence: recorded.sequence,
        journalEntries: recorded.journalEntries,
        namespaces: recorded.namespaces,
      };
  const configuredIdentity = options.entityIdentity ?? deps.entityIdentity;
  const entityIdentity = configuredIdentity === false ? undefined : configuredIdentity;
  const configuredTrust = options.trustMode ?? deps.trustMode;
  const trustMode =
    configuredTrust === false || configuredTrust === undefined
      ? 'accepted'
      : configuredTrust;
  const view = entityIdentity === 'canonical'
    ? canonicalizeKnowledge(literalClauses, literalSources, trustMode)
    : literalKnowledge(literalClauses, literalSources, trustMode);
  const clauses = view.clauses;
  const trustResult = trustMode === 'accepted' ? {} : { trustMode };
  const relatedOptions = resolvedRelatedKnowledgeOptions(options.relatedKnowledge);
  const relatedResult = (): { relatedKnowledge?: KnowledgeSearchResult } =>
    relatedOptions === undefined
      ? {}
      : {
          relatedKnowledge: searchKnowledge(
            literalClauses,
            question,
            literalSources,
            {
              ...(relatedOptions.limit === undefined
                ? {}
                : { limit: relatedOptions.limit }),
              ...(relatedOptions.kinds === undefined
                ? {}
                : { kinds: relatedOptions.kinds }),
              ...(entityIdentity === undefined ? {} : { entityIdentity }),
              ...(trustMode === 'accepted' ? {} : { trustMode }),
            }
          ),
        };
  const aggregatePredicates = new Map<
    string,
    Array<{ op: AggregateOperator; outputPosition: number }>
  >();
  for (const clause of clauses) {
    if (!isAggregateRule(clause)) continue;
    const outputPosition = clause.head.args.findIndex(
      (term) => term.type === 'var' && term.name === clause.aggregate.as
    );
    const signatures = aggregatePredicates.get(predKey(clause.head)) ?? [];
    signatures.push({ op: clause.aggregate.op, outputPosition });
    aggregatePredicates.set(predKey(clause.head), signatures);
  }
  if (clauses.length === 0) {
    return {
      status: 'unanswerable',
      query: null,
      bindings: [],
      ...relatedResult(),
      ...trustResult,
      ...(recordedSnapshot === undefined ? {} : { recordedSnapshot }),
    };
  }
  const schemaPredicateLimit =
    options.schemaPredicateLimit ?? deps.recallSchemaPredicateLimit;
  const schemaByteLimit = options.schemaByteLimit ?? deps.recallSchemaByteLimit;
  let initialSelection: RecallSchemaSelection;
  try {
    initialSelection = selectRecallSchema(clauses, question, {
      sourceIndex: view.sources,
      ...(schemaPredicateLimit === undefined
        ? {}
        : { predicateLimit: schemaPredicateLimit }),
      ...(schemaByteLimit === undefined ? {} : { byteLimit: schemaByteLimit }),
    });
  } catch (error) {
    if (!(error instanceof RecallSchemaBudgetError)) throw error;
    try {
      initialSelection = selectRecallSchema(clauses, question, {
        sourceIndex: view.sources,
        predicateLimit: MAX_RECALL_SCHEMA_PREDICATES,
        ...(schemaByteLimit === undefined ? {} : { byteLimit: schemaByteLimit }),
      });
    } catch (widenError) {
      if (widenError instanceof RecallSchemaBudgetError) {
        return {
          status: 'schema_budget_exhausted',
          query: null,
          bindings: [],
          ...relatedResult(),
          ...trustResult,
          ...(recordedSnapshot === undefined ? {} : { recordedSnapshot }),
        };
      }
      throw widenError;
    }
  }

  interface PassResult {
    outcome: RecallSchemaAttemptOutcome;
    query: string | null;
    bindings: Record<string, string>[];
    explanation?: ExplainKnowledgeResult;
    rowTrust?: KnowledgeTrust[];
    queryReview?: RecallQueryReview;
  }

  const runPass = async (selection: RecallSchemaSelection): Promise<PassResult> => {
    assertSafeForExternalLlm(selection.summary, 'memory schema');
    const messages: ChatMessage[] = [
      {
        role: 'system',
        content: queryGenSystemPrompt(selection.summary, options.queryPromptVariant),
      },
      { role: 'user', content: question },
    ];
    const validateResponse = (response: string): QuerySpec | null => {
      if (UNANSWERABLE_RE.test(response)) return null;
      const parsed = parseQuerySpec(response);
      validateQuerySpec(
        parsed,
        selection.availablePredicates,
        question,
        aggregatePredicates,
        options.queryPromptVariant !== 'baseline'
      );
      return entityIdentity === 'canonical'
        ? view.resolver.canonicalizeQuery(parsed).query
        : parsed;
    };
    const evaluate = (query: QuerySpec, queryText: string): PassResult => {
      if (options.explain || trustMode === 'include_tentative') {
        const explanation = explainKnowledge(
          literalClauses,
          queryText,
          literalSources,
          {
            ...(options.proofLimit === undefined
              ? {}
              : { maxProofsPerRow: options.proofLimit }),
            ...(entityIdentity === undefined ? {} : { entityIdentity }),
            ...(trustMode === 'accepted' ? {} : { trustMode }),
            ...(options.graphSelector === undefined
              ? {}
              : { graphSelector: options.graphSelector }),
          }
        );
        const bindings = explanation.rows.map((row) => row.bindings);
        return {
          outcome: bindings.length > 0 ? 'answered' : 'empty',
          query: queryText,
          bindings,
          ...(trustMode === 'include_tentative'
            ? { rowTrust: explanationRowTrust(explanation) }
            : {}),
          ...(options.explain ? { explanation } : {}),
        };
      }
      const bindings = evaluateQuerySpec(clauses, query).map((binding: Bindings) =>
        Object.fromEntries(
          Object.entries(binding).map(([name, term]) => [name, serializeTerm(term)])
        )
      );
      return {
        outcome: bindings.length > 0 ? 'answered' : 'empty',
        query: queryText,
        bindings,
      };
    };

    let query = await completeWithRetry(deps.llm, messages, validateResponse);
    if (query === null) {
      return { outcome: 'unanswerable', query: null, bindings: [] };
    }
    let queryText = serializeQuerySpec(query);
    let result = evaluate(query, queryText);
    if (result.outcome === 'answered') {
      const ambiguity = answeredQueryAmbiguity(
        query,
        selection,
        clauses,
        question
      );
      if (ambiguity === undefined) return result;

      const originalQuery = queryText;
      const reviewPrompt = answeredQueryReviewPrompt(
        question,
        originalQuery,
        result.bindings.slice(0, MAX_QUERY_REVIEW_ROWS),
        ambiguity.reasons,
        ambiguity.competingPredicates
      );
      assertSafeForExternalLlm(reviewPrompt, 'query review evidence');
      const reviewMessages: ChatMessage[] = [
        ...messages,
        { role: 'assistant', content: `?- ${originalQuery}.` },
        { role: 'user', content: reviewPrompt },
      ];
      query = await completeWithRetry(deps.llm, reviewMessages, validateResponse);
      if (query === null) {
        return {
          outcome: 'unanswerable',
          query: null,
          bindings: [],
          queryReview: {
            originalQuery,
            reviewedQuery: null,
            reasons: ambiguity.reasons,
            competingPredicates: ambiguity.competingPredicates,
            outcome: 'unanswerable',
          },
        };
      }
      queryText = serializeQuerySpec(query);
      const queryReview: RecallQueryReview = {
        originalQuery,
        reviewedQuery: queryText,
        reasons: ambiguity.reasons,
        competingPredicates: ambiguity.competingPredicates,
        outcome: queryText === originalQuery ? 'repeated' : 'corrected',
      };
      if (queryReview.outcome === 'repeated') return { ...result, queryReview };
      result = evaluate(query, queryText);
      return { ...result, queryReview };
    }

    const fallbackMessages: ChatMessage[] = [
      ...messages,
      { role: 'assistant', content: `?- ${queryText}.` },
      {
        role: 'user',
        content: `The query ${queryText} returned no results. If it correctly expresses the question, repeat it unchanged: an empty result is valid evidence that no stored fact matches. Try ONE alternative only if the first query mistranslated the question. Output exactly ?- ${UNANSWERABLE}. only when the schema cannot express the question at all, never merely because the result was empty.`,
      },
    ];
    query = await completeWithRetry(deps.llm, fallbackMessages, validateResponse);
    if (query === null) {
      return { outcome: 'unanswerable', query: null, bindings: [] };
    }
    queryText = serializeQuerySpec(query);
    result = evaluate(query, queryText);
    return result;
  };

  const attempts: RecallSchemaAttempt[] = [];
  const queryReviews: RecallQueryReview[] = [];
  let finalSelection = initialSelection;
  let pass = await runPass(finalSelection);
  if (pass.queryReview !== undefined) queryReviews.push(pass.queryReview);
  const recordAttempt = () => {
    attempts.push({
      detailedPredicates: finalSelection.selectedPredicates.length,
      advertisedPredicates: finalSelection.advertisedPredicates,
      catalogComplete: finalSelection.catalogComplete,
      schemaComplete: finalSelection.schemaComplete,
      summaryBytes: finalSelection.summaryBytes,
      outcome: pass.outcome,
    });
  };
  recordAttempt();

  if (
    pass.outcome !== 'answered' &&
    !finalSelection.schemaComplete &&
    finalSelection.totalPredicates <= MAX_RECALL_SCHEMA_PREDICATES &&
    finalSelection.selectedPredicates.length < finalSelection.totalPredicates
  ) {
    try {
      finalSelection = selectRecallSchema(clauses, question, {
        sourceIndex: view.sources,
        predicateLimit: finalSelection.totalPredicates,
        ...(schemaByteLimit === undefined ? {} : { byteLimit: schemaByteLimit }),
      });
      pass = await runPass(finalSelection);
      if (pass.queryReview !== undefined) queryReviews.push(pass.queryReview);
      recordAttempt();
    } catch (error) {
      if (!(error instanceof RecallSchemaBudgetError)) {
        throw error;
      }
      finalSelection = initialSelection;
    }
  }

  const includePruning =
    initialSelection.pruned || !initialSelection.schemaComplete || attempts.length > 1;
  const pruning = includePruning
    ? {
        pruning: {
          ...recallSchemaDiagnostics(finalSelection),
          initialSelectedPredicates: [...initialSelection.selectedPredicates],
          attempts,
        },
      }
    : {};
  const { outcome, queryReview: _queryReview, ...retrieval } = pass;
  const reviewResult = queryReviews.length === 0 ? {} : { queryReviews };
  const snapshotResult = recordedSnapshot === undefined ? {} : { recordedSnapshot };
  if (pass.outcome === 'answered') {
    return {
      status: 'answered',
      ...retrieval,
      ...reviewResult,
      ...pruning,
      ...trustResult,
      ...snapshotResult,
    };
  }
  if (!finalSelection.schemaComplete) {
    return {
      status: 'schema_budget_exhausted',
      ...retrieval,
      ...relatedResult(),
      ...reviewResult,
      ...pruning,
      ...trustResult,
      ...snapshotResult,
    };
  }
  let whyNotResult:
    | { whyNot: ExplainWhyNotResult }
    | { whyNotUnavailable: RecallWhyNotUnavailable }
    | Record<string, never> = {};
  if (outcome === 'empty' && retrieval.query !== null) {
    try {
      whyNotResult = {
        whyNot: explainWhyNot(
          literalClauses,
          retrieval.query,
          literalSources,
          {
            ...(options.proofLimit === undefined
              ? {}
              : { maxProofsPerRow: options.proofLimit }),
            ...(entityIdentity === undefined ? {} : { entityIdentity }),
            ...(trustMode === 'accepted' ? {} : { trustMode }),
          }
        ),
      };
    } catch (error) {
      if (!(error instanceof EngineLimitError)) throw error;
      whyNotResult = {
        whyNotUnavailable: {
          reason: 'diagnostic_limit',
          message: error.message,
        },
      };
    }
  }
  return {
    status: outcome === 'empty' ? 'no_match' : 'unanswerable',
    ...retrieval,
    ...whyNotResult,
    ...relatedResult(),
    ...reviewResult,
    ...pruning,
    ...trustResult,
    ...snapshotResult,
  };
}

export async function recallQuestion(
  deps: PipelineDeps,
  question: string,
  namespaces: string[] | '*' = ['default'],
  options: RecallOptions = {}
): Promise<RecallResult> {
  const answerMode = resolvedRecallAnswerMode(
    options.answerMode ?? deps.recallAnswerMode
  );
  const answerModeResult =
    answerMode === 'natural' ? {} : { answerMode };
  const retrieval = await retrieveQuestion(
    deps,
    question,
    namespaces,
    answerMode === 'evidence' ? { ...options, explain: true } : options
  );
  if (retrieval.query === null) {
    return {
      answer:
        retrieval.status === 'schema_budget_exhausted'
          ? 'Recall reached its schema budget before it could rule out relevant memories.'
          : 'I have no relevant memories to answer that.',
      ...answerModeResult,
      ...retrieval,
    };
  }

  if (retrieval.status === 'schema_budget_exhausted') {
    return {
      answer: 'Recall reached its schema budget before it could rule out relevant memories.',
      ...answerModeResult,
      ...retrieval,
    };
  }

  if (retrieval.status === 'no_match') {
    return {
      answer:
        retrieval.whyNot?.summary ??
        `No stored result matches ${retrieval.query}.`,
      ...answerModeResult,
      ...retrieval,
    };
  }

  if (answerMode === 'deterministic') {
    return {
      answer: deterministicRecallAnswer(
        retrieval.query,
        retrieval.bindings,
        retrieval.rowTrust
      ),
      answerMode,
      ...retrieval,
    };
  }
  if (answerMode === 'evidence') {
    if (retrieval.explanation === undefined) {
      throw new Error('evidence recall requires explanation evidence');
    }
    return {
      answer: evidenceRecallAnswer(
        retrieval.query,
        retrieval.bindings,
        retrieval.explanation,
        retrieval.rowTrust
      ),
      answerMode,
      ...retrieval,
    };
  }

  const phrasing = phrasingUserPrompt(
    question,
    retrieval.query,
    retrieval.bindings,
    retrieval.trustMode,
    retrieval.rowTrust
  );
  assertSafeForExternalLlm(phrasing, 'recall evidence');
  const answer = await deps.llm.complete([
    { role: 'system', content: PHRASING_SYSTEM_PROMPT },
    {
      role: 'user',
      content: phrasing,
    },
  ]);
  return { answer: answer.trim(), ...retrieval };
}
