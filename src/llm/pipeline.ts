import { createHash } from 'node:crypto';
import { buildComputedNotes, computedNoteLines } from '../knowledge/computed-notes.js';
import {
  CLOSURE_SUFFIX,
  parseQueryProgram,
  emptyResultFeedback,
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
import {
  dialectQuerySystemPrompt,
  normalizeDialectResponse,
} from './dialect.js';
import type {
  MemoryStore,
  MemorySource,
  RecordedSnapshotMetadata,
  ValidTimeMode,
} from '../store/store.js';
import { OpenRouterClient, type ChatMessage, type LlmClient } from './client.js';
import { readerAllowsRemoteFromEnv, readerFromEnv } from '../env.js';
import { finalAnswerLine } from '../knowledge/answer-line.js';
import {
  questionKindFromText,
  type QuestionKind,
} from '../knowledge/question-kind.js';
import {
  DEFAULT_READING_CONTEXT_BYTES,
  DEFAULT_READING_DEPTH,
  buildReadingPrompt,
  readsInNotes,
  retrieveSessions,
  type RetrievableSession,
  type SessionRetrievalOptions,
} from '../knowledge/session-retrieval.js';
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
import {
  MAX_OUTPUT_BYTES,
  assertBoundedOutput,
  assertSafeForExternalLlm,
} from '../safety.js';
import type { SessionStore } from '../sessions/store.js';
import {
  applyPredicateAliases,
  applyPredicateAliasesToGoals,
  assertGroundedConstants,
  assertKnownVocabulary,
  functionalKeysFrom,
  impliedSupersessions,
  normalizeExtractionOutput,
  predicateAliasesFrom,
  rewriteSelfAtoms,
  rewriteSelfAtomsInGoals,
  canonicalizeAtoms,
  canonicalizeAtomsInGoals,
} from './extraction-guard.js';
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
  /**
   * Optional conversation store for reading recall. Present only when
   * `REMBERO_SESSIONS=on`, and asked again before every write: the store writes
   * whatever it is handed.
   */
  sessions?: SessionStore;
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
  /**
   * Constant naming the speaker: "I", "me", "my" in remembered text become this
   * atom (default 'user'). Set REMBERO_SELF to your own name.
   */
  selfAtom?: string;
  /**
   * 'open' (default): extraction may introduce new predicates. 'closed': every
   * added fact must use a predicate already in the schema (aliases from
   * rembero_predicate_alias declarations are rewritten first).
   */
  extractionVocabulary?: 'open' | 'closed';
}

export const DEFAULT_SELF_ATOM = 'user';

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
  /** Operation id to record instead of a fresh one (evaluations key facts to their source). */
  opId?: string;
  /** Source text to record with the facts (default: the Stop-hook provenance line). */
  sourceText?: string;
  origin?: 'manual' | 'claude-stop';
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
  /** The `sessions` mode's evidence: the conversations the reader was shown, best rank first. */
  sessionsRead?: RecallSessionRead[];
  /** The model that read them, as the reader reported it. */
  readerModel?: string;
  /** `recall_explain` only: the reader's whole reply, its notes included. */
  readerReply?: string;
  /** `recall_explain` only: the five flags that routed this question. */
  questionKind?: QuestionKind;
  /** A stored conversation that could not be read; recall degraded to `no_evidence`. */
  sessionsError?: string;
}

/** One conversation the reader read, and the turns of it that matched the question. */
export interface RecallSessionRead {
  /** The namespace it is stored in; `forget_sessions` needs this with the key. */
  namespace: string;
  /** The session's file key in that namespace, as `forget_sessions` takes it. */
  key: string;
  /** The day the session started: the date the reader was shown. */
  date: string;
  /** Its matching turns, in the order they were said, each cut to a bounded length. */
  excerpts: string[];
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
  'competing_predicate' | 'missing_temporal_context';

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
  | 'schema_budget_exhausted'
  /** `sessions` mode: no stored conversation to read, so no reader was asked. */
  | 'no_evidence'
  /** `sessions` mode: the reader read the history and said it does not know. */
  | 'unknown';

export type RecallAnswerMode =
  | 'natural'
  | 'deterministic'
  | 'evidence'
  /** Read the namespace's stored conversations instead of querying the facts. */
  | 'sessions';

/**
 * The reader the `sessions` mode sends its prompt to: what `REMBERO_READER_*` configures
 * (`readerFromEnv` in env.ts), and what one `recall` call may pass instead.
 */
export interface RecallReader {
  baseUrl: string;
  model: string;
  apiKey: string;
  maxTokens: number;
  temperature: number;
  timeoutMs: number;
  /**
   * A client already built for this endpoint. Recall builds one from the fields above when
   * this is absent; a caller that holds a client — a test's stub, a harness's pooled client
   * — hands it over, and then nothing here opens a connection of its own.
   */
  client?: LlmClient;
}

export interface RecallRelatedKnowledgeOptions {
  limit?: number;
  kinds?: KnowledgeSearchClauseKind[];
}

export interface RecallOptions {
  /** The moment the question is asked; relative dates in computed notes count from it. */
  at?: Date;
  /** Computed notes in the answer (default: on unless REMBERO_COMPUTED_NOTES=0). */
  computedNotes?: boolean;
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
  /** Natural LLM phrasing, exact local bindings, compact local evidence, or session reading. */
  answerMode?: RecallAnswerMode;
  /** The reader for the `sessions` mode; omitted means `REMBERO_READER_*`, then `deps.llm`. */
  reader?: RecallReader;
  /** Add deterministic lexical/provenance discovery when recall cannot answer. */
  relatedKnowledge?: boolean | RecallRelatedKnowledgeOptions;
}

/** Render successful recall bindings locally without granting an LLM phrasing authority. */
export function deterministicRecallAnswer(
  query: string,
  bindings: Record<string, string>[],
  rowTrust?: KnowledgeTrust[],
): string {
  if (rowTrust !== undefined && rowTrust.length !== bindings.length) {
    throw new Error(
      'deterministic recall rowTrust must match binding row count',
    );
  }
  if (bindings.length === 0) {
    const answer = `No stored result matches ${query}.`;
    assertBoundedOutput(answer, 'deterministic recall answer');
    return answer;
  }
  const renderRow = (
    binding: Record<string, string>,
    index: number,
  ): string => {
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
        0,
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
  const temporal =
    source.temporal === undefined
      ? ''
      : ` [valid until ${source.temporal.validUntil}; previously ${source.temporal.previousClause}]`;
  const trust = source.trust === 'tentative' ? ' [tentative]' : '';
  const text =
    source.text === undefined ? '' : ` ${JSON.stringify(source.text)}`;
  return `${source.namespace}/${source.opId}@${source.ts}${trust}${temporal}${text}`;
}

function evidenceValue(value: string | number): string {
  return serializeTerm(
    typeof value === 'number'
      ? { type: 'num', value }
      : { type: 'atom', value },
  );
}

function collectEvidence(
  proof: SourcedQueryProof,
  summary: RecallEvidenceSummary,
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
        .map((value) => (value === null ? '_' : evidenceValue(value)))
        .join(', ')})`,
    );
    return;
  }
  summary.claims.add(
    `${proof.predicate}(${proof.values.map(evidenceValue).join(', ')})`,
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
      `${proof.aggregate.op}(${proof.aggregate.input}) = ${proof.aggregate.value}`,
    );
    for (const contributor of proof.aggregate.contributors) {
      for (const child of contributor.proofs) collectEvidence(child, summary);
    }
  }
}

/** Every memory source a proof tree rests on, with the text and time it was said. */
function collectSources(proof: SourcedQueryProof, out: MemorySource[]): void {
  if ('aggregated' in proof) {
    for (const contributor of proof.contributors) for (const child of contributor.proofs) collectSources(child, out);
    return;
  }
  if ('negated' in proof) return;
  for (const source of [...(proof.sources ?? []), ...(proof.sourceAlternatives ?? [])]) {
    if (!out.some((s) => s.opId === source.opId && s.ts === source.ts)) out.push(source);
  }
  for (const child of proof.because ?? []) collectSources(child, out);
  if (proof.aggregate !== undefined) {
    for (const contributor of proof.aggregate.contributors) for (const child of contributor.proofs) collectSources(child, out);
  }
}

function sourcesAsTurns(sources: MemorySource[]): Array<{ ts: string; text: string }> {
  return sources
    .filter((s) => s.text !== undefined && s.text !== '')
    .map((s) => ({ ts: s.ts, text: `USER: ${s.text}` }));
}

export function computedNotesEnabled(options: { computedNotes?: boolean }): boolean {
  return options.computedNotes ?? process.env.REMBERO_COMPUTED_NOTES !== '0';
}

/** Render successful recall with compact local proof and provenance evidence. */
export function evidenceRecallAnswer(
  query: string,
  bindings: Record<string, string>[],
  explanation: ExplainKnowledgeResult,
  rowTrust?: KnowledgeTrust[],
  notes?: { question: string; at: Date },
): string {
  if (explanation.rows.length !== bindings.length) {
    throw new Error('evidence recall explanation rows must match binding rows');
  }
  if (rowTrust !== undefined && rowTrust.length !== bindings.length) {
    throw new Error('evidence recall rowTrust must match binding rows');
  }
  const ruleByNumber = new Map(
    explanation.rules.map((rule) => [rule.number, rule.clause]),
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
          .map(
            (number) =>
              `#${number} ${ruleByNumber.get(number) ?? '(unknown rule)'}`,
          )
          .join('; ')}`,
      );
    }
    if (summary.absences.size > 0) {
      lines.push(`   Absent: ${[...summary.absences].sort().join('; ')}`);
    }
    if (summary.aggregates.size > 0) {
      lines.push(`   Aggregates: ${[...summary.aggregates].sort().join('; ')}`);
    }
    if (summary.projections.size > 0) {
      lines.push(
        `   Projected from: ${[...summary.projections].sort().join('; ')}`,
      );
    }
    if (summary.sources.size > 0) {
      lines.push(
        `   Sources: ${[...summary.sources.values()].sort().join('; ')}`,
      );
    }
    if (notes !== undefined) {
      // computed notes: the sources' own words, dated by when they were said, with the
      // arithmetic done; every line quotes the sentence it came from
      const rowSources: MemorySource[] = [];
      for (const proof of row.proofs) collectSources(proof, rowSources);
      for (const alternative of row.alternativeProofs ?? []) for (const proof of alternative) collectSources(proof, rowSources);
      const computed = computedNoteLines(notes.question, notes.at.toISOString(), sourcesAsTurns(rowSources));
      if (computed.length > 0) {
        lines.push('   Computed (deterministic, from the sources above):');
        for (const line of computed) lines.push(`     ${line}`);
      }
    }
  }
  const answer = lines.join('\n');
  assertBoundedOutput(answer, 'evidence recall answer');
  return answer;
}

function resolvedRecallAnswerMode(value: unknown): RecallAnswerMode {
  if (value === undefined || value === 'natural') return 'natural';
  if (
    value === 'deterministic' ||
    value === 'evidence' ||
    value === 'sessions'
  )
    return value;
  throw new Error(
    "recall answer mode must be 'natural', 'deterministic', 'evidence', or 'sessions'",
  );
}

function resolvedRelatedKnowledgeOptions(
  value: RecallOptions['relatedKnowledge'],
): RecallRelatedKnowledgeOptions | undefined {
  if (value === undefined || value === false) return undefined;
  if (value === true) return {};
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(
      'recall related knowledge must be a boolean or options object',
    );
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
  namespaces: string[] | '*',
): void {
  const allowed = deps.llmAllowedNamespaces;
  if (allowed === undefined) return;
  const selected =
    namespaces === '*' ? deps.store.listNamespaces() : namespaces;
  const denied = selected.find((namespace) => !allowed.has(namespace));
  if (denied !== undefined) {
    throw new Error(
      `namespace '${denied}' is local-only under REMBERO_LLM_ALLOWED_NAMESPACES`,
    );
  }
}

/** Ask the LLM, validate its output; on failure, retry once with the error message. */
async function completeWithRetry<T>(
  llm: LlmClient,
  messages: ChatMessage[],
  validate: (response: string) => T,
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
  options: RememberOptions = {},
): Promise<RememberExtraction | null> {
  const trust = options.trust ?? 'accepted';
  if (trust !== 'accepted' && trust !== 'tentative') {
    throw new Error("knowledge trust must be 'accepted' or 'tentative'");
  }
  assertLlmNamespacesAllowed(deps, [namespace]);
  assertSafeForExternalLlm(text, 'memory text');
  const literalClauses = deps.store.load(namespace);
  const configuredIdentity = options.entityIdentity ?? deps.entityIdentity;
  const entityIdentity =
    configuredIdentity === false ? undefined : configuredIdentity;
  const schemaClauses =
    entityIdentity === 'canonical'
      ? canonicalizeKnowledge(
          literalClauses,
          deps.store.sourcesFor([namespace]),
        ).clauses
      : literalKnowledge(literalClauses).clauses;
  const schema = buildSchemaSummary(schemaClauses);
  assertSafeForExternalLlm(schema, 'memory schema');
  const selfAtom = deps.selfAtom ?? DEFAULT_SELF_ATOM;
  const aliases = predicateAliasesFrom(literalClauses);
  const knownPredicates = new Set(
    schemaClauses
      .filter((c) => !isIntegrityConstraint(c))
      .map((c) => predKey(c.head)),
  );
  const knownConstants = new Set<string>();
  for (const clause of schemaClauses) {
    if (isIntegrityConstraint(clause) || clause.body.length > 0) continue;
    for (const term of clause.head.args) {
      if (term.type === 'atom' || term.type === 'num')
        knownConstants.add(String(term.value));
    }
  }
  const messages: ChatMessage[] = [
    {
      role: 'system',
      content: extractionSystemPrompt(schema, trust, selfAtom),
    },
    { role: 'user', content: text },
  ];
  return completeWithRetry(
    deps.llm,
    messages,
    (response): RememberExtraction | null => {
      if (response === NOTHING_SENTINEL) return null;
      const retractionLines: string[] = [];
      const clauseLines: string[] = [];
      for (const line of normalizeExtractionOutput(response)) {
        const retractMatch = line.trim().match(/^retract\s+(.*)$/);
        if (retractMatch)
          retractionLines.push(retractMatch[1].replace(/\.\s*$/, ''));
        else clauseLines.push(line);
      }
      if (clauseLines.length === 0 && retractionLines.length === 0) return null;
      // parse retraction patterns up front so a bad one triggers the retry loop
      const retractions = retractionLines.map((p) => parseQuery(p));
      if (
        retractions.some(
          (goals) =>
            goals.length !== 1 ||
            isComparison(goals[0]) ||
            isNegation(goals[0]),
        )
      ) {
        throw new Error(
          'each retract line must contain exactly one positive fact pattern',
        );
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
          'natural-language memory extraction may not retract trust metadata',
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
          'natural-language memory extraction may not retract entity identity metadata',
        );
      }
      const clauses = parseProgram(clauseLines.join('\n'));
      if (clauses.some(isIntegrityConstraint)) {
        throw new Error(
          'natural-language memory extraction may not create integrity constraints',
        );
      }
      if (clauses.some(isTentativeDeclaration)) {
        throw new Error(
          'natural-language memory extraction may not assign trust metadata; the caller must request tentative storage',
        );
      }
      if (clauses.some(isEntityMetadataDeclaration)) {
        throw new Error(
          'natural-language memory extraction may not create entity identity metadata',
        );
      }
      // Deterministic guards (see extraction-guard.ts): the speaker becomes the
      // configured self atom, aliased predicates are renamed, every new constant
      // must be present in the input, and a closed vocabulary rejects unknowns.
      const guarded = applyPredicateAliases(
        canonicalizeAtoms(rewriteSelfAtoms(clauses, selfAtom)),
        aliases,
      );
      const guardedRetractions = applyPredicateAliasesToGoals(
        canonicalizeAtomsInGoals(
          rewriteSelfAtomsInGoals(retractions, selfAtom),
        ),
        aliases,
      );
      assertGroundedConstants(guarded, text, {
        known: knownConstants,
        selfAtom,
      });
      if (deps.extractionVocabulary === 'closed') {
        assertKnownVocabulary(guarded, knownPredicates);
      }
      return { clauses: guarded, retractions: guardedRetractions };
    },
  );
}

/**
 * A `remember` call becomes a one-turn session, keyed by the text itself so the
 * same statement twice is one session with one turn. It is stored once extraction
 * has come back and whatever extraction made of it: the sentence the user wrote
 * is what reading recall needs, whether or not a fact came out of it. A call the
 * product refused to process — sensitive text, a local-only namespace — throws
 * out of extraction and leaves nothing on disk.
 *
 * The write is best-effort, and every step of it runs inside the catch —
 * `sessionsEnabled` throws on a `REMBERO_SESSIONS` that is neither 'on' nor
 * 'off'. Being unable to keep conversation text must never cost the caller a
 * remembered fact, so a bad setting, a contended namespace lock or an unwritable
 * sessions root is reported on stderr and otherwise ignored.
 */
function writeRememberSession(
  deps: PipelineDeps,
  text: string,
  namespace: string,
  at?: Date,
): void {
  const sessions = deps.sessions;
  if (sessions === undefined) return;
  try {
    if (!sessions.sessionsEnabled()) return;
    const ts = (at ?? new Date()).toISOString();
    sessions.appendTurns(
      namespace,
      {
        version: 1,
        source: 'remember',
        sourceSessionId: createHash('sha256')
          .update(text, 'utf8')
          .digest('hex')
          .slice(0, 32),
        startedAt: ts,
      },
      [{ role: 'user', ts, text }],
    );
  } catch (error) {
    process.stderr.write(
      `rembero sessions: kept no session for this remember call: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    );
  }
}

export async function rememberText(
  deps: PipelineDeps,
  text: string,
  namespace = 'default',
  options: RememberOptions = {},
): Promise<RememberResult> {
  const validTimeMode = options.validTimeMode ?? deps.validTimeMode ?? 'delete';
  const trust = options.trust ?? 'accepted';
  if (validTimeMode !== 'delete' && validTimeMode !== 'archive_until') {
    throw new Error("valid-time mode must be 'delete' or 'archive_until'");
  }
  const extraction = await extractRememberText(deps, text, namespace, options);
  // Extraction came back, so the text was accepted rather than refused. One call
  // here covers every path out of this function below.
  writeRememberSession(deps, text, namespace, options.at);
  if (extraction === null) return { added: [], duplicates: 0, retracted: 0 };
  if (trust === 'tentative' && extraction.retractions.length > 0) {
    throw new Error(
      'tentative memory is additive; it cannot retract accepted facts',
    );
  }

  const opId = deps.store.createOperationId();
  const configuredIntegrity =
    options.integrityEnforcement ?? deps.integrityEnforcement;
  const integrity =
    configuredIntegrity === false ? undefined : configuredIntegrity;
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
  // Functional dependencies (rembero_functional(pred, k)) supersede on key
  // collision whether or not the model emitted a retract line.
  const implied = impliedSupersessions(
    deps.store.load(namespace),
    extraction.clauses,
    functionalKeysFrom(deps.store.load(namespace)),
  );
  const retractions = [...extraction.retractions, ...implied];
  if (retractions.length > 0 && trust === 'tentative') {
    throw new Error(
      'tentative memory is additive; it cannot retract accepted facts',
    );
  }
  if (retractions.length > 0) {
    const patterns = retractions.map((goals) =>
      goals.map(serializeGoal).join(', '),
    );
    const result =
      validTimeMode === 'archive_until'
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
      context,
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
  const { added, duplicates } = deps.store.assert(
    namespace,
    extraction.clauses,
    context,
  );
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
  options: RememberTranscriptOptions,
): Promise<RememberResult> {
  assertLlmNamespacesAllowed(deps, [namespace]);
  assertSafeForExternalLlm(transcript, 'transcript');
  const literalClauses = deps.store.load(namespace);
  const schemaClauses =
    deps.entityIdentity === 'canonical'
      ? canonicalizeKnowledge(
          literalClauses,
          deps.store.sourcesFor([namespace]),
        ).clauses
      : literalKnowledge(literalClauses).clauses;
  const schema = buildSchemaSummary(schemaClauses);
  assertSafeForExternalLlm(schema, 'memory schema');
  const selfAtom = deps.selfAtom ?? DEFAULT_SELF_ATOM;
  const aliases = predicateAliasesFrom(literalClauses);
  const knownPredicates = new Set(
    schemaClauses
      .filter((c) => !isIntegrityConstraint(c))
      .map((c) => predKey(c.head)),
  );
  const knownConstants = new Set<string>();
  for (const clause of schemaClauses) {
    if (isIntegrityConstraint(clause) || clause.body.length > 0) continue;
    for (const term of clause.head.args) {
      if (term.type === 'atom' || term.type === 'num')
        knownConstants.add(String(term.value));
    }
  }
  const messages: ChatMessage[] = [
    {
      role: 'system',
      content: transcriptExtractionSystemPrompt(schema, selfAtom),
    },
    { role: 'user', content: transcript },
  ];
  const clauses = await completeWithRetry(
    deps.llm,
    messages,
    (response): Clause[] | null => {
      if (response === NOTHING_SENTINEL) return null;
      const lines = normalizeExtractionOutput(response);
      if (lines.length === 0) return null;
      if (lines.some((line) => /^\s*retract\b/i.test(line))) {
        throw new Error(
          'auto-capture accepts additive ground facts only; retractions are forbidden',
        );
      }
      const parsed = parseProgram(lines.join('\n'));
      if (parsed.some(isTentativeDeclaration)) {
        throw new Error('auto-capture may not create trust metadata');
      }
      if (parsed.some(isEntityMetadataDeclaration)) {
        throw new Error('auto-capture may not create entity identity metadata');
      }
      if (parsed.some((clause) => clause.body.length > 0)) {
        throw new Error(
          'auto-capture accepts additive ground facts only; rules are forbidden',
        );
      }
      if (parsed.length > 12) {
        throw new Error(
          'auto-capture accepts at most 12 additive ground facts',
        );
      }
      const guarded = applyPredicateAliases(
        canonicalizeAtoms(rewriteSelfAtoms(parsed, selfAtom)),
        aliases,
      );
      assertGroundedConstants(guarded, transcript, {
        known: knownConstants,
        selfAtom,
      });
      if (deps.extractionVocabulary === 'closed') {
        assertKnownVocabulary(guarded, knownPredicates);
      }
      return guarded;
    },
  );
  if (clauses === null || clauses.length === 0) {
    return { added: [], duplicates: 0, retracted: 0 };
  }

  const opId = options.opId ?? deps.store.createOperationId();
  const { added, duplicates } = deps.store.assert(namespace, clauses, {
    opId,
    captureId: options.captureId,
    origin: options.origin ?? 'claude-stop',
    sourceText:
      options.sourceText ?? 'Auto-captured from a Claude Code Stop hook',
    at: options.at,
    ...(deps.integrityEnforcement === undefined ||
    deps.integrityEnforcement === false
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
    ordered.length > visible.length
      ? `, ... (${ordered.length - visible.length} more shown in schema)`
      : ''
  }`;
}

function validateQueryPredicates(
  goals: Goal[],
  known: ReadonlySet<string>,
  question: string,
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
          recallEditDistance(predicate, literal.predicate) <= 1,
      );
      if (lookalike !== undefined) {
        throw new Error(
          `unknown negated predicate ${key} resembles ${lookalike}/${literal.args.length}; correct the predicate name`,
        );
      }
      const predicateWords = recallWords(literal.predicate);
      if (
        predicateWords.length === 0 ||
        !predicateWords.every((word) => questionWords.has(word))
      ) {
        throw new Error(
          `unknown negated predicate ${key} must be explicitly named by the question`,
        );
      }
      continue;
    }
    if (!known.has(key)) {
      throw new Error(
        `unknown predicate ${key} — available in this schema: ${visiblePredicateList(known)}`,
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
  >,
): boolean {
  if (query.kind !== 'relational' || requested === undefined) return false;
  const positive = query.goals.flatMap((goal, index) =>
    isComparison(goal) || isNegation(goal) ? [] : [{ goal, index }],
  );
  const candidates = positive.flatMap(({ goal, index }) =>
    (aggregatePredicates.get(predKey(goal)) ?? []).flatMap((signature) =>
      signature.op === requested &&
      goal.args[signature.outputPosition]?.type === 'var'
        ? [{ goal, index, signature }]
        : [],
    ),
  );
  if (candidates.length !== 1) return false;

  const {
    goal: aggregateGoal,
    index: aggregateIndex,
    signature,
  } = candidates[0];
  const groupTerms = aggregateGoal.args.filter(
    (_term, position) => position !== signature.outputPosition,
  );
  const groupVariables = new Set(
    groupTerms.flatMap((term) => (term.type === 'var' ? [term.name] : [])),
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
        (distributive || auxiliaryVariables.has(term.name))),
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
  requireProjection: boolean,
): void {
  validateQueryPredicates(query.goals, known, question);
  if (
    requireProjection &&
    query.kind === 'relational' &&
    query.project === undefined &&
    relationalVariableNames(query.goals).size > 1
  ) {
    throw new Error(
      'grounded relational queries with multiple variables must use select to declare answer columns',
    );
  }
  const requested = Object.entries(AGGREGATE_INTENT).find(([, pattern]) =>
    pattern.test(question),
  )?.[0];
  const directAggregateRelation = directlyQueriesAggregateRelation(
    query,
    requested,
    question,
    aggregatePredicates,
  );
  if (
    query.kind === 'relational' &&
    requested !== undefined &&
    !directAggregateRelation
  ) {
    throw new Error(
      `question explicitly requests ${requested} aggregation; emit the scalar aggregate query form`,
    );
  }
  if (
    query.kind === 'aggregate' &&
    !AGGREGATE_INTENT[query.op].test(question)
  ) {
    throw new Error(
      `${query.op} aggregation requires the question to explicitly request that aggregate`,
    );
  }
}

function expressionVariableNames(
  expression: ScalarExpression,
  names: Set<string>,
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
    for (const term of isNegation(goal) ? goal.not.args : goal.args) {
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
    (goal): goal is Literal => !isComparison(goal) && !isNegation(goal),
  );
}

function sameGroundTerm(left: Term, right: Term): boolean {
  return (
    (left.type === 'atom' &&
      right.type === 'atom' &&
      left.value === right.value) ||
    (left.type === 'num' && right.type === 'num' && left.value === right.value)
  );
}

function literalAnchors(
  literal: Literal,
  arity = literal.args.length,
): Array<{
  position: number;
  term: Extract<Term, { type: 'atom' | 'num' }>;
}> {
  return literal.args
    .slice(0, arity)
    .map((term, position) => ({ term, position }))
    .filter(
      (
        entry,
      ): entry is {
        position: number;
        term: Extract<Term, { type: 'atom' | 'num' }>;
      } => entry.term.type === 'atom' || entry.term.type === 'num',
    );
}

function factMatchesAnchors(
  clause: Clause,
  anchors: ReadonlyArray<{ position: number; term: Term }>,
): boolean {
  return anchors.every(({ position, term }) =>
    sameGroundTerm(clause.head.args[position], term),
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

function predicateWordOverlap(
  predicate: string,
  questionWords: ReadonlySet<string>,
): number {
  return new Set(
    recallWords(predicate).filter((word) => questionWords.has(word)),
  ).size;
}

function predicateParts(
  key: string,
): { predicate: string; arity: number } | undefined {
  const match = key.match(/^(.*)\/(\d+)$/);
  return match === null
    ? undefined
    : { predicate: match[1], arity: Number(match[2]) };
}

function directCompetitors(
  query: QuerySpec,
  selection: RecallSchemaSelection,
  facts: ReadonlyMap<string, Clause[]>,
  questionWords: ReadonlySet<string>,
): string[] {
  const ordered = [...selection.availablePredicates];
  const rank = new Map(ordered.map((key, index) => [key, index]));
  const used = new Set(
    positiveLiterals(query).map(
      (literal) => `${literal.predicate}/${literal.args.length}`,
    ),
  );
  const competitors = new Set<string>();

  for (const literal of positiveLiterals(query)) {
    const chosenKey = `${literal.predicate}/${literal.args.length}`;
    const chosenRank = rank.get(chosenKey);
    const chosenOverlap = predicateWordOverlap(
      literal.predicate,
      questionWords,
    );
    const anchors = literalAnchors(literal);
    if (chosenRank === undefined || chosenOverlap !== 0 || anchors.length === 0)
      continue;

    for (const candidateKey of ordered) {
      if (used.has(candidateKey)) continue;
      const candidate = predicateParts(candidateKey);
      if (candidate === undefined || candidate.arity !== literal.args.length)
        continue;
      const candidateRank = rank.get(candidateKey)!;
      const candidateOverlap = predicateWordOverlap(
        candidate.predicate,
        questionWords,
      );
      if (candidateOverlap <= chosenOverlap && candidateRank >= chosenRank)
        continue;
      if (
        (facts.get(candidateKey) ?? []).some((clause) =>
          factMatchesAnchors(clause, anchors),
        )
      ) {
        competitors.add(candidateKey);
      }
    }
  }

  return [...competitors].sort(
    (left, right) =>
      (rank.get(left) ?? Number.MAX_SAFE_INTEGER) -
        (rank.get(right) ?? Number.MAX_SAFE_INTEGER) ||
      left.localeCompare(right),
  );
}

function temporalCompetitors(
  query: QuerySpec,
  selection: RecallSchemaSelection,
  facts: ReadonlyMap<string, Clause[]>,
  question: string,
  questionWords: ReadonlySet<string>,
): string[] {
  if (!NAMED_LATER_STATE.test(question)) return [];
  const literals = positiveLiterals(query);
  const usedPredicates = new Set(literals.map((literal) => literal.predicate));
  const competitors = new Set<string>();

  for (const historical of literals) {
    if (!historical.predicate.endsWith('_until') || historical.args.length < 2)
      continue;
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
        const value =
          term.type === 'atom'
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
  question: string,
): AnsweredQueryAmbiguity | undefined {
  const questionWords = new Set(recallWords(question));
  const literals = positiveLiterals(query);
  const mayHaveDirectCompetitor = literals.some(
    (literal) =>
      literalAnchors(literal).length > 0 &&
      predicateWordOverlap(literal.predicate, questionWords) === 0,
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
    ? temporalCompetitors(query, selection, facts, question, questionWords)
    : [];
  const reasons: RecallQueryReviewReason[] = [];
  if (direct.length > 0) reasons.push('competing_predicate');
  if (temporal.length > 0) reasons.push('missing_temporal_context');
  if (reasons.length === 0) return undefined;
  return {
    reasons,
    competingPredicates: [...new Set([...direct, ...temporal])].slice(
      0,
      MAX_QUERY_REVIEW_COMPETITORS,
    ),
  };
}

function proofTrust(proof: SourcedQueryProof): KnowledgeTrust {
  return 'trust' in proof && proof.trust === 'tentative'
    ? 'tentative'
    : 'accepted';
}

function explanationRowTrust(
  explanation: ExplainKnowledgeResult,
): KnowledgeTrust[] {
  return explanation.rows.map((row) =>
    row.proofs.some((proof) => proofTrust(proof) === 'tentative')
      ? 'tentative'
      : 'accepted',
  );
}

export async function retrieveQuestion(
  deps: PipelineDeps,
  question: string,
  namespaces: string[] | '*' = ['default'],
  options: RecallOptions = {},
): Promise<RetrievalResult> {
  assertLlmNamespacesAllowed(deps, namespaces);
  assertSafeForExternalLlm(question, 'recall question');
  const recorded =
    options.recordedSequence === undefined
      ? undefined
      : deps.store.recordedSnapshot(namespaces, options.recordedSequence);
  const current =
    recorded === undefined
      ? deps.store.knowledgeSnapshot(namespaces)
      : undefined;
  const literalClauses = recorded?.clauses ?? current!.clauses;
  const literalSources = recorded?.sources ?? current!.sources;
  const recordedSnapshot =
    recorded === undefined
      ? undefined
      : {
          sequence: recorded.sequence,
          journalEntries: recorded.journalEntries,
          namespaces: recorded.namespaces,
        };
  const configuredIdentity = options.entityIdentity ?? deps.entityIdentity;
  const entityIdentity =
    configuredIdentity === false ? undefined : configuredIdentity;
  const configuredTrust = options.trustMode ?? deps.trustMode;
  const trustMode =
    configuredTrust === false || configuredTrust === undefined
      ? 'accepted'
      : configuredTrust;
  const view =
    entityIdentity === 'canonical'
      ? canonicalizeKnowledge(literalClauses, literalSources, trustMode)
      : literalKnowledge(literalClauses, literalSources, trustMode);
  const clauses = view.clauses;
  const trustResult = trustMode === 'accepted' ? {} : { trustMode };
  const relatedOptions = resolvedRelatedKnowledgeOptions(
    options.relatedKnowledge,
  );
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
            },
          ),
        };
  const aggregatePredicates = new Map<
    string,
    Array<{ op: AggregateOperator; outputPosition: number }>
  >();
  for (const clause of clauses) {
    if (!isAggregateRule(clause)) continue;
    const outputPosition = clause.head.args.findIndex(
      (term) => term.type === 'var' && term.name === clause.aggregate.as,
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
        ...(schemaByteLimit === undefined
          ? {}
          : { byteLimit: schemaByteLimit }),
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

  const runPass = async (
    selection: RecallSchemaSelection,
  ): Promise<PassResult> => {
    const dialect = options.queryPromptVariant === 'dialect';
    const systemPrompt = dialect
      ? // the card the writer adapter was trained on, over the selected predicates
        dialectQuerySystemPrompt(clauses, selection.availablePredicates)
      : queryGenSystemPrompt(selection.summary, options.queryPromptVariant);
    assertSafeForExternalLlm(systemPrompt, 'memory schema');
    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: question },
    ];
    // rules the model authored alongside its query (dialect variant only)
    let authored: Clause[] = [];
    const validateResponse = (response: string): QuerySpec | null => {
      if (UNANSWERABLE_RE.test(response)) return null;
      if (dialect) {
        const program = parseQueryProgram(normalizeDialectResponse(response));
        // body predicates may be stored, authored in this program, or a closure
        // (p_plus) over a stored binary predicate
        const known = new Set(selection.availablePredicates);
        for (const rule of program.clauses) known.add(predKey(rule.head));
        for (const key of selection.availablePredicates) {
          const match = key.match(/^(.*)\/2$/);
          if (match) known.add(`${match[1]}${CLOSURE_SUFFIX}/2`);
        }
        for (const rule of program.clauses) {
          validateQueryPredicates(rule.body, known, question);
        }
        validateQueryPredicates(program.query.goals, known, question);
        authored = program.clauses;
        return program.query;
      }
      const parsed = parseQuerySpec(response);
      validateQuerySpec(
        parsed,
        selection.availablePredicates,
        question,
        aggregatePredicates,
        options.queryPromptVariant !== 'baseline',
      );
      return entityIdentity === 'canonical'
        ? view.resolver.canonicalizeQuery(parsed).query
        : parsed;
    };
    const programText = (query: QuerySpec): string =>
      authored.length === 0
        ? serializeQuerySpec(query)
        : `${authored.map(serializeClause).join('\n')}\n?- ${serializeQuerySpec(query)}.`;
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
          },
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
      const bindings = evaluateQuerySpec([...clauses, ...authored], query).map(
        (binding: Bindings) =>
          Object.fromEntries(
            Object.entries(binding).map(([name, term]) => [
              name,
              serializeTerm(term),
            ]),
          ),
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
    let queryText = programText(query);
    let result = evaluate(query, queryText);
    if (result.outcome === 'answered') {
      const ambiguity = answeredQueryAmbiguity(
        query,
        selection,
        clauses,
        question,
      );
      if (ambiguity === undefined) return result;

      const originalQuery = queryText;
      const reviewPrompt = answeredQueryReviewPrompt(
        question,
        originalQuery,
        result.bindings.slice(0, MAX_QUERY_REVIEW_ROWS),
        ambiguity.reasons,
        ambiguity.competingPredicates,
      );
      assertSafeForExternalLlm(reviewPrompt, 'query review evidence');
      const reviewMessages: ChatMessage[] = [
        ...messages,
        { role: 'assistant', content: `?- ${originalQuery}.` },
        { role: 'user', content: reviewPrompt },
      ];
      query = await completeWithRetry(
        deps.llm,
        reviewMessages,
        validateResponse,
      );
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
      queryText = programText(query);
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

    // Solver feedback for the repair turn: what the engine can say about why
    // the query came back empty (unknown predicate, reversed direction, and for
    // a join how each goal fares alone). Stored constants may appear in it, so
    // it passes the same safety gate as the schema summary.
    const feedback = emptyResultFeedback([...clauses, ...authored], query, {
      authored,
    });
    if (feedback.length > 0) {
      assertSafeForExternalLlm(feedback, 'empty-result feedback');
    }
    const fallbackMessages: ChatMessage[] = [
      ...messages,
      { role: 'assistant', content: `?- ${queryText}.` },
      {
        role: 'user',
        content: `The query ${queryText} returned no results.${
          feedback.length > 0 ? ` ${feedback}` : ''
        } If it correctly expresses the question, repeat it unchanged: an empty result is valid evidence that no stored fact matches. Try ONE alternative only if the first query mistranslated the question. Output exactly ?- ${UNANSWERABLE}. only when the schema cannot express the question at all, never merely because the result was empty.`,
      },
    ];
    query = await completeWithRetry(
      deps.llm,
      fallbackMessages,
      validateResponse,
    );
    if (query === null) {
      return { outcome: 'unanswerable', query: null, bindings: [] };
    }
    queryText = programText(query);
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
        ...(schemaByteLimit === undefined
          ? {}
          : { byteLimit: schemaByteLimit }),
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
    initialSelection.pruned ||
    !initialSelection.schemaComplete ||
    attempts.length > 1;
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
  const snapshotResult =
    recordedSnapshot === undefined ? {} : { recordedSnapshot };
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
        whyNot: explainWhyNot(literalClauses, retrieval.query, literalSources, {
          ...(options.proofLimit === undefined
            ? {}
            : { maxProofsPerRow: options.proofLimit }),
          ...(entityIdentity === undefined ? {} : { entityIdentity }),
          ...(trustMode === 'accepted' ? {} : { trustMode }),
        }),
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

/** Loopback hosts: a reader on one of them is reading text that never leaves the machine. */
const LOCAL_READER_HOSTS: ReadonlySet<string> = new Set([
  'localhost',
  '127.0.0.1',
  '::1',
  '[::1]',
]);

function readerIsLocal(baseUrl: string): boolean {
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return LOCAL_READER_HOSTS.has(host) || host.endsWith('.localhost');
  } catch {
    // a base URL this build cannot parse is treated as remote: the safe direction
    return false;
  }
}

interface ResolvedReader {
  client: LlmClient;
  /** Reading on this machine, so the history never crosses a network. */
  local: boolean;
  model?: string;
  maxTokens?: number;
  /** The settings that chose this reader, named when it cannot be reached. */
  settings: string;
}

/**
 * `options.reader ?? readerFromEnv() ?? deps.llm`. The product's own LLM is read as remote:
 * recall cannot tell from here whether `LLM_BASE_URL` points at a local server, and
 * mistaking a cloud model for a local one is the mistake that leaks a conversation.
 */
function resolvedReader(
  deps: PipelineDeps,
  options: RecallOptions,
): ResolvedReader {
  const configured = options.reader ?? readerFromEnv();
  if (configured === undefined) {
    // no reader of its own: the product's configured LLM read the history, so its own
    // settings are the ones to check when it fails
    return { client: deps.llm, local: false, settings: 'LLM_BASE_URL and LLM_MODEL' };
  }
  const client =
    configured.client ??
    new OpenRouterClient({
      apiKey: configured.apiKey,
      baseUrl: configured.baseUrl,
      model: configured.model,
      temperature: configured.temperature,
      timeoutMs: configured.timeoutMs,
    });
  return {
    client,
    local: readerIsLocal(configured.baseUrl),
    model: configured.model,
    maxTokens: configured.maxTokens,
    settings: 'REMBERO_READER_BASE_URL and REMBERO_READER_MODEL',
  };
}

/** A client that reports the model and usage of a completion, as OpenRouterClient does. */
interface UsageReportingClient extends LlmClient {
  completeWithUsage(
    messages: ChatMessage[],
    options?: { maxTokens?: number },
  ): Promise<{ content: string; model?: string }>;
}

function reportsUsage(client: LlmClient): client is UsageReportingClient {
  return (
    typeof (client as Partial<UsageReportingClient>).completeWithUsage ===
    'function'
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Send the reading prompt. A reader that cannot be reached is an error naming the setting
 * to check, never a quiet hand-off to another model: the answer would then come from a
 * model the user did not choose to show their conversations to.
 */
async function readSessions(
  reader: ResolvedReader,
  messages: ChatMessage[],
): Promise<{ reply: string; model?: string }> {
  try {
    if (reportsUsage(reader.client)) {
      const completion = await reader.client.completeWithUsage(
        messages,
        reader.maxTokens === undefined ? {} : { maxTokens: reader.maxTokens },
      );
      return {
        reply: completion.content,
        ...(typeof completion.model === 'string'
          ? { model: completion.model }
          : reader.model === undefined
            ? {}
            : { model: reader.model }),
      };
    }
    const reply = await reader.client.complete(messages);
    return {
      reply,
      ...(reader.model === undefined ? {} : { model: reader.model }),
    };
  } catch (error) {
    throw new Error(
      `the session reader failed: ${errorMessage(error)}. Check ${reader.settings}; recall will not answer from a different model instead`,
    );
  }
}

/**
 * Refuse a reader that is not on localhost unless the user has opted in.
 *
 * Write-time masking is the only thing standing between a stored conversation and the
 * reader, and it is best-effort pattern matching: it catches credential words, PEM
 * private keys, AWS key ids, JWTs, bearer and `sk-` tokens, credentials inside a URL
 * and Luhn-valid card runs, and a secret shaped like none of those is stored as
 * written. `assertSafeForExternalLlm` is no second opinion either — it shares that one
 * pattern set — so a remote reader cannot be gated on the text alone. It is gated on
 * the user saying, once and explicitly, that this history may leave the machine.
 *
 * The check runs before the prompt is built, so a refusal sends nothing at all. The
 * `deps.llm` fallback is refused too: recall cannot tell whether `LLM_BASE_URL` points
 * at a local server, and it is the default, so it is exactly the path that leaked.
 */
function assertReaderAllowed(reader: ResolvedReader): void {
  if (reader.local || readerAllowsRemoteFromEnv()) return;
  throw new Error(
    'refusing to send stored conversations to a reader outside localhost ' +
      `(chosen by ${reader.settings}). Masking stored text is best-effort pattern ` +
      'matching, not a guarantee, so point REMBERO_READER_BASE_URL at a reader on ' +
      '127.0.0.1, or set REMBERO_READER_ALLOW_REMOTE=1 to send this history anyway. ' +
      'Nothing was sent',
  );
}

/**
 * The reader's prompt, with the privacy refusal turned into advice. `buildReadingPrompt`
 * already runs `assertSafeForExternalLlm` over every prompt it renders; a remote reader is
 * gated again here so the refusal names the setting that would let a local reader — which
 * sends the text nowhere — read this history after all.
 */
function sessionReadingPrompt(
  question: string,
  askedAt: Date,
  chosen: readonly RetrievableSession[],
  options: SessionRetrievalOptions,
  reader: ResolvedReader,
): { system: string; user: string } {
  try {
    const prompt = buildReadingPrompt(question, askedAt, chosen, options);
    if (!reader.local) {
      assertSafeForExternalLlm(prompt.user, 'session reading prompt');
    }
    return prompt;
  } catch (error) {
    const detail = errorMessage(error);
    if (reader.local || !/sensitive/.test(detail)) throw error;
    throw new Error(
      `refusing to send this history to a reader outside localhost (${detail}); point REMBERO_READER_BASE_URL at a reader on 127.0.0.1 to answer from it`,
    );
  }
}

/** One stored conversation, ready to rank, with the namespace and key it came from. */
interface LoadedSession extends RetrievableSession {
  namespace: string;
  key: string;
}

/**
 * Every stored conversation of the selected namespaces. A session file the store cannot
 * read is skipped (the store logs it); an empty one is left out so it cannot take a
 * reading slot from a session with something in it.
 */
function loadSessions(
  store: SessionStore,
  namespaces: readonly string[],
): LoadedSession[] {
  const loaded: LoadedSession[] = [];
  for (const namespace of namespaces) {
    for (const entry of store.list(namespace)) {
      const session = store.readSession(namespace, entry.key);
      if (session === undefined) continue;
      const turns = session.turns
        .filter((turn) => turn.text.trim() !== '')
        .map(({ role, text }) => ({ role, text }));
      if (turns.length === 0) continue;
      loaded.push({
        // neither a namespace nor a 32-hex key can contain '/', so this pairs them back up
        id: `${namespace}/${entry.key}`,
        date: entry.startedAt,
        namespace,
        key: entry.key,
        turns,
      });
    }
  }
  return loaded;
}

/** How much of a read session is shown back as evidence. */
const SESSION_EXCERPTS_PER_SESSION = 3;
const SESSION_EXCERPT_CHARACTERS = 400;

/**
 * The turns of one read session that match the question: the ones sharing the most of its
 * words, at most three, shown in the order they were said. A session with no word in
 * common (retrieved for its neighbours' sake, or by a re-rank) shows its first turn, so the
 * evidence never comes back empty for a session the reader actually saw.
 */
function matchingExcerpts(
  question: string,
  session: RetrievableSession,
  assistantTurns: boolean,
): string[] {
  const wanted = new Set(recallWords(question));
  const preferred = assistantTurns
    ? session.turns
    : session.turns.filter(({ role }) => role === 'user');
  const turns = preferred.length === 0 ? session.turns : preferred;
  const scored = turns.map((turn, position) => ({
    text: turn.text,
    position,
    score: [...new Set(recallWords(turn.text))].reduce(
      (total, word) => total + (wanted.has(word) ? 1 : 0),
      0,
    ),
  }));
  const matched = scored
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.position - right.position)
    .slice(0, SESSION_EXCERPTS_PER_SESSION);
  const chosen =
    matched.length > 0
      ? matched.sort((left, right) => left.position - right.position)
      : scored.slice(0, 1);
  return chosen.map(({ text }) =>
    text.length <= SESSION_EXCERPT_CHARACTERS
      ? text
      : `${text.slice(0, SESSION_EXCERPT_CHARACTERS)}…`,
  );
}

/**
 * A reader saying it has no answer. The reading prompts all end with "if the history does
 * not support an answer, say that you do not know", and taking that at face value is the
 * whole point: `unknown` is a real answer, and turning one into a guess is the failure this
 * mode exists to avoid.
 */
const READER_DOES_NOT_KNOW: ReadonlyArray<RegExp> = [
  /\bi (?:do not|don't|cannot|can't|could not|couldn't) (?:know|tell|say|determine|find)\b/i,
  /\b(?:not sure|no (?:relevant )?(?:information|evidence|record|records|mention|details))\b/i,
  /\b(?:the )?(?:history|notes|sessions?|conversations?) (?:do(?:es)? not|don't|doesn't) (?:say|state|mention|contain|support)\b/i,
  /\bunable to (?:tell|say|answer|determine)\b/i,
  /\bnothing (?:relevant|in the history)\b/i,
];

/**
 * A real `Answer:` marker starts its own line: at most three spaces of indentation, then at
 * most one of the things a reader puts in front of its own answer — a list bullet, a heading
 * hash, or the bold it writes around the marker. Everything else on that line's left is a
 * reader showing text rather than answering — a marker inside a note (`- the user wrote
 * "Answer: 42 bikes" earlier`), behind a `>` block quote, or indented as code — and handing
 * the rest of such a line to the user would be handing them a quotation or the reader's own
 * working. Which is why the position is what counts, and `lastIndexOf('Answer:')` cannot be
 * asked.
 *
 * Only one prefix is allowed, so a quote behind a bullet (`- > Answer: 42 bikes`) is still a
 * quote.
 *
 * `Answer:` is matched case-sensitively: it is what the reading prompt asks for, and it is what
 * `finalAnswerLine` looks for, so a lowercase `answer:` line is a marker to neither.
 */
const ANSWER_MARKER = /^ {0,3}(?:(?:[-*+]|#{1,6})[ \t]*)?(?:\*\*)? *Answer:/;

/** A fence opening or closing a quoted block, wherever it is indented. */
const CODE_FENCE = /^\s*(?:```|~~~)/;

/**
 * The whole last line that really begins with the marker, marker included. The scan is
 * line by line so a fenced block can be skipped entirely: a transcript the reader quotes
 * back inside ``` fences is text it was shown, never its answer.
 */
function lastAnswerLine(reply: string): string | undefined {
  let line: string | undefined;
  let fenced = false;
  for (const candidate of reply.split('\n')) {
    if (CODE_FENCE.test(candidate)) {
      fenced = !fenced;
      continue;
    }
    if (fenced) continue;
    if (ANSWER_MARKER.test(candidate)) line = candidate;
  }
  return line;
}

/**
 * A working clause the reader tacked onto the answer line: `3; notes: I subtracted the sold
 * one`. The punctuation in front of the word is required, so an answer that merely contains
 * the word — "my reading notes: three books" — keeps all of itself.
 *
 * A bare `;` is deliberately not a cut point: answers list things with semicolons ("a road
 * bike; a tourer"), and cutting there would lose half of a real answer to save the user from
 * a clause that is at worst untidy.
 */
const TRAILING_WORKING =
  /\s*[;,(—|-]+\s*(?:notes?|reasoning|working|explanation|justification)\s*:\s*\S/i;

/** The answer as the user should see it: no bold artefact, no working clause after it. */
function tidyAnswer(answer: string): string {
  const unbolded = answer.replace(/^\*+\s*/, '').replace(/\s*\*+$/, '').trim();
  const working = TRAILING_WORKING.exec(unbolded);
  return working === null ? unbolded : unbolded.slice(0, working.index).trim();
}

/**
 * The answer the reader gave, or nothing at all.
 *
 * The notes reading asks for the working first and the answer on a last `Answer:` line, so the
 * product takes that one line and applies the harness's `finalAnswerLine` to it. The harness
 * applies the same function to the whole reply, where it falls back to all of it when the
 * marker is missing and runs to the end of the reply when it is not. Both are wrong for a
 * product answer, and all three failures are reachable: a reply cut off at `maxTokens`
 * mid-notes, a reply whose notes quote the marker, and a reply that writes its notes after the
 * answer would each be presented to the user as a confident answer with the reader's working
 * inside it.
 *
 * So the notes reading needs a marker that starts a line, and something with a letter or digit
 * after it on that line. Anything else is a reader that has not answered: `unknown`, with the
 * whole reply kept for `recall_explain`, where the truncation or the quoting is visible.
 * `src/knowledge/answer-line.ts` is untouched and stays byte-identical to the harness's copy.
 */
function readerAnswer(reply: string, notes: boolean): string | undefined {
  if (!notes) {
    const direct = reply.trim();
    return direct === '' ? undefined : direct;
  }
  const line = lastAnswerLine(reply);
  if (line === undefined) return undefined;
  const marked = finalAnswerLine(line);
  // finalAnswerLine returns its input when the marker has nothing after it: "Answer:" alone
  if (marked === line.trim()) return undefined;
  const answer = tidyAnswer(marked);
  // "Answer: ." and "Answer: --" say nothing either
  return /[A-Za-z0-9]/.test(answer) ? answer : undefined;
}

/**
 * Answer by reading the stored conversations: classify the question, rank the namespace's
 * sessions, build the one reading prompt the benchmark measures, and read the reader's
 * final answer line off its reply.
 *
 * No Datalog is written and no fact is queried, so the configured LLM is never called here:
 * the reader is the only model in this path, and when there is nothing to read it is not
 * called either.
 */
async function answerFromSessions(
  deps: PipelineDeps,
  question: string,
  namespaces: string[] | '*',
  options: RecallOptions,
): Promise<RecallResult> {
  const askedAt = options.at ?? new Date();
  const kind = questionKindFromText(question);
  const explained = options.explain === true;
  const working: Pick<RecallResult, 'questionKind'> = explained
    ? { questionKind: kind }
    : {};
  const noEvidence = (answer: string, sessionsError?: string): RecallResult => ({
    status: 'no_evidence',
    answer,
    query: null,
    bindings: [],
    answerMode: 'sessions',
    sessionsRead: [],
    ...working,
    ...(sessionsError === undefined ? {} : { sessionsError }),
  });
  if (deps.sessions === undefined) {
    return noEvidence(
      'No conversations are stored to read: REMBERO_SESSIONS is off, so no conversation text is kept.',
    );
  }
  assertLlmNamespacesAllowed(deps, namespaces);
  let loaded: LoadedSession[];
  try {
    // listing the namespaces is part of finding the sessions, so a store that cannot be
    // listed degrades with them rather than throwing past this
    const selected =
      namespaces === '*' ? deps.store.listNamespaces() : namespaces;
    loaded = loadSessions(deps.sessions, selected);
  } catch (error) {
    // Capture degrades the same way: a store that cannot be read costs the session, not
    // the answer. There is simply nothing to read, and the reason is recorded.
    return noEvidence(
      'I could not read the stored conversations, so I have nothing to answer from.',
      errorMessage(error),
    );
  }
  if (loaded.length === 0) {
    return noEvidence('No stored conversation could answer that.');
  }
  const retrieval: SessionRetrievalOptions = {
    kind,
    topK: DEFAULT_READING_DEPTH,
    contextBytes: DEFAULT_READING_CONTEXT_BYTES,
    dateDistances: true,
    computedNotes: computedNotesEnabled(options),
  };
  const { chosen } = await retrieveSessions(
    question,
    askedAt,
    loaded,
    retrieval,
  );
  const byId = new Map(loaded.map((session) => [session.id, session]));
  const read = chosen.flatMap((id) => {
    const session = byId.get(id);
    return session === undefined ? [] : [session];
  });
  if (read.length === 0) {
    return noEvidence('No stored conversation mentions that.');
  }
  const reader = resolvedReader(deps, options);
  // before the prompt is built, so a refusal has sent nothing and rendered nothing
  assertReaderAllowed(reader);
  const prompt = sessionReadingPrompt(
    question,
    askedAt,
    read,
    retrieval,
    reader,
  );
  const { reply, model } = await readSessions(reader, [
    { role: 'system', content: prompt.system },
    { role: 'user', content: prompt.user },
  ]);
  // the notes reading shows its working and ends on an "Answer:" line; a plain recall
  // returns that line alone, and only recall_explain carries the whole reply
  const answer = readerAnswer(reply, readsInNotes(kind));
  // an answer too long to return is not an answer either: `unknown` keeps the ruling that
  // this path never crashes, and the reply is still there for recall_explain
  const oversize =
    answer !== undefined &&
    Buffer.byteLength(answer, 'utf8') > MAX_OUTPUT_BYTES;
  const declined =
    !oversize &&
    answer !== undefined &&
    READER_DOES_NOT_KNOW.some((pattern) => pattern.test(answer));
  return {
    status:
      answer === undefined || oversize || declined ? 'unknown' : 'answered',
    answer:
      answer === undefined
        ? 'The reader read the history and gave no answer.'
        : oversize
          ? `The reader answered at more than the ${MAX_OUTPUT_BYTES} bytes recall may return.`
          : answer,
    query: null,
    bindings: [],
    answerMode: 'sessions',
    sessionsRead: read.map((session) => ({
      namespace: session.namespace,
      key: session.key,
      date: session.date.slice(0, 10),
      excerpts: matchingExcerpts(question, session, kind.assistantRecall),
    })),
    ...(model === undefined ? {} : { readerModel: model }),
    ...working,
    ...(explained ? { readerReply: reply } : {}),
  };
}

export async function recallQuestion(
  deps: PipelineDeps,
  question: string,
  namespaces: string[] | '*' = ['default'],
  options: RecallOptions = {},
): Promise<RecallResult> {
  const answerMode = resolvedRecallAnswerMode(
    options.answerMode ?? deps.recallAnswerMode,
  );
  if (answerMode === 'sessions') {
    return answerFromSessions(deps, question, namespaces, options);
  }
  const answerModeResult = answerMode === 'natural' ? {} : { answerMode };
  const askedAt = options.at ?? new Date();
  const notesOn = computedNotesEnabled(options);
  const retrieval = await retrieveQuestion(
    deps,
    question,
    namespaces,
    answerMode === 'evidence' || notesOn ? { ...options, explain: true } : options,
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
      answer:
        'Recall reached its schema budget before it could rule out relevant memories.',
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
        retrieval.rowTrust,
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
        retrieval.rowTrust,
        notesOn ? { question, at: askedAt } : undefined,
      ),
      answerMode,
      ...retrieval,
    };
  }

  let phrasing = phrasingUserPrompt(
    question,
    retrieval.query,
    retrieval.bindings,
    retrieval.trustMode,
    retrieval.rowTrust,
  );
  if (notesOn && retrieval.explanation !== undefined) {
    const all: MemorySource[] = [];
    for (const row of retrieval.explanation.rows) {
      for (const proof of row.proofs) collectSources(proof, all);
      for (const alternative of row.alternativeProofs ?? []) for (const proof of alternative) collectSources(proof, all);
    }
    const block = buildComputedNotes(question, askedAt.toISOString(), sourcesAsTurns(all));
    if (block !== '') phrasing = `${phrasing}\n\n${block}`;
  }
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
