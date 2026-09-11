import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  addLlmUsage,
  emptyLlmUsageTotals,
  type ChatMessage,
  type LlmCompletion,
  type LlmUsage,
  type LlmUsageTotals,
} from '../llm/client.js';
import { recallWords } from '../llm/schema.js';
import { assertSafeForExternalLlm } from '../safety.js';
import {
  searchKnowledge,
  type KnowledgeSearchResult,
} from '../knowledge/search.js';
import {
  isRecommendationIntent,
  MemoryEmbeddingCache,
  prepareSemanticKnowledge,
  SEMANTIC_CHUNK_CHARACTERS,
  SEMANTIC_CHUNK_OVERLAP,
  semanticSearchKnowledge,
  type SemanticKnowledgeSearchResult,
} from '../knowledge/semantic-search.js';
import type { EmbeddingClient, EmbeddingUsage } from '../llm/embeddings.js';
import type { Clause } from '../engine/index.js';
import { rememberTranscriptText } from '../llm/pipeline.js';
import type { LlmClient } from '../llm/client.js';
import { engineRecall as runEngineRecall } from './engine-recall.js';
import {
  expandByEntities,
  interleaveSessions,
} from '../knowledge/entity-retrieval.js';
import { MemoryStore } from '../store/store.js';
import {
  longMemEvalSessionText,
  LONGMEMEVAL_S_COMMIT,
  LONGMEMEVAL_S_SHA256,
  scoreLongMemEvalRetrievedSessions,
  type LongMemEvalInstance,
  type LongMemEvalQuestionResult,
} from './longmemeval.js';

export const LONGMEMEVAL_ANSWER_VERSION =
  'remembero.longmemeval-answer.v1' as const;
export const DEFAULT_LONGMEMEVAL_ANSWER_TOP_K = 4;
export const DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K = 5;
export const DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K = 5;
export const DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES = 56 * 1024;
export const MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES = 160 * 1024;
export const LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS = 16_384;
export const LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE = 315;
export const DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES = new Set([
  'single-session-preference',
  'multi-session',
]);

/**
 * How haystack sessions become memory.
 *   raw        one placeholder fact per session carrying the transcript as source text
 *              (the pipeline's original form: retrieval is lexical over the raw text)
 *   extracted  the product's transcript extraction runs on every session; only the
 *              facts it writes exist, so a session with no facts is unretrievable
 *   hybrid     both: the placeholder keeps every session retrievable and the extracted
 *              facts add predicate/constant text for retrieval to match
 */
export type LongMemEvalFormation = 'raw' | 'extracted' | 'hybrid';
export const LONGMEMEVAL_FORMATION_LABELS = {
  raw: 'durable-raw-session-facts',
  extracted: 'extracted-transcript-facts',
  hybrid: 'raw-session-facts-plus-extracted',
} as const;
/** Sessions longer than this are cut before extraction (the served model has a context limit). */
export const DEFAULT_LONGMEMEVAL_EXTRACTION_CHARACTERS = 16_000;

export interface LongMemEvalExtractionStats {
  calls: number;
  sessionsWithFacts: number;
  facts: number;
  errors: number;
  /** Error messages, trimmed to their first clause, with counts. */
  errorKinds: Record<string, number>;
  usage: LlmUsage | null;
  /** Sessions whose extraction was replayed from the cache instead of calling the model. */
  cached?: number;
}

/** "constant 'x' is not in the input; store only..." -> "constant '…' is not in the input" */
export function extractionErrorKind(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message
    .split(/[;\n(]/)[0]!
    .replace(/'[^']*'/g, "'…'")
    .replace(/\b\d+\b/g, 'N')
    .trim()
    .slice(0, 90);
}

export interface LongMemEvalCompletionClient {
  readonly model: string;
  completeWithUsage(
    messages: ChatMessage[],
    options?: { maxTokens?: number },
  ): Promise<LlmCompletion>;
}

export interface LongMemEvalEngineRecall {
  status: 'answered' | 'empty' | 'unparsable' | 'error';
  query: string | null;
  rows: number;
  attempts: number;
  ms: number;
  usage: LlmUsage | null;
  error?: string;
}

export interface LongMemEvalAnswerObservation {
  questionId: string;
  questionType: string;
  abstention: boolean;
  status: 'judged' | 'error';
  correct: boolean | null;
  retrievedSessionIds: string[];
  contextSessionIds: string[];
  contextRoles: 'all' | 'user';
  redactedRetrievedSessions: number;
  retrievalRoute: 'local' | 'semantic';
  embeddingModel: string | null;
  embeddingCalls: number;
  embeddingUsage: EmbeddingUsage | null;
  semanticPreparationCalls: number;
  semanticPreparationUsage: EmbeddingUsage | null;
  retrieval: LongMemEvalQuestionResult | null;
  context: LongMemEvalQuestionResult | null;
  hypothesis: string | null;
  judgeResponse: string | null;
  readerUsage: LlmUsage | null;
  judgeUsage: LlmUsage | null;
  /** Present when formation ran the product extraction. */
  extraction?: LongMemEvalExtractionStats;
  /** Present when a time-range extractor ran: the range it read off the question, or null. */
  temporalRange?: { start: string; end: string } | null;
  temporalRangeUsage?: LlmUsage | null;
  /** Present when the memory engine's own recall ran over the remembered facts. */
  engineRecall?: LongMemEvalEngineRecall;
  formationMs: number;
  semanticPreparationMs: number;
  retrievalMs: number;
  readerMs: number;
  judgeMs: number;
  totalMs: number;
  userTurnMs: number;
  error?: string;
}

export interface LongMemEvalAnswerSummary {
  questions: number;
  judgedQuestions: number;
  errors: number;
  correct: number;
  accuracy: number;
  judgedAccuracy: number;
  abstentionAccuracy: number;
  fullContextEvidenceAccuracy: number;
  incompleteContextEvidenceAccuracy: number;
  retrievalRecallAtK: number;
  contextRecallAtK: number;
  redactedRetrievedSessions: number;
  medianFormationMs: number;
  p95FormationMs: number;
  medianSemanticPreparationMs: number;
  p95SemanticPreparationMs: number;
  medianRetrievalMs: number;
  p95RetrievalMs: number;
  medianReaderMs: number;
  p95ReaderMs: number;
  medianJudgeMs: number;
  p95JudgeMs: number;
  medianTotalMs: number;
  p95TotalMs: number;
  medianUserTurnMs: number;
  p95UserTurnMs: number;
  readerUsage: LlmUsageTotals;
  judgeUsage: LlmUsageTotals;
  extractionUsage: LlmUsageTotals & {
    sessionsWithFacts: number;
    facts: number;
    errors: number;
    errorKinds: Record<string, number>;
  };
  embeddingUsage: {
    calls: number;
    promptTokens: number;
    totalTokens: number;
    costResponses: number;
    costUsd: number;
  };
  semanticPreparationUsage: {
    calls: number;
    promptTokens: number;
    totalTokens: number;
    costResponses: number;
    costUsd: number;
  };
}

export interface LongMemEvalAnswerRun {
  schemaVersion: typeof LONGMEMEVAL_ANSWER_VERSION;
  generatedAt: string;
  dataset: {
    id: 'xiaowu0162/longmemeval-cleaned';
    split: 'longmemeval_s_cleaned';
    selection: 'dev' | 'test' | 'all';
    commit: typeof LONGMEMEVAL_S_COMMIT;
    sha256: string;
  };
  readerModel: string;
  judgeModel: string;
  embeddingModel: string | null;
  judgeProtocol: 'longmemeval-official-compatible-v1';
  formation: (typeof LONGMEMEVAL_FORMATION_LABELS)[LongMemEvalFormation];
  extractionModel: string | null;
  /** Every knob that shaped the run, so a results file explains itself. */
  settings?: {
    aggregationReaderModel: string | null;
    temporalRangeModel: string | null;
    readingStrategy: 'direct' | 'notes' | 'two-call';
    hybridRetrieval: 'shared' | 'reserved' | 'keyed';
    retrievalUnit: 'session' | 'turn';
    entityRetrieval: boolean;
    engineRecall?: boolean;
    engineRecallQuestionTypes?: string[] | null;
    dateDistances?: boolean;
    hybridQuestionTypes: string[] | null;
    factsInContext: boolean;
    readerMaxTokens: number;
  };
  retrieval:
    'remembero-local-source-search' | 'remembero-adaptive-source-search';
  answerContextPolicy: 'user-turns-except-assistant-memory';
  semanticQuestionTypes: string[];
  multiSessionSemanticMaximumLexicalScore: number;
  semanticPreparation: 'cold' | 'prepared';
  topK: number;
  multiSessionTopK: number;
  temporalTopK: number;
  sourceCharacters: number;
  contextBytes: number;
  summary: LongMemEvalAnswerSummary;
  byQuestionType: Record<string, LongMemEvalAnswerSummary>;
  observations: LongMemEvalAnswerObservation[];
}

function mean(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function mergeEmbeddingUsage(
  left: EmbeddingUsage | null,
  right: EmbeddingUsage,
): EmbeddingUsage {
  const add = (a: number | null | undefined, b: number | null) =>
    a === null && b === null ? null : (a ?? 0) + (b ?? 0);
  return {
    promptTokens: add(left?.promptTokens, right.promptTokens),
    totalTokens: add(left?.totalTokens, right.totalTokens),
    costUsd: add(left?.costUsd, right.costUsd),
  };
}

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil((sorted.length - 1) * quantile)] ?? 0;
}

function boundedUtf8(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, 'utf8') <= maxBytes) return value;
  let low = 0;
  let high = value.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (Buffer.byteLength(value.slice(0, middle), 'utf8') <= maxBytes)
      low = middle;
    else high = middle - 1;
  }
  return value.slice(0, low);
}

function sourceWindow(
  text: string,
  question: string,
  maxBytes: number,
  focusCharacterOffset?: number,
): string {
  const boundedSource = text.slice(0, LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS);
  if (Buffer.byteLength(boundedSource, 'utf8') <= maxBytes)
    return boundedSource;
  if (focusCharacterOffset !== undefined) {
    const approximateCharacters = Math.max(
      1,
      Math.min(boundedSource.length, maxBytes),
    );
    const start = Math.max(
      0,
      Math.min(
        boundedSource.length - approximateCharacters,
        focusCharacterOffset -
          Math.floor((approximateCharacters - SEMANTIC_CHUNK_CHARACTERS) / 2),
      ),
    );
    return boundedUtf8(boundedSource.slice(start), maxBytes);
  }
  const words = [
    ...new Set(recallWords(question).filter((word) => word.length >= 3)),
  ];
  const approximateCharacters = Math.max(
    1,
    Math.min(boundedSource.length, maxBytes),
  );
  let bestStart = 0;
  let bestScore = -1;
  for (let start = 0; start < boundedSource.length; start += 512) {
    const candidate = boundedSource
      .slice(start, start + approximateCharacters)
      .toLowerCase();
    const score = words.reduce(
      (total, word) => total + (candidate.includes(word) ? 1 : 0),
      0,
    );
    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
    if (start + approximateCharacters >= boundedSource.length) break;
  }
  return boundedUtf8(boundedSource.slice(bestStart), maxBytes);
}

/** USER:/ASSISTANT: blocks, the shape the product's transcript capture and its training data use. */
export function longMemEvalTranscript(
  session: LongMemEvalInstance['haystack_sessions'][number],
  options: { assistantCharacters?: number } = {},
): string {
  // Assistant turns are ~87% of the characters and, by the extraction contract, never a
  // source of facts; keeping only their head leaves the user's words in a small prompt.
  const limit = options.assistantCharacters;
  return session
    .map(({ role, content }) => {
      const text =
        role !== 'user' && limit !== undefined && content.length > limit
          ? `${content.slice(0, limit)} […]`
          : content;
      return `${role === 'user' ? 'USER' : 'ASSISTANT'}: ${text}`;
    })
    .join('\n\n');
}

function sumLlmUsage(a: LlmUsage, b: LlmUsage): LlmUsage {
  const add = (x: number | null, y: number | null) =>
    x === null && y === null ? null : (x ?? 0) + (y ?? 0);
  return {
    promptTokens: add(a.promptTokens, b.promptTokens),
    completionTokens: add(a.completionTokens, b.completionTokens),
    totalTokens: add(a.totalTokens, b.totalTokens),
    cachedPromptTokens: add(a.cachedPromptTokens, b.cachedPromptTokens),
    reasoningTokens: add(a.reasoningTokens, b.reasoningTokens),
    costUsd: add(a.costUsd, b.costUsd),
  };
}

/** The text after the last "Answer:" marker, or the whole reply when there is none. */
export function finalAnswerLine(reply: string): string {
  const index = reply.lastIndexOf('Answer:');
  if (index < 0) return reply.trim();
  const answer = reply.slice(index + 'Answer:'.length).trim();
  return answer === '' ? reply.trim() : answer;
}

export function temporalRangePrompt(
  question: string,
  questionDate: string,
): string {
  return `Today is ${questionDate}. A user asks their assistant:

"${question}"

If the question refers to a specific period of time (for example "last month", "in March", "two weeks ago", "the week before my trip", "this year"), reply with the absolute date range it refers to as JSON: {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}. Be generous at the edges (widen by a few days). If the question carries no time cue at all, reply exactly {"none":true}. Reply with the JSON only.`;
}

/** First JSON object in the reply; a well-formed range or null. Anything else is null too. */
export function parseTemporalRange(
  reply: string,
): { start: string; end: string } | null {
  const match = /\{[^{}]*\}/.exec(reply);
  if (match === null) return null;
  try {
    const parsed = JSON.parse(match[0]) as {
      start?: unknown;
      end?: unknown;
      none?: unknown;
    };
    if (parsed.none === true) return null;
    const iso = /^\d{4}-\d{2}-\d{2}$/;
    if (
      typeof parsed.start === 'string' &&
      typeof parsed.end === 'string' &&
      iso.test(parsed.start) &&
      iso.test(parsed.end) &&
      parsed.start <= parsed.end
    ) {
      return { start: parsed.start, end: parsed.end };
    }
  } catch {
    // not JSON: no range
  }
  return null;
}

function datasetDate(value: string): Date {
  const match =
    /^(\d{4})\/(\d{2})\/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})$/.exec(value);
  if (match === null) throw new Error(`invalid LongMemEval date '${value}'`);
  return new Date(
    Date.UTC(
      Number(match[1]),
      Number(match[2]) - 1,
      Number(match[3]),
      Number(match[4]),
      Number(match[5]),
    ),
  );
}

function validateOptions(topK: number, contextBytes: number): void {
  if (!Number.isSafeInteger(topK) || topK < 1 || topK > 100) {
    throw new Error('LongMemEval answer topK must be an integer from 1 to 100');
  }
  if (
    !Number.isSafeInteger(contextBytes) ||
    contextBytes < 4_096 ||
    contextBytes > MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES
  ) {
    throw new Error(
      `LongMemEval answer context bytes must be an integer from 4096 to ${MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES}`,
    );
  }
}

interface AnswerContext {
  messages: ChatMessage[];
  contextSessionIds: string[];
  redactedRetrievedSessions: number;
}

const MAX_ENGINE_RENDER_CHARACTERS = 4_000;

/** Question dates come as "2024/03/01 (Fri) 09:00"; sessions as ISO instants. */
function questionDay(questionDate: string): string {
  const match = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(questionDate);
  return match === null
    ? questionDate.slice(0, 10)
    : `${match[1]}-${match[2]}-${match[3]}`;
}

/**
 * "63 days, about 9 weeks or 2 months, before the question date 2024-03-01": the
 * arithmetic a reader would otherwise do itself, done once and exactly.
 */
export function describeDistance(
  sessionTs: string,
  questionDate: string,
): string {
  const day = questionDay(questionDate);
  const from = Date.UTC(
    Number(sessionTs.slice(0, 4)),
    Number(sessionTs.slice(5, 7)) - 1,
    Number(sessionTs.slice(8, 10)),
  );
  const to = Date.UTC(
    Number(day.slice(0, 4)),
    Number(day.slice(5, 7)) - 1,
    Number(day.slice(8, 10)),
  );
  const days = Math.round((to - from) / 86_400_000);
  if (!Number.isFinite(days)) return `question date ${questionDate}`;
  const relation = days >= 0 ? 'before' : 'after';
  const abs = Math.abs(days);
  const parts = [`${abs} day${abs === 1 ? '' : 's'}`];
  if (abs >= 14) {
    const weeks = Math.round(abs / 7);
    const months = Math.round(abs / 30.44);
    parts.push(
      abs >= 60
        ? `about ${weeks} weeks or ${months} month${months === 1 ? '' : 's'}`
        : `about ${weeks} weeks`,
    );
  }
  return `${parts.join(', ')}${parts.length > 1 ? ',' : ''} ${relation} the question date ${day}`;
}

export function buildLongMemEvalAnswerContext(
  instance: LongMemEvalInstance,
  rankedSources: Array<{
    opId: string;
    ts: string;
    text?: string;
    redacted?: true;
    focusCharacterOffset?: number;
    /** Extracted facts of this session that matched the question, shown dated to the reader. */
    facts?: string[];
  }>,
  contextBytes = DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
  extraFacts: Array<{ clause: string; ts: string }> = [],
  reading: 'direct' | 'notes' | 'enumerate' = 'direct',
  engine?: { query: string; rendered: string },
  dateDistances = false,
): AnswerContext {
  validateOptions(Math.max(1, rankedSources.length), contextBytes);
  const usable = rankedSources.filter(
    (source) => source.redacted !== true && source.text !== undefined,
  );
  const perSessionBytes = Math.max(
    256,
    Math.floor(contextBytes / Math.max(1, usable.length)),
  );
  const selected = usable.map((source, rank) => {
    const facts =
      source.facts !== undefined && source.facts.length > 0
        ? `Remembered facts (stated in this session): ${source.facts.join(' ')}\n`
        : '';
    const dateLine = dateDistances
      ? `Session date: ${source.ts.slice(0, 10)} (${describeDistance(source.ts, instance.question_date)})`
      : `Session date: ${source.ts}`;
    const header = `### Retrieved session ${rank + 1}\n${dateLine}\n${facts}`;
    const body = sourceWindow(
      source.text!,
      instance.question,
      Math.max(1, perSessionBytes - Buffer.byteLength(header, 'utf8') - 2),
      source.focusCharacterOffset,
    );
    return { ...source, rank, section: `${header}${body}\n` };
  });
  const history = selected
    .sort(
      (left, right) =>
        left.ts.localeCompare(right.ts) || left.rank - right.rank,
    )
    .map(({ section }) => section)
    .join('\n');
  const remembered =
    extraFacts.length === 0
      ? ''
      : `\n### Supplementary remembered facts (extracted earlier, with the session date; they may be unrelated to the question, so ignore any that do not concern it and never treat them as evidence on their own)\n${extraFacts
          .map(({ ts, clause }) => `- ${ts}: ${clause}`)
          .join('\n')}\n`;
  const engineBlock =
    engine === undefined
      ? ''
      : `\n### Memory engine result\nThe memory system wrote this Datalog program over the facts it remembered from the whole history (every session, not only the chats above) and executed it. The rows are exact for the remembered facts, but the program may be broader than the question and facts can be missing or misread, so keep only the rows that fit the question, cross-check with the chats, and prefer the chats where they disagree.\nProgram: ${engine.query.replace(/\s*\n\s*/g, ' ')}\n${engine.rendered.slice(0, MAX_ENGINE_RENDER_CHARACTERS)}\n`;
  const user = `History chats:\n\n${history || '[no safe relevant history retrieved]'}\n${remembered}${engineBlock}Current date: ${instance.question_date}\nQuestion: ${instance.question}\nAnswer:`;
  assertSafeForExternalLlm(user, 'LongMemEval answer prompt');
  const system =
    instance.question_type === 'single-session-preference'
      ? 'Use the supplied history to personalize the answer. You may use general knowledge for recommendations, but do not invent facts about the user. Briefly make the remembered preference or context driving the answer explicit.'
      : reading === 'enumerate'
        ? 'Do not answer the question yet. From the supplied history, list every item relevant to the question, one per line, each with its session date and the exact detail the history states (a count, a name, a date, an amount, a quote). Include every occurrence across sessions, keep duplicates apart, and add nothing the history does not say. If nothing is relevant, write "No relevant items." Output the list only.'
        : reading === 'notes'
          ? 'Answer only from the supplied history. Work in two steps. First, under "Notes:", list every relevant item the history states, one per line, each with its session date and the exact detail (a count, a name, a date, an amount). Then, on a final line starting with "Answer:", give the answer derived from those notes, concise and with the arithmetic or ordering made explicit when the question needs it. If the notes do not support an answer, the Answer line must say that you do not know. Do not invent details.'
          : 'Answer only from the supplied history. If it does not support an answer, say that you do not know. Be concise and do not invent details.';
  return {
    messages: [
      {
        role: 'system',
        content: system,
      },
      { role: 'user', content: user },
    ],
    contextSessionIds: selected
      .sort((left, right) => left.rank - right.rank)
      .map(({ opId }) => opId),
    redactedRetrievedSessions: rankedSources.length - usable.length,
  };
}

export function buildLongMemEvalJudgePrompt(
  instance: LongMemEvalInstance,
  hypothesis: string,
): string {
  const common = `Question: ${instance.question}\n\nCorrect answer: ${instance.answer}\n\nModel response: ${hypothesis}`;
  if (instance.question_id.endsWith('_abs')) {
    return `Judge whether the response correctly identifies the question as unanswerable. It may say the information is incomplete or provide other information while noting that the requested information is unavailable.\n\n${common}\n\nAnswer yes or no only.`;
  }
  if (instance.question_type === 'single-session-preference') {
    return `The correct answer is a personalization rubric. Judge yes when the response recalls and correctly uses relevant personal information; it need not cover every rubric point.\n\n${common}\n\nAnswer yes or no only.`;
  }
  if (instance.question_type === 'knowledge-update') {
    return `Judge yes when the response contains the updated correct answer. Earlier information may also appear, provided the update is clear.\n\n${common}\n\nAnswer yes or no only.`;
  }
  if (instance.question_type === 'temporal-reasoning') {
    return `Judge yes when the response is equivalent to the correct answer or contains every required intermediate result. Do not penalize a one-unit error in a requested count of days, weeks, or months. A partial answer is incorrect.\n\n${common}\n\nAnswer yes or no only.`;
  }
  return `Judge yes when the response contains an equivalent correct answer or every intermediate result required to derive it. A partial response is incorrect.\n\n${common}\n\nAnswer yes or no only.`;
}

export function parseLongMemEvalJudgeLabel(value: string): boolean {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[.!]+$/g, '');
  if (normalized === 'yes') return true;
  if (normalized === 'no') return false;
  throw new Error('LongMemEval judge must answer yes or no only');
}

export async function evaluateLongMemEvalAnswerInstance(
  instance: LongMemEvalInstance,
  reader: LongMemEvalCompletionClient,
  judge: LongMemEvalCompletionClient,
  options: {
    topK?: number;
    multiSessionTopK?: number;
    temporalTopK?: number;
    contextBytes?: number;
    embeddings?: EmbeddingClient;
    semanticQuestionTypes?: ReadonlySet<string>;
    multiSessionSemanticMaximumLexicalScore?: number;
    prepareSemantic?: boolean;
    formation?: LongMemEvalFormation;
    /** Model that runs the product's transcript extraction (extracted/hybrid formations). */
    extractor?: LongMemEvalCompletionClient;
    extractionCharacters?: number;
    extractionMaxTokens?: number;
    /** Show each retrieved session's matched extracted facts to the reader (default true). */
    factsInContext?: boolean;
    /**
     * hybrid only. 'shared' (default): raw and extracted facts compete for the same top-k
     * session slots. 'reserved': top-k is filled from raw session text exactly as in raw
     * formation, and extracted facts are searched separately and appended to the reader's
     * context as a dated block (their sessions count as retrieved). 'keyed': the extracted
     * facts are prepended to the session's own key (its source text) and not indexed as
     * separate documents, the LongMemEval paper's fact-augmented keys; the facts stay in the
     * store for entity retrieval.
     */
    hybridRetrieval?: 'shared' | 'reserved' | 'keyed';
    /** hybrid/extracted only: question types that use the extractor; other types run raw. */
    hybridQuestionTypes?: ReadonlySet<string>;
    /** reserved only: minimum lexical score for an appended fact (default 1). */
    reservedMinimumScore?: number;
    /**
     * hybrid/extracted: one hop over the extracted facts from the question (shared
     * relation-and-subject or shared entity); the sessions found take alternate top-k
     * slots with the lexical ranking. See knowledge/entity-retrieval.ts.
     */
    entityRetrieval?: boolean;
    /**
     * 'session' (default): one document per session. 'turn': one document per user turn,
     * scored separately and aggregated to sessions (sum of 1/log2(rank+1) over a session's
     * matching turns); whole sessions are returned to the reader.
     */
    retrievalUnit?: 'session' | 'turn';
    /**
     * Time-aware retrieval (the paper's time-aware query expansion): a frontier model reads
     * the absolute date range a question refers to, given the question date, or refuses when
     * there is no time cue; sessions inside the range are ranked ahead of those outside.
     */
    temporalRangeExtractor?: LongMemEvalCompletionClient;
    /** Question types that get the range treatment (default: temporal-reasoning). */
    temporalRangeQuestionTypes?: ReadonlySet<string>;
    /**
     * Engine recall: after formation, the memory system's own recall path authors a Datalog
     * query over the facts it remembered from the whole history (the writer model does the
     * authoring), executes it, and the reader sees the query and its rows ahead of the
     * chats. This is the moonshot composition: counting, latest-value and chaining come
     * from the engine, the reader reads.
     */
    /**
     * Each retrieved session's header states its distance to the question date in days,
     * weeks and months, so the reader copies an interval instead of computing one.
     */
    dateDistances?: boolean;
    engineRecall?: {
      /** The writer: authors the Datalog query (usage is accounted with extraction). */
      llm: LongMemEvalCompletionClient;
      /** Question types that get it (default: all). */
      questionTypes?: ReadonlySet<string>;
    };
    /**
     * 'direct' (default) or 'notes': the reader first lists every relevant dated item, then
     * gives a final "Answer:" line, which alone is judged (the paper's Chain-of-Note reading).
     */
    readingStrategy?: 'direct' | 'notes' | 'two-call';
    /** Question types read with notes (default: multi-session, temporal-reasoning, knowledge-update). */
    notesQuestionTypes?: ReadonlySet<string>;
    /** A reader used only for the aggregation question types (same set as notesQuestionTypes). */
    aggregationReader?: LongMemEvalCompletionClient;
    /** Completion budget per reader call (default 4096; reasoning readers need more). */
    readerMaxTokens?: number;
    /**
     * Directory of per-session extraction results keyed by extractor model and transcript
     * hash. Extraction is deterministic enough to replay, and it is two hours of a full dev
     * run; with a warm cache a formation or retrieval experiment costs only reader time.
     */
    extractionCacheDir?: string;
    /** Cut each assistant turn to this many characters before extraction (default: no cut). */
    extractionAssistantCharacters?: number;
  } = {},
): Promise<LongMemEvalAnswerObservation> {
  // formation can be routed by question type: a type outside the set runs raw formation,
  // which also spares the extraction calls for it
  const formation: LongMemEvalFormation =
    options.formation !== undefined &&
    options.formation !== 'raw' &&
    options.hybridQuestionTypes !== undefined &&
    !options.hybridQuestionTypes.has(instance.question_type)
      ? 'raw'
      : (options.formation ?? 'raw');
  if (formation !== 'raw' && options.extractor === undefined) {
    throw new Error(`formation "${formation}" needs an extractor client`);
  }
  const topK = options.topK ?? DEFAULT_LONGMEMEVAL_ANSWER_TOP_K;
  const effectiveTopK =
    instance.question_type === 'multi-session'
      ? (options.multiSessionTopK ?? DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K)
      : instance.question_type === 'temporal-reasoning'
        ? (options.temporalTopK ?? DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K)
        : topK;
  const contextBytes =
    options.contextBytes ?? DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES;
  validateOptions(effectiveTopK, contextBytes);
  const multiSessionSemanticMaximumLexicalScore =
    options.multiSessionSemanticMaximumLexicalScore ??
    LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE;
  if (
    !Number.isFinite(multiSessionSemanticMaximumLexicalScore) ||
    multiSessionSemanticMaximumLexicalScore < 0 ||
    multiSessionSemanticMaximumLexicalScore > 10_000
  ) {
    throw new Error(
      'multi-session semantic maximum lexical score must be from 0 to 10000',
    );
  }
  const root = mkdtempSync(join(tmpdir(), 'remembero-longmemeval-answer-'));
  const started = performance.now();
  let formationMs = 0;
  let semanticPreparationMs = 0;
  let retrievalMs = 0;
  let readerMs = 0;
  let judgeMs = 0;
  let retrievedSessionIds: string[] = [];
  let contextSessionIds: string[] = [];
  const contextRoles =
    instance.question_type === 'single-session-assistant'
      ? ('all' as const)
      : ('user' as const);
  let redactedRetrievedSessions = 0;
  let retrievalRoute: 'local' | 'semantic' = 'local';
  let embeddingModel: string | null = null;
  let embeddingCalls = 0;
  let embeddingUsage: EmbeddingUsage | null = null;
  let semanticPreparationCalls = 0;
  let semanticPreparationUsage: EmbeddingUsage | null = null;
  let retrieval: LongMemEvalQuestionResult | null = null;
  let context: LongMemEvalQuestionResult | null = null;
  let hypothesis: string | null = null;
  let judgeResponse: string | null = null;
  let readerUsage: LlmUsage | null = null;
  let judgeUsage: LlmUsage | null = null;
  let extraction: LongMemEvalExtractionStats | undefined;
  let temporalRange: { start: string; end: string } | null | undefined;
  let temporalRangeUsage: LlmUsage | null = null;
  let engineRecall: LongMemEvalEngineRecall | undefined;
  try {
    const store = new MemoryStore(root);
    const formationStarted = performance.now();
    const sourceSessionIds = new Map<string, string>();
    const userSourceText = new Map<string, string>();
    // session id -> what the reader needs if the session is reached other than lexically
    const sessionRecords = new Map<string, { ts: string; text: string }>();
    if (formation !== 'raw') {
      extraction = {
        calls: 0,
        sessionsWithFacts: 0,
        facts: 0,
        errors: 0,
        errorKinds: {},
        usage: null,
      };
    }
    const extractor = options.extractor;
    const stats = extraction;
    // the product pipeline takes a plain LlmClient; adapt the usage-reporting client
    const extractorLlm =
      extractor === undefined || stats === undefined
        ? undefined
        : {
            complete: async (messages: ChatMessage[]): Promise<string> => {
              stats.calls += 1;
              const completion = await extractor.completeWithUsage(messages, {
                maxTokens: options.extractionMaxTokens ?? 512,
              });
              stats.usage =
                stats.usage === null
                  ? completion.usage
                  : sumLlmUsage(stats.usage, completion.usage);
              return completion.content;
            },
          };
    for (const [index, session] of instance.haystack_sessions.entries()) {
      const operationId = `longmemeval:${index}:${instance.haystack_session_ids[index]!}`;
      sourceSessionIds.set(operationId, instance.haystack_session_ids[index]!);
      const userText = longMemEvalSessionText(
        session.filter(({ role }) => role === 'user'),
      );
      userSourceText.set(
        operationId,
        userText === '' ? longMemEvalSessionText(session) : userText,
      );
      const sessionText = longMemEvalSessionText(session);
      const at = datasetDate(instance.haystack_dates[index]!);
      sessionRecords.set(instance.haystack_session_ids[index]!, {
        ts: at.toISOString(),
        text:
          contextRoles === 'user'
            ? userSourceText.get(operationId)!
            : sessionText,
      });
      let sessionFacts: string[] = [];
      if (extractorLlm !== undefined && stats !== undefined) {
        // hybrid keeps the raw fact under the session's id; the extracted facts need
        // their own id (the store refuses to reuse one) that still maps to the session
        const factsOperationId =
          formation === 'hybrid' ? `${operationId}:facts` : operationId;
        sourceSessionIds.set(
          factsOperationId,
          instance.haystack_session_ids[index]!,
        );
        userSourceText.set(factsOperationId, userSourceText.get(operationId)!);
        const budget =
          options.extractionCharacters ??
          DEFAULT_LONGMEMEVAL_EXTRACTION_CHARACTERS;
        const transcript = longMemEvalTranscript(session, {
          ...(options.extractionAssistantCharacters === undefined
            ? {}
            : { assistantCharacters: options.extractionAssistantCharacters }),
        }).slice(0, budget);
        const cachePath =
          options.extractionCacheDir === undefined
            ? undefined
            : join(
                options.extractionCacheDir,
                `${createHash('sha256')
                  .update(`${extractor!.model}\n${transcript}`)
                  .digest('hex')
                  .slice(0, 40)}.json`,
              );
        const cached =
          cachePath !== undefined && existsSync(cachePath)
            ? (JSON.parse(readFileSync(cachePath, 'utf8')) as {
                facts: string[];
                error?: string;
              })
            : undefined;
        if (cached !== undefined) {
          stats.cached = (stats.cached ?? 0) + 1;
          if (cached.error !== undefined) {
            stats.errors += 1;
            stats.errorKinds[cached.error] =
              (stats.errorKinds[cached.error] ?? 0) + 1;
          } else if (cached.facts.length > 0) {
            store.assert('longmemeval', cached.facts.join('\n'), {
              opId: factsOperationId,
              sourceText: sessionText,
              at,
            });
            stats.sessionsWithFacts += 1;
            stats.facts += cached.facts.length;
            sessionFacts = cached.facts;
          }
        } else {
          try {
            const result = await rememberTranscriptText(
              { store, llm: extractorLlm },
              transcript,
              'longmemeval',
              {
                captureId: operationId,
                opId: factsOperationId,
                sourceText: sessionText,
                origin: 'manual',
                at,
              },
            );
            if (result.added.length > 0) stats.sessionsWithFacts += 1;
            stats.facts += result.added.length;
            sessionFacts = result.added;
            if (cachePath !== undefined) {
              mkdirSync(options.extractionCacheDir!, { recursive: true });
              writeFileSync(cachePath, JSON.stringify({ facts: result.added }));
            }
          } catch (error) {
            // a refused or malformed extraction leaves the session raw (hybrid) or absent (extracted)
            stats.errors += 1;
            const kind = extractionErrorKind(error);
            stats.errorKinds[kind] = (stats.errorKinds[kind] ?? 0) + 1;
            if (cachePath !== undefined) {
              mkdirSync(options.extractionCacheDir!, { recursive: true });
              writeFileSync(
                cachePath,
                JSON.stringify({ facts: [], error: kind }),
              );
            }
          }
        }
      }
      if (formation !== 'extracted') {
        // fact-augmented key: the session's own facts lead its source text, so lexical
        // scoring sees them without a separate document competing for a slot
        const keyedFormation =
          formation === 'hybrid' && options.hybridRetrieval === 'keyed';
        // rendered as words: "bought(user, road_bike)." -> "bought user road bike", so a
        // whitespace tokenizer sees the same terms the question uses
        const factKey =
          keyedFormation && sessionFacts.length > 0
            ? `Remembered facts: ${sessionFacts
                .map((fact) => fact.replace(/[_(),.']+/g, ' ').trim())
                .join('. ')}.\n\n`
            : '';
        if (options.retrievalUnit === 'turn') {
          // one document per user turn; the session is what comes back to the reader
          let turnIndex = 0;
          for (const turn of session) {
            // assistant turns are indexed too: assistant-memory questions ask what the
            // assistant said, and the first turn-only run lost six of them
            if (turn.content.trim() === '') continue;
            const turnOperationId = `${operationId}:t${turnIndex}`;
            sourceSessionIds.set(
              turnOperationId,
              instance.haystack_session_ids[index]!,
            );
            store.assert(
              'longmemeval',
              `longmem_turn(session_${index}, turn_${turnIndex}).`,
              {
                opId: turnOperationId,
                sourceText: `${turnIndex === 0 ? factKey : ''}${turn.content}`,
                at,
              },
            );
            turnIndex += 1;
          }
          if (turnIndex === 0) {
            store.assert('longmemeval', `longmem_session(session_${index}).`, {
              opId: operationId,
              sourceText: `${factKey}${sessionText}`,
              at,
            });
          }
        } else {
          store.assert('longmemeval', `longmem_session(session_${index}).`, {
            opId: operationId,
            sourceText: `${factKey}${sessionText}`,
            at,
          });
        }
      }
    }
    formationMs = performance.now() - formationStarted;
    const snapshot = store.knowledgeSnapshot(['longmemeval']);
    const retrievalStarted = performance.now();
    // time-aware: read a date range off the question before searching
    if (
      options.temporalRangeExtractor !== undefined &&
      (
        options.temporalRangeQuestionTypes ?? new Set(['temporal-reasoning'])
      ).has(instance.question_type)
    ) {
      const completion = await options.temporalRangeExtractor.completeWithUsage(
        [
          {
            role: 'user',
            content: temporalRangePrompt(
              instance.question,
              instance.question_date,
            ),
          },
        ],
        { maxTokens: 1024 },
      );
      temporalRangeUsage = completion.usage;
      temporalRange = parseTemporalRange(completion.content);
    }
    const inRange = (ts: string): boolean =>
      temporalRange !== null &&
      temporalRange !== undefined &&
      ts.slice(0, 10) >= temporalRange.start &&
      ts.slice(0, 10) <= temporalRange.end;
    const rangeFirst = <T extends { ts: string }>(items: T[]): T[] =>
      temporalRange === null || temporalRange === undefined
        ? items
        : [
            ...items.filter((i) => inRange(i.ts)),
            ...items.filter((i) => !inRange(i.ts)),
          ];
    const semanticQuestionTypes =
      options.semanticQuestionTypes ??
      DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES;
    const reserved =
      formation === 'hybrid' && options.hybridRetrieval === 'reserved';
    const keyed = formation === 'hybrid' && options.hybridRetrieval === 'keyed';
    const isPlaceholder = (c: Clause) =>
      c.head.predicate === 'longmem_session' ||
      c.head.predicate === 'longmem_turn';
    // reserved and keyed: placeholders fill top-k on their own (keyed carries the facts in
    // the placeholder's text); reserved also searches the extracted facts apart
    const rawClauses =
      reserved || keyed
        ? snapshot.clauses.filter(isPlaceholder)
        : snapshot.clauses;
    const factClauses = reserved
      ? snapshot.clauses.filter((c) => !isPlaceholder(c))
      : [];
    const turnUnit = options.retrievalUnit === 'turn';
    const lexical = searchKnowledge(
      rawClauses,
      instance.question,
      snapshot.sources,
      {
        // extracted formations hold several facts per session; fetch more so top-k
        // still counts distinct sessions after de-duplication below
        limit: Math.min(
          100,
          Math.max(
            // a range needs candidates beyond top-k to promote from
            temporalRange ? effectiveTopK * 4 : 0,
            turnUnit
              ? effectiveTopK * 6
              : formation === 'raw' || reserved || keyed
                ? effectiveTopK
                : effectiveTopK * 8,
          ),
        ),
        minimumScore: 1,
        kinds: ['fact'],
        sourceCharacterLimit: LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS,
      },
    );
    const useSemantic =
      options.embeddings !== undefined &&
      semanticQuestionTypes.has(instance.question_type) &&
      ((instance.question_type === 'single-session-preference' &&
        isRecommendationIntent(instance.question)) ||
        (instance.question_type === 'multi-session' &&
          (lexical.results[0]?.score ?? 0) <=
            multiSessionSemanticMaximumLexicalScore));
    let search: KnowledgeSearchResult | SemanticKnowledgeSearchResult;
    if (useSemantic) {
      const semanticCache = new MemoryEmbeddingCache();
      if (options.prepareSemantic === true) {
        const preparationStarted = performance.now();
        let after: string | undefined;
        do {
          const prepared = await prepareSemanticKnowledge(
            snapshot.clauses,
            snapshot.sources,
            options.embeddings!,
            {
              cache: semanticCache,
              limit: 100,
              kinds: ['fact'],
              ...(after === undefined ? {} : { after }),
            },
          );
          semanticPreparationCalls += prepared.providerCalls;
          semanticPreparationUsage = mergeEmbeddingUsage(
            semanticPreparationUsage,
            prepared.providerUsage,
          );
          after =
            prepared.status === 'more'
              ? (prepared.nextCursor ?? undefined)
              : undefined;
          if (prepared.status === 'complete') break;
        } while (after !== undefined);
        semanticPreparationMs = performance.now() - preparationStarted;
      }
      const semantic = await semanticSearchKnowledge(
        snapshot.clauses,
        instance.question,
        snapshot.sources,
        options.embeddings!,
        {
          limit:
            formation === 'raw'
              ? effectiveTopK
              : Math.min(100, effectiveTopK * 8),
          candidateLimit: 100,
          kinds: ['fact'],
          cache: semanticCache,
        },
      );
      retrievalRoute = 'semantic';
      embeddingModel = options.embeddings!.model;
      embeddingCalls = semantic.providerCalls;
      embeddingUsage = semantic.providerUsage;
      search = semantic;
    } else {
      search = lexical;
    }
    retrievalMs = Math.max(
      0,
      performance.now() - retrievalStarted - semanticPreparationMs,
    );
    // extracted facts that matched, grouped by session, minus the placeholder plumbing
    const matchedFacts = new Map<string, string[]>();
    for (const result of search.results) {
      const source = result.sources[0];
      if (
        source === undefined ||
        result.clause.startsWith('longmem_session(') ||
        result.clause.startsWith('longmem_turn(')
      )
        continue;
      const session = sourceSessionIds.get(source.opId) ?? source.opId;
      const list = matchedFacts.get(session) ?? [];
      if (list.length < 24) list.push(result.clause);
      matchedFacts.set(session, list);
    }
    const seenSessions = new Set<string>();
    const rankedSources = search.results.flatMap((result) => {
      const source = result.sources[0];
      return source === undefined
        ? []
        : [
            {
              opId: sourceSessionIds.get(source.opId) ?? source.opId,
              ts: source.ts,
              facts:
                options.factsInContext === false
                  ? []
                  : (matchedFacts.get(
                      sourceSessionIds.get(source.opId) ?? source.opId,
                    ) ?? []),
              text:
                contextRoles === 'user'
                  ? (userSourceText.get(source.opId) ?? source.text)
                  : source.text,
              ...('semanticChunkIndex' in result
                ? {
                    focusCharacterOffset:
                      result.semanticChunkIndex *
                      (SEMANTIC_CHUNK_CHARACTERS - SEMANTIC_CHUNK_OVERLAP),
                  }
                : {}),
              ...(source.redacted === true ? { redacted: true as const } : {}),
            },
          ];
    });
    // one entry per session, best rank first, then the usual top-k
    const dedupedSources = rangeFirst(
      rankedSources.filter(({ opId }) => {
        if (seenSessions.has(opId)) return false;
        seenSessions.add(opId);
        return true;
      }),
    ).slice(0, effectiveTopK);
    if (turnUnit) {
      // aggregate matching turns to their sessions: sum of 1/log2(rank+1), whole session back
      const sessionScore = new Map<string, number>();
      search.results.forEach((result, position) => {
        const source = result.sources[0];
        if (source === undefined) return;
        const session = sourceSessionIds.get(source.opId) ?? source.opId;
        sessionScore.set(
          session,
          (sessionScore.get(session) ?? 0) + 1 / Math.log2(position + 2),
        );
      });
      const ordered = rangeFirst(
        [...sessionScore.entries()]
          .sort((a, b) => b[1] - a[1])
          .map(([session]) => ({
            session,
            ts: sessionRecords.get(session)?.ts ?? '',
          })),
      )
        .slice(0, effectiveTopK)
        .map(({ session }) => session);
      dedupedSources.length = 0;
      for (const session of ordered) {
        const record = sessionRecords.get(session);
        if (record === undefined) continue;
        dedupedSources.push({
          opId: session,
          ts: record.ts,
          text: record.text,
          facts:
            options.factsInContext === false
              ? []
              : (matchedFacts.get(session) ?? []),
        });
      }
    }
    rankedSources.length = 0;
    rankedSources.push(...dedupedSources);
    // entity retrieval: one hop over the extracted facts, interleaved with the lexical order
    if (options.entityRetrieval === true && formation !== 'raw') {
      const hits = expandByEntities(
        snapshot.clauses.filter((c) => c.head.predicate !== 'longmem_session'),
        instance.question,
        snapshot.sources,
        { selfAtom: 'user', maxSessions: effectiveTopK * 2 },
      );
      const bySession = new Map<string, (typeof hits)[number]>();
      for (const hit of hits) {
        const session = sourceSessionIds.get(hit.opId) ?? hit.opId;
        if (!bySession.has(session)) bySession.set(session, hit);
      }
      const order = interleaveSessions(
        rankedSources.map(({ opId }) => opId),
        [...bySession.keys()],
        effectiveTopK,
      );
      const lexicalBySession = new Map(rankedSources.map((r) => [r.opId, r]));
      const merged = order.flatMap((session) => {
        const lexicalEntry = lexicalBySession.get(session);
        if (lexicalEntry !== undefined) return [lexicalEntry];
        const record = sessionRecords.get(session);
        const hit = bySession.get(session);
        if (record === undefined || hit === undefined) return [];
        return [
          {
            opId: session,
            ts: record.ts,
            text: record.text,
            facts: options.factsInContext === false ? [] : hit.facts,
          },
        ];
      });
      rankedSources.length = 0;
      rankedSources.push(...merged);
    }
    retrievedSessionIds = rankedSources.map(({ opId }) => opId);
    // reserved: matched extracted facts ride along as a dated block, up to 3k of them
    const extraFacts: Array<{ clause: string; ts: string }> = [];
    if (reserved && factClauses.length > 0) {
      const factSearch = searchKnowledge(
        factClauses,
        instance.question,
        snapshot.sources,
        {
          limit: Math.min(100, effectiveTopK * 3),
          minimumScore: options.reservedMinimumScore ?? 1,
          kinds: ['fact'],
          sourceCharacterLimit: LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS,
        },
      );
      for (const result of factSearch.results) {
        const source = result.sources[0];
        if (source === undefined || source.redacted === true) continue;
        extraFacts.push({ clause: result.clause, ts: source.ts });
        const session = sourceSessionIds.get(source.opId) ?? source.opId;
        if (!retrievedSessionIds.includes(session))
          retrievedSessionIds.push(session);
      }
    }
    const firstResult = search.results[0];
    const topScore =
      firstResult === undefined
        ? 0
        : 'semanticScore' in firstResult
          ? firstResult.semanticScore
          : firstResult.score;
    retrieval = scoreLongMemEvalRetrievedSessions(
      instance,
      retrievedSessionIds,
      retrievalMs,
      topScore,
    );
    const aggregationType = (
      options.notesQuestionTypes ??
      new Set(['multi-session', 'temporal-reasoning', 'knowledge-update'])
    ).has(instance.question_type);
    const notes = options.readingStrategy === 'notes' && aggregationType;
    const twoCall = options.readingStrategy === 'two-call' && aggregationType;
    let engine: { query: string; rendered: string } | undefined;
    if (
      options.engineRecall !== undefined &&
      (options.engineRecall.questionTypes === undefined ||
        options.engineRecall.questionTypes.has(instance.question_type))
    ) {
      const engineStarted = performance.now();
      const writer = options.engineRecall.llm;
      let engineUsage: LlmUsage | null = null;
      const authoring: LlmClient = {
        complete: async (messages) => {
          if (process.env.REMBERO_ENGINE_DEBUG) {
            const bytes = messages.reduce((a, m) => a + m.content.length, 0);
            process.stderr.write(
              `[engine-recall] ${instance.question_id} prompt ${bytes} chars, ${messages.length} messages\n`,
            );
          }
          const completion = await writer.completeWithUsage(messages, {
            maxTokens: 512,
          });
          if (process.env.REMBERO_ENGINE_DEBUG) {
            process.stderr.write(
              `[engine-recall] ${instance.question_id} system: ${messages[0]?.content.slice(0, 1500).replace(/\n/g, ' | ')}\n[engine-recall] ${instance.question_id} last user: ${messages.at(-1)?.content.slice(0, 600).replace(/\n/g, ' | ')}\n[engine-recall] ${instance.question_id} response: ${completion.content.slice(0, 300).replace(/\n/g, ' | ')}\n`,
            );
          }
          engineUsage =
            engineUsage === null
              ? completion.usage
              : sumLlmUsage(engineUsage, completion.usage);
          return completion.content;
        },
      };
      try {
        const recall = await runEngineRecall(
          store,
          'longmemeval',
          instance.question,
          authoring,
        );
        engineRecall = {
          status: recall.status,
          query: recall.program,
          rows: recall.rows,
          attempts: recall.attempts,
          ms: performance.now() - engineStarted,
          usage: engineUsage,
          ...(recall.error === undefined ? {} : { error: recall.error }),
        };
        if (recall.status === 'answered' && recall.program !== null) {
          engine = { query: recall.program, rendered: recall.rendered };
        }
      } catch (error) {
        engineRecall = {
          status: 'error',
          query: null,
          rows: 0,
          attempts: 0,
          ms: performance.now() - engineStarted,
          usage: engineUsage,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }
    const answerContext = buildLongMemEvalAnswerContext(
      instance,
      rankedSources,
      contextBytes,
      extraFacts,
      twoCall ? 'enumerate' : notes ? 'notes' : 'direct',
      engine,
      options.dateDistances === true,
    );
    contextSessionIds = [
      ...answerContext.contextSessionIds,
      ...retrievedSessionIds.filter(
        (id) => !answerContext.contextSessionIds.includes(id),
      ),
    ];
    redactedRetrievedSessions = answerContext.redactedRetrievedSessions;
    context = scoreLongMemEvalRetrievedSessions(
      instance,
      contextSessionIds,
      retrievalMs,
      topScore,
    );
    const readerStarted = performance.now();
    const activeReader =
      aggregationType && options.aggregationReader !== undefined
        ? options.aggregationReader
        : reader;
    const readerCompletion = await activeReader.completeWithUsage(
      answerContext.messages,
      {
        maxTokens: options.readerMaxTokens ?? 4_096,
      },
    );
    readerUsage = readerCompletion.usage;
    if (twoCall) {
      // second call: answer from the enumerated items only, the history left behind
      const enumeration = readerCompletion.content.trim();
      const answerCompletion = await activeReader.completeWithUsage(
        [
          {
            role: 'system',
            content:
              "Answer the question using only the dated items listed below, which were extracted from the user's chat history. Make any counting, summing or ordering explicit, then give the answer. If the items do not support an answer, say that you do not know. Do not invent details.",
          },
          {
            role: 'user',
            content: `Relevant items from the history:\n${enumeration || 'No relevant items.'}\n\nCurrent date: ${instance.question_date}\nQuestion: ${instance.question}\nAnswer:`,
          },
        ],
        { maxTokens: options.readerMaxTokens ?? 4_096 },
      );
      hypothesis = answerCompletion.content.trim();
      readerUsage = sumLlmUsage(readerUsage, answerCompletion.usage);
    } else {
      hypothesis = notes
        ? finalAnswerLine(readerCompletion.content)
        : readerCompletion.content.trim();
    }
    readerMs = performance.now() - readerStarted;
    const judgeStarted = performance.now();
    const judgeCompletion = await judge.completeWithUsage(
      [
        {
          role: 'user',
          content: buildLongMemEvalJudgePrompt(instance, hypothesis),
        },
      ],
      { maxTokens: 16 },
    );
    judgeMs = performance.now() - judgeStarted;
    judgeResponse = judgeCompletion.content.trim();
    judgeUsage = judgeCompletion.usage;
    return {
      questionId: instance.question_id,
      questionType: instance.question_type,
      abstention: instance.question_id.endsWith('_abs'),
      status: 'judged',
      correct: parseLongMemEvalJudgeLabel(judgeResponse),
      retrievedSessionIds,
      contextSessionIds,
      contextRoles,
      redactedRetrievedSessions,
      retrievalRoute,
      embeddingModel,
      embeddingCalls,
      embeddingUsage,
      semanticPreparationCalls,
      semanticPreparationUsage,
      retrieval,
      context,
      hypothesis,
      judgeResponse,
      readerUsage,
      judgeUsage,
      ...(extraction === undefined ? {} : { extraction }),
      ...(temporalRange === undefined
        ? {}
        : { temporalRange, temporalRangeUsage }),
      ...(engineRecall === undefined ? {} : { engineRecall }),
      formationMs,
      semanticPreparationMs,
      retrievalMs,
      readerMs,
      judgeMs,
      totalMs: performance.now() - started,
      userTurnMs: retrievalMs + readerMs + judgeMs,
    };
  } catch (error) {
    return {
      questionId: instance.question_id,
      questionType: instance.question_type,
      abstention: instance.question_id.endsWith('_abs'),
      status: 'error',
      correct: null,
      retrievedSessionIds,
      contextSessionIds,
      contextRoles,
      redactedRetrievedSessions,
      retrievalRoute,
      embeddingModel,
      embeddingCalls,
      embeddingUsage,
      semanticPreparationCalls,
      semanticPreparationUsage,
      retrieval,
      context,
      hypothesis,
      judgeResponse,
      readerUsage,
      judgeUsage,
      ...(extraction === undefined ? {} : { extraction }),
      ...(temporalRange === undefined
        ? {}
        : { temporalRange, temporalRangeUsage }),
      ...(engineRecall === undefined ? {} : { engineRecall }),
      formationMs,
      semanticPreparationMs,
      retrievalMs,
      readerMs,
      judgeMs,
      totalMs: performance.now() - started,
      userTurnMs: retrievalMs + readerMs + judgeMs,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

export function summarizeLongMemEvalAnswers(
  observations: readonly LongMemEvalAnswerObservation[],
): LongMemEvalAnswerSummary {
  const judged = observations.filter(
    (
      observation,
    ): observation is LongMemEvalAnswerObservation & { correct: boolean } =>
      observation.status === 'judged' && observation.correct !== null,
  );
  const correct = judged.filter(({ correct: value }) => value).length;
  const abstentions = judged.filter(({ abstention }) => abstention);
  const fullEvidence = judged.filter(
    ({ context: value }) => value?.strictEvidenceCoverage === true,
  );
  const incompleteEvidence = judged.filter(
    ({ context: value }) => value?.strictEvidenceCoverage !== true,
  );
  let readerUsage = emptyLlmUsageTotals();
  let judgeUsage = emptyLlmUsageTotals();
  let extractionTotals = emptyLlmUsageTotals();
  const extractionCounts = {
    sessionsWithFacts: 0,
    facts: 0,
    errors: 0,
    errorKinds: {} as Record<string, number>,
  };
  const embeddingUsage = {
    calls: 0,
    promptTokens: 0,
    totalTokens: 0,
    costResponses: 0,
    costUsd: 0,
  };
  const semanticPreparationUsage = {
    calls: 0,
    promptTokens: 0,
    totalTokens: 0,
    costResponses: 0,
    costUsd: 0,
  };
  for (const observation of observations) {
    if (observation.readerUsage !== null)
      readerUsage = addLlmUsage(readerUsage, observation.readerUsage);
    if (observation.judgeUsage !== null)
      judgeUsage = addLlmUsage(judgeUsage, observation.judgeUsage);
    if (observation.extraction !== undefined) {
      // addLlmUsage counts one call per invocation; the extraction ran many per question
      if (observation.extraction.usage !== null) {
        extractionTotals = addLlmUsage(
          extractionTotals,
          observation.extraction.usage,
        );
        extractionTotals.calls += observation.extraction.calls - 1;
      } else {
        extractionTotals.calls += observation.extraction.calls;
      }
      extractionCounts.sessionsWithFacts +=
        observation.extraction.sessionsWithFacts;
      extractionCounts.facts += observation.extraction.facts;
      extractionCounts.errors += observation.extraction.errors;
      for (const [kind, count] of Object.entries(
        observation.extraction.errorKinds ?? {},
      )) {
        extractionCounts.errorKinds[kind] =
          (extractionCounts.errorKinds[kind] ?? 0) + count;
      }
    }
    embeddingUsage.calls += observation.embeddingCalls;
    embeddingUsage.promptTokens +=
      observation.embeddingUsage?.promptTokens ?? 0;
    embeddingUsage.totalTokens += observation.embeddingUsage?.totalTokens ?? 0;
    if (
      observation.embeddingUsage !== null &&
      observation.embeddingUsage !== undefined &&
      observation.embeddingUsage.costUsd !== null
    ) {
      embeddingUsage.costResponses += observation.embeddingCalls;
      embeddingUsage.costUsd += observation.embeddingUsage.costUsd;
    }
    semanticPreparationUsage.calls += observation.semanticPreparationCalls ?? 0;
    semanticPreparationUsage.promptTokens +=
      observation.semanticPreparationUsage?.promptTokens ?? 0;
    semanticPreparationUsage.totalTokens +=
      observation.semanticPreparationUsage?.totalTokens ?? 0;
    if (
      observation.semanticPreparationUsage?.costUsd !== null &&
      observation.semanticPreparationUsage !== null &&
      observation.semanticPreparationUsage !== undefined
    ) {
      semanticPreparationUsage.costResponses +=
        observation.semanticPreparationCalls ?? 0;
      semanticPreparationUsage.costUsd +=
        observation.semanticPreparationUsage.costUsd;
    }
  }
  return {
    questions: observations.length,
    judgedQuestions: judged.length,
    errors: observations.length - judged.length,
    correct,
    accuracy: observations.length === 0 ? 0 : correct / observations.length,
    judgedAccuracy: judged.length === 0 ? 0 : correct / judged.length,
    abstentionAccuracy: mean(
      abstentions.map(({ correct: value }) => (value ? 1 : 0)),
    ),
    fullContextEvidenceAccuracy: mean(
      fullEvidence.map(({ correct: value }) => (value ? 1 : 0)),
    ),
    incompleteContextEvidenceAccuracy: mean(
      incompleteEvidence.map(({ correct: value }) => (value ? 1 : 0)),
    ),
    retrievalRecallAtK: mean(
      observations.flatMap(({ retrieval: value }) =>
        value?.recallAtK === null || value?.recallAtK === undefined
          ? []
          : [value.recallAtK],
      ),
    ),
    contextRecallAtK: mean(
      observations.flatMap(({ context: value }) =>
        value?.recallAtK === null || value?.recallAtK === undefined
          ? []
          : [value.recallAtK],
      ),
    ),
    redactedRetrievedSessions: observations.reduce(
      (sum, value) => sum + value.redactedRetrievedSessions,
      0,
    ),
    medianFormationMs: percentile(
      observations.map(({ formationMs: value }) => value),
      0.5,
    ),
    p95FormationMs: percentile(
      observations.map(({ formationMs: value }) => value),
      0.95,
    ),
    medianSemanticPreparationMs: percentile(
      observations.map(({ semanticPreparationMs: value }) => value ?? 0),
      0.5,
    ),
    p95SemanticPreparationMs: percentile(
      observations.map(({ semanticPreparationMs: value }) => value ?? 0),
      0.95,
    ),
    medianRetrievalMs: percentile(
      observations.map(({ retrievalMs: value }) => value),
      0.5,
    ),
    p95RetrievalMs: percentile(
      observations.map(({ retrievalMs: value }) => value),
      0.95,
    ),
    medianReaderMs: percentile(
      observations.map(({ readerMs: value }) => value),
      0.5,
    ),
    p95ReaderMs: percentile(
      observations.map(({ readerMs: value }) => value),
      0.95,
    ),
    medianJudgeMs: percentile(
      observations.map(({ judgeMs: value }) => value),
      0.5,
    ),
    p95JudgeMs: percentile(
      observations.map(({ judgeMs: value }) => value),
      0.95,
    ),
    medianTotalMs: percentile(
      observations.map(({ totalMs: value }) => value),
      0.5,
    ),
    p95TotalMs: percentile(
      observations.map(({ totalMs: value }) => value),
      0.95,
    ),
    medianUserTurnMs: percentile(
      observations.map(({ userTurnMs: value }) => value ?? 0),
      0.5,
    ),
    p95UserTurnMs: percentile(
      observations.map(({ userTurnMs: value }) => value ?? 0),
      0.95,
    ),
    readerUsage,
    judgeUsage,
    extractionUsage: { ...extractionTotals, ...extractionCounts },
    embeddingUsage,
    semanticPreparationUsage,
  };
}

export function longMemEvalAnswerRun(
  observations: LongMemEvalAnswerObservation[],
  readerModel: string,
  judgeModel: string,
  options: {
    topK?: number;
    multiSessionTopK?: number;
    temporalTopK?: number;
    contextBytes?: number;
    generatedAt?: string;
    embeddingModel?: string | null;
    selection?: 'dev' | 'test' | 'all';
    sha256?: string;
    semanticQuestionTypes?: ReadonlySet<string>;
    multiSessionSemanticMaximumLexicalScore?: number;
    prepareSemantic?: boolean;
    formation?: LongMemEvalFormation;
    extractionModel?: string | null;
    settings?: LongMemEvalAnswerRun['settings'];
  } = {},
): LongMemEvalAnswerRun {
  const topK = options.topK ?? DEFAULT_LONGMEMEVAL_ANSWER_TOP_K;
  const multiSessionTopK =
    options.multiSessionTopK ?? DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K;
  const temporalTopK =
    options.temporalTopK ?? DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K;
  const contextBytes =
    options.contextBytes ?? DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES;
  validateOptions(topK, contextBytes);
  validateOptions(multiSessionTopK, contextBytes);
  validateOptions(temporalTopK, contextBytes);
  const questionTypes = [
    ...new Set(observations.map(({ questionType }) => questionType)),
  ].sort();
  return {
    schemaVersion: LONGMEMEVAL_ANSWER_VERSION,
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    dataset: {
      id: 'xiaowu0162/longmemeval-cleaned',
      split: 'longmemeval_s_cleaned',
      selection: options.selection ?? 'all',
      commit: LONGMEMEVAL_S_COMMIT,
      sha256: options.sha256 ?? LONGMEMEVAL_S_SHA256,
    },
    readerModel,
    judgeModel,
    embeddingModel: options.embeddingModel ?? null,
    judgeProtocol: 'longmemeval-official-compatible-v1',
    formation: LONGMEMEVAL_FORMATION_LABELS[options.formation ?? 'raw'],
    ...(options.settings === undefined ? {} : { settings: options.settings }),
    extractionModel: options.extractionModel ?? null,
    retrieval:
      options.embeddingModel === undefined || options.embeddingModel === null
        ? 'remembero-local-source-search'
        : 'remembero-adaptive-source-search',
    answerContextPolicy: 'user-turns-except-assistant-memory',
    semanticQuestionTypes: [
      ...(options.semanticQuestionTypes ??
        DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES),
    ].sort(),
    multiSessionSemanticMaximumLexicalScore:
      options.multiSessionSemanticMaximumLexicalScore ??
      LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE,
    semanticPreparation: options.prepareSemantic === true ? 'prepared' : 'cold',
    topK,
    multiSessionTopK,
    temporalTopK,
    sourceCharacters: LONGMEMEVAL_ANSWER_SOURCE_CHARACTERS,
    contextBytes,
    summary: summarizeLongMemEvalAnswers(observations),
    byQuestionType: Object.fromEntries(
      questionTypes.map((questionType) => [
        questionType,
        summarizeLongMemEvalAnswers(
          observations.filter((value) => value.questionType === questionType),
        ),
      ]),
    ),
    observations,
  };
}
