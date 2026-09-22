/**
 * Run the page-scale benchmark over one assembled tier.
 *
 * The clients are injected, so the whole loop is testable without a model: the reader and the
 * judge are the same `LongMemEvalCompletionClient` shape the chat harness uses, and retrieval is
 * the product's own `retrieveSessions` / `buildReadingPrompt` — a different corpus, not a
 * different reader. Nothing here reads a dataset label to decide how to answer: the question's
 * kind comes from its text, as it does in the product.
 */

import { performance } from 'node:perf_hooks';
import { finalAnswerLine } from '../knowledge/answer-line.js';
import { questionKindFromText } from '../knowledge/question-kind.js';
import {
  buildReadingPrompt,
  retrieveSessions,
  type RetrievableSession,
  type SessionRetrievalOptions,
} from '../knowledge/session-retrieval.js';
import { windowHoldsPage, windowPages, windowsAsSessions, type DocumentPage } from './document-corpus.js';
import {
  scoreRetrieval,
  summariseTier,
  type DocumentQuestion,
  type DocumentTier,
  type QuestionOutcome,
  type TierSummary,
} from './document-recall.js';
import { mapConcurrent } from './map-concurrent.js';
import { scoreAgainstXlDocBench, scoreWithLocaleTolerance, xlAnswerType } from './xl-docbench-score.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';

/** The phrases a reader uses when the pages do not answer the question. */
const DECLINED = [
  // the product's reading prompt tells the reader to "say that you do not know", so its own
  // wording has to be recognised here — XL-DocBench's rule does not accept it, which is why the
  // two abstention numbers in a tier row can disagree
  'do not know',
  "don't know",
  'dont know',
  'no information',
  'not mentioned',
  'not stated',
  'not specified',
  'not provided',
  'does not say',
  "doesn't say",
  'cannot be determined',
  'cannot determine',
  "can't determine",
  'cannot answer',
  'unable to answer',
  'no answer',
  'not found in',
  'not in the',
  'unanswerable',
  'unknown',
  'insufficient information',
];

/** Did the reader decline rather than answer? Judged on the final answer line alone. */
export function declinedToAnswer(answer: string): boolean {
  const text = answer.toLowerCase();
  return DECLINED.some((phrase) => text.includes(phrase));
}

/**
 * The judge prompt. A document question has one gold answer and no personalisation rubric, so
 * this is the chat harness's contract minus the cases a document cannot have.
 */
export function buildDocumentJudgePrompt(question: DocumentQuestion, response: string): string {
  const common = `Question: ${question.question}\nResponse: ${response}`;
  if (question.answer === null) {
    return `Judge whether the response correctly says the document does not answer the question. Answer yes when it declines or reports the information as absent, no when it states an answer.\n\n${common}\n\nAnswer yes or no only.`;
  }
  return `Judge whether the response gives the correct answer to a question about a long document. Wording, rounding of a figure to the same value, and extra correct detail are all acceptable; a different value, a different entity, or a partial answer is not.\n\nCorrect answer: ${question.answer}\n${common}\n\nAnswer yes or no only.`;
}

/** yes/no, strictly: a judge that says anything else is a broken run, not a wrong answer. */
export function parseDocumentJudgeLabel(value: string): boolean {
  const first = value.trim().toLowerCase().replace(/[^a-z]/g, ' ').trim().split(/\s+/)[0];
  if (first === 'yes') return true;
  if (first === 'no') return false;
  throw new Error('document judge must answer yes or no only');
}

export interface DocumentRecallOptions {
  reader: LongMemEvalCompletionClient;
  /**
   * Optional second opinion. The headline accuracy is the benchmark's own rule, so a run without
   * a judge is a complete run; a judge only says how much a strict rule undercounts.
   */
  judge?: LongMemEvalCompletionClient;
  /** Retrieval depth for a plain lookup; aggregation and temporal keep their own defaults. */
  topK: number;
  contextBytes: number;
  pagesPerWindow: number;
  windowBytes: number;
  maxTokens?: number;
  /**
   * Skip retrieval and show the reader exactly the windows holding the gold evidence pages. Not a
   * result the product can claim — it is the ceiling, the answer to "how much of the miss is
   * retrieval's fault?". A tier's oracle accuracy bounds its real accuracy from above.
   */
  oraclePages?: boolean;
  /** Dollars per million tokens, for an endpoint that does not report its own cost. */
  price?: { inputPerMillion: number; outputPerMillion: number };
  concurrency: number;
  /** Called after each question so a long tier reports progress as it goes. */
  onQuestion?: (outcome: QuestionOutcome, index: number, total: number) => void;
}

export interface TierResult {
  summary: TierSummary;
  outcomes: QuestionOutcome[];
  sessions: RetrievableSession[];
  errors: Array<{ questionId: string; message: string }>;
}

/** Retrieval, reading and judging for every question of one tier. */
export async function evaluateDocumentTier(
  tier: DocumentTier,
  pages: DocumentPage[],
  options: DocumentRecallOptions,
): Promise<TierResult> {
  const windows = windowPages(pages, {
    pagesPerWindow: options.pagesPerWindow,
    maxBytes: options.windowBytes,
  });
  const sessions = windowsAsSessions(windows);
  const byId = new Map(sessions.map((session) => [session.id, session]));
  const askedAt = new Date(sessions[sessions.length - 1]?.date ?? Date.now());
  const errors: Array<{ questionId: string; message: string }> = [];

  const results = await mapConcurrent(
    tier.questions,
    options.concurrency,
    async (question, index) => {
      try {
        const outcome = await evaluateDocumentQuestion(question, sessions, byId, askedAt, options);
        options.onQuestion?.(outcome, index, tier.questions.length);
        return outcome;
      } catch (error) {
        errors.push({
          questionId: question.id,
          message: error instanceof Error ? error.message : String(error),
        });
        return undefined;
      }
    },
  );

  const outcomes = results.filter((outcome): outcome is QuestionOutcome => outcome !== undefined);
  return { summary: summariseTier(tier, windows.length, outcomes), outcomes, sessions, errors };
}

/** One question: rank the windows, read the chosen ones, judge the final answer line. */
export async function evaluateDocumentQuestion(
  question: DocumentQuestion,
  sessions: readonly RetrievableSession[],
  byId: Map<string, RetrievableSession>,
  askedAt: Date,
  options: DocumentRecallOptions,
): Promise<QuestionOutcome> {
  const kind = questionKindFromText(question.question);
  const retrieval: SessionRetrievalOptions = {
    kind,
    topK: options.topK,
    contextBytes: options.contextBytes,
    dateDistances: false,
    computedNotes: false,
  };
  const started = performance.now();
  const chosenSessions =
    options.oraclePages === true
      ? oracleSessions(sessions, question.evidencePages, options.topK)
      : (await retrieveSessions(question.question, askedAt, sessions, retrieval)).chosen
          .map((id) => byId.get(id))
          .filter((session): session is RetrievableSession => session !== undefined);
  const retrievalMs = performance.now() - started;
  const prompt = buildReadingPrompt(question.question, askedAt, chosenSessions, retrieval);
  const readStarted = performance.now();
  const completion = await options.reader.completeWithUsage(
    [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user },
    ],
    options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens },
  );
  const answer = finalAnswerLine(completion.content);
  const readerMs = performance.now() - readStarted;
  const latencyMs = performance.now() - started;
  const promptTokens = completion.usage?.promptTokens ?? 0;
  const completionTokens = completion.usage?.completionTokens ?? 0;
  const readerCostUsd =
    completion.usage?.costUsd ??
    (options.price === undefined
      ? undefined
      : (promptTokens * options.price.inputPerMillion + completionTokens * options.price.outputPerMillion) / 1e6);

  const answerType = xlAnswerType(question.answerFormat, question.verificationRule);
  const scored = scoreAgainstXlDocBench(answer, question.answer ?? '', answerType);
  const locale = scoreWithLocaleTolerance(answer, question.answer ?? '', answerType);
  const judged =
    options.judge === undefined
      ? undefined
      : parseDocumentJudgeLabel(
          (
            await options.judge.completeWithUsage([
              { role: 'user', content: buildDocumentJudgePrompt(question, answer) },
            ])
          ).content,
        );
  return {
    question,
    retrieval: scoreRetrieval(chosenSessions, question.evidencePages),
    answer,
    correct: scored.accuracy === 1,
    correctLocale: locale.localeAccuracy === 1,
    tokenF1: scored.tokenF1,
    anls: scored.anls,
    ...(judged === undefined ? {} : { judged }),
    abstained: declinedToAnswer(answer),
    latencyMs,
    contextBytes: Buffer.byteLength(prompt.user, 'utf8'),
    readerTokens: {
      prompt: promptTokens,
      completion: completionTokens,
      reasoning: completion.usage?.reasoningTokens ?? 0,
    },
    ...(readerCostUsd === undefined ? {} : { readerCostUsd }),
    retrievalMs,
    readerMs,
  };
}

/**
 * The windows holding the gold evidence pages, in page order, capped at the same depth a real
 * retrieval gets — otherwise the oracle would also be given more context than the product.
 */
export function oracleSessions(
  sessions: readonly RetrievableSession[],
  evidencePages: readonly number[],
  depth: number,
): RetrievableSession[] {
  const wanted = sessions.filter((session) => {
    const range = pageRangeOf(session.id);
    return range !== undefined && evidencePages.some((page) => windowHoldsPage(range, page));
  });
  return wanted.slice(0, depth);
}

function pageRangeOf(id: string): { firstPage: number; lastPage: number } | undefined {
  const match = /^pages-(\d+)-(\d+)$/.exec(id);
  if (match === null) return undefined;
  return { firstPage: Number(match[1]), lastPage: Number(match[2]) };
}
