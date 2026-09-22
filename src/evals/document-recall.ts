/**
 * The page-scale recall benchmark: one document of N pages, questions whose evidence sits on a
 * known page, and three numbers per tier — did retrieval reach the page, did the reader answer,
 * and how much of what it read was worth reading.
 *
 * A tier is built by concatenating labelled documents until the page count is exact (100, 500,
 * 1000), remapping each question's evidence pages by the offset its document landed at. Pages the
 * questions do not touch are not padding: they are the distractors that make a 1000-page tier
 * harder than a 100-page one, which is the whole point of the sweep.
 */

import type { PageWindow } from './document-corpus.js';
import { pageRangeFromId, windowHoldsPage } from './document-corpus.js';

/** A labelled question over a document: gold answer plus the pages that justify it. */
export interface DocumentQuestion {
  id: string;
  question: string;
  /** The gold answer, or null when the document cannot answer it (an abstention case). */
  answer: string | null;
  /** Pages of the assembled tier that hold the evidence; empty for an unanswerable question. */
  evidencePages: number[];
  /** Which source document it came from, kept so a tier's misses can be traced back. */
  sourceDocument: string;
  /** The dataset's own question type, recorded but never used to steer answering. */
  datasetKind?: string;
  /** XL-DocBench's answer format (Str, Int, Float, None, ...), which picks its scoring rule. */
  answerFormat?: string;
  /** XL-DocBench's verification rule, used when the format alone does not name a type. */
  verificationRule?: string;
  /** The document this question belongs to, when a tier holds several. */
  documentId?: string;
}

/** One source document inside an assembled tier. */
export interface TierMember {
  id: string;
  title: string;
  sourceUrl: string;
  sha256: string;
  pages: number;
  /** Page 1 of this document is this page of the tier. */
  pageOffset: number;
}

/** A tier: an assembled document of an exact page count, with its labelled questions. */
export interface DocumentTier {
  tier: string;
  pages: number;
  members: TierMember[];
  questions: DocumentQuestion[];
}

/** What one question's retrieval achieved, in pages rather than sessions. */
export interface RetrievalScore {
  /** Did any chosen window hold a gold evidence page? */
  evidenceHit: boolean;
  /** Fraction of the gold pages that ended up in front of the reader. */
  pageRecall: number;
  /** Fraction of the chosen windows that held at least one gold page. */
  windowPrecision: number;
  pagesShown: number;
  windowsShown: number;
}

/**
 * Score a retrieval against the labels. Precision is per window, not per page: a window is the
 * unit the retrieval actually chose, and counting pages would punish a correct window for the
 * three neighbouring pages it carries along by design.
 */
export function scoreRetrieval(
  chosen: Array<{ id: string }>,
  evidencePages: number[],
): RetrievalScore {
  const ranges = chosen
    .map((session) => pageRangeFromId(session.id))
    .filter((range): range is { firstPage: number; lastPage: number } => range !== undefined);
  const pagesShown = ranges.reduce((sum, range) => sum + (range.lastPage - range.firstPage + 1), 0);
  if (evidencePages.length === 0) {
    return { evidenceHit: false, pageRecall: 0, windowPrecision: 0, pagesShown, windowsShown: ranges.length };
  }
  const found = evidencePages.filter((page) => ranges.some((range) => windowHoldsPage(range, page)));
  const useful = ranges.filter((range) => evidencePages.some((page) => windowHoldsPage(range, page)));
  return {
    evidenceHit: found.length > 0,
    pageRecall: found.length / evidencePages.length,
    windowPrecision: ranges.length === 0 ? 0 : useful.length / ranges.length,
    pagesShown,
    windowsShown: ranges.length,
  };
}

/** The window a page falls in, for turning a dataset's evidence page into a gold window id. */
export function windowForPage(windows: PageWindow[], page: number): PageWindow | undefined {
  return windows.find((window) => windowHoldsPage(window, page));
}

/** Per-question outcome: retrieval, the reader's answer, and the judge's verdict. */
export interface QuestionOutcome {
  question: DocumentQuestion;
  retrieval: RetrievalScore;
  answer: string;
  /** The benchmark's own rule: 1 or 0, relaxed by answer type. */
  correct: boolean;
  /** The same rule once a decimal comma is read as a decimal point (a diagnostic, not the number). */
  correctLocale: boolean;
  /** Their two surface-form diagnostics, reported alongside accuracy. */
  tokenF1: number;
  anls: number;
  /** A judge's opinion, when one was asked; the rule is the headline either way. */
  judged?: boolean;
  /** An unanswerable question is right only when the reader declines to answer. */
  abstained: boolean;
  latencyMs: number;
  contextBytes: number;
  /** What the read cost, so a tier's price per answer can be reported alongside its accuracy. */
  readerTokens?: { prompt: number; completion: number; reasoning?: number };
  /**
   * Dollars for the read: what the endpoint reported (OpenRouter does), else the tokens priced at
   * the caller's rate, else undefined. A local reader priced at zero is zero, not unknown.
   */
  readerCostUsd?: number;
  /** Where the time went: ranking the pages, and the reader producing its answer. */
  retrievalMs?: number;
  readerMs?: number;
}

export interface TierSummary {
  tier: string;
  pages: number;
  windows: number;
  questions: number;
  answerable: number;
  unanswerable: number;
  /** Correct answers over all questions, the headline. */
  accuracy: number;
  /** Correct answers over answerable questions only. */
  answerableAccuracy: number;
  /** Accuracy once a decimal comma counts as a decimal point. */
  localeAccuracy: number;
  /** Of the answerable questions, how often retrieval put a gold page in front of the reader. */
  evidenceHitRate: number;
  meanPageRecall: number;
  meanWindowPrecision: number;
  /** Answered when it should have declined: the hallucination rate on unanswerable questions. */
  falseAnswerRate: number;
  meanLatencyMs: number;
  meanContextBytes: number;
  /** XL-DocBench's secondary diagnostics over the same answers. */
  meanTokenF1: number;
  meanAnls: number;
  /** The speed lens: time to rank pages versus time to read them, and the tail. */
  meanRetrievalMs: number;
  meanReaderMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  /** The cost lens: tokens per answer and dollars per answer, where the price is known. */
  meanPromptTokens: number;
  meanCompletionTokens: number;
  meanReasoningTokens: number;
  costPerAnswerUsd?: number;
  totalCostUsd?: number;
  /** Dollars per correct answer by the judge (else the rule): what one right answer costs. */
  costPerCorrectUsd?: number;
  /** Judge accuracy over the questions a judge saw, when one was asked. */
  judgedAccuracy?: number;
  judgedQuestions: number;
}

/** Roll per-question outcomes into the tier row that goes in the results table. */
export function summariseTier(
  tier: DocumentTier,
  windows: number,
  outcomes: QuestionOutcome[],
): TierSummary {
  const answerable = outcomes.filter((outcome) => outcome.question.answer !== null);
  const unanswerable = outcomes.filter((outcome) => outcome.question.answer === null);
  const judged = outcomes.filter((outcome) => outcome.judged !== undefined);
  const latencies = outcomes.map((o) => o.latencyMs).sort((a, b) => a - b);
  const quantile = (q: number) =>
    latencies.length === 0 ? 0 : latencies[Math.min(latencies.length - 1, Math.floor(q * latencies.length))]!;
  const priced = outcomes.filter((o) => o.readerCostUsd !== undefined);
  const totalCost = priced.length === outcomes.length && outcomes.length > 0
    ? priced.reduce((sum, o) => sum + o.readerCostUsd!, 0)
    : undefined;
  const rightAnswers = judged.length > 0
    ? judged.filter((o) => o.judged === true).length
    : outcomes.filter((o) => o.correct).length;
  const mean = (values: number[]) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);
  return {
    tier: tier.tier,
    pages: tier.pages,
    windows,
    questions: outcomes.length,
    answerable: answerable.length,
    unanswerable: unanswerable.length,
    accuracy: outcomes.length === 0 ? 0 : outcomes.filter((o) => o.correct).length / outcomes.length,
    answerableAccuracy: answerable.length === 0 ? 0 : answerable.filter((o) => o.correct).length / answerable.length,
    localeAccuracy: outcomes.length === 0 ? 0 : outcomes.filter((o) => o.correctLocale).length / outcomes.length,
    evidenceHitRate: answerable.length === 0 ? 0 : answerable.filter((o) => o.retrieval.evidenceHit).length / answerable.length,
    meanPageRecall: mean(answerable.map((o) => o.retrieval.pageRecall)),
    meanWindowPrecision: mean(answerable.map((o) => o.retrieval.windowPrecision)),
    falseAnswerRate: unanswerable.length === 0 ? 0 : unanswerable.filter((o) => !o.abstained).length / unanswerable.length,
    meanLatencyMs: mean(outcomes.map((o) => o.latencyMs)),
    meanContextBytes: mean(outcomes.map((o) => o.contextBytes)),
    meanRetrievalMs: mean(outcomes.map((o) => o.retrievalMs ?? 0)),
    meanReaderMs: mean(outcomes.map((o) => o.readerMs ?? 0)),
    p50LatencyMs: quantile(0.5),
    p95LatencyMs: quantile(0.95),
    meanPromptTokens: mean(outcomes.map((o) => o.readerTokens?.prompt ?? 0)),
    meanCompletionTokens: mean(outcomes.map((o) => o.readerTokens?.completion ?? 0)),
    meanReasoningTokens: mean(outcomes.map((o) => o.readerTokens?.reasoning ?? 0)),
    ...(totalCost === undefined
      ? {}
      : {
          totalCostUsd: totalCost,
          costPerAnswerUsd: totalCost / outcomes.length,
          ...(rightAnswers === 0 ? {} : { costPerCorrectUsd: totalCost / rightAnswers }),
        }),
    meanTokenF1: mean(answerable.map((o) => o.tokenF1)),
    meanAnls: mean(answerable.map((o) => o.anls)),
    ...(judged.length === 0 ? {} : { judgedAccuracy: judged.filter((o) => o.judged === true).length / judged.length }),
    judgedQuestions: judged.length,
  };
}
