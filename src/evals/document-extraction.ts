/**
 * The extraction half of the page-scale benchmark: what the writer pulls out of a page, and
 * whether it is true.
 *
 * Two numbers, both against evidence rather than a vocabulary:
 *
 *   precision — a sampled fact is shown to a judge beside the page it came from, and counted only
 *   when the page supports it. This is the number that matters for a memory product: a store full
 *   of confident inventions is worse than an empty one.
 *
 *   recall — for each labelled question, the facts extracted from its evidence pages are shown to
 *   the judge with the gold answer. The facts score when the answer follows from them alone. So
 *   recall is measured against what the document was asked for, not against a fact list nobody
 *   wrote.
 *
 * Extraction is sampled, because a thousand pages through a local 4B writer is hours of GPU: the
 * sample always contains every labelled evidence page, so recall is exact, and precision carries
 * the sample size it was measured on.
 */

import type { DocumentPage } from './document-corpus.js';
import type { DocumentQuestion } from './document-recall.js';
import { mapConcurrent } from './map-concurrent.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';

/** Whatever the product's writer does to one page, as this benchmark needs to see it. */
export type PageExtractor = (page: DocumentPage) => Promise<{ facts: string[] }>;

export interface ExtractionOptions {
  extract: PageExtractor;
  judge: LongMemEvalCompletionClient;
  /** Pages sampled beyond the evidence pages, spread evenly through the document. */
  samplePages: number;
  /** Facts judged for support, at most, spread across the sampled pages. */
  judgeFacts: number;
  concurrency: number;
  onPage?: (page: number, facts: number, index: number, total: number) => void;
}

export interface FactVerdict {
  page: number;
  fact: string;
  supported: boolean;
}

export interface QuestionCoverage {
  questionId: string;
  evidencePages: number[];
  facts: number;
  answered: boolean;
}

export interface ExtractionSummary {
  pagesSampled: number;
  evidencePagesSampled: number;
  facts: number;
  factsPerPage: number;
  pagesWithNoFacts: number;
  extractionErrors: number;
  factsJudged: number;
  /** Supported facts over judged facts: how much of the store is true. */
  precision: number;
  questionsScored: number;
  /** Questions whose answer follows from the facts extracted off their evidence pages. */
  recall: number;
}

export interface ExtractionResult {
  summary: ExtractionSummary;
  verdicts: FactVerdict[];
  coverage: QuestionCoverage[];
  perPage: Array<{ page: number; facts: string[]; error?: string }>;
}

/**
 * Which pages to extract from: every labelled evidence page, plus an even spread of the rest, so
 * the precision sample is not all front matter.
 */
export function samplePagesToExtract(
  pages: DocumentPage[],
  questions: DocumentQuestion[],
  extra: number,
): number[] {
  const evidence = new Set<number>();
  for (const question of questions) for (const page of question.evidencePages) evidence.add(page);
  const chosen = new Set(evidence);
  if (extra > 0 && pages.length > 0) {
    const stride = Math.max(1, Math.floor(pages.length / extra));
    for (let page = 1; page <= pages.length && chosen.size < evidence.size + extra; page += stride) {
      chosen.add(page);
    }
  }
  return [...chosen].filter((page) => page >= 1 && page <= pages.length).sort((a, b) => a - b);
}

/** Judge one fact against the page it was taken from. */
export function buildFactSupportPrompt(fact: string, pageText: string): string {
  return (
    'Does the page below support this stored fact? Answer yes only when the page states it, ' +
    'directly or by plain paraphrase. Answer no when the fact adds, changes or infers anything ' +
    'the page does not say.\n\n' +
    `Stored fact: ${fact}\n\nPage:\n${pageText.slice(0, 12_000)}\n\nAnswer yes or no only.`
  );
}

/** Judge whether the extracted facts alone answer a labelled question. */
export function buildFactCoveragePrompt(
  question: DocumentQuestion,
  facts: string[],
): string {
  return (
    'Do the stored facts below contain the answer to the question? Answer yes only when the ' +
    'correct answer can be read off the facts without consulting the document.\n\n' +
    `Question: ${question.question}\nCorrect answer: ${question.answer ?? '(none: the document does not answer it)'}\n\n` +
    `Stored facts:\n${facts.length === 0 ? '(none)' : facts.join('\n')}\n\nAnswer yes or no only.`
  );
}

function yesNo(value: string): boolean {
  const first = value.trim().toLowerCase().replace(/[^a-z]/g, ' ').trim().split(/\s+/)[0];
  if (first === 'yes') return true;
  if (first === 'no') return false;
  throw new Error('fact judge must answer yes or no only');
}

/** Extract from the sampled pages, then measure precision and recall over what came out. */
export async function measureDocumentExtraction(
  pages: DocumentPage[],
  questions: DocumentQuestion[],
  options: ExtractionOptions,
): Promise<ExtractionResult> {
  const sampled = samplePagesToExtract(pages, questions, options.samplePages);
  const byNumber = new Map(pages.map((page) => [page.page, page]));
  const perPage: Array<{ page: number; facts: string[]; error?: string }> = [];
  let extractionErrors = 0;

  await mapConcurrent(sampled, options.concurrency, async (pageNumber, index) => {
    const page = byNumber.get(pageNumber);
    if (page === undefined) return;
    try {
      const { facts } = await options.extract(page);
      perPage.push({ page: pageNumber, facts });
      options.onPage?.(pageNumber, facts.length, index, sampled.length);
    } catch (error) {
      extractionErrors += 1;
      perPage.push({
        page: pageNumber,
        facts: [],
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
  perPage.sort((left, right) => left.page - right.page);

  const allFacts = perPage.flatMap((entry) => entry.facts.map((fact) => ({ page: entry.page, fact })));
  const toJudge = evenSample(allFacts, options.judgeFacts);
  const verdicts = await mapConcurrent(toJudge, options.concurrency, async ({ page, fact }) => {
    const text = byNumber.get(page)?.text ?? '';
    const reply = await options.judge.completeWithUsage([
      { role: 'user', content: buildFactSupportPrompt(fact, text) },
    ]);
    return { page, fact, supported: yesNo(reply.content) };
  });

  const answerable = questions.filter((question) => question.answer !== null && question.evidencePages.length > 0);
  const factsOfPage = new Map(perPage.map((entry) => [entry.page, entry.facts]));
  const coverage = await mapConcurrent(answerable, options.concurrency, async (question) => {
    const facts = question.evidencePages.flatMap((page) => factsOfPage.get(page) ?? []);
    const reply = await options.judge.completeWithUsage([
      { role: 'user', content: buildFactCoveragePrompt(question, facts) },
    ]);
    return {
      questionId: question.id,
      evidencePages: question.evidencePages,
      facts: facts.length,
      answered: yesNo(reply.content),
    };
  });

  const evidencePages = new Set(answerable.flatMap((question) => question.evidencePages));
  return {
    summary: {
      pagesSampled: sampled.length,
      evidencePagesSampled: sampled.filter((page) => evidencePages.has(page)).length,
      facts: allFacts.length,
      factsPerPage: sampled.length === 0 ? 0 : allFacts.length / sampled.length,
      pagesWithNoFacts: perPage.filter((entry) => entry.facts.length === 0 && entry.error === undefined).length,
      extractionErrors,
      factsJudged: verdicts.length,
      precision: verdicts.length === 0 ? 0 : verdicts.filter((verdict) => verdict.supported).length / verdicts.length,
      questionsScored: coverage.length,
      recall: coverage.length === 0 ? 0 : coverage.filter((entry) => entry.answered).length / coverage.length,
    },
    verdicts,
    coverage,
    perPage,
  };
}

/** An even sample, so the judged facts are not all from the first pages extracted. */
export function evenSample<T>(values: readonly T[], limit: number): T[] {
  if (limit <= 0 || values.length <= limit) return [...values];
  const stride = values.length / limit;
  const picked: T[] = [];
  for (let index = 0; picked.length < limit; index += 1) {
    const at = Math.floor(index * stride);
    if (at >= values.length) break;
    picked.push(values[at]!);
  }
  return picked;
}
