import { describe, expect, it } from 'vitest';
import {
  buildDocumentJudgePrompt,
  declinedToAnswer,
  evaluateDocumentTier,
  oracleSessions,
  parseDocumentJudgeLabel,
} from '../src/evals/document-recall-run.js';
import { windowPages, windowsAsSessions } from '../src/evals/document-corpus.js';
import type { DocumentQuestion, DocumentTier } from '../src/evals/document-recall.js';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import type { LongMemEvalCompletionClient } from '../src/evals/longmemeval-answer.js';

const completion = (content: string): LlmCompletion => ({
  content,
  usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15, cost: null },
});

/** A reader that answers from whatever text it was shown, so retrieval decides the outcome. */
const readerThatQuotes = (needle: string, reply: string): LongMemEvalCompletionClient => ({
  model: 'fake-reader',
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    const shown = messages.map((message) => message.content).join('\n');
    return completion(shown.includes(needle) ? `Answer: ${reply}` : 'Answer: no information');
  },
});

const judgeAgainstGold = (gold: string): LongMemEvalCompletionClient => ({
  model: 'fake-judge',
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    return completion(messages[0]!.content.includes(`Response: ${gold}`) ? 'yes' : 'no');
  },
});

const pagesWithFact = (count: number, factPage: number, fact: string) =>
  Array.from({ length: count }, (_unused, index) => ({
    page: index + 1,
    text:
      index + 1 === factPage
        ? `Section ${index + 1}. ${fact}`
        : `Section ${index + 1}. Routine boilerplate about procurement schedules and annexes.`,
  }));

const question = (overrides: Partial<DocumentQuestion> = {}): DocumentQuestion => ({
  id: 'q1',
  question: 'What was the remediation budget for the Kestrel programme?',
  answer: '4.2 million pounds',
  evidencePages: [7],
  sourceDocument: 'doc-a',
  ...overrides,
});

const tierOf = (questions: DocumentQuestion[], pages: number): DocumentTier => ({
  tier: 'test',
  pages,
  members: [],
  questions,
});

describe('declinedToAnswer', () => {
  it('spots the phrases a reader uses when the pages do not answer', () => {
    expect(declinedToAnswer('No information about that in the document.')).toBe(true);
    expect(declinedToAnswer('The figure is not stated.')).toBe(true);
    expect(declinedToAnswer('4.2 million pounds')).toBe(false);
  });

  it("recognises the product prompt's own wording", () => {
    expect(declinedToAnswer('I do not know.')).toBe(true);
    expect(declinedToAnswer("I don't know from these pages.")).toBe(true);
  });
});

describe('buildDocumentJudgePrompt', () => {
  it('shows the gold answer for an answerable question', () => {
    const prompt = buildDocumentJudgePrompt(question(), '4.2 million');
    expect(prompt).toContain('Correct answer: 4.2 million pounds');
    expect(prompt).toContain('Answer yes or no only.');
  });

  it('asks for a declining answer when the document cannot answer', () => {
    const prompt = buildDocumentJudgePrompt(question({ answer: null, evidencePages: [] }), 'unknown');
    expect(prompt).toContain('does not answer the question');
    expect(prompt).not.toContain('Correct answer:');
  });
});

describe('parseDocumentJudgeLabel', () => {
  it('accepts yes and no in any casing or punctuation', () => {
    expect(parseDocumentJudgeLabel('Yes.')).toBe(true);
    expect(parseDocumentJudgeLabel('  no\n')).toBe(false);
  });

  it('refuses anything else rather than scoring it', () => {
    expect(() => parseDocumentJudgeLabel('probably')).toThrow(/yes or no/);
  });
});

describe('evaluateDocumentTier', () => {
  const fact = 'The Kestrel programme remediation budget was 4.2 million pounds.';

  it('scores a hit when retrieval finds the evidence page and the reader answers', async () => {
    const pages = pagesWithFact(40, 7, fact);
    const result = await evaluateDocumentTier(tierOf([question()], pages.length), pages, {
      reader: readerThatQuotes('4.2 million pounds', '4.2 million pounds'),
      judge: judgeAgainstGold('4.2 million pounds'),
      topK: 4,
      contextBytes: 24 * 1024,
      pagesPerWindow: 4,
      windowBytes: 12 * 1024,
      concurrency: 1,
    });
    expect(result.errors).toEqual([]);
    expect(result.summary.questions).toBe(1);
    expect(result.summary.accuracy).toBe(1);
    expect(result.summary.evidenceHitRate).toBe(1);
    expect(result.outcomes[0]!.retrieval.pageRecall).toBe(1);
    expect(result.summary.windows).toBe(10);
  });

  it('records a page-miss as a wrong answer, not an error', async () => {
    // the fact sits on page 150 of 200, and the question's words match the boilerplate on every
    // page, so a depth-4 retrieval cannot reach it: a miss the scorer must record as an answer
    // that was never possible, not as a broken run
    const pages = pagesWithFact(200, 150, fact);
    const result = await evaluateDocumentTier(
      tierOf([question({ question: 'Which annexe lists procurement schedules?', evidencePages: [150] })], pages.length),
      pages,
      {
        reader: readerThatQuotes('4.2 million pounds', '4.2 million pounds'),
        judge: judgeAgainstGold('4.2 million pounds'),
        topK: 4,
        contextBytes: 24 * 1024,
        pagesPerWindow: 4,
        windowBytes: 12 * 1024,
        concurrency: 1,
      },
    );
    expect(result.summary.accuracy).toBe(0);
    expect(result.outcomes[0]!.answer).toBe('no information');
    expect(result.outcomes[0]!.abstained).toBe(true);
  });

  it('collects a reader failure as an error and still summarises the rest', async () => {
    const pages = pagesWithFact(12, 3, fact);
    const failing: LongMemEvalCompletionClient = {
      model: 'broken',
      async completeWithUsage() {
        throw new Error('reader endpoint refused the connection');
      },
    };
    const result = await evaluateDocumentTier(tierOf([question()], pages.length), pages, {
      reader: failing,
      judge: judgeAgainstGold('x'),
      topK: 4,
      contextBytes: 24 * 1024,
      pagesPerWindow: 4,
      windowBytes: 12 * 1024,
      concurrency: 1,
    });
    expect(result.outcomes).toEqual([]);
    expect(result.errors).toEqual([
      { questionId: 'q1', message: 'reader endpoint refused the connection' },
    ]);
    expect(result.summary.questions).toBe(0);
  });

  it('counts the reader tokens a read cost', async () => {
    const pages = pagesWithFact(8, 2, fact);
    const result = await evaluateDocumentTier(tierOf([question({ evidencePages: [2] })], pages.length), pages, {
      reader: readerThatQuotes('4.2 million pounds', '4.2 million pounds'),
      judge: judgeAgainstGold('4.2 million pounds'),
      topK: 4,
      contextBytes: 24 * 1024,
      pagesPerWindow: 4,
      windowBytes: 12 * 1024,
      concurrency: 1,
    });
    expect(result.outcomes[0]!.readerTokens).toEqual({ prompt: 10, completion: 5 });
  });
});

describe('oracleSessions', () => {
  const sessions = windowsAsSessions(
    windowPages(
      Array.from({ length: 40 }, (_unused, index) => ({ page: index + 1, text: `page ${index + 1}` })),
      { pagesPerWindow: 4 },
    ),
  );

  it('returns exactly the windows holding the gold pages', () => {
    expect(oracleSessions(sessions, [7, 33], 4).map((session) => session.id)).toEqual([
      'pages-0005-0008',
      'pages-0033-0036',
    ]);
  });

  it('is empty for an unanswerable question, so the ceiling is not inflated', () => {
    expect(oracleSessions(sessions, [], 4)).toEqual([]);
  });

  it('never shows more windows than a real retrieval of the same depth would', () => {
    expect(oracleSessions(sessions, [1, 6, 11, 16, 21, 26], 4)).toHaveLength(4);
  });
});
