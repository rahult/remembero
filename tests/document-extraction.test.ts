import { describe, expect, it } from 'vitest';
import {
  buildFactCoveragePrompt,
  buildFactSupportPrompt,
  evenSample,
  measureDocumentExtraction,
  samplePagesToExtract,
} from '../src/evals/document-extraction.js';
import type { DocumentQuestion } from '../src/evals/document-recall.js';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import type { LongMemEvalCompletionClient } from '../src/evals/longmemeval-answer.js';

const pages = (count: number) =>
  Array.from({ length: count }, (_unused, index) => ({
    page: index + 1,
    text: `page ${index + 1} states that budget line ${index + 1} was approved`,
  }));

const question = (id: string, evidencePages: number[], answer: string | null = 'approved'): DocumentQuestion => ({
  id,
  question: `was budget line ${evidencePages[0] ?? 0} approved?`,
  answer,
  evidencePages,
  sourceDocument: 'doc',
});

/**
 * A judge that only ever reads the evidence it is given: the page for a support prompt, the
 * stored facts for a coverage prompt. Reading the whole prompt would let the gold answer printed
 * in the coverage prompt stand in for facts that were never extracted.
 */
const wordJudge = (word: string): LongMemEvalCompletionClient => ({
  model: 'fake-judge',
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    const prompt = messages[0]!.content;
    const marker = prompt.includes('Stored facts:') ? 'Stored facts:' : 'Page:';
    const evidence = prompt.slice(prompt.indexOf(marker) + marker.length);
    return {
      content: evidence.includes(word) ? 'yes' : 'no',
      usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2, cost: null },
    };
  },
});

describe('samplePagesToExtract', () => {
  it('always includes every labelled evidence page', () => {
    const sampled = samplePagesToExtract(pages(100), [question('a', [7, 93])], 5);
    expect(sampled).toContain(7);
    expect(sampled).toContain(93);
  });

  it('spreads the extra pages through the document rather than taking the front', () => {
    const sampled = samplePagesToExtract(pages(100), [], 5);
    expect(sampled).toEqual([1, 21, 41, 61, 81]);
  });

  it('asks for nothing extra when the extra budget is zero', () => {
    expect(samplePagesToExtract(pages(50), [question('a', [3])], 0)).toEqual([3]);
  });

  it('drops an evidence page that is not in the document', () => {
    expect(samplePagesToExtract(pages(10), [question('a', [99])], 0)).toEqual([]);
  });
});

describe('evenSample', () => {
  it('returns everything when the limit is not binding', () => {
    expect(evenSample([1, 2, 3], 10)).toEqual([1, 2, 3]);
  });

  it('samples across the range instead of taking a prefix', () => {
    expect(evenSample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)).toEqual([1, 3, 5, 7, 9]);
  });

  it('returns nothing for a zero limit', () => {
    expect(evenSample([1, 2, 3], 0)).toEqual([1, 2, 3]);
  });
});

describe('prompts', () => {
  it('shows the fact and its page for a support judgement', () => {
    const prompt = buildFactSupportPrompt('approved(line_7).', 'page 7 states that budget line 7 was approved');
    expect(prompt).toContain('Stored fact: approved(line_7).');
    expect(prompt).toContain('page 7 states');
  });

  it('says plainly when a question had no facts to answer from', () => {
    expect(buildFactCoveragePrompt(question('a', [3]), [])).toContain('(none)');
  });
});

describe('measureDocumentExtraction', () => {
  it('measures precision over judged facts and recall over labelled questions', async () => {
    const result = await measureDocumentExtraction(
      pages(20),
      [question('q1', [4]), question('q2', [12])],
      {
        extract: async (page) => ({
          facts: [`approved(line_${page.page}).`, `mentioned(page_${page.page}).`],
        }),
        judge: wordJudge('approved'),
        samplePages: 2,
        judgeFacts: 4,
        concurrency: 2,
      },
    );
    expect(result.summary.pagesSampled).toBe(4);
    expect(result.summary.evidencePagesSampled).toBe(2);
    expect(result.summary.facts).toBe(8);
    expect(result.summary.factsPerPage).toBe(2);
    expect(result.summary.factsJudged).toBe(4);
    // half of every page's facts mention "approved", and the sample alternates between the two
    expect(result.summary.precision).toBeGreaterThan(0);
    expect(result.summary.questionsScored).toBe(2);
    expect(result.summary.recall).toBe(1);
  });

  it('counts a writer failure without losing the rest of the sample', async () => {
    const result = await measureDocumentExtraction(pages(10), [question('q1', [2])], {
      extract: async (page) => {
        if (page.page === 2) throw new Error('writer returned unparsable output');
        return { facts: [`approved(line_${page.page}).`] };
      },
      judge: wordJudge('approved'),
      samplePages: 2,
      judgeFacts: 10,
      concurrency: 1,
    });
    expect(result.summary.extractionErrors).toBe(1);
    expect(result.perPage.find((entry) => entry.page === 2)!.error).toMatch(/unparsable/);
    // the question's only evidence page failed, so its answer is not in the store
    expect(result.summary.recall).toBe(0);
    expect(result.summary.facts).toBeGreaterThan(0);
  });

  it('reports a page that yielded no facts separately from one that errored', async () => {
    const result = await measureDocumentExtraction(pages(6), [], {
      extract: async () => ({ facts: [] }),
      judge: wordJudge('approved'),
      samplePages: 3,
      judgeFacts: 5,
      concurrency: 1,
    });
    expect(result.summary.pagesWithNoFacts).toBe(3);
    expect(result.summary.extractionErrors).toBe(0);
    expect(result.summary.factsJudged).toBe(0);
    expect(result.summary.precision).toBe(0);
  });

  it('skips unanswerable questions, which have no evidence pages to extract from', async () => {
    const result = await measureDocumentExtraction(pages(8), [question('q1', [], null)], {
      extract: async (page) => ({ facts: [`approved(line_${page.page}).`] }),
      judge: wordJudge('approved'),
      samplePages: 2,
      judgeFacts: 2,
      concurrency: 1,
    });
    expect(result.summary.questionsScored).toBe(0);
    expect(result.summary.recall).toBe(0);
  });
});
