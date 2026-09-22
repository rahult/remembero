import { describe, expect, it } from 'vitest';
import {
  PAGE_BREAK,
  pageRangeFromId,
  pagesFromPageBreaks,
  windowHoldsPage,
  windowId,
  windowPages,
  windowsAsSessions,
} from '../src/evals/document-corpus.js';
import { scoreRetrieval, summariseTier, windowForPage, type QuestionOutcome } from '../src/evals/document-recall.js';

const pagesOf = (count: number, body = (page: number) => `page ${page} body`) =>
  Array.from({ length: count }, (_unused, index) => ({ page: index + 1, text: body(index + 1) }));

describe('pagesFromPageBreaks', () => {
  it('numbers pages from one and drops only a trailing empty part', () => {
    const pages = pagesFromPageBreaks(['a', 'b', 'c'].join(PAGE_BREAK) + PAGE_BREAK);
    expect(pages).toEqual([
      { page: 1, text: 'a' },
      { page: 2, text: 'b' },
      { page: 3, text: 'c' },
    ]);
  });

  it('keeps a blank page, because dropping it would shift every later page number', () => {
    const pages = pagesFromPageBreaks(['first', '   ', 'third'].join(PAGE_BREAK));
    expect(pages.map((page) => page.page)).toEqual([1, 2, 3]);
    expect(pages[2]!.text).toBe('third');
  });
});

describe('windowPages', () => {
  it('cuts fixed runs of pages and carries the range in the id', () => {
    const windows = windowPages(pagesOf(10), { pagesPerWindow: 4 });
    expect(windows.map((window) => window.id)).toEqual([
      'pages-0001-0004',
      'pages-0005-0008',
      'pages-0009-0010',
    ]);
    expect(windows[0]).toMatchObject({ firstPage: 1, lastPage: 4 });
  });

  it('never straddles a page across two windows, so a gold page maps to exactly one', () => {
    const windows = windowPages(pagesOf(30), { pagesPerWindow: 3 });
    for (let page = 1; page <= 30; page += 1) {
      expect(windows.filter((window) => windowHoldsPage(window, page))).toHaveLength(1);
    }
  });

  it('cuts a window short when a dense page would blow the byte budget', () => {
    const dense = pagesOf(6, () => 'x'.repeat(5_000));
    const windows = windowPages(dense, { pagesPerWindow: 4, maxBytes: 12 * 1024 });
    expect(windows.map((window) => window.id)).toEqual([
      'pages-0001-0002',
      'pages-0003-0004',
      'pages-0005-0006',
    ]);
  });

  it('keeps a page that is bigger than the budget on its own rather than dropping it', () => {
    const huge = pagesOf(3, () => 'x'.repeat(20_000));
    const windows = windowPages(huge, { pagesPerWindow: 4, maxBytes: 12 * 1024 });
    expect(windows.map((window) => window.id)).toEqual([
      'pages-0001-0001',
      'pages-0002-0002',
      'pages-0003-0003',
    ]);
  });

  it('labels each page inside a window so the reader can cite one', () => {
    const [window] = windowPages(pagesOf(2), { pagesPerWindow: 2 });
    expect(window!.text).toContain('[page 1]');
    expect(window!.text).toContain('[page 2]');
  });
});

describe('window ids', () => {
  it('round-trips a page range', () => {
    expect(pageRangeFromId(windowId(7, 11))).toEqual({ firstPage: 7, lastPage: 11 });
  });

  it('returns undefined for an id that is not a page window', () => {
    expect(pageRangeFromId('answer_7f3a')).toBeUndefined();
  });
});

describe('windowsAsSessions', () => {
  it('dates windows in page order so date ordering stays inert', () => {
    const sessions = windowsAsSessions(windowPages(pagesOf(9), { pagesPerWindow: 3 }));
    const dates = sessions.map((session) => Date.parse(session.date));
    expect(dates).toEqual([...dates].sort((a, b) => a - b));
    expect(sessions[0]!.turns).toHaveLength(1);
    expect(sessions[0]!.turns[0]!.role).toBe('user');
  });
});

describe('scoreRetrieval', () => {
  const windows = windowPages(pagesOf(40), { pagesPerWindow: 4 });

  it('counts a hit when a chosen window holds a gold page', () => {
    const score = scoreRetrieval([{ id: 'pages-0005-0008' }, { id: 'pages-0021-0024' }], [7]);
    expect(score.evidenceHit).toBe(true);
    expect(score.pageRecall).toBe(1);
    expect(score.windowPrecision).toBe(0.5);
    expect(score.pagesShown).toBe(8);
  });

  it('reports partial page recall when only some evidence pages are shown', () => {
    const score = scoreRetrieval([{ id: 'pages-0001-0004' }], [2, 90]);
    expect(score.pageRecall).toBe(0.5);
    expect(score.evidenceHit).toBe(true);
  });

  it('misses cleanly when no chosen window holds a gold page', () => {
    const score = scoreRetrieval([{ id: 'pages-0001-0004' }], [30]);
    expect(score).toMatchObject({ evidenceHit: false, pageRecall: 0, windowPrecision: 0 });
  });

  it('ignores ids that carry no page range', () => {
    expect(scoreRetrieval([{ id: 'not-a-window' }], [3]).windowsShown).toBe(0);
  });

  it('scores an unanswerable question as no hit rather than dividing by zero', () => {
    const score = scoreRetrieval([{ id: 'pages-0001-0004' }], []);
    expect(score).toMatchObject({ evidenceHit: false, pageRecall: 0, windowPrecision: 0, pagesShown: 4 });
  });

  it('finds the window a gold page belongs to', () => {
    expect(windowForPage(windows, 13)!.id).toBe('pages-0013-0016');
    expect(windowForPage(windows, 999)).toBeUndefined();
  });
});

describe('summariseTier', () => {
  const question = (id: string, answer: string | null, evidencePages: number[]) => ({
    id,
    question: `q ${id}`,
    answer,
    evidencePages,
    sourceDocument: 'doc-a',
  });

  const outcome = (
    id: string,
    answer: string | null,
    evidencePages: number[],
    overrides: Partial<QuestionOutcome> = {},
  ): QuestionOutcome => ({
    question: question(id, answer, evidencePages),
    retrieval: scoreRetrieval([{ id: 'pages-0001-0004' }], evidencePages),
    answer: 'said something',
    correct: true,
    correctLocale: true,
    abstained: false,
    latencyMs: 1_000,
    contextBytes: 20_000,
    ...overrides,
  });

  it('separates accuracy over all questions from accuracy over answerable ones', () => {
    const tier = { tier: '100p', pages: 100, members: [], questions: [] };
    const summary = summariseTier(tier, 25, [
      outcome('a', 'yes', [2]),
      outcome('b', 'no', [3], { correct: false }),
      outcome('c', null, [], { correct: true, abstained: true }),
    ]);
    expect(summary.questions).toBe(3);
    expect(summary.answerable).toBe(2);
    expect(summary.unanswerable).toBe(1);
    expect(summary.accuracy).toBeCloseTo(2 / 3);
    expect(summary.answerableAccuracy).toBe(0.5);
    expect(summary.evidenceHitRate).toBe(1);
    expect(summary.falseAnswerRate).toBe(0);
  });

  it('counts an answer to an unanswerable question as a false answer', () => {
    const tier = { tier: '500p', pages: 500, members: [], questions: [] };
    const summary = summariseTier(tier, 125, [
      outcome('a', null, [], { correct: false, abstained: false }),
      outcome('b', null, [], { correct: true, abstained: true }),
    ]);
    expect(summary.falseAnswerRate).toBe(0.5);
    expect(summary.answerableAccuracy).toBe(0);
  });
});
