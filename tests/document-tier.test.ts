import { describe, expect, it } from 'vitest';
import { assembleTier, interleaveSources, type TierSource } from '../src/evals/document-tier.js';

const source = (
  id: string,
  pageCount: number,
  questions: TierSource['questions'] = [],
): TierSource => ({
  id,
  title: `Document ${id}`,
  sourceUrl: `https://example.invalid/${id}.pdf`,
  sha256: 'f'.repeat(64),
  pages: Array.from({ length: pageCount }, (_unused, index) => ({
    page: index + 1,
    text: `${id} page ${index + 1}`,
  })),
  questions,
});

describe('assembleTier', () => {
  it('lands on the exact page target and records each document offset', () => {
    const { tier, pages } = assembleTier('100p', 100, [source('a', 60), source('b', 60)]);
    expect(pages).toHaveLength(100);
    expect(tier.members.map((member) => [member.id, member.pageOffset, member.pages])).toEqual([
      ['a', 0, 60],
      ['b', 60, 40],
    ]);
    expect(pages[99]!.text).toBe('b page 40');
  });

  it('renumbers pages across the concatenation', () => {
    const { pages } = assembleTier('10p', 10, [source('a', 5), source('b', 5)]);
    expect(pages.map((page) => page.page)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(pages[5]!.text).toBe('b page 1');
  });

  it('shifts evidence pages by the offset of their document', () => {
    const { tier } = assembleTier('10p', 10, [
      source('a', 5, [{ id: 'q1', question: 'first?', answer: 'yes', evidencePages: [3] }]),
      source('b', 5, [{ id: 'q2', question: 'second?', answer: 'no', evidencePages: [1, 4] }]),
    ]);
    expect(tier.questions).toEqual([
      expect.objectContaining({ id: 'q1', evidencePages: [3], sourceDocument: 'a' }),
      expect.objectContaining({ id: 'q2', evidencePages: [6, 9], sourceDocument: 'b' }),
    ]);
  });

  it('drops a question whose evidence the truncation cut away, and says why', () => {
    const { tier, droppedQuestions } = assembleTier('6p', 6, [
      source('a', 5),
      source('b', 5, [
        { id: 'kept', question: 'in?', answer: 'yes', evidencePages: [1] },
        { id: 'cut', question: 'out?', answer: 'yes', evidencePages: [4] },
      ]),
    ]);
    expect(tier.questions.map((question) => question.id)).toEqual(['kept']);
    expect(droppedQuestions).toEqual([
      { id: 'cut', reason: 'evidence page 4 of b falls past the 1-page truncation' },
    ]);
  });

  it('keeps an unanswerable question, which has no evidence to cut', () => {
    const { tier } = assembleTier('4p', 4, [
      source('a', 4, [{ id: 'none', question: 'absent?', answer: null, evidencePages: [] }]),
    ]);
    expect(tier.questions[0]).toMatchObject({ id: 'none', answer: null, evidencePages: [] });
  });

  it('ignores sources beyond the target rather than overshooting', () => {
    const { tier } = assembleTier('5p', 5, [source('a', 5), source('b', 5)]);
    expect(tier.members.map((member) => member.id)).toEqual(['a']);
  });

  it('refuses a tier the sources cannot fill', () => {
    expect(() => assembleTier('500p', 500, [source('a', 10)])).toThrow(/supplied 10/);
  });

  it('refuses a non-positive target', () => {
    expect(() => assembleTier('0p', 0, [source('a', 10)])).toThrow(/positive page target/);
  });
});

describe('interleaveSources', () => {
  it('alternates labelled documents with filler so labels are not bunched at the front', () => {
    const ordered = interleaveSources([
      source('l1', 10, [{ id: 'q1', question: 'a?', answer: 'x', evidencePages: [1] }]),
      source('l2', 10, [{ id: 'q2', question: 'b?', answer: 'y', evidencePages: [1] }]),
      source('f1', 10),
      source('f2', 10),
      source('f3', 10),
    ]);
    expect(ordered.map((entry) => entry.id)).toEqual(['l1', 'f1', 'l2', 'f2', 'f3']);
  });
});
