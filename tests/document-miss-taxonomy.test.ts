import { describe, expect, it } from 'vitest';
import {
  buildTaxonomyPrompt,
  deterministicCause,
  parseShape,
  parseTaxonomyReply,
  type MissInput,
} from '../src/evals/document-miss-taxonomy.js';

const miss = (overrides: Partial<MissInput> = {}): MissInput => ({
  id: 'q',
  question: 'How many?',
  gold: '3',
  answer: '4',
  evidencePages: [2, 9],
  pageRecall: 1,
  evidenceHit: true,
  ...overrides,
});

describe('deterministicCause', () => {
  it('calls a miss with no gold page shown not_found', () => {
    expect(deterministicCause(miss({ evidenceHit: false, pageRecall: 0 }))).toBe('not_found');
  });

  it('calls a miss with some gold pages shown partly_found', () => {
    expect(deterministicCause(miss({ pageRecall: 0.5 }))).toBe('partly_found');
  });

  it('blames retrieval when the oracle run got it right with the same pages', () => {
    expect(deterministicCause(miss({ oracleCorrect: true }))).toBe('retrieval_order');
  });

  it('marks an answered unanswerable question before anything else', () => {
    expect(deterministicCause(miss({ gold: null, evidenceHit: false }))).toBe('answered_unanswerable');
  });

  it('leaves a full-evidence miss for the labeller', () => {
    expect(deterministicCause(miss({ oracleCorrect: false }))).toBeUndefined();
  });
});

describe('taxonomy prompt and reply', () => {
  it('shows the quotes and the answers', () => {
    const prompt = buildTaxonomyPrompt(miss(), ['There are three standards.']);
    expect(prompt).toContain('Gold answer: 3');
    expect(prompt).toContain('- There are three standards.');
  });

  it('parses a two-line reply', () => {
    expect(parseTaxonomyReply('cause: computation_error\nshape: count')).toEqual({
      cause: 'computation_error',
      shape: 'count',
    });
  });

  it('refuses an invented cause', () => {
    expect(() => parseTaxonomyReply('cause: bad_vibes\nshape: count')).toThrow(/cause/);
  });

  it('parses a one-word shape', () => {
    expect(parseShape('Set_difference.')).toBe('set_difference');
  });
});
