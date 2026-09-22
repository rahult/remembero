import { describe, expect, it } from 'vitest';
import {
  anlsScore,
  extractNumber,
  normalizeAnswer,
  normalizedLevenshteinSimilarity,
  scoreAgainstXlDocBench,
  scoreWithLocaleTolerance,
  xlAccuracyScore,
  xlAnswerType,
  xlTokenF1,
  withDecimalCommaAsPoint,
} from '../src/evals/xl-docbench-score.js';

describe('xlAnswerType', () => {
  it('maps the dataset formats', () => {
    expect(xlAnswerType('Str')).toBe('entity');
    expect(xlAnswerType('Int')).toBe('numeric');
    expect(xlAnswerType('Float')).toBe('numeric');
    expect(xlAnswerType('None')).toBe('unanswerable');
    expect(xlAnswerType('Percentage')).toBe('percentage');
  });

  it('falls back to the verification rule for an unmapped format', () => {
    expect(xlAnswerType('Quantity', 'numeric_tolerance')).toBe('numeric');
    expect(xlAnswerType('Option', 'choice_exact_match')).toBe('single_choice');
  });

  it('defaults a missing format to an entity', () => {
    expect(xlAnswerType(undefined)).toBe('entity');
    expect(xlAnswerType('')).toBe('entity');
  });
});

describe('normalizeAnswer', () => {
  it('strips an answer preamble, punctuation and articles', () => {
    expect(normalizeAnswer('Answer: The European Commission,')).toBe('european commission');
    expect(normalizeAnswer('the answer is  4.2%  ')).toBe('4.2%');
  });

  it('keeps the characters a number needs', () => {
    expect(normalizeAnswer('-12.5%')).toBe('-12.5%');
  });
});

describe('extractNumber', () => {
  it('reads the first number, ignoring separators', () => {
    expect(extractNumber('1,234.5 tonnes')).toBe(1234.5);
    expect(extractNumber('about -3')).toBe(-3);
  });

  it('returns undefined when there is no number', () => {
    expect(extractNumber('several')).toBeUndefined();
  });
});

describe('xlAccuracyScore', () => {
  it('accepts a numeric answer inside five per cent', () => {
    expect(xlAccuracyScore('approximately 102 pages', '100', 'numeric')).toBe(1);
    expect(xlAccuracyScore('120 pages', '100', 'numeric')).toBe(0);
  });

  it('handles a zero gold exactly', () => {
    expect(xlAccuracyScore('0', '0', 'numeric')).toBe(1);
    expect(xlAccuracyScore('1', '0', 'numeric')).toBe(0);
  });

  it('accepts a gold string contained in a longer answer', () => {
    expect(xlAccuracyScore('The body responsible is the European Commission.', 'European Commission', 'entity')).toBe(1);
  });

  it('accepts a near-identical string on edit similarity', () => {
    expect(xlAccuracyScore('Europen Commission', 'European Commission', 'entity')).toBe(1);
    expect(xlAccuracyScore('World Bank', 'European Commission', 'entity')).toBe(0);
  });

  it('scores an unanswerable question on the declining phrases only', () => {
    expect(xlAccuracyScore('This cannot be determined from the document.', '', 'unanswerable')).toBe(1);
    expect(xlAccuracyScore('unanswerable', '', 'unanswerable')).toBe(1);
    expect(xlAccuracyScore('4.2 million', '', 'unanswerable')).toBe(0);
  });

  it('counts a context overflow as the correct answer to an unanswerable question', () => {
    expect(xlAccuracyScore('context_overflow', '', 'unanswerable')).toBe(1);
  });

  it('reads a boolean either way round', () => {
    expect(xlAccuracyScore('Yes, it does.', 'yes', 'boolean')).toBe(1);
    expect(xlAccuracyScore('No.', 'yes', 'boolean')).toBe(0);
    expect(xlAccuracyScore('perhaps', 'yes', 'boolean')).toBe(0);
  });

  it('compares a single choice by its letter', () => {
    expect(xlAccuracyScore('Option C is correct', 'C', 'single_choice')).toBe(1);
    expect(xlAccuracyScore('B', 'C', 'single_choice')).toBe(0);
  });
});

describe('xlTokenF1 and ANLS', () => {
  it('is one for an exact answer and zero for no overlap', () => {
    expect(xlTokenF1('European Commission', 'the European Commission')).toBe(1);
    expect(xlTokenF1('World Bank', 'European Commission')).toBe(0);
  });

  it('is partial for a partial overlap', () => {
    expect(xlTokenF1('European Council', 'European Commission')).toBeCloseTo(0.5);
  });

  it('treats an empty gold as answered only by an empty prediction', () => {
    expect(xlTokenF1('', '')).toBe(1);
    expect(xlTokenF1('something', '')).toBe(0);
  });

  it('zeroes an ANLS below the 0.5 threshold', () => {
    expect(anlsScore('abcd', 'abce')).toBeCloseTo(0.75);
    expect(anlsScore('abcd', 'wxyz')).toBe(0);
    expect(normalizedLevenshteinSimilarity('', '')).toBe(1);
  });
});

describe('scoreAgainstXlDocBench', () => {
  it('reports the three numbers their evaluator reports', () => {
    expect(scoreAgainstXlDocBench('Answer: European Commission', 'European Commission', 'entity')).toEqual({
      accuracy: 1,
      tokenF1: 1,
      anls: expect.closeTo(0.66, 1),
    });
  });
});

describe('withDecimalCommaAsPoint', () => {
  it('reads a decimal comma as a decimal point', () => {
    expect(withDecimalCommaAsPoint('63,1%')).toBe('63.1%');
    expect(withDecimalCommaAsPoint('ongeveer 89,9 procent')).toBe('ongeveer 89.9 procent');
  });

  it('leaves a thousands separator alone', () => {
    expect(withDecimalCommaAsPoint('1,234 cases')).toBe('1,234 cases');
  });
});

describe('scoreWithLocaleTolerance', () => {
  it('rescues a correct answer written with a decimal comma', () => {
    expect(scoreWithLocaleTolerance('dat is ongeveer 63,1%', '63.1%', 'entity')).toEqual({
      accuracy: 0,
      localeAccuracy: 1,
    });
  });

  it('does not rescue a wrong answer', () => {
    expect(scoreWithLocaleTolerance('dat is ongeveer 44,2%', '63.1%', 'entity')).toEqual({
      accuracy: 0,
      localeAccuracy: 0,
    });
  });

  it('leaves an already-correct answer untouched', () => {
    expect(scoreWithLocaleTolerance('63.1%', '63.1%', 'entity')).toEqual({
      accuracy: 1,
      localeAccuracy: 1,
    });
  });
});
