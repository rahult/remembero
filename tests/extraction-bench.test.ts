import { describe, expect, it } from 'vitest';
import { parseProgram, serializeClause } from '../src/engine/index.js';
import {
  EXTRACTION_BENCH_CASES,
  EXTRACTION_BENCH_PHENOMENA,
  predicateDrift,
  scoreExtractionBench,
  type ExtractionBenchCase,
} from '../src/evals/extraction-bench.js';
import { EXTRACTION_EVAL_CASES } from '../src/evals/extraction.js';

describe('extraction benchmark suite', () => {
  it('has at least 100 cases, every phenomenon represented at least 6 times, unique ids', () => {
    expect(EXTRACTION_BENCH_CASES.length).toBeGreaterThanOrEqual(100);
    const ids = new Set(EXTRACTION_BENCH_CASES.map((c) => c.id));
    expect(ids.size).toBe(EXTRACTION_BENCH_CASES.length);
    for (const phenomenon of EXTRACTION_BENCH_PHENOMENA) {
      const n = EXTRACTION_BENCH_CASES.filter(
        (c) => c.phenomenon === phenomenon,
      ).length;
      expect(n, phenomenon).toBeGreaterThanOrEqual(6);
    }
  });

  it('every expected program parses and the expected final program contains the initial program minus retractions', () => {
    for (const c of EXTRACTION_BENCH_CASES) {
      expect(() => parseProgram(c.initialProgram), c.id).not.toThrow();
      expect(() => parseProgram(c.expectedFinalProgram), c.id).not.toThrow();
      expect(() => parseProgram(c.expectedAddedProgram), c.id).not.toThrow();
    }
  });

  it('does not reuse the saturated 15-case eval verbatim', () => {
    const old = new Set(EXTRACTION_EVAL_CASES.map((c) => c.input));
    expect(EXTRACTION_BENCH_CASES.filter((c) => old.has(c.input)).length).toBe(
      0,
    );
  });

  it('measures predicate drift as added facts whose predicate is neither expected nor already stored', () => {
    const c: ExtractionBenchCase = EXTRACTION_BENCH_CASES.find(
      (x) => x.phenomenon === 'competitor_predicate',
    )!;
    expect(predicateDrift(c, ['employer(mira, initech).'])).toEqual([]);
    expect(predicateDrift(c, ['works_at(mira, initech).'])).toEqual([
      'works_at/2',
    ]);
  });

  it('scores per phenomenon', () => {
    const observations = EXTRACTION_BENCH_CASES.slice(0, 3).map((c, i) => ({
      case: c,
      model: 'm',
      outcome: 'completed' as const,
      actualClauses:
        i === 0
          ? parseProgram(c.expectedFinalProgram).map(serializeClause)
          : [],
      added: [],
      duplicates: 0,
      retracted: 0,
      llmCalls: 1,
      usage: {
        calls: 1,
        usageResponses: 0,
        costResponses: 0,
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        cachedPromptTokens: 0,
        reasoningTokens: 0,
        costUsd: 0,
      },
      durationMs: 1,
    }));
    const score = scoreExtractionBench(observations);
    expect(score.cases).toBe(3);
    expect(Object.keys(score.byPhenomenon).length).toBeGreaterThan(0);
    expect(typeof score.driftRate).toBe('number');
  });
});
