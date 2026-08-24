import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { DEFAULT_MODEL } from '../src/llm/client.js';
import { PRODUCT_SHIP_EVIDENCE } from '../src/document/product-ship-evidence.js';

interface FrozenRun {
  model: string;
  score: {
    cases: number;
    accuracy: number;
    unexpectedErrors?: number;
    errors?: number;
    totalTokens: number;
    costUsd: number;
    durationMs: number;
  };
}

function runs(file: string): FrozenRun[] {
  const path = fileURLToPath(new URL(`../docs/research/results/${file}`, import.meta.url));
  return (JSON.parse(readFileSync(path, 'utf8')) as { runs: FrozenRun[] }).runs;
}

describe('frozen frontier model release matrix', () => {
  it('selects the only tested model with perfect current recall and extraction', () => {
    const recall = runs('frontier-recall-v1-summary.json');
    const extraction = runs('frontier-extraction-v1-summary.json');
    const byModel = (items: FrozenRun[], model: string) =>
      items.find((run) => run.model === model)?.score;

    expect(recall.map((run) => run.model)).toEqual([
      'openai/gpt-5.4',
      'anthropic/claude-sonnet-5',
      'google/gemini-3.1-pro-preview',
    ]);
    for (const run of recall) {
      expect(run.score).toMatchObject({ cases: 26, accuracy: 1, errors: 0 });
      expect(run.score.totalTokens).toBeGreaterThan(0);
      expect(run.score.costUsd).toBeGreaterThan(0);
      expect(run.score.durationMs).toBeGreaterThan(0);
    }
    expect(byModel(extraction, 'anthropic/claude-sonnet-5')).toMatchObject({
      cases: 15,
      accuracy: 1,
      unexpectedErrors: 0,
    });
    expect(byModel(extraction, 'google/gemini-3.1-pro-preview')).toMatchObject({
      cases: 15,
      accuracy: 1,
      unexpectedErrors: 0,
    });
    expect(byModel(extraction, 'openai/gpt-5.4')).toMatchObject({
      cases: 15,
      accuracy: 14 / 15,
      unexpectedErrors: 0,
    });
    expect(DEFAULT_MODEL).toBe('anthropic/claude-sonnet-5');
  });

  it('keeps the economy model available but outside the strict default gate', () => {
    const lunaRecall = runs('economy-luna-recall-v1-summary.json')[0]!;
    const lunaExtraction = runs('economy-luna-extraction-v1-summary.json')[0]!;

    expect(lunaRecall).toMatchObject({
      model: 'openai/gpt-5.6-luna',
      score: { cases: 26, accuracy: 25 / 26, errors: 1 },
    });
    expect(lunaExtraction).toMatchObject({
      model: 'openai/gpt-5.6-luna',
      score: { cases: 15, accuracy: 1, unexpectedErrors: 0 },
    });
    expect(lunaRecall.score.costUsd).toBeLessThan(
      runs('frontier-recall-v1-summary.json').find(
        (run) => run.model === 'anthropic/claude-sonnet-5'
      )!.score.costUsd
    );
  });

  it('binds the ship decision to the measured matrix and optional OCR boundary', () => {
    const path = fileURLToPath(
      new URL('../docs/research/results/product-ship-v1-summary.json', import.meta.url)
    );
    const summary = JSON.parse(readFileSync(path, 'utf8')) as {
      decision: string;
      defaultModel: { model: string };
      frontierMatrix: Array<{
        model: string;
        recallAccuracyPercent: number;
        extractionAccuracyPercent: number;
      }>;
      liveOcr: { status: string };
      gates: Record<string, string>;
    };

    expect(summary).toMatchObject({
      decision: 'ready_local_alpha_with_live_ocr_disabled',
      defaultModel: { model: DEFAULT_MODEL },
      liveOcr: { status: 'blocked_optional' },
      gates: {
        documentProof: 'pass',
        memorgRoundTrip: 'pass',
        packagePolicy: 'pass_under_8m_no_pdfs_or_python_bytecode',
        liveUnlimitedOcr: 'blocked_optional',
      },
    });
    const selected = summary.frontierMatrix.find((entry) => entry.model === DEFAULT_MODEL);
    expect(selected).toMatchObject({
      recallAccuracyPercent: 100,
      extractionAccuracyPercent: 100,
    });
    expect(PRODUCT_SHIP_EVIDENCE).toMatchObject({
      decision: summary.decision,
      defaultModel: summary.defaultModel.model,
      deterministicDocumentAccuracyPercent: 100,
      deterministicTokens: 0,
      deterministicProviderCostUsd: 0,
      liveOcrStatus: summary.liveOcr.status,
    });
    expect(PRODUCT_SHIP_EVIDENCE.models).toHaveLength(summary.frontierMatrix.length);
  });
});
