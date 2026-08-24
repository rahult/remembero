export interface ProductShipModelEvidence {
  model: string;
  role: 'default' | 'frontier' | 'economy';
  recallAccuracyPercent: number;
  recallTokens: number;
  recallCostUsd: number;
  recallDurationMs: number;
  extractionAccuracyPercent: number;
  extractionTokens: number;
  extractionCostUsd: number;
  extractionDurationMs: number;
}

export interface ProductShipEvidence {
  generatedAt: string;
  decision: 'ready_local_alpha_with_live_ocr_disabled';
  defaultModel: string;
  testFiles: number;
  passingTests: number;
  skippedTests: number;
  deterministicDocumentQuestions: number;
  deterministicDocumentAccuracyPercent: number;
  deterministicModelCalls: 0;
  deterministicTokens: 0;
  deterministicProviderCostUsd: 0;
  models: ProductShipModelEvidence[];
  liveOcrStatus: 'blocked_optional';
  boundary: string;
}

/** Frozen from the 21 August 2026 release matrix and product ship gate. */
export const PRODUCT_SHIP_EVIDENCE: ProductShipEvidence = {
  generatedAt: '2026-08-21T01:08:25.213+10:00',
  decision: 'ready_local_alpha_with_live_ocr_disabled',
  defaultModel: 'anthropic/claude-sonnet-5',
  testFiles: 55,
  passingTests: 747,
  skippedTests: 1,
  deterministicDocumentQuestions: 12,
  deterministicDocumentAccuracyPercent: 100,
  deterministicModelCalls: 0,
  deterministicTokens: 0,
  deterministicProviderCostUsd: 0,
  models: [
    {
      model: 'anthropic/claude-sonnet-5', role: 'default',
      recallAccuracyPercent: 100, recallTokens: 169067, recallCostUsd: 0.345182, recallDurationMs: 145440.32,
      extractionAccuracyPercent: 100, extractionTokens: 20703, extractionCostUsd: 0.045646, extractionDurationMs: 53959.498,
    },
    {
      model: 'google/gemini-3.1-pro-preview', role: 'frontier',
      recallAccuracyPercent: 100, recallTokens: 142250, recallCostUsd: 0.35604, recallDurationMs: 147129.66,
      extractionAccuracyPercent: 100, extractionTokens: 18811, extractionCostUsd: 0.077842, extractionDurationMs: 62662.878,
    },
    {
      model: 'openai/gpt-5.4', role: 'frontier',
      recallAccuracyPercent: 100, recallTokens: 110721, recallCostUsd: 0.22724, recallDurationMs: 74912.061,
      extractionAccuracyPercent: 93.3, extractionTokens: 13321, extractionCostUsd: 0.035703, extractionDurationMs: 22629.118,
    },
    {
      model: 'openai/gpt-5.6-luna', role: 'economy',
      recallAccuracyPercent: 96.2, recallTokens: 120454, recallCostUsd: 0.020846, recallDurationMs: 103216.549,
      extractionAccuracyPercent: 100, extractionTokens: 13507, extractionCostUsd: 0.003229, extractionDurationMs: 27044.601,
    },
  ],
  liveOcrStatus: 'blocked_optional',
  boundary:
    'Ready for a loopback local alpha. Live Unlimited-OCR quality remains disabled until a successful real-page provider run completes.',
};
