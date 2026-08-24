export interface LiveOcrEvidenceMetric {
  passed: number;
  total: number;
  percent: number;
}

export interface LiveOcrDocumentEvidence {
  documentId: string;
  kind: string;
  status: 'error';
  errorCategory: 'provider_quota';
}

export interface LiveOcrEvidenceSnapshot {
  generatedAt: string;
  source: 'official_baidu_zerogpu';
  model: 'baidu/Unlimited-OCR';
  mode: 'gundam';
  status: 'blocked';
  documentCount: number;
  completedDocuments: number;
  errorDocuments: number;
  operationalMessage: string;
  requiredFieldRecall: LiveOcrEvidenceMetric;
  readingOrderRecall: LiveOcrEvidenceMetric;
  readingOrderOrder: LiveOcrEvidenceMetric;
  groundingCoordinateCoverage: LiveOcrEvidenceMetric;
  tableDetection: LiveOcrEvidenceMetric;
  totalLatencyMs: number;
  averageDocumentLatencyMs: number;
  maximumDocumentLatencyMs: number;
  documents: LiveOcrDocumentEvidence[];
  authorityBoundary: string;
}

/** Frozen from docs/research/results/unlimited-ocr-real-v1-summary.json. */
export const LIVE_OCR_EVIDENCE: LiveOcrEvidenceSnapshot = {
  generatedAt: '2026-08-20T13:59:22.107Z',
  source: 'official_baidu_zerogpu',
  model: 'baidu/Unlimited-OCR',
  mode: 'gundam',
  status: 'blocked',
  documentCount: 4,
  completedDocuments: 0,
  errorDocuments: 4,
  operationalMessage:
    'Authenticated provider request reached the official Space, but inference did not start: ZeroGPU quota reported 90 seconds requested and 0 seconds remaining.',
  requiredFieldRecall: { passed: 0, total: 0, percent: 0 },
  readingOrderRecall: { passed: 0, total: 0, percent: 0 },
  readingOrderOrder: { passed: 0, total: 0, percent: 0 },
  groundingCoordinateCoverage: { passed: 0, total: 0, percent: 0 },
  tableDetection: { passed: 0, total: 0, percent: 0 },
  totalLatencyMs: 0,
  averageDocumentLatencyMs: 0,
  maximumDocumentLatencyMs: 0,
  documents: [
    { documentId: 'irs-w9-english-p1', kind: 'government_form', status: 'error', errorCategory: 'provider_quota' },
    { documentId: 'irs-w9-spanish-p1', kind: 'multilingual_government_form', status: 'error', errorCategory: 'provider_quota' },
    { documentId: 'mathbridge-paper-p4', kind: 'technical_research_paper', status: 'error', errorCategory: 'provider_quota' },
    { documentId: 'un-multilingualism-p20', kind: 'visual_publication', status: 'error', errorCategory: 'provider_quota' },
  ],
  authorityBoundary:
    'No real-page model quality score is claimed. The deterministic showcase uses downloaded PDFs plus reviewed source text until a quota-backed live run completes.',
};
