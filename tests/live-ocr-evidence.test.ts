import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { LIVE_OCR_EVIDENCE } from '../src/document/live-ocr-evidence.js';

interface FrozenRealLiveReport {
  generatedAt: string;
  selectedDocumentIds: string[];
  documents: Array<{ documentId: string; kind: string; status: string; error?: string }>;
  aggregate: {
    documentCount: number;
    completedDocuments: number;
    errorDocuments: number;
    status: string;
  };
}

describe('frozen real-page OCR attempt', () => {
  it('reports provider quota as an operational block, never as a quality score', () => {
    const path = fileURLToPath(
      new URL(
        '../docs/research/results/unlimited-ocr-real-v1-summary.json',
        import.meta.url
      )
    );
    const report = JSON.parse(readFileSync(path, 'utf8')) as FrozenRealLiveReport;

    expect(report.aggregate).toMatchObject({
      status: 'fail',
      documentCount: 4,
      completedDocuments: 0,
      errorDocuments: 4,
    });
    expect(report.documents.every((document) => document.status === 'error')).toBe(true);
    expect(report.documents.every((document) => document.error?.includes('ZeroGPU quota'))).toBe(true);

    expect(LIVE_OCR_EVIDENCE).toMatchObject({
      generatedAt: report.generatedAt,
      status: 'blocked',
      documentCount: report.aggregate.documentCount,
      completedDocuments: report.aggregate.completedDocuments,
      errorDocuments: report.aggregate.errorDocuments,
    });
    expect(LIVE_OCR_EVIDENCE.documents.map((document) => document.documentId)).toEqual(
      report.selectedDocumentIds
    );
    expect(LIVE_OCR_EVIDENCE.operationalMessage).toMatch(/inference did not start/i);
    expect(LIVE_OCR_EVIDENCE.authorityBoundary).toMatch(/no real-page model quality score/i);
  });
});
