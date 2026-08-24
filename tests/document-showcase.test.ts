import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  askDocumentQuestion,
  documentFixtureIds,
  materializeDocumentCatalog,
  materializeDocumentShowcase,
  seedDocumentShowcase,
} from '../src/document/showcase.js';
import { evaluateDocumentShowcases } from '../src/evals/document-showcase.js';
import { MemoryStore } from '../src/store/store.js';

function documentStore(label: string): MemoryStore {
  return new MemoryStore(mkdtempSync(join(tmpdir(), `rembero-document-${label}-`)));
}

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

describe('real document showcase corpus', () => {
  it('materializes source-attributed real PDFs and rendered page hashes', () => {
    const { document, parse, questions, defaultQuestionId } = materializeDocumentShowcase();
    const catalog = materializeDocumentCatalog();

    expect(document).toMatchObject({
      id: 'irs-w9-english',
      fileName: 'irs-form-w9-en.pdf',
      parserMode: 'source_text_reviewed',
      source: {
        publisher: 'Internal Revenue Service',
        pdfPageCount: 6,
        sha256: '2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53',
      },
    });
    expect(document.pages[0]).toMatchObject({
      pageNumber: 1,
      imageUrl: '/documents/irs-form-w9-en-p1.png',
      imageSha256: '05ebe549fcd6805423e1757b45b9901bd3d14ccd5d80281c06ce249f05948dda',
    });
    expect(defaultQuestionId).toBe('w9-destination');
    expect(documentFixtureIds()).toEqual([
      'irs-w9-english',
      'irs-w9-spanish',
      'mathbridge-paper',
      'un-multilingualism',
    ]);
    expect(catalog.map((entry) => entry.id)).toEqual(documentFixtureIds());
    expect(questions.map((question) => question.id)).toEqual([
      'w9-destination',
      'w9-identity-match',
      'w9-deadline',
    ]);
    expect(parse).toMatchObject({
      status: 'ready',
      pageCount: 1,
      regionCount: 6,
      acceptedClaimCount: 4,
      proposedClaimCount: 1,
      acceptedClaimCoveragePercent: 100,
      pageCoveragePercent: 100,
    });
  });

  it('matches every advertised PDF and rendered-page hash to the checked-in bytes', () => {
    const files = new Map([
      ['irs-w9-english', ['irs-form-w9-en.pdf', 'irs-form-w9-en-p1.png']],
      ['irs-w9-spanish', ['irs-form-w9-es.pdf', 'irs-form-w9-es-p1.png']],
      ['mathbridge-paper', ['mathbridge-paper.pdf', 'mathbridge-paper-p4.png']],
      ['un-multilingualism', ['un-multilingualism.pdf', 'un-multilingualism-p20.png']],
    ]);

    for (const [documentId, [pdfName, imageName]] of files) {
      const { document } = materializeDocumentShowcase(documentId);
      const pdfPath = fileURLToPath(
        new URL(`../benchmarks/document-ocr/real/${pdfName}`, import.meta.url)
      );
      const imagePath = fileURLToPath(
        new URL(`../benchmarks/document-ocr/real/pages/${imageName}`, import.meta.url)
      );
      expect(sha256(pdfPath)).toBe(document.source.sha256);
      expect(sha256(imagePath)).toBe(document.pages[0]?.imageSha256);
    }
  });

  it('seeds accepted facts and reviewed rules idempotently per PDF namespace', () => {
    const store = documentStore('seed');

    expect(seedDocumentShowcase(store, 'irs-w9-english')).toMatchObject({
      seeded: true,
      added: 6,
      duplicates: 0,
    });
    expect(seedDocumentShowcase(store, 'irs-w9-english')).toMatchObject({
      seeded: false,
      added: 0,
      duplicates: 6,
    });
    expect(seedDocumentShowcase(store, 'mathbridge-paper')).toMatchObject({
      seeded: true,
      added: 7,
      duplicates: 0,
    });
  });

  it('proves the W-9 submission route from two accepted claims on one real region', () => {
    const store = documentStore('w9-route');
    seedDocumentShowcase(store, 'irs-w9-english');

    const result = askDocumentQuestion(store, 'w9-destination', 'irs-w9-english');

    expect(result).toMatchObject({
      status: 'answered',
      answer: 'Give the completed W-9 to the requester; do not send it to the IRS.',
      query: 'safe_submission(w9_en, Recipient)',
      bindings: [{ Recipient: 'requester' }],
      steps: expect.arrayContaining([
        expect.objectContaining({ clause: 'form_recipient(w9_en, requester).' }),
        expect.objectContaining({ clause: 'prohibited_recipient(w9_en, irs).' }),
        expect.objectContaining({
          clause: 'safe_submission(Form, Recipient) :- form_recipient(Form, Recipient), prohibited_recipient(Form, irs).',
        }),
      ]),
    });
    expect(result.sources.filter((source) => source.regionId === 'w9-en-region-destination')).toHaveLength(2);
  });

  it('derives MathBridge scale and context-limit comparisons from reviewed page 4 facts', () => {
    const store = documentStore('mathbridge');
    seedDocumentShowcase(store, 'mathbridge-paper');

    expect(askDocumentQuestion(store, 'mathbridge-scale', 'mathbridge-paper')).toMatchObject({
      status: 'answered',
      answer: 'Yes. The page reports approximately 23 million retained data points, above the reviewed 20 million threshold.',
      sources: expect.arrayContaining([
        expect.objectContaining({ regionId: 'mathbridge-region-retained' }),
      ]),
    });
    expect(
      askDocumentQuestion(store, 'mathbridge-context-limit', 'mathbridge-paper')
    ).toMatchObject({
      status: 'answered',
      sources: expect.arrayContaining([
        expect.objectContaining({ regionId: 'mathbridge-region-limits' }),
      ]),
    });
  });

  it('keeps a future UN commitment proposed and outside proof', () => {
    const store = documentStore('un-abstention');
    seedDocumentShowcase(store, 'un-multilingualism');

    const result = askDocumentQuestion(
      store,
      'un-future-commitment',
      'un-multilingualism'
    );

    expect(result).toMatchObject({
      status: 'unsupported',
      query: 'guaranteed_future_expansion(un, Year)',
      bindings: [],
      steps: [],
      sources: [],
    });
    expect(result.relatedEvidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'raw_region', regionId: 'un-region-future' }),
        expect.objectContaining({
          kind: 'proposed_claim',
          clause: 'future_coverage_expansion(un, likely).',
          badge: 'Proposed only',
        }),
      ])
    );
  });

  it('evaluates all four real PDFs with perfect deterministic scorecards', () => {
    const report = evaluateDocumentShowcases();

    expect(report.aggregate).toMatchObject({
      status: 'pass',
      documentCount: 4,
      questionCount: 12,
    });
    expect(report.documents).toHaveLength(4);
    expect(report.documents.every((document) => document.status === 'pass')).toBe(true);
    expect(report.aggregate.metrics).toMatchObject({
      parseCoverage: { percent: 100 },
      answerAccuracy: { percent: 100 },
      sourceRecall: { percent: 100 },
      proofGrounding: { percent: 100 },
      abstentionCorrectness: { percent: 100 },
      idempotency: { percent: 100 },
    });
  });
});
