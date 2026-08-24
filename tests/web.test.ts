import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MemoryStore } from '../src/store/store.js';
import { RemberoWebService, WebServiceError } from '../src/web/service.js';
import { startWebServer } from '../src/web/server.js';

function webService(label: string): RemberoWebService {
  return new RemberoWebService({
    store: new MemoryStore(mkdtempSync(join(tmpdir(), `rembero-web-${label}-`))),
    llmConfigured: false,
  });
}

describe('Remembero web use-case service', () => {
  it('seeds one sourced personal workspace idempotently', () => {
    const service = webService('seed');

    expect(service.seedDemo()).toEqual({ seeded: true, added: 15 });
    expect(service.seedDemo()).toEqual({ seeded: false, added: 0 });
    expect(service.bootstrap()).toMatchObject({
      namespace: 'personal',
      llmConfigured: false,
      empty: false,
      counts: {
        facts: 12,
        rules: 3,
        constraints: 0,
        sourcedPercent: 100,
      },
      health: {
        status: 'healthy',
        clauseCount: 15,
        provenance: { sourceCoveragePercent: 100 },
      },
    });
  });

  it('answers a guided personal question with real rule proof and sources', async () => {
    const service = webService('ask');
    service.seedDemo();

    const result = await service.ask({
      question: 'Who is collaborating on Atlas?',
      presetId: 'collaborators',
    });

    expect(result).toMatchObject({
      mode: 'guided-local',
      status: 'answered',
      query: 'collaborator(Person, atlas)',
      answer: 'Maya is collaborating on Atlas.',
      bindings: [{ Person: 'maya' }],
      evidence: {
        claims: expect.arrayContaining([
          'project_owner(atlas, rahul)',
          'project_contributor(atlas, maya)',
        ]),
        rules: [
          expect.objectContaining({
            clause:
              'collaborator(Person, Project) :- project_owner(Project, Owner), project_contributor(Project, Person), Owner != Person.',
          }),
        ],
        sources: expect.arrayContaining([
          expect.objectContaining({ opId: 'web-demo-atlas-session-v1' }),
        ]),
      },
      explanation: { rows: [{ bindings: { Person: 'maya' } }] },
    });
  });

  it('returns the document showcase with normalized parse metadata and anchored proof', () => {
    const service = webService('document');
    service.parseDocument();

    const result = service.documentShowcase();

    expect(result).toMatchObject({
      documents: expect.arrayContaining([
        expect.objectContaining({ id: 'irs-w9-english' }),
        expect.objectContaining({ id: 'irs-w9-spanish' }),
        expect.objectContaining({ id: 'mathbridge-paper' }),
        expect.objectContaining({ id: 'un-multilingualism' }),
      ]),
      parse: {
        status: 'ready',
        pageCount: 1,
        acceptedClaimCount: 4,
        proposedClaimCount: 1,
      },
      defaultQuestionId: 'w9-destination',
      questions: [
        expect.objectContaining({ id: 'w9-destination' }),
        expect.objectContaining({ id: 'w9-identity-match' }),
        expect.objectContaining({ id: 'w9-deadline' }),
      ],
      document: {
        fileName: 'irs-form-w9-en.pdf',
        parserMode: 'source_text_reviewed',
        source: {
          publisher: 'Internal Revenue Service',
          sha256: '2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53',
        },
      },
      evaluation: {
        status: 'pass',
        metrics: {
          proofGrounding: { percent: 100 },
        },
      },
      proof: {
        questionId: 'w9-destination',
        status: 'answered',
        answer: 'Give the completed W-9 to the requester; do not send it to the IRS.',
      },
      liveOcrEvidence: {
        status: 'blocked',
        documentCount: 4,
        completedDocuments: 0,
        errorDocuments: 4,
      },
      memorgExport: {
        format: 'remembero-memorg-import',
        version: 1,
        targetVersion: '0.1.2',
        itemCount: 66,
        downloadUrl: '/documents/document-intelligence.memorg.json',
      },
      shipEvidence: {
        decision: 'ready_local_alpha_with_live_ocr_disabled',
        defaultModel: 'anthropic/claude-sonnet-5',
        testFiles: 55,
        passingTests: 747,
        deterministicDocumentAccuracyPercent: 100,
        deterministicTokens: 0,
        deterministicProviderCostUsd: 0,
        liveOcrStatus: 'blocked_optional',
        models: expect.arrayContaining([
          expect.objectContaining({
            model: 'anthropic/claude-sonnet-5',
            role: 'default',
            recallAccuracyPercent: 100,
            extractionAccuracyPercent: 100,
          }),
          expect.objectContaining({
            model: 'openai/gpt-5.6-luna',
            role: 'economy',
          }),
        ]),
      },
    });
    expect(result.proof.sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'accepted_fact',
          regionId: 'w9-en-region-destination',
        }),
        expect.objectContaining({
          kind: 'accepted_fact',
          regionId: 'w9-en-region-destination',
        }),
      ])
    );
  });

  it('keeps document showcase reads mutation-free before explicit parsing', () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-web-document-read-')));
    const service = new RemberoWebService({ store, llmConfigured: false });

    const result = service.documentShowcase();

    expect(store.load('documents')).toHaveLength(0);
    expect(result).toMatchObject({
      parse: { seededCount: 0, duplicateCount: 0 },
      proof: { status: 'unsupported', steps: [], sources: [] },
    });
  });

  it('returns parse reruns idempotently and keeps unsupported document evidence honest', () => {
    const service = webService('document-boundary');

    const firstParse = service.parseDocument();
    expect(firstParse).toMatchObject({
      documents: expect.arrayContaining([
        expect.objectContaining({ id: 'irs-w9-english' }),
      ]),
      parse: {
        seededCount: 6,
        duplicateCount: 0,
      },
      document: {
        fileName: 'irs-form-w9-en.pdf',
      },
    });
    const secondParse = service.parseDocument();
    expect(secondParse).toMatchObject({
      parse: {
        seededCount: 0,
        duplicateCount: 6,
      },
    });

    expect(() => service.askDocument({ questionId: 'unknown' })).toThrow(
      /allowlisted/i
    );
    expect(
      service.askDocument({
        documentId: 'un-multilingualism',
        questionId: 'un-future-commitment',
      })
    ).toMatchObject({
      status: 'unsupported',
      relatedEvidence: expect.arrayContaining([
        expect.objectContaining({ kind: 'raw_region', regionId: 'un-region-future' }),
      ]),
    });
  });

  it('switches document snapshots by allowlisted document id without mutating unrelated namespaces', () => {
    const service = webService('document-switch');

    const before = service.documentShowcase({ documentId: 'mathbridge-paper' });
    expect(before.proof.status).toBe('unsupported');

    const parsed = service.parseDocument({ documentId: 'mathbridge-paper' });
    expect(parsed).toMatchObject({
      document: { id: 'mathbridge-paper' },
      proof: {
        questionId: 'mathbridge-scale',
        answer: 'Yes. The page reports approximately 23 million retained data points, above the reviewed 20 million threshold.',
      },
    });

    const untouched = service.documentShowcase({ documentId: 'un-multilingualism' });
    expect(untouched.proof.status).toBe('unsupported');
  });

  it('keeps a failed guided question separate from related discovery', async () => {
    const service = webService('related');
    service.seedDemo();

    const result = await service.ask({
      question: 'What gift does Maya want?',
      presetId: 'gift',
    });

    expect(result).toMatchObject({
      mode: 'guided-local',
      status: 'no_match',
      query: 'prefers_gift(maya, Gift)',
      bindings: [],
      whyNot: { status: 'blocked' },
      relatedKnowledge: { status: 'matches' },
    });
    expect(result.answer).toMatch(/don't have a supported answer/i);
    expect(result.relatedKnowledge.results.length).toBeGreaterThan(0);
  });

  it('searches source phrases and browses the explicit Maya neighborhood', () => {
    const service = webService('discovery');
    service.seedDemo();

    const search = service.search({ text: 'vendor security review' });
    expect(search).toMatchObject({ status: 'matches' });
    expect(search.results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sources: expect.arrayContaining([
            expect.objectContaining({
              text: expect.stringContaining('vendor security review'),
            }),
          ]),
          reasons: expect.arrayContaining([
            expect.objectContaining({ kind: 'source_phrase' }),
          ]),
        }),
      ])
    );

    const graph = service.graph({ focus: 'maya' });
    expect(graph.selection.selectedClaims).toBeGreaterThan(0);
    expect(graph.graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'entity', value: 'maya' }),
        expect.objectContaining({ kind: 'claim', predicate: 'project_contributor' }),
      ])
    );
  });

  it('stores a structured fact with durable source evidence', () => {
    const service = webService('capture');
    service.seedDemo();

    expect(
      service.addMemory({
        subject: 'maya',
        predicate: 'prefers_channel',
        object: 'signal',
        sourceText: 'Maya asked me to use Signal for project updates.',
      })
    ).toMatchObject({
      status: 'saved',
      clause: 'prefers_channel(maya, signal).',
      added: 1,
      duplicate: false,
    });
    expect(service.search({ text: 'Signal project updates' }).results[0]).toMatchObject({
      clause: 'prefers_channel(maya, signal).',
      sources: [
        expect.objectContaining({
          text: 'Maya asked me to use Signal for project updates.',
        }),
      ],
    });
  });

  it('requires a configured model for arbitrary questions but keeps guided recall local', async () => {
    const service = webService('model-boundary');
    service.seedDemo();

    await expect(
      service.ask({ question: 'Tell me something surprising.' })
    ).rejects.toMatchObject<WebServiceError>({
      code: 'model_not_configured',
      status: 400,
    });
    await expect(
      service.ask({ question: 'Who owns Atlas?', presetId: 'owner' })
    ).resolves.toMatchObject({ status: 'answered', answer: 'Rahul owns Atlas.' });
  });

  it('serves the real same-origin JSON workflow over loopback HTTP', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-web-http-'));
    const running = await startWebServer({ root, port: 0, seedDemo: true });
    try {
      const bootstrap = await fetch(`${running.url}/api/bootstrap`).then((response) =>
        response.json()
      );
      expect(bootstrap).toMatchObject({
        namespace: 'personal',
        counts: { facts: 12, rules: 3, sourcedPercent: 100 },
      });

      const answerResponse = await fetch(`${running.url}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: running.url,
        },
        body: JSON.stringify({
          question: 'Who is collaborating on Atlas?',
          presetId: 'collaborators',
        }),
      });
      expect(answerResponse.status).toBe(200);
      await expect(answerResponse.json()).resolves.toMatchObject({
        status: 'answered',
        answer: 'Maya is collaborating on Atlas.',
      });

      const documentResponse = await fetch(`${running.url}/api/document`).then((response) =>
        response.json()
      );
      expect(documentResponse).toMatchObject({
        documents: expect.arrayContaining([
          expect.objectContaining({ id: 'irs-w9-english' }),
          expect.objectContaining({ id: 'irs-w9-spanish' }),
          expect.objectContaining({ id: 'mathbridge-paper' }),
          expect.objectContaining({ id: 'un-multilingualism' }),
        ]),
        parse: { pageCount: 1, acceptedClaimCount: 4 },
        defaultQuestionId: 'w9-destination',
        document: { fileName: 'irs-form-w9-en.pdf' },
        proof: { status: 'answered' },
      });

      const parseResponse = await fetch(`${running.url}/api/document?documentId=mathbridge-paper`).then((response) =>
        response.json()
      );
      expect(parseResponse).toMatchObject({
        document: { id: 'mathbridge-paper' },
        proof: {
          questionId: 'mathbridge-scale',
          answer: 'Yes. The page reports approximately 23 million retained data points, above the reviewed 20 million threshold.',
        },
      });

      const memorgResponse = await fetch(`${running.url}/api/document/memorg`);
      expect(memorgResponse.status).toBe(200);
      await expect(memorgResponse.json()).resolves.toMatchObject({
        format: 'remembero-memorg-import',
        version: 1,
        target: {
          package: 'memorg',
          version: '0.1.2',
          method: 'MemorgSystem.create_memory_item',
        },
        items: expect.arrayContaining([
          expect.objectContaining({ key: 'remembero-document-intelligence' }),
          expect.objectContaining({ key: 'document-mathbridge-paper', item_type: 'document' }),
        ]),
      });

      const reparseResponse = await fetch(`${running.url}/api/document/parse`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: running.url,
        },
        body: JSON.stringify({ documentId: 'un-multilingualism' }),
      });
      expect(reparseResponse.status).toBe(200);
      await expect(reparseResponse.json()).resolves.toMatchObject({
        parse: expect.objectContaining({ status: 'ready' }),
        document: {
          id: 'un-multilingualism',
          parserMode: 'source_text_reviewed',
        },
      });

      const documentAsk = await fetch(`${running.url}/api/document/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: running.url,
        },
        body: JSON.stringify({
          documentId: 'un-multilingualism',
          questionId: 'un-future-commitment',
        }),
      });
      expect(documentAsk.status).toBe(200);
      await expect(documentAsk.json()).resolves.toMatchObject({
        status: 'unsupported',
        relatedEvidence: expect.arrayContaining([
          expect.objectContaining({ kind: 'raw_region', regionId: 'un-region-future' }),
        ]),
      });

      const invalidQuestion = await fetch(`${running.url}/api/document/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: running.url,
        },
        body: JSON.stringify({ questionId: 'unknown' }),
      });
      expect(invalidQuestion.status).toBe(400);
      await expect(invalidQuestion.json()).resolves.toMatchObject({
        error: 'invalid_document_question',
      });

      const invalidDocumentRead = await fetch(
        `${running.url}/api/document?documentId=not-allowlisted`
      );
      expect(invalidDocumentRead.status).toBe(400);
      await expect(invalidDocumentRead.json()).resolves.toMatchObject({
        error: 'invalid_document_id',
      });

      for (const path of ['/api/document/parse', '/api/document/ask']) {
        const invalidDocumentPost = await fetch(`${running.url}${path}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Origin: running.url,
          },
          body: JSON.stringify({
            documentId: 'not-allowlisted',
            ...(path.endsWith('/ask') ? { questionId: 'collaborators' } : {}),
          }),
        });
        expect(invalidDocumentPost.status).toBe(400);
        await expect(invalidDocumentPost.json()).resolves.toMatchObject({
          error: 'invalid_document_id',
        });
      }

      const rejected = await fetch(`${running.url}/api/seed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Origin: 'https://example.invalid',
        },
        body: '{}',
      });
      expect(rejected.status).toBe(403);
      await expect(rejected.json()).resolves.toMatchObject({
        error: 'origin_rejected',
      });
    } finally {
      await running.close();
    }
  });

  it('refuses every non-loopback bind because the web console has no authentication', async () => {
    await expect(
      startWebServer({ host: '0.0.0.0', port: 0, seedDemo: false })
    ).rejects.toThrow(/loopback hosts only/i);
  });
});
