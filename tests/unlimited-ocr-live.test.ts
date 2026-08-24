import { describe, expect, it, vi } from 'vitest';
import { fileURLToPath } from 'node:url';

import {
  formatUnlimitedOcrLiveHelp,
  UNLIMITED_OCR_LIVE_SAVE_MARKER,
  UnlimitedOcrLiveClient,
  evaluateUnlimitedOcrLive,
  loadUnlimitedOcrLiveCorpus,
  parseUnlimitedOcrLiveCliArgs,
} from '../src/evals/unlimited-ocr-live.js';

function fixturePath(name: string): string {
  return fileURLToPath(new URL(`../benchmarks/document-ocr/real/pages/${name}`, import.meta.url));
}

function sseResponse(frames: string[]): Response {
  return new Response(frames.join('\n\n'), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function crlfSseResponse(frames: string[]): Response {
  return new Response(frames.map((frame) => frame.replace(/\n/g, '\r\n')).join('\r\n\r\n'), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function rawGroundedTextFromReference(referenceText: string): string {
  return (
    referenceText
      .split('\n')
      .map(
        (line, index) =>
          `<|det|>text [0, ${index}, 100, ${index + 1}]<|/det|>${line}`
      )
      .join('\n') +
    `\n${UNLIMITED_OCR_LIVE_SAVE_MARKER}\n`
  );
}

function successfulOutput(document: ReturnType<typeof loadUnlimitedOcrLiveCorpus>['documents'][number]): string {
  return [
    ...document.expectedReadingOrder,
    ...document.requiredFields,
    ...(document.expectedTable ? ['column one | column two | column three'] : []),
  ].join('\n');
}

describe('UnlimitedOcrLiveClient', () => {
  it('uploads, starts, and consumes the Gradio SSE contract', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/gradio_api/upload')) {
        expect(init?.method).toBe('POST');
        expect(init?.headers).toEqual({ Authorization: 'Bearer hf_test_token' });
        return new Response(JSON.stringify(['/tmp/gradio/uploaded/irs-form-w9-en-p1.png']), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/gradio_api/call/v2/run_ocr')) {
        expect(init?.method).toBe('POST');
        expect(init?.headers).toMatchObject({
          Authorization: 'Bearer hf_test_token',
          'Content-Type': 'application/json',
        });
        const body = JSON.parse(String(init?.body));
        expect(body).toMatchObject({
          mode: 'gundam',
          prompt: 'document parsing.',
          image_path: {
            path: '/tmp/gradio/uploaded/irs-form-w9-en-p1.png',
            orig_name: 'irs-form-w9-en-p1.png',
            mime_type: 'image/png',
            meta: { _type: 'gradio.FileData' },
          },
        });
        return new Response(JSON.stringify({ event_id: 'evt-contract' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      expect(url.endsWith('/gradio_api/call/run_ocr/evt-contract')).toBe(true);
      expect(init?.headers).toMatchObject({
        Authorization: 'Bearer hf_test_token',
        Accept: 'text/event-stream',
      });
      return crlfSseResponse([
        'event: generating\ndata: [{"text":"<|det|>title [1, 2, 3, 4]<|/det|>Form W-9\\n<|det|>text [2, 3, 4, 5]<|/det|>Give form to the requester\\n===============save results:===============\\n","done":false}]',
        'event: generating\ndata: [{"text":"Form W-9\\nGive form to the requester","done":true}]',
        'event: complete\ndata: [{"text":"Form W-9\\nGive form to the requester","done":true}]',
      ]);
    });

    const client = new UnlimitedOcrLiveClient(
      {
        spaceUrl: 'https://example.hf.space',
        allowedSpaceHosts: ['example.hf.space'],
        mode: 'gundam',
        prompt: 'document parsing.',
        hfToken: 'hf_test_token',
      },
      fetchFn as typeof fetch
    );
    const result = await client.parseImage(fixturePath('irs-form-w9-en-p1.png'));

    expect(fetchFn).toHaveBeenCalledTimes(3);
    expect(result.rawGroundedText).toContain('<|det|>title [1, 2, 3, 4]<|/det|>Form W-9');
    expect(result.rawGroundedText).not.toContain(UNLIMITED_OCR_LIVE_SAVE_MARKER);
    expect(result.finalText).toBe('Form W-9\nGive form to the requester');
    expect(result.markdown).toBe('Form W-9\nGive form to the requester');
    expect(result.blocks).toEqual([
      { category: 'title', coordinates: [1, 2, 3, 4], text: 'Form W-9' },
      { category: 'text', coordinates: [2, 3, 4, 5], text: 'Give form to the requester' },
    ]);
  });

  it('allows only explicit HTTPS Space hosts without embedded credentials', () => {
    expect(
      () => new UnlimitedOcrLiveClient({ spaceUrl: 'http://baidu-unlimited-ocr.hf.space' })
    ).toThrow(/must use https/i);
    expect(
      () =>
        new UnlimitedOcrLiveClient({
          spaceUrl: 'https://token@baidu-unlimited-ocr.hf.space',
        })
    ).toThrow(/must not contain credentials/i);
    expect(
      () => new UnlimitedOcrLiveClient({ spaceUrl: 'https://attacker.example' })
    ).toThrow(/not allowlisted/i);
  });
});

describe('evaluateUnlimitedOcrLive', () => {
  it('scores the full fixture corpus and aggregates gates', async () => {
    const corpus = loadUnlimitedOcrLiveCorpus();
    const eventIds = corpus.documents.map((document) => `evt-${document.id}`);
    const finalByEventId = new Map(
      corpus.documents.map((document, index) => [eventIds[index], successfulOutput(document)])
    );
    const rawByEventId = new Map(
      corpus.documents.map((document, index) => [
        eventIds[index],
        rawGroundedTextFromReference(successfulOutput(document)),
      ])
    );
    let uploadIndex = 0;
    let runIndex = 0;

    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/gradio_api/upload')) {
        const document = corpus.documents[uploadIndex];
        uploadIndex += 1;
        return new Response(JSON.stringify([`/tmp/gradio/${document.id}.png`]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/gradio_api/call/v2/run_ocr')) {
        const eventId = eventIds[runIndex];
        runIndex += 1;
        return new Response(JSON.stringify({ event_id: eventId }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      const eventId = url.split('/').at(-1) ?? '';
      const rawText = rawByEventId.get(eventId) ?? '';
      const finalText = finalByEventId.get(eventId) ?? '';
      return sseResponse([
        `event: generating\ndata: ${JSON.stringify([{ text: rawText, done: false }])}`,
        `event: generating\ndata: ${JSON.stringify([{ text: finalText, done: true }])}`,
        `event: complete\ndata: ${JSON.stringify([{ text: finalText, done: true }])}`,
      ]);
    });

    const report = await evaluateUnlimitedOcrLive({
      spaceUrl: 'https://example.hf.space',
      allowedSpaceHosts: ['example.hf.space'],
      fetchFn: fetchFn as typeof fetch,
    });

    expect(report.spaceUrl).toBe('https://example.hf.space');
    expect(report.mode).toBe('gundam');
    expect(report.aggregate.status).toBe('pass');
    expect(report.selectedDocumentCount).toBe(4);
    expect(report.availableDocumentCount).toBe(4);
    expect(report.selectedDocumentIds).toEqual([
      'irs-w9-english-p1',
      'irs-w9-spanish-p1',
      'mathbridge-paper-p4',
      'un-multilingualism-p20',
    ]);
    expect(report.aggregate.documentCount).toBe(4);
    expect(report.aggregate.errorDocuments).toBe(0);
    expect(report.aggregate.checks.noErrors).toBe(true);
    expect(report.aggregate.requiredFieldRecall.percent).toBe(100);
    expect(report.aggregate.readingOrderRecall.percent).toBe(100);
    expect(report.aggregate.readingOrderOrder.percent).toBe(100);
    expect(report.aggregate.groundingCoordinateCoverage.percent).toBe(100);
    expect(report.aggregate.tableDetection.percent).toBe(100);
    expect(report.aggregate.normalizedSimilarity).toBeUndefined();
    expect(report.documents.every((document) => document.status === 'pass')).toBe(true);
    expect(report.documents.find((document) => document.documentId === 'mathbridge-paper-p4')?.tableDetection.actual).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(corpus.documents.length * 3);
  });

  it('filters to selected documents and validates requested ids', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/gradio_api/upload')) {
        return new Response(JSON.stringify(['/tmp/gradio/irs-form-w9-en-p1.png']), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/gradio_api/call/v2/run_ocr')) {
        return new Response(JSON.stringify({ event_id: 'evt-irs-w9-english-p1' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      const corpus = loadUnlimitedOcrLiveCorpus();
      const document = corpus.documents.find((entry) => entry.id === 'irs-w9-english-p1')!;
      const text = successfulOutput(document);
      return sseResponse([
        `event: generating\ndata: ${JSON.stringify([{ text: rawGroundedTextFromReference(text), done: false }])}`,
        `event: complete\ndata: ${JSON.stringify([{ text, done: true }])}`,
      ]);
    });

    const filtered = await evaluateUnlimitedOcrLive({
      spaceUrl: 'https://example.hf.space',
      allowedSpaceHosts: ['example.hf.space'],
      documentIds: ['irs-w9-english-p1'],
      fetchFn: fetchFn as typeof fetch,
    });
    expect(filtered.selectedDocumentCount).toBe(1);
    expect(filtered.availableDocumentCount).toBe(4);
    expect(filtered.selectedDocumentIds).toEqual(['irs-w9-english-p1']);
    expect(filtered.aggregate.documentCount).toBe(1);
    expect(filtered.documents).toHaveLength(1);
    expect(filtered.documents[0]?.documentId).toBe('irs-w9-english-p1');

    await expect(
      evaluateUnlimitedOcrLive({
        documentIds: ['missing-doc'],
        fetchFn: fetchFn as typeof fetch,
      })
    ).rejects.toThrow(/unknown document id\(s\): missing-doc/i);
  });

  it('keeps provider failures classified as errors', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/gradio_api/upload')) {
        return new Response(JSON.stringify(['/tmp/gradio/irs-form-w9-en-p1.png']), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/gradio_api/call/v2/run_ocr')) {
        return new Response(JSON.stringify({ event_id: 'evt-irs-w9-english-p1' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return sseResponse([
        'event: error\ndata: {"error":"ZeroGPU quota exceeded","title":"quota"}',
      ]);
    });

    const report = await evaluateUnlimitedOcrLive({
      documentIds: ['irs-w9-english-p1'],
      fetchFn: fetchFn as typeof fetch,
    });

    expect(report.aggregate.status).toBe('fail');
    expect(report.aggregate.errorDocuments).toBe(1);
    expect(report.aggregate.checks.noErrors).toBe(false);
    expect(report.documents[0]?.status).toBe('error');
    expect(report.documents[0]?.error).toMatch(/provider error/i);
  });
});

describe('CLI args', () => {
  it('parses repeatable document filters and help', () => {
    expect(
      parseUnlimitedOcrLiveCliArgs([
        '--document',
        'irs-w9-english-p1',
        '--document',
        'mathbridge-paper-p4',
        '--json',
        '--allow-failed-output',
        '--help',
      ])
    ).toMatchObject({
      documentIds: ['irs-w9-english-p1', 'mathbridge-paper-p4'],
      json: true,
      allowFailedOutput: true,
      help: true,
    });
  });

  it('rejects missing document values and formats help text', () => {
    expect(() => parseUnlimitedOcrLiveCliArgs(['--document'])).toThrow(
      /--document requires a value/i
    );
    expect(formatUnlimitedOcrLiveHelp()).toContain('HF_TOKEN');
    expect(formatUnlimitedOcrLiveHelp()).toContain('--document <id>');
  });
});
