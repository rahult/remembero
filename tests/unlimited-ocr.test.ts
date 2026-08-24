import { describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_UNLIMITED_OCR_MODEL,
  UNLIMITED_OCR_MULTI_PAGE_WINDOW,
  UNLIMITED_OCR_NGRAM_SIZE,
  UNLIMITED_OCR_PROMPT,
  UNLIMITED_OCR_SINGLE_PAGE_WINDOW,
  UnlimitedOcrClient,
  parseUnlimitedOcrGrounding,
} from '../src/document/unlimited-ocr.js';

function dataUrl(text: string): string {
  return `data:image/png;base64,${Buffer.from(text).toString('base64')}`;
}

describe('UnlimitedOcrClient', () => {
  it('sends the required vLLM request contract for a single page', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe('http://localhost:8000/v1/chat/completions');
      expect(init?.method).toBe('POST');
      expect(init?.headers).toMatchObject({
        Authorization: 'Bearer EMPTY',
        'Content-Type': 'application/json',
      });
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      const body = JSON.parse(String(init?.body));
      expect(body).toEqual({
        model: DEFAULT_UNLIMITED_OCR_MODEL,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: UNLIMITED_OCR_PROMPT },
              {
                type: 'image_url',
                image_url: { url: dataUrl('page-1') },
              },
            ],
          },
        ],
        max_tokens: 8192,
        temperature: 0,
        skip_special_tokens: false,
        vllm_xargs: {
          ngram_size: UNLIMITED_OCR_NGRAM_SIZE,
          window_size: UNLIMITED_OCR_SINGLE_PAGE_WINDOW,
        },
      });
      return new Response(
        JSON.stringify({
          model: DEFAULT_UNLIMITED_OCR_MODEL,
          choices: [{ message: { content: '<|det|>title [1, 2, 3, 4]<|/det|>Hello' } }],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    });
    const client = new UnlimitedOcrClient(
      { baseUrl: 'http://localhost:8000/v1', allowPrivateNetwork: true },
      fetchFn as typeof fetch
    );
    await expect(client.parse([{ dataUrl: dataUrl('page-1') }])).resolves.toMatchObject({
      markdown: 'Hello',
      blocks: [{ category: 'title', coordinates: [1, 2, 3, 4], text: 'Hello' }],
    });
  });

  it('uses the multi-page window size when multiple images are passed', async () => {
    const fetchFn = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.vllm_xargs.window_size).toBe(UNLIMITED_OCR_MULTI_PAGE_WINDOW);
      expect(body.messages[0].content).toHaveLength(3);
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: 'Page 1\n\nPage 2' } }],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    });
    const client = new UnlimitedOcrClient(
      { baseUrl: 'http://localhost:8000/v1', allowPrivateNetwork: true },
      fetchFn as typeof fetch
    );
    const result = await client.parse([
      { dataUrl: dataUrl('page-1') },
      { dataUrl: dataUrl('page-2') },
    ]);
    expect(result.markdown).toBe('Page 1\n\nPage 2');
    expect(result.blocks).toEqual([]);
  });

  it('rejects bounded input and oversized responses', async () => {
    const client = new UnlimitedOcrClient({
      baseUrl: 'http://localhost:8000/v1',
      allowPrivateNetwork: true,
      maxImages: 1,
      maxTotalImageBytes: 5,
      maxResponseBytes: 16,
    });
    await expect(
      client.parse([{ dataUrl: dataUrl('page-1') }, { dataUrl: dataUrl('page-2') }])
    ).rejects.toThrow(/exceeded 1 images/i);

    const oversizedFetch = vi.fn(async () =>
      new Response(JSON.stringify({ choices: [{ message: { content: '0123456789abcdef' } }] }), {
        status: 200,
        headers: { 'content-length': '100' },
      })
    );
    const oversizedClient = new UnlimitedOcrClient(
      { baseUrl: 'http://localhost:8000/v1', allowPrivateNetwork: true, maxResponseBytes: 32 },
      oversizedFetch as typeof fetch
    );
    await expect(oversizedClient.parse([{ dataUrl: dataUrl('x') }])).rejects.toThrow(
      /response exceeded 32 bytes/i
    );
  });

  it('fails closed on malformed responses', async () => {
    const fetchFn = vi.fn(async () =>
      new Response(JSON.stringify({ choices: [{ message: { content: '<|ref|>oops' } }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );
    const client = new UnlimitedOcrClient(
      { baseUrl: 'http://localhost:8000/v1', allowPrivateNetwork: true },
      fetchFn as typeof fetch
    );
    await expect(client.parse([{ dataUrl: dataUrl('page-1') }])).rejects.toThrow(
      /unterminated <\|ref\|>/i
    );
  });

  it('requires explicit private-network authority and rejects credentialed URLs', () => {
    expect(
      () => new UnlimitedOcrClient({ baseUrl: 'http://localhost:8000/v1' })
    ).toThrow(/requires allowPrivateNetwork/i);
    expect(
      () =>
        new UnlimitedOcrClient({
          baseUrl: 'https://user:secret@example.com/v1',
        })
    ).toThrow(/must not contain credentials/i);
    expect(
      () => new UnlimitedOcrClient({ baseUrl: 'http://example.com/v1' })
    ).toThrow(/public baseUrl must use https/i);
  });
});

describe('parseUnlimitedOcrGrounding', () => {
  it('unwraps ref blocks and keeps category plus numeric coordinates', () => {
    const parsed = parseUnlimitedOcrGrounding(
      [
        '# Title',
        '<|ref|><|det|>title [10, 20, 30, 40]<|/det|>Quarterly Review<|/ref|>',
        '',
        '<|det|>text [1.5, 2, 3, 4.25]<|/det|>Revenue grew 20%.',
      ].join('\n')
    );
    expect(parsed.markdown).toBe('# Title\nQuarterly Review\n\nRevenue grew 20%.');
    expect(parsed.blocks).toEqual([
      {
        category: 'title',
        coordinates: [10, 20, 30, 40],
        text: 'Quarterly Review',
      },
      {
        category: 'text',
        coordinates: [1.5, 2, 3, 4.25],
        text: 'Revenue grew 20%.',
      },
    ]);
  });
});
