import { describe, expect, it } from 'vitest';
import { OpenRouterEmbeddingClient } from '../src/llm/embeddings.js';

describe('embedding client rate-limit handling', () => {
  it('retries a 429 with a pause and succeeds on a later attempt', async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      if (calls < 3) return new Response('rate limited', { status: 429 });
      return new Response(
        JSON.stringify({
          data: [{ index: 0, embedding: [0.1, 0.2] }],
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }) as unknown as typeof fetch;
    const client = new OpenRouterEmbeddingClient(
      { apiKey: 'k', baseUrl: 'https://example.invalid/v1', model: 'm' },
      fetchFn,
    );
    const started = Date.now();
    const result = await client.embed(['hello']);
    expect(calls).toBe(3);
    expect(result.vectors[0]).toEqual([0.1, 0.2]);
    // two pauses happened (short in tests, but present)
    expect(Date.now() - started).toBeGreaterThanOrEqual(100);
  });
});
