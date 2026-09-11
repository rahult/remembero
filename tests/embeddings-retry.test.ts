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

  it('retries a 400 that is really the local runner dying mid-request', async () => {
    // Ollama answers 400 when its llama runner subprocess drops the connection
    // ("Post .../tokenize: EOF"); the runner restarts, so the request is worth retrying
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      if (calls === 1)
        return new Response(
          JSON.stringify({
            error: {
              message:
                'Post "http://127.0.0.1:61954/tokenize": read tcp 127.0.0.1:63295->127.0.0.1:61954: read: connection reset by peer',
            },
          }),
          { status: 400 },
        );
      return new Response(
        JSON.stringify({
          data: [{ index: 0, embedding: [0.3, 0.4] }],
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }) as unknown as typeof fetch;
    const client = new OpenRouterEmbeddingClient(
      { apiKey: 'k', baseUrl: 'http://127.0.0.1:11434/v1', model: 'm' },
      fetchFn,
    );
    const result = await client.embed(['hello']);
    expect(calls).toBe(2);
    expect(result.vectors[0]).toEqual([0.3, 0.4]);
  });

  it('does not retry an ordinary 400', async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({ error: { message: 'model "nope" not found' } }),
        { status: 400 },
      );
    }) as unknown as typeof fetch;
    const client = new OpenRouterEmbeddingClient(
      { apiKey: 'k', baseUrl: 'http://127.0.0.1:11434/v1', model: 'nope' },
      fetchFn,
    );
    await expect(client.embed(['hello'])).rejects.toThrow(/not found/);
    expect(calls).toBe(1);
  });
});
