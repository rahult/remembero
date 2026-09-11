import { describe, expect, it } from 'vitest';
import { OpenRouterClient } from '../src/llm/client.js';

const ok = (content: string) =>
  new Response(
    JSON.stringify({
      choices: [{ message: { content }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
    }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  );

describe('chat client rate-limit handling', () => {
  it('backs off and retries a 429 up to four attempts', async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      if (calls < 4) return new Response('rate limited', { status: 429 });
      return ok('fine');
    }) as unknown as typeof fetch;
    const client = new OpenRouterClient(
      { apiKey: 'k', baseUrl: 'https://example.invalid/v1', model: 'm' },
      fetchFn,
    );
    const started = Date.now();
    const result = await client.completeWithUsage([
      { role: 'user', content: 'hi' },
    ]);
    expect(result.content).toBe('fine');
    expect(calls).toBe(4);
    // three pauses, growing
    expect(Date.now() - started).toBeGreaterThanOrEqual(300);
  });

  it('gives up after the fourth 429', async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      return new Response('rate limited', { status: 429 });
    }) as unknown as typeof fetch;
    const client = new OpenRouterClient(
      { apiKey: 'k', baseUrl: 'https://example.invalid/v1', model: 'm' },
      fetchFn,
    );
    await expect(
      client.completeWithUsage([{ role: 'user', content: 'hi' }]),
    ).rejects.toThrow(/429/);
    expect(calls).toBe(4);
  });
});
