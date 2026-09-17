import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_LLM_TIMEOUT_MS,
  OpenRouterClient,
  llmTuningFromFlags,
} from '../src/llm/client.js';
import { parseArgs } from '../src/evals/run-longmemeval-answer.js';
import { distillRunIdentity } from '../src/training/reader-distill.js';

const ok = () =>
  new Response(
    JSON.stringify({
      choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
    }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  );

/** A client whose every request body is captured verbatim. */
function recordingClient(
  config: ConstructorParameters<typeof OpenRouterClient>[0],
): { client: OpenRouterClient; bodies: string[] } {
  const bodies: string[] = [];
  const fetchFn = (async (_url: unknown, init: RequestInit) => {
    bodies.push(String(init.body));
    return ok();
  }) as unknown as typeof fetch;
  return { client: new OpenRouterClient(config, fetchFn), bodies };
}

const runnerSource = (): string =>
  readFileSync(
    new URL('../src/evals/run-longmemeval-answer.ts', import.meta.url),
    'utf8',
  );

afterEach(() => {
  vi.restoreAllMocks();
});

describe('LLM client temperature and timeout', () => {
  it('sends the same bytes as before when neither option is given', async () => {
    const { client, bodies } = recordingClient({
      apiKey: 'k',
      baseUrl: 'https://x.test/v1',
      model: 'm',
    });
    await client.complete([{ role: 'user', content: 'hi' }]);
    expect(bodies[0]).toBe(
      '{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0}',
    );
  });

  it('defaults the request timeout to 60s and uses timeoutMs when given', async () => {
    const timeout = vi.spyOn(AbortSignal, 'timeout');
    const plain = recordingClient({
      apiKey: 'k',
      baseUrl: 'https://x.test/v1',
      model: 'm',
    });
    await plain.client.complete([{ role: 'user', content: 'hi' }]);
    expect(DEFAULT_LLM_TIMEOUT_MS).toBe(60_000);
    expect(timeout).toHaveBeenLastCalledWith(DEFAULT_LLM_TIMEOUT_MS);
    const slow = recordingClient({
      apiKey: 'k',
      baseUrl: 'https://x.test/v1',
      model: 'm',
      timeoutMs: 300_000,
    });
    await slow.client.complete([{ role: 'user', content: 'hi' }]);
    expect(timeout).toHaveBeenLastCalledWith(300_000);
  });

  it('sends the configured temperature, as Kimi K3 requires temperature 1', async () => {
    const { client, bodies } = recordingClient({
      apiKey: 'k',
      baseUrl: 'https://api.moonshot.ai/v1',
      model: 'kimi-k3',
      temperature: 1,
    });
    await client.completeWithUsage([{ role: 'user', content: 'hi' }], {
      maxTokens: 42,
    });
    const body = JSON.parse(bodies[0]) as {
      temperature: number;
      max_tokens: number;
    };
    expect(body.temperature).toBe(1);
    expect(body.max_tokens).toBe(42);
  });

  it('keeps the retry loop under a configured temperature', async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls += 1;
      if (calls < 3) return new Response('rate limited', { status: 429 });
      return ok();
    }) as unknown as typeof fetch;
    const client = new OpenRouterClient(
      {
        apiKey: 'k',
        baseUrl: 'https://x.test/v1',
        model: 'm',
        temperature: 1,
        timeoutMs: 120_000,
      },
      fetchFn,
    );
    expect(await client.complete([{ role: 'user', content: 'hi' }])).toBe('ok');
    expect(calls).toBe(3);
  });

  it('rejects an out-of-range temperature or timeout before any request', () => {
    const base = { apiKey: 'k', baseUrl: 'https://x.test/v1', model: 'm' };
    const fetchFn = vi.fn() as unknown as typeof fetch;
    expect(
      () => new OpenRouterClient({ ...base, temperature: 2.5 }, fetchFn),
    ).toThrow(/temperature/i);
    expect(
      () => new OpenRouterClient({ ...base, temperature: -1 }, fetchFn),
    ).toThrow(/temperature/i);
    expect(
      () => new OpenRouterClient({ ...base, temperature: Number.NaN }, fetchFn),
    ).toThrow(/temperature/i);
    expect(
      () => new OpenRouterClient({ ...base, timeoutMs: 999 }, fetchFn),
    ).toThrow(/timeout/i);
    expect(
      () => new OpenRouterClient({ ...base, timeoutMs: 600_001 }, fetchFn),
    ).toThrow(/timeout/i);
    expect(
      () => new OpenRouterClient({ ...base, timeoutMs: 1000.5 }, fetchFn),
    ).toThrow(/timeout/i);
    expect(fetchFn).not.toHaveBeenCalled();
  });
});

describe('tuning flags for CLI-built clients', () => {
  it('reads an unprefixed pair and leaves both out when absent', () => {
    expect(llmTuningFromFlags(['--model', 'kimi-k3'])).toEqual({});
    expect(
      llmTuningFromFlags([
        '--model',
        'kimi-k3',
        '--temperature',
        '1',
        '--timeout-ms',
        '300000',
      ]),
    ).toEqual({ temperature: 1, timeoutMs: 300_000 });
  });

  it('reads a prefixed pair, so the student and the teacher are tuned apart', () => {
    expect(
      llmTuningFromFlags(
        ['--temperature', '0.7', '--student-temperature', '1'],
        'student-',
      ),
    ).toEqual({ temperature: 1 });
    expect(
      llmTuningFromFlags(['--student-timeout-ms', '180000'], 'student-'),
    ).toEqual({ timeoutMs: 180_000 });
  });

  it('names the offending flag when the value is bad or missing', () => {
    expect(() => llmTuningFromFlags(['--temperature', '3'])).toThrow(
      /--temperature/,
    );
    expect(() => llmTuningFromFlags(['--timeout-ms', '10'])).toThrow(
      /--timeout-ms/,
    );
    expect(() => llmTuningFromFlags(['--temperature'])).toThrow(
      /--temperature needs a value/,
    );
  });
});

describe('the distiller pins the teacher tuning in run.json', () => {
  it('drops the knobs when they were left at the defaults', () => {
    const identity = distillRunIdentity({
      contract: 'dd+notes@24576',
      teacher: 'z-ai/glm-5.3-flash',
      seed: 7,
      splitSeed: 7,
      typeWeights: 'default',
      trainCount: 3000,
      labels: 'data/real/labels.jsonl',
    });
    expect(identity).not.toHaveProperty('temperature');
    expect(identity).not.toHaveProperty('timeoutMs');
  });

  it('keeps them when the run set them', () => {
    expect(
      distillRunIdentity({
        contract: 'dd+notes@24576',
        teacher: 'kimi-k3',
        seed: 7,
        splitSeed: 7,
        typeWeights: 'default',
        trainCount: 3000,
        labels: 'data/real/labels.jsonl',
        temperature: 1,
        timeoutMs: 300_000,
      }),
    ).toMatchObject({ temperature: 1, timeoutMs: 300_000 });
  });
});

describe('LongMemEval answer runner reader tuning flags', () => {
  it('defaults all three to undefined', () => {
    const args = parseArgs([]);
    expect(args.readerTemperature).toBeUndefined();
    expect(args.readerTimeoutMs).toBeUndefined();
    expect(args.judgeTemperature).toBeUndefined();
  });

  it('parses the reader and judge knobs', () => {
    const args = parseArgs([
      '--reader-model',
      'kimi-k3',
      '--reader-temperature',
      '1',
      '--reader-timeout-ms',
      '300000',
      '--judge-temperature',
      '0.2',
    ]);
    expect(args.readerTemperature).toBe(1);
    expect(args.readerTimeoutMs).toBe(300_000);
    expect(args.judgeTemperature).toBe(0.2);
  });

  it('rejects out-of-range values', () => {
    expect(() => parseArgs(['--reader-temperature', '4'])).toThrow(
      /--reader-temperature/,
    );
    expect(() => parseArgs(['--reader-timeout-ms', '999'])).toThrow(
      /--reader-timeout-ms/,
    );
    expect(() => parseArgs(['--reader-timeout-ms'])).toThrow(/needs a value/);
  });

  it('documents the flags next to --reader-max-tokens', () => {
    const source = runnerSource();
    expect(source).toMatch(/--reader-temperature <n>/);
    expect(source).toMatch(/--reader-timeout-ms <n>/);
    expect(source).toMatch(/--judge-temperature <n>/);
  });
});
