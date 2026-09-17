/**
 * Protocol conformance for the two shipped Python bridges. These spawn the real adapter
 * process through the same client the runner uses, so they prove the wire contract end to
 * end: one long-lived process, one JSON line per question, ids that came from the request,
 * and a store that does not survive into the next question.
 *
 * They need `uv` on PATH and a warm dependency cache; the first run of a bridge resolves
 * its pins and downloads the 67 MB embedding model, so the suite skips when `uv` is absent.
 */
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { loadExternalAdapterManifest } from '../src/evals/memory-stack-external.js';
import { openMemorySystem } from '../src/evals/memory-systems-builtin.js';
import {
  MEMORY_SYSTEMS_PROTOCOL_VERSION,
  memorySystemRequestFor,
} from '../src/evals/memory-systems-protocol.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';

const BRIDGE_TIMEOUT_MS = 600_000;

function haveUv(): boolean {
  try {
    execFileSync('uv', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function manifestPath(adapter: string): string {
  return fileURLToPath(
    new URL(`../benchmarks/adapters/${adapter}/memory-systems.json`, import.meta.url),
  );
}

/** First question: the evidence session is the only one that mentions graduating. */
function graduationInstance(): LongMemEvalInstance {
  return {
    question_id: 'bridge_one',
    question_type: 'single-session-user',
    question: 'What degree did I graduate with?',
    answer: 'Business Administration',
    question_date: '2023/07/10 (Mon) 09:00',
    haystack_session_ids: ['noise', 'evidence'],
    haystack_dates: ['2023/05/15 (Mon) 02:21', '2023/06/01 (Thu) 11:04'],
    haystack_sessions: [
      [
        { role: 'user', content: 'Compare credit card rewards for me.' },
        { role: 'assistant', content: 'Here are three cards worth a look.' },
      ],
      [
        { role: 'user', content: 'I graduate with a degree in Business Administration.' },
        { role: 'assistant', content: 'Congratulations on the degree.' },
      ],
    ],
    answer_session_ids: ['evidence'],
  };
}

/**
 * Second question: entirely different sessions, but deliberately the *same* query as the
 * first. A store that survived the first question would rank its graduation session top
 * here, so the assertion that 'evidence' is absent has something to catch.
 */
function bicycleInstance(): LongMemEvalInstance {
  return {
    question_id: 'bridge_two',
    question_type: 'single-session-user',
    question: 'What degree did I graduate with?',
    answer: 'green',
    question_date: '2023/08/02 (Wed) 09:00',
    haystack_session_ids: ['weather', 'bicycle'],
    haystack_dates: ['2023/07/01 (Sat) 08:00', '2023/07/20 (Thu) 18:30'],
    haystack_sessions: [
      [
        { role: 'user', content: 'Will it rain in Lisbon next week?' },
        { role: 'assistant', content: 'Showers on Tuesday, otherwise dry.' },
      ],
      [
        { role: 'user', content: 'I painted my bicycle green over the weekend.' },
        { role: 'assistant', content: 'Green is a fine colour for a bicycle.' },
      ],
    ],
    answer_session_ids: ['bicycle'],
  };
}

describe.skipIf(!haveUv())('memory-systems bridges over rembero.memory-systems.v1', () => {
  for (const [adapter, adapterId] of [
    ['langgraph-fastembed', 'langgraph-inmemory-fastembed'],
    ['llamaindex-fastembed', 'llamaindex-vectormemory-fastembed'],
  ] as const) {
    it(`${adapter} declares the memory-systems protocol in its manifest`, async () => {
      const manifest = await loadExternalAdapterManifest(manifestPath(adapter));
      expect(manifest.protocol).toBe(MEMORY_SYSTEMS_PROTOCOL_VERSION);
      expect(manifest.descriptor.id).toBe(adapterId);
      expect(manifest.command.executable).toBe('uv');
      expect(manifest.command.args).toEqual(['run', '--script', 'memory_systems_bridge.py']);
      expect(manifest.descriptor.capabilities.rankedRetrieval).toBe(true);
      expect(manifest.descriptor.capabilities.answerRows).toBe(false);
    });

    it(
      `${adapter} answers two questions from one process with a fresh store each time`,
      async () => {
        const client = await openMemorySystem(manifestPath(adapter), {
          timeoutMs: BRIDGE_TIMEOUT_MS,
        });
        try {
          const firstRequest = memorySystemRequestFor(graduationInstance(), 'retrieval', 2);
          const first = await client.request(firstRequest);
          const firstIds = (first.retrieved ?? []).map(({ sessionId }) => sessionId);
          expect(firstIds.length).toBeGreaterThan(0);
          expect(firstIds.length).toBeLessThanOrEqual(firstRequest.topK);
          // parseMemorySystemResponse already refuses an id outside the request; assert the
          // ranking itself, which the parser cannot judge.
          expect(firstIds[0]).toBe('evidence');
          expect(first.retrieved?.map(({ rank }) => rank)).toEqual(
            firstIds.map((_, index) => index + 1),
          );
          // A pure retriever takes no lane it cannot serve, and spends nothing doing it.
          expect(first.unsupported).toEqual(['memories']);
          expect(first.memories).toEqual([]);
          expect(first.usage).toEqual({
            modelCalls: 0,
            inputTokens: 0,
            outputTokens: 0,
            costUsd: 0,
          });
          expect(first.wallMs?.ingest).toBeGreaterThan(0);

          const secondRequest = memorySystemRequestFor(bicycleInstance(), 'retrieval', 2);
          const second = await client.request(secondRequest);
          const secondIds = (second.retrieved ?? []).map(({ sessionId }) => sessionId);
          expect(secondIds.length).toBeGreaterThan(0);
          expect(secondIds.length).toBeLessThanOrEqual(secondRequest.topK);
          // Same query as the first question: a store that carried over would put its
          // graduation session at the top of this list.
          expect(secondIds).not.toContain('evidence');
          expect(secondIds).not.toContain('noise');
          expect(secondIds.every((id) => id === 'weather' || id === 'bicycle')).toBe(true);
          expect(second.unsupported).toEqual(['memories']);
        } finally {
          await client.close();
        }
      },
      BRIDGE_TIMEOUT_MS,
    );
  }
});
