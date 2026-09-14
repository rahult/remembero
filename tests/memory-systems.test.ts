import { describe, expect, it } from 'vitest';
import {
  MEMORY_SYSTEMS_PROTOCOL_VERSION,
  memorySystemDate,
  memorySystemRequestFor,
  parseMemorySystemResponse,
  summarizeMemorySystemUsage,
} from '../src/evals/memory-systems-protocol.js';
import { createMemorySystemProcess } from '../src/evals/memory-systems-process.js';
import type { LongMemEvalInstance } from '../src/evals/longmemeval.js';

function instance(): LongMemEvalInstance {
  return {
    question_id: 'ba358f49_abs',
    question_type: 'temporal-reasoning',
    question: 'How long ago did I graduate?',
    answer: 'Business Administration, 2019',
    question_date: '2023/07/10 (Mon) 09:00',
    haystack_session_ids: ['noise', 'evidence'],
    haystack_dates: ['2023/05/15 (Mon) 02:21', '2023/06/01 (Thu) 11:04'],
    haystack_sessions: [
      [
        { role: 'user', content: 'Compare credit card rewards.' },
        { role: 'assistant', content: 'Here are three.' },
      ],
      [
        { role: 'user', content: 'I graduate with Business Administration.' },
        { role: 'assistant', content: 'Congratulations.' },
      ],
    ],
    answer_session_ids: ['evidence'],
  };
}

// A stub adapter: one process, one JSON line per question, noise on stderr.
const STUB = `
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const request = JSON.parse(line);
  process.stderr.write('adapter chatter that must never be parsed\\n');
  const hits = request.sessions.filter((session) =>
    session.turns.some((turn) => turn.content.includes('graduate')));
  process.stdout.write(JSON.stringify({
    questionId: request.questionId,
    retrieved: hits.map((session, index) => ({
      sessionId: session.id, rank: index + 1, score: 1 - index / 10,
    })),
    memories: hits.map((session) => ({
      text: 'the user graduated with Business Administration',
      sessionIds: [session.id],
      at: session.date,
    })),
    unsupported: [],
    usage: { modelCalls: 2, inputTokens: 100, outputTokens: 10, costUsd: 0.0001 },
    wallMs: { ingest: 41, search: 2 },
  }) + '\\n');
});
`;

const SILENT = 'setInterval(() => {}, 1000);';

describe('rembero.memory-systems.v1', () => {
  it('normalizes dataset dates and never shows the adapter the gold answer', () => {
    expect(memorySystemDate('2023/07/10 (Mon) 09:00')).toBe('2023-07-10');
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(request.protocolVersion).toBe(MEMORY_SYSTEMS_PROTOCOL_VERSION);
    expect(request.topK).toBe(4);
    expect(request.lanes).toEqual(['retrieval']);
    expect(request.sessions.map(({ id, date }) => [id, date])).toEqual([
      ['noise', '2023-05-15'],
      ['evidence', '2023-06-01'],
    ]);
    const serialized = JSON.stringify(request);
    expect(serialized).not.toContain('Business Administration, 2019');
    expect(serialized).not.toContain('answer_session_ids');
  });

  it('rejects a session id that was not in the request', () => {
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(() =>
      parseMemorySystemResponse(
        { questionId: request.questionId, retrieved: [{ sessionId: 'invented', rank: 1 }] },
        request,
      ),
    ).toThrow(/not in the request/);
  });

  it('rejects an answer to a different question and surfaces a reported error', () => {
    const request = memorySystemRequestFor(instance(), 'retrieval', 4);
    expect(() => parseMemorySystemResponse({ questionId: 'other' }, request)).toThrow(
      /expected ba358f49_abs/,
    );
    expect(() =>
      parseMemorySystemResponse({ questionId: request.questionId, error: 'qdrant died' }, request),
    ).toThrow(/qdrant died/);
  });

  it('keeps one process alive across questions and discards stderr', async () => {
    const client = createMemorySystemProcess({
      id: 'stub',
      executable: process.execPath,
      args: ['-e', STUB],
      timeoutMs: 10_000,
    });
    try {
      const first = await client.request(memorySystemRequestFor(instance(), 'retrieval', 4));
      const second = await client.request(memorySystemRequestFor(instance(), 'memories', 4));
      expect(first.retrieved).toEqual([{ sessionId: 'evidence', rank: 1, score: 1 }]);
      expect(second.memories?.[0]?.sessionIds).toEqual(['evidence']);
      expect(second.usage).toEqual({
        modelCalls: 2,
        inputTokens: 100,
        outputTokens: 10,
        costUsd: 0.0001,
      });
      expect(second.wallMs).toEqual({ ingest: 41, search: 2 });
    } finally {
      await client.close();
    }
  });

  it('kills a silent adapter at the timeout', async () => {
    const client = createMemorySystemProcess({
      id: 'silent',
      executable: process.execPath,
      args: ['-e', SILENT],
      timeoutMs: 400,
    });
    try {
      await expect(
        client.request(memorySystemRequestFor(instance(), 'retrieval', 4)),
      ).rejects.toThrow(/timed out after 400ms/);
    } finally {
      await client.close();
    }
  });

  it('totals adapter usage across observations', () => {
    const summary = summarizeMemorySystemUsage([
      {
        memorySystem: {
          id: 'stub',
          lane: 'retrieval',
          usage: { modelCalls: 2, inputTokens: 100, outputTokens: 10, costUsd: 0.01 },
          wallMs: { ingest: 40, search: 2 },
          returnedSessions: 1,
          returnedMemories: 0,
          memoryBytes: { supplied: 0, kept: 0, dropped: 0 },
          unsupported: [],
        },
      },
      {
        memorySystem: {
          id: 'stub',
          lane: 'retrieval',
          usage: { modelCalls: 3, inputTokens: 200, outputTokens: 20, costUsd: 0.02 },
          wallMs: { ingest: 60, search: 4 },
          returnedSessions: 1,
          returnedMemories: 0,
          memoryBytes: { supplied: 0, kept: 0, dropped: 0 },
          unsupported: [],
        },
      },
      {},
    ]);
    expect(summary).toEqual({
      questions: 2,
      modelCalls: 5,
      inputTokens: 300,
      outputTokens: 30,
      costUsd: 0.03,
      medianIngestMs: 50,
      medianSearchMs: 3,
      droppedMemoryBytes: 0,
    });
  });
});
