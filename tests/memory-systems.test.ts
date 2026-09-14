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

// An adapter that hangs on one question and answers the next, to prove a killed child does not
// take the question after it down with it.
const HANGS_ON_MARKED = `
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const request = JSON.parse(line);
  if (request.question.includes('hang')) return;
  process.stdout.write(JSON.stringify({
    questionId: request.questionId,
    retrieved: [{ sessionId: 'evidence', rank: 1 }],
  }) + '\\n');
});
`;

// An adapter that prints a startup banner on stdout before it ever answers.
const BANNER = `
const readline = require('node:readline');
process.stdout.write('fastembed: loading model (this is not an answer)\\n');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const request = JSON.parse(line);
  process.stdout.write('progress: 100%\\n');
  process.stdout.write(JSON.stringify({
    questionId: request.questionId,
    retrieved: [{ sessionId: 'evidence', rank: 1 }],
  }) + '\\n');
});
`;

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

  it('answers the question after a timeout through a fresh child', async () => {
    const client = createMemorySystemProcess({
      id: 'flaky',
      executable: process.execPath,
      args: ['-e', HANGS_ON_MARKED],
      timeoutMs: 400,
    });
    try {
      const hung = instance();
      hung.question = 'please hang on this one';
      await expect(
        client.request(memorySystemRequestFor(hung, 'retrieval', 4)),
      ).rejects.toThrow(/timed out after 400ms/);
      // The killed child must not settle this one with "exited with SIGKILL".
      const next = await client.request(memorySystemRequestFor(instance(), 'retrieval', 4));
      expect(next.retrieved).toEqual([{ sessionId: 'evidence', rank: 1 }]);
    } finally {
      await client.close();
    }
  });

  it('skips non-JSON stdout chatter instead of failing the question', async () => {
    const client = createMemorySystemProcess({
      id: 'noisy',
      executable: process.execPath,
      args: ['-e', BANNER],
      timeoutMs: 10_000,
    });
    try {
      const first = await client.request(memorySystemRequestFor(instance(), 'retrieval', 4));
      expect(first.retrieved).toEqual([{ sessionId: 'evidence', rank: 1 }]);
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

import {
  evaluateLongMemEvalAnswerInstance,
  longMemEvalAnswerRun,
  type LongMemEvalCompletionClient,
} from '../src/evals/longmemeval-answer.js';
import type { ChatMessage, LlmCompletion } from '../src/llm/client.js';
import {
  createBuiltinMemorySystem,
  isBuiltinMemorySystem,
} from '../src/evals/memory-systems-builtin.js';
import type {
  MemorySystemClient,
  MemorySystemResponse,
} from '../src/evals/memory-systems-protocol.js';

class ScriptedClient implements LongMemEvalCompletionClient {
  readonly calls: ChatMessage[][] = [];
  constructor(
    readonly model: string,
    private readonly outputs: string[],
  ) {}
  async completeWithUsage(messages: ChatMessage[]): Promise<LlmCompletion> {
    this.calls.push(structuredClone(messages));
    const content = this.outputs.shift();
    if (content === undefined) throw new Error('scripted completion exhausted');
    return {
      content,
      model: this.model,
      usage: {
        promptTokens: 10,
        completionTokens: 2,
        totalTokens: 12,
        cachedPromptTokens: 0,
        reasoningTokens: 0,
        costUsd: 0.001,
      },
    };
  }
}

function fixedClient(
  id: string,
  response: Omit<MemorySystemResponse, 'questionId'>,
): MemorySystemClient {
  return {
    id,
    async request(request) {
      return { questionId: request.questionId, ...response };
    },
    async close() {},
  };
}

describe('the memory-system seam', () => {
  it('replaces the lexical search with the adapter ranking in the retrieval lane', async () => {
    const reader = new ScriptedClient('reader', ['Business Administration.']);
    const judge = new ScriptedClient('judge', ['yes']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      // not the _abs id the other cases use: retrieval recall is scored only for
      // questions that have evidence to find
      { ...instance(), question_id: 'ba358f49' },
      reader,
      judge,
      {
        topK: 4,
        contextBytes: 24_576,
        semanticQuestionTypes: new Set<string>(),
        memorySystem: {
          lane: 'retrieval',
          client: fixedClient('stub', {
            retrieved: [{ sessionId: 'evidence', rank: 1, score: 0.83 }],
            memories: [],
            unsupported: ['memories'],
            usage: {
              modelCalls: 48,
              inputTokens: 120_000,
              outputTokens: 3_000,
              costUsd: 0.03,
            },
            wallMs: { ingest: 41_000, search: 120 },
          }),
        },
      },
    );
    expect(observation.status).toBe('judged');
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.retrieval?.recallAtK).toBe(1);
    expect(observation.memorySystem).toMatchObject({
      id: 'stub',
      lane: 'retrieval',
      usage: { modelCalls: 48, costUsd: 0.03 },
      wallMs: { ingest: 41_000, search: 120 },
      returnedSessions: 1,
    });
    expect(reader.calls[0]?.[1]?.content).toContain(
      'I graduate with Business Administration',
    );
    expect(reader.calls[0]?.[1]?.content).not.toContain('credit card rewards');
  });

  it('reads from memory text alone in the memories lane and records what it dropped', async () => {
    const reader = new ScriptedClient('reader', ['Business Administration.']);
    const judge = new ScriptedClient('judge', ['yes']);
    const filler = 'x'.repeat(3_000);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      {
        topK: 4,
        // small on purpose: the second memory cannot fit, so it is dropped and counted
        contextBytes: 4_096,
        semanticQuestionTypes: new Set<string>(),
        memorySystem: {
          lane: 'memories',
          client: fixedClient('stub', {
            retrieved: [],
            memories: [
              {
                text: "User's degree is Business Administration",
                sessionIds: ['evidence'],
                at: '2023-06-01',
              },
              { text: filler, sessionIds: ['noise'], at: '2023-05-15' },
            ],
            unsupported: [],
            usage: {
              modelCalls: 3,
              inputTokens: 10,
              outputTokens: 1,
              costUsd: 0.001,
            },
            wallMs: { ingest: 10, search: 1 },
          }),
        },
      },
    );
    expect(observation.status).toBe('judged');
    expect(reader.calls[0]?.[1]?.content).toContain(
      "MEMORY: User's degree is Business Administration",
    );
    expect(reader.calls[0]?.[1]?.content).not.toContain(
      'I graduate with Business Administration',
    );
    expect(observation.retrievedSessionIds).toEqual(['evidence']);
    expect(observation.memorySystem?.returnedMemories).toBe(2);
    expect(observation.memorySystem?.memoryBytes.dropped).toBeGreaterThan(2_000);
    expect(observation.memorySystem?.memoryBytes.kept).toBeLessThan(4_096);
  });

  it('fails the question when the memories lane comes back empty', async () => {
    const reader = new ScriptedClient('reader', ['unused']);
    const judge = new ScriptedClient('judge', ['unused']);
    const observation = await evaluateLongMemEvalAnswerInstance(
      instance(),
      reader,
      judge,
      {
        topK: 4,
        semanticQuestionTypes: new Set<string>(),
        memorySystem: {
          lane: 'memories',
          client: fixedClient('retriever-only', {
            retrieved: [{ sessionId: 'evidence', rank: 1 }],
            memories: [],
            unsupported: ['memories'],
          }),
        },
      },
    );
    expect(observation.status).toBe('error');
    expect(observation.error).toMatch(/memories lane is unsupported/);
  });

  it('ranks with BM25 and returns the whole haystack newest first for full context', async () => {
    expect(isBuiltinMemorySystem('builtin:bm25')).toBe(true);
    expect(isBuiltinMemorySystem('builtin:nothing')).toBe(false);
    const request = memorySystemRequestFor(instance(), 'retrieval', 2);
    const bm25 = createBuiltinMemorySystem('builtin:bm25', {});
    const ranked = await bm25.request(request);
    expect(ranked.retrieved?.[0]?.sessionId).toBe('evidence');
    expect(ranked.unsupported).toEqual(['memories']);
    const full = createBuiltinMemorySystem('builtin:full-context', {});
    const everything = await full.request(request);
    expect(everything.retrieved?.map(({ sessionId }) => sessionId)).toEqual([
      'evidence',
      'noise',
    ]);
    await bm25.close();
    await full.close();
  });

  it('names the memory system in the run artifact without changing the schema', () => {
    const run = longMemEvalAnswerRun(
      [],
      'glm-5.3-flash:cloud',
      'deepseek-chat',
      {
        memorySystemId: 'builtin:bm25',
        topK: 4,
        multiSessionTopK: 15,
        temporalTopK: 10,
        contextBytes: 24_576,
        settings: {
          aggregationReaderModel: null,
          temporalRangeModel: null,
          readingStrategy: 'direct',
          hybridRetrieval: 'shared',
          retrievalUnit: 'session',
          entityRetrieval: false,
          hybridQuestionTypes: null,
          factsInContext: true,
          readerMaxTokens: 440,
          memorySystem: 'builtin:bm25',
          memoryLane: 'retrieval',
        },
      },
    );
    expect(run.retrieval).toBe('memory-system:builtin:bm25');
    expect(run.schemaVersion).toBe('remembero.longmemeval-answer.v1');
    expect(run.settings?.memoryLane).toBe('retrieval');
  });
});
