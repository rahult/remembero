import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_READING_CANDIDATE_SESSIONS,
  MAX_READING_CANDIDATE_TURNS,
  cappedReadingCandidates,
  recallQuestion,
  type RecallReader,
} from '../src/llm/pipeline.js';
import {
  DEFAULT_PRODUCT_READING_CONTEXT_BYTES,
  DEFAULT_READER_MAX_TOKENS,
  MAX_CONFIGURED_READING_CONTEXT_BYTES,
  MIN_READING_CONTEXT_BYTES,
  readerFromEnv,
  readingContextBytesFromEnv,
  recallAnswerModeFromEnv,
} from '../src/env.js';
import { MAX_KNOWLEDGE_SEARCH_CLAUSES } from '../src/knowledge/search.js';
import { MAX_READING_CONTEXT_BYTES } from '../src/knowledge/session-retrieval.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { MemoryStore } from '../src/store/store.js';
import { SessionStore, type SessionSource } from '../src/sessions/store.js';
import { MAX_OUTPUT_BYTES } from '../src/safety.js';

const ASKED_AT = new Date('2024-03-01T09:00:00.000Z');

/** The reader under test: scripted reply, every prompt it was handed recorded. */
class StubReaderClient implements LlmClient {
  prompts: ChatMessage[][] = [];
  constructor(private reply: string) {}
  async complete(messages: ChatMessage[]): Promise<string> {
    this.prompts.push(messages);
    return this.reply;
  }
}

/** The product's configured LLM: a sessions recall must not reach for it. */
class NeverCalledLlm implements LlmClient {
  async complete(): Promise<string> {
    throw new Error('the configured LLM was called');
  }
}

/** An LlmClient with scripted replies, for the Datalog path. */
class ScriptedLlm implements LlmClient {
  calls: ChatMessage[][] = [];
  constructor(private responses: string[]) {}
  async complete(messages: ChatMessage[]): Promise<string> {
    this.calls.push(messages);
    const next = this.responses.shift();
    if (next === undefined) throw new Error('ScriptedLlm ran out of responses');
    return next;
  }
}

function reader(client: LlmClient, baseUrl = 'http://127.0.0.1:8084/v1'): RecallReader {
  return {
    baseUrl,
    model: 'reader-v7',
    apiKey: 'local',
    maxTokens: 512,
    temperature: 0,
    timeoutMs: 5_000,
    client,
  };
}

let store: MemoryStore;
let sessions: SessionStore;

function storeSession(
  sourceSessionId: string,
  startedAt: string,
  turns: Array<{ role: 'user' | 'assistant'; text: string }>,
): string {
  const source: SessionSource = 'import';
  sessions.appendTurns(
    'default',
    { version: 1, source, sourceSessionId, startedAt },
    turns.map((turn, index) => ({
      ...turn,
      ts: new Date(Date.parse(startedAt) + index * 1_000).toISOString(),
    })),
  );
  return sessions.sessionKey(source, sourceSessionId);
}

beforeEach(() => {
  store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-sessions-recall-')));
  sessions = new SessionStore({
    root: mkdtempSync(join(tmpdir(), 'rembero-sessions-recall-store-')),
    capBytes: 1024 * 1024,
    log: () => {},
  });
});

describe('the sessions answer mode', () => {
  it('answers from the one stored session that holds the evidence', async () => {
    storeSession('museums', '2024-01-12T09:00:00.000Z', [
      { role: 'user', text: 'I visited three museums in Lisbon and spent 42 euros on tickets.' },
      { role: 'assistant', text: 'Lisbon has a wonderful museum scene.' },
    ]);
    const bikes = storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
      { role: 'assistant', text: 'Two bikes is a tidy stable.' },
    ]);
    const client = new StubReaderClient(
      'Notes:\n- 2024-02-27: the user owns two bikes\nAnswer: Two bikes.',
    );
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
    );
    expect(result.status).toBe('answered');
    expect(result.answerMode).toBe('sessions');
    expect(result.answer).toBe('Two bikes.');
    // a plain recall never shows the reader's working
    expect(result.answer).not.toContain('Notes:');
    expect(result.readerReply).toBeUndefined();
    expect(result.readerModel).toBe('reader-v7');
    const read = result.sessionsRead ?? [];
    expect(read[0]?.namespace).toBe('default');
    expect(read[0]?.key).toBe(bikes);
    expect(read[0]?.date).toBe('2024-02-27');
    expect(read[0]?.excerpts.join(' ')).toContain('two bikes');
    expect(client.prompts).toHaveLength(1);
    expect(client.prompts[0]?.at(-1)?.content).toContain('two bikes');
  });

  it('refuses a reader outside localhost until REMBERO_READER_ALLOW_REMOTE is set', async () => {
    // Nothing here is sensitive by any pattern, so the prompt gate has no reason to
    // fire: the reader's own address is the reason, because masking is best-effort
    // and a cloud reader would see whatever the patterns missed.
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const client = new StubReaderClient('Answer: Two bikes.');
    await expect(
      recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          reader: reader(client, 'https://api.example.com/v1'),
        },
      ),
    ).rejects.toThrow(/REMBERO_READER_ALLOW_REMOTE/);
    expect(client.prompts).toEqual([]);
  });

  it('reads with a reader outside localhost once REMBERO_READER_ALLOW_REMOTE=1', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const client = new StubReaderClient('Answer: Two bikes.');
    process.env.REMBERO_READER_ALLOW_REMOTE = '1';
    try {
      const result = await recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          reader: reader(client, 'https://api.example.com/v1'),
        },
      );
      expect(result.status).toBe('answered');
      expect(result.answer).toBe('Two bikes.');
    } finally {
      delete process.env.REMBERO_READER_ALLOW_REMOTE;
    }
  });

  it('refuses the configured-LLM fallback too, because it is not known to be local', async () => {
    // No REMBERO_READER_* at all, so the reader is `deps.llm` — the cloud model the
    // product is configured with. That is the default, and it is the path that sent a
    // transcript's `.env` dump off the machine.
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const llm = new StubReaderClient('Answer: Two bikes.');
    await expect(
      recallQuestion(
        { store, llm, sessions },
        'How many bikes do I own now?',
        ['default'],
        { answerMode: 'sessions', at: ASKED_AT },
      ),
    ).rejects.toThrow(/REMBERO_READER_ALLOW_REMOTE/);
    expect(llm.prompts).toEqual([]);
  });

  it('shows the reader working and the question kind only to recall_explain', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const reply = 'Notes:\n- 2024-02-27: two bikes\nAnswer: Two bikes.';
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      {
        answerMode: 'sessions',
        at: ASKED_AT,
        explain: true,
        reader: reader(new StubReaderClient(reply)),
      },
    );
    expect(result.answer).toBe('Two bikes.');
    expect(result.readerReply).toBe(reply);
    expect(result.questionKind).toMatchObject({ aggregation: true, update: true });
  });

  it('returns no_evidence without calling the reader when the namespace is empty', async () => {
    const client = new StubReaderClient('Answer: Two bikes.');
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
    );
    expect(result.status).toBe('no_evidence');
    expect(result.sessionsRead ?? []).toEqual([]);
    expect(client.prompts).toEqual([]);
  });

  it('reports unknown when the reader says it does not know', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const client = new StubReaderClient('Notes:\n- nothing relevant\nAnswer: I do not know.');
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
    );
    expect(result.status).toBe('unknown');
    expect(client.prompts).toHaveLength(1);
  });

  it('never passes a reply without an answer line off as an answer', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    // a reader cut off at maxTokens mid-notes, one that reached the marker and stopped,
    // and one that said nothing at all
    const truncated = [
      'Notes:\n- 2024-02-27: the user mentions bikes\n- 2024-01-12: museums',
      'Notes:\n- 2024-02-27: the user mentions bikes\nAnswer:',
      '',
    ];
    for (const reply of truncated) {
      const result = await recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          explain: true,
          reader: reader(new StubReaderClient(reply)),
        },
      );
      expect(result.status).toBe('unknown');
      expect(result.answer).not.toContain('Notes:');
      // the working is still there for recall_explain, so the truncation is visible
      expect(result.readerReply).toBe(reply);
    }
  });

  it('takes the answer off the last real answer line, and nothing around it', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const shapes: Array<{ reply: string; answer: string | null }> = [
      // a marker quoted inside a note is not an answer line
      {
        reply:
          'Notes:\n- the user wrote "Answer: 42 bikes" earlier\n- 2024-01-12: museums',
        answer: null,
      },
      // notes written after the answer stay out of it
      {
        reply: 'Answer: Two bikes.\nNotes:\n- reasoning: subtracted the sold one',
        answer: 'Two bikes.',
      },
      // a line with no letter or digit says nothing
      { reply: 'Notes:\n- nothing useful\nAnswer: .', answer: null },
      // several answer lines: the last real one wins
      {
        reply:
          'Answer: One bike.\nNotes:\n- 2024-02-27: sold the road bike\nAnswer: Two bikes.',
        answer: 'Two bikes.',
      },
      // a block quote is the reader showing an earlier turn, not answering
      {
        reply:
          'Notes:\n- an earlier assistant turn said:\n> Answer: 42 bikes\n- that was stale',
        answer: null,
      },
      // a quoted transcript inside a fence, indented, is not an answer either
      {
        reply:
          'Notes:\n- the transcript contained:\n```\n  Answer: 42 bikes\n```\n- stale',
        answer: null,
      },
      // a quoted line that carries the reader's reasoning is the worst case of all
      {
        reply:
          'Notes:\n- I reasoned:\n> Answer: 42 bikes because he said three minus one',
        answer: null,
      },
      // a real answer line after a quoted one still answers
      {
        reply:
          'Notes:\n- an earlier turn said:\n> Answer: 42 bikes\n- that was stale\nAnswer: Two bikes.',
        answer: 'Two bikes.',
      },
      // bold around the marker, and working tacked onto the answer line
      {
        reply: '**Answer:** 3; notes: I subtracted the sold one',
        answer: '3',
      },
      // the marker is case-sensitive, as the prompt asks for it and finalAnswerLine reads it
      { reply: 'Notes:\n- 2024-02-27: two bikes\nanswer: two', answer: null },
      // a semicolon is not a cut point: an answer may list things
      {
        reply: 'Answer: a road bike; a tourer',
        answer: 'a road bike; a tourer',
      },
      // the prefixes a reader puts in front of its own answer still answer
      { reply: 'Notes:\n- 2024-02-27: sold one\n- Answer: Two bikes.', answer: 'Two bikes.' },
      { reply: 'Notes:\n- 2024-02-27: sold one\n## Answer: Two bikes.', answer: 'Two bikes.' },
      // a quote behind a bullet is still a quote
      {
        reply: 'Notes:\n- the transcript said:\n- > Answer: 42 bikes',
        answer: null,
      },
      // a real answer line after a fenced transcript answers
      {
        reply: 'Notes:\n```\nAnswer: 42 bikes\n```\nAnswer: Two bikes.',
        answer: 'Two bikes.',
      },
    ];
    for (const { reply, answer } of shapes) {
      const result = await recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          reader: reader(new StubReaderClient(reply)),
        },
      );
      if (answer === null) {
        expect(result.status).toBe('unknown');
      } else {
        expect(result.status).toBe('answered');
        expect(result.answer).toBe(answer);
      }
      expect(result.answer).not.toContain('Notes:');
      expect(result.answer).not.toContain('reasoning:');
      expect(result.answer).not.toContain('42 bikes');
    }
  });

  it('reports unknown rather than throwing on a reply too long to return', async () => {
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      {
        answerMode: 'sessions',
        at: ASKED_AT,
        reader: reader(
          new StubReaderClient(`Answer: ${'x'.repeat(MAX_OUTPUT_BYTES + 1)}`),
        ),
      },
    );
    expect(result.status).toBe('unknown');
    expect(result.answer.length).toBeLessThan(200);
  });

  it('refuses a sensitive prompt for a non-local reader, naming the setting', async () => {
    storeSession('keys', '2024-02-20T09:00:00.000Z', [
      { role: 'user', text: 'I rotated the deployment api_key yesterday and filed it in the vault.' },
    ]);
    const client = new StubReaderClient('Answer: yesterday.');
    await expect(
      recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'Is my api_key: hunter2abc the one I rotated?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          reader: reader(client, 'https://api.example.com/v1'),
        },
      ),
    ).rejects.toThrow(/REMBERO_READER_BASE_URL/);
    expect(client.prompts).toEqual([]);
  });

  it('leaves the evidence mode on the Datalog path', async () => {
    store.assert('default', 'owns(user, two_bikes).');
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const llm = new ScriptedLlm(['?- owns(user, What).']);
    const result = await recallQuestion(
      { store, llm, sessions },
      'What do I own?',
      ['default'],
      { answerMode: 'evidence', at: ASKED_AT },
    );
    expect(result.status).toBe('answered');
    expect(result.answerMode).toBe('evidence');
    expect(result.query).toBe('owns(user, What)');
    expect(result.sessionsRead).toBeUndefined();
    expect(llm.calls).toHaveLength(1);
  });
});

describe('the reader settings', () => {
  it('keeps no reader of its own without a base URL', () => {
    expect(readerFromEnv({})).toBeUndefined();
    expect(readerFromEnv({ REMBERO_READER_BASE_URL: '  ' })).toBeUndefined();
  });

  it('fills the defaults and trims the base URL', () => {
    expect(
      readerFromEnv({
        REMBERO_READER_BASE_URL: 'http://127.0.0.1:8084/v1/',
        REMBERO_READER_MODEL: 'reader-v7',
      }),
    ).toEqual({
      baseUrl: 'http://127.0.0.1:8084/v1',
      model: 'reader-v7',
      apiKey: 'local',
      maxTokens: DEFAULT_READER_MAX_TOKENS,
      temperature: 0,
      timeoutMs: 60_000,
    });
  });

  it('refuses a base URL with no model, and an unusable knob', () => {
    expect(() =>
      readerFromEnv({ REMBERO_READER_BASE_URL: 'http://127.0.0.1:8084/v1' }),
    ).toThrow(/REMBERO_READER_MODEL/);
    expect(() =>
      readerFromEnv({
        REMBERO_READER_BASE_URL: 'http://127.0.0.1:8084/v1',
        REMBERO_READER_MODEL: 'reader-v7',
        REMBERO_READER_MAX_TOKENS: '99999',
      }),
    ).toThrow(/REMBERO_READER_MAX_TOKENS/);
  });

  it('accepts sessions as the configured answer mode', () => {
    expect(
      recallAnswerModeFromEnv({ REMBERO_RECALL_ANSWER_MODE: 'sessions' }),
    ).toBe('sessions');
  });
});

/** A candidate as the product hands it to the ranking: an id, a date and its turns. */
function candidate(id: string, date: string, turns: number) {
  return {
    id,
    date,
    turns: Array.from({ length: turns }, (_unused, index) => ({
      role: 'user' as const,
      text: `${id} turn ${index}`,
    })),
  };
}

describe('the reading candidate cap', () => {
  it('leaves a store under both caps exactly as it was', () => {
    const loaded = [
      candidate('a', '2024-01-01T09:00:00.000Z', 3),
      candidate('b', '2024-02-01T09:00:00.000Z', 4),
    ];
    expect(cappedReadingCandidates(loaded)).toEqual(loaded);
  });

  it('keeps the most recent sessions, in the order it was given them', () => {
    const loaded = [
      candidate('oldest', '2024-01-01T09:00:00.000Z', 1),
      candidate('newest', '2024-03-01T09:00:00.000Z', 1),
      candidate('middle', '2024-02-01T09:00:00.000Z', 1),
    ];
    // most recent two, but still in the caller's order, so a store under the cap and
    // one just over it rank the same sessions in the same sequence
    expect(cappedReadingCandidates(loaded, { sessions: 2 }).map(({ id }) => id)).toEqual([
      'newest',
      'middle',
    ]);
  });

  it('spends the turn budget newest first, keeping the newest turns of a long session', () => {
    const loaded = [
      candidate('old', '2024-01-01T09:00:00.000Z', 5),
      candidate('new', '2024-03-01T09:00:00.000Z', 3),
    ];
    const capped = cappedReadingCandidates(loaded, { turns: 5 });
    expect(capped.map(({ id }) => id)).toEqual(['old', 'new']);
    // 'new' takes 3 of the 5, and 'old' contributes its last 2 turns
    expect(capped[0]?.turns.map(({ text }) => text)).toEqual([
      'old turn 3',
      'old turn 4',
    ]);
    expect(capped[1]?.turns).toHaveLength(3);
  });

  it('drops a session entirely once the turn budget is spent', () => {
    const loaded = [
      candidate('old', '2024-01-01T09:00:00.000Z', 4),
      candidate('new', '2024-03-01T09:00:00.000Z', 4),
    ];
    expect(
      cappedReadingCandidates(loaded, { turns: 4 }).map(({ id }) => id),
    ).toEqual(['new']);
  });

  it('answers over 100,020 turns, where the clause limit used to reach the user', () => {
    // The measured break: 100,020 non-empty turns made searchKnowledge throw
    // "knowledge search exceeds 100000 clauses" in 692 ms, which unsearchableQuestion
    // does not match, so every sessions recall failed until the sessions were deleted.
    // its own store, because the shared one caps at 1 MB and would evict as we write
    const bulk = new SessionStore({
      root: mkdtempSync(join(tmpdir(), 'rembero-sessions-recall-bulk-')),
      capBytes: 400 * 1024 * 1024,
      log: () => {},
    });
    for (let index = 0; index < 30; index += 1) {
      const startedAt = new Date(Date.UTC(2024, 0, 1 + index, 9)).toISOString();
      bulk.appendTurns(
        'default',
        {
          version: 1,
          source: 'import',
          sourceSessionId: `bulk-${index}`,
          startedAt,
        },
        Array.from({ length: 3_334 }, (_unused, turn) => ({
          role: (turn % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
          ts: new Date(Date.parse(startedAt) + turn * 1_000).toISOString(),
          text: `session ${index} turn ${turn}: I rode my bike to the market and bought pears`,
        })),
      );
    }
    const stored = bulk
      .list('default')
      .reduce((total, entry) => total + entry.turns, 0);
    expect(stored).toBe(100_020);
    return expect(
      recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions: bulk },
        'How many bikes do I own now?',
        ['default'],
        {
          answerMode: 'sessions',
          at: ASKED_AT,
          reader: reader(new StubReaderClient('Answer: Two bikes.')),
        },
      ),
    ).resolves.toMatchObject({ status: 'answered', answer: 'Two bikes.' });
  });

  it('caps below the clause limit that used to break every sessions recall', () => {
    // retrieveSessions indexes one clause per non-empty turn, and searchKnowledge
    // refuses more than MAX_KNOWLEDGE_SEARCH_CLAUSES of them with a message
    // unsearchableQuestion does not match, so the throw reached the user.
    expect(MAX_READING_CANDIDATE_TURNS).toBeLessThan(MAX_KNOWLEDGE_SEARCH_CLAUSES);
    // the session unit indexes one clause per session, so that count is bounded too
    expect(MAX_READING_CANDIDATE_SESSIONS).toBeLessThan(MAX_KNOWLEDGE_SEARCH_CLAUSES);
  });
});

/** Three sessions long enough that the reading budget, not the source window, decides. */
function storeLongSessions(): void {
  for (const [index, startedAt] of [
    '2024-02-25T09:00:00.000Z',
    '2024-02-26T09:00:00.000Z',
    '2024-02-27T09:00:00.000Z',
  ].entries()) {
    storeSession(`long-${index}`, startedAt, [
      {
        role: 'user',
        text: `I own two bikes. ${'pears and bikes and markets. '.repeat(3_000)}`,
      },
    ]);
  }
}

describe('the product reading budget', () => {
  it('defaults to the 24576 bytes reader v7 was trained on', () => {
    expect(DEFAULT_PRODUCT_READING_CONTEXT_BYTES).toBe(24_576);
    expect(readingContextBytesFromEnv({})).toBe(24_576);
    expect(
      readingContextBytesFromEnv({ REMBERO_READING_CONTEXT_BYTES: '8192' }),
    ).toBe(8_192);
  });

  it('agrees with the bounds the shared retrieval module enforces', () => {
    expect(MAX_CONFIGURED_READING_CONTEXT_BYTES).toBe(MAX_READING_CONTEXT_BYTES);
    expect(MIN_READING_CONTEXT_BYTES).toBe(4_096);
  });

  it('refuses a budget that is not an integer in range, naming the setting', () => {
    for (const configured of ['', 'lots', '24k', '-1', '1024', '1048576']) {
      expect(
        () => readingContextBytesFromEnv({ REMBERO_READING_CONTEXT_BYTES: configured }),
        configured,
      ).toThrow(/REMBERO_READING_CONTEXT_BYTES/);
    }
  });

  it('spends the default budget on the history and not the old 56 KB', async () => {
    // three long sessions, so the even split across them is what the budget decides:
    // at 56 KB each would render its whole 16 KB source window, at 24576 they share it
    storeLongSessions();
    const client = new StubReaderClient('Answer: Two bikes.');
    const result = await recallQuestion(
      { store, llm: new NeverCalledLlm(), sessions },
      'How many bikes do I own now?',
      ['default'],
      { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
    );
    expect(result.status).toBe('answered');
    const prompt = client.prompts[0]?.at(-1)?.content ?? '';
    const bytes = Buffer.byteLength(prompt, 'utf8');
    expect(bytes).toBeGreaterThan(20_000);
    expect(bytes).toBeLessThan(32_000);
  });

  it('honours a smaller configured budget', async () => {
    storeLongSessions();
    const client = new StubReaderClient('Answer: Two bikes.');
    process.env.REMBERO_READING_CONTEXT_BYTES = '8192';
    try {
      const result = await recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
      );
      expect(result.status).toBe('answered');
      expect(
        Buffer.byteLength(client.prompts[0]?.at(-1)?.content ?? '', 'utf8'),
      ).toBeLessThan(12_000);
    } finally {
      delete process.env.REMBERO_READING_CONTEXT_BYTES;
    }
  });

  it('degrades a retrieval failure to no_evidence with the reason recorded', async () => {
    // Retrieval used to sit outside the degrade path loadSessions has, so anything it
    // threw reached the user. A budget this build cannot read is the reachable case.
    storeSession('bikes', '2024-02-27T09:00:00.000Z', [
      { role: 'user', text: 'I now own two bikes after selling the old road bike.' },
    ]);
    const client = new StubReaderClient('Answer: Two bikes.');
    process.env.REMBERO_READING_CONTEXT_BYTES = 'twenty-four kilobytes';
    try {
      const result = await recallQuestion(
        { store, llm: new NeverCalledLlm(), sessions },
        'How many bikes do I own now?',
        ['default'],
        { answerMode: 'sessions', at: ASKED_AT, reader: reader(client) },
      );
      expect(result.status).toBe('no_evidence');
      expect(result.sessionsError).toMatch(/REMBERO_READING_CONTEXT_BYTES/);
      expect(result.sessionsRead ?? []).toEqual([]);
      expect(client.prompts).toEqual([]);
    } finally {
      delete process.env.REMBERO_READING_CONTEXT_BYTES;
    }
  });
});
