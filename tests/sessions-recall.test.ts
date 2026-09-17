import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { recallQuestion, type RecallReader } from '../src/llm/pipeline.js';
import {
  DEFAULT_READER_MAX_TOKENS,
  readerFromEnv,
  recallAnswerModeFromEnv,
} from '../src/env.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { MemoryStore } from '../src/store/store.js';
import { SessionStore, type SessionSource } from '../src/sessions/store.js';

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
    expect(read[0]?.key).toBe(bikes);
    expect(read[0]?.date).toBe('2024-02-27');
    expect(read[0]?.excerpts.join(' ')).toContain('two bikes');
    expect(client.prompts).toHaveLength(1);
    expect(client.prompts[0]?.at(-1)?.content).toContain('two bikes');
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
