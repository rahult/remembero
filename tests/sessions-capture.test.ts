import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  autoCaptureClaudeStop,
  type AutoCaptureOptions,
} from '../src/autocapture/capture.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { rememberText } from '../src/llm/pipeline.js';
import { SessionStore } from '../src/sessions/store.js';
import { MemoryStore } from '../src/store/store.js';

class ScriptedLlm implements LlmClient {
  calls: ChatMessage[][] = [];

  constructor(private readonly responses: string[]) {}

  async complete(messages: ChatMessage[]): Promise<string> {
    this.calls.push(messages);
    const response = this.responses.shift();
    if (response === undefined) throw new Error('ScriptedLlm ran out of responses');
    return response;
  }
}

const CAPTURE_AT = new Date('2026-08-17T02:00:00.000Z');

let root: string;
let claudeConfigDir: string;
let transcriptPath: string;
let sessionsDir: string;
let store: MemoryStore;
let sessions: SessionStore;
let previousSessionsSetting: string | undefined;

function transcriptLine(
  role: 'user' | 'assistant',
  content: unknown,
  timestamp?: string,
): string {
  return JSON.stringify({
    type: role,
    message: { role, content },
    ...(timestamp === undefined ? {} : { timestamp }),
  });
}

function writeTranscript(lines: string[]): void {
  writeFileSync(transcriptPath, `${lines.join('\n')}\n`, 'utf8');
}

function stopInput(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    session_id: 'session-1',
    transcript_path: transcriptPath,
    cwd: root,
    hook_event_name: 'Stop',
    stop_hook_active: false,
    last_assistant_message: 'Understood.',
    ...overrides,
  });
}

function captureOptions(
  overrides: Partial<AutoCaptureOptions> = {},
): AutoCaptureOptions {
  return {
    namespace: 'default',
    dailyCap: 5,
    tailBytes: 16 * 1024,
    claudeConfigDir,
    now: CAPTURE_AT,
    ...overrides,
  };
}

/** The stored turns of the one session the Claude Code stop hook writes. */
function storedTurns(): ReturnType<SessionStore['readSession']> {
  return sessions.readSession(
    'default',
    sessions.sessionKey('claude-code', 'session-1'),
  );
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'rembero-sessions-capture-'));
  claudeConfigDir = join(root, '.claude');
  const projectDir = join(claudeConfigDir, 'projects', 'project');
  transcriptPath = join(projectDir, 'session-1.jsonl');
  mkdirSync(projectDir, { recursive: true });
  store = new MemoryStore(join(root, 'memory'));
  sessionsDir = join(root, 'sessions');
  sessions = new SessionStore({ root: sessionsDir, capBytes: 1024 * 1024 });
  previousSessionsSetting = process.env.REMBERO_SESSIONS;
  process.env.REMBERO_SESSIONS = 'on';
});

afterEach(() => {
  if (previousSessionsSetting === undefined) {
    delete process.env.REMBERO_SESSIONS;
  } else {
    process.env.REMBERO_SESSIONS = previousSessionsSetting;
  }
});

describe('capture writes conversation turns to the session store', () => {
  it('appends both roles of the transcript tail with transcript timestamps', async () => {
    writeTranscript([
      transcriptLine(
        'user',
        'I prefer dark mode and live in Melbourne.',
        '2026-08-17T01:58:00.000Z',
      ),
      transcriptLine('assistant', 'I will remember those preferences.'),
    ]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);

    const result = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    expect(result.status).toBe('captured');
    expect(result.sessionTurns).toEqual({ appended: 2, skipped: 0 });
    const session = storedTurns()!;
    expect(session.header).toMatchObject({
      version: 1,
      source: 'claude-code',
      sourceSessionId: 'session-1',
      startedAt: '2026-08-17T01:58:00.000Z',
      cwd: root,
    });
    expect(session.turns.map((turn) => [turn.index, turn.role, turn.ts])).toEqual([
      [0, 'user', '2026-08-17T01:58:00.000Z'],
      // No timestamp on the entry, so the capture time stands in.
      [1, 'assistant', CAPTURE_AT.toISOString()],
    ]);
    expect(session.turns[0].text).toBe('I prefer dark mode and live in Melbourne.');
    expect(session.turns[1].text).toBe('I will remember those preferences.');
  });

  it('never stores a turn twice when stop hooks re-read an overlapping tail', async () => {
    const first = transcriptLine('user', 'I prefer dark mode.');
    const second = transcriptLine('assistant', 'Noted.');
    writeTranscript([first, second]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).',
      'lives_in(user, melbourne).',
    ]);

    await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );
    // The same tail again: the fact store refuses it as a duplicate capture, and
    // nothing new reaches the session either.
    const repeat = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );
    expect(repeat.status).toBe('skipped');
    expect(storedTurns()!.turns).toHaveLength(2);

    // A grown transcript: the two stored turns are skipped by hash, the new one appended.
    writeTranscript([first, second, transcriptLine('user', 'I live in Melbourne.')]);
    const grown = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );
    expect(grown.sessionTurns).toEqual({ appended: 1, skipped: 2 });
    expect(storedTurns()!.turns.map((turn) => turn.text)).toEqual([
      'I prefer dark mode.',
      'Noted.',
      'I live in Melbourne.',
    ]);
  });

  it('lets the stop hook succeed when the session store cannot be written', async () => {
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);
    // A live namespace lock this process will never release: appendTurns waits its
    // bounded 250 ms and then fails fast.
    mkdirSync(sessionsDir, { recursive: true });
    writeFileSync(
      join(sessionsDir, '.session-default.lock'),
      `${JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() })}\n`,
      'utf8',
    );

    const result = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    expect(result.status).toBe('captured');
    expect(result.added).toEqual(['prefers_theme(user, dark).']);
    expect(result.sessionTurns).toBeUndefined();
    expect(result.sessionError).toMatch(/session lock/i);
    expect(storedTurns()).toBeUndefined();
  });

  it('stores nothing without a session store, or with REMBERO_SESSIONS off', async () => {
    writeTranscript([
      transcriptLine('user', 'I prefer dark mode and live in Melbourne.'),
    ]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).',
      'lives_in(user, melbourne).',
    ]);

    const result = await autoCaptureClaudeStop(
      { store, llm },
      stopInput(),
      captureOptions(),
    );
    expect(result).toMatchObject({
      status: 'captured',
      added: ['prefers_theme(user, dark).'],
    });
    expect(result.sessionTurns).toBeUndefined();
    expect(result.sessionError).toBeUndefined();
    expect(existsSync(sessionsDir)).toBe(false);

    process.env.REMBERO_SESSIONS = 'off';
    writeTranscript([transcriptLine('user', 'I live in Melbourne.')]);
    const off = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );
    expect(off.status).toBe('captured');
    expect(off.sessionTurns).toBeUndefined();
    expect(existsSync(sessionsDir)).toBe(false);
  });
});

describe('remember writes a one-turn session', () => {
  it('stores the remembered text as a user turn under source remember', async () => {
    const llm = new ScriptedLlm(['owns(user, bike).']);

    const result = await rememberText(
      { store, llm, sessions },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );

    expect(result.added).toEqual(['owns(user, bike).']);
    const entries = sessions.list('default');
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({ source: 'remember', turns: 1 });
    const session = sessions.readSession('default', entries[0].key)!;
    expect(session.header).toMatchObject({
      source: 'remember',
      startedAt: CAPTURE_AT.toISOString(),
    });
    expect(session.header.sourceSessionId).toMatch(/^[0-9a-f]{32}$/);
    expect(session.turns).toHaveLength(1);
    expect(session.turns[0]).toMatchObject({
      index: 0,
      role: 'user',
      ts: CAPTURE_AT.toISOString(),
      text: 'I bought a road bike',
    });

    // The same text twice is one session with one turn: the turn hash is stored.
    await rememberText(
      { store, llm: new ScriptedLlm(['owns(user, bike).']), sessions },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );
    expect(sessions.list('default')).toHaveLength(1);
    expect(sessions.readSession('default', entries[0].key)!.turns).toHaveLength(1);
  });

  it('stores nothing without a session store, or with REMBERO_SESSIONS off', async () => {
    await rememberText(
      { store, llm: new ScriptedLlm(['owns(user, bike).']) },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );
    expect(existsSync(sessionsDir)).toBe(false);

    process.env.REMBERO_SESSIONS = 'off';
    await rememberText(
      { store, llm: new ScriptedLlm(['owns(user, canoe).']), sessions },
      'I bought a canoe',
      'default',
      { at: CAPTURE_AT },
    );
    expect(existsSync(sessionsDir)).toBe(false);
  });

  it('answers even when the session write fails', async () => {
    mkdirSync(sessionsDir, { recursive: true });
    writeFileSync(
      join(sessionsDir, '.session-default.lock'),
      `${JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() })}\n`,
      'utf8',
    );

    const result = await rememberText(
      { store, llm: new ScriptedLlm(['owns(user, bike).']), sessions },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );

    expect(result.added).toEqual(['owns(user, bike).']);
    expect(
      readdirSync(sessionsDir).filter((name) => name.endsWith('.jsonl')),
    ).toEqual([]);
  });
});
