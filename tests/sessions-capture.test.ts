import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  autoCaptureClaudeStop,
  type AutoCaptureOptions,
} from '../src/autocapture/capture.js';
import { transcriptWindowBytes } from '../src/autocapture/transcript.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { createServer } from '../src/mcp/server.js';
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
    // That repeat is refused by the fact store before the session write, so the
    // skipping itself is asserted directly: the same two turns handed to the store
    // again are both recognised and neither is stored twice.
    expect(
      sessions.appendTurns(
        'default',
        {
          version: 1,
          source: 'claude-code',
          sourceSessionId: 'session-1',
          startedAt: CAPTURE_AT.toISOString(),
        },
        [
          {
            role: 'user',
            ts: CAPTURE_AT.toISOString(),
            text: 'I prefer dark mode.',
          },
          { role: 'assistant', ts: CAPTURE_AT.toISOString(), text: 'Noted.' },
        ],
      ),
    ).toMatchObject({ appended: 0, skipped: 2 });
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

  it('keeps a short answer the user repeats inside one window', async () => {
    // "ok" twice in one conversation is two events. Deduplicating within the batch
    // would drop the second and shift every later turn's index, which is exactly
    // the ordering the reader depends on; only turns already stored are skipped.
    writeTranscript([
      transcriptLine('user', 'ok', '2026-08-17T01:50:00.000Z'),
      transcriptLine('assistant', 'Shall I carry on?', '2026-08-17T01:51:00.000Z'),
      transcriptLine('user', 'ok', '2026-08-17T01:52:00.000Z'),
      transcriptLine('assistant', 'Done.', '2026-08-17T01:53:00.000Z'),
    ]);
    // Nothing memorable is said, so extraction may ask more than once; the session
    // write is what this test is about.
    const llm = new ScriptedLlm(['', '', '']);

    const result = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput({ last_assistant_message: 'Done.' }),
      captureOptions(),
    );

    expect(result.sessionTurns).toEqual({ appended: 4, skipped: 0 });
    expect(storedTurns()!.turns.map((turn) => [turn.index, turn.text])).toEqual([
      [0, 'ok'],
      [1, 'Shall I carry on?'],
      [2, 'ok'],
      [3, 'Done.'],
    ]);
  });

  it('never appends a turn older than the ones already stored', async () => {
    // A window with no user text is re-read four times further back, so a capture
    // can see turns older than the stored ones. Appending those would give a newer
    // index to an older ts and the file would stop being in time order.
    writeTranscript([
      transcriptLine('user', 'I prefer dark mode.', '2026-08-17T02:00:00.000Z'),
      transcriptLine('assistant', 'Noted.', '2026-08-17T02:01:00.000Z'),
    ]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).',
      'lives_in(user, melbourne).',
    ]);
    await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    writeTranscript([
      // Older than the stored turns, and never stored: dropped, not appended.
      transcriptLine('user', 'I set this up last week.', '2026-08-17T01:00:00.000Z'),
      transcriptLine('user', 'I live in Melbourne.', '2026-08-17T02:02:00.000Z'),
    ]);
    const later = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    expect(later.sessionTurns).toEqual({ appended: 1, skipped: 0 });
    expect(storedTurns()!.turns.map((turn) => [turn.index, turn.ts])).toEqual([
      [0, '2026-08-17T02:00:00.000Z'],
      [1, '2026-08-17T02:01:00.000Z'],
      [2, '2026-08-17T02:02:00.000Z'],
    ]);
    expect(storedTurns()!.turns.map((turn) => turn.text)).not.toContain(
      'I set this up last week.',
    );
  });

  it('keeps a new turn beside an assistant answer larger than tailBytes', async () => {
    // A code-heavy assistant answer is routinely bigger than the extraction tail's
    // budget, and `tail.turns` is untruncated. Spending the budget on it would drop
    // the user's message in the same batch, which the ts floor would then bar for
    // good.
    writeTranscript([
      transcriptLine('user', 'I prefer dark mode.', '2026-08-17T01:00:00.000Z'),
      transcriptLine('assistant', 'Noted.', '2026-08-17T01:01:00.000Z'),
    ]);
    const llm = new ScriptedLlm([
      'prefers_theme(user, dark).',
      'lives_in(user, melbourne).',
    ]);
    await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions({ tailBytes: 3000 }),
    );

    writeTranscript([
      transcriptLine(
        'user',
        'remember I moved to Melbourne',
        '2026-08-17T02:05:00.000Z',
      ),
      transcriptLine('assistant', 'x'.repeat(4000), '2026-08-17T02:06:00.000Z'),
    ]);
    const second = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions({ tailBytes: 3000 }),
    );

    expect(second.sessionTurns).toEqual({ appended: 2, skipped: 0 });
    expect(storedTurns()!.turns.map((turn) => turn.ts)).toEqual([
      '2026-08-17T01:00:00.000Z',
      '2026-08-17T01:01:00.000Z',
      '2026-08-17T02:05:00.000Z',
      '2026-08-17T02:06:00.000Z',
    ]);
    expect(storedTurns()!.turns[2].text).toBe('remember I moved to Melbourne');
  });

  it('bounds a first capture to the read window, and still dates it from the start', async () => {
    // The store masks and hashes every turn it is handed, inside the namespace
    // lock, so a window that widened backwards past its budget must not reach it
    // whole. Only a widened window can: everything an unwidened one holds fits the
    // budget by construction, which is why a sliding session loses nothing.
    const budget = transcriptWindowBytes(3000);
    const turnText = (marker: string) => `${marker} ${'x'.repeat(2048)}`;
    const minute = (index: number) =>
      `2026-08-17T01:${String(index).padStart(2, '0')}:00.000Z`;
    // The only user turn is at the very start, so the base 64 KB window holds no
    // user text and the scan widens to the whole file.
    writeTranscript([
      transcriptLine('user', turnText('oldest'), minute(0)),
      ...Array.from({ length: 45 }, (_, index) =>
        transcriptLine('assistant', turnText(`answer ${index}`), minute(index + 1)),
      ),
    ]);
    const llm = new ScriptedLlm(['% nothing']);

    const result = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions({ tailBytes: 3000 }),
    );

    expect(result.status).toBe('empty');
    const turns = storedTurns()!.turns;
    expect(turns.length).toBeGreaterThan(1);
    expect(turns.length).toBeLessThan(46);
    expect(
      turns.reduce((sum, turn) => sum + Buffer.byteLength(turn.text, 'utf8'), 0),
    ).toBeLessThanOrEqual(budget);
    // The newest turns, contiguous and in ts order, the oldest ones dropped.
    expect(turns.at(-1)!.text.startsWith('answer 44')).toBe(true);
    expect(turns.map((turn) => turn.ts)).toEqual(
      [...turns].map((turn) => turn.ts).sort(),
    );
    expect(turns.some((turn) => turn.text.startsWith('oldest'))).toBe(false);
    // The dropped turns still date the session: it did not begin where the budget
    // happened to start keeping.
    expect(storedTurns()!.header.startedAt).toBe(minute(0));
  });

  it('stores a batch in ts order even when the transcript is not', async () => {
    writeTranscript([
      transcriptLine('assistant', 'Noted.', '2026-08-17T02:05:00.000Z'),
      transcriptLine('user', 'I prefer dark mode.', '2026-08-17T02:04:00.000Z'),
    ]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);

    await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    expect(storedTurns()!.turns.map((turn) => [turn.index, turn.ts])).toEqual([
      [0, '2026-08-17T02:04:00.000Z'],
      [1, '2026-08-17T02:05:00.000Z'],
    ]);
    expect(storedTurns()!.header.startedAt).toBe('2026-08-17T02:04:00.000Z');
  });

  it('keeps the capture when REMBERO_SESSIONS is neither on nor off', async () => {
    // sessionsEnabledFromEnv throws on any other value, and that must not reach
    // the stop hook: the facts are already committed by then.
    process.env.REMBERO_SESSIONS = 'true';
    writeTranscript([transcriptLine('user', 'I prefer dark mode.')]);
    const llm = new ScriptedLlm(['prefers_theme(user, dark).']);

    const result = await autoCaptureClaudeStop(
      { store, llm, sessions },
      stopInput(),
      captureOptions(),
    );

    expect(result.status).toBe('captured');
    expect(result.added).toEqual(['prefers_theme(user, dark).']);
    expect(result.sessionError).toMatch(/REMBERO_SESSIONS/);
    expect(result.sessionTurns).toBeUndefined();
    expect(existsSync(sessionsDir)).toBe(false);
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

describe('a bad sessions setting costs sessions and nothing else', () => {
  it('starts the MCP server anyway', () => {
    process.env.REMBERO_SESSIONS = 'true';
    expect(() =>
      createServer({ store, llm: new ScriptedLlm([]) }),
    ).not.toThrow();

    process.env.REMBERO_SESSIONS = 'on';
    process.env.REMBERO_SESSION_CAP_BYTES = 'lots';
    try {
      expect(() =>
        createServer({ store, llm: new ScriptedLlm([]) }),
      ).not.toThrow();
    } finally {
      delete process.env.REMBERO_SESSION_CAP_BYTES;
    }
  });

  it('runs an unrelated CLI command anyway', () => {
    const home = join(root, 'cli-home');
    const query = (env: Record<string, string>) =>
      spawnSync(process.execPath, [resolve('dist/cli.js'), 'query', 'pet(a, B)'], {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home, ...env },
      });

    const unreadable = query({ REMBERO_SESSIONS: 'true' });
    expect(unreadable.status).toBe(0);
    expect(unreadable.stderr).toMatch(/REMBERO_SESSIONS/);

    const badCap = query({
      REMBERO_SESSIONS: 'on',
      REMBERO_SESSION_CAP_BYTES: 'lots',
    });
    expect(badCap.status).toBe(0);
    expect(badCap.stderr).toMatch(/REMBERO_SESSION_CAP_BYTES/);
    expect(existsSync(join(home, 'sessions'))).toBe(false);
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

  it('stores the turn even when extraction finds no fact to keep', async () => {
    const result = await rememberText(
      { store, llm: new ScriptedLlm(['% nothing']), sessions },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );

    expect(result.added).toEqual([]);
    const entries = sessions.list('default');
    expect(entries).toHaveLength(1);
    expect(sessions.readSession('default', entries[0].key)!.turns).toHaveLength(1);
  });

  it('keeps nothing on disk for a remember the product refused', async () => {
    // The namespace allowlist and the sensitive-text refusal both throw out of
    // extraction; text the product would not process is text it must not store.
    await expect(
      rememberText(
        {
          store,
          llm: new ScriptedLlm(['owns(user, bike).']),
          sessions,
          llmAllowedNamespaces: new Set(['other']),
        },
        'I bought a road bike',
        'default',
        { at: CAPTURE_AT },
      ),
    ).rejects.toThrow(/local-only/i);
    expect(existsSync(sessionsDir)).toBe(false);
  });

  it('answers when REMBERO_SESSIONS is neither on nor off', async () => {
    process.env.REMBERO_SESSIONS = 'true';

    const result = await rememberText(
      { store, llm: new ScriptedLlm(['owns(user, bike).']), sessions },
      'I bought a road bike',
      'default',
      { at: CAPTURE_AT },
    );

    expect(result.added).toEqual(['owns(user, bike).']);
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
