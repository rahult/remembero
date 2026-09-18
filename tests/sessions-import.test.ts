import { spawnSync } from 'node:child_process';
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { importClaudeTranscript } from '../src/sessions/import.js';
import { SessionStore } from '../src/sessions/store.js';

/** The `sessionId` Claude Code stamps on every line of a real transcript. */
const SESSION_ID = 'a1b2c3d4-past-chat';
/** The transcript's own timestamp; the second line carries none on purpose. */
const FIRST_TS = '2026-09-17T04:05:06.000Z';
/** The file's modification time, which the timestamp-less turn must inherit. */
const FILE_MTIME = new Date('2026-09-18T07:00:00.000Z');

let root: string;
let transcriptPath: string;
let sessions: SessionStore;
let previousSessionsSetting: string | undefined;

function transcriptLine(
  role: 'user' | 'assistant',
  content: string,
  options: { timestamp?: string; sessionId?: string } = {},
): string {
  return JSON.stringify({
    type: role,
    message: { role, content },
    ...(options.timestamp === undefined ? {} : { timestamp: options.timestamp }),
    ...(options.sessionId === undefined
      ? {}
      : { sessionId: options.sessionId }),
  });
}

function writeTranscript(path: string, lines: string[]): void {
  writeFileSync(path, `${lines.join('\n')}\n`, 'utf8');
  utimesSync(path, FILE_MTIME, FILE_MTIME);
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'rembero-sessions-import-'));
  transcriptPath = join(root, 'past-chat.jsonl');
  writeTranscript(transcriptPath, [
    transcriptLine('user', 'I moved to Melbourne in March.', {
      timestamp: FIRST_TS,
      sessionId: SESSION_ID,
    }),
    transcriptLine('assistant', 'Noted, Melbourne since March.', {
      sessionId: SESSION_ID,
    }),
  ]);
  sessions = new SessionStore({
    root: join(root, 'sessions'),
    capBytes: 1024 * 1024,
  });
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

describe('importClaudeTranscript', () => {
  it('stores both turns of a transcript file as one imported session', () => {
    const result = importClaudeTranscript(sessions, 'scratch', transcriptPath);

    expect(result).toEqual({
      key: sessions.sessionKey('import', SESSION_ID),
      appended: 2,
      skipped: 0,
    });
    const stored = sessions.readSession('scratch', result.key)!;
    expect(stored.header).toMatchObject({
      version: 1,
      source: 'import',
      sourceSessionId: SESSION_ID,
      startedAt: FIRST_TS,
    });
    expect(stored.turns.map((turn) => [turn.index, turn.role, turn.text])).toEqual([
      [0, 'user', 'I moved to Melbourne in March.'],
      [1, 'assistant', 'Noted, Melbourne since March.'],
    ]);
    // Its own timestamp when it has one, the file's modification time otherwise,
    // and every one of them a timestamp the reader can subtract dates from.
    expect(stored.turns[0].ts).toBe(FIRST_TS);
    expect(stored.turns[1].ts).toBe(FILE_MTIME.toISOString());
    for (const turn of stored.turns) {
      expect(Number.isNaN(Date.parse(turn.ts))).toBe(false);
    }
  });

  it('appends nothing when the same file is imported twice', () => {
    importClaudeTranscript(sessions, 'scratch', transcriptPath);
    const again = importClaudeTranscript(sessions, 'scratch', transcriptPath);

    expect(again).toMatchObject({ appended: 0, skipped: 2 });
    const stored = sessions.readSession('scratch', again.key)!;
    expect(stored.turns).toHaveLength(2);
    expect(sessions.list('scratch')).toHaveLength(1);
  });

  it('reads the whole file rather than a tail', () => {
    const long = join(root, 'long-chat.jsonl');
    const lines: string[] = [];
    for (let index = 0; index < 400; index += 1) {
      lines.push(
        transcriptLine('user', `turn ${index} ${'x'.repeat(200)}`, {
          sessionId: 'long-chat-session',
        }),
      );
    }
    writeTranscript(long, lines);

    const result = importClaudeTranscript(sessions, 'scratch', long);

    expect(result.appended).toBe(400);
    const stored = sessions.readSession('scratch', result.key)!;
    expect(stored.turns[0].text).toContain('turn 0 ');
  });

  it('keeps a turn the conversation repeats, and its place in the order', () => {
    const repeated = join(root, 'repeated.jsonl');
    writeTranscript(repeated, [
      transcriptLine('user', 'ok', { sessionId: 'repeated-session' }),
      transcriptLine('assistant', 'sure', { sessionId: 'repeated-session' }),
      transcriptLine('user', 'ok', { sessionId: 'repeated-session' }),
      transcriptLine('assistant', 'done', { sessionId: 'repeated-session' }),
    ]);

    const result = importClaudeTranscript(sessions, 'scratch', repeated);

    expect(result).toMatchObject({ appended: 4, skipped: 0 });
    expect(
      sessions.readSession('scratch', result.key)!.turns.map((turn) => turn.text),
    ).toEqual(['ok', 'sure', 'ok', 'done']);
    // Re-reading the same conversation still stores each of its turns once.
    expect(
      importClaudeTranscript(sessions, 'scratch', repeated),
    ).toMatchObject({ appended: 0, skipped: 4 });
    expect(sessions.readSession('scratch', result.key)!.turns).toHaveLength(4);
  });

  it('keeps two conversations that share a file name apart', () => {
    const conversations = ['a', 'b'].map((name) => {
      const dir = join(root, name);
      mkdirSync(dir);
      const path = join(dir, 'chat.jsonl');
      writeTranscript(path, [
        transcriptLine('user', `conversation ${name} opened`, {
          sessionId: `session-${name}`,
        }),
        transcriptLine('assistant', `that was ${name}`, {
          sessionId: `session-${name}`,
        }),
      ]);
      return importClaudeTranscript(sessions, 'scratch', path);
    });

    expect(conversations[0].key).not.toBe(conversations[1].key);
    expect(sessions.list('scratch')).toHaveLength(2);
    for (const imported of conversations) {
      expect(imported).toMatchObject({ appended: 2, skipped: 0 });
    }
  });

  it('keeps two same-named transcripts apart with no session id to go on', () => {
    const keys = ['a', 'b'].map((name) => {
      const dir = join(root, `no-id-${name}`);
      mkdirSync(dir);
      const path = join(dir, 'chat.jsonl');
      writeTranscript(path, [
        transcriptLine('user', `conversation ${name} opened`),
        transcriptLine('assistant', `that was ${name}`),
      ]);
      const result = importClaudeTranscript(sessions, 'scratch', path);
      expect(result).toMatchObject({ appended: 2, skipped: 0 });
      return result.key;
    });

    expect(keys[0]).not.toBe(keys[1]);
    expect(sessions.list('scratch')).toHaveLength(2);
  });

  it('is one session when the same conversation arrives under two file names', () => {
    const copy = join(root, 'renamed-copy.jsonl');
    writeTranscript(copy, [
      transcriptLine('user', 'I moved to Melbourne in March.', {
        timestamp: FIRST_TS,
        sessionId: SESSION_ID,
      }),
      transcriptLine('assistant', 'Noted, Melbourne since March.', {
        sessionId: SESSION_ID,
      }),
      transcriptLine('user', 'And I kept the old flat.', {
        sessionId: SESSION_ID,
      }),
    ]);

    const first = importClaudeTranscript(sessions, 'scratch', transcriptPath);
    const second = importClaudeTranscript(sessions, 'scratch', copy);

    expect(second.key).toBe(first.key);
    expect(second).toMatchObject({ appended: 1, skipped: 2 });
    expect(sessions.list('scratch')).toHaveLength(1);
  });

  it('appends to the same session when a transcript has grown since the import', () => {
    const growing = join(root, 'growing.jsonl');
    const opening = [
      transcriptLine('user', 'day one'),
      transcriptLine('assistant', 'noted'),
    ];
    writeTranscript(growing, opening);

    const first = importClaudeTranscript(sessions, 'scratch', growing);
    appendFileSync(
      growing,
      `${transcriptLine('user', 'day two')}\n${transcriptLine('assistant', 'noted again')}\n`,
      'utf8',
    );
    const second = importClaudeTranscript(sessions, 'scratch', growing);

    // No session id to go on, so identity rests on the opening turn, which the new
    // conversation did not move.
    expect(second.key).toBe(first.key);
    expect(second).toMatchObject({ appended: 2, skipped: 2 });
    expect(sessions.list('scratch')).toHaveLength(1);
    expect(
      sessions.readSession('scratch', first.key)!.turns.map((turn) => turn.text),
    ).toEqual(['day one', 'noted', 'day two', 'noted again']);
  });

  it('refuses and writes nothing when sessions are off', () => {
    delete process.env.REMBERO_SESSIONS;
    const off = new SessionStore({
      root: join(root, 'sessions-off'),
      capBytes: 1024 * 1024,
    });

    expect(() => importClaudeTranscript(off, 'scratch', transcriptPath)).toThrow(
      /REMBERO_SESSIONS/,
    );
    expect(existsSync(join(root, 'sessions-off'))).toBe(false);
  });

  it('names a transcript path that does not exist', () => {
    expect(() =>
      importClaudeTranscript(sessions, 'scratch', join(root, 'absent.jsonl')),
    ).toThrow(/no transcript file at .*absent\.jsonl/);
  });

  it('refuses a symbolic link as a transcript, saying so', () => {
    const link = join(root, 'linked.jsonl');
    symlinkSync(transcriptPath, link);

    expect(() => importClaudeTranscript(sessions, 'scratch', link)).toThrow(
      /symbolic link/,
    );
  });

  it('refuses a transcript path that is not a regular file', () => {
    expect(() => importClaudeTranscript(sessions, 'scratch', root)).toThrow(
      /non-regular/,
    );
  });
});

describe('remembero sessions CLI', () => {
  const cli = (argv: string[], env: Record<string, string | undefined> = {}) => {
    const childEnv: Record<string, string> = {};
    for (const [key, value] of Object.entries({
      ...process.env,
      REMBERO_HOME: join(root, 'home'),
      REMBERO_SESSIONS: 'on',
      ...env,
    })) {
      if (value !== undefined) childEnv[key] = value;
    }
    return spawnSync(process.execPath, [resolve('dist/cli.js'), ...argv], {
      encoding: 'utf8',
      env: childEnv,
    });
  };
  const storedFile = (key: string) =>
    join(root, 'home', 'sessions', 'scratch', `${key}.jsonl`);

  it('imports, lists and forgets one session', () => {
    const key = sessions.sessionKey('import', SESSION_ID);

    const imported = cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);
    expect(imported.stderr).toBe('');
    expect(imported.status).toBe(0);
    expect(imported.stdout).toMatch(/past-chat\.jsonl/);
    expect(imported.stdout).toMatch(/2 appended/);
    expect(existsSync(storedFile(key))).toBe(true);

    const twice = cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);
    expect(twice.status).toBe(0);
    expect(twice.stdout).toMatch(/0 appended/);
    expect(twice.stdout).toMatch(/2 skipped/);

    const listed = cli(['sessions', 'list', '-n', 'scratch']);
    expect(listed.status).toBe(0);
    expect(listed.stdout).toContain(key);
    expect(listed.stdout).toContain('import');
    expect(listed.stdout).toMatch(/2 turn/);

    const forgotten = cli(['sessions', 'forget', key, '-n', 'scratch']);
    expect(forgotten.status).toBe(0);
    expect(forgotten.stdout).toMatch(/forgot/);
    expect(existsSync(storedFile(key))).toBe(false);
    expect(cli(['sessions', 'list', '-n', 'scratch']).stdout).toMatch(/no sessions/);
  });

  it('forgets a whole namespace with --all', () => {
    cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);

    const forgotten = cli(['sessions', 'forget', '--all', '-n', 'scratch']);

    expect(forgotten.status).toBe(0);
    expect(forgotten.stdout).toMatch(/1 session/);
    expect(existsSync(join(root, 'home', 'sessions', 'scratch'))).toBe(false);
  });

  it('refuses forget without a key or --all', () => {
    const result = cli(['sessions', 'forget', '-n', 'scratch']);

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/--all/);
  });

  it('still lists and forgets what is on disk with sessions off', () => {
    const key = sessions.sessionKey('import', SESSION_ID);
    cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);

    // Turning sessions off stops new conversation text being kept; it must never
    // trap what is already stored beyond the user's reach.
    const listed = cli(['sessions', 'list', '-n', 'scratch'], {
      REMBERO_SESSIONS: 'off',
    });
    expect(listed.status).toBe(0);
    expect(listed.stdout).toContain(key);

    const forgotten = cli(['sessions', 'forget', key, '-n', 'scratch'], {
      REMBERO_SESSIONS: undefined,
    });
    expect(forgotten.status).toBe(0);
    expect(forgotten.stdout).toMatch(/forgot session/);
    expect(existsSync(storedFile(key))).toBe(false);
  });

  it('lists and forgets with a cap this build cannot read', () => {
    // A cap is irrelevant to deleting, but sessionStoreEvenIfOff built a store whose
    // constructor read REMBERO_SESSION_CAP_BYTES, so one typo trapped every stored
    // conversation: the user could neither see what was kept nor delete it.
    const key = sessions.sessionKey('import', SESSION_ID);
    cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);

    const listed = cli(['sessions', 'list', '-n', 'scratch'], {
      REMBERO_SESSION_CAP_BYTES: 'two hundred megabytes',
    });
    expect(listed.status).toBe(0);
    expect(listed.stdout).toContain(key);

    const forgotten = cli(['sessions', 'forget', key, '-n', 'scratch'], {
      REMBERO_SESSION_CAP_BYTES: 'two hundred megabytes',
    });
    expect(forgotten.status).toBe(0);
    expect(forgotten.stdout).toMatch(/forgot session/);
    expect(existsSync(storedFile(key))).toBe(false);
  });

  it('still refuses import on a cap it cannot read, naming that setting', () => {
    // The delete path's default cap must not make a bad cap silently acceptable to
    // the path that actually writes and evicts.
    const bad = cli(['sessions', 'import', transcriptPath, '-n', 'scratch'], {
      REMBERO_SESSION_CAP_BYTES: 'two hundred megabytes',
    });

    expect(bad.status).toBe(1);
    expect(bad.stderr).toMatch(/REMBERO_SESSION_CAP_BYTES/);
  });

  it('refuses import with sessions off, naming the setting and writing nothing', () => {
    const off = cli(['sessions', 'import', transcriptPath, '-n', 'scratch'], {
      REMBERO_SESSIONS: undefined,
    });

    expect(off.status).toBe(1);
    expect(off.stderr).toMatch(/REMBERO_SESSIONS/);
    expect(existsSync(join(root, 'home', 'sessions'))).toBe(false);
  });

  it('names an unreadable sessions setting instead of importing', () => {
    const bad = cli(['sessions', 'import', transcriptPath, '-n', 'scratch'], {
      REMBERO_SESSIONS: 'yes',
    });

    expect(bad.status).toBe(1);
    expect(bad.stderr).toMatch(/REMBERO_SESSIONS/);
    // Once, not once from the store's note and again from the refusal.
    expect(bad.stderr.match(/REMBERO_SESSIONS/g)).toHaveLength(1);
    expect(existsSync(join(root, 'home', 'sessions'))).toBe(false);
  });

  it('refuses list and forget on an unreadable sessions setting too', () => {
    const key = sessions.sessionKey('import', SESSION_ID);
    cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);

    for (const argv of [
      ['sessions', 'list', '-n', 'scratch'],
      ['sessions', 'forget', key, '-n', 'scratch'],
    ]) {
      const result = cli(argv, { REMBERO_SESSIONS: 'yes' });
      expect(result.status).toBe(1);
      expect(result.stderr.match(/REMBERO_SESSIONS/g)).toHaveLength(1);
    }
    expect(existsSync(storedFile(key))).toBe(true);
  });
});
