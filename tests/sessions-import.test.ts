import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { importClaudeTranscript } from '../src/sessions/import.js';
import { SessionStore } from '../src/sessions/store.js';

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
  timestamp?: string,
): string {
  return JSON.stringify({
    type: role,
    message: { role, content },
    ...(timestamp === undefined ? {} : { timestamp }),
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
    transcriptLine('user', 'I moved to Melbourne in March.', FIRST_TS),
    transcriptLine('assistant', 'Noted, Melbourne since March.'),
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
      key: sessions.sessionKey('import', 'past-chat'),
      appended: 2,
      skipped: 0,
    });
    const stored = sessions.readSession('scratch', result.key)!;
    expect(stored.header).toMatchObject({
      version: 1,
      source: 'import',
      sourceSessionId: 'past-chat',
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
      lines.push(transcriptLine('user', `turn ${index} ${'x'.repeat(200)}`));
    }
    writeTranscript(long, lines);

    const result = importClaudeTranscript(sessions, 'scratch', long);

    expect(result.appended).toBe(400);
    const stored = sessions.readSession('scratch', result.key)!;
    expect(stored.turns[0].text).toContain('turn 0 ');
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

  it('refuses a transcript path that is not a regular file', () => {
    expect(() =>
      importClaudeTranscript(sessions, 'scratch', root),
    ).toThrow(/regular/);
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

  it('imports, lists and forgets one session', () => {
    const key = sessions.sessionKey('import', 'past-chat');

    const imported = cli(['sessions', 'import', transcriptPath, '-n', 'scratch']);
    expect(imported.stderr).toBe('');
    expect(imported.status).toBe(0);
    expect(imported.stdout).toMatch(/past-chat\.jsonl/);
    expect(imported.stdout).toMatch(/2 appended/);
    expect(existsSync(join(root, 'home', 'sessions', 'scratch', `${key}.jsonl`))).toBe(
      true,
    );

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
    expect(existsSync(join(root, 'home', 'sessions', 'scratch', `${key}.jsonl`))).toBe(
      false,
    );
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
    expect(existsSync(join(root, 'home', 'sessions'))).toBe(false);
  });
});
