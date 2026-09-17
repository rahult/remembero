import {
  appendFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { SessionStore, sessionsRoot } from '../src/sessions/store.js';
import {
  DEFAULT_SESSION_CAP_BYTES,
  sessionCapBytesFromEnv,
  sessionsEnabledFromEnv,
} from '../src/env.js';

const header = {
  version: 1,
  source: 'claude-code',
  sourceSessionId: 'abc',
  startedAt: '2026-09-18T00:00:00.000Z',
} as const;

describe('session store', () => {
  let tmp: string;
  let store: SessionStore;

  beforeAll(() => {
    tmp = mkdtempSync(join(tmpdir(), 'rembero-sessions-'));
    store = new SessionStore({ root: tmp, capBytes: 1024 * 1024 });
  });

  it('appends turns and skips ones it already has', () => {
    const first = store.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
      { role: 'assistant', ts: '2026-09-18T00:00:01.000Z', text: 'Nice, which model?' },
    ]);
    expect(first.appended).toBe(2);
    const second = store.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
      { role: 'user', ts: '2026-09-18T00:00:02.000Z', text: 'Rode 40 km today' },
    ]);
    expect(second).toMatchObject({ appended: 1, skipped: 1 });
    const key = store.sessionKey('claude-code', 'abc');
    expect(store.readSession('default', key)!.turns.map((t) => t.index)).toEqual([0, 1, 2]);
  });

  it('writes a header line then one JSON line per turn', () => {
    const key = store.sessionKey('claude-code', 'abc');
    const lines = readFileSync(join(tmp, 'default', `${key}.jsonl`), 'utf8')
      .trimEnd()
      .split('\n');
    expect(lines).toHaveLength(4);
    expect(JSON.parse(lines[0])).toMatchObject({
      version: 1,
      source: 'claude-code',
      sourceSessionId: 'abc',
    });
    expect(JSON.parse(lines[1])).toMatchObject({ index: 0, role: 'user' });
    expect(store.readSession('default', key)!.header).toMatchObject({
      sourceSessionId: 'abc',
    });
    const entry = store.list('default').find((e) => e.key === key)!;
    expect(entry).toMatchObject({ turns: 3, source: 'claude-code' });
    expect(entry.lastTs).toBe('2026-09-18T00:00:02.000Z');
    expect(entry.bytes).toBeGreaterThan(0);
  });

  it('masks a secret and records it', () => {
    const result = store.appendTurns('default', { ...header, sourceSessionId: 'sec' }, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'my api key = sk-abc123456789' },
    ]);
    expect(result.masked).toBe(1);
    const key = store.sessionKey('claude-code', 'sec');
    expect(store.readSession('default', key)!.turns[0].text).toContain('[redacted]');
  });

  it('drops the oldest session when the cap is exceeded', () => {
    const tmp2 = mkdtempSync(join(tmpdir(), 'rembero-sessions-cap-'));
    const logged: string[] = [];
    const small = new SessionStore({
      root: tmp2,
      capBytes: 400,
      log: (message) => logged.push(message),
    });
    small.appendTurns('default', { ...header, sourceSessionId: 'old', startedAt: '2026-09-01T00:00:00.000Z' }, [
      { role: 'user', ts: '2026-09-01T00:00:00.000Z', text: 'x'.repeat(300) },
    ]);
    const result = small.appendTurns('default', { ...header, sourceSessionId: 'new', startedAt: '2026-09-18T00:00:00.000Z' }, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'y'.repeat(300) },
    ]);
    expect(result.dropped).toEqual([small.sessionKey('claude-code', 'old')]);
    expect(small.list('default').map((e) => e.key)).toEqual([small.sessionKey('claude-code', 'new')]);
    expect(logged.join('\n')).toContain(small.sessionKey('claude-code', 'old'));
  });

  it('deletes a session and its index entry', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-del-'));
    const owner = new SessionStore({ root, capBytes: 1024 * 1024 });
    owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    ]);
    const key = owner.sessionKey('claude-code', 'abc');
    expect(owner.deleteSession('default', key)).toBe(true);
    expect(owner.list('default')).toEqual([]);
    expect(owner.readSession('default', key)).toBeUndefined();
    expect(owner.deleteSession('default', key)).toBe(false);
  });

  it('deletes a whole namespace and reports how many sessions went', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-ns-'));
    const owner = new SessionStore({ root, capBytes: 1024 * 1024 });
    for (const id of ['one', 'two']) {
      owner.appendTurns('scratch', { ...header, sourceSessionId: id }, [
        { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: `session ${id}` },
      ]);
    }
    expect(owner.list('scratch')).toHaveLength(2);
    expect(owner.deleteNamespace('scratch')).toBe(2);
    expect(owner.list('scratch')).toEqual([]);
    expect(owner.deleteNamespace('scratch')).toBe(0);
  });

  it('appends on a clean line after a torn write', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-torn-'));
    const logged: string[] = [];
    const owner = new SessionStore({
      root,
      capBytes: 1024 * 1024,
      log: (message) => logged.push(message),
    });
    owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    ]);
    const key = owner.sessionKey('claude-code', 'abc');
    const path = join(root, 'default', `${key}.jsonl`);
    appendFileSync(path, '{"index":1,"role":"user","ts":"2026');
    const result = owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:02.000Z', text: 'Rode 40 km today' },
    ]);
    expect(result.appended).toBe(1);
    const session = owner.readSession('default', key)!;
    expect(session.turns.map((t) => t.text)).toEqual([
      'I bought a road bike',
      'Rode 40 km today',
    ]);
    expect(logged.join('\n')).toContain('unreadable turn');
  });

  it('quarantines a torn header and keeps the rest of the namespace working', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-header-'));
    const logged: string[] = [];
    const owner = new SessionStore({
      root,
      capBytes: 1024 * 1024,
      log: (message) => logged.push(message),
    });
    owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    ]);
    owner.appendTurns('default', { ...header, sourceSessionId: 'other' }, [
      { role: 'user', ts: '2026-09-18T00:00:01.000Z', text: 'A different session' },
    ]);
    const key = owner.sessionKey('claude-code', 'abc');
    const other = owner.sessionKey('claude-code', 'other');
    const path = join(root, 'default', `${key}.jsonl`);
    const lines = readFileSync(path, 'utf8').trimEnd().split('\n');
    // A crash while the header was being written.
    writeFileSync(
      path,
      [`{"version":1,"source":"claude-c`, ...lines.slice(1)].join('\n') + '\n'
    );
    // ...and no index.json to fall back on.
    rmSync(join(root, 'default', 'index.json'));

    expect(owner.readSession('default', key)).toBeUndefined();
    expect(owner.list('default').map((e) => e.key)).toEqual([other]);
    expect(
      owner.appendTurns('default', { ...header, sourceSessionId: 'other' }, [
        { role: 'user', ts: '2026-09-18T00:00:02.000Z', text: 'Still writing' },
      ]).appended
    ).toBe(1);

    const result = owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:03.000Z', text: 'After the tear' },
    ]);
    expect(result.appended).toBe(1);
    expect(existsSync(`${path}.corrupt`)).toBe(true);
    expect(owner.readSession('default', key)!.turns.map((t) => t.text)).toEqual([
      'After the tear',
    ]);
    expect(logged.join('\n')).toContain('unreadable');
  });

  it('never reuses a turn index after a line is dropped', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-index-'));
    const owner = new SessionStore({
      root,
      capBytes: 1024 * 1024,
      log: () => {},
    });
    owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
      { role: 'assistant', ts: '2026-09-18T00:00:01.000Z', text: 'Nice, which model?' },
    ]);
    const key = owner.sessionKey('claude-code', 'abc');
    const path = join(root, 'default', `${key}.jsonl`);
    const lines = readFileSync(path, 'utf8').trimEnd().split('\n');
    // The turn at index 0 is now unreadable; the turn at index 1 survives.
    writeFileSync(
      path,
      [lines[0], '{"index":0,"role":"user"', lines[2]].join('\n') + '\n'
    );
    const result = owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:02.000Z', text: 'Rode 40 km today' },
      { role: 'user', ts: '2026-09-18T00:00:03.000Z', text: 'And 20 more' },
    ]);
    expect(result.appended).toBe(2);
    const indexes = owner.readSession('default', key)!.turns.map((t) => t.index);
    expect(indexes).toEqual([1, 2, 3]);
    expect(new Set(indexes).size).toBe(indexes.length);
  });

  it('holds a per-namespace lock across the append and the index refresh', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-lock-'));
    let onLog: (message: string) => void = () => {};
    const first = new SessionStore({
      root,
      capBytes: 400,
      log: (message) => onLog(message),
    });
    const second = new SessionStore({ root, capBytes: 400, log: () => {} });
    first.appendTurns(
      'default',
      { ...header, sourceSessionId: 'old', startedAt: '2026-09-01T00:00:00.000Z' },
      [{ role: 'user', ts: '2026-09-01T00:00:00.000Z', text: 'x'.repeat(300) }]
    );
    let nested: unknown;
    onLog = () => {
      if (nested !== undefined) return;
      try {
        second.appendTurns('default', { ...header, sourceSessionId: 'rival' }, [
          { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'racing' },
        ]);
        nested = 'appended without waiting for the lock';
      } catch (error) {
        nested = error;
      }
    };
    // The drop inside the locked region is where the second writer tries to cut in.
    first.appendTurns(
      'default',
      { ...header, sourceSessionId: 'new', startedAt: '2026-09-18T00:00:00.000Z' },
      [{ role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'y'.repeat(300) }]
    );
    expect(nested).toBeInstanceOf(Error);
    expect((nested as Error).message).toMatch(/lock/i);
    onLog = () => {};
    // The lock is released, so the rival can write once the first writer is done.
    expect(
      second.appendTurns('default', { ...header, sourceSessionId: 'rival' }, [
        { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'racing' },
      ]).appended
    ).toBe(1);
  });

  it('takes over a stale lock whose owner is gone, and leaves none behind', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-stale-'));
    const owner = new SessionStore({ root, capBytes: 1024 * 1024 });
    const lockPath = join(root, '.session-default.lock');
    writeFileSync(
      lockPath,
      `${JSON.stringify({ pid: 2147483646, createdAt: '2026-09-01T00:00:00.000Z' })}\n`
    );
    const longAgo = Date.now() / 1000 - 120;
    utimesSync(lockPath, longAgo, longAgo);
    expect(
      owner.appendTurns('default', header, [
        { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
      ]).appended
    ).toBe(1);
    expect(existsSync(lockPath)).toBe(false);
  });

  it('logs a line that parses but is not a turn', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-junk-'));
    const logged: string[] = [];
    const owner = new SessionStore({
      root,
      capBytes: 1024 * 1024,
      log: (message) => logged.push(message),
    });
    owner.appendTurns('default', header, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I bought a road bike' },
    ]);
    const key = owner.sessionKey('claude-code', 'abc');
    appendFileSync(
      join(root, 'default', `${key}.jsonl`),
      '{"note":"a line that is not a turn"}\n'
    );
    expect(owner.readSession('default', key)!.turns).toHaveLength(1);
    expect(logged.join('\n')).toContain('not a turn');
  });

  it('rejects a turn timestamp that is not a timestamp', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-ts-'));
    const owner = new SessionStore({ root, capBytes: 1024 * 1024 });
    expect(() =>
      owner.appendTurns('default', header, [
        { role: 'user', ts: 'whenever', text: 'I bought a road bike' },
      ])
    ).toThrow(/ts/i);
    // Nothing is written when a batch is rejected.
    expect(owner.list('default')).toEqual([]);
    expect(() =>
      owner.appendTurns('default', header, [
        { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'fine' },
        { role: 'user', ts: '', text: 'not fine' },
      ])
    ).toThrow(/ts/i);
    expect(owner.list('default')).toEqual([]);
  });

  it('rejects a namespace that is not [a-z0-9_-]+', () => {
    expect(() => store.list('../etc')).toThrow();
    expect(() => store.appendTurns('../etc', header, [])).toThrow();
    expect(() => store.deleteNamespace('Default')).toThrow();
    expect(() => store.readSession('default', '../../etc/passwd')).toThrow();
  });

  it('reads the root and the cap from the environment', () => {
    expect(sessionsRoot({ REMBERO_HOME: '/tmp/rembero-home' })).toBe(
      join('/tmp/rembero-home', 'sessions')
    );
    expect(sessionsEnabledFromEnv({})).toBe(false);
    expect(sessionsEnabledFromEnv({ REMBERO_SESSIONS: 'on' })).toBe(true);
    expect(() => sessionsEnabledFromEnv({ REMBERO_SESSIONS: 'yes' })).toThrow(
      /REMBERO_SESSIONS/
    );
    expect(sessionCapBytesFromEnv({})).toBe(DEFAULT_SESSION_CAP_BYTES);
    expect(
      sessionCapBytesFromEnv({ REMBERO_SESSION_CAP_BYTES: '2097152' })
    ).toBe(2 * 1024 * 1024);
    expect(() =>
      sessionCapBytesFromEnv({ REMBERO_SESSION_CAP_BYTES: '1024' })
    ).toThrow(/REMBERO_SESSION_CAP_BYTES/);
    expect(() =>
      sessionCapBytesFromEnv({ REMBERO_SESSION_CAP_BYTES: 'lots' })
    ).toThrow(/integer/i);
    expect(store.sessionsEnabled()).toBe(
      sessionsEnabledFromEnv(process.env)
    );
  });
});
