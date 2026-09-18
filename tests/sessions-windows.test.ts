import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SESSION_WINDOW_BYTES,
  DEFAULT_SESSION_WINDOW_GAP_MS,
  DEFAULT_SESSION_WINDOW_TURNS,
  sessionWindowBytesFromEnv,
  sessionWindowGapMsFromEnv,
  sessionWindowTurnsFromEnv,
} from '../src/env.js';
import { importClaudeTranscript } from '../src/sessions/import.js';
import { SessionStore, type SessionHeader } from '../src/sessions/store.js';
import {
  parseSessionWindowId,
  windowSessionTurns,
} from '../src/sessions/windows.js';
import { forgetSessionsTool } from '../src/mcp/tools.js';

const HOUR_MS = 60 * 60 * 1000;
const START = Date.parse('2026-09-13T08:00:00.000Z');

const header = (sourceSessionId: string, startedAt: string): SessionHeader => ({
  version: 1,
  source: 'import',
  sourceSessionId,
  startedAt,
});

/** A store whose windows are small enough to read in a test. */
function store(
  options: { windowTurns?: number; windowGapMs?: number; windowBytes?: number } = {},
) {
  return new SessionStore({
    root: mkdtempSync(join(tmpdir(), 'rembero-sessions-windows-')),
    capBytes: 1024 * 1024,
    windowTurns: options.windowTurns ?? 4,
    windowGapMs: options.windowGapMs ?? 6 * HOUR_MS,
    windowBytes: options.windowBytes ?? 64 * 1024,
    log: () => {},
  });
}

/** `count` turns, `spacingMs` apart, alternating roles. */
function turns(
  count: number,
  spacingMs = 1_000,
  at = START,
): Array<{ role: 'user' | 'assistant'; ts: string; text: string }> {
  return Array.from({ length: count }, (_unused, index) => ({
    role: (index % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
    ts: new Date(at + index * spacingMs).toISOString(),
    text: `turn ${index}`,
  }));
}

describe('windowSessionTurns: the rule that cuts a conversation into reading units', () => {
  it('starts a new window at the turn limit', () => {
    const grouped = windowSessionTurns(turns(10), {
      turns: 4,
      gapMs: HOUR_MS,
      bytes: 8_192,
    });
    expect(grouped.map((window) => window.length)).toEqual([4, 4, 2]);
  });

  it('starts a new window when the conversation was put down and picked up later', () => {
    const sitting = turns(3);
    const nextMorning = turns(2, 1_000, START + 20 * HOUR_MS);
    const grouped = windowSessionTurns([...sitting, ...nextMorning], {
      turns: 40,
      gapMs: 6 * HOUR_MS,
      bytes: 8_192,
    });
    expect(grouped.map((window) => window.length)).toEqual([3, 2]);
    expect(grouped[1]![0]!.ts).toBe(nextMorning[0]!.ts);
  });

  it('starts a new window when the text would outgrow what the reader is shown', () => {
    // One turn is not one size: a 40-turn window of the real transcript came to 63 KB
    // around a single 16.8 KB paste, and everything after that paste was past the
    // reader's 16,384-character source cap however well it ranked.
    const long = { ts: new Date(START).toISOString(), text: 'x'.repeat(5_000) };
    const short = (index: number) => ({
      ts: new Date(START + index * 1_000).toISOString(),
      text: `short ${index}`,
    });
    const grouped = windowSessionTurns(
      [long, short(1), short(2), { ...long, ts: short(3).ts }, short(4)],
      { turns: 40, gapMs: HOUR_MS, bytes: 8_192 },
    );
    expect(grouped.map((window) => window.length)).toEqual([3, 2]);
  });

  it('keeps a turn larger than the byte bound whole, in a window of its own', () => {
    const huge = { ts: new Date(START).toISOString(), text: 'x'.repeat(20_000) };
    const grouped = windowSessionTurns(
      [
        { ts: new Date(START - 1_000).toISOString(), text: 'before' },
        huge,
        { ts: new Date(START + 1_000).toISOString(), text: 'after' },
      ],
      { turns: 40, gapMs: HOUR_MS, bytes: 8_192 },
    );
    expect(grouped.map((window) => window.map((turn) => turn.text.length))).toEqual([
      [6],
      [20_000],
      [5],
    ]);
  });

  it('never splits a turn: the windows concatenate back to the conversation', () => {
    const conversation = [
      ...turns(9),
      ...turns(5, 1_000, START + 30 * HOUR_MS),
      ...turns(3, 1_000, START + 31 * HOUR_MS),
    ];
    const grouped = windowSessionTurns(conversation, {
      turns: 4,
      gapMs: 6 * HOUR_MS,
      bytes: 8_192,
    });
    expect(grouped.flat()).toEqual(conversation);
    // every turn lands in exactly one window, whole
    expect(grouped.reduce((total, window) => total + window.length, 0)).toBe(
      conversation.length,
    );
    expect(grouped.every((window) => window.length > 0)).toBe(true);
  });

  it('keeps a turn whose timestamp cannot be read with its neighbours', () => {
    // Nothing the store writes, but a hand-edited file can hold one: an unmeasurable
    // gap must not open a window.
    const grouped = windowSessionTurns(
      [
        { ts: new Date(START).toISOString() },
        { ts: 'whenever' },
        { ts: new Date(START + 2_000).toISOString() },
      ],
      { turns: 40, gapMs: HOUR_MS, bytes: 8_192 },
    );
    expect(grouped).toHaveLength(1);
  });

  it('reads a window id, and a bare key as the whole session', () => {
    expect(parseSessionWindowId('a'.repeat(32))).toEqual({ key: 'a'.repeat(32) });
    expect(parseSessionWindowId(`${'a'.repeat(32)}#3`)).toEqual({
      key: 'a'.repeat(32),
      window: 3,
    });
    expect(() => parseSessionWindowId(`${'a'.repeat(32)}#x`)).toThrow(/window/);
  });

  it('takes its defaults, and its settings, from the environment', () => {
    expect(DEFAULT_SESSION_WINDOW_TURNS).toBe(40);
    expect(DEFAULT_SESSION_WINDOW_GAP_MS).toBe(6 * HOUR_MS);
    expect(DEFAULT_SESSION_WINDOW_BYTES).toBe(8 * 1024);
    expect(sessionWindowTurnsFromEnv({})).toBe(DEFAULT_SESSION_WINDOW_TURNS);
    expect(sessionWindowBytesFromEnv({})).toBe(DEFAULT_SESSION_WINDOW_BYTES);
    expect(
      sessionWindowBytesFromEnv({ REMBERO_SESSION_WINDOW_BYTES: '4096' }),
    ).toBe(4_096);
    expect(() =>
      sessionWindowBytesFromEnv({ REMBERO_SESSION_WINDOW_BYTES: '10' }),
    ).toThrow(/REMBERO_SESSION_WINDOW_BYTES/);
    expect(sessionWindowGapMsFromEnv({})).toBe(DEFAULT_SESSION_WINDOW_GAP_MS);
    expect(
      sessionWindowTurnsFromEnv({ REMBERO_SESSION_WINDOW_TURNS: '12' }),
    ).toBe(12);
    expect(
      sessionWindowGapMsFromEnv({ REMBERO_SESSION_WINDOW_GAP_MS: '600000' }),
    ).toBe(600_000);
    expect(() =>
      sessionWindowTurnsFromEnv({ REMBERO_SESSION_WINDOW_TURNS: '1' }),
    ).toThrow(/REMBERO_SESSION_WINDOW_TURNS/);
    expect(() =>
      sessionWindowTurnsFromEnv({ REMBERO_SESSION_WINDOW_TURNS: 'lots' }),
    ).toThrow(/integer/i);
    expect(() =>
      sessionWindowGapMsFromEnv({ REMBERO_SESSION_WINDOW_GAP_MS: '10' }),
    ).toThrow(/REMBERO_SESSION_WINDOW_GAP_MS/);
  });
});

describe('the session store offers windows, not whole conversations', () => {
  it('lists one entry per window, each with its own range and turn count', () => {
    const owner = store({ windowTurns: 4 });
    owner.appendTurns('scratch', header('long', new Date(START).toISOString()), turns(10));
    const key = owner.sessionKey('import', 'long');

    const windows = owner.list('scratch');
    expect(windows.map((window) => window.windowKey)).toEqual([
      `${key}#0`,
      `${key}#1`,
      `${key}#2`,
    ]);
    expect(windows.map((window) => window.turns)).toEqual([4, 4, 2]);
    expect(windows.every((window) => window.key === key)).toBe(true);
    expect(windows.every((window) => window.windows === 3)).toBe(true);
    // each window's own time range, not the conversation's
    expect(windows[0]!.startedAt).toBe(new Date(START).toISOString());
    expect(windows[0]!.lastTs).toBe(new Date(START + 3_000).toISOString());
    expect(windows[1]!.startedAt).toBe(new Date(START + 4_000).toISOString());
    expect(windows[2]!.lastTs).toBe(new Date(START + 9_000).toISOString());
    expect(windows.every((window) => window.bytes > 0)).toBe(true);
    // the whole conversation is still one session, one file
    expect(owner.listSessions('scratch').map((entry) => entry.key)).toEqual([key]);
    expect(owner.listSessions('scratch')[0]!.turns).toBe(10);
  });

  it('splits on a six-hour silence, so a window is one sitting', () => {
    const owner = store({ windowTurns: 40, windowGapMs: 6 * HOUR_MS });
    owner.appendTurns('scratch', header('days', new Date(START).toISOString()), [
      ...turns(3),
      ...turns(2, 1_000, START + 26 * HOUR_MS),
      ...turns(2, 1_000, START + 52 * HOUR_MS),
    ]);

    const windows = owner.list('scratch');
    expect(windows.map((window) => window.turns)).toEqual([3, 2, 2]);
    expect(windows.map((window) => window.startedAt.slice(0, 10))).toEqual([
      '2026-09-13',
      '2026-09-14',
      '2026-09-15',
    ]);
  });

  it('reads one window by its id, and the whole session by its key', () => {
    const owner = store({ windowTurns: 4 });
    owner.appendTurns('scratch', header('long', new Date(START).toISOString()), turns(10));
    const key = owner.sessionKey('import', 'long');

    expect(
      owner.readSession('scratch', `${key}#1`)!.turns.map((turn) => turn.text),
    ).toEqual(['turn 4', 'turn 5', 'turn 6', 'turn 7']);
    expect(owner.readSession('scratch', key)!.turns).toHaveLength(10);
    expect(owner.readSession('scratch', `${key}#9`)).toBeUndefined();
    expect(
      owner
        .readSessionWindows('scratch', key)
        .map((window) => window.turns.length),
    ).toEqual([4, 4, 2]);
  });

  it('appends into the open window and renumbers nothing already stored', () => {
    const owner = store({ windowTurns: 4 });
    const key = owner.sessionKey('import', 'growing');
    owner.appendTurns('scratch', header('growing', new Date(START).toISOString()), turns(6));
    const before = owner.list('scratch').map((window) => window.turns);
    expect(before).toEqual([4, 2]);

    owner.appendTurns(
      'scratch',
      header('growing', new Date(START).toISOString()),
      turns(4, 1_000, START + 6_000).map((turn, index) => ({
        ...turn,
        text: `later ${index}`,
      })),
    );

    expect(owner.list('scratch').map((window) => window.turns)).toEqual([4, 4, 2]);
    expect(
      owner.readSession('scratch', `${key}#0`)!.turns.map((turn) => turn.text),
    ).toEqual(['turn 0', 'turn 1', 'turn 2', 'turn 3']);
  });

  it('forgets every window of a session from its session key', () => {
    const owner = store({ windowTurns: 4 });
    owner.appendTurns('scratch', header('long', new Date(START).toISOString()), turns(10));
    owner.appendTurns('scratch', header('other', new Date(START).toISOString()), turns(2));
    const key = owner.sessionKey('import', 'long');
    expect(owner.list('scratch').filter((w) => w.key === key)).toHaveLength(3);

    expect(
      forgetSessionsTool({ sessions: owner }, { namespace: 'scratch', key }),
    ).toEqual({ deleted: 1 });

    expect(owner.list('scratch').filter((w) => w.key === key)).toEqual([]);
    expect(owner.readSession('scratch', `${key}#0`)).toBeUndefined();
    expect(owner.readSession('scratch', key)).toBeUndefined();
    // the other conversation is untouched
    expect(owner.list('scratch').map((w) => w.windowKey)).toEqual([
      `${owner.sessionKey('import', 'other')}#0`,
    ]);
  });

  it('forgets the whole conversation when handed one of its window ids', () => {
    // Windows are a reading unit over one append-only file, so there is no way to
    // delete one and keep the rest; recall hands the user a window id, and forgetting
    // must not leave most of the conversation behind.
    const owner = store({ windowTurns: 4 });
    owner.appendTurns('scratch', header('long', new Date(START).toISOString()), turns(10));
    const key = owner.sessionKey('import', 'long');

    expect(
      forgetSessionsTool(
        { sessions: owner },
        { namespace: 'scratch', key: `${key}#1` },
      ),
    ).toEqual({ deleted: 1 });
    expect(owner.list('scratch')).toEqual([]);
  });

  it('reads a session file written before windowing existed as windows', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-migration-'));
    const owner = new SessionStore({
      root,
      capBytes: 1024 * 1024,
      windowTurns: 4,
      windowGapMs: 6 * HOUR_MS,
      windowBytes: 64 * 1024,
      log: () => {},
    });
    const key = owner.sessionKey('import', 'old');
    const dir = join(root, 'scratch');
    mkdirSync(dir, { recursive: true });
    const stored = turns(10);
    // exactly what the store wrote before windowing: a header line, one turn per line,
    // and an index.json with no windows in it
    writeFileSync(
      join(dir, `${key}.jsonl`),
      [
        JSON.stringify(header('old', new Date(START).toISOString())),
        ...stored.map((turn, index) =>
          JSON.stringify({ ...turn, index, hash: `hash-${index}` }),
        ),
      ].join('\n') + '\n',
      'utf8',
    );
    writeFileSync(
      join(dir, 'index.json'),
      `${JSON.stringify({
        version: 1,
        sessions: [
          {
            key,
            source: 'import',
            startedAt: new Date(START).toISOString(),
            lastTs: stored.at(-1)!.ts,
            turns: stored.length,
            bytes: 1_234,
          },
        ],
      })}\n`,
      'utf8',
    );
    const fileBefore = readFileSync(join(dir, `${key}.jsonl`), 'utf8');

    const windows = owner.list('scratch');

    expect(windows.map((window) => window.windowKey)).toEqual([
      `${key}#0`,
      `${key}#1`,
      `${key}#2`,
    ]);
    expect(windows.map((window) => window.turns)).toEqual([4, 4, 2]);
    expect(
      owner.readSession('scratch', `${key}#2`)!.turns.map((turn) => turn.text),
    ).toEqual(['turn 8', 'turn 9']);
    // read, never rewritten
    expect(readFileSync(join(dir, `${key}.jsonl`), 'utf8')).toBe(fileBefore);
  });

  it('is idempotent when the same transcript is imported again', () => {
    const previous = process.env.REMBERO_SESSIONS;
    process.env.REMBERO_SESSIONS = 'on';
    try {
      const root = mkdtempSync(join(tmpdir(), 'rembero-sessions-reimport-'));
      const owner = new SessionStore({
        root: join(root, 'sessions'),
        capBytes: 8 * 1024 * 1024,
        windowTurns: 4,
        windowGapMs: 6 * HOUR_MS,
        windowBytes: 64 * 1024,
        log: () => {},
      });
      const transcript = join(root, 'chat.jsonl');
      writeFileSync(
        transcript,
        `${turns(10)
          .map((turn) =>
            JSON.stringify({
              type: turn.role,
              sessionId: 'reimport-session',
              timestamp: turn.ts,
              message: { role: turn.role, content: turn.text },
            }),
          )
          .join('\n')}\n`,
        'utf8',
      );

      const first = importClaudeTranscript(owner, 'scratch', transcript);
      const windowsAfterFirst = owner.list('scratch');
      const again = importClaudeTranscript(owner, 'scratch', transcript);

      expect(first).toMatchObject({ appended: 10, skipped: 0 });
      expect(again).toMatchObject({ appended: 0, skipped: 10 });
      expect(again.key).toBe(first.key);
      expect(owner.list('scratch')).toEqual(windowsAfterFirst);
      expect(owner.listSessions('scratch')).toHaveLength(1);
    } finally {
      if (previous === undefined) delete process.env.REMBERO_SESSIONS;
      else process.env.REMBERO_SESSIONS = previous;
    }
  });

  it('offers a 468-turn conversation to retrieval as a dozen windows', () => {
    // The measured shape of one real Claude Code session: 468 turns over five days,
    // where retrieval had a single candidate and the reading budget kept its first 7%.
    const owner = new SessionStore({
      root: mkdtempSync(join(tmpdir(), 'rembero-sessions-real-shape-')),
      capBytes: 8 * 1024 * 1024,
      log: () => {},
    });
    const spread = Array.from({ length: 468 }, (_unused, index) => ({
      role: (index % 5 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
      // five days of conversation, ~13 minutes between turns
      ts: new Date(START + index * 13 * 60 * 1_000).toISOString(),
      text: `turn ${index}: ${'conversation text. '.repeat(20)}`,
    }));
    owner.appendTurns('scratch', header('real', spread[0]!.ts), spread);

    const windows = owner.list('scratch');
    // the default 40-turn limit alone gives twelve, and the six-hour gap gives more
    expect(windows.length).toBeGreaterThanOrEqual(12);
    expect(owner.listSessions('scratch')).toHaveLength(1);
    // every turn is offered exactly once, in order, and no window holds the lot
    expect(windows.reduce((total, window) => total + window.turns, 0)).toBe(468);
    expect(Math.max(...windows.map((window) => window.turns))).toBeLessThanOrEqual(
      DEFAULT_SESSION_WINDOW_TURNS,
    );
    expect(new Set(windows.map((window) => window.windowKey)).size).toBe(
      windows.length,
    );
    expect(
      owner
        .readSessionWindows('scratch', owner.sessionKey('import', 'real'))
        .flatMap((window) => window.turns.map((turn) => turn.index)),
    ).toEqual(spread.map((_unused, index) => index));
  });
});
