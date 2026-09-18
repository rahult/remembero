import { createHash, randomUUID } from 'node:crypto';
import {
  appendFileSync,
  closeSync,
  existsSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import {
  DEFAULT_SESSION_CAP_BYTES,
  DEFAULT_SESSION_WINDOW_BYTES,
  DEFAULT_SESSION_WINDOW_GAP_MS,
  DEFAULT_SESSION_WINDOW_TURNS,
  sessionCapBytesFromEnv,
  sessionWindowBytesFromEnv,
  sessionWindowGapMsFromEnv,
  sessionWindowTurnsFromEnv,
  sessionsEnabledFromEnv,
} from '../env.js';
import {
  containsSensitiveText,
  maskSensitiveSpans,
  REDACTED_SOURCE,
} from '../safety.js';
import {
  parseSessionWindowId,
  sessionWindowKey,
  windowSessionTurns,
  type SessionWindowLimits,
} from './windows.js';

/** A namespace becomes a directory name, so nothing but these characters is allowed. */
const NAMESPACE_PATTERN = /^[a-z0-9_-]+$/;
/** A session key is the first 32 hex characters of a digest, so no id can escape the directory. */
const SESSION_KEY_PATTERN = /^[0-9a-f]{32}$/;
const INDEX_FILE = 'index.json';
/** What an unreadable session file is renamed to, plus `.1`, `.2`, … when one is taken. */
const QUARANTINE_SUFFIX = '.corrupt';
/** Numbered suffixes tried before falling back to a random one, so the search is bounded. */
const MAX_QUARANTINE_SUFFIXES = 1_000;
/** `<key>.jsonl.corrupt`, `<key>.jsonl.corrupt.7`, `<key>.jsonl.corrupt.<uuid>`. */
function quarantinedName(key: string, name: string): boolean {
  return (
    name === `${key}.jsonl${QUARANTINE_SUFFIX}` ||
    name.startsWith(`${key}.jsonl${QUARANTINE_SUFFIX}.`)
  );
}

// The same lock discipline as the fact store's `withLock` (src/store/store.ts): a
// `wx` create for the lock, the owner's pid inside it, and an age-plus-liveness
// test before a crashed writer's lock is taken over.
//
// The wait is short rather than asynchronous. `appendTurns` returns its result
// synchronously — the brief's interface, and capture runs inside a Claude Code stop
// hook — so an async lock would change that signature and every caller. A
// synchronous wait parks the whole event loop, which inside the MCP server means
// the server answers nothing while it waits, so the stall is bounded instead: 250 ms
// in 5 ms sleeps, 50 sleeps at worst, and then a contended writer fails fast with an
// error naming the lock. Failing is safe here because turns are identified by hash:
// the next stop hook re-reads the same transcript tail and stores each turn once.
const LOCK_WAIT_MS = 250;
const LOCK_STALE_MS = 30_000;
const LOCK_RETRY_MS = 5;
const sleepCell = new Int32Array(new SharedArrayBuffer(4));

export type SessionSource = 'claude-code' | 'remember' | 'import';

export interface SessionTurn {
  index: number;
  role: 'user' | 'assistant';
  ts: string;
  text: string;
  hash: string;
}

export interface SessionHeader {
  version: 1;
  source: SessionSource;
  sourceSessionId: string;
  startedAt: string;
  cwd?: string;
}

/**
 * One window of a session, as the namespace index records it.
 *
 * `bytes` is the window's own stored turn lines, where a session entry's `bytes` is the
 * whole file: the cap counts files, and a window is not a file.
 */
export interface SessionWindowSummary {
  /** 0-based ordinal within the session, in file order. */
  window: number;
  startedAt: string;
  lastTs: string;
  turns: number;
  bytes: number;
}

export interface SessionIndexEntry {
  key: string;
  source: SessionSource;
  startedAt: string;
  lastTs: string;
  turns: number;
  bytes: number;
  /**
   * The session's windows, written when the session is appended to. Absent in an
   * `index.json` written before windowing existed; `list` then derives them from the
   * file, so an old store reads as windows without being rewritten.
   */
  windows?: SessionWindowSummary[];
}

/** One window of one session: what `list` offers and what retrieval ranks. */
export interface SessionWindowEntry extends SessionWindowSummary {
  /** The session's file key. `forget_sessions` takes this and removes every window. */
  key: string;
  /** `<key>#<window>`, the id `readSession` takes. */
  windowKey: string;
  source: SessionSource;
  /** How many windows this session has, so a reader of one knows what it is part of. */
  windows: number;
}

/** One window's turns, read from the file. */
export interface SessionWindowRead {
  key: string;
  window: number;
  windowKey: string;
  windows: number;
  header: SessionHeader;
  startedAt: string;
  lastTs: string;
  turns: SessionTurn[];
}

export interface SessionAppendResult {
  appended: number;
  skipped: number;
  masked: number;
  dropped: string[];
}

type SessionFileState =
  | { kind: 'absent' }
  | { kind: 'unreadable'; reason: string }
  | { kind: 'session'; header: SessionHeader; turns: SessionTurn[] };

export interface SessionStoreOptions {
  root?: string;
  capBytes?: number;
  /** At most this many turns in one window (default `REMBERO_SESSION_WINDOW_TURNS`). */
  windowTurns?: number;
  /** The silence that starts a new window (default `REMBERO_SESSION_WINDOW_GAP_MS`). */
  windowGapMs?: number;
  /** At most this much turn text in one window (default `REMBERO_SESSION_WINDOW_BYTES`). */
  windowBytes?: number;
  /** Where cap drops are reported; stderr by default. */
  log?: (message: string) => void;
}

/** `${REMBERO_HOME ?? ~/.rembero}/sessions`, beside the fact store's `memory/`. */
export function sessionsRoot(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.REMBERO_HOME ?? join(homedir(), '.rembero'), 'sessions');
}

/**
 * The conversation store when `REMBERO_SESSIONS=on`, and nothing at all when it
 * is off or when either sessions setting is invalid. An unusable
 * `REMBERO_SESSIONS` or `REMBERO_SESSION_CAP_BYTES` throws from
 * `sessionsEnabledFromEnv` and from the constructor, and the CLI and the MCP
 * server build this store for every command and every tool call: a mistyped
 * sessions setting must cost the user their sessions, not `query`, `recall` and
 * `forget` as well. The failure is named on stderr and the caller proceeds with
 * no store, which is exactly the documented off state.
 *
 * `log` takes that note somewhere other than stderr. A caller that reports an
 * unusable setting itself — `remembero sessions`, which refuses and says why —
 * holds the note back rather than printing the same sentence twice.
 */
export function sessionStoreFromEnv(
  options: { log?: (message: string) => void } = {},
): SessionStore | undefined {
  const log =
    options.log ??
    ((message: string) => {
      process.stderr.write(`${message}\n`);
    });
  try {
    return sessionsEnabledFromEnv() ? new SessionStore() : undefined;
  } catch (error) {
    log(
      `rembero sessions: keeping no sessions: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    return undefined;
  }
}

/**
 * A store for the sessions already on disk, whether or not new ones are being
 * kept.
 *
 * `sessionStoreFromEnv` hands back nothing both when `REMBERO_SESSIONS` is off and
 * when it cannot be read, which is right for writing. Listing and forgetting are
 * different: turning sessions off must not trap what is already stored beyond the
 * user's reach, and deletion is the direction that has to keep working in exactly
 * the state a privacy-minded user picks. So `off` and unset get a plain store here,
 * and only a setting this build cannot read refuses — then nothing knows what the
 * user asked for. Every surface that forgets sessions goes through this, so the CLI
 * and the `forget_sessions` tool cannot drift apart.
 */
export function sessionStoreEvenIfOff(
  configured?: SessionStore,
): SessionStore {
  if (configured !== undefined) return configured;
  // Throws on a `REMBERO_SESSIONS` that is neither 'on' nor 'off': then nothing knows
  // what the user asked for, and refusing is right.
  sessionsEnabledFromEnv();
  // The cap is not. It says how much new conversation text a namespace may keep, which
  // has nothing to do with reading the index or deleting a file, and this store's
  // callers — `sessions list`, `sessions forget`, `forget_sessions` — do only those.
  // Reading it here meant one typo in REMBERO_SESSION_CAP_BYTES trapped every stored
  // conversation: the user could neither see what was kept nor delete it, in the very
  // state a privacy-minded user is most likely to be in. So the delete path takes the
  // default cap and never consults the setting. The write path still does, and still
  // refuses under its own name. The window settings are the same kind of thing: they
  // say how a conversation is cut up for reading, not whether it can be listed or
  // deleted, so a typo in one must not trap what is on disk either.
  return new SessionStore({
    capBytes: DEFAULT_SESSION_CAP_BYTES,
    windowTurns: DEFAULT_SESSION_WINDOW_TURNS,
    windowGapMs: DEFAULT_SESSION_WINDOW_GAP_MS,
    windowBytes: DEFAULT_SESSION_WINDOW_BYTES,
  });
}

function assertNamespace(namespace: string): void {
  if (!NAMESPACE_PATTERN.test(namespace)) {
    throw new Error(
      `session namespace must match ${String(NAMESPACE_PATTERN)}: ${namespace}`,
    );
  }
}

function assertSessionKey(key: string): void {
  if (!SESSION_KEY_PATTERN.test(key)) {
    throw new Error(`session key must be 32 hex characters: ${key}`);
  }
}

function assertHeader(header: SessionHeader): void {
  if (header.version !== 1) throw new Error('session header version must be 1');
  if (
    header.source !== 'claude-code' &&
    header.source !== 'remember' &&
    header.source !== 'import'
  ) {
    throw new Error(
      "session source must be 'claude-code', 'remember', or 'import'",
    );
  }
  if (typeof header.sourceSessionId !== 'string' || header.sourceSessionId === '') {
    throw new Error('session header needs a sourceSessionId');
  }
  if (Number.isNaN(Date.parse(header.startedAt))) {
    throw new Error(`session startedAt is not a timestamp: ${header.startedAt}`);
  }
  if (header.cwd !== undefined && typeof header.cwd !== 'string') {
    throw new Error('session cwd must be a string when present');
  }
}

/**
 * A turn's `ts` reaches the index as `lastTs` and the reader's date-distance
 * arithmetic from there, so it is checked the way a header's `startedAt` is.
 */
function assertTurn(turn: {
  role: 'user' | 'assistant';
  ts: string;
  text: string;
}): void {
  if (turn.role !== 'user' && turn.role !== 'assistant') {
    throw new Error(`session turn role must be user or assistant: ${turn.role}`);
  }
  if (typeof turn.text !== 'string') {
    throw new Error('session turn text must be a string');
  }
  if (typeof turn.ts !== 'string' || Number.isNaN(Date.parse(turn.ts))) {
    throw new Error(`session turn ts is not a timestamp: ${turn.ts}`);
  }
}

/** Only the header's known fields reach the file, in a stable order. */
function headerLine(header: SessionHeader): string {
  const stored: SessionHeader = {
    version: 1,
    source: header.source,
    sourceSessionId: header.sourceSessionId,
    startedAt: header.startedAt,
  };
  if (header.cwd !== undefined) stored.cwd = header.cwd;
  return JSON.stringify(stored);
}

/**
 * A turn's identity for the one job it has: recognising a turn this session has
 * already stored, when a later overlapping read window hands it over again. The
 * same role and stored text is the same turn.
 */
function turnHash(role: string, text: string): string {
  return createHash('sha256').update(`${role}\n${text}`, 'utf8').digest('hex');
}

/**
 * Mask the secret and keep the rest; a turn that still looks sensitive after
 * masking is stored as the placeholder rather than stored in the clear.
 */
function maskTurnText(text: string): { text: string; masked: boolean } {
  const spans = maskSensitiveSpans(text);
  // Three ways a turn loses all of its text rather than just the secret span:
  //
  // - the masker had to cut a span short (a call longer than its span cap, or a
  //   blank line inside one), so part of the secret may still be in the text while
  //   nothing in it matches a pattern any more — the credential word that gave it
  //   away is inside the span that was masked;
  // - the masked text still trips the detector;
  // - the detector saw a secret in the original that the masker did not touch.
  //   Nothing should reach that last one today, since the two share one pattern
  //   set, so it guards against the two drifting apart.
  if (
    spans.truncated ||
    containsSensitiveText(spans.text) ||
    (spans.masked === 0 && containsSensitiveText(text))
  ) {
    return { text: REDACTED_SOURCE, masked: true };
  }
  return { text: spans.text, masked: spans.masked > 0 };
}

/**
 * A crash mid-append can leave a line with no newline; appending straight after it
 * would splice the next turn into the torn one and lose it too.
 */
function endsWithNewline(path: string): boolean {
  const { size } = statSync(path);
  if (size === 0) return true;
  const handle = openSync(path, 'r');
  try {
    const tail = Buffer.alloc(1);
    readSync(handle, tail, 0, 1, size - 1);
    return tail[0] === 0x0a;
  } finally {
    closeSync(handle);
  }
}

/** A lock whose owner is still running is never taken over, however old it looks. */
function lockOwnerAlive(lockPath: string): boolean {
  try {
    const owner = JSON.parse(readFileSync(lockPath, 'utf8')) as {
      pid?: unknown;
    };
    if (!Number.isSafeInteger(owner.pid) || (owner.pid as number) <= 0) {
      return false;
    }
    try {
      process.kill(owner.pid as number, 0);
      return true;
    } catch (error) {
      // EPERM means the process exists but belongs to someone else.
      return (error as NodeJS.ErrnoException).code === 'EPERM';
    }
  } catch {
    // A crashed writer can leave an empty or partial lock; age alone gates cleanup.
    return false;
  }
}

/**
 * How many bytes of the file a window's turns occupy: their stored lines, newline each.
 * The header line belongs to no window, so the windows of a session sum to a little less
 * than the file.
 */
function storedTurnBytes(turns: readonly SessionTurn[]): number {
  return turns.reduce(
    (total, turn) => total + Buffer.byteLength(JSON.stringify(turn), 'utf8') + 1,
    0,
  );
}

function byStartedAt(a: SessionIndexEntry, b: SessionIndexEntry): number {
  return (
    a.startedAt.localeCompare(b.startedAt) || a.key.localeCompare(b.key)
  );
}

function isTurnRecord(value: unknown): value is SessionTurn {
  if (typeof value !== 'object' || value === null) return false;
  const turn = value as Partial<SessionTurn>;
  return (
    Number.isInteger(turn.index) &&
    (turn.role === 'user' || turn.role === 'assistant') &&
    typeof turn.ts === 'string' &&
    typeof turn.text === 'string' &&
    typeof turn.hash === 'string'
  );
}

/** Window summaries an index entry can be believed about: ordinals 0..n-1, in order. */
function areWindowSummaries(
  value: unknown,
): value is SessionWindowSummary[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (window, ordinal) =>
        typeof window === 'object' &&
        window !== null &&
        (window as SessionWindowSummary).window === ordinal &&
        typeof (window as SessionWindowSummary).startedAt === 'string' &&
        typeof (window as SessionWindowSummary).lastTs === 'string' &&
        Number.isInteger((window as SessionWindowSummary).turns) &&
        Number.isInteger((window as SessionWindowSummary).bytes),
    )
  );
}

function isHeaderRecord(value: unknown): value is SessionHeader {
  if (typeof value !== 'object' || value === null) return false;
  const header = value as Partial<SessionHeader>;
  return (
    header.version === 1 &&
    typeof header.source === 'string' &&
    typeof header.sourceSessionId === 'string' &&
    typeof header.startedAt === 'string'
  );
}

/**
 * Append-only JSONL sessions under `<root>/<namespace>/<key>.jsonl`, with one
 * `index.json` per namespace rewritten atomically after each append. Writing is
 * gated by `REMBERO_SESSIONS`: capture and import ask `sessionsEnabled()` before
 * they call `appendTurns`, so a store built for a test or an explicit root still
 * works while the product setting is off.
 */
export class SessionStore {
  private readonly root: string;
  private readonly capBytes: number;
  private readonly windowLimits: SessionWindowLimits;
  private readonly log: (message: string) => void;
  private readonly heldLocks = new Set<string>();

  constructor(options: SessionStoreOptions = {}) {
    this.root = options.root ?? sessionsRoot();
    this.capBytes = options.capBytes ?? sessionCapBytesFromEnv();
    this.windowLimits = {
      turns: options.windowTurns ?? sessionWindowTurnsFromEnv(),
      gapMs: options.windowGapMs ?? sessionWindowGapMsFromEnv(),
      bytes: options.windowBytes ?? sessionWindowBytesFromEnv(),
    };
    this.log =
      options.log ??
      ((message: string) => {
        process.stderr.write(`${message}\n`);
      });
  }

  sessionsEnabled(): boolean {
    return sessionsEnabledFromEnv();
  }

  sessionKey(source: SessionSource, sourceSessionId: string): string {
    return createHash('sha256')
      .update(`${source}\0${sourceSessionId}`, 'utf8')
      .digest('hex')
      .slice(0, 32);
  }

  appendTurns(
    namespace: string,
    header: SessionHeader,
    turns: Array<{ role: 'user' | 'assistant'; ts: string; text: string }>,
  ): SessionAppendResult {
    assertNamespace(namespace);
    assertHeader(header);
    // The whole batch is checked before anything is written, so a bad turn in the
    // middle of a batch cannot leave half of it stored.
    for (const turn of turns) assertTurn(turn);
    if (turns.length === 0) {
      return { appended: 0, skipped: 0, masked: 0, dropped: [] };
    }
    return this.withNamespaceLock(namespace, () =>
      this.appendTurnsUnlocked(namespace, header, turns),
    );
  }

  private appendTurnsUnlocked(
    namespace: string,
    header: SessionHeader,
    turns: Array<{ role: 'user' | 'assistant'; ts: string; text: string }>,
  ): SessionAppendResult {
    const key = this.sessionKey(header.source, header.sourceSessionId);
    const dir = join(this.root, namespace);
    const path = join(dir, `${key}.jsonl`);
    const state = this.inspectSessionFile(path);
    if (state.kind === 'unreadable') this.quarantine(path, state.reason);
    const existing = state.kind === 'session' ? state : undefined;
    // Only what this session already holds on disk. A batch's own repeats are
    // kept: "ok" typed twice in one conversation is two events, and dropping the
    // second would both lose text and shift every later turn's index, which the
    // reader's ordering depends on. An identical turn re-read across overlapping
    // capture windows is still skipped, which is the point of hashing at all:
    // consecutive stop hooks see the same tail and must store each turn once.
    const seen = new Set(existing?.turns.map((turn) => turn.hash) ?? []);
    // Continue past the highest index the file still holds: a dropped line must
    // not hand its index to a new turn.
    let index = (existing?.turns.at(-1)?.index ?? -1) + 1;
    let appended = 0;
    let skipped = 0;
    let masked = 0;
    const lines: string[] = existing === undefined ? [headerLine(header)] : [];
    for (const turn of turns) {
      const stored = maskTurnText(turn.text);
      if (stored.masked) masked += 1;
      const hash = turnHash(turn.role, stored.text);
      if (seen.has(hash)) {
        skipped += 1;
        continue;
      }
      const record: SessionTurn = {
        index,
        role: turn.role,
        ts: turn.ts,
        text: stored.text,
        hash,
      };
      lines.push(JSON.stringify(record));
      index += 1;
      appended += 1;
    }
    if (lines.length > 0) {
      mkdirSync(dir, { recursive: true, mode: 0o700 });
      const prefix =
        existing !== undefined && !endsWithNewline(path) ? '\n' : '';
      appendFileSync(path, `${prefix}${lines.join('\n')}\n`, {
        encoding: 'utf8',
        mode: 0o600,
      });
    }
    const dropped = this.refreshIndex(namespace, dir, key);
    return { appended, skipped, masked, dropped };
  }

  /**
   * One stored conversation, or one window of it.
   *
   * `<key>` is the whole session, in file order, exactly as before windowing: the file
   * is the conversation, and a caller that wants all of it still asks for it that way.
   * `<key>#<n>` is that window alone — the unit retrieval ranks and the reader reads —
   * and is `undefined` when the session holds no such window.
   */
  readSession(
    namespace: string,
    id: string,
  ): { header: SessionHeader; turns: SessionTurn[] } | undefined {
    assertNamespace(namespace);
    const { key, window } = parseSessionWindowId(id);
    assertSessionKey(key);
    const session = this.readSessionFile(
      join(this.root, namespace, `${key}.jsonl`),
    );
    if (session === undefined || window === undefined) return session;
    const turns = this.windowsOf(session.turns)[window];
    return turns === undefined ? undefined : { header: session.header, turns };
  }

  /**
   * Every window of one session, from a single read of its file.
   *
   * Reading the windows one at a time would parse a 346 KB file once per window, and
   * the reading path wants all of them: this is the call it makes.
   */
  readSessionWindows(namespace: string, key: string): SessionWindowRead[] {
    assertNamespace(namespace);
    assertSessionKey(key);
    const session = this.readSessionFile(
      join(this.root, namespace, `${key}.jsonl`),
    );
    if (session === undefined) return [];
    const grouped = this.windowsOf(session.turns);
    return grouped.map((turns, window) => ({
      key,
      window,
      windowKey: sessionWindowKey(key, window),
      windows: grouped.length,
      header: session.header,
      startedAt: turns[0]?.ts ?? session.header.startedAt,
      lastTs: turns.at(-1)?.ts ?? session.header.startedAt,
      turns,
    }));
  }

  /** This store's window rule, applied to one session's turns in file order. */
  private windowsOf(turns: readonly SessionTurn[]): SessionTurn[][] {
    return windowSessionTurns(turns, this.windowLimits);
  }

  /**
   * Every namespace that holds a session directory, sorted.
   *
   * A `'*'` recall expands through the fact store, which knows nothing about a namespace
   * that holds conversations and not one fact — an imported transcript's namespace, most
   * obviously. So the session store has to be asked too, and the two lists unioned.
   */
  listNamespaces(): string[] {
    let names: string[];
    try {
      names = readdirSync(this.root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name);
    } catch {
      // no sessions root yet: nothing is stored, which is not a failure
      return [];
    }
    return names.filter((name) => NAMESPACE_PATTERN.test(name)).sort();
  }

  /**
   * The namespace's windows, oldest session first and each session's windows in order.
   *
   * This is the unit: `sessions list` shows a window's own time range and turn count,
   * and the reading path ranks one candidate per window. Every entry still carries the
   * session key, so what a user sees here is what `forget_sessions` takes.
   *
   * The summaries come from the index, which the append wrote them into. A session whose
   * entry predates windowing has none, and its file is windowed here instead — one read
   * per such session, until the next append records them.
   */
  list(namespace: string): SessionWindowEntry[] {
    return this.listSessions(namespace).flatMap((entry) => {
      const windows = entry.windows ?? this.windowSummaries(namespace, entry);
      return windows.map((window) => ({
        ...window,
        key: entry.key,
        windowKey: sessionWindowKey(entry.key, window.window),
        source: entry.source,
        windows: windows.length,
      }));
    });
  }

  /** The namespace's whole sessions: one entry per file, as the byte cap counts them. */
  listSessions(namespace: string): SessionIndexEntry[] {
    assertNamespace(namespace);
    const dir = join(this.root, namespace);
    if (!existsSync(dir)) return [];
    return this.readIndex(dir).sort(byStartedAt);
  }

  /** A pre-windowing entry's windows, derived from its file. */
  private windowSummaries(
    namespace: string,
    entry: SessionIndexEntry,
  ): SessionWindowSummary[] {
    const session = this.readSessionFile(
      join(this.root, namespace, `${entry.key}.jsonl`),
    );
    return this.summariesOf(session?.turns ?? [], entry.startedAt);
  }

  /** One session's windows as the index records them. */
  private summariesOf(
    turns: readonly SessionTurn[],
    startedAt: string,
  ): SessionWindowSummary[] {
    const grouped = this.windowsOf(turns);
    // A header with no turn still names a session the user can see and forget, so it
    // gets one empty window rather than disappearing from the listing.
    if (grouped.length === 0) {
      return [
        { window: 0, startedAt, lastTs: startedAt, turns: 0, bytes: 0 },
      ];
    }
    return grouped.map((window, ordinal) => ({
      window: ordinal,
      startedAt: window[0]!.ts,
      lastTs: window.at(-1)!.ts,
      turns: window.length,
      bytes: storedTurnBytes(window),
    }));
  }

  /**
   * Forget one stored conversation: the file, its index entry and every quarantined copy.
   *
   * A window id (`<key>#<n>`) forgets the whole session too. Windows are a reading unit
   * over one append-only file, not files of their own, and the user is handed window ids
   * by recall: taking one and deleting only part of the conversation would need the file
   * rewritten, and answering "forget that" with "some of it is still here" is worse than
   * the plain reading. `forget_sessions` documents the session key for this reason.
   */
  deleteSession(namespace: string, id: string): boolean {
    assertNamespace(namespace);
    const { key } = parseSessionWindowId(id);
    assertSessionKey(key);
    return this.withNamespaceLock(namespace, () => {
      const dir = join(this.root, namespace);
      const path = join(dir, `${key}.jsonl`);
      // A quarantined sibling can outlive the live file, so the copies are removed
      // whether or not <key>.jsonl is still there: forgetting must leave nothing
      // readable, and `sessions list` never showed these to say they were there.
      const quarantined = this.unlinkQuarantined(dir, key);
      if (!existsSync(path)) return quarantined;
      unlinkSync(path);
      this.writeIndex(
        dir,
        this.readIndex(dir)
          .filter((entry) => entry.key !== key)
          .sort(byStartedAt),
      );
      return true;
    });
  }

  deleteNamespace(namespace: string): number {
    assertNamespace(namespace);
    return this.withNamespaceLock(namespace, () => {
      const dir = join(this.root, namespace);
      if (!existsSync(dir)) return 0;
      const removed = this.sessionFiles(dir).length;
      rmSync(dir, { recursive: true, force: true });
      return removed;
    });
  }

  private sessionFiles(dir: string): string[] {
    return readdirSync(dir).filter((name) =>
      SESSION_KEY_PATTERN.test(name.replace(/\.jsonl$/, '')) &&
      name.endsWith('.jsonl'),
    );
  }

  /**
   * A torn or missing header must not brick the session, nor the namespace's index
   * rebuild, so the state is reported rather than thrown: the reader skips an
   * unreadable file and `appendTurns` quarantines it and starts a fresh one.
   */
  private inspectSessionFile(path: string): SessionFileState {
    if (!existsSync(path)) return { kind: 'absent' };
    const lines = readFileSync(path, 'utf8')
      .split('\n')
      .filter((line) => line.trim() !== '');
    if (lines.length === 0) return { kind: 'absent' };
    let header: unknown;
    try {
      header = JSON.parse(lines[0]);
    } catch {
      return { kind: 'unreadable', reason: 'its header line is unreadable' };
    }
    if (!isHeaderRecord(header)) {
      return {
        kind: 'unreadable',
        reason: 'its first line is not a readable session header',
      };
    }
    const turns: SessionTurn[] = [];
    for (const line of lines.slice(1)) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        // A crash between the write and the rename can leave a torn last line.
        this.log(`rembero sessions: ignoring an unreadable turn in ${path}`);
        continue;
      }
      if (isTurnRecord(parsed)) {
        turns.push(parsed);
      } else {
        this.log(
          `rembero sessions: ignoring a line that is not a turn in ${path}`,
        );
      }
    }
    return { kind: 'session', header, turns };
  }

  private readSessionFile(
    path: string,
  ): { header: SessionHeader; turns: SessionTurn[] } | undefined {
    const state = this.inspectSessionFile(path);
    if (state.kind === 'session') {
      return { header: state.header, turns: state.turns };
    }
    if (state.kind === 'unreadable') {
      this.log(`rembero sessions: skipping ${path} because ${state.reason}`);
    }
    return undefined;
  }

  /**
   * Move an unreadable file aside so a fresh session can take its place.
   *
   * A second tear must not overwrite the first quarantined copy: those turns are the
   * user's, they are all that is left of them, and destroying them silently is the one
   * thing quarantine exists to avoid. So the name is suffixed until it is free. The
   * search is bounded — a namespace that has somehow torn a thousand times gets a
   * random name rather than an unbounded loop — and `deleteSession` removes every
   * suffix, so a forgotten session leaves none of them behind.
   */
  private quarantine(path: string, reason: string): void {
    let target = `${path}${QUARANTINE_SUFFIX}`;
    for (let attempt = 1; existsSync(target); attempt += 1) {
      if (attempt >= MAX_QUARANTINE_SUFFIXES) {
        target = `${path}${QUARANTINE_SUFFIX}.${randomUUID()}`;
        break;
      }
      target = `${path}${QUARANTINE_SUFFIX}.${attempt}`;
    }
    renameSync(path, target);
    this.log(
      `rembero sessions: quarantined ${path} as ${target} because ${reason}; ` +
        'a fresh session file starts in its place',
    );
  }

  /** Recompute this session's entry, then drop whole sessions until the namespace fits. */
  private refreshIndex(namespace: string, dir: string, key: string): string[] {
    const entry = this.entryFor(dir, key);
    const entries = this.readIndex(dir).filter((other) => other.key !== key);
    if (entry !== undefined) entries.push(entry);
    let total = entries.reduce((sum, other) => sum + other.bytes, 0);
    const dropped = new Set<string>();
    if (total > this.capBytes) {
      // Never the session just written: a single session larger than the cap
      // would otherwise be deleted by the append that created it.
      for (const candidate of entries
        .filter((other) => other.key !== key)
        .sort(byStartedAt)) {
        if (total <= this.capBytes) break;
        this.unlinkIfPresent(join(dir, `${candidate.key}.jsonl`));
        // the same reasoning as deleteSession: an evicted session leaves no readable copy
        this.unlinkQuarantined(dir, candidate.key);
        total -= candidate.bytes;
        dropped.add(candidate.key);
        this.log(
          `rembero sessions: dropped ${candidate.key} from namespace ${namespace} ` +
            `(${candidate.bytes} bytes, started ${candidate.startedAt}) to stay under ` +
            `the ${this.capBytes} byte cap`,
        );
      }
    }
    this.writeIndex(
      dir,
      entries.filter((other) => !dropped.has(other.key)).sort(byStartedAt),
    );
    return [...dropped];
  }

  private entryFor(dir: string, key: string): SessionIndexEntry | undefined {
    const path = join(dir, `${key}.jsonl`);
    const session = this.readSessionFile(path);
    if (session === undefined) return undefined;
    const last = session.turns.at(-1);
    return {
      key,
      source: session.header.source,
      startedAt: session.header.startedAt,
      lastTs: last?.ts ?? session.header.startedAt,
      turns: session.turns.length,
      bytes: statSync(path).size,
      // Written at append time, from the turns now on disk. The rule is a left fold, so
      // an append extends the last window or opens a new one and renumbers nothing.
      windows: this.summariesOf(session.turns, session.header.startedAt),
    };
  }

  /** `index.json` when it is readable, otherwise rebuilt from the session files. */
  private readIndex(dir: string): SessionIndexEntry[] {
    const path = join(dir, INDEX_FILE);
    if (existsSync(path)) {
      try {
        const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'));
        const sessions = (parsed as { sessions?: unknown }).sessions;
        if (Array.isArray(sessions)) {
          return sessions
            .filter(
              (entry): entry is SessionIndexEntry =>
                typeof entry === 'object' &&
                entry !== null &&
                SESSION_KEY_PATTERN.test(String((entry as SessionIndexEntry).key)) &&
                existsSync(join(dir, `${(entry as SessionIndexEntry).key}.jsonl`)),
            )
            // An entry whose windows cannot be read — an old index that has none, or a
            // hand-edited one — keeps everything else it says, and `list` derives its
            // windows from the file instead of showing a broken time range.
            .map((entry) =>
              areWindowSummaries(entry.windows)
                ? entry
                : ({
                    key: entry.key,
                    source: entry.source,
                    startedAt: entry.startedAt,
                    lastTs: entry.lastTs,
                    turns: entry.turns,
                    bytes: entry.bytes,
                  } satisfies SessionIndexEntry),
            );
        }
      } catch {
        this.log(`rembero sessions: rebuilding an unreadable ${path}`);
      }
    }
    if (!existsSync(dir)) return [];
    const rebuilt: SessionIndexEntry[] = [];
    for (const name of this.sessionFiles(dir)) {
      const entry = this.entryFor(dir, name.replace(/\.jsonl$/, ''));
      if (entry !== undefined) rebuilt.push(entry);
    }
    return rebuilt;
  }

  private writeIndex(dir: string, sessions: SessionIndexEntry[]): void {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    const path = join(dir, INDEX_FILE);
    const tmp = `${path}.tmp-${process.pid}-${randomUUID()}`;
    const text = `${JSON.stringify({ version: 1, sessions }, null, 2)}\n`;
    try {
      writeFileSync(tmp, text, { encoding: 'utf8', mode: 0o600, flag: 'wx' });
      renameSync(tmp, path);
    } catch (error) {
      this.unlinkIfPresent(tmp);
      throw error;
    }
  }

  /**
   * One writer per namespace. The index is read, modified and rewritten, and the
   * turn hashes are read before the append, so two writers in one namespace — a
   * second stop hook, or capture while `sessions import` runs — would otherwise
   * lose an index entry or duplicate a turn, last writer winning.
   *
   * This mirrors `MemoryStore.withLock` (src/store/store.ts:935) rather than
   * calling it: that method is private to a class built around the fact store's
   * root, and `src/store/store.ts` is not mine to change in this pass. Lifting the
   * two into one `src/store/file-lock.ts` is the obvious follow-up.
   */
  private withNamespaceLock<T>(namespace: string, operation: () => T): T {
    const name = `session-${namespace}`;
    if (this.heldLocks.has(name)) return operation();
    mkdirSync(this.root, { recursive: true, mode: 0o700 });
    const lockPath = join(this.root, `.${name}.lock`);
    const deadline = Date.now() + LOCK_WAIT_MS;
    // Every retry path checks this, including the ones that loop straight back:
    // a lock being created and removed under us must not spin without end.
    const assertDeadline = (): void => {
      if (Date.now() >= deadline) {
        throw new Error(`timed out waiting for session lock '${name}'`);
      }
    };
    let descriptor: number | undefined;
    let ownedDevice: number | undefined;
    let ownedInode: number | undefined;
    while (descriptor === undefined) {
      try {
        const acquired = openSync(lockPath, 'wx', 0o600);
        try {
          writeFileSync(
            acquired,
            `${JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() })}\n`,
            'utf8',
          );
          const owned = fstatSync(acquired);
          ownedDevice = owned.dev;
          ownedInode = owned.ino;
          descriptor = acquired;
        } catch (error) {
          closeSync(acquired);
          this.unlinkIfPresent(lockPath);
          throw error;
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
        try {
          const lock = lstatSync(lockPath);
          if (lock.isSymbolicLink()) {
            throw new Error(`refusing symbolic-link lock file ${lockPath}`);
          }
          if (
            Date.now() - lock.mtimeMs > LOCK_STALE_MS &&
            !lockOwnerAlive(lockPath)
          ) {
            unlinkSync(lockPath);
            assertDeadline();
            continue;
          }
        } catch (statError) {
          if ((statError as NodeJS.ErrnoException).code === 'ENOENT') {
            // The owner released it between our create and our stat.
            assertDeadline();
            continue;
          }
          throw statError;
        }
        assertDeadline();
        Atomics.wait(sleepCell, 0, 0, LOCK_RETRY_MS);
      }
    }
    try {
      this.heldLocks.add(name);
      return operation();
    } finally {
      this.heldLocks.delete(name);
      closeSync(descriptor);
      try {
        const current = lstatSync(lockPath);
        if (current.dev === ownedDevice && current.ino === ownedInode) {
          unlinkSync(lockPath);
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
    }
  }

  /** Remove every quarantined copy of one session. True when there was at least one. */
  private unlinkQuarantined(dir: string, key: string): boolean {
    let removed = false;
    let names: string[];
    try {
      names = readdirSync(dir);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
      throw error;
    }
    for (const name of names) {
      if (!quarantinedName(key, name)) continue;
      this.unlinkIfPresent(join(dir, name));
      removed = true;
    }
    return removed;
  }

  private unlinkIfPresent(path: string): void {
    try {
      unlinkSync(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }
}
