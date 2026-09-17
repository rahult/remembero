import { createHash, randomUUID } from 'node:crypto';
import {
  appendFileSync,
  closeSync,
  existsSync,
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
import { sessionCapBytesFromEnv, sessionsEnabledFromEnv } from '../env.js';
import {
  containsSensitiveText,
  maskSensitiveSpans,
  REDACTED_SOURCE,
} from '../safety.js';

/** A namespace becomes a directory name, so nothing but these characters is allowed. */
const NAMESPACE_PATTERN = /^[a-z0-9_-]+$/;
/** A session key is the first 32 hex characters of a digest, so no id can escape the directory. */
const SESSION_KEY_PATTERN = /^[0-9a-f]{32}$/;
const INDEX_FILE = 'index.json';

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

export interface SessionIndexEntry {
  key: string;
  source: SessionSource;
  startedAt: string;
  lastTs: string;
  turns: number;
  bytes: number;
}

export interface SessionAppendResult {
  appended: number;
  skipped: number;
  masked: number;
  dropped: string[];
}

export interface SessionStoreOptions {
  root?: string;
  capBytes?: number;
  /** Where cap drops are reported; stderr by default. */
  log?: (message: string) => void;
}

/** `${REMBERO_HOME ?? ~/.rembero}/sessions`, beside the fact store's `memory/`. */
export function sessionsRoot(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.REMBERO_HOME ?? join(homedir(), '.rembero'), 'sessions');
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

/** A turn's identity: the same role and stored text is the same turn. */
function turnHash(role: string, text: string): string {
  return createHash('sha256').update(`${role}\n${text}`, 'utf8').digest('hex');
}

/**
 * Mask the secret and keep the rest; a turn that still looks sensitive after
 * masking is stored as the placeholder rather than stored in the clear.
 */
function maskTurnText(text: string): { text: string; masked: boolean } {
  const spans = maskSensitiveSpans(text);
  if (containsSensitiveText(spans.text)) {
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
  private readonly log: (message: string) => void;

  constructor(options: SessionStoreOptions = {}) {
    this.root = options.root ?? sessionsRoot();
    this.capBytes = options.capBytes ?? sessionCapBytesFromEnv();
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
    const key = this.sessionKey(header.source, header.sourceSessionId);
    if (turns.length === 0) {
      return { appended: 0, skipped: 0, masked: 0, dropped: [] };
    }
    const dir = join(this.root, namespace);
    const path = join(dir, `${key}.jsonl`);
    const existing = this.readSessionFile(path);
    const seen = new Set(existing?.turns.map((turn) => turn.hash) ?? []);
    let index = existing?.turns.length ?? 0;
    let appended = 0;
    let skipped = 0;
    let masked = 0;
    const lines: string[] = existing === undefined ? [headerLine(header)] : [];
    for (const turn of turns) {
      if (turn.role !== 'user' && turn.role !== 'assistant') {
        throw new Error(`session turn role must be user or assistant: ${turn.role}`);
      }
      const stored = maskTurnText(turn.text);
      if (stored.masked) masked += 1;
      const hash = turnHash(turn.role, stored.text);
      if (seen.has(hash)) {
        skipped += 1;
        continue;
      }
      seen.add(hash);
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

  readSession(
    namespace: string,
    key: string,
  ): { header: SessionHeader; turns: SessionTurn[] } | undefined {
    assertNamespace(namespace);
    assertSessionKey(key);
    return this.readSessionFile(join(this.root, namespace, `${key}.jsonl`));
  }

  list(namespace: string): SessionIndexEntry[] {
    assertNamespace(namespace);
    const dir = join(this.root, namespace);
    if (!existsSync(dir)) return [];
    return this.readIndex(dir).sort(byStartedAt);
  }

  deleteSession(namespace: string, key: string): boolean {
    assertNamespace(namespace);
    assertSessionKey(key);
    const dir = join(this.root, namespace);
    const path = join(dir, `${key}.jsonl`);
    if (!existsSync(path)) return false;
    unlinkSync(path);
    this.writeIndex(
      dir,
      this.readIndex(dir)
        .filter((entry) => entry.key !== key)
        .sort(byStartedAt),
    );
    return true;
  }

  deleteNamespace(namespace: string): number {
    assertNamespace(namespace);
    const dir = join(this.root, namespace);
    if (!existsSync(dir)) return 0;
    const removed = this.sessionFiles(dir).length;
    rmSync(dir, { recursive: true, force: true });
    return removed;
  }

  private sessionFiles(dir: string): string[] {
    return readdirSync(dir).filter((name) =>
      SESSION_KEY_PATTERN.test(name.replace(/\.jsonl$/, '')) &&
      name.endsWith('.jsonl'),
    );
  }

  private readSessionFile(
    path: string,
  ): { header: SessionHeader; turns: SessionTurn[] } | undefined {
    if (!existsSync(path)) return undefined;
    const lines = readFileSync(path, 'utf8')
      .split('\n')
      .filter((line) => line.trim() !== '');
    if (lines.length === 0) return undefined;
    const header: unknown = JSON.parse(lines[0]);
    if (!isHeaderRecord(header)) {
      throw new Error(`session file has no header line: ${path}`);
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
      if (isTurnRecord(parsed)) turns.push(parsed);
    }
    return { header, turns };
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
          return sessions.filter(
            (entry): entry is SessionIndexEntry =>
              typeof entry === 'object' &&
              entry !== null &&
              SESSION_KEY_PATTERN.test(String((entry as SessionIndexEntry).key)) &&
              existsSync(join(dir, `${(entry as SessionIndexEntry).key}.jsonl`)),
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

  private unlinkIfPresent(path: string): void {
    try {
      unlinkSync(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }
}
