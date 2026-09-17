import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, type Stats } from 'node:fs';
import { basename, resolve } from 'node:path';
import {
  parseTranscriptMessages,
  type TranscriptMessage,
} from '../autocapture/transcript.js';
import type { SessionStore } from './store.js';

/**
 * Import reads a whole transcript into memory, so the file is bounded first. A
 * Claude Code transcript of this size holds tens of thousands of turns; anything
 * larger is a mistaken path rather than a conversation.
 */
export const MAX_IMPORT_TRANSCRIPT_BYTES = 64 * 1024 * 1024;

/** The same shape `parseClaudeStopHookInput` accepts for a Claude session id. */
const SESSION_ID_PATTERN = /^[a-zA-Z0-9._-]{1,256}$/;

/**
 * The `sessionId` Claude Code writes on every line of a transcript, from the
 * first line that carries a usable one.
 *
 * This is read here rather than taken from `parseTranscriptMessages`, which
 * returns turns and no file-level metadata. The scan walks lines by index instead
 * of splitting, because the file can be tens of megabytes and the answer is almost
 * always on line one.
 */
function transcriptSessionId(text: string): string | undefined {
  let start = 0;
  while (start < text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const line = text.slice(start, end);
    start = end + 1;
    if (line.trim() === '') continue;
    let entry: unknown;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      continue;
    }
    const id = (entry as { sessionId?: unknown }).sessionId;
    if (typeof id === 'string' && SESSION_ID_PATTERN.test(id)) return id;
  }
  return undefined;
}

/**
 * What identifies this transcript's conversation.
 *
 * The transcript's own `sessionId` when it has one: two files copied out of one
 * conversation are then one session however they were named, and two unrelated
 * conversations that happen to share a file name — `/a/chat.jsonl` and
 * `/b/chat.jsonl` — stay apart, where a name alone would have merged them under
 * whichever header was written first.
 *
 * Failing that, the file name plus a digest of its opening turn. That digest is
 * over the start of the conversation, not the whole file, for the two properties
 * import needs at once: it does not move when the same file is imported again, and
 * it does not move when the file has since grown, so a transcript re-imported
 * after more conversation appends its new turns to the session it already has.
 */
function importSessionId(
  text: string,
  file: string,
  first: TranscriptMessage | undefined,
): string {
  const carried = transcriptSessionId(text);
  if (carried !== undefined) return carried;
  const name = basename(file).replace(/\.[^.]+$/, '');
  if (first === undefined) return name;
  const opening = createHash('sha256')
    .update(`${first.role}\n${first.text}`, 'utf8')
    .digest('hex')
    .slice(0, 12);
  return `${name}-${opening}`;
}

/** A path that is missing, a link, or not a file at all, said plainly. */
function transcriptStat(file: string): Stats {
  let stat: Stats;
  try {
    stat = lstatSync(file);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error(`no transcript file at ${file}`);
    }
    throw error;
  }
  // A link is named as a link rather than lumped in with the rest: it is the one
  // refusal a user is likely to have meant something reasonable by.
  if (stat.isSymbolicLink()) {
    throw new Error(`refusing a symbolic link as a transcript file: ${file}`);
  }
  if (!stat.isFile()) {
    throw new Error(`refusing a non-regular transcript file: ${file}`);
  }
  return stat;
}

export interface SessionImportResult {
  /** The stored session's key, `sha256('import\0<session id>')` truncated. */
  key: string;
  appended: number;
  skipped: number;
}

/**
 * Store one Claude Code transcript file as an imported session.
 *
 * Unlike capture, which re-reads an overlapping tail after every stop hook, an
 * import is explicit and one-off: it reads the whole file, so a conversation
 * whose early turns are long past the tail window still lands. Re-importing the
 * same file appends nothing, because the store skips a turn whose hash this
 * session already holds; see `importSessionId` for what makes two files the same
 * conversation.
 */
export function importClaudeTranscript(
  store: SessionStore,
  namespace: string,
  path: string,
): SessionImportResult {
  // The store writes whenever it is asked to, so the gate is here: importing is
  // the one path that puts a past conversation on disk, and the setting is what
  // the user consented with.
  if (!store.sessionsEnabled()) {
    throw new Error(
      'importing sessions needs REMBERO_SESSIONS=on; nothing was written',
    );
  }
  const file = resolve(path);
  const stat = transcriptStat(file);
  if (stat.size > MAX_IMPORT_TRANSCRIPT_BYTES) {
    throw new Error(
      `transcript ${file} exceeds ${MAX_IMPORT_TRANSCRIPT_BYTES} bytes`,
    );
  }
  const text = readFileSync(file, 'utf8');
  const turns = parseTranscriptMessages(text);
  const sourceSessionId = importSessionId(text, file, turns[0]);
  if (sourceSessionId === '') {
    throw new Error(`transcript ${file} has no usable session id or file name`);
  }
  const key = store.sessionKey('import', sourceSessionId);
  // Every stored `ts` must satisfy `Date.parse`, because the index's `lastTs` and
  // the reader's date arithmetic come from it. A transcript entry carries its own
  // timestamp when it has a usable one; otherwise the whole file is dated by its
  // modification time, which is the closest thing to when the conversation ended.
  const fileTime = stat.mtime.getTime();
  const fallbackTs = new Date(
    Number.isNaN(fileTime) ? Date.now() : fileTime,
  ).toISOString();
  const stamped = turns.map((turn) => ({
    role: turn.role,
    ts: turn.ts ?? fallbackTs,
    text: turn.text,
  }));
  if (stamped.length === 0) return { key, appended: 0, skipped: 0 };
  const result = store.appendTurns(
    namespace,
    {
      version: 1,
      source: 'import',
      sourceSessionId,
      startedAt: stamped[0].ts,
    },
    stamped,
  );
  return { key, appended: result.appended, skipped: result.skipped };
}
