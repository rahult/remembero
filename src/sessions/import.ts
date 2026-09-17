import { lstatSync, readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { parseTranscriptMessages } from '../autocapture/transcript.js';
import type { SessionStore } from './store.js';

/**
 * Import reads a whole transcript into memory, so the file is bounded first. A
 * Claude Code transcript of this size holds tens of thousands of turns; anything
 * larger is a mistaken path rather than a conversation.
 */
export const MAX_IMPORT_TRANSCRIPT_BYTES = 64 * 1024 * 1024;

export interface SessionImportResult {
  /** The stored session's key, `sha256('import\0<file basename>')` truncated. */
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
 * same file appends nothing, because the store identifies a turn by the hash of
 * its role and stored text.
 *
 * The session is keyed on the file's basename without its extension, so the same
 * transcript imported twice is one session whatever directory it was copied to,
 * while two differently named files stay apart.
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
  const stat = lstatSync(file);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`refusing non-regular transcript file ${file}`);
  }
  if (stat.size > MAX_IMPORT_TRANSCRIPT_BYTES) {
    throw new Error(
      `transcript ${file} exceeds ${MAX_IMPORT_TRANSCRIPT_BYTES} bytes`,
    );
  }
  const sourceSessionId = basename(file).replace(/\.[^.]+$/, '');
  if (sourceSessionId === '') {
    throw new Error(`transcript ${file} has no usable file name`);
  }
  const key = store.sessionKey('import', sourceSessionId);
  const turns = parseTranscriptMessages(readFileSync(file, 'utf8'));
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
