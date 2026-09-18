/**
 * Cutting one stored conversation into windows: the unit retrieval ranks and the
 * reader reads.
 *
 * A session file stays one file per conversation, because that is what identity,
 * idempotent re-import and `forget_sessions` are built on. What changes is the unit
 * above it. Measured on a real Claude Code transcript: 468 turns and 346 KB over five
 * days in one session, against about 30 turns and a few kilobytes for a LongMemEval
 * session. With one session in a namespace, ranking whole sessions has nothing to
 * choose between, and the product's 24,576-byte reading budget then keeps the first
 * ~7% of the file — its continuation summary — while the 35 turns that mentioned the
 * answer were never shown.
 *
 * So a window is a stretch of consecutive turns: at most `turns` of them, and never
 * spanning a silence longer than `gapMs`. Both bounds matter. The turn count keeps a
 * window near the size the reader was trained on; the gap keeps yesterday's thread out
 * of today's, which is what makes a window's own date worth showing.
 *
 * The rule is a left fold over the turns in file order, so a window's membership never
 * depends on what comes after it: appending turns extends the last window or opens a
 * new one and renumbers nothing. That is what lets the same function assign windows at
 * write time and re-derive them when reading a file written before windowing existed.
 */

/**
 * The three bounds; the `sessionWindow*FromEnv` readers in env.ts carry the product's
 * defaults and the settings that change them.
 */
export interface SessionWindowLimits {
  /** At most this many turns in one window. */
  turns: number;
  /** A gap longer than this between consecutive turns starts a new window. */
  gapMs: number;
  /**
   * At most this much turn text in one window.
   *
   * The turn count alone is not enough, because one turn is not one size: in the
   * measured transcript a pasted continuation summary ran to 16.8 KB on its own, and a
   * 40-turn window around it came to 63 KB. The reader only ever sees the first 16,384
   * characters of a source (`READING_SOURCE_CHARACTERS`) and only its share of the
   * reading budget of that, so everything after that one turn was unreachable however
   * well it ranked. A window bounded in bytes is a window the reader can be shown whole.
   */
  bytes: number;
}

/** `<key>#<window>`, the id `readSession` takes and recall reports. */
export const SESSION_WINDOW_SEPARATOR = '#';

export function sessionWindowKey(key: string, window: number): string {
  return `${key}${SESSION_WINDOW_SEPARATOR}${window}`;
}

/**
 * A session id split into the file it names and the window it names, if any.
 *
 * `undefined` for a window means the whole session: that is what `readSession` was
 * given before windowing and what `forget_sessions` is documented to take, and both
 * must keep working.
 */
export function parseSessionWindowId(id: string): {
  key: string;
  window?: number;
} {
  const separator = id.indexOf(SESSION_WINDOW_SEPARATOR);
  if (separator === -1) return { key: id };
  const ordinal = id.slice(separator + 1);
  if (!/^\d+$/.test(ordinal)) {
    throw new Error(`session window must be a number: ${id}`);
  }
  const window = Number(ordinal);
  if (!Number.isSafeInteger(window)) {
    throw new Error(`session window is out of range: ${id}`);
  }
  return { key: id.slice(0, separator), window };
}

/**
 * The turns of one session, grouped into windows, in file order.
 *
 * A turn is never split across windows — a window is a list of whole turn records — so
 * concatenating the result gives back exactly the input, in order. A turn longer than
 * the byte bound therefore keeps its window to itself rather than being cut: one
 * enormous paste must not take the turns around it out of reach, and cutting it would
 * store something the user never said. An unparseable `ts` (nothing the store writes,
 * but a hand-edited file can hold one) carries the previous turn's time rather than
 * opening a window on a gap nobody can measure.
 */
export function windowSessionTurns<T extends { ts: string; text?: string }>(
  turns: readonly T[],
  limits: SessionWindowLimits,
): T[][] {
  const windows: T[][] = [];
  let current: T[] = [];
  let used = 0;
  let previous: number | undefined;
  for (const turn of turns) {
    const at = Date.parse(turn.ts);
    const time = Number.isNaN(at) ? previous : at;
    // A jump in either direction ends the window: capture sorts a batch by time, but a
    // file that already holds an out-of-order pair should still keep the two sittings
    // apart rather than fold a whole day into one window.
    const gap =
      previous === undefined || time === undefined
        ? 0
        : Math.abs(time - previous);
    const bytes = Buffer.byteLength(turn.text ?? '', 'utf8');
    if (
      current.length >= limits.turns ||
      (current.length > 0 && gap > limits.gapMs) ||
      (current.length > 0 && used + bytes > limits.bytes)
    ) {
      windows.push(current);
      current = [];
      used = 0;
    }
    current.push(turn);
    used += bytes;
    previous = time;
  }
  if (current.length > 0) windows.push(current);
  return windows;
}
