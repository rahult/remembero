import { createHash } from 'node:crypto';
import type { LlmClient } from '../llm/client.js';
import { rememberTranscriptText } from '../llm/pipeline.js';
import type { MemoryStore } from '../store/store.js';
import type { IntegrityEnforcementOptions } from '../knowledge/enforcement.js';
import type { KnowledgeCheckEnforcementOptions } from '../knowledge/check-enforcement.js';
import type { EntityIdentityMode } from '../knowledge/identity.js';
import {
  DEFAULT_AUTO_CAPTURE_DAILY_CAP,
  validateAutoCaptureDailyCap,
  validateAutoCaptureTailBytes,
} from './hooks.js';
import {
  DEFAULT_TRANSCRIPT_TAIL_BYTES,
  parseClaudeStopHookInput,
  readClaudeTranscriptTail,
  transcriptWindowBytes,
} from './transcript.js';
import type { SessionStore } from '../sessions/store.js';

export interface AutoCaptureDeps {
  store: MemoryStore;
  llm: LlmClient;
  /**
   * Optional conversation store. Present only when `REMBERO_SESSIONS=on`, and
   * asked again here: the store writes whatever it is handed, so the setting is
   * checked on the way in rather than inside it.
   */
  sessions?: SessionStore;
  llmAllowedNamespaces?: ReadonlySet<string>;
  integrityEnforcement?: IntegrityEnforcementOptions | false;
  knowledgeCheckEnforcement?: KnowledgeCheckEnforcementOptions | false;
  entityIdentity?: EntityIdentityMode | false;
  /** Constant naming the speaker in captured text (default 'user'; REMBERO_SELF). */
  selfAtom?: string;
  /** 'closed' restricts captured facts to predicates already in the schema. */
  extractionVocabulary?: 'open' | 'closed';
}

export interface AutoCaptureOptions {
  namespace?: string;
  dailyCap?: number;
  tailBytes?: number;
  claudeConfigDir?: string;
  now?: Date;
}

export interface AutoCaptureResult {
  captureId: string;
  status: 'captured' | 'empty' | 'skipped';
  added: string[];
  duplicates: number;
  reason?: 'duplicate' | 'daily_cap' | 'no_user_text';
  /** Turns written to the session store, when one is configured and it accepted them. */
  sessionTurns?: { appended: number; skipped: number };
  /** Why the session write was given up on; the capture itself still succeeded. */
  sessionError?: string;
}

/** Session turns, by the store's `appendTurns` signature. */
type StampedTurn = { role: 'user' | 'assistant'; ts: string; text: string };

/**
 * The turns this session does not already hold, in `ts` order, and the window's
 * own start whether or not its first turn is one of them.
 *
 * Two things have to be kept off every stop hook. One is old turns: a window with
 * no user text is re-read four times further back (up to 16 MB), so a capture can
 * see turns far older than the ones already stored, and appending those would put
 * a newer `index` on an older `ts` — the file order `readSession` returns would
 * stop matching time. `lastTs` comes from the namespace index rather than the
 * session file, so the high-water mark costs one small read instead of a second
 * parse of a file that grows without bound. The other is volume: the store masks
 * and hashes every turn it is handed, inside the namespace lock, so a widened
 * window of megabytes would be re-masked on every hook.
 *
 * The budget for that is the read window's own `transcriptWindowBytes`, not the
 * extraction tail's much smaller `tailBytes`. Everything an unwidened window
 * holds fits in it by construction, so no new turn is dropped while a session
 * slides forward — which matters because the `ts` floor makes a drop permanent,
 * and because a hook whose write failed must be able to recover its turns from
 * the next window. Only a widened window can exceed the budget, and that is the
 * first capture into a long conversation, where storing the newest part of it is
 * the point. A turn larger than the whole budget is kept on its own account
 * rather than spending it, so one code-heavy assistant answer cannot evict the
 * user's message beside it.
 */
function sessionTurnsToWrite(
  sessions: SessionStore,
  namespace: string,
  sourceSessionId: string,
  turns: Array<{ role: 'user' | 'assistant'; ts?: string; text: string }>,
  at: string,
  tailBytes: number,
): { startedAt?: string; turns: StampedTurn[] } {
  const key = sessions.sessionKey('claude-code', sourceSessionId);
  const lastTs = sessions
    .list(namespace)
    .find((entry) => entry.key === key)?.lastTs;
  const floor = lastTs === undefined ? undefined : Date.parse(lastTs);
  const stamped = turns.map((turn) => ({
    role: turn.role,
    ts: turn.ts ?? at,
    text: turn.text,
  }));
  // The whole window's start, kept or not: a bounded first capture must not claim
  // the conversation began at the oldest turn it happened to keep.
  let startedAt: string | undefined;
  for (const turn of stamped) {
    if (startedAt === undefined || Date.parse(turn.ts) < Date.parse(startedAt)) {
      startedAt = turn.ts;
    }
  }
  // Turns sharing the last stored timestamp stay: several transcript entries can
  // carry one timestamp, and the store skips the stored ones by hash.
  const fresh = stamped.filter(
    (turn) => floor === undefined || Date.parse(turn.ts) >= floor,
  );
  const budget = transcriptWindowBytes(tailBytes);
  const bounded: StampedTurn[] = [];
  let used = 0;
  for (let index = fresh.length - 1; index >= 0; index -= 1) {
    const turn = fresh[index];
    const bytes = Buffer.byteLength(turn.text, 'utf8');
    if (bytes > budget) {
      bounded.push(turn);
      continue;
    }
    if (bounded.length > 0 && used + bytes > budget) break;
    bounded.push(turn);
    used += bytes;
  }
  // Transcript order first, then a stable sort by time: entries can be written out
  // of order, and the file's own order is what `readSession` hands the reader.
  bounded.reverse();
  bounded.sort((left, right) => Date.parse(left.ts) - Date.parse(right.ts));
  return { ...(startedAt === undefined ? {} : { startedAt }), turns: bounded };
}

/**
 * Store the transcript's turns as a session, and never let that failure reach the
 * stop hook: an invalid sessions setting, a contended namespace lock, a full disk
 * or a read-only sessions root must not cost the user their captured facts. Every
 * step runs inside the catch for that reason, the setting check included —
 * `sessionsEnabled` throws on a `REMBERO_SESSIONS` that is neither 'on' nor
 * 'off'. The next stop hook re-reads an overlapping tail and the store skips by
 * turn hash, so a lost write is recovered rather than a lost turn.
 */
function writeSessionTurns(
  deps: AutoCaptureDeps,
  namespace: string,
  input: ReturnType<typeof parseClaudeStopHookInput>,
  turns: Array<{ role: 'user' | 'assistant'; ts?: string; text: string }>,
  now: Date,
  tailBytes: number,
): Pick<AutoCaptureResult, 'sessionTurns' | 'sessionError'> {
  const sessions = deps.sessions;
  if (sessions === undefined) return {};
  try {
    if (!sessions.sessionsEnabled()) return {};
    const selected = sessionTurnsToWrite(
      sessions,
      namespace,
      input.sessionId,
      turns,
      now.toISOString(),
      tailBytes,
    );
    if (selected.turns.length === 0) {
      return { sessionTurns: { appended: 0, skipped: 0 } };
    }
    const result = sessions.appendTurns(
      namespace,
      {
        version: 1,
        source: 'claude-code',
        sourceSessionId: input.sessionId,
        startedAt: selected.startedAt ?? selected.turns[0].ts,
        cwd: input.cwd,
      },
      selected.turns,
    );
    return {
      sessionTurns: { appended: result.appended, skipped: result.skipped },
    };
  } catch (error) {
    return {
      sessionError: error instanceof Error ? error.message : String(error),
    };
  }
}

function safeFailureReason(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (
    typeof error === 'object' &&
    error !== null &&
    (error as { code?: unknown }).code === 'integrity_violation'
  ) {
    return 'integrity_violation';
  }
  if (/sensitive/i.test(message)) return 'sensitive_text';
  if (/namespace.+local-only/i.test(message)) return 'namespace_denied';
  if (/transcript|Stop hook/i.test(message)) return 'invalid_transcript';
  if (/parse|expected|ground facts|retract|rule/i.test(message))
    return 'invalid_extraction';
  if (/fetch|HTTP|OpenRouter|LLM|model|response/i.test(message))
    return 'llm_error';
  return 'capture_error';
}

export async function autoCaptureClaudeStop(
  deps: AutoCaptureDeps,
  rawHookInput: string,
  options: AutoCaptureOptions = {},
): Promise<AutoCaptureResult> {
  const namespace = options.namespace ?? 'default';
  const dailyCap = validateAutoCaptureDailyCap(
    options.dailyCap ?? DEFAULT_AUTO_CAPTURE_DAILY_CAP,
  );
  const tailBytes = validateAutoCaptureTailBytes(
    options.tailBytes ?? DEFAULT_TRANSCRIPT_TAIL_BYTES,
  );
  const now = options.now ?? new Date();
  const captureId = deps.store.createOperationId();

  let input: ReturnType<typeof parseClaudeStopHookInput>;
  let tail: ReturnType<typeof readClaudeTranscriptTail>;
  try {
    input = parseClaudeStopHookInput(rawHookInput);
    tail = readClaudeTranscriptTail(input, {
      claudeConfigDir: options.claudeConfigDir,
      tailBytes,
      userOnly: true,
    });
  } catch (error) {
    try {
      deps.store.recordAutoCaptureSkip(namespace, safeFailureReason(error), {
        captureId,
        at: now,
      });
    } catch {
      try {
        deps.store.recordAutoCaptureEmergency(
          namespace,
          captureId,
          'journal_unavailable',
          now,
        );
      } catch {
        // The input error remains the useful failure to surface.
      }
    }
    throw error;
  }

  if (tail.userMessageCount === 0 || tail.text.trim() === '') {
    try {
      deps.store.recordAutoCaptureSkip(namespace, 'no_user_text', {
        captureId,
        at: now,
      });
    } catch (error) {
      try {
        deps.store.recordAutoCaptureEmergency(
          namespace,
          captureId,
          'journal_unavailable',
          now,
        );
      } catch {
        // The primary journal error is surfaced below.
      }
      throw error;
    }
    return {
      captureId,
      status: 'skipped',
      reason: 'no_user_text',
      added: [],
      duplicates: 0,
    };
  }

  const fingerprint = createHash('sha256')
    .update(`${input.sessionId}\0${tail.text}`, 'utf8')
    .digest('hex');
  let reservation: ReturnType<MemoryStore['reserveAutoCapture']>;
  try {
    reservation = deps.store.reserveAutoCapture(namespace, {
      captureId,
      fingerprint,
      sessionId: input.sessionId,
      tailBytes: tail.bytes,
      dailyCap,
      at: now,
    });
  } catch (error) {
    try {
      deps.store.recordAutoCaptureEmergency(
        namespace,
        captureId,
        'journal_unavailable',
        now,
      );
    } catch {
      // The original journal failure remains the most useful error to surface.
    }
    throw error;
  }
  if (!reservation.reserved) {
    return {
      captureId,
      status: 'skipped',
      reason: reservation.reason,
      added: [],
      duplicates: 0,
    };
  }

  try {
    const result = await rememberTranscriptText(
      {
        store: deps.store,
        llm: deps.llm,
        llmAllowedNamespaces: deps.llmAllowedNamespaces,
        integrityEnforcement: deps.integrityEnforcement,
        knowledgeCheckEnforcement: deps.knowledgeCheckEnforcement,
        entityIdentity: deps.entityIdentity,
      },
      tail.text,
      namespace,
      { captureId, at: now },
    );
    const status =
      result.added.length === 0 && result.duplicates === 0
        ? 'empty'
        : 'captured';
    const session = writeSessionTurns(
      deps,
      namespace,
      input,
      tail.turns,
      now,
      tailBytes,
    );
    try {
      deps.store.finishAutoCapture(
        namespace,
        captureId,
        status,
        { added: result.added.length, duplicates: result.duplicates },
        now,
      );
    } catch (error) {
      try {
        deps.store.recordAutoCaptureEmergency(
          namespace,
          captureId,
          'journal_unavailable',
          now,
        );
      } catch {
        // The primary journal error is surfaced below.
      }
      throw error;
    }
    return {
      captureId,
      status,
      added: result.added,
      duplicates: result.duplicates,
      ...session,
    };
  } catch (error) {
    try {
      deps.store.finishAutoCapture(
        namespace,
        captureId,
        'failed',
        { reason: safeFailureReason(error) },
        now,
      );
    } catch {
      try {
        deps.store.recordAutoCaptureEmergency(
          namespace,
          captureId,
          'journal_unavailable',
          now,
        );
      } catch {
        // The original capture error remains the useful failure to surface.
      }
    }
    throw error;
  }
}
