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

/**
 * Store the transcript's turns as a session, and never let that failure reach the
 * stop hook: a contended namespace lock, a full disk or a read-only sessions root
 * must not cost the user their captured facts. The next stop hook re-reads an
 * overlapping tail and the store skips by turn hash, so a lost write is recovered
 * rather than a lost turn.
 */
function writeSessionTurns(
  deps: AutoCaptureDeps,
  namespace: string,
  input: ReturnType<typeof parseClaudeStopHookInput>,
  turns: Array<{ role: 'user' | 'assistant'; ts?: string; text: string }>,
  now: Date,
): Pick<AutoCaptureResult, 'sessionTurns' | 'sessionError'> {
  const sessions = deps.sessions;
  if (sessions === undefined || !sessions.sessionsEnabled()) return {};
  const at = now.toISOString();
  const stamped = turns.map((turn) => ({
    role: turn.role,
    ts: turn.ts ?? at,
    text: turn.text,
  }));
  if (stamped.length === 0) return {};
  try {
    const result = sessions.appendTurns(
      namespace,
      {
        version: 1,
        source: 'claude-code',
        sourceSessionId: input.sessionId,
        startedAt: stamped[0].ts,
        cwd: input.cwd,
      },
      stamped,
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
