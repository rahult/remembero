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

export interface AutoCaptureDeps {
  store: MemoryStore;
  llm: LlmClient;
  llmAllowedNamespaces?: ReadonlySet<string>;
  integrityEnforcement?: IntegrityEnforcementOptions | false;
  knowledgeCheckEnforcement?: KnowledgeCheckEnforcementOptions | false;
  entityIdentity?: EntityIdentityMode | false;
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
  if (/parse|expected|ground facts|retract|rule/i.test(message)) return 'invalid_extraction';
  if (/fetch|HTTP|OpenRouter|LLM|model|response/i.test(message)) return 'llm_error';
  return 'capture_error';
}

export async function autoCaptureClaudeStop(
  deps: AutoCaptureDeps,
  rawHookInput: string,
  options: AutoCaptureOptions = {}
): Promise<AutoCaptureResult> {
  const namespace = options.namespace ?? 'default';
  const dailyCap = validateAutoCaptureDailyCap(
    options.dailyCap ?? DEFAULT_AUTO_CAPTURE_DAILY_CAP
  );
  const tailBytes = validateAutoCaptureTailBytes(
    options.tailBytes ?? DEFAULT_TRANSCRIPT_TAIL_BYTES
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
        deps.store.recordAutoCaptureEmergency(namespace, captureId, 'journal_unavailable', now);
      } catch {
        // The input error remains the useful failure to surface.
      }
    }
    throw error;
  }

  if (tail.userMessageCount === 0 || tail.text.trim() === '') {
    try {
      deps.store.recordAutoCaptureSkip(namespace, 'no_user_text', { captureId, at: now });
    } catch (error) {
      try {
        deps.store.recordAutoCaptureEmergency(namespace, captureId, 'journal_unavailable', now);
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
      deps.store.recordAutoCaptureEmergency(namespace, captureId, 'journal_unavailable', now);
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
      { captureId, at: now }
    );
    const status = result.added.length === 0 && result.duplicates === 0
      ? 'empty'
      : 'captured';
    try {
      deps.store.finishAutoCapture(
        namespace,
        captureId,
        status,
        { added: result.added.length, duplicates: result.duplicates },
        now
      );
    } catch (error) {
      try {
        deps.store.recordAutoCaptureEmergency(namespace, captureId, 'journal_unavailable', now);
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
    };
  } catch (error) {
    try {
      deps.store.finishAutoCapture(
        namespace,
        captureId,
        'failed',
        { reason: safeFailureReason(error) },
        now
      );
    } catch {
      try {
        deps.store.recordAutoCaptureEmergency(namespace, captureId, 'journal_unavailable', now);
      } catch {
        // The original capture error remains the useful failure to surface.
      }
    }
    throw error;
  }
}
