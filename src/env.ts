import { existsSync, lstatSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from 'dotenv';
import {
  DEFAULT_RECALL_SCHEMA_PREDICATES,
  MAX_RECALL_SCHEMA_PREDICATES,
} from './llm/schema.js';
import type { ValidTimeMode } from './store/store.js';
import type { IntegrityEnforcementOptions } from './knowledge/enforcement.js';
import type { KnowledgeCheckEnforcementOptions } from './knowledge/check-enforcement.js';
import {
  MAX_KNOWLEDGE_CHECK_SUITE_BYTES,
  parseKnowledgeCheckSuite,
} from './knowledge/checks.js';
import type { EntityIdentityMode } from './knowledge/identity.js';
import type {
  McpToolProfile,
  RecallAnswerMode,
  RecallReader,
} from './llm/pipeline.js';
import {
  DEFAULT_LLM_TEMPERATURE,
  DEFAULT_LLM_TIMEOUT_MS,
  checkedTemperature,
  checkedTimeoutMs,
} from './llm/client.js';

/**
 * Load .env from the current directory and from the package root (so the CLI
 * works no matter where it is launched from). Existing env vars win.
 */
export function loadEnv(): void {
  const candidates = [
    join(process.cwd(), '.env'),
    join(dirname(fileURLToPath(import.meta.url)), '..', '.env'),
  ];
  for (const path of candidates) {
    if (existsSync(path)) config({ path, quiet: true });
  }
}

export function validTimeModeFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): ValidTimeMode {
  const configured = env.REMBERO_VALID_TIME_MODE ?? 'delete';
  if (configured === 'delete' || configured === 'archive_until')
    return configured;
  throw new Error(
    "REMBERO_VALID_TIME_MODE must be 'delete' or 'archive_until'",
  );
}

export function recallSchemaPredicateLimitFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT;
  if (configured === undefined) return DEFAULT_RECALL_SCHEMA_PREDICATES;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT must be an integer');
  }
  const parsed = Number(configured);
  if (
    !Number.isSafeInteger(parsed) ||
    parsed < 1 ||
    parsed > MAX_RECALL_SCHEMA_PREDICATES
  ) {
    throw new Error(
      `REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT must be from 1 to ${MAX_RECALL_SCHEMA_PREDICATES}`,
    );
  }
  return parsed;
}

export function mcpToolProfileFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): McpToolProfile {
  const configured = env.REMBERO_MCP_PROFILE ?? 'full';
  if (configured === 'core' || configured === 'full') return configured;
  throw new Error("REMBERO_MCP_PROFILE must be 'core' or 'full'");
}

/** REMBERO_SELF: the constant that names the speaker in remembered text (default 'user'). */
export function selfAtomFromEnv(env: NodeJS.ProcessEnv = process.env): string {
  const configured = env.REMBERO_SELF?.trim();
  if (configured === undefined || configured === '') return 'user';
  if (!/^[a-z][a-z0-9_]*$/.test(configured)) {
    throw new Error(
      'REMBERO_SELF must be a lowercase snake_case constant, e.g. rahul',
    );
  }
  return configured;
}

/** REMBERO_EXTRACTION_VOCABULARY: 'open' (default) or 'closed' (only schema predicates may be added). */
export function extractionVocabularyFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): 'open' | 'closed' {
  const configured = env.REMBERO_EXTRACTION_VOCABULARY ?? 'open';
  if (configured === 'open' || configured === 'closed') return configured;
  throw new Error("REMBERO_EXTRACTION_VOCABULARY must be 'open' or 'closed'");
}

/**
 * REMBERO_RECALL_ANSWER_MODE: 'evidence' (default) renders the query's rows and their
 * proof locally with no model call; 'deterministic' renders bare bindings; 'natural'
 * asks the configured LLM to phrase the answer, the only mode that sends recalled
 * facts to a model. Evidence became the default in 0.57: the phrasing leg costs a
 * frontier call per recall and, on the agent-boundary benchmark, only ever lost answers
 * the query had already got right (docs/research/SELF-HOSTED-ROADMAP.md).
 */
export function recallAnswerModeFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): RecallAnswerMode {
  const configured = env.REMBERO_RECALL_ANSWER_MODE ?? 'evidence';
  if (
    configured === 'natural' ||
    configured === 'deterministic' ||
    configured === 'evidence' ||
    configured === 'sessions'
  )
    return configured;
  throw new Error(
    "REMBERO_RECALL_ANSWER_MODE must be 'natural', 'deterministic', 'evidence', or 'sessions'",
  );
}

/** A reader left at its defaults reads one prompt of notes and an answer line. */
export const DEFAULT_READER_MAX_TOKENS = 4_096;

/**
 * The reader the `sessions` answer mode sends its prompt to: `REMBERO_READER_BASE_URL`
 * plus `_MODEL`, `_API_KEY`, `_MAX_TOKENS`, `_TEMPERATURE`, `_TIMEOUT_MS`. Nothing at all
 * when no base URL is set, which is how recall falls back to the product's configured LLM.
 *
 * A base URL without a model is a configuration mistake rather than a reason to answer
 * from a model the user did not choose, so it throws. The API key defaults to `local`
 * because a llama.cpp or Ollama endpoint accepts any bearer token, and a local reader is
 * the point of the setting.
 */
export function readerFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): RecallReader | undefined {
  const baseUrl = env.REMBERO_READER_BASE_URL?.trim();
  if (baseUrl === undefined || baseUrl === '') return undefined;
  const model = env.REMBERO_READER_MODEL?.trim();
  if (model === undefined || model === '') {
    throw new Error(
      'REMBERO_READER_MODEL is required when REMBERO_READER_BASE_URL is set',
    );
  }
  const configuredMaxTokens = env.REMBERO_READER_MAX_TOKENS;
  let maxTokens = DEFAULT_READER_MAX_TOKENS;
  if (configuredMaxTokens !== undefined) {
    if (!/^\d+$/.test(configuredMaxTokens)) {
      throw new Error('REMBERO_READER_MAX_TOKENS must be an integer');
    }
    maxTokens = Number(configuredMaxTokens);
    if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || maxTokens > 16_384) {
      throw new Error(
        'REMBERO_READER_MAX_TOKENS must be an integer from 1 to 16384',
      );
    }
  }
  const temperature = env.REMBERO_READER_TEMPERATURE;
  const timeoutMs = env.REMBERO_READER_TIMEOUT_MS;
  return {
    baseUrl: baseUrl.replace(/\/$/, ''),
    model,
    apiKey: env.REMBERO_READER_API_KEY ?? 'local',
    maxTokens,
    temperature:
      temperature === undefined
        ? DEFAULT_LLM_TEMPERATURE
        : checkedTemperature(temperature, 'REMBERO_READER_TEMPERATURE'),
    timeoutMs:
      timeoutMs === undefined
        ? DEFAULT_LLM_TIMEOUT_MS
        : checkedTimeoutMs(timeoutMs, 'REMBERO_READER_TIMEOUT_MS'),
  };
}

/**
 * How many bytes of retrieved history the product's `sessions` answer mode spends on
 * one question.
 *
 * The published number describes a benchmark arm that ran `--context-bytes 24576`, and
 * reader v7 was trained on 24 KB prompts, so the product must read the same budget or
 * its answers are not the ones that were measured. The shared module's
 * `DEFAULT_READING_CONTEXT_BYTES` (56 KB) stays as it is: it is the harness's own
 * fallback for arms that do not pass the flag, and changing it would move every
 * archived comparison.
 */
export const DEFAULT_PRODUCT_READING_CONTEXT_BYTES = 24 * 1024;
/**
 * The range `validateReadingOptions` (knowledge/session-retrieval.ts) accepts, repeated
 * here rather than imported so a setting lookup does not pull the whole retrieval module
 * into `env.ts`. A test pins the two to agree.
 */
export const MIN_READING_CONTEXT_BYTES = 4_096;
export const MAX_CONFIGURED_READING_CONTEXT_BYTES = 160 * 1024;

/** REMBERO_READING_CONTEXT_BYTES: the reading budget, 24576 by default. */
export function readingContextBytesFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_READING_CONTEXT_BYTES;
  if (configured === undefined) return DEFAULT_PRODUCT_READING_CONTEXT_BYTES;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_READING_CONTEXT_BYTES must be an integer');
  }
  const parsed = Number(configured);
  if (
    !Number.isSafeInteger(parsed) ||
    parsed < MIN_READING_CONTEXT_BYTES ||
    parsed > MAX_CONFIGURED_READING_CONTEXT_BYTES
  ) {
    throw new Error(
      `REMBERO_READING_CONTEXT_BYTES must be an integer from ${MIN_READING_CONTEXT_BYTES} to ${MAX_CONFIGURED_READING_CONTEXT_BYTES}`,
    );
  }
  return parsed;
}

/**
 * REMBERO_READER_ALLOW_REMOTE: `1` lets the `sessions` answer mode send stored
 * conversations to a reader that is not on localhost. Anything else, including
 * unset, keeps them on the machine.
 *
 * The store masks secrets on the way in, but that masking is best-effort pattern
 * matching and not a guarantee: it recognises credential words, PEM private keys,
 * AWS key ids, JWTs, bearer and `sk-` tokens, credentials inside a URL and
 * Luhn-valid card runs, and a secret with none of those shapes is stored as
 * written. A reader on 127.0.0.1 reads that text without it leaving the machine,
 * so a non-local reader — including the product's own configured LLM, which recall
 * cannot tell is local — is refused until the user says otherwise here.
 *
 * A typo is read as "not set": the safe direction is to send nothing, and recall
 * says which setting to correct.
 */
export function readerAllowsRemoteFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return env.REMBERO_READER_ALLOW_REMOTE?.trim() === '1';
}

export function integrityEnforcementFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): IntegrityEnforcementOptions | undefined {
  // The gate is on by default. no_new_violations is migration-safe: stores that
  // already contain violations keep working, but no write may add one.
  // 'off' is an explicit opt-out; `remembero init` registers the server strict.
  const mode = env.REMBERO_INTEGRITY_MODE ?? 'no_new_violations';
  if (mode === 'off') return undefined;
  if (mode !== 'strict' && mode !== 'no_new_violations') {
    throw new Error(
      "REMBERO_INTEGRITY_MODE must be 'off', 'strict', or 'no_new_violations'",
    );
  }
  const configuredNamespaces = env.REMBERO_INTEGRITY_NAMESPACES;
  if (configuredNamespaces === undefined) return { mode };
  if (configuredNamespaces === '*') return { mode, namespaces: '*' };
  const namespaces = configuredNamespaces
    .split(',')
    .map((value) => value.trim());
  if (namespaces.some((value) => value.length === 0)) {
    throw new Error(
      "REMBERO_INTEGRITY_NAMESPACES must be '*' or a comma-separated namespace list",
    );
  }
  return { mode, namespaces };
}

export function knowledgeCheckEnforcementFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): KnowledgeCheckEnforcementOptions | undefined {
  const mode = env.REMBERO_CHECK_MODE ?? 'off';
  if (mode === 'off') return undefined;
  if (mode !== 'strict' && mode !== 'no_regressions') {
    throw new Error(
      "REMBERO_CHECK_MODE must be 'off', 'strict', or 'no_regressions'",
    );
  }
  const configuredPath = env.REMBERO_CHECK_SUITE;
  if (configuredPath === undefined || configuredPath.trim() === '') {
    throw new Error(
      'REMBERO_CHECK_SUITE is required when REMBERO_CHECK_MODE is active',
    );
  }
  const path = resolve(configuredPath);
  const stat = lstatSync(path);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error('refusing non-regular REMBERO_CHECK_SUITE file');
  }
  if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
    throw new Error(
      `REMBERO_CHECK_SUITE exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`,
    );
  }
  const suite = parseKnowledgeCheckSuite(readFileSync(path, 'utf8'));
  const configuredNamespaces = env.REMBERO_CHECK_NAMESPACES;
  if (configuredNamespaces === undefined) return { mode, suite };
  if (configuredNamespaces === '*') return { mode, suite, namespaces: '*' };
  const namespaces = configuredNamespaces
    .split(',')
    .map((value) => value.trim());
  if (namespaces.some((value) => value.length === 0)) {
    throw new Error(
      "REMBERO_CHECK_NAMESPACES must be '*' or a comma-separated namespace list",
    );
  }
  return { mode, suite, namespaces };
}

/** The session store holds 200 MB of conversation text per namespace before it drops the oldest. */
export const DEFAULT_SESSION_CAP_BYTES = 200 * 1024 * 1024;
/** Below a megabyte the cap would drop a session as fast as capture writes one. */
export const MIN_SESSION_CAP_BYTES = 1024 * 1024;

/**
 * REMBERO_SESSIONS: 'off' (default) or 'on'. Off means no conversation text is
 * stored at all; capture and import check this before they write.
 */
export function sessionsEnabledFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const configured = env.REMBERO_SESSIONS ?? 'off';
  if (configured === 'on') return true;
  if (configured === 'off') return false;
  throw new Error("REMBERO_SESSIONS must be 'on' or 'off'");
}

/** REMBERO_SESSION_CAP_BYTES: per-namespace byte cap for stored sessions. */
export function sessionCapBytesFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_SESSION_CAP_BYTES;
  if (configured === undefined) return DEFAULT_SESSION_CAP_BYTES;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_SESSION_CAP_BYTES must be an integer');
  }
  const parsed = Number(configured);
  if (!Number.isSafeInteger(parsed) || parsed < MIN_SESSION_CAP_BYTES) {
    throw new Error(
      `REMBERO_SESSION_CAP_BYTES must be at least ${MIN_SESSION_CAP_BYTES}`,
    );
  }
  return parsed;
}

/**
 * How a long conversation is cut into the units retrieval ranks and the reader reads.
 *
 * A LongMemEval session is about 30 turns and a few kilobytes; one real Claude Code
 * conversation measured 468 turns and 346 KB over five days. Ranking whole sessions
 * then has nothing to choose between, and the reading budget keeps whichever 7% of
 * the file comes first. 40 turns is a little over a LongMemEval session, so the unit
 * the reader sees is the size the reader was trained on, and six hours splits a
 * conversation resumed the next morning from the one it continues.
 */
export const DEFAULT_SESSION_WINDOW_TURNS = 40;
export const DEFAULT_SESSION_WINDOW_GAP_MS = 6 * 60 * 60 * 1000;
/**
 * And at most 8 KB of turn text, because one turn is not one size. The reading budget is
 * 24,576 bytes over four sessions, about 6 KB each, and the reader is never shown more
 * than the first 16,384 characters of a source: a window past that size has a tail
 * nothing can reach. Measured, a 40-turn window of the real transcript came to 63 KB
 * around a single 16.8 KB paste, and the sentence that answered the question sat behind
 * it. A turn larger than this keeps a window to itself rather than being cut.
 */
export const DEFAULT_SESSION_WINDOW_BYTES = 8 * 1024;
/** A window of one turn is no window at all: nothing would ever be read with its neighbours. */
export const MIN_SESSION_WINDOW_TURNS = 2;
/** Below a minute a window would break on the pause inside a single exchange. */
export const MIN_SESSION_WINDOW_GAP_MS = 60 * 1000;
/** Below a kilobyte nearly every turn would be a window of its own. */
export const MIN_SESSION_WINDOW_BYTES = 1024;

/** REMBERO_SESSION_WINDOW_TURNS: how many turns one window holds at most. */
export function sessionWindowTurnsFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_SESSION_WINDOW_TURNS;
  if (configured === undefined) return DEFAULT_SESSION_WINDOW_TURNS;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_SESSION_WINDOW_TURNS must be an integer');
  }
  const parsed = Number(configured);
  if (!Number.isSafeInteger(parsed) || parsed < MIN_SESSION_WINDOW_TURNS) {
    throw new Error(
      `REMBERO_SESSION_WINDOW_TURNS must be at least ${MIN_SESSION_WINDOW_TURNS}`,
    );
  }
  return parsed;
}

/** REMBERO_SESSION_WINDOW_GAP_MS: the silence that starts a new window. */
export function sessionWindowGapMsFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_SESSION_WINDOW_GAP_MS;
  if (configured === undefined) return DEFAULT_SESSION_WINDOW_GAP_MS;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_SESSION_WINDOW_GAP_MS must be an integer');
  }
  const parsed = Number(configured);
  if (!Number.isSafeInteger(parsed) || parsed < MIN_SESSION_WINDOW_GAP_MS) {
    throw new Error(
      `REMBERO_SESSION_WINDOW_GAP_MS must be at least ${MIN_SESSION_WINDOW_GAP_MS}`,
    );
  }
  return parsed;
}

/** REMBERO_SESSION_WINDOW_BYTES: how much turn text one window holds at most. */
export function sessionWindowBytesFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const configured = env.REMBERO_SESSION_WINDOW_BYTES;
  if (configured === undefined) return DEFAULT_SESSION_WINDOW_BYTES;
  if (!/^\d+$/.test(configured)) {
    throw new Error('REMBERO_SESSION_WINDOW_BYTES must be an integer');
  }
  const parsed = Number(configured);
  if (!Number.isSafeInteger(parsed) || parsed < MIN_SESSION_WINDOW_BYTES) {
    throw new Error(
      `REMBERO_SESSION_WINDOW_BYTES must be at least ${MIN_SESSION_WINDOW_BYTES}`,
    );
  }
  return parsed;
}

export function entityIdentityFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): EntityIdentityMode | undefined {
  const configured = env.REMBERO_ENTITY_IDENTITY ?? 'off';
  if (configured === 'off') return undefined;
  if (configured === 'canonical') return configured;
  throw new Error("REMBERO_ENTITY_IDENTITY must be 'off' or 'canonical'");
}
