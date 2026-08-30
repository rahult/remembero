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
import type { McpToolProfile, RecallAnswerMode } from './llm/pipeline.js';

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

export function validTimeModeFromEnv(env: NodeJS.ProcessEnv = process.env): ValidTimeMode {
  const configured = env.REMBERO_VALID_TIME_MODE ?? 'delete';
  if (configured === 'delete' || configured === 'archive_until') return configured;
  throw new Error("REMBERO_VALID_TIME_MODE must be 'delete' or 'archive_until'");
}

export function recallSchemaPredicateLimitFromEnv(
  env: NodeJS.ProcessEnv = process.env
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
      `REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT must be from 1 to ${MAX_RECALL_SCHEMA_PREDICATES}`
    );
  }
  return parsed;
}

export function mcpToolProfileFromEnv(
  env: NodeJS.ProcessEnv = process.env
): McpToolProfile {
  const configured = env.REMBERO_MCP_PROFILE ?? 'full';
  if (configured === 'core' || configured === 'full') return configured;
  throw new Error("REMBERO_MCP_PROFILE must be 'core' or 'full'");
}

export function recallAnswerModeFromEnv(
  env: NodeJS.ProcessEnv = process.env
): RecallAnswerMode {
  const configured = env.REMBERO_RECALL_ANSWER_MODE ?? 'natural';
  if (
    configured === 'natural' ||
    configured === 'deterministic' ||
    configured === 'evidence'
  ) return configured;
  throw new Error(
    "REMBERO_RECALL_ANSWER_MODE must be 'natural', 'deterministic', or 'evidence'"
  );
}

export function integrityEnforcementFromEnv(
  env: NodeJS.ProcessEnv = process.env
): IntegrityEnforcementOptions | undefined {
  const mode = env.REMBERO_INTEGRITY_MODE ?? 'off';
  if (mode === 'off') return undefined;
  if (mode !== 'strict' && mode !== 'no_new_violations') {
    throw new Error(
      "REMBERO_INTEGRITY_MODE must be 'off', 'strict', or 'no_new_violations'"
    );
  }
  const configuredNamespaces = env.REMBERO_INTEGRITY_NAMESPACES;
  if (configuredNamespaces === undefined) return { mode };
  if (configuredNamespaces === '*') return { mode, namespaces: '*' };
  const namespaces = configuredNamespaces.split(',').map((value) => value.trim());
  if (namespaces.some((value) => value.length === 0)) {
    throw new Error(
      "REMBERO_INTEGRITY_NAMESPACES must be '*' or a comma-separated namespace list"
    );
  }
  return { mode, namespaces };
}

export function knowledgeCheckEnforcementFromEnv(
  env: NodeJS.ProcessEnv = process.env
): KnowledgeCheckEnforcementOptions | undefined {
  const mode = env.REMBERO_CHECK_MODE ?? 'off';
  if (mode === 'off') return undefined;
  if (mode !== 'strict' && mode !== 'no_regressions') {
    throw new Error(
      "REMBERO_CHECK_MODE must be 'off', 'strict', or 'no_regressions'"
    );
  }
  const configuredPath = env.REMBERO_CHECK_SUITE;
  if (configuredPath === undefined || configuredPath.trim() === '') {
    throw new Error('REMBERO_CHECK_SUITE is required when REMBERO_CHECK_MODE is active');
  }
  const path = resolve(configuredPath);
  const stat = lstatSync(path);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error('refusing non-regular REMBERO_CHECK_SUITE file');
  }
  if (stat.size > MAX_KNOWLEDGE_CHECK_SUITE_BYTES) {
    throw new Error(`REMBERO_CHECK_SUITE exceeds ${MAX_KNOWLEDGE_CHECK_SUITE_BYTES} bytes`);
  }
  const suite = parseKnowledgeCheckSuite(readFileSync(path, 'utf8'));
  const configuredNamespaces = env.REMBERO_CHECK_NAMESPACES;
  if (configuredNamespaces === undefined) return { mode, suite };
  if (configuredNamespaces === '*') return { mode, suite, namespaces: '*' };
  const namespaces = configuredNamespaces.split(',').map((value) => value.trim());
  if (namespaces.some((value) => value.length === 0)) {
    throw new Error(
      "REMBERO_CHECK_NAMESPACES must be '*' or a comma-separated namespace list"
    );
  }
  return { mode, suite, namespaces };
}

export function entityIdentityFromEnv(
  env: NodeJS.ProcessEnv = process.env
): EntityIdentityMode | undefined {
  const configured = env.REMBERO_ENTITY_IDENTITY ?? 'off';
  if (configured === 'off') return undefined;
  if (configured === 'canonical') return configured;
  throw new Error("REMBERO_ENTITY_IDENTITY must be 'off' or 'canonical'");
}
