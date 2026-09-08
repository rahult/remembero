import { describe, expect, it } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  entityIdentityFromEnv,
  integrityEnforcementFromEnv,
  knowledgeCheckEnforcementFromEnv,
  recallAnswerModeFromEnv,
  recallSchemaPredicateLimitFromEnv,
  validTimeModeFromEnv,
} from '../src/env.js';

describe('recallAnswerModeFromEnv', () => {
  it('defaults to natural phrasing and accepts deterministic rendering', () => {
    expect(recallAnswerModeFromEnv({})).toBe('natural');
    expect(
      recallAnswerModeFromEnv({ REMBERO_RECALL_ANSWER_MODE: 'deterministic' })
    ).toBe('deterministic');
    expect(
      recallAnswerModeFromEnv({ REMBERO_RECALL_ANSWER_MODE: 'evidence' })
    ).toBe('evidence');
  });

  it('fails closed on an unknown answer mode', () => {
    expect(() =>
      recallAnswerModeFromEnv({ REMBERO_RECALL_ANSWER_MODE: 'creative' })
    ).toThrow(/natural.*deterministic.*evidence/i);
  });
});
import {
  DEFAULT_RECALL_SCHEMA_PREDICATES,
  MAX_RECALL_SCHEMA_PREDICATES,
} from '../src/llm/schema.js';

describe('entityIdentityFromEnv', () => {
  it('is off by default and accepts the explicit canonical projection', () => {
    expect(entityIdentityFromEnv({})).toBeUndefined();
    expect(entityIdentityFromEnv({ REMBERO_ENTITY_IDENTITY: 'off' })).toBeUndefined();
    expect(entityIdentityFromEnv({ REMBERO_ENTITY_IDENTITY: 'canonical' })).toBe(
      'canonical'
    );
  });

  it('fails closed on an unknown identity mode', () => {
    expect(() =>
      entityIdentityFromEnv({ REMBERO_ENTITY_IDENTITY: 'global' })
    ).toThrow(/must be 'off' or 'canonical'/i);
  });
});

describe('validTimeModeFromEnv', () => {
  it('defaults to deletion and accepts only the two documented modes', () => {
    expect(validTimeModeFromEnv({})).toBe('delete');
    expect(validTimeModeFromEnv({ REMBERO_VALID_TIME_MODE: 'delete' })).toBe('delete');
    expect(validTimeModeFromEnv({ REMBERO_VALID_TIME_MODE: 'archive_until' })).toBe(
      'archive_until'
    );
  });

  it('fails closed on an unknown mode', () => {
    expect(() => validTimeModeFromEnv({ REMBERO_VALID_TIME_MODE: 'archive' })).toThrow(
      /must be 'delete' or 'archive_until'/i
    );
  });
});

describe('recallSchemaPredicateLimitFromEnv', () => {
  it('uses the bounded default and accepts an explicit limit', () => {
    expect(recallSchemaPredicateLimitFromEnv({})).toBe(
      DEFAULT_RECALL_SCHEMA_PREDICATES
    );
    expect(
      recallSchemaPredicateLimitFromEnv({
        REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT: '48',
      })
    ).toBe(48);
  });

  it('fails closed on malformed and out-of-range values', () => {
    for (const value of ['0', '-1', '1.5', 'many', String(MAX_RECALL_SCHEMA_PREDICATES + 1)]) {
      expect(() =>
        recallSchemaPredicateLimitFromEnv({
          REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT: value,
        })
      ).toThrow(/REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT/);
    }
  });
});

describe('integrityEnforcementFromEnv', () => {
  it('defaults to no_new_violations, allows an explicit off, and parses both modes', () => {
    // The gate is the highest-value feature the benchmarks measured; it must be on
    // for a default install. no_new_violations is migration-safe for stores that
    // already contain violations.
    expect(integrityEnforcementFromEnv({})).toEqual({ mode: 'no_new_violations' });
    expect(integrityEnforcementFromEnv({ REMBERO_INTEGRITY_MODE: 'off' })).toBeUndefined();
    expect(
      integrityEnforcementFromEnv({ REMBERO_INTEGRITY_MODE: 'strict' })
    ).toEqual({ mode: 'strict' });
    expect(
      integrityEnforcementFromEnv({
        REMBERO_INTEGRITY_MODE: 'no_new_violations',
        REMBERO_INTEGRITY_NAMESPACES: 'policy,work',
      })
    ).toEqual({
      mode: 'no_new_violations',
      namespaces: ['policy', 'work'],
    });
    expect(
      integrityEnforcementFromEnv({
        REMBERO_INTEGRITY_MODE: 'strict',
        REMBERO_INTEGRITY_NAMESPACES: '*',
      })
    ).toEqual({ mode: 'strict', namespaces: '*' });
  });

  it('rejects invalid modes and namespace lists', () => {
    expect(() =>
      integrityEnforcementFromEnv({ REMBERO_INTEGRITY_MODE: 'warn' })
    ).toThrow(/must be 'off', 'strict', or 'no_new_violations'/i);
    expect(() =>
      integrityEnforcementFromEnv({
        REMBERO_INTEGRITY_MODE: 'strict',
        REMBERO_INTEGRITY_NAMESPACES: 'policy,,work',
      })
    ).toThrow(/comma-separated namespace list/i);
  });
});

describe('knowledgeCheckEnforcementFromEnv', () => {
  it('loads and normalizes a bounded suite with both modes and namespaces', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-check-env-'));
    const file = join(root, 'checks.json');
    writeFileSync(
      file,
      JSON.stringify({
        version: 1,
        checks: [
          { name: 'required', query: 'required(a)', expect: { kind: 'nonempty' } },
        ],
      })
    );
    expect(knowledgeCheckEnforcementFromEnv({})).toBeUndefined();
    expect(
      knowledgeCheckEnforcementFromEnv({
        REMBERO_CHECK_MODE: 'strict',
        REMBERO_CHECK_SUITE: file,
        REMBERO_CHECK_NAMESPACES: 'default,shared',
      })
    ).toMatchObject({
      mode: 'strict',
      namespaces: ['default', 'shared'],
      suite: { version: 1, checks: [{ name: 'required' }] },
    });
    expect(
      knowledgeCheckEnforcementFromEnv({
        REMBERO_CHECK_MODE: 'no_regressions',
        REMBERO_CHECK_SUITE: file,
        REMBERO_CHECK_NAMESPACES: '*',
      })
    ).toMatchObject({ mode: 'no_regressions', namespaces: '*' });
  });

  it('fails closed on missing suites, invalid modes, and namespace gaps', () => {
    expect(() =>
      knowledgeCheckEnforcementFromEnv({ REMBERO_CHECK_MODE: 'strict' })
    ).toThrow(/REMBERO_CHECK_SUITE is required/i);
    expect(() =>
      knowledgeCheckEnforcementFromEnv({ REMBERO_CHECK_MODE: 'warn' })
    ).toThrow(/strict.*no_regressions/i);
    const root = mkdtempSync(join(tmpdir(), 'rembero-check-env-invalid-'));
    const file = join(root, 'checks.json');
    writeFileSync(
      file,
      JSON.stringify({
        version: 1,
        checks: [
          { name: 'valid', query: 'valid(a)', expect: { kind: 'nonempty' } },
        ],
      })
    );
    expect(() =>
      knowledgeCheckEnforcementFromEnv({
        REMBERO_CHECK_MODE: 'strict',
        REMBERO_CHECK_SUITE: file,
        REMBERO_CHECK_NAMESPACES: 'default,,shared',
      })
    ).toThrow(/comma-separated namespace list/i);
  });
});
