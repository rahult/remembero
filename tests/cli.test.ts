import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  MAX_INPUT_BYTES,
  assertBoundedOutput,
  stringifyBoundedResult,
} from '../src/safety.js';
import { MemoryStore, knowledgeProgramDigest } from '../src/store/store.js';
import { computeMemoryProposalDigest } from '../src/knowledge/memory-proposal.js';
import { serializeClause } from '../src/engine/index.js';

describe('CLI ingress limits', () => {
  it('exposes the remembero command name through successful help', () => {
    const root = mkdtempSync(join(tmpdir(), 'remembero-cli-help-'));
    const home = join(root, 'home');
    const result = spawnSync(process.execPath, [resolve('dist/cli.js'), '--help'], {
      encoding: 'utf8',
      env: { ...process.env, REMBERO_HOME: home },
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toMatch(/^remembero — logic-based memory/);
    expect(result.stderr).toBe('');
    expect(existsSync(home)).toBe(false);
  });

  it('fails closed before returning an oversized JSON result', () => {
    expect(() => stringifyBoundedResult({ value: 'oversized' }, 'test result', 8)).toThrow(
      /test result exceeds 8 bytes/i
    );
  });

  it('rejects non-finite numbers instead of silently serializing them as null', () => {
    expect(() => stringifyBoundedResult({ value: Number.NaN }, 'test result')).toThrow(
      /non-finite/i
    );
    expect(() =>
      stringifyBoundedResult({ value: Number.POSITIVE_INFINITY }, 'test result')
    ).toThrow(/non-finite/i);
  });

  it('fails closed before printing an oversized plain-text recall answer', () => {
    expect(() => assertBoundedOutput('oversized', 'CLI recall answer', 8)).toThrow(
      /CLI recall answer exceeds 8 bytes/i
    );
  });
  it('rejects an oversized import before reading or mutating the store', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-limit-'));
    const file = join(root, 'oversized.dl');
    const home = join(root, 'home');
    writeFileSync(file, 'x'.repeat(MAX_INPUT_BYTES + 1));

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'import', 'default', file],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/import file exceeds 65536 bytes/i);
    expect(existsSync(join(home, 'memory', 'default.dl'))).toBe(false);
  });

  it('validates the recall schema limit before any external request', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-schema-limit-'));
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'recall',
        'What is remembered?',
        '--schema-predicate-limit',
        '0',
      ],
      {
        encoding: 'utf8',
        env: {
          ...process.env,
          REMBERO_HOME: join(root, 'home'),
          LLM_API_KEY: 'test-only-key',
          REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT: '32',
        },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/schema predicate limit must be from 1 to 256/i);
  });

  it('prints the explicit recall status when memory is empty', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-recall-status-'));
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'recall',
        'What is remembered?',
        '--related',
        '--related-limit',
        '2',
      ],
      {
        encoding: 'utf8',
        env: {
          ...process.env,
          REMBERO_HOME: join(root, 'home'),
          LLM_API_KEY: 'test-only-key',
          REMBERO_RECALL_SCHEMA_PREDICATE_LIMIT: '32',
        },
      }
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('status: unanswerable');
    expect(result.stdout).toContain(
      'Related knowledge (discovery only; not an answer or proof):'
    );
    expect(result.stdout).toContain('No local lexical matches.');
  });

  it('routes proposal-first memory through local secret rejection without writing', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-memory-proposal-'));
    const home = join(root, 'home');
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'propose-memory',
        'My token is ghp_supersecretvalue.',
      ],
      {
        encoding: 'utf8',
        env: {
          ...process.env,
          REMBERO_HOME: home,
          LLM_API_KEY: 'unused-test-key',
        },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/refusing to send sensitive memory text/i);
    expect(existsSync(join(home, 'memory', 'default.dl'))).toBe(false);
  });

  it('applies one reviewed accepted-memory proposal file idempotently', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-apply-memory-'));
    const home = join(root, 'home');
    const proposalFile = join(root, 'memory-proposal.json');
    const payload = {
      version: 1 as const,
      baselineDigest: knowledgeProgramDigest(
        ['default'],
        new Map([['default', []]])
      ),
      namespace: 'default',
      namespaces: ['default'],
      sourceText: 'Rahul has a pet named Luna.',
      validTimeMode: 'delete' as const,
      addClauses: ['pet(rahul, luna).'],
      removeClauses: [],
    };
    writeFileSync(
      proposalFile,
      JSON.stringify({
        ...payload,
        proposalDigest: computeMemoryProposalDigest(payload),
      })
    );
    const apply = () =>
      spawnSync(
        process.execPath,
        [
          resolve('dist/cli.js'),
          'apply-memory',
          proposalFile,
          '--op-id',
          'cli-reviewed-memory',
        ],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );

    const first = apply();
    expect(first.status).toBe(0);
    expect(JSON.parse(first.stdout)).toMatchObject({
      opId: 'cli-reviewed-memory',
      sequence: 1,
      added: [expect.any(Object)],
      audit: { topology: { factCount: 1 } },
    });
    expect(JSON.parse(apply().stdout)).toEqual(JSON.parse(first.stdout));
    const queried = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'query', 'pet(rahul, Name)'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(JSON.parse(queried.stdout)).toEqual([{ Name: 'luna' }]);
  });

  it('rejects an invalid proof limit before evaluating an explanation', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-proof-limit-'));
    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'explain', 'answer(a)', '--proof-limit', '17'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: join(root, 'home') },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/proof limit must be from 1 to 16/i);
  });

  it('prints alternative proof witnesses through the explain command', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-proofs-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'left(a). right(a). answer(X) :- left(X). answer(X) :- right(X).'
    );

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'explain', 'answer(a)', '--proof-limit', '2'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(0);
    const payload = JSON.parse(result.stdout);
    expect(payload.rows[0]).toMatchObject({
      proofs: [expect.objectContaining({ rule: 1 })],
      alternativeProofs: [[expect.objectContaining({ rule: 2 })]],
    });
  });

  it('replays explicit write operation ids and reports deterministic conflicts', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-op-id-'));
    const home = join(root, 'home');
    const runAssert = (clause: string) =>
      spawnSync(
        process.execPath,
        [resolve('dist/cli.js'), 'assert', clause, '--op-id', 'cli-assert-retry'],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );

    const first = runAssert('retry_fact(alpha).');
    const replay = runAssert('retry_fact(alpha).');
    const conflict = runAssert('retry_fact(beta).');

    expect(first.status).toBe(0);
    expect(replay.status).toBe(0);
    expect(JSON.parse(replay.stdout)).toEqual(JSON.parse(first.stdout));
    expect(conflict.status).toBe(4);
    expect(JSON.parse(conflict.stderr)).toEqual({
      error: 'operation_conflict',
      message: "assert operation 'cli-assert-retry' was already used for another mutation",
      operation: 'assert',
      namespace: 'default',
      opId: 'cli-assert-retry',
    });

    const firstForget = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'forget',
        'retry_fact(_)',
        '--op-id',
        'cli-forget-retry',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    const replayForget = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'forget',
        'retry_fact( _ )',
        '--op-id',
        'cli-forget-retry',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(firstForget.status).toBe(0);
    expect(replayForget.status).toBe(0);
    expect(firstForget.stdout).toBe('removed 1 clause(s)\n');
    expect(replayForget.stdout).toBe(firstForget.stdout);
  });

  it('keeps tentative CLI facts hidden until explicitly reviewed and accepted', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-trust-'));
    const home = join(root, 'home');
    const run = (args: string[]) =>
      spawnSync(process.execPath, [resolve('dist/cli.js'), ...args], {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      });

    const asserted = run([
      'assert',
      'status(mira, active).',
      '--namespace',
      'personal',
      '--trust',
      'tentative',
      '--op-id',
      'cli-tentative-status',
    ]);
    expect(asserted.status).toBe(0);
    expect(JSON.parse(asserted.stdout)).toMatchObject({
      added: ['status(mira, active).'],
    });

    const hidden = run([
      'query',
      'status(mira, State)',
      '--namespaces',
      'personal',
    ]);
    expect(JSON.parse(hidden.stdout)).toEqual([]);
    const included = run([
      'explain',
      'status(mira, State)',
      '--namespaces',
      'personal',
      '--trust',
      'include_tentative',
    ]);
    expect(JSON.parse(included.stdout)).toMatchObject({
      trustMode: 'include_tentative',
      rows: [{ bindings: { State: 'active' }, proofs: [{ trust: 'tentative' }] }],
    });
    const claims = run(['claims', '--namespaces', 'personal']);
    expect(JSON.parse(claims.stdout)).toMatchObject({
      count: 1,
      claims: [{ clause: 'status(mira, active).', namespace: 'personal' }],
    });

    const accepted = run([
      'accept',
      'status(mira, active).',
      '--namespace',
      'personal',
      '--op-id',
      'cli-accept-status',
    ]);
    expect(accepted.status).toBe(0);
    expect(JSON.parse(accepted.stdout)).toMatchObject({
      action: 'accept',
      resolved: 1,
      added: ['status(mira, active).'],
    });
    const visible = run([
      'query',
      'status(mira, State)',
      '--namespaces',
      'personal',
    ]);
    expect(JSON.parse(visible.stdout)).toEqual([{ State: 'active' }]);

    expect(
      run([
        'assert',
        'prefers(mira, tea).',
        '--namespace',
        'personal',
        '--trust',
        'tentative',
      ]).status
    ).toBe(0);
    const rejected = run([
      'reject',
      'prefers(mira, tea).',
      '--namespace',
      'personal',
      '--op-id',
      'cli-reject-preference',
    ]);
    expect(JSON.parse(rejected.stdout)).toMatchObject({
      action: 'reject',
      resolved: 1,
      added: [],
    });
    expect(JSON.parse(run(['claims', '--namespaces', 'personal']).stdout)).toMatchObject({
      count: 0,
    });
  });

  it('rotates and lists journal checkpoints without changing recorded CLI reads', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-checkpoint-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'item(a).', { opId: 'first' });
    store.assert('default', 'item(b).', { opId: 'second' });
    const run = (args: string[]) =>
      spawnSync(process.execPath, [resolve('dist/cli.js'), ...args], {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      });

    const preview = run([
      'checkpoint',
      '--dry-run',
      '--op-id',
      'cli-checkpoint',
      '--at',
      '2026-08-17T02:00:00.000Z',
    ]);
    expect(JSON.parse(preview.stdout)).toMatchObject({
      rotated: true,
      sequence: 2,
    });
    expect(existsSync(join(home, 'memory', 'journal.log'))).toBe(true);

    const compacted = run([
      'checkpoint',
      '--op-id',
      'cli-checkpoint',
      '--at',
      '2026-08-17T02:00:00.000Z',
    ]);
    expect(compacted.status).toBe(0);
    expect(JSON.parse(compacted.stdout)).toMatchObject({
      rotated: true,
      sequence: 2,
      segmentCount: 1,
    });
    expect(JSON.parse(run(['checkpoints']).stdout)).toMatchObject({
      count: 1,
      checkpoints: [{ sequence: 2 }],
    });
    expect(
      JSON.parse(
        run(['query', 'item(Value)', '--as-of-sequence', '1']).stdout
      )
    ).toMatchObject({
      bindings: [{ Value: 'a' }],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('previews counterfactual result and integrity changes without writing', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-what-if-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'status(mira, active). :- status(Person, active), status(Person, paused).',
      { opId: 'baseline' }
    );
    const journalBefore = readFileSync(join(home, 'memory', 'journal.log'), 'utf8');

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'what-if',
        'status(mira, State)',
        '--assume',
        'status(mira, paused).',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      changed: true,
      candidate: { rows: [{ bindings: { State: 'active' } }, { bindings: { State: 'paused' } }] },
      resultDelta: { added: [{ bindings: { State: 'paused' } }] },
      integrityDelta: {
        candidate: { status: 'violations', violationCount: 1 },
        introduced: [{ bindings: { Person: 'mira' } }],
      },
    });
    expect(readFileSync(join(home, 'memory', 'journal.log'), 'utf8')).toBe(
      journalBefore
    );
    expect(store.load('default').map(serializeClause)).toEqual([
      'status(mira, active).',
      ':- status(Person, active), status(Person, paused).',
    ]);
  });

  it('previews rule, topology, audit, and coverage changes without writing', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-rule-what-if-'));
    const home = join(root, 'home');
    const suite = join(root, 'checks.json');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'base(a).', { opId: 'rule-baseline' });
    writeFileSync(
      suite,
      JSON.stringify({
        version: 1,
        coverage: { minimumPercent: 100 },
        checks: [
          {
            name: 'derived fact',
            query: 'derived(a)',
            expect: { kind: 'nonempty' },
          },
        ],
      })
    );
    const journalBefore = readFileSync(join(home, 'memory', 'journal.log'), 'utf8');

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'what-if',
        'derived(X)',
        '--assume-rule',
        'derived(X) :- base(X).',
        '--check-suite',
        suite,
        '--as-of-sequence',
        '1',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      changed: true,
      recordedSnapshot: { sequence: 1, journalEntries: 1 },
      application: { assumedRules: ['derived(X) :- base(X).'] },
      candidate: { rows: [{ bindings: { X: 'a' } }] },
      ruleAuditDelta: {
        baseline: { topology: { ruleCount: 0 } },
        candidate: { topology: { ruleCount: 1 } },
      },
      checkDelta: {
        baseline: { status: 'failed' },
        candidate: { status: 'passed', coverage: { percent: 100 } },
        fixed: ['derived fact'],
      },
    });
    expect(readFileSync(join(home, 'memory', 'journal.log'), 'utf8')).toBe(
      journalBefore
    );
    expect(store.clausesFor(['default'])).toHaveLength(1);
  });

  it('applies one reviewed rule proposal file atomically and idempotently', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-apply-rule-'));
    const home = join(root, 'home');
    const proposalFile = join(root, 'proposal.json');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'base(a).', { opId: 'apply-baseline' });

    const preview = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'what-if',
        'derived(X)',
        '--assume-rule',
        'derived(X) :- base(X).',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(preview.status).toBe(0);
    writeFileSync(proposalFile, preview.stdout);

    const apply = () =>
      spawnSync(
        process.execPath,
        [
          resolve('dist/cli.js'),
          'apply-rule-change',
          proposalFile,
          '--op-id',
          'cli-reviewed-rule',
        ],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );
    const first = apply();
    expect(first.status).toBe(0);
    expect(JSON.parse(first.stdout)).toMatchObject({
      opId: 'cli-reviewed-rule',
      sequence: 2,
      added: [expect.any(Object)],
      audit: { topology: { ruleCount: 1 } },
    });
    const replay = apply();
    expect(replay.status).toBe(0);
    expect(JSON.parse(replay.stdout)).toEqual(JSON.parse(first.stdout));

    const queried = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'query', 'derived(X)'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(JSON.parse(queried.stdout)).toEqual([{ X: 'a' }]);
  });

  it('explains rule blockers and sourced nearby facts without an LLM', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-why-not-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'works_at(mira, initech). colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
      { opId: 'employment-source' }
    );

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'why-not',
        'colleague(mira, rahul)',
        '--failure-limit',
        '16',
        '--diagnostic-depth',
        '6',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'blocked',
      failures: [
        {
          reason: 'rules_blocked',
          rules: [
            {
              failures: [
                {
                  reason: 'missing_fact',
                  goal: 'works_at(rahul, initech)',
                  nearby: [
                    {
                      fact: 'works_at(mira, initech).',
                      explanation: {
                        rows: [{ proofs: [{ sources: [{ opId: 'employment-source' }] }] }],
                      },
                    },
                  ],
                },
              ],
            },
          ],
        },
      ],
      graph: {
        nodes: expect.arrayContaining([
          expect.objectContaining({ kind: 'failure', reason: 'missing_fact' }),
        ]),
      },
    });
  });

  it('exports a focused rule and policy topology without an LLM', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-topology-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `employee(alice).
       eligible(X) :- employee(X), \\+ suspended(X).
       :- eligible(X), blocked(X).`,
      { opId: 'topology-source' }
    );

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'topology',
        'eligible',
        '--direction',
        'both',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      predicates: [
        { key: 'blocked/1', openInput: true },
        { key: 'eligible/1', stratum: 1 },
        { key: 'employee/1' },
        { key: 'suspended/1', openInput: true },
      ],
      rules: [{ sources: [{ opId: 'topology-source' }] }],
      constraints: [{ sources: [{ opId: 'topology-source' }] }],
      selection: { focus: 'eligible/1', direction: 'both' },
      openNegatedInputs: ['suspended/1'],
    });
  });

  it('diffs exact recorded states with optional query impact', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-diff-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'item(a).', { opId: 'before' });
    store.assert('default', 'item(b).', { opId: 'after' });

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'diff',
        '1',
        '2',
        '--query',
        'item(Value)',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      from: { sequence: 1, journalEntries: 2 },
      to: { sequence: 2, journalEntries: 2 },
      clauses: { added: [{ clause: 'item(b).' }] },
      queryImpact: {
        added: [{ bindings: { Value: 'b' } }],
        unchangedCount: 1,
      },
    });
  });

  it('prints minimal verified repair proposals without mutating knowledge', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-repair-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'employee(bob). ready(X) :- employee(X), badge(X), trained(X).',
      { opId: 'baseline' }
    );

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'repair',
        'ready(bob)',
        '--plan-limit',
        '4',
        '--repair-steps',
        '3',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'repairable',
      plans: [
        {
          assume: ['badge(bob).', 'trained(bob).'],
          candidate: { rows: [{ bindings: {} }] },
        },
      ],
    });
    expect(store.load('default').map(serializeClause)).toEqual([
      'employee(bob).',
      'ready(X) :- employee(X), badge(X), trained(X).',
    ]);
  });

  it('audits rule hazards with an operational warning exit code', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-rule-audit-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'employee(bob). eligible(X) :- employee(X), \\+ blocked(X).',
      { opId: 'baseline' }
    );

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'audit-rules', 'eligible', '--direction', 'upstream'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(2);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'review',
      warningCount: 1,
      findings: [
        {
          severity: 'warning',
          code: 'open_negated_input',
          predicateKeys: ['blocked/1'],
        },
      ],
      topology: { selection: { focus: 'eligible/1', direction: 'upstream' } },
      graph: {
        nodes: expect.arrayContaining([
          expect.objectContaining({ kind: 'finding', code: 'open_negated_input' }),
        ]),
      },
    });
  });

  it('searches local facts and sources without an LLM', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-search-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'dentist(rahul, chen).', {
      opId: 'dentist-source',
      sourceText: 'Rahul dentist is Doctor Chen.',
    });
    store.assert('default', 'pet(rahul, luna).', {
      opId: 'pet-source',
      sourceText: 'Rahul cat is Luna.',
    });

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'search',
        'Doctor Chen',
        '--kind',
        'fact',
        '--search-limit',
        '5',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'matches',
      results: [
        {
          rank: 1,
          clause: 'dentist(rahul, chen).',
          sources: [{ opId: 'dentist-source' }],
          reasons: expect.arrayContaining([
            expect.objectContaining({ kind: 'source_phrase' }),
          ]),
        },
      ],
    });
  });

  it('browses an explicit entity neighborhood without an LLM', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-browse-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'works_at(mira, acme). works_at(rahul, acme).', {
      opId: 'employment',
    });
    store.assert('default', 'lives_in(rahul, melbourne).', {
      opId: 'home',
    });

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'browse',
        'mira',
        '--browse-depth',
        '3',
        '--claim-limit',
        '10',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'matches',
      selection: {
        focus: 'mira',
        depth: 3,
        selectedClaims: 3,
      },
      graph: {
        nodes: expect.arrayContaining([
          expect.objectContaining({ kind: 'claim', predicate: 'lives_in' }),
        ]),
      },
    });
  });

  it('finds shortest explicit entity paths without an LLM', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-connect-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `works_at(mira, acme). works_at(rahul, acme). lives_in(rahul, melbourne).
       colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.`,
      { opId: 'connection-source' }
    );

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'connect',
        'mira',
        'rahul',
        '--path-depth',
        '2',
        '--path-limit',
        '3',
        '--include-derived',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'connected',
      shortestHops: 1,
      searchComplete: true,
      includeDerived: true,
      paths: expect.arrayContaining([
        expect.objectContaining({
          entities: ['mira', 'rahul'],
          segments: [
            expect.objectContaining({
              predicate: 'colleague',
              derived: true,
              rule: 1,
            }),
          ],
        }),
      ]),
      graph: {
        nodes: expect.arrayContaining([
          expect.objectContaining({
            kind: 'claim',
            sources: [expect.objectContaining({ opId: 'connection-source' })],
          }),
        ]),
      },
    });
  });

  it('exports and verifies a content-addressed bundle without mutating memory', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-bundle-'));
    const home = join(root, 'home');
    const file = join(root, 'knowledge.json');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('personal', 'pet(rahul, luna).', {
      opId: 'pet-source',
      sourceText: 'My cat is Luna.',
    });

    const exported = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'bundle', '--namespaces', 'personal'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );
    expect(exported.status).toBe(0);
    const bundle = JSON.parse(exported.stdout);
    expect(bundle).toMatchObject({
      format: 'rembero-knowledge-bundle',
      view: { kind: 'current' },
      namespaces: [
        {
          namespace: 'personal',
          clauses: [
            {
              clause: 'pet(rahul, luna).',
              sources: [{ opId: 'pet-source', text: 'My cat is Luna.' }],
            },
          ],
        },
      ],
    });
    writeFileSync(file, exported.stdout);
    const verified = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'verify-bundle', file],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(verified.status).toBe(0);
    expect(JSON.parse(verified.stdout)).toMatchObject({
      valid: true,
      namespaces: ['personal'],
      clauseCount: 1,
      sourceCount: 1,
    });
    expect(store.load('personal').map(serializeClause)).toEqual([
      'pet(rahul, luna).',
    ]);
  });

  it('exports and verifies the real-PDF Memorg memory without creating a local store', () => {
    const root = mkdtempSync(join(tmpdir(), 'remembero-cli-document-memorg-'));
    const home = join(root, 'home');
    const file = join(root, 'document-intelligence.memorg.json');
    const exported = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'document-memorg'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(exported.status).toBe(0);
    expect(existsSync(home)).toBe(false);
    const artifact = JSON.parse(exported.stdout);
    expect(artifact).toMatchObject({
      format: 'remembero-memorg-import',
      version: 1,
      target: { package: 'memorg', version: '0.1.2' },
      sha256: '5890e2945a534d0b871f0fa70fdd54b918704bbd8e753544a2dacef8a09ca531',
    });
    expect(artifact.items).toHaveLength(66);

    writeFileSync(file, exported.stdout);
    const verified = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'verify-document-memorg', file],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(verified.status).toBe(0);
    expect(JSON.parse(verified.stdout)).toMatchObject({
      valid: true,
      itemCount: 66,
      documentCount: 4,
      acceptedClaimCount: 17,
      proposedClaimCount: 4,
    });
    expect(existsSync(home)).toBe(false);
  });

  it('runs knowledge regression files with CI-friendly pass and failure exits', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-checks-'));
    const home = join(root, 'home');
    const passingFile = join(root, 'passing.json');
    const failingFile = join(root, 'failing.json');
    const coverageFile = join(root, 'coverage.json');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'item(a). item(b). copy(X) :- item(X).', { opId: 'items' });
    writeFileSync(
      passingFile,
      JSON.stringify({
        version: 1,
        checks: [
          {
            name: 'items',
            query: 'item(X)',
            expect: {
              kind: 'rows',
              order: 'set',
              rows: [{ X: 'b' }, { X: 'a' }],
            },
          },
        ],
      })
    );
    writeFileSync(
      failingFile,
      JSON.stringify({
        version: 1,
        checks: [
          {
            name: 'missing item',
            query: 'item(c)',
            expect: { kind: 'nonempty' },
          },
        ],
      })
    );
    writeFileSync(
      coverageFile,
      JSON.stringify({
        version: 1,
        coverage: { minimumPercent: 100 },
        checks: [
          {
            name: 'facts only',
            query: 'item(X)',
            expect: {
              kind: 'rows',
              order: 'set',
              rows: [{ X: 'a' }, { X: 'b' }],
            },
          },
        ],
      })
    );
    const run = (file: string) =>
      spawnSync(process.execPath, [resolve('dist/cli.js'), 'test-knowledge', file], {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      });
    const passing = run(passingFile);
    expect(passing.status).toBe(0);
    expect(JSON.parse(passing.stdout)).toMatchObject({
      status: 'passed',
      passedCount: 1,
    });
    const failing = run(failingFile);
    expect(failing.status).toBe(2);
    expect(JSON.parse(failing.stdout)).toMatchObject({
      status: 'failed',
      failedCount: 1,
      checks: [
        {
          whyNot: {
            status: 'blocked',
            failures: [{ reason: 'missing_fact', goal: 'item(c)' }],
          },
        },
      ],
    });
    const coverage = run(coverageFile);
    expect(coverage.status).toBe(2);
    expect(JSON.parse(coverage.stdout)).toMatchObject({
      status: 'failed',
      passedCount: 1,
      failedCount: 0,
      coveragePassed: false,
      coverage: {
        totalRules: 1,
        coveredRules: 0,
        minimumPercent: 100,
        passed: false,
      },
    });
  });

  it('profiles deterministic query work without timing fields', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-profile-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `${Array.from(
        { length: 200 },
        (_, index) => `related(person_${index}, topic_${index % 5}).`
      ).join('\n')}
       selected(person_199).
       relevant(X, Y) :- selected(X), related(X, Y).`,
      { opId: 'profile-program' }
    );
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'profile',
        'relevant(X, Y)',
        '--compare-scan',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(result.status).toBe(0);
    const payload = JSON.parse(result.stdout);
    expect(payload).toMatchObject({
      equivalent: true,
      explanation: {
        rows: [{ bindings: { X: 'person_199', Y: 'topic_4' } }],
      },
      workReduction: { candidateFactsAvoided: expect.any(Number) },
    });
    expect(JSON.stringify(payload)).not.toMatch(/duration|elapsed|millisecond/i);
  });

  it('supersedes multiple fact patterns with exact valid-time archives and safe retries', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-supersede-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('personal', 'works_at(mira, acme). title(mira, engineer).', {
      opId: 'prior-employment',
    });
    const run = (replacement: string, at = '2026-08-16T16:59:00.000Z') =>
      spawnSync(
        process.execPath,
        [
          resolve('dist/cli.js'),
          'supersede',
          replacement,
          '--namespace',
          'personal',
          '--pattern',
          'works_at(mira, _)',
          '--pattern',
          'title(mira, _)',
          '--at',
          at,
          '--op-id',
          'cli-employment-correction',
        ],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );

    const first = run('works_at(mira, initech). title(mira, lead).');
    const replay = run('works_at(mira, initech). title(mira, lead).');
    const conflict = run('works_at(mira, other). title(mira, lead).');

    expect(first.status).toBe(0);
    expect(JSON.parse(first.stdout)).toEqual({
      added: ['works_at(mira, initech).', 'title(mira, lead).'],
      duplicates: 0,
      retracted: 2,
      archived: [
        "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
        "title_until(mira, engineer, '2026-08-16T16:59:00.000Z').",
      ],
      opId: 'cli-employment-correction',
    });
    expect(replay.status).toBe(0);
    expect(replay.stdout).toBe(first.stdout);
    expect(conflict.status).toBe(4);
    expect(JSON.parse(conflict.stderr)).toMatchObject({
      error: 'operation_conflict',
      operation: 'supersede',
      namespace: 'personal',
      opId: 'cli-employment-correction',
    });
    expect(new MemoryStore(join(home, 'memory')).load('personal').map(serializeClause).sort())
      .toEqual([
        'title(mira, lead).',
        "title_until(mira, engineer, '2026-08-16T16:59:00.000Z').",
        'works_at(mira, initech).',
        "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
      ].sort());

    new MemoryStore(join(home, 'memory')).assert(
      'personal',
      'temporary_assignment(mira, atlas).'
    );
    const ended = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'supersede',
        '--namespace',
        'personal',
        '--pattern',
        'temporary_assignment(mira, _)',
        '--at',
        '2026-08-17T00:00:00.000Z',
        '--op-id',
        'cli-assignment-ended',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(ended.status).toBe(0);
    expect(JSON.parse(ended.stdout)).toEqual({
      added: [],
      duplicates: 0,
      retracted: 1,
      archived: [
        "temporary_assignment_until(mira, atlas, '2026-08-17T00:00:00.000Z').",
      ],
      opId: 'cli-assignment-ended',
    });
  });

  it('requires supersede patterns and a canonical UTC timestamp', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-supersede-invalid-'));
    const home = join(root, 'home');
    new MemoryStore(join(home, 'memory')).assert('default', 'status(mira, active).');
    const run = (extra: string[]) =>
      spawnSync(
        process.execPath,
        [resolve('dist/cli.js'), 'supersede', 'status(mira, paused).', ...extra],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );

    const noPattern = run([]);
    expect(noPattern.status).toBe(1);
    expect(noPattern.stderr).toMatch(/requires at least one fact pattern/i);
    const invalidAt = run([
      '--pattern',
      'status(mira, _)',
      '--at',
      '2026-08-16 16:59:00',
    ]);
    expect(invalidAt.status).toBe(1);
    expect(invalidAt.stderr).toMatch(/canonical UTC timestamp/i);
    const destructiveMode = run([
      '--pattern',
      'status(mira, _)',
      '--valid-time-mode',
      'delete',
    ]);
    expect(destructiveMode.status).toBe(1);
    expect(destructiveMode.stderr).toMatch(/always preserves _until history/i);
    expect(new MemoryStore(join(home, 'memory')).load('default').map(serializeClause)).toEqual([
      'status(mira, active).',
    ]);
  });

  it('rejects operation ids on commands without idempotent write semantics', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-op-id-command-'));
    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'query', 'f(X)', '--op-id', 'unsupported'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: join(root, 'home') },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(
      /--op-id is available for assert, accept, reject, supersede, forget, import, checkpoint, apply-rule-change, and apply-memory/i
    );
  });

  it('rejects recall answer mode on non-recall commands', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-answer-mode-'));
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'query',
        'item(X)',
        '--answer-mode',
        'deterministic',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: join(root, 'home') },
      }
    );
    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/available only for serve, recall, or recall-explain/i);
  });

  it('queries an exact recorded snapshot without changing current knowledge', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-recorded-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert('default', 'status(mira, active).', { opId: 'before' });
    store.replace('default', ['status(mira, _)'], 'status(mira, paused).', {
      opId: 'after',
    });

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'query',
        'status(mira, State)',
        '--as-of-sequence',
        '1',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      bindings: [{ State: 'active' }],
      recordedSnapshot: {
        sequence: 1,
        journalEntries: 2,
        namespaces: ['default'],
      },
    });
    expect(store.load('default').map(serializeClause)).toEqual(['status(mira, paused).']);
  });

  it('exports one complete result support graph without changing query rows', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-graph-select-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `parent(alice, bob). parent(bob, carol). parent(carol, dan).
       ancestor(X, Y) :- parent(X, Y).
       ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).`
    );

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'explain',
        'ancestor(alice, Descendant)',
        '--graph-result',
        '2',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(0);
    const payload = JSON.parse(result.stdout);
    expect(payload.rows.map((row: { bindings: Record<string, string> }) => row.bindings)).toEqual([
      { Descendant: 'bob' },
      { Descendant: 'carol' },
      { Descendant: 'dan' },
    ]);
    expect(payload.graphSelection).toMatchObject({
      selector: { kind: 'result', row: 2 },
    });
    expect(payload.graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'claim', predicate: 'parent', values: ['bob', 'carol'] }),
      ])
    );
    expect(payload.graph.nodes).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'claim', predicate: 'parent', values: ['carol', 'dan'] }),
      ])
    );
  });

  it('rejects ambiguous graph selectors before evaluating a query', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-graph-invalid-'));
    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'explain',
        'answer(a)',
        '--graph-result',
        '1',
        '--graph-support',
        'claim:answer',
      ],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: join(root, 'home') },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/mutually exclusive/i);
  });

  it('keeps literal queries unchanged and enables explicit canonical identity reads', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-identity-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(works_at, 2, 0).
       works_at('Mira Patel', acme).`
    );

    const literal = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'query', 'works_at(mira, Company)'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    const canonical = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'query',
        'works_at(mira, Company)',
        '--entity-identity',
        'canonical',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(literal.status).toBe(0);
    expect(JSON.parse(literal.stdout)).toEqual([]);
    expect(canonical.status).toBe(0);
    expect(JSON.parse(canonical.stdout)).toEqual([{ Company: 'acme' }]);
  });

  it('checks explicit integrity constraints and exits 2 when violations exist', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-integrity-'));
    const home = join(root, 'home');
    const asserted = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'assert',
        'status(mira, active). status(mira, terminated). :- status(Person, active), status(Person, terminated).',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(asserted.status).toBe(0);
    expect(JSON.parse(asserted.stdout).added).toHaveLength(3);

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'check', '--max-violations', '10'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(2);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'violations',
      constraintCount: 1,
      violationCount: 1,
      checks: [{ rows: [{ bindings: { Person: 'mira' } }] }],
    });
  });

  it('returns meaningful exit status for immutable knowledge health', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-health-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'employee(alice). :- employee(X), suspended(X).',
      { opId: 'health-baseline' }
    );
    store.assert('default', 'suspended(alice).', { opId: 'health-violation' });
    const runHealth = (extra: string[] = []) =>
      spawnSync(
        process.execPath,
        [resolve('dist/cli.js'), 'health', '--namespaces', 'default', ...extra],
        { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
      );

    const current = runHealth();
    expect(current.status).toBe(3);
    expect(JSON.parse(current.stdout)).toMatchObject({
      status: 'violations',
      integrity: { violationCount: 1 },
    });
    const recorded = runHealth(['--as-of-sequence', '1']);
    expect(recorded.status).toBe(2);
    expect(JSON.parse(recorded.stdout)).toMatchObject({
      status: 'review',
      integrity: { violationCount: 0 },
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('enforces an environment-configured knowledge suite on direct writes', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-check-enforcement-'));
    const home = join(root, 'home');
    const suite = join(root, 'checks.json');
    writeFileSync(
      suite,
      JSON.stringify({
        version: 1,
        checks: [
          {
            name: 'forbidden stays absent',
            query: 'forbidden(a)',
            expect: { kind: 'empty' },
          },
        ],
      })
    );
    const env = {
      ...process.env,
      REMBERO_HOME: home,
      REMBERO_CHECK_MODE: 'strict',
      REMBERO_CHECK_SUITE: suite,
      REMBERO_CHECK_NAMESPACES: 'default',
    };
    const safe = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'assert', 'safe(a).', '--op-id', 'safe-check-write'],
      { encoding: 'utf8', env }
    );
    expect(safe.status).toBe(0);

    const blocked = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'assert',
        'forbidden(a).',
        '--op-id',
        'blocked-check-write',
      ],
      { encoding: 'utf8', env }
    );
    expect(blocked.status).toBe(8);
    expect(JSON.parse(blocked.stderr)).toMatchObject({
      error: 'knowledge_check_enforcement',
      mode: 'strict',
      candidate: { status: 'failed' },
    });
    expect(
      new MemoryStore(join(home, 'memory')).load('default').map(serializeClause)
    ).toEqual(['safe(a).']);
  });

  it('returns a focused current or recorded conflict view with meaningful exit status', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-conflicts-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'status(mira, active). :- status(Person, active), status(Person, terminated).',
      { opId: 'baseline' }
    );
    store.assert('default', 'status(mira, terminated).', { opId: 'later' });

    const current = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'conflicts', 'mira'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(current.status).toBe(2);
    expect(JSON.parse(current.stdout)).toMatchObject({
      status: 'violations',
      focus: 'mira',
      matchingViolationCount: 1,
      clusterCount: 1,
      clusters: [{ focus: 'mira', rows: [{ focusBinding: 'Person' }] }],
    });

    const recorded = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'conflicts',
        'mira',
        '--as-of-sequence',
        '1',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );
    expect(recorded.status).toBe(0);
    expect(JSON.parse(recorded.stdout)).toMatchObject({
      status: 'consistent',
      focus: 'mira',
      matchingViolationCount: 0,
      clusterCount: 0,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('returns zero for a consistent constrained knowledge base', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-integrity-clean-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'status(mira, active). :- status(Person, active), status(Person, terminated).'
    );

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'check'],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      status: 'consistent',
      violationCount: 0,
    });
  });

  it('rejects a violating write atomically with structured evidence and exit 3', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-enforcement-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      'active(mira). :- active(Person), suspended(Person).'
    );
    const before = readFileSync(join(home, 'memory', 'default.dl'), 'utf8');

    const result = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'assert',
        'suspended(mira).',
        '--integrity-mode',
        'strict',
      ],
      { encoding: 'utf8', env: { ...process.env, REMBERO_HOME: home } }
    );

    expect(result.status).toBe(3);
    expect(result.stdout).toBe('');
    expect(JSON.parse(result.stderr)).toMatchObject({
      error: 'integrity_violation',
      mode: 'strict',
      introducedViolationCount: 1,
      candidate: {
        checks: [{ rows: [{ bindings: { Person: 'mira' } }] }],
      },
    });
    expect(readFileSync(join(home, 'memory', 'default.dl'), 'utf8')).toBe(before);
  });

  it('supports migration-friendly no-new-violations enforcement from the environment', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-enforcement-migrate-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    store.assert(
      'default',
      `active(mira). suspended(mira).
       :- active(Person), suspended(Person).`
    );
    const env = {
      ...process.env,
      REMBERO_HOME: home,
      REMBERO_INTEGRITY_MODE: 'no_new_violations',
    };

    const unrelated = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'assert', 'project(atlas).'],
      { encoding: 'utf8', env }
    );
    expect(unrelated.status).toBe(0);

    const newViolation = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'assert', 'active(alex). suspended(alex).'],
      { encoding: 'utf8', env }
    );
    expect(newViolation.status).toBe(3);
    expect(JSON.parse(newViolation.stderr)).toMatchObject({
      mode: 'no_new_violations',
      baselineViolationCount: 1,
      introducedViolationCount: 1,
    });
  });
});

describe('auto-capture CLI', () => {
  it('fails closed when a settings option is missing its path', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-hooks-missing-'));
    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'init-hooks', '--settings'],
      {
        encoding: 'utf8',
        env: { ...process.env, CLAUDE_CONFIG_DIR: root },
      }
    );

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('--settings requires a value');
    expect(existsSync(join(root, 'settings.json'))).toBe(false);
  });

  it('installs and removes only its managed Claude hook', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-hooks-'));
    const settingsPath = join(root, 'settings.json');
    writeFileSync(
      settingsPath,
      JSON.stringify({ hooks: { Stop: [{ hooks: [{ type: 'command', command: 'existing' }] }] } }),
      'utf8'
    );

    const install = spawnSync(
      process.execPath,
      [
        resolve('dist/cli.js'),
        'init-hooks',
        '--settings',
        settingsPath,
        '--namespace',
        'personal',
        '--daily-cap',
        '3',
        '--tail-bytes',
        '8192',
      ],
      { encoding: 'utf8', env: { ...process.env } }
    );
    expect(install.status).toBe(0);
    expect(install.stdout).toContain('installed Remembero Claude hook');
    const installed = JSON.parse(readFileSync(settingsPath, 'utf8'));
    const handlers = installed.hooks.Stop.flatMap(
      (group: { hooks: Record<string, unknown>[] }) => group.hooks
    );
    expect(handlers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ command: 'existing' }),
        expect.objectContaining({
          type: 'command',
          async: true,
          args: expect.arrayContaining(['remember', '--batch', 'personal']),
        }),
      ])
    );

    const remove = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'init-hooks', '--remove', '--settings', settingsPath],
      { encoding: 'utf8', env: { ...process.env } }
    );
    expect(remove.status).toBe(0);
    const removed = JSON.parse(readFileSync(settingsPath, 'utf8'));
    expect(removed.hooks.Stop).toEqual([
      { hooks: [{ type: 'command', command: 'existing' }] },
    ]);
  });

  it('lists and prunes numbered auto-captured facts end to end', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-review-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory'));
    const captureId = 'capture-review-cli';
    const opId = 'operation-review-cli';
    const now = new Date();
    store.note('personal', 'auto_capture', {
      captureId,
      status: 'started',
      source: 'claude-stop',
      sessionId: 'session-review-cli',
    }, now);
    store.assert('personal', 'prefers_theme(user, dark).', {
      captureId,
      opId,
      origin: 'claude-stop',
      sourceText: 'Auto-captured from a Claude Code Stop hook',
      at: now,
    });
    store.finishAutoCapture('personal', captureId, 'captured', { added: 1 }, now);

    const review = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'review', '--namespace', 'personal', '--json'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );
    expect(review.status).toBe(0);
    expect(JSON.parse(review.stdout).facts).toEqual([
      expect.objectContaining({
        clause: 'prefers_theme(user, dark).',
        current: true,
      }),
    ]);

    const prune = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'review', '--namespace', 'personal', '--forget', '1'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );
    expect(prune.status).toBe(0);
    expect(prune.stdout).toContain('removed 1 auto-captured fact');
    expect(store.load('personal').map(serializeClause)).toEqual([]);
  });

  it('prints temporal history as JSON with deterministic event ordering', () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-cli-history-'));
    const home = join(root, 'home');
    const store = new MemoryStore(join(home, 'memory')) as MemoryStore & {
      supersede: (
        namespace: string,
        patterns: string[],
        replacements: string,
        context?: Record<string, unknown>
      ) => unknown;
    };

    store.assert('personal', 'works_at(mira, acme).', {
      opId: 'source-1',
      sourceText: 'Mira works at Acme.',
      at: new Date('2026-08-10T09:00:00.000Z'),
    });
    store.supersede('personal', ['works_at(mira, _)'], 'works_at(mira, initech).', {
      opId: 'source-2',
      sourceText: 'Mira now works at Initech.',
      at: new Date('2026-08-16T16:59:00.000Z'),
    });

    const result = spawnSync(
      process.execPath,
      [resolve('dist/cli.js'), 'history', 'works_at(mira, _)', '--namespace', 'personal', '--json'],
      {
        encoding: 'utf8',
        env: { ...process.env, REMBERO_HOME: home },
      }
    );

    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({
      pattern: 'works_at(mira, _)',
      namespaces: ['personal'],
      events: [
        expect.objectContaining({
          sequence: 1,
          position: 0,
          action: 'asserted',
          clause: 'works_at(mira, acme).',
        }),
        expect.objectContaining({
          sequence: 2,
          position: 0,
          action: 'superseded',
          clause: 'works_at(mira, acme).',
          archivedAs: "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
          validUntil: '2026-08-16T16:59:00.000Z',
        }),
        expect.objectContaining({
          sequence: 2,
          position: 2,
          action: 'asserted',
          clause: 'works_at(mira, initech).',
          current: true,
        }),
      ],
    });
  });
});
