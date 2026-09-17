import { describe, it, expect, beforeEach } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SessionStore } from '../src/sessions/store.js';
import { MemoryStore, OperationConflictError } from '../src/store/store.js';
import { IntegrityViolationError } from '../src/knowledge/enforcement.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { MAX_INPUT_BYTES, MAX_NAMESPACE_COUNT } from '../src/safety.js';
import { serializeClause } from '../src/engine/index.js';
import {
  assertFactsTool,
  assertTentativeTool,
  checkIntegrityTool,
  checkpointJournalTool,
  conflictViewsTool,
  explainQueryTool,
  forgetSessionsTool,
  forgetTool,
  listMemoriesTool,
  listCheckpointsTool,
  queryTool,
  resolveTentativeTool,
  reviewTentativeTool,
  recallExplainTool,
  rememberTool,
  proposeMemoryTool,
  applyMemoryProposalTool,
  recallTool,
  supersedeFactsTool,
  whatIfTool,
  applyRuleChangeProposalTool,
  whyNotTool,
  topologyTool,
  recordedDiffTool,
  repairPlanTool,
  auditRulesTool,
  searchKnowledgeTool,
  browseKnowledgeGraphTool,
  connectKnowledgeGraphTool,
  exportKnowledgeBundleTool,
  verifyKnowledgeBundleTool,
  runKnowledgeChecksTool,
  profileKnowledgeTool,
  knowledgeHealthTool,
} from '../src/mcp/tools.js';
import { serializeKnowledgeBundle } from '../src/knowledge/bundle.js';

class ScriptedLlm implements LlmClient {
  constructor(private responses: string[]) {}
  async complete(_messages: ChatMessage[]): Promise<string> {
    const next = this.responses.shift();
    if (next === undefined) throw new Error('out of responses');
    return next;
  }
}

let store: MemoryStore;

beforeEach(() => {
  store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-tools-')));
});

describe('MCP tool handlers', () => {
  it('remember extracts via LLM and stores', async () => {
    // constants must be grounded in the input: "luna" is, "luna_the_cat" is not
    const llm = new ScriptedLlm(['pet(rahul, luna).']);
    const result = await rememberTool(
      { store, llm, selfAtom: 'rahul' },
      { text: 'My cat is called Luna' },
    );
    expect(result.added).toEqual(['pet(rahul, luna).']);
    expect(result.opId).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('recall answers from memory', async () => {
    store.assert('default', 'pet(rahul, luna).');
    const llm = new ScriptedLlm(['?- pet(rahul, X).', 'Your cat is Luna.']);
    const result = await recallTool(
      { store, llm },
      { question: 'What is my cat called?' },
    );
    expect(result.answer).toBe('Your cat is Luna.');
    expect(result.bindings).toEqual([{ X: 'luna' }]);
  });

  it('recall renders successful bindings locally in deterministic answer mode', async () => {
    store.assert('default', 'pet(rahul, luna).');
    const llm = new ScriptedLlm(['?- pet(rahul, Name).']);
    const result = await recallTool(
      { store, llm },
      { question: 'What is my cat called?', answerMode: 'deterministic' },
    );
    expect(result).toMatchObject({
      status: 'answered',
      answerMode: 'deterministic',
      answer: 'Result for pet(rahul, Name): Name = luna.',
    });
  });

  it('recall answers from an explicit recorded snapshot', async () => {
    store.assert('default', 'status(mira, active).', { opId: 'past' });
    store.replace('default', ['status(mira, _)'], 'status(mira, paused).', {
      opId: 'current',
    });
    const llm = new ScriptedLlm([
      '?- status(mira, State).',
      'Mira was active.',
    ]);

    const result = await recallTool(
      { store, llm },
      { question: 'What was Mira status?', recordedSequence: 1 },
    );

    expect(result.answer).toBe('Mira was active.');
    expect(result.bindings).toEqual([{ State: 'active' }]);
    expect(result.recordedSnapshot).toMatchObject({
      sequence: 1,
      journalEntries: 2,
    });
  });

  it('assert_facts takes raw Datalog without any LLM', async () => {
    const result = assertFactsTool(
      { store },
      { clauses: 'f(a). g(X) :- f(X).' },
    );
    expect(result.added).toEqual(['f(a).', 'g(X) :- f(X).']);
    expect(result.duplicates).toBe(0);
    expect(result.opId).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('assert_facts and forget forward caller operation ids for safe retries', () => {
    const asserted = assertFactsTool(
      { store },
      { clauses: 'f(a). f(b).', opId: 'tool-assert' },
    );
    expect(
      assertFactsTool(
        { store },
        { clauses: 'f(a). f(b).', opId: 'tool-assert' },
      ),
    ).toEqual(asserted);
    expect(() =>
      assertFactsTool({ store }, { clauses: 'g(a).', opId: 'tool-assert' }),
    ).toThrow(OperationConflictError);

    const forgotten = forgetTool(
      { store },
      { pattern: 'f(_)', opId: 'tool-forget' },
    );
    expect(
      forgetTool({ store }, { pattern: 'f( _ )', opId: 'tool-forget' }),
    ).toEqual(forgotten);
    expect(forgotten).toEqual({ removed: 2, opId: 'tool-forget' });
  });

  it('reviews and resolves tentative facts without mixing them into accepted reads', () => {
    expect(() =>
      assertFactsTool(
        { store },
        { clauses: "rembero_tentative('status(mira, active).')." },
      ),
    ).toThrow(/use assert_tentative/i);
    expect(
      assertTentativeTool(
        { store },
        {
          namespace: 'personal',
          clauses: 'status(mira, active).',
          opId: 'tentative-status',
        },
      ),
    ).toEqual({
      added: ['status(mira, active).'],
      duplicates: 0,
      opId: 'tentative-status',
    });
    expect(
      queryTool(
        { store },
        { namespaces: ['personal'], query: 'status(mira, State)' },
      ).bindings,
    ).toEqual([]);
    expect(
      explainQueryTool(
        { store },
        {
          namespaces: ['personal'],
          query: 'status(mira, State)',
          trustMode: 'include_tentative',
        },
      ),
    ).toMatchObject({
      trustMode: 'include_tentative',
      rows: [
        {
          bindings: { State: 'active' },
          proofs: [{ trust: 'tentative' }],
        },
      ],
    });
    expect(
      reviewTentativeTool({ store }, { namespaces: ['personal'] }),
    ).toMatchObject({
      count: 1,
      claims: [
        {
          namespace: 'personal',
          clause: 'status(mira, active).',
          sources: [{ opId: 'tentative-status' }],
        },
      ],
    });
    const declarationPattern = "rembero_tentative('status(mira, active).')";
    expect(() =>
      forgetTool(
        { store },
        { namespace: 'personal', pattern: declarationPattern },
      ),
    ).toThrow(/use resolveTentative/i);
    expect(() =>
      supersedeFactsTool(
        { store },
        {
          namespace: 'personal',
          patterns: [declarationPattern],
          replacements: 'status(mira, paused).',
        },
      ),
    ).toThrow(/use resolveTentative/i);
    expect(
      resolveTentativeTool(
        { store },
        {
          namespace: 'personal',
          clauses: 'status(mira, active).',
          action: 'accept',
          opId: 'accept-status',
        },
      ),
    ).toEqual({
      action: 'accept',
      resolved: 1,
      added: ['status(mira, active).'],
      duplicates: 0,
      opId: 'accept-status',
    });
    expect(
      queryTool(
        { store },
        { namespaces: ['personal'], query: 'status(mira, State)' },
      ).bindings,
    ).toEqual([{ State: 'active' }]);
    expect(
      reviewTentativeTool({ store }, { namespaces: ['personal'] }),
    ).toEqual({
      claims: [],
      count: 0,
    });
    expect(
      queryTool(
        { store },
        {
          namespaces: ['personal'],
          query: 'status(mira, State)',
          trustMode: 'include_tentative',
          recordedSequence: 1,
        },
      ),
    ).toMatchObject({
      bindings: [{ State: 'active' }],
      trustMode: 'include_tentative',
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('keeps tentative claims outside policy unless an audit explicitly includes them', () => {
    store.assert(
      'personal',
      'active(mira). :- active(Person), suspended(Person).',
    );
    store.assertTentative('personal', 'suspended(mira).');

    expect(
      checkIntegrityTool({ store }, { namespaces: ['personal'] }).status,
    ).toBe('consistent');
    expect(
      checkIntegrityTool(
        { store },
        { namespaces: ['personal'], trustMode: 'include_tentative' },
      ),
    ).toMatchObject({
      status: 'violations',
      trustMode: 'include_tentative',
      checks: [
        {
          rows: [
            {
              proofs: [
                expect.objectContaining({ predicate: 'active' }),
                expect.objectContaining({
                  predicate: 'suspended',
                  trust: 'tentative',
                }),
              ],
            },
          ],
        },
      ],
    });
    expect(
      conflictViewsTool(
        { store },
        {
          namespaces: ['personal'],
          focus: 'mira',
          trustMode: 'include_tentative',
        },
      ),
    ).toMatchObject({
      trustMode: 'include_tentative',
      matchingViolationCount: 1,
      clusters: [{ focus: 'mira' }],
    });
    expect(
      listMemoriesTool({ store }, { namespaces: ['personal'] }).predicates.map(
        (group) => group.predicate,
      ),
    ).not.toContain('suspended/1');
    expect(
      listMemoriesTool(
        { store },
        { namespaces: ['personal'], trustMode: 'include_tentative' },
      ),
    ).toMatchObject({
      trustMode: 'include_tentative',
      predicates: expect.arrayContaining([
        expect.objectContaining({ predicate: 'suspended/1' }),
      ]),
    });
  });

  it('supersedes explicit facts atomically with exact archives and retry safety', () => {
    store.assert('personal', 'works_at(mira, acme). title(mira, engineer).', {
      opId: 'prior-employment',
    });
    const request = {
      patterns: ['works_at(mira, _)', 'title(mira, _)'],
      replacements: 'works_at(mira, initech). title(mira, lead).',
      namespace: 'personal',
      at: '2026-08-16T16:59:00.000Z',
      opId: 'employment-correction',
    };

    const first = supersedeFactsTool({ store }, request);
    const replay = supersedeFactsTool({ store }, request);

    expect(replay).toEqual(first);
    expect(first).toEqual({
      added: ['works_at(mira, initech).', 'title(mira, lead).'],
      duplicates: 0,
      retracted: 2,
      archived: [
        "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
        "title_until(mira, engineer, '2026-08-16T16:59:00.000Z').",
      ],
      opId: 'employment-correction',
    });
    expect(store.load('personal').map(serializeClause).sort()).toEqual(
      [
        'title(mira, lead).',
        "title_until(mira, engineer, '2026-08-16T16:59:00.000Z').",
        'works_at(mira, initech).',
        "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
      ].sort(),
    );
    expect(() =>
      supersedeFactsTool(
        { store },
        { ...request, replacements: 'works_at(mira, other).' },
      ),
    ).toThrow(OperationConflictError);
    expect(() =>
      supersedeFactsTool(
        { store },
        { ...request, at: '2026-08-16T17:00:00.000Z' },
      ),
    ).toThrow(OperationConflictError);
    const { at: _at, ...withoutAt } = request;
    expect(() => supersedeFactsTool({ store }, withoutAt)).toThrow(
      OperationConflictError,
    );

    store.assert('personal', 'status(mira, active).');
    const implicitTime = {
      patterns: ['status(mira, _)'],
      replacements: 'status(mira, paused).',
      namespace: 'personal',
      opId: 'implicit-time-correction',
    };
    expect(supersedeFactsTool({ store }, implicitTime)).toEqual(
      supersedeFactsTool({ store }, implicitTime),
    );

    store.assert('personal', 'temporary_assignment(mira, atlas).');
    expect(
      supersedeFactsTool(
        { store },
        {
          patterns: ['temporary_assignment(mira, _)'],
          namespace: 'personal',
          at: '2026-08-17T00:00:00.000Z',
          opId: 'assignment-ended',
        },
      ),
    ).toEqual({
      added: [],
      duplicates: 0,
      retracted: 1,
      archived: [
        "temporary_assignment_until(mira, atlas, '2026-08-17T00:00:00.000Z').",
      ],
      opId: 'assignment-ended',
    });

    store.assert('personal', 'stable(value).');
    const noOp = supersedeFactsTool(
      { store },
      {
        patterns: ['arrived(_)'],
        replacements: 'stable(value).',
        namespace: 'personal',
        opId: 'no-op-correction',
      },
    );
    expect(noOp).toMatchObject({ retracted: 0, added: [], duplicates: 1 });
    store.assert('personal', 'arrived(later).');
    expect(
      supersedeFactsTool(
        { store },
        {
          patterns: ['arrived(_)'],
          replacements: 'stable(value).',
          namespace: 'personal',
          opId: 'no-op-correction',
        },
      ),
    ).toEqual(noOp);
    expect(store.load('personal').map(serializeClause)).toContain(
      'arrived(later).',
    );
    expect(
      store
        .load('personal')
        .map(serializeClause)
        .some((clause) => clause.startsWith('arrived_until(')),
    ).toBe(false);
  });

  it('rejects ambiguous supersession timestamps before mutation', () => {
    store.assert('default', 'status(mira, active).');
    expect(() =>
      supersedeFactsTool(
        { store },
        {
          patterns: ['status(mira, _)'],
          replacements: 'status(mira, paused).',
          at: '2026-08-16 16:59:00',
        },
      ),
    ).toThrow(/canonical UTC timestamp/i);
    expect(store.load('default').map(serializeClause)).toEqual([
      'status(mira, active).',
    ]);
  });

  it('write tools share atomic integrity enforcement and structured rejection', () => {
    store.assert('default', 'active(mira). :- active(X), suspended(X).');
    expect(() =>
      assertFactsTool(
        { store, integrityEnforcement: { mode: 'strict' } },
        { clauses: 'suspended(mira).' },
      ),
    ).toThrow(IntegrityViolationError);
    expect(store.load('default').map(serializeClause)).toEqual([
      'active(mira).',
      ':- active(X), suspended(X).',
    ]);

    store.assert('default', 'manager(mira, rahul).');
    store.assert('default', ':- active(Person), \\+ manager(Person, _).');
    expect(() =>
      forgetTool(
        { store, integrityEnforcement: { mode: 'strict' } },
        { pattern: 'manager(mira, _)' },
      ),
    ).toThrow(IntegrityViolationError);

    store.assert('default', 'status(mira, active). :- status(X, suspended).');
    expect(() =>
      supersedeFactsTool(
        { store, integrityEnforcement: { mode: 'strict' } },
        {
          patterns: ['status(mira, _)'],
          replacements: 'status(mira, suspended).',
          at: '2026-08-16T16:59:00.000Z',
        },
      ),
    ).toThrow(IntegrityViolationError);
    expect(store.load('default').map(serializeClause)).toContain(
      'status(mira, active).',
    );
    expect(store.load('default').map(serializeClause)).not.toContain(
      "status_until(mira, active, '2026-08-16T16:59:00.000Z').",
    );
  });

  it('query evaluates raw Datalog and returns bindings', () => {
    store.assert('default', 'f(a). f(b). g(X) :- f(X), X != a.');
    const result = queryTool({ store }, { query: 'g(X)' });
    expect(result.bindings).toEqual([{ X: 'b' }]);
    store.assert('default', 'edge(a, b). edge(b, c).');
    expect(
      queryTool(
        { store },
        { query: 'select End where edge(a, Mid), edge(Mid, End)' },
      ).bindings,
    ).toEqual([{ End: 'c' }]);
  });

  it('proposes accepted natural-language memory without invoking a writer', async () => {
    store.assert('default', 'works_at(mira, acme).', { opId: 'proposal-base' });
    const result = await proposeMemoryTool(
      {
        store,
        llm: new ScriptedLlm([
          'retract works_at(mira, _).\nworks_at(mira, initech).',
        ]),
      },
      {
        text: 'Mira now works at Initech.',
        namespace: 'default',
      },
    );

    expect(result).toMatchObject({
      changed: true,
      proposal: {
        removeClauses: ['works_at(mira, acme).'],
        addClauses: ['works_at(mira, initech).'],
      },
    });
    expect(store.load('default').map(serializeClause)).toEqual([
      'works_at(mira, acme).',
    ]);
    const applied = applyMemoryProposalTool(
      { store },
      {
        proposal: JSON.stringify(result.proposal),
        opId: 'tool-reviewed-memory',
      },
    );
    expect(applied).toMatchObject({
      opId: 'tool-reviewed-memory',
      removed: [expect.any(Object)],
      added: [expect.any(Object)],
    });
    expect(store.load('default').map(serializeClause)).toEqual([
      'works_at(mira, initech).',
    ]);
  });

  it('simulates query and integrity impact without invoking a writer', () => {
    store.assert(
      'default',
      'status(mira, active). :- status(Person, active), status(Person, paused).',
      { opId: 'what-if-baseline' },
    );
    const result = whatIfTool(
      { store },
      {
        query: 'status(mira, State)',
        assume: 'status(mira, paused).',
      },
    );
    expect(result).toMatchObject({
      changed: true,
      resultDelta: {
        added: [{ bindings: { State: 'paused' } }],
        removed: [],
      },
      integrityDelta: {
        candidate: { status: 'violations', violationCount: 1 },
        introduced: [{ bindings: { Person: 'mira' } }],
      },
    });
    expect(store.clausesFor(['default'])).toHaveLength(2);
  });

  it('simulates rule and coverage impact from an exact recorded baseline', () => {
    store.assert('default', 'base(a).', { opId: 'recorded-base' });
    store.assert('default', 'base(b).', { opId: 'later-base' });
    const result = whatIfTool(
      { store },
      {
        query: 'derived(X)',
        assumeRules: 'derived(X) :- base(X).',
        recordedSequence: 1,
        checkSuite: JSON.stringify({
          version: 1,
          coverage: { minimumPercent: 100 },
          checks: [
            {
              name: 'recorded derivation',
              query: 'derived(a)',
              expect: { kind: 'nonempty' },
            },
          ],
        }),
      },
    );

    expect(result).toMatchObject({
      changed: true,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
      candidate: { rows: [{ bindings: { X: 'a' } }] },
      ruleAuditDelta: {
        baseline: { topology: { ruleCount: 0 } },
        candidate: { topology: { ruleCount: 1 } },
      },
      checkDelta: {
        baseline: { status: 'failed' },
        candidate: { status: 'passed', coverage: { percent: 100 } },
        fixed: ['recorded derivation'],
      },
    });
  });

  it('applies a reviewed current rule proposal through the tool boundary', () => {
    store.assert('default', 'base(a).', { opId: 'apply-base' });
    const preview = whatIfTool(
      { store },
      {
        query: 'derived(X)',
        assumeRules: 'derived(X) :- base(X).',
      },
    );
    const applied = applyRuleChangeProposalTool(
      { store },
      {
        proposal: JSON.stringify(preview.ruleProposal),
        opId: 'tool-reviewed-rule',
      },
    );

    expect(applied).toMatchObject({
      opId: 'tool-reviewed-rule',
      sequence: 2,
      added: [expect.any(Object)],
      audit: { topology: { ruleCount: 1 } },
    });
    expect(queryTool({ store }, { query: 'derived(X)' }).bindings).toEqual([
      { X: 'a' },
    ]);
  });

  it('reports current and recorded deterministic knowledge health through the tool boundary', () => {
    store.assert('default', 'base(a). derived(X) :- base(X).', {
      opId: 'health-base',
    });
    store.assert('default', 'later(b).', { opId: 'health-later' });

    expect(
      knowledgeHealthTool({ store }, { namespaces: ['default'] }),
    ).toMatchObject({
      status: 'healthy',
      clauseCount: 3,
      rules: { topology: { ruleCount: 1 } },
      provenance: { sourceCoveragePercent: 100 },
    });
    expect(
      knowledgeHealthTool(
        { store },
        { namespaces: ['default'], recordedSequence: 1 },
      ),
    ).toMatchObject({
      status: 'healthy',
      clauseCount: 2,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('explains current and recorded query blockers through the tool boundary', () => {
    store.assert('default', 'status(mira, active).', { opId: 'before' });
    store.replace('default', ['status(mira, _)'], 'status(mira, paused).', {
      opId: 'after',
    });

    expect(
      whyNotTool({ store }, { query: 'status(mira, active)' }),
    ).toMatchObject({
      status: 'blocked',
      failures: [
        {
          reason: 'missing_fact',
          nearby: [
            {
              fact: 'status(mira, paused).',
              explanation: {
                rows: [{ proofs: [{ sources: [{ opId: 'after' }] }] }],
              },
            },
          ],
        },
      ],
    });
    expect(
      whyNotTool(
        { store },
        { query: 'status(mira, active)', recordedSequence: 1 },
      ),
    ).toMatchObject({
      status: 'satisfied',
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
      explanation: { rows: [{ proofs: [{ sources: [{ opId: 'before' }] }] }] },
    });
  });

  it('maps focused current topology and exact recorded rule history', () => {
    store.assert('default', 'base(a). middle(X) :- base(X).', {
      opId: 'topology-before',
    });
    store.assert('default', 'output(X) :- middle(X).', {
      opId: 'topology-after',
    });

    expect(
      topologyTool({ store }, { focus: 'output', direction: 'upstream' }),
    ).toMatchObject({
      predicateCount: 3,
      ruleCount: 2,
      predicates: [{ key: 'base/1' }, { key: 'middle/1' }, { key: 'output/1' }],
      selection: { focus: 'output/1', direction: 'upstream' },
    });
    expect(topologyTool({ store }, { recordedSequence: 1 })).toMatchObject({
      predicateCount: 2,
      ruleCount: 1,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
      rules: [{ sources: [{ opId: 'topology-before' }] }],
    });
  });

  it('diffs recorded clauses and optional query proofs through the tool boundary', () => {
    store.assert('default', 'item(a).', { opId: 'diff-before' });
    store.assert('default', 'item(b).', { opId: 'diff-after' });

    expect(
      recordedDiffTool(
        { store },
        { fromSequence: 1, toSequence: 2, query: 'item(Value)' },
      ),
    ).toMatchObject({
      changed: true,
      clauses: {
        added: [{ clause: 'item(b).', sources: [{ opId: 'diff-after' }] }],
        removed: [],
      },
      topology: {
        changedNodes: [
          {
            before: { key: 'item/1', factCount: 1 },
            after: { key: 'item/1', factCount: 2 },
          },
        ],
      },
      queryImpact: {
        added: [{ bindings: { Value: 'b' } }],
        unchangedCount: 1,
      },
    });
  });

  it('returns verified proposal-only repairs through the tool boundary', () => {
    store.assert(
      'default',
      'employee(bob). eligible(X) :- employee(X), badge(X).',
      { opId: 'repair-baseline' },
    );

    expect(
      repairPlanTool(
        { store },
        { query: 'eligible(bob)', maxPlans: 4, maxSteps: 3 },
      ),
    ).toMatchObject({
      status: 'repairable',
      plans: [
        {
          assume: ['badge(bob).'],
          candidate: { rows: [{ bindings: {} }] },
          noNewViolationsSafe: true,
        },
      ],
    });
    expect(store.clausesFor(['default'])).toHaveLength(2);
  });

  it('audits current and recorded rule health through the tool boundary', () => {
    store.assert(
      'default',
      'employee(bob). eligible(X) :- employee(X), \\+ blocked(X).',
      { opId: 'audit-before' },
    );
    store.assert('default', 'blocked(bob).', { opId: 'audit-after' });

    expect(auditRulesTool({ store }, {})).toMatchObject({
      status: 'advisory',
      warningCount: 0,
      findings: [
        expect.objectContaining({
          code: 'inactive_derived_predicate',
          predicateKeys: ['eligible/1'],
        }),
      ],
    });
    expect(auditRulesTool({ store }, { recordedSequence: 1 })).toMatchObject({
      status: 'review',
      warningCount: 1,
      findings: [
        expect.objectContaining({
          code: 'open_negated_input',
          predicateKeys: ['blocked/1'],
        }),
      ],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('searches current and recorded knowledge locally through the tool boundary', () => {
    store.assert('default', 'dentist(rahul, chen).', {
      opId: 'dentist-source',
      sourceText: 'Rahul dentist is Chen.',
    });
    store.assert('default', 'pet(rahul, luna).', {
      opId: 'pet-source',
      sourceText: 'Rahul cat is Luna.',
    });

    expect(
      searchKnowledgeTool(
        { store },
        { text: 'cat Luna', kinds: ['fact'], limit: 5 },
      ),
    ).toMatchObject({
      status: 'matches',
      results: [
        {
          clause: 'pet(rahul, luna).',
          sources: [{ opId: 'pet-source' }],
        },
      ],
    });
    expect(
      searchKnowledgeTool({ store }, { text: 'cat Luna', recordedSequence: 1 }),
    ).toMatchObject({
      status: 'no_match',
      results: [],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('browses current and recorded explicit graph neighborhoods through the tool boundary', () => {
    store.assert('default', 'works_at(mira, acme).', {
      opId: 'mira-source',
    });
    store.assert('default', 'works_at(rahul, acme).', {
      opId: 'rahul-source',
    });

    expect(
      browseKnowledgeGraphTool({ store }, { focus: 'acme', depth: 1 }),
    ).toMatchObject({
      status: 'matches',
      selection: { selectedClaims: 2, totalGroundFacts: 2 },
      graph: {
        nodes: expect.arrayContaining([
          expect.objectContaining({
            kind: 'claim',
            values: ['mira', 'acme'],
          }),
          expect.objectContaining({
            kind: 'claim',
            values: ['rahul', 'acme'],
          }),
        ]),
      },
    });
    expect(
      browseKnowledgeGraphTool(
        { store },
        { focus: 'acme', recordedSequence: 1 },
      ),
    ).toMatchObject({
      selection: { selectedClaims: 1, totalGroundFacts: 1 },
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('connects entities through current and recorded explicit graph paths', () => {
    store.assert('default', 'works_at(mira, acme).', { opId: 'mira-source' });
    store.assert('default', 'works_at(rahul, acme).', { opId: 'rahul-source' });
    store.assert(
      'default',
      'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
      { opId: 'colleague-rule' },
    );

    expect(
      connectKnowledgeGraphTool(
        { store },
        { from: 'mira', to: 'rahul', maxDepth: 2, includeDerived: true },
      ),
    ).toMatchObject({
      status: 'connected',
      shortestHops: 1,
      includeDerived: true,
      paths: expect.arrayContaining([
        expect.objectContaining({
          entities: ['mira', 'rahul'],
          segments: [
            expect.objectContaining({ predicate: 'colleague', derived: true }),
          ],
        }),
      ]),
    });
    expect(
      connectKnowledgeGraphTool(
        { store },
        {
          from: 'mira',
          to: 'rahul',
          maxDepth: 2,
          includeDerived: true,
          recordedSequence: 3,
        },
      ),
    ).toMatchObject({
      status: 'connected',
      shortestHops: 1,
      includeDerived: true,
      recordedSnapshot: { sequence: 3, journalEntries: 3 },
    });
    expect(
      connectKnowledgeGraphTool(
        { store },
        {
          from: 'mira',
          to: 'rahul',
          maxDepth: 2,
          includeDerived: true,
          recordedSequence: 1,
        },
      ),
    ).toMatchObject({
      status: 'no_path',
      searchComplete: true,
      recordedSnapshot: { sequence: 1, journalEntries: 3 },
    });
  });

  it('exports and verifies current and recorded bundles through the tool boundary', () => {
    store.assert('first', 'item(a).', { opId: 'first-source' });
    store.assert('second', 'item(b).', { opId: 'second-source' });
    const current = exportKnowledgeBundleTool({ store }, {});
    expect(current.namespaces.map(({ namespace }) => namespace)).toEqual([
      'first',
      'second',
    ]);
    expect(
      verifyKnowledgeBundleTool({ bundle: serializeKnowledgeBundle(current) }),
    ).toMatchObject({
      valid: true,
      namespaceCount: 2,
      clauseCount: 2,
      sourceCount: 2,
    });

    const recorded = exportKnowledgeBundleTool(
      { store },
      { namespaces: ['first', 'second'], recordedSequence: 1 },
    );
    expect(recorded).toMatchObject({
      view: { kind: 'recorded', sequence: 1, journalEntries: 2 },
      namespaces: [
        { namespace: 'first', clauses: [{ clause: 'item(a).' }] },
        { namespace: 'second', clauses: [] },
      ],
    });
  });

  it('runs current and recorded knowledge suites through the tool boundary', () => {
    store.assert('default', 'item(a).', { opId: 'before' });
    store.assert('default', 'item(b).', { opId: 'after' });
    const suite = JSON.stringify({
      version: 1,
      checks: [
        {
          name: 'only first item',
          query: 'item(X)',
          expect: {
            kind: 'rows',
            order: 'exact',
            rows: [{ X: 'a' }],
          },
        },
      ],
    });
    expect(runKnowledgeChecksTool({ store }, { suite })).toMatchObject({
      status: 'failed',
      failedCount: 1,
      checks: [{ unexpectedRows: [{ X: 'b' }] }],
    });
    expect(
      runKnowledgeChecksTool({ store }, { suite, recordedSequence: 1 }),
    ).toMatchObject({
      status: 'passed',
      passedCount: 1,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('keeps coverage-only suite failure distinct from row failures', () => {
    store.assert('default', 'base(a). derived(X) :- base(X).', {
      opId: 'coverage-program',
    });
    const result = runKnowledgeChecksTool(
      { store },
      {
        suite: JSON.stringify({
          version: 1,
          coverage: { minimumPercent: 100 },
          checks: [
            {
              name: 'base only',
              query: 'base(a)',
              expect: { kind: 'nonempty' },
            },
          ],
        }),
      },
    );
    expect(result).toMatchObject({
      status: 'failed',
      passedCount: 1,
      failedCount: 0,
      coveragePassed: false,
      coverage: {
        totalRules: 1,
        coveredRules: 0,
        percent: 0,
        minimumPercent: 100,
      },
    });
  });

  it('profiles current and recorded deterministic work through the tool boundary', () => {
    store.assert(
      'default',
      `${Array.from(
        { length: 200 },
        (_, index) => `related(person_${index}, topic_${index % 5}).`,
      ).join('\n')}
       selected(person_199).
       relevant(X, Y) :- selected(X), related(X, Y).`,
      { opId: 'profile-program' },
    );
    const result = profileKnowledgeTool(
      { store },
      { query: 'relevant(X, Y)', compareFullScan: true, recordedSequence: 1 },
    );
    expect(result).toMatchObject({
      equivalent: true,
      explanation: {
        rows: [{ bindings: { X: 'person_199', Y: 'topic_4' } }],
      },
      indexed: { indexedRelationLookups: expect.any(Number) },
      fullScan: { indexedRelationLookups: 0 },
      workReduction: { candidateFactsAvoided: expect.any(Number) },
      recordedSnapshot: { sequence: 1, journalEntries: 1 },
    });
    expect(result.workReduction!.candidateFactsAvoided).toBeGreaterThan(100);
  });

  it('query and explain read deterministic recorded snapshots with past sources', () => {
    store.assert(
      'default',
      'works_at(mira, acme). coworker(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
      { opId: 'past-source', sourceText: 'Mira worked at Acme.' },
    );
    store.assert('default', 'works_at(rahul, acme).', {
      opId: 'second-source',
    });
    store.replace(
      'default',
      ['works_at(mira, _)'],
      'works_at(mira, initech).',
      {
        opId: 'current-source',
      },
    );

    expect(queryTool({ store }, { query: 'works_at(mira, C)' })).toEqual({
      bindings: [{ C: 'initech' }],
    });
    const past = queryTool(
      { store },
      { query: 'coworker(mira, Who)', recordedSequence: 2 },
    );
    expect(past).toEqual({
      bindings: [{ Who: 'rahul' }],
      recordedSnapshot: {
        sequence: 2,
        journalEntries: 3,
        namespaces: ['default'],
      },
    });
    const explained = explainQueryTool(
      { store },
      { query: 'works_at(mira, Company)', recordedSequence: 1 },
    );
    expect(explained.rows[0]).toMatchObject({
      bindings: { Company: 'acme' },
      proofs: [
        { sources: [{ opId: 'past-source', text: 'Mira worked at Acme.' }] },
      ],
    });
    expect(explained.recordedSnapshot?.sequence).toBe(1);
  });

  it('checkpoints the active journal without changing tool snapshot sequences', () => {
    store.assert('default', 'item(a).', { opId: 'checkpoint-a' });
    store.assert('default', 'item(b).', { opId: 'checkpoint-b' });
    expect(
      checkpointJournalTool(
        { store },
        {
          opId: 'tool-checkpoint',
          at: '2026-08-17T02:00:00.000Z',
          dryRun: true,
        },
      ),
    ).toMatchObject({ rotated: true, sequence: 2, segmentCount: 1 });
    const checkpoint = checkpointJournalTool(
      { store },
      {
        opId: 'tool-checkpoint',
        at: '2026-08-17T02:00:00.000Z',
      },
    );
    expect(checkpoint).toMatchObject({
      rotated: true,
      sequence: 2,
      checkpoint: { opId: 'tool-checkpoint' },
    });
    expect(listCheckpointsTool({ store })).toMatchObject({
      count: 1,
      checkpoints: [{ sequence: 2 }],
    });
    expect(
      queryTool({ store }, { query: 'item(Value)', recordedSequence: 1 }),
    ).toMatchObject({
      bindings: [{ Value: 'a' }],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('recomputes aggregate-derived facts from the selected recorded snapshot', () => {
    store.assert(
      'default',
      `member(red, alice).
       team_size(Team, Count) :- count(*) as Count where member(Team, Person).`,
      { opId: 'aggregate-baseline' },
    );
    store.assert('default', 'member(red, bob).', { opId: 'aggregate-later' });

    expect(
      queryTool({ store }, { query: 'team_size(red, Count)' }).bindings,
    ).toEqual([{ Count: '2' }]);
    expect(
      explainQueryTool(
        { store },
        { query: 'team_size(red, Count)', recordedSequence: 1 },
      ),
    ).toMatchObject({
      rows: [
        {
          bindings: { Count: '1' },
          proofs: [
            {
              aggregate: {
                value: 1,
                contributors: [
                  { proofs: [{ sources: [{ opId: 'aggregate-baseline' }] }] },
                ],
              },
            },
          ],
        },
      ],
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
  });

  it('applies integrity, listing, and explicit identity to the selected snapshot', () => {
    store.assert(
      'default',
      "rembero_alias('Mira Patel', mira). rembero_entity_position(active, 1, 0). active('Mira Patel'). :- active(Person), suspended(Person).",
      { opId: 'baseline' },
    );
    store.assert('default', 'suspended(mira).', { opId: 'later' });

    expect(
      checkIntegrityTool(
        { store },
        { recordedSequence: 1, entityIdentity: 'canonical' },
      ),
    ).toMatchObject({
      status: 'consistent',
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
    expect(
      checkIntegrityTool({ store }, { entityIdentity: 'canonical' }).status,
    ).toBe('violations');
    expect(
      conflictViewsTool(
        { store },
        { recordedSequence: 1, entityIdentity: 'canonical' },
      ),
    ).toMatchObject({
      status: 'consistent',
      matchingViolationCount: 0,
      recordedSnapshot: { sequence: 1, journalEntries: 2 },
    });
    expect(
      conflictViewsTool(
        { store },
        { focus: "'Mira Patel'", entityIdentity: 'canonical' },
      ),
    ).toMatchObject({
      status: 'violations',
      focus: 'mira',
      matchingViolationCount: 1,
      clusters: [{ focus: 'mira', rows: [{ focusBinding: 'Person' }] }],
    });
    const listed = listMemoriesTool({ store }, { recordedSequence: 1 });
    expect(
      listed.predicates.some((group) => group.predicate === 'suspended/1'),
    ).toBe(false);
    expect(listed.recordedSnapshot?.sequence).toBe(1);
    expect(
      queryTool(
        { store },
        {
          query: 'active(mira)',
          recordedSequence: 1,
          entityIdentity: 'canonical',
        },
      ).bindings,
    ).toEqual([{ yes: 'true' }]); // ground goal: one boolean row
  });

  it('query and explain_query accept arithmetic comparison filters', () => {
    store.assert(
      'default',
      'score(alice, 20). score(bob, 14). baseline(team, 10). ahead(X) :- score(X, S), baseline(team, B), S > B + 5.',
    );
    expect(queryTool({ store }, { query: 'ahead(Person)' })).toEqual({
      bindings: [{ Person: 'alice' }],
    });
    expect(
      explainQueryTool({ store }, { query: 'score(Person, S), S / 2 >= 10' })
        .rows,
    ).toHaveLength(1);
  });

  it('query and explain_query expose exact scalar aggregation', () => {
    store.assert('default', 'works_at(alice, acme). works_at(bob, acme).', {
      opId: 'aggregate-source',
    });
    const query = 'count(*) as Count where works_at(Person, acme)';

    expect(queryTool({ store }, { query })).toEqual({
      bindings: [{ Count: '2' }],
    });
    const explained = explainQueryTool({ store }, { query });
    expect(explained.rows[0]).toMatchObject({
      bindings: { Count: '2' },
      proofs: [
        {
          aggregated: true,
          op: 'count',
          value: 2,
          contributors: [
            { bindings: { Person: 'alice' } },
            { bindings: { Person: 'bob' } },
          ],
        },
      ],
    });
    expect(explained.graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'aggregate', value: 2 }),
      ]),
    );
  });

  it('rejects oversized inputs and namespace fan-out before evaluation', () => {
    expect(() =>
      queryTool({ store }, { query: 'x'.repeat(MAX_INPUT_BYTES + 1) }),
    ).toThrow(/query exceeds/i);
    expect(() =>
      listMemoriesTool(
        { store },
        {
          namespaces: Array.from(
            { length: MAX_NAMESPACE_COUNT + 1 },
            (_, i) => `ns${i}`,
          ),
        },
      ),
    ).toThrow(/namespace list exceeds/i);
  });

  it('explain_query returns proof sources and a query-scoped graph', () => {
    store.assert(
      'default',
      'works_at(rahul, acme). works_at(mira, acme). colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.',
      { opId: 'source-1', sourceText: 'Rahul and Mira work at Acme.' },
    );
    const result = explainQueryTool(
      { store },
      { query: 'colleague(rahul, Who)' },
    );
    expect(result.rows[0].bindings).toEqual({ Who: 'mira' });
    expect(result.rows[0].proofs[0]).toMatchObject({
      predicate: 'colleague',
      rule: 1,
      because: [
        { predicate: 'works_at', sources: [{ opId: 'source-1' }] },
        { predicate: 'works_at', sources: [{ opId: 'source-1' }] },
      ],
    });
    expect(result.graph.nodes.some((node) => node.kind === 'result')).toBe(
      true,
    );
  });

  it('explain_query returns bounded alternative proofs only when requested', () => {
    store.assert(
      'default',
      'left(a). right(a). answer(X) :- left(X). answer(X) :- right(X).',
    );

    const primary = explainQueryTool({ store }, { query: 'answer(a)' });
    const expanded = explainQueryTool(
      { store },
      { query: 'answer(a)', proofLimit: 2 },
    );

    expect(primary.rows[0]).not.toHaveProperty('alternativeProofs');
    expect(expanded.rows[0].proofs[0]).toMatchObject({ rule: 1 });
    expect(expanded.rows[0].alternativeProofs).toEqual([
      [expect.objectContaining({ rule: 2 })],
    ]);
    expect(expanded.graph.nodes.some((node) => node.kind === 'proof')).toBe(
      true,
    );
  });

  it('check_integrity returns proof-bearing violations without mutating memory', () => {
    store.assert(
      'default',
      'active(mira). suspended(mira). :- active(X), suspended(X).',
      { opId: 'integrity-input' },
    );
    const before = store.load('default');

    const result = checkIntegrityTool({ store }, { maxViolations: 10 });

    expect(result).toMatchObject({
      status: 'violations',
      constraintCount: 1,
      violationCount: 1,
      checks: [{ rows: [{ bindings: { X: 'mira' } }] }],
    });
    expect(store.load('default')).toEqual(before);
  });

  it('recall_explain keeps the answer and adds deterministic evidence', async () => {
    store.assert('default', 'pet(rahul, luna).', {
      opId: 'pet-source',
      sourceText: 'My cat is called Luna.',
    });
    const llm = new ScriptedLlm(['?- pet(rahul, Name).', 'Your cat is Luna.']);
    const result = await recallExplainTool(
      { store, llm },
      { question: 'What is my cat called?' },
    );
    expect(result.answer).toBe('Your cat is Luna.');
    expect(result.bindings).toEqual([{ Name: 'luna' }]);
    expect(result.explanation?.rows[0].proofs[0]).toMatchObject({
      predicate: 'pet',
      sources: [{ opId: 'pet-source', text: 'My cat is called Luna.' }],
    });
  });

  it('recall_explain threads the proof limit through generated-query evaluation', async () => {
    store.assert(
      'default',
      'left(a). right(a). answer(X) :- left(X). answer(X) :- right(X).',
    );
    const llm = new ScriptedLlm([
      '?- answer(a).',
      'The answer is supported twice.',
    ]);

    const result = await recallExplainTool(
      { store, llm },
      { question: 'Is a an answer?', proofLimit: 2 },
    );

    expect(result.explanation?.rows[0].proofs[0]).toMatchObject({ rule: 1 });
    expect(result.explanation?.rows[0].alternativeProofs).toEqual([
      [expect.objectContaining({ rule: 2 })],
    ]);
  });

  it('query can span all namespaces with *', () => {
    store.assert('work', 'works_at(rahul, acme).');
    store.assert('home', 'lives_in(rahul, sydney).');
    const result = queryTool(
      { store },
      { query: 'works_at(P, _), lives_in(P, C)', namespaces: '*' },
    );
    expect(result.bindings).toEqual([{ P: 'rahul', C: 'sydney' }]);
  });

  it('forget retracts matching facts', () => {
    store.assert('default', 'f(a). f(b).');
    const result = forgetTool({ store }, { pattern: 'f(_)' });
    expect(result.removed).toBe(2);
    expect(result.opId).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('list_memories groups clauses by predicate', () => {
    store.assert('default', 'f(a). f(b). g(X) :- f(X).');
    const result = listMemoriesTool({ store }, {});
    expect(result.predicates).toEqual([
      { predicate: 'f/1', facts: ['f(a).', 'f(b).'] },
      { predicate: 'g/1', facts: [], rules: ['g(X) :- f(X).'] },
    ]);
  });

  it('list_memories exposes integrity constraints separately from predicates', () => {
    store.assert('default', 'active(mira). :- active(X), suspended(X).');
    expect(listMemoriesTool({ store }, {})).toEqual({
      predicates: [{ predicate: 'active/1', facts: ['active(mira).'] }],
      constraints: [':- active(X), suspended(X).'],
    });
  });

  it('list_memories filters by predicate name', () => {
    store.assert(
      'default',
      'f(a). g(b). :- f(X), blocked(X). :- g(X), hidden(X).',
    );
    const result = listMemoriesTool({ store }, { predicate: 'f' });
    expect(result).toEqual({
      predicates: [{ predicate: 'f/1', facts: ['f(a).'] }],
      constraints: [':- f(X), blocked(X).'],
    });
  });
});

describe('forget_sessions', () => {
  const session = {
    version: 1,
    source: 'import',
    sourceSessionId: 'past-chat',
    startedAt: '2026-09-18T00:00:00.000Z',
  } as const;

  const seededSessions = (): { sessions: SessionStore; key: string } => {
    const sessions = new SessionStore({
      root: mkdtempSync(join(tmpdir(), 'rembero-tools-sessions-')),
      capBytes: 1024 * 1024,
    });
    sessions.appendTurns('scratch', session, [
      { role: 'user', ts: '2026-09-18T00:00:00.000Z', text: 'I moved to Melbourne.' },
    ]);
    sessions.appendTurns(
      'scratch',
      { ...session, sourceSessionId: 'other-chat' },
      [{ role: 'user', ts: '2026-09-18T00:01:00.000Z', text: 'I ride a road bike.' }],
    );
    return { sessions, key: sessions.sessionKey('import', 'past-chat') };
  };

  it('deletes one session by key', () => {
    const { sessions, key } = seededSessions();

    expect(forgetSessionsTool({ sessions }, { namespace: 'scratch', key })).toEqual({
      deleted: 1,
    });
    expect(sessions.readSession('scratch', key)).toBeUndefined();
    expect(sessions.list('scratch')).toHaveLength(1);
  });

  it('reports nothing deleted for a key it does not hold', () => {
    const { sessions } = seededSessions();

    expect(
      forgetSessionsTool(
        { sessions },
        { namespace: 'scratch', key: 'f'.repeat(32) },
      ),
    ).toEqual({ deleted: 0 });
    expect(sessions.list('scratch')).toHaveLength(2);
  });

  it('deletes every session in the namespace with all', () => {
    const { sessions } = seededSessions();

    expect(
      forgetSessionsTool({ sessions }, { namespace: 'scratch', all: true }),
    ).toEqual({ deleted: 2 });
    expect(sessions.list('scratch')).toEqual([]);
  });

  it('refuses a call that names neither a key nor all', () => {
    const { sessions } = seededSessions();

    expect(() => forgetSessionsTool({ sessions }, { namespace: 'scratch' })).toThrow(
      /key/,
    );
    expect(sessions.list('scratch')).toHaveLength(2);
  });

  it('refuses a call that names both a key and all', () => {
    const { sessions, key } = seededSessions();

    expect(() =>
      forgetSessionsTool({ sessions }, { namespace: 'scratch', key, all: true }),
    ).toThrow(/both/);
    expect(sessions.list('scratch')).toHaveLength(2);
  });

  it('names the setting when no session store is configured', () => {
    expect(() => forgetSessionsTool({}, { all: true })).toThrow(/REMBERO_SESSIONS/);
  });
});
