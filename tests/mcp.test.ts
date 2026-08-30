import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { createServer } from '../src/mcp/server.js';
import { serializeClause } from '../src/engine/index.js';
import { MemoryStore } from '../src/store/store.js';

class ScriptedLlm implements LlmClient {
  calls = 0;
  constructor(private responses: string[]) {}
  async complete(_messages: ChatMessage[]): Promise<string> {
    this.calls += 1;
    const response = this.responses.shift();
    if (response === undefined) throw new Error('out of responses');
    return response;
  }
}

describe('MCP tool profiles', () => {
  it('exposes only the core memory surface under the core profile', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-profile-')));
    const server = createServer({
      store,
      llm: new ScriptedLlm([]),
      toolProfile: 'core',
    });
    const client = new Client({ name: 'rembero-profile-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name).sort()).toEqual(
        [
          'assert_facts',
          'check_integrity',
          'explain_query',
          'forget',
          'history',
          'list_memories',
          'query',
          'recall',
          'recall_explain',
          'remember',
          'search_knowledge',
          'supersede_facts',
        ]
      );
    } finally {
      await client.close();
      await server.close();
    }
  });
});

describe('MCP server default namespace', () => {
  it('routes namespace-less tool calls to the configured default namespace', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-ns-')));
    const server = createServer({
      store,
      llm: new ScriptedLlm([]),
      defaultNamespace: 'proj-atlas',
    });
    const client = new Client({ name: 'rembero-ns-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'city(sydney).' },
      });
      expect(store.load('proj-atlas').length).toBe(1);
      expect(store.load('default').length).toBe(0);

      const result = await client.callTool({
        name: 'query',
        arguments: { query: 'city(X)' },
      });
      const text = (result.content as { type: string; text: string }[])
        .map((block) => block.text)
        .join('\n');
      expect(text).toContain('sydney');
    } finally {
      await client.close();
      await server.close();
    }
  });
});

describe('MCP explanation surfaces', () => {
  it('registers and executes explain_query and recall_explain over the real protocol', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-')));
    store.assert('default', 'pet(rahul, luna).', {
      opId: 'mcp-source',
      sourceText: 'My cat is called Luna.',
    });
    store.assert('default', 'employee(alice). employee(bob). suspended(bob).');
    store.assert('default', ':- employee(X), suspended(X).', {
      opId: 'mcp-integrity-policy',
    });
    store.assert('default', 'score(alice, 20). score(bob, 14). baseline(team, 10).');
    store.assert(
      'default',
      'left(a). right(a). answer(X) :- left(X). answer(X) :- right(X).'
    );
    store.assert(
      'default',
      `edge(a, b). edge(b, c).
       reachable(X, Y) :- edge(X, Y).
       reachable(X, Y) :- edge(X, Z), reachable(Z, Y).`
    );
    store.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(works_at, 2, 0).
       works_at('Mira Patel', acme).`,
      { opId: 'mcp-identity-source' }
    );
    const server = createServer({
      store,
      llm: new ScriptedLlm(['?- pet(rahul, Name).', 'Your cat is Luna.']),
      embeddings: {
        model: 'test/embedding',
        async embed(inputs) {
          return {
            model: this.model,
            vectors: inputs.map(() => [1, 0]),
            usage: { promptTokens: inputs.length, totalTokens: inputs.length, costUsd: 0 },
          };
        },
      },
    });
    const client = new Client({ name: 'rembero-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      expect(client.getServerVersion()).toEqual({ name: 'rembero', version: '0.54.0' });
      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name)).toEqual(
        expect.arrayContaining([
          'explain_query',
          'recall_explain',
          'check_integrity',
          'conflict_views',
          'what_if',
          'apply_rule_change',
          'why_not',
          'knowledge_topology',
          'diff_recorded_knowledge',
          'plan_query_repair',
          'audit_rules',
          'search_knowledge',
          'semantic_search_knowledge',
          'prepare_semantic_search',
          'browse_knowledge_graph',
          'connect_knowledge_graph',
          'export_knowledge_bundle',
          'verify_knowledge_bundle',
          'run_knowledge_checks',
          'profile_query',
          'assert_tentative',
          'review_tentative',
          'resolve_tentative',
          'checkpoint_journal',
          'list_checkpoints',
          'history',
          'propose_memory',
          'apply_memory_proposal',
          'knowledge_health',
          'supersede_facts',
        ])
      );

      const health = await client.callTool({
        name: 'knowledge_health',
        arguments: { namespaces: ['default'] },
      });
      const healthText = health.content.find((item) => item.type === 'text');
      expect(JSON.parse(healthText?.type === 'text' ? healthText.text : '')).toMatchObject({
        status: 'violations',
        stateDigest: expect.stringMatching(/^[a-f0-9]{64}$/),
        integrity: { violationCount: 1 },
        provenance: { sourceCoveragePercent: 100 },
      });

      const simulated = await client.callTool({
        name: 'what_if',
        arguments: {
          query: 'employee(Person)',
          assume: 'employee(carol).',
          without: ['employee(bob)'],
        },
      });
      const simulatedText = simulated.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(simulatedText?.type === 'text' ? simulatedText.text : '')
      ).toMatchObject({
        changed: true,
        resultDelta: {
          added: [{ bindings: { Person: 'carol' } }],
          removed: [{ bindings: { Person: 'bob' } }],
        },
        integrityDelta: {
          baseline: { status: 'violations', violationCount: 1 },
          candidate: { status: 'consistent', violationCount: 0 },
          resolved: [{ bindings: { X: 'bob' } }],
        },
      });
      expect(store.clausesFor(['default']).some((clause) =>
        clause.head.args.some((term) => term.type === 'atom' && term.value === 'carol')
      )).toBe(false);

      const whyNot = await client.callTool({
        name: 'why_not',
        arguments: {
          query: 'employee(carol)',
          maxFailures: 8,
          maxCandidatesPerFailure: 2,
        },
      });
      const whyNotText = whyNot.content.find((item) => item.type === 'text');
      expect(JSON.parse(whyNotText?.type === 'text' ? whyNotText.text : '')).toMatchObject({
        status: 'blocked',
        failures: [
          {
            reason: 'missing_fact',
            goal: 'employee(carol)',
            nearby: [
              { fact: 'employee(alice).' },
              { fact: 'employee(bob).' },
            ],
          },
        ],
      });

      const topology = await client.callTool({
        name: 'knowledge_topology',
        arguments: { focus: 'answer', direction: 'upstream' },
      });
      const topologyText = topology.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(topologyText?.type === 'text' ? topologyText.text : '')
      ).toMatchObject({
        predicateCount: 3,
        ruleCount: 2,
        predicates: [{ key: 'answer/1' }, { key: 'left/1' }, { key: 'right/1' }],
        selection: { focus: 'answer/1', direction: 'upstream' },
      });

      const diff = await client.callTool({
        name: 'diff_recorded_knowledge',
        arguments: {
          fromSequence: 0,
          toSequence: 1,
          query: 'pet(rahul, Name)',
        },
      });
      const diffText = diff.content.find((item) => item.type === 'text');
      expect(JSON.parse(diffText?.type === 'text' ? diffText.text : '')).toMatchObject({
        changed: true,
        clauses: {
          added: [{ kind: 'fact', clause: 'pet(rahul, luna).' }],
        },
        queryImpact: {
          added: [{ bindings: { Name: 'luna' } }],
        },
      });

      const simulatedRule = await client.callTool({
        name: 'what_if',
        arguments: {
          query: 'connected(a, Y)',
          assumeRules: 'connected(X, Y) :- edge(X, Y).',
          checkSuite: JSON.stringify({
            version: 1,
            coverage: { minimumPercent: 1 },
            checks: [
              {
                name: 'direct connection',
                query: 'connected(a, b)',
                expect: { kind: 'nonempty' },
              },
            ],
          }),
        },
      });
      const simulatedRuleText = simulatedRule.content.find(
        (item) => item.type === 'text'
      );
      const simulatedRulePayload = JSON.parse(
        simulatedRuleText?.type === 'text' ? simulatedRuleText.text : ''
      );
      expect(simulatedRulePayload).toMatchObject({
        changed: true,
        application: {
          assumedRules: ['connected(X, Y) :- edge(X, Y).'],
        },
        candidate: { rows: [{ bindings: { Y: 'b' } }] },
        ruleAuditDelta: {
          candidate: { topology: { rules: expect.any(Array) } },
        },
        checkDelta: {
          baseline: { status: 'failed' },
          candidate: { status: 'passed' },
          fixed: ['direct connection'],
        },
      });

      const appliedRule = await client.callTool({
        name: 'apply_rule_change',
        arguments: {
          proposal: JSON.stringify(simulatedRulePayload.ruleProposal),
          opId: 'mcp-reviewed-rule',
        },
      });
      const appliedRuleText = appliedRule.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(appliedRuleText?.type === 'text' ? appliedRuleText.text : '')
      ).toMatchObject({
        opId: 'mcp-reviewed-rule',
        added: [expect.any(Object)],
        removed: [],
        audit: { topology: { ruleCount: 5 } },
        checks: { status: 'passed' },
      });

      const repairs = await client.callTool({
        name: 'plan_query_repair',
        arguments: {
          query: 'answer(c)',
          maxPlans: 4,
          maxSteps: 2,
        },
      });
      const repairsText = repairs.content.find((item) => item.type === 'text');
      const repairsPayload = JSON.parse(
        repairsText?.type === 'text' ? repairsText.text : ''
      );
      expect(repairsPayload).toMatchObject({
        status: 'repairable',
        plans: expect.arrayContaining([
          expect.objectContaining({ assume: ['left(c).'] }),
          expect.objectContaining({ assume: ['right(c).'] }),
        ]),
      });
      expect(repairsPayload.plans).toHaveLength(2);

      const audit = await client.callTool({
        name: 'audit_rules',
        arguments: { focus: 'answer', direction: 'upstream' },
      });
      const auditText = audit.content.find((item) => item.type === 'text');
      expect(JSON.parse(auditText?.type === 'text' ? auditText.text : '')).toMatchObject({
        status: 'clean',
        warningCount: 0,
        infoCount: 0,
        findings: [],
        topology: { selection: { focus: 'answer/1', direction: 'upstream' } },
      });

      const searched = await client.callTool({
        name: 'search_knowledge',
        arguments: { text: 'cat Luna', kinds: ['fact'], limit: 5 },
      });
      const searchedText = searched.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(searchedText?.type === 'text' ? searchedText.text : '')
      ).toMatchObject({
        status: 'matches',
        results: [
          {
            rank: 1,
            clause: 'pet(rahul, luna).',
            sources: [{ opId: 'mcp-source', text: 'My cat is called Luna.' }],
          },
        ],
      });

      const semantic = await client.callTool({
        name: 'semantic_search_knowledge',
        arguments: { text: 'recommend cat Luna advice', kinds: ['fact'], limit: 5 },
      });
      const semanticText = semantic.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(semanticText?.type === 'text' ? semanticText.text : '')
      ).toMatchObject({
        status: 'matches',
        route: 'semantic',
        model: 'test/embedding',
        cacheMisses: expect.any(Number),
        results: expect.arrayContaining([
          expect.objectContaining({ clause: 'pet(rahul, luna).' }),
        ]),
      });

      const prepared = await client.callTool({
        name: 'prepare_semantic_search',
        arguments: { namespaces: ['default'], kinds: ['fact'], limit: 2 },
      });
      const preparedText = prepared.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(preparedText?.type === 'text' ? preparedText.text : '')
      ).toMatchObject({
        status: expect.stringMatching(/complete|more/),
        selectedCount: 2,
        cacheHits: expect.any(Number),
        cacheMisses: expect.any(Number),
      });

      const browsed = await client.callTool({
        name: 'browse_knowledge_graph',
        arguments: { focus: 'rahul', depth: 1, maxClaims: 10 },
      });
      const browsedText = browsed.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(browsedText?.type === 'text' ? browsedText.text : '')
      ).toMatchObject({
        status: 'matches',
        selection: { focus: 'rahul', selectedClaims: 1 },
        graph: {
          nodes: expect.arrayContaining([
            expect.objectContaining({ kind: 'claim', predicate: 'pet' }),
          ]),
        },
      });
      const browsedPayload = JSON.parse(
        browsedText?.type === 'text' ? browsedText.text : ''
      );
      const petClaim = browsedPayload.graph.nodes.find(
        (node: { kind: string; predicate?: string }) =>
          node.kind === 'claim' && node.predicate === 'pet'
      );
      expect(petClaim.sources).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ opId: 'mcp-source' }),
        ])
      );

      const connected = await client.callTool({
        name: 'connect_knowledge_graph',
        arguments: { from: 'rahul', to: 'luna', maxDepth: 1 },
      });
      const connectedText = connected.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(connectedText?.type === 'text' ? connectedText.text : '')
      ).toMatchObject({
        status: 'connected',
        shortestHops: 1,
        paths: [
          {
            entities: ['rahul', 'luna'],
            segments: [
              {
                predicate: 'pet',
                from: 'rahul',
                to: 'luna',
                fromPosition: 0,
                toPosition: 1,
              },
            ],
          },
        ],
      });

      const derivedConnection = await client.callTool({
        name: 'connect_knowledge_graph',
        arguments: { from: 'a', to: 'c', maxDepth: 2, includeDerived: true },
      });
      const derivedConnectionText = derivedConnection.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          derivedConnectionText?.type === 'text' ? derivedConnectionText.text : ''
        )
      ).toMatchObject({
        status: 'connected',
        shortestHops: 1,
        includeDerived: true,
        paths: [
          {
            entities: ['a', 'c'],
            segments: [
              expect.objectContaining({
                predicate: 'reachable',
                derived: true,
                rule: 4,
              }),
            ],
          },
        ],
        claimProofs: [
          expect.objectContaining({
            derived: true,
            proof: expect.objectContaining({ predicate: 'reachable', rule: 4 }),
          }),
        ],
      });

      const exportedBundle = await client.callTool({
        name: 'export_knowledge_bundle',
        arguments: { namespaces: ['default'], recordedSequence: 1 },
      });
      const exportedBundleText = exportedBundle.content.find(
        (item) => item.type === 'text'
      );
      const bundleText =
        exportedBundleText?.type === 'text' ? exportedBundleText.text : '';
      expect(JSON.parse(bundleText)).toMatchObject({
        format: 'rembero-knowledge-bundle',
        view: { kind: 'recorded', sequence: 1 },
        namespaces: [
          {
            namespace: 'default',
            clauses: [{ clause: 'pet(rahul, luna).' }],
          },
        ],
      });
      const verifiedBundle = await client.callTool({
        name: 'verify_knowledge_bundle',
        arguments: { bundle: bundleText },
      });
      const verifiedBundleText = verifiedBundle.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          verifiedBundleText?.type === 'text' ? verifiedBundleText.text : ''
        )
      ).toMatchObject({ valid: true, clauseCount: 1, sourceCount: 1 });

      const suite = JSON.stringify({
        version: 1,
        checks: [
          {
            name: 'pet remembered',
            query: 'pet(rahul, Name)',
            expect: {
              kind: 'rows',
              order: 'exact',
              rows: [{ Name: 'luna' }],
            },
          },
          {
            name: 'no dragon',
            query: 'dragon(Name)',
            expect: { kind: 'empty' },
          },
        ],
      });
      const checked = await client.callTool({
        name: 'run_knowledge_checks',
        arguments: { suite, recordedSequence: 1 },
      });
      const checkedText = checked.content.find((item) => item.type === 'text');
      expect(JSON.parse(checkedText?.type === 'text' ? checkedText.text : '')).toMatchObject({
        status: 'passed',
        checkCount: 2,
        passedCount: 2,
        recordedSnapshot: { sequence: 1 },
      });

      const profiled = await client.callTool({
        name: 'profile_query',
        arguments: { query: 'answer(a)', compareFullScan: true },
      });
      const profiledText = profiled.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(profiledText?.type === 'text' ? profiledText.text : '')
      ).toMatchObject({
        equivalent: true,
        explanation: { rows: [{ bindings: {} }] },
        indexed: { candidateFactsVisited: expect.any(Number) },
        fullScan: { candidateFactsVisited: expect.any(Number) },
      });

      const asserted = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'retry_fact(alpha).', opId: 'mcp-assert-retry' },
      });
      const assertedText = asserted.content.find((item) => item.type === 'text');
      const assertedPayload = JSON.parse(
        assertedText?.type === 'text' ? assertedText.text : ''
      );
      expect(assertedPayload).toMatchObject({
        added: ['retry_fact(alpha).'],
        duplicates: 0,
        opId: 'mcp-assert-retry',
      });
      const replayed = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'retry_fact(alpha).', opId: 'mcp-assert-retry' },
      });
      const replayedText = replayed.content.find((item) => item.type === 'text');
      expect(JSON.parse(replayedText?.type === 'text' ? replayedText.text : '')).toEqual(
        assertedPayload
      );
      const conflict = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'retry_fact(beta).', opId: 'mcp-assert-retry' },
      });
      expect(conflict.isError).toBe(true);
      const conflictText = conflict.content.find((item) => item.type === 'text');
      expect(JSON.parse(conflictText?.type === 'text' ? conflictText.text : '')).toEqual({
        error: 'operation_conflict',
        message: "assert operation 'mcp-assert-retry' was already used for another mutation",
        operation: 'assert',
        namespace: 'default',
        opId: 'mcp-assert-retry',
      });

      const tentative = await client.callTool({
        name: 'assert_tentative',
        arguments: {
          clauses: 'tentative_note(atlas).',
          opId: 'mcp-tentative-note',
        },
      });
      const tentativeText = tentative.content.find((item) => item.type === 'text');
      expect(JSON.parse(tentativeText?.type === 'text' ? tentativeText.text : '')).toEqual({
        added: ['tentative_note(atlas).'],
        duplicates: 0,
        opId: 'mcp-tentative-note',
      });
      const hiddenTentative = await client.callTool({
        name: 'query',
        arguments: { query: 'tentative_note(atlas)' },
      });
      const hiddenTentativeText = hiddenTentative.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          hiddenTentativeText?.type === 'text' ? hiddenTentativeText.text : ''
        ).bindings
      ).toEqual([]);
      const includedTentative = await client.callTool({
        name: 'explain_query',
        arguments: {
          query: 'tentative_note(atlas)',
          trustMode: 'include_tentative',
        },
      });
      const includedTentativeText = includedTentative.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          includedTentativeText?.type === 'text' ? includedTentativeText.text : ''
        )
      ).toMatchObject({
        trustMode: 'include_tentative',
        rows: [{ proofs: [{ trust: 'tentative' }] }],
      });
      const reviewedTentative = await client.callTool({
        name: 'review_tentative',
        arguments: {},
      });
      const reviewedTentativeText = reviewedTentative.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          reviewedTentativeText?.type === 'text' ? reviewedTentativeText.text : ''
        )
      ).toMatchObject({ count: 1, claims: [{ clause: 'tentative_note(atlas).' }] });
      const resolvedTentative = await client.callTool({
        name: 'resolve_tentative',
        arguments: {
          clauses: 'tentative_note(atlas).',
          action: 'accept',
          opId: 'mcp-accept-note',
        },
      });
      const resolvedTentativeText = resolvedTentative.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          resolvedTentativeText?.type === 'text' ? resolvedTentativeText.text : ''
        )
      ).toMatchObject({ action: 'accept', resolved: 1, added: ['tentative_note(atlas).'] });

      const checkpointed = await client.callTool({
        name: 'checkpoint_journal',
        arguments: {
          opId: 'mcp-checkpoint',
          at: '2026-08-17T02:00:00.000Z',
        },
      });
      const checkpointedText = checkpointed.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          checkpointedText?.type === 'text' ? checkpointedText.text : ''
        )
      ).toMatchObject({ rotated: true, segmentCount: 1 });
      const checkpoints = await client.callTool({
        name: 'list_checkpoints',
        arguments: {},
      });
      const checkpointsText = checkpoints.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(checkpointsText?.type === 'text' ? checkpointsText.text : '')
      ).toMatchObject({ count: 1, checkpoints: [{ opId: 'mcp-checkpoint' }] });

      await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'status(mira, active).', opId: 'mcp-status-source' },
      });
      const superseded = await client.callTool({
        name: 'supersede_facts',
        arguments: {
          patterns: ['status(mira, _)'],
          replacements: 'status(mira, paused).',
          at: '2026-08-16T16:59:00.000Z',
          opId: 'mcp-status-correction',
        },
      });
      const supersededText = superseded.content.find((item) => item.type === 'text');
      const supersededPayload = JSON.parse(
        supersededText?.type === 'text' ? supersededText.text : ''
      );
      expect(supersededPayload).toEqual({
        added: ['status(mira, paused).'],
        duplicates: 0,
        retracted: 1,
        archived: [
          "status_until(mira, active, '2026-08-16T16:59:00.000Z').",
        ],
        opId: 'mcp-status-correction',
      });
      const supersededReplay = await client.callTool({
        name: 'supersede_facts',
        arguments: {
          patterns: ['status(mira, _)'],
          replacements: 'status(mira, paused).',
          at: '2026-08-16T16:59:00.000Z',
          opId: 'mcp-status-correction',
        },
      });
      const supersededReplayText = supersededReplay.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          supersededReplayText?.type === 'text' ? supersededReplayText.text : ''
        )
      ).toEqual(supersededPayload);
      const supersedeConflict = await client.callTool({
        name: 'supersede_facts',
        arguments: {
          patterns: ['status(mira, _)'],
          replacements: 'status(mira, away).',
          at: '2026-08-16T16:59:00.000Z',
          opId: 'mcp-status-correction',
        },
      });
      expect(supersedeConflict.isError).toBe(true);
      const supersedeConflictText = supersedeConflict.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          supersedeConflictText?.type === 'text' ? supersedeConflictText.text : ''
        )
      ).toMatchObject({
        error: 'operation_conflict',
        operation: 'supersede',
        namespace: 'default',
        opId: 'mcp-status-correction',
      });
      expect(store.load('default').map(({ head }) => head.predicate)).toEqual(
        expect.arrayContaining(['status', 'status_until'])
      );
      await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'temporary_assignment(mira, atlas).' },
      });
      const ended = await client.callTool({
        name: 'supersede_facts',
        arguments: {
          patterns: ['temporary_assignment(mira, _)'],
          at: '2026-08-17T00:00:00.000Z',
          opId: 'mcp-assignment-ended',
        },
      });
      const endedText = ended.content.find((item) => item.type === 'text');
      expect(JSON.parse(endedText?.type === 'text' ? endedText.text : '')).toEqual({
        added: [],
        duplicates: 0,
        retracted: 1,
        archived: [
          "temporary_assignment_until(mira, atlas, '2026-08-17T00:00:00.000Z').",
        ],
        opId: 'mcp-assignment-ended',
      });

      const forgotten = await client.callTool({
        name: 'forget',
        arguments: { pattern: 'retry_fact(_)', opId: 'mcp-forget-retry' },
      });
      const forgottenText = forgotten.content.find((item) => item.type === 'text');
      const forgottenPayload = JSON.parse(
        forgottenText?.type === 'text' ? forgottenText.text : ''
      );
      expect(forgottenPayload).toEqual({ removed: 1, opId: 'mcp-forget-retry' });
      const forgottenReplay = await client.callTool({
        name: 'forget',
        arguments: { pattern: 'retry_fact( _ )', opId: 'mcp-forget-retry' },
      });
      const forgottenReplayText = forgottenReplay.content.find(
        (item) => item.type === 'text'
      );
      expect(
        JSON.parse(
          forgottenReplayText?.type === 'text' ? forgottenReplayText.text : ''
        )
      ).toEqual(forgottenPayload);

      const explained = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'pet(rahul, Name)' },
      });
      const explainText = explained.content.find((item) => item.type === 'text');
      expect(explainText?.type).toBe('text');
      const explainPayload = JSON.parse(explainText?.type === 'text' ? explainText.text : '');
      expect(explainPayload.rows[0].proofs[0]).toMatchObject({
        predicate: 'pet',
        sources: [{ opId: 'mcp-source' }],
      });

      const alternatives = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'answer(a)', proofLimit: 2 },
      });
      const alternativesText = alternatives.content.find((item) => item.type === 'text');
      const alternativesPayload = JSON.parse(
        alternativesText?.type === 'text' ? alternativesText.text : ''
      );
      expect(alternativesPayload.rows[0]).toMatchObject({
        proofs: [expect.objectContaining({ rule: 1 })],
        alternativeProofs: [[expect.objectContaining({ rule: 2 })]],
      });
      expect(alternativesPayload.graph.nodes).toEqual(
        expect.arrayContaining([expect.objectContaining({ kind: 'proof' })])
      );

      const integrity = await client.callTool({
        name: 'check_integrity',
        arguments: {
          maxViolations: 10,
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      const integrityText = integrity.content.find((item) => item.type === 'text');
      const integrityPayload = JSON.parse(
        integrityText?.type === 'text' ? integrityText.text : ''
      );
      expect(integrityPayload).toMatchObject({
        status: 'violations',
        constraintCount: 1,
        violationCount: 1,
        checks: [
          {
            sources: [{ opId: 'mcp-integrity-policy' }],
            rows: [{ bindings: { X: 'bob' } }],
            graphSelection: { selector: { kind: 'result', row: 1 } },
          },
        ],
      });

      const conflicts = await client.callTool({
        name: 'conflict_views',
        arguments: {
          focus: 'bob',
          maxViolations: 10,
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      const conflictsText = conflicts.content.find((item) => item.type === 'text');
      const conflictsPayload = JSON.parse(
        conflictsText?.type === 'text' ? conflictsText.text : ''
      );
      expect(conflictsPayload).toMatchObject({
        status: 'violations',
        focus: 'bob',
        matchingViolationCount: 1,
        clusterCount: 1,
        clusters: [
          {
            focus: 'bob',
            rows: [{ focusBinding: 'X', bindings: { X: 'bob' } }],
            graphSelection: { selector: { kind: 'result', row: 1 } },
          },
        ],
      });

      const negated = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'employee(X), \\+ suspended(X)' },
      });
      const negatedText = negated.content.find((item) => item.type === 'text');
      const negatedPayload = JSON.parse(
        negatedText?.type === 'text' ? negatedText.text : ''
      );
      expect(negatedPayload.rows).toEqual([
        expect.objectContaining({ bindings: { X: 'alice' } }),
      ]);
      expect(negatedPayload.graph.nodes).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ kind: 'absence', predicate: 'suspended' }),
        ])
      );

      const aggregated = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'count(*) as Count where employee(Person)' },
      });
      const aggregatedText = aggregated.content.find((item) => item.type === 'text');
      const aggregatedPayload = JSON.parse(
        aggregatedText?.type === 'text' ? aggregatedText.text : ''
      );
      expect(aggregatedPayload.rows).toEqual([
        expect.objectContaining({ bindings: { Count: '2' } }),
      ]);
      expect(aggregatedPayload.graph.nodes).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ kind: 'aggregate', op: 'count', value: 2 }),
        ])
      );

      store.assert(
        'default',
        'employee_count(Count) :- count(*) as Count where employee(Person).'
      );
      const aggregateRule = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'employee_count(Count)' },
      });
      const aggregateRuleText = aggregateRule.content.find(
        (item) => item.type === 'text'
      );
      const aggregateRulePayload = JSON.parse(
        aggregateRuleText?.type === 'text' ? aggregateRuleText.text : ''
      );
      expect(aggregateRulePayload).toMatchObject({
        rows: [
          {
            bindings: { Count: '2' },
            proofs: [
              {
                predicate: 'employee_count',
                aggregate: {
                  aggregated: true,
                  op: 'count',
                  value: 2,
                  contributors: [{ bindings: { Person: 'alice' } }, { bindings: { Person: 'bob' } }],
                },
              },
            ],
          },
        ],
        graph: { nodes: expect.arrayContaining([expect.objectContaining({ kind: 'aggregate' })]) },
      });

      const arithmetic = await client.callTool({
        name: 'query',
        arguments: {
          query: 'score(Person, Points), baseline(team, Base), Points > Base + 5',
        },
      });
      const arithmeticText = arithmetic.content.find((item) => item.type === 'text');
      const arithmeticPayload = JSON.parse(
        arithmeticText?.type === 'text' ? arithmeticText.text : ''
      );
      expect(arithmeticPayload.bindings).toEqual([
        { Person: 'alice', Points: '20', Base: '10' },
      ]);

      const identity = await client.callTool({
        name: 'explain_query',
        arguments: {
          query: 'works_at(mira, Company)',
          entityIdentity: 'canonical',
        },
      });
      const identityText = identity.content.find((item) => item.type === 'text');
      const identityPayload = JSON.parse(
        identityText?.type === 'text' ? identityText.text : ''
      );
      expect(identityPayload.rows[0]).toMatchObject({
        bindings: { Company: 'acme' },
        proofs: [
          {
            sources: [
              expect.objectContaining({
                projectedFrom: "works_at('Mira Patel', acme).",
                identityRewrites: [
                  expect.objectContaining({ original: 'Mira Patel', canonical: 'mira' }),
                ],
              }),
            ],
          },
        ],
      });

      const recalled = await client.callTool({
        name: 'recall_explain',
        arguments: {
          question: 'What is my cat called?',
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      const recallText = recalled.content.find((item) => item.type === 'text');
      const recallPayload = JSON.parse(recallText?.type === 'text' ? recallText.text : '');
      expect(recallPayload).toMatchObject({
        answer: 'Your cat is Luna.',
        bindings: [{ Name: 'luna' }],
        explanation: {
          rows: [{ bindings: { Name: 'luna' } }],
          graphSelection: { selector: { kind: 'result', row: 1 } },
        },
      });

      store.assert('default', 'works_at(mira, acme).', {
        opId: 'history-1',
        sourceText: 'Mira works at Acme.',
        at: new Date('2026-08-10T09:00:00.000Z'),
      });
      (
        store as MemoryStore & {
          supersede: (
            namespace: string,
            patterns: string[],
            replacements: string,
            context?: Record<string, unknown>
          ) => unknown;
        }
      ).supersede('default', ['works_at(mira, _)'], 'works_at(mira, initech).', {
        opId: 'history-2',
        sourceText: 'Mira now works at Initech.',
        at: new Date('2026-08-16T16:59:00.000Z'),
      });
      const historical = await client.callTool({
        name: 'history',
        arguments: { pattern: 'works_at(mira, _)', namespaces: ['default'] },
      });
      const historyText = historical.content.find((item) => item.type === 'text');
      const historyPayload = JSON.parse(historyText?.type === 'text' ? historyText.text : '');
      expect(historyPayload).toMatchObject({
        pattern: 'works_at(mira, _)',
        namespaces: ['default'],
        events: [
          expect.objectContaining({
            action: 'asserted',
            clause: 'works_at(mira, acme).',
          }),
          expect.objectContaining({
            action: 'superseded',
            clause: 'works_at(mira, acme).',
            archivedAs: "works_at_until(mira, acme, '2026-08-16T16:59:00.000Z').",
          }),
          expect.objectContaining({
            action: 'asserted',
            clause: 'works_at(mira, initech).',
            current: true,
          }),
        ],
      });
      const recorded = await client.callTool({
        name: 'query',
        arguments: {
          query: 'works_at(mira, Company)',
          recordedSequence: historyPayload.events[0].sequence,
        },
      });
      const recordedText = recorded.content.find((item) => item.type === 'text');
      const recordedPayload = JSON.parse(
        recordedText?.type === 'text' ? recordedText.text : ''
      );
      expect(recordedPayload).toMatchObject({
        bindings: [{ Company: 'acme' }],
        recordedSnapshot: { sequence: historyPayload.events[0].sequence },
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('combines recorded snapshots, canonical focus, and graph selection for conflicts', async () => {
    const store = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-mcp-recorded-conflicts-'))
    );
    store.assert(
      'default',
      `rembero_alias('Mira Patel', mira).
       rembero_entity_position(active, 1, 0).
       active('Mira Patel').
       :- active(Person), suspended(Person).`,
      { opId: 'recorded-conflict-baseline' }
    );
    store.assert('default', 'suspended(mira).', {
      opId: 'recorded-conflict-later',
    });
    const server = createServer({
      store,
      llm: new ScriptedLlm([]),
    });
    const client = new Client({ name: 'rembero-recorded-conflict-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const response = await client.callTool({
        name: 'conflict_views',
        arguments: {
          focus: "'Mira Patel'",
          entityIdentity: 'canonical',
          recordedSequence: 2,
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      const responseText = response.content.find((item) => item.type === 'text');
      const payload = JSON.parse(
        responseText?.type === 'text' ? responseText.text : ''
      );
      expect(payload).toMatchObject({
        status: 'violations',
        focus: 'mira',
        matchingViolationCount: 1,
        recordedSnapshot: { sequence: 2, journalEntries: 2 },
        clusters: [
          {
            focus: 'mira',
            graphSelection: { selector: { kind: 'result', row: 1 } },
            rows: [{ bindings: { Person: 'mira' } }],
          },
        ],
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('returns inspectable non-empty query reviews over the real recall protocol', async () => {
    const store = new MemoryStore(
      mkdtempSync(join(tmpdir(), 'rembero-mcp-recall-review-'))
    );
    store.assert(
      'default',
      'uses_language(atlas, rust). project_owner(atlas, rahul).'
    );
    const server = createServer({
      store,
      llm: new ScriptedLlm([
        '?- uses_language(atlas, Value).',
        '?- project_owner(atlas, Owner).',
        'Rahul owns Atlas.',
      ]),
    });
    const client = new Client({ name: 'rembero-recall-review-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const response = await client.callTool({
        name: 'recall',
        arguments: { question: 'Who owns Atlas?' },
      });
      const responseText = response.content.find((item) => item.type === 'text');
      const payload = JSON.parse(
        responseText?.type === 'text' ? responseText.text : ''
      );
      expect(payload).toMatchObject({
        status: 'answered',
        answer: 'Rahul owns Atlas.',
        query: 'project_owner(atlas, Owner)',
        bindings: [{ Owner: 'rahul' }],
        queryReviews: [
          {
            originalQuery: 'uses_language(atlas, Value)',
            reviewedQuery: 'project_owner(atlas, Owner)',
            reasons: ['competing_predicate'],
            competingPredicates: ['project_owner/2'],
            outcome: 'corrected',
          },
        ],
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('returns proof-bearing write rejection and preserves memory over the real protocol', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-enforce-')));
    store.assert(
      'default',
      'active(mira). :- active(Person), suspended(Person).'
    );
    const before = store.load('default');
    const server = createServer({
      store,
      llm: new ScriptedLlm([]),
      integrityEnforcement: { mode: 'strict' },
    });
    const client = new Client({ name: 'rembero-enforcement-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const rejected = await client.callTool({
        name: 'assert_facts',
        arguments: {
          clauses: 'suspended(mira).',
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      expect(rejected.isError).toBe(true);
      const text = rejected.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        error: 'integrity_violation',
        mode: 'strict',
        introducedViolationCount: 1,
        candidate: {
          checks: [
            {
              rows: [{ bindings: { Person: 'mira' } }],
              graphSelection: { selector: { kind: 'result', row: 1 } },
            },
          ],
        },
      });
      expect(store.load('default')).toEqual(before);

      const rejectedSupersede = await client.callTool({
        name: 'supersede_facts',
        arguments: {
          patterns: ['active(mira)'],
          replacements: 'active(mira). suspended(mira).',
          at: '2026-08-16T16:59:00.000Z',
          opId: 'rejected-temporal-correction',
          graphSelector: { kind: 'result', row: 1 },
        },
      });
      expect(rejectedSupersede.isError).toBe(true);
      const supersedeText = rejectedSupersede.content.find((item) => item.type === 'text');
      const supersedePayload = JSON.parse(
        supersedeText?.type === 'text' ? supersedeText.text : ''
      );
      expect(supersedePayload).toMatchObject({
        error: 'integrity_violation',
        mode: 'strict',
        introducedViolationCount: 1,
      });
      expect(store.load('default')).toEqual(before);

      const weakened = await client.callTool({
        name: 'assert_facts',
        arguments: {
          clauses: 'suspended(mira).',
          integrityMode: 'no_new_violations',
        },
      });
      expect(weakened.isError).toBe(true);
      const weakenedText = weakened.content.find((item) => item.type === 'text');
      expect(weakenedText?.type === 'text' ? weakenedText.text : '').toMatch(
        /cannot weaken strict server integrity enforcement/i
      );
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('applies the configured valid-time mode through the real remember tool', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-temporal-')));
    store.assert('default', 'works_at(mira, acme).', { opId: 'mcp-before' });
    const server = createServer({
      store,
      validTimeMode: 'archive_until',
      llm: new ScriptedLlm(['retract works_at(mira, _).\nworks_at(mira, initech).']),
    });
    const client = new Client({ name: 'rembero-temporal-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const remembered = await client.callTool({
        name: 'remember',
        arguments: { text: 'Mira now works at Initech' },
      });
      const text = remembered.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        retracted: 1,
        archived: [expect.stringMatching(/^works_at_until\(mira, acme, '/)],
      });
      expect(store.load('default').map((clause) => clause.head.predicate)).toEqual(
        expect.arrayContaining(['works_at', 'works_at_until'])
      );
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('applies a bounded schema slice through the real recall tool', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-schema-')));
    store.assert(
      'default',
      `${Array.from({ length: 40 }, (_, index) => `alpha_${index}(value_${index}).`).join('\n')}
       zeta_relation(target, answer).`
    );
    const server = createServer({
      store,
      llm: new ScriptedLlm([
        '?- zeta_relation(target, Value).',
        'The stored answer is answer.',
      ]),
    });
    const client = new Client({ name: 'rembero-schema-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const recalled = await client.callTool({
        name: 'recall',
        arguments: {
          question: 'Find the requested information',
          schemaPredicateLimit: 2,
        },
      });
      const text = recalled.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        status: 'answered',
        answer: 'The stored answer is answer.',
        query: 'zeta_relation(target, Value)',
        bindings: [{ Value: 'answer' }],
        pruning: {
          totalPredicates: 41,
          selectedPredicates: expect.not.arrayContaining(['zeta_relation/2']),
          catalogComplete: true,
        },
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('returns grounded negative recall without a phrasing model call', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-negative-')));
    store.assert('default', 'works_at(maya, acme).', { opId: 'maya-source' });
    const llm = new ScriptedLlm([
      '?- works_at(zoe, Company).',
      '?- works_at(zoe, Company).',
    ]);
    const server = createServer({ store, llm });
    const client = new Client({ name: 'rembero-negative-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const recalled = await client.callTool({
        name: 'recall',
        arguments: {
          question: 'Where does Zoe work?',
          relatedKnowledge: true,
          relatedLimit: 1,
          relatedKinds: ['fact'],
        },
      });
      const text = recalled.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        status: 'no_match',
        query: 'works_at(zoe, Company)',
        bindings: [],
        whyNot: {
          status: 'blocked',
          summary:
            'No stored result matches works_at(zoe, Company). Required fact works_at(zoe, Company) is missing.',
        },
        relatedKnowledge: {
          status: 'matches',
          limit: 1,
          results: [
            {
              clause: 'works_at(maya, acme).',
              sources: [{ opId: 'maya-source' }],
            },
          ],
        },
      });
      expect(payload.answer).toBe(payload.whyNot.summary);
      expect(llm.calls).toBe(2);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('renders positive recall locally when deterministic answer mode is requested', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-answer-mode-')));
    store.assert('default', 'works_at(maya, acme).', { opId: 'maya-source' });
    const llm = new ScriptedLlm([
      '?- works_at(maya, Company).',
      '?- works_at(maya, Company).',
    ]);
    const server = createServer({ store, llm });
    const client = new Client({ name: 'rembero-answer-mode-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const recalled = await client.callTool({
        name: 'recall',
        arguments: {
          question: 'Where does Maya work?',
          answerMode: 'deterministic',
        },
      });
      const text = recalled.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        status: 'answered',
        answerMode: 'deterministic',
        answer: 'Result for works_at(maya, Company): Company = acme.',
      });
      const evidenced = await client.callTool({
        name: 'recall',
        arguments: {
          question: 'Where does Maya work?',
          answerMode: 'evidence',
        },
      });
      const evidenceText = evidenced.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(evidenceText?.type === 'text' ? evidenceText.text : '')
      ).toMatchObject({
        status: 'answered',
        answerMode: 'evidence',
        answer: expect.stringContaining('Sources: default/maya-source@'),
        explanation: { rows: [{ bindings: { Company: 'acme' } }] },
      });
      expect(llm.calls).toBe(2);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('applies REMBERO_RECALL_ANSWER_MODE as the MCP server default', async () => {
    const previousMode = process.env.REMBERO_RECALL_ANSWER_MODE;
    process.env.REMBERO_RECALL_ANSWER_MODE = 'deterministic';
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-answer-env-')));
    store.assert('default', 'project(atlas).', { opId: 'project-source' });
    const llm = new ScriptedLlm(['?- project(atlas).']);
    const server = createServer({ store, llm });
    const client = new Client({ name: 'rembero-answer-env-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const recalled = await client.callTool({
        name: 'recall',
        arguments: { question: 'Is Atlas a project?' },
      });
      const text = recalled.content.find((item) => item.type === 'text');
      expect(JSON.parse(text?.type === 'text' ? text.text : '')).toMatchObject({
        answerMode: 'deterministic',
        answer: 'The query project(atlas) is supported.',
      });
      expect(llm.calls).toBe(1);
    } finally {
      await client.close();
      await server.close();
      if (previousMode === undefined) delete process.env.REMBERO_RECALL_ANSWER_MODE;
      else process.env.REMBERO_RECALL_ANSWER_MODE = previousMode;
    }
  });

  it('applies REMBERO_VALID_TIME_MODE through a programmatic MCP server', async () => {
    const previousMode = process.env.REMBERO_VALID_TIME_MODE;
    process.env.REMBERO_VALID_TIME_MODE = 'archive_until';
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-temporal-env-')));
    store.assert('default', 'works_at(mira, acme).', { opId: 'mcp-env-before' });
    const server = createServer({
      store,
      llm: new ScriptedLlm(['retract works_at(mira, _).\nworks_at(mira, initech).']),
    });
    const client = new Client({ name: 'rembero-temporal-env-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const remembered = await client.callTool({
        name: 'remember',
        arguments: { text: 'Mira now works at Initech' },
      });
      const text = remembered.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        retracted: 1,
        archived: [expect.stringMatching(/^works_at_until\(mira, acme, '/)],
      });
      expect(store.load('default').map((clause) => clause.head.predicate)).toEqual(
        expect.arrayContaining(['works_at', 'works_at_until'])
      );
    } finally {
      await client.close();
      await server.close();
      if (previousMode === undefined) delete process.env.REMBERO_VALID_TIME_MODE;
      else process.env.REMBERO_VALID_TIME_MODE = previousMode;
    }
  });

  it('proposes accepted memory changes over MCP without mutating', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-proposal-')));
    store.assert('default', 'works_at(mira, acme).', { opId: 'proposal-before' });
    const server = createServer({
      store,
      llm: new ScriptedLlm([
        'retract works_at(mira, _).\nworks_at(mira, initech).',
      ]),
    });
    const client = new Client({ name: 'rembero-proposal-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const proposed = await client.callTool({
        name: 'propose_memory',
        arguments: {
          text: 'Mira now works at Initech.',
          checkSuite: JSON.stringify({
            version: 1,
            checks: [
              {
                name: 'new employer',
                query: 'works_at(mira, initech)',
                expect: { kind: 'nonempty' },
              },
            ],
          }),
        },
      });
      const text = proposed.content.find((item) => item.type === 'text');
      const payload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(payload).toMatchObject({
        changed: true,
        proposal: {
          removeClauses: ['works_at(mira, acme).'],
          addClauses: ['works_at(mira, initech).'],
          checkSuite: expect.any(String),
        },
        checkDelta: { candidate: { status: 'passed' } },
      });
      expect(store.load('default').map(serializeClause)).toEqual([
        'works_at(mira, acme).',
      ]);
      const applied = await client.callTool({
        name: 'apply_memory_proposal',
        arguments: {
          proposal: JSON.stringify(payload.proposal),
          opId: 'mcp-reviewed-memory',
        },
      });
      const appliedText = applied.content.find((item) => item.type === 'text');
      expect(
        JSON.parse(appliedText?.type === 'text' ? appliedText.text : '')
      ).toMatchObject({
        opId: 'mcp-reviewed-memory',
        removed: [expect.any(Object)],
        added: [expect.any(Object)],
        checks: { status: 'passed' },
      });
      expect(store.load('default').map(serializeClause)).toEqual([
        'works_at(mira, initech).',
      ]);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('enforces a server-level knowledge suite across raw MCP writers', async () => {
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-mcp-check-guard-')));
    const server = createServer({
      store,
      llm: new ScriptedLlm([]),
      knowledgeCheckEnforcement: {
        mode: 'strict',
        namespaces: ['default'],
        suite: {
          version: 1,
          checks: [
            {
              name: 'forbidden stays absent',
              query: 'forbidden(a)',
              expect: { kind: 'empty' },
            },
          ],
        },
      },
    });
    const client = new Client({ name: 'rembero-check-guard-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const safe = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'safe(a).', opId: 'mcp-safe-check' },
      });
      expect(safe.isError).not.toBe(true);
      const blocked = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'forbidden(a).', opId: 'mcp-blocked-check' },
      });
      expect(blocked.isError).toBe(true);
      const text = blocked.content.find((item) => item.type === 'text');
      expect(JSON.parse(text?.type === 'text' ? text.text : '')).toMatchObject({
        error: 'knowledge_check_enforcement',
        mode: 'strict',
        candidate: { status: 'failed' },
      });
      expect(store.load('default').map(serializeClause)).toEqual(['safe(a).']);
    } finally {
      await client.close();
      await server.close();
    }
  });
});
