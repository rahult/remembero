import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { createServer } from '../src/mcp/server.js';
import { listMemoriesTool, lookupTool } from '../src/mcp/tools.js';
import { MemoryStore } from '../src/store/store.js';

class SilentLlm implements LlmClient {
  async complete(_messages: ChatMessage[]): Promise<string> {
    return '';
  }
}

function seeded(): MemoryStore {
  const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'rembero-core-')));
  store.assert(
    'default',
    `reports_to(maya, liam).
     reports_to(liam, ava).
     reports_to(ava, dana).
     works_on(maya, atlas).
     works_on(tom, atlas).
     prefers_meeting(maya, morning).
     rembero_arg_names(reports_to, person, manager).
     :- reports_to(X, X).`,
    { opId: 'seed' },
  );
  return store;
}

async function coreClient(store: MemoryStore) {
  const server = createServer({
    store,
    llm: new SilentLlm(),
    toolProfile: 'core',
  });
  const client = new Client({ name: 'core-test', version: '1.0.0' });
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);
  return { server, client };
}

describe('small-model core profile', () => {
  it('exposes at most eight tools, each with at most four parameters', async () => {
    const { server, client } = await coreClient(seeded());
    try {
      const { tools } = await client.listTools();
      expect(tools.map((t) => t.name).sort()).toEqual([
        'assert_facts',
        'forget',
        'list_memories',
        'lookup',
        'query',
        'recall',
        'remember',
        'supersede_facts',
      ]);
      for (const tool of tools) {
        const params = Object.keys(
          (tool.inputSchema as { properties?: Record<string, unknown> })
            .properties ?? {},
        );
        expect(
          params.length,
          `${tool.name}: ${params.join(', ')}`,
        ).toBeLessThanOrEqual(4);
      }
    } finally {
      await client.close();
      await server.close();
    }
  });

  it('list_memories defaults to a schema listing with counts, samples and argument names', () => {
    const result = listMemoriesTool(
      { store: seeded() },
      { namespaces: ['default'], mode: 'schema' },
    );
    const reports = result.schema?.find(
      (entry) => entry.predicate === 'reports_to/2',
    );
    expect(reports).toMatchObject({
      predicate: 'reports_to/2',
      arity: 2,
      count: 3,
      args: ['person', 'manager'],
    });
    expect(reports?.samples.length).toBeLessThanOrEqual(3);
    expect(
      result.schema?.some((entry) =>
        entry.predicate.startsWith('rembero_arg_names'),
      ),
    ).toBe(false);
    expect(result.constraints).toEqual([':- reports_to(X, X).']);
    // full mode is still available and unchanged
    const full = listMemoriesTool(
      { store: seeded() },
      { namespaces: ['default'], mode: 'full' },
    );
    expect(
      full.predicates.find((g) => g.predicate === 'reports_to/2')?.facts,
    ).toHaveLength(3);
  });

  it('lookup compiles subject/object slots into a query, with transitive closure on request', () => {
    const store = seeded();
    expect(
      lookupTool(
        { store },
        { predicate: 'reports_to', subject: 'maya', namespaces: ['default'] },
      ).bindings,
    ).toEqual([{ object: 'liam' }]);
    expect(
      lookupTool(
        { store },
        {
          predicate: 'reports_to',
          subject: 'maya',
          transitive: true,
          namespaces: ['default'],
        },
      ).bindings.map((b) => b.object),
    ).toEqual(['liam', 'ava', 'dana']);
    expect(
      lookupTool(
        { store },
        {
          predicate: 'reports_to',
          object: 'dana',
          transitive: true,
          namespaces: ['default'],
        },
      )
        .bindings.map((b) => b.subject)
        .sort(),
    ).toEqual(['ava', 'liam', 'maya']);
    expect(
      lookupTool(
        { store },
        {
          predicate: 'reports_to',
          subject: 'maya',
          object: 'dana',
          transitive: true,
          namespaces: ['default'],
        },
      ).bindings,
    ).toEqual([{ yes: 'true' }]);
    expect(() =>
      lookupTool(
        { store },
        { predicate: 'Reports To', namespaces: ['default'] },
      ),
    ).toThrow(/predicate must be lowercase/);
  });

  it('turns parser errors into hints a model can act on', async () => {
    const { server, client } = await coreClient(seeded());
    try {
      const quoted = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'works_at("Mira Chen", initech).' },
      });
      const quotedText = quoted.content.find((c) => c.type === 'text');
      expect(quotedText?.type === 'text' ? quotedText.text : '').toMatch(
        /single quotes.*'Mira Chen'/s,
      );
      const upper = await client.callTool({
        name: 'assert_facts',
        arguments: { clauses: 'works_at(Mira, initech).' },
      });
      const upperText = upper.content.find((c) => c.type === 'text');
      expect(upperText?.type === 'text' ? upperText.text : '').toMatch(
        /lowercase.*works_at\(mira, initech\)/s,
      );
    } finally {
      await client.close();
      await server.close();
    }
  });
});
