import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { explainQueryTool, queryTool } from '../src/mcp/tools.js';
import { MemoryStore } from '../src/store/store.js';

function storeWithChain(): MemoryStore {
  const store = new MemoryStore(
    mkdtempSync(join(tmpdir(), 'rembero-query-surface-')),
  );
  store.assert(
    'default',
    `reports_to(maya, liam).
     reports_to(liam, ava).
     reports_to(ava, dana).
     waits_on(atlas, vendor).
     waits_on(vendor, legal).
     waits_on(legal, freeze).`,
    { opId: 'seed' },
  );
  return store;
}

describe('query surfaces share one normalizer', () => {
  it('MCP query accepts a rule program and answers the sink rule', () => {
    const store = storeWithChain();
    const result = queryTool(
      { store },
      {
        query: `reach(X) :- waits_on(atlas, X).
                reach(X) :- reach(M), waits_on(M, X).
                root(R) :- reach(R), \\+ waits_on(R, _).`,
        namespaces: ['default'],
      },
    );
    expect(result.bindings).toEqual([{ R: 'freeze' }]);
  });

  it('MCP query answers a ground goal with one boolean row instead of [{}] or []', () => {
    const store = storeWithChain();
    expect(
      queryTool(
        { store },
        { query: 'reports_to_plus(maya, dana).', namespaces: ['default'] },
      ).bindings,
    ).toEqual([{ yes: 'true' }]);
    expect(
      queryTool(
        { store },
        { query: '?- reports_to_plus(dana, maya).', namespaces: ['default'] },
      ).bindings,
    ).toEqual([{ yes: 'false' }]);
  });

  it('MCP explain_query accepts the same rule-program form', () => {
    const store = storeWithChain();
    const result = explainQueryTool(
      { store },
      {
        query: 'above(M) :- reports_to_plus(maya, M).',
        namespaces: ['default'],
      },
    );
    expect(result.rows.map((row) => row.bindings)).toEqual([
      { M: 'liam' },
      { M: 'ava' },
      { M: 'dana' },
    ]);
    // the authored rule is in the catalog so proofs resolve
    expect(result.rules.map((rule) => rule.clause)).toContain(
      'above(M) :- reports_to_plus(maya, M).',
    );
  });
});

describe('query surfaces report diagnostics instead of silent empties', () => {
  it('MCP query throws an actionable error for an unknown predicate or a capitalized constant', () => {
    const store = storeWithChain();
    const unknown = queryTool(
      { store },
      { query: 'report_to(maya, M)', namespaces: ['default'] },
    );
    expect(unknown.bindings).toEqual([]);
    expect(unknown.diagnostics?.[0].message).toMatch(
      /unknown predicate report_to\/2.*reports_to\/2/s,
    );
    expect(() =>
      queryTool(
        { store },
        { query: 'reports_to(Maya, M)', namespaces: ['default'] },
      ),
    ).toThrow(/'Maya' is a variable.*reports_to\(maya, M\)/s);
    expect(() =>
      queryTool(
        { store },
        { query: 'reports_to_plus_plus(maya, M)', namespaces: ['default'] },
      ),
    ).toThrow(/suffix appears twice/);
  });

  it('MCP query attaches a direction hint when a reversed query comes back empty', () => {
    const store = storeWithChain();
    const result = queryTool(
      { store },
      { query: 'reports_to_plus(dana, M)', namespaces: ['default'] },
    );
    expect(result.bindings).toEqual([]);
    expect(result.diagnostics?.[0]).toMatchObject({ code: 'direction' });
    expect(result.diagnostics?.[0].message).toMatch(
      /reports_to_plus\(M, dana\)/,
    );
    const fine = queryTool(
      { store },
      { query: 'reports_to_plus(maya, M)', namespaces: ['default'] },
    );
    expect(fine.diagnostics).toBeUndefined();
  });
});

describe('SQLite bridge reports the same diagnostics', () => {
  it('throws for a capitalized constant and a doubled closure suffix', async () => {
    const { openDatalogDatabase } = await import('../src/sqlite/extension.js');
    const database = await openDatalogDatabase(':memory:');
    try {
      database.exec(`
        CREATE TABLE reports_to(person TEXT, manager TEXT);
        INSERT INTO reports_to VALUES ('maya','liam'),('liam','ava');
      `);
      // portable path (negation routes there): data-dependent checks apply
      expect(() =>
        database.datalogQuery(
          'q(M) :- reports_to(Maya, M), \\+ reports_to(M, _).',
        ),
      ).toThrow(/'Maya' is a variable.*reports_to\(maya, M\)/s);
      // native path: the data-free closure checks still apply
      expect(() =>
        database.datalogQuery('q(M) :- reports_to_plus(maya, M, N).'),
      ).toThrow(/closure predicates are binary/);
      expect(() =>
        database.datalogQuery('q(M) :- reports_to_plus_plus(maya, M).'),
      ).toThrow(/suffix appears twice/);
      expect(
        database
          .datalogQuery('q(M) :- reports_to_plus(maya, M).')
          .map((r) => r.M)
          .sort(),
      ).toEqual(['ava', 'liam']);
    } finally {
      database.close();
    }
  });
});

describe('explain_query shares the diagnostics', () => {
  it('throws for a capitalized constant and attaches the unknown-predicate hint', () => {
    const store = storeWithChain();
    expect(() =>
      explainQueryTool(
        { store },
        { query: 'reports_to(Maya, M)', namespaces: ['default'] },
      ),
    ).toThrow(/'Maya' is a variable/);
    const unknown = explainQueryTool(
      { store },
      { query: 'report_to(maya, M)', namespaces: ['default'] },
    );
    expect(unknown.rows).toEqual([]);
    expect(unknown.diagnostics?.[0].message).toMatch(/reports_to\/2/);
  });
});
