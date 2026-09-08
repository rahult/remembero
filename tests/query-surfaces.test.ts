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
