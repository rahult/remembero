import { describe, expect, it } from 'vitest';
import {
  evaluate,
  evaluateWithProof,
  expandClosurePredicates,
  parseProgram,
  parseQuery,
  serializeClause,
  serializeTerm,
  type Bindings,
} from '../src/engine/index.js';

/** Render bindings as sorted "X=a Y=b" rows for order-independent assertions. */
function rows(bindings: Bindings[]): string[] {
  return bindings
    .map((binding) =>
      Object.entries(binding)
        .map(([name, term]) => `${name}=${serializeTerm(term)}`)
        .sort()
        .join(' '),
    )
    .sort();
}

const CHAIN = `
reports_to(maya, liam).
reports_to(liam, ava).
reports_to(ava, dana).
reports_to(tom, liam).
`;

describe('closure predicates: on-demand transitive closure', () => {
  it('answers p_plus as one-or-more hops over a binary predicate p', () => {
    const program = parseProgram(CHAIN);
    const result = evaluate(program, parseQuery('reports_to_plus(maya, M).'));
    expect(rows(result)).toEqual(['M=ava', 'M=dana', 'M=liam']);
  });

  it('answers the reverse direction: everyone below a node', () => {
    const program = parseProgram(CHAIN);
    const result = evaluate(program, parseQuery('reports_to_plus(P, dana).'));
    expect(rows(result)).toEqual(['P=ava', 'P=liam', 'P=maya', 'P=tom']);
  });

  it('terminates on a cyclic graph and includes the start node in its own closure', () => {
    const program = parseProgram(`
      edge(a, b).
      edge(b, c).
      edge(c, a).
    `);
    const result = evaluate(program, parseQuery('edge_plus(a, X).'));
    expect(rows(result)).toEqual(['X=a', 'X=b', 'X=c']);
  });

  it('lets a user-defined p_plus win over synthesis', () => {
    const program = parseProgram(`
      ${CHAIN}
      reports_to_plus(maya, someone_else).
    `);
    const result = evaluate(program, parseQuery('reports_to_plus(maya, M).'));
    expect(rows(result)).toEqual(['M=someone_else']);
  });

  it('leaves a non-binary _plus reference alone', () => {
    const program = parseProgram(`review_slot(atlas, monday, morning).`);
    const result = evaluate(
      program,
      parseQuery('review_slot_plus(atlas, D, W).'),
    );
    expect(rows(result)).toEqual([]);
  });

  it('composes with negation to find the root of a chain', () => {
    const program = parseProgram(`
      waits_on(atlas, vendor_security_review).
      waits_on(vendor_security_review, legal_signoff).
      waits_on(legal_signoff, procurement_freeze).
      root(R) :- waits_on_plus(atlas, R), \\+ waits_on(R, _).
    `);
    const result = evaluate(program, parseQuery('root(R).'));
    expect(rows(result)).toEqual(['R=procurement_freeze']);
  });

  it('shows the synthesized closure rule in the proof ladder', () => {
    const program = parseProgram(CHAIN);
    const explained = evaluateWithProof(
      program,
      parseQuery('reports_to_plus(maya, dana).'),
    );
    expect(explained).toHaveLength(1);
    const serialized = JSON.stringify(explained[0].proofs);
    expect(serialized).toContain('reports_to_plus');
    const synthesized = expandClosurePredicates(
      program,
      parseQuery('reports_to_plus(maya, dana).'),
    )
      .filter((clause) => clause.head.predicate === 'reports_to_plus')
      .map(serializeClause);
    expect(synthesized).toEqual([
      'reports_to_plus(X, Y) :- reports_to(X, Y).',
      'reports_to_plus(X, Y) :- reports_to(X, Z), reports_to_plus(Z, Y).',
    ]);
  });

  it('returns the same program array when nothing references a closure', () => {
    const program = parseProgram(CHAIN);
    expect(
      expandClosurePredicates(program, parseQuery('reports_to(maya, M).')),
    ).toBe(program);
  });
});

describe('closure predicates: knowledge explain catalog', () => {
  it('lists synthesized closure rules in the explain rule catalog so proofs resolve', async () => {
    const { explainKnowledge } = await import('../src/knowledge/graph.js');
    const result = explainKnowledge(
      parseProgram(CHAIN),
      'reports_to_plus(maya, dana)',
    );
    expect(result.rows).toHaveLength(1);
    const ruleNumbers = new Set<number>();
    const walk = (proof: { rule?: number; because?: unknown[] }) => {
      if (proof.rule !== undefined) ruleNumbers.add(proof.rule);
      for (const child of proof.because ?? [])
        walk(child as { rule?: number; because?: unknown[] });
    };
    for (const proof of result.rows[0].proofs) walk(proof);
    expect(ruleNumbers.size).toBeGreaterThan(0);
    const catalog = new Map(
      result.rules.map((rule) => [rule.number, rule.clause]),
    );
    for (const number of ruleNumbers) {
      expect(catalog.get(number)).toMatch(/^reports_to_plus\(X, Y\) :- /);
    }
  });
});

describe('closure predicates: SQLite bridge', () => {
  it('routes a single-rule _plus program to the portable engine and answers the chain', async () => {
    const { openDatalogDatabase, sqliteDatalogExecutionMode } =
      await import('../src/sqlite/extension.js');
    const program = 'above(M) :- reports_to_plus(maya, M).';
    expect(sqliteDatalogExecutionMode(program)).toBe('portable');
    const database = await openDatalogDatabase(':memory:');
    try {
      database.exec(`
        CREATE TABLE reports_to(person TEXT, manager TEXT);
        INSERT INTO reports_to VALUES ('maya','liam'),('liam','ava'),('ava','dana'),('tom','liam');
      `);
      const result = database
        .datalogQuery(program)
        .map((row) => row.M)
        .sort();
      expect(result).toEqual(['ava', 'dana', 'liam']);
      expect(() => database.datalogSql(program)).toThrow(
        /closure.*cannot be compiled to one SQLite SELECT/i,
      );
    } finally {
      database.close();
    }
  });

  it('still routes an authored p_plus rule program natively when nothing is synthesized', async () => {
    const { sqliteDatalogExecutionMode } =
      await import('../src/sqlite/extension.js');
    expect(
      sqliteDatalogExecutionMode('reports_to_plus(X, Y) :- reports_to(X, Y).'),
    ).toBe('native');
  });
});
