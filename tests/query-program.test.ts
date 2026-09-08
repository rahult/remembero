import { describe, expect, it } from 'vitest';
import {
  evaluate,
  parseProgram,
  parseQueryProgram,
  serializeClause,
  serializeGoal,
} from '../src/engine/index.js';

const FACTS = parseProgram(`
  waits_on(atlas, vendor).
  waits_on(vendor, legal).
  waits_on(legal, freeze).
  reports_to(maya, liam).
  reports_to(liam, ava).
`);

function goalText(query: { kind: string; goals?: unknown[] }): string {
  return (query.goals as Parameters<typeof serializeGoal>[0][])
    .map(serializeGoal)
    .join(', ');
}

describe('parseQueryProgram: one normalizer for every query surface', () => {
  it('accepts a bare goal list with or without a trailing period', () => {
    for (const input of ['reports_to(maya, M)', 'reports_to(maya, M).']) {
      const qp = parseQueryProgram(input);
      expect(qp.clauses).toEqual([]);
      expect(qp.target).toBe('goals');
      expect(qp.ground).toBe(false);
      expect(goalText(qp.query)).toBe('reports_to(maya, M)');
    }
  });

  it('accepts the ?- form and flags a ground query for a boolean answer', () => {
    const qp = parseQueryProgram('?- reports_to(maya, liam).');
    expect(qp.ground).toBe(true);
    expect(qp.target).toBe('goals');
    expect(parseQueryProgram('reports_to(maya, liam).').ground).toBe(true);
    expect(parseQueryProgram('reports_to(maya, _).').ground).toBe(false);
  });

  it('accepts aggregate query specs unchanged', () => {
    const qp = parseQueryProgram('count(*) as N where reports_to(X, Y)');
    expect(qp.query.kind).toBe('aggregate');
    expect(qp.ground).toBe(false);
  });

  it('picks the unique sink rule as the query target, not the first rule', () => {
    // The published few-shot shape: helper first, answer last.
    const qp = parseQueryProgram(`
      reach(X) :- waits_on(atlas, X).
      reach(X) :- reach(M), waits_on(M, X).
      root(R) :- reach(R), \\+ waits_on(R, _).
    `);
    expect(qp.target).toBe('sink');
    expect(goalText(qp.query)).toBe('root(R)');
    expect(qp.clauses.map(serializeClause)).toHaveLength(3);
    const rows = evaluate(
      [...FACTS, ...qp.clauses],
      qp.query.kind === 'relational' ? qp.query.goals : [],
    );
    expect(rows.map((r) => (r.R as { value: string }).value)).toEqual([
      'freeze',
    ]);
  });

  it('ignores self-references when finding the sink, so a recursive predicate is a target', () => {
    const qp = parseQueryProgram(`
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `);
    expect(qp.target).toBe('sink');
    expect(goalText(qp.query)).toBe('path(X, Y)');
  });

  it('treats a single rule as its own sink', () => {
    const qp = parseQueryProgram('above(M) :- reports_to_plus(maya, M).');
    expect(qp.target).toBe('sink');
    expect(goalText(qp.query)).toBe('above(M)');
  });

  it('errors when a program has more than one sink, naming them and the fix', () => {
    expect(() =>
      parseQueryProgram(`
        a(X) :- reports_to(X, _).
        b(Y) :- waits_on(Y, _).
      `),
    ).toThrow(/derives a\/1 and b\/1.*\?- a\(X\)\./s);
  });

  it('lets an explicit ?- line inside a program choose the target', () => {
    const qp = parseQueryProgram(`
      a(X) :- reports_to(X, _).
      b(Y) :- waits_on(Y, _).
      ?- b(Y).
    `);
    expect(qp.target).toBe('explicit');
    expect(goalText(qp.query)).toBe('b(Y)');
    expect(qp.clauses).toHaveLength(2);
  });

  it('rejects a program that is only facts or constraints as not a query', () => {
    expect(() => parseQueryProgram(':- reports_to(X, X).')).toThrow(
      /integrity constraint/i,
    );
  });
});
