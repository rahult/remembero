import { describe, expect, it } from 'vitest';
import {
  diagnoseQuery,
  emptyResultFeedback,
  parseProgram,
  parseQueryProgram,
} from '../src/engine/index.js';

const FACTS = parseProgram(`
  status(atlas, blocked).
  status(beacon, active).
  reports_to(maya, liam).
  reports_to(liam, ava).
  waits_on(atlas, vendor).
  works_on(maya, atlas).
  works_on(liam, beacon).
  promised(liam, maya, atlas).
  promised(ava, liam, beacon).
  prefers_meeting(maya, morning).
  employed(rahul) :- works_at(rahul, _).
`);

function diagnose(query: string, emptyResult = true) {
  const program = parseQueryProgram(query);
  return diagnoseQuery([...FACTS, ...program.clauses], program.query, {
    emptyResult,
    authored: program.clauses,
  });
}

describe('diagnoseQuery: the model gets told what went wrong', () => {
  it('reports an unknown predicate with near-miss names as a warning on an empty result', () => {
    // a warning, not an error: a positive goal over an unknown predicate is an
    // honest closed-world "no", and negated unknowns are the normal way to say
    // "nothing is suspended"
    const [d] = diagnose('statuses(P, S)');
    expect(d.severity).toBe('warning');
    expect(d.code).toBe('unknown_predicate');
    expect(diagnose('reports_to(maya, M), \\+ suspended(M)')).toEqual([]);
    expect(diagnose('statuses(P, S)', false)).toEqual([]);
    expect(d.message).toMatch(/unknown predicate statuses\/2/);
    expect(d.message).toMatch(/status\/2/);
  });

  it('reports a wrong arity as unknown, naming the arity that exists', () => {
    const [d] = diagnose('status(P)');
    expect(d.code).toBe('unknown_predicate');
    expect(d.message).toMatch(/status\/1.*status\/2/s);
  });

  it('accepts a rule-derived predicate and a synthesizable closure as known', () => {
    expect(diagnose('employed(X)')).toEqual([]);
    expect(diagnose('reports_to_plus(maya, M)')).toEqual([]);
  });

  it('rejects a closure of the wrong arity and a doubled suffix as errors', () => {
    expect(diagnose('reports_to_plus(maya, M, N)')[0]).toMatchObject({
      severity: 'error',
      code: 'closure_arity',
    });
    expect(diagnose('reports_to_plus_plus(maya, M)')[0]).toMatchObject({
      severity: 'error',
      code: 'closure_double_suffix',
    });
    expect(diagnose('reports_to_plus_plus(maya, M)')[0].message).toMatch(
      /reports_to_plus\(/,
    );
  });

  it('flags a singleton uppercase variable that names a known constant in that slot', () => {
    const [d] = diagnose('reports_to(Maya, M)', false);
    expect(d).toMatchObject({
      severity: 'error',
      code: 'capitalized_constant',
    });
    expect(d.message).toMatch(/Maya.*variable.*maya.*reports_to\(maya, M\)/s);
    // a variable that appears twice is a join, not a typo
    expect(diagnose('reports_to(Maya, X), reports_to(X, Maya)', false)).toEqual(
      [],
    );
  });

  it('hints at argument direction when a constant never appears in that slot and the result is empty', () => {
    const [d] = diagnose('reports_to(liam, maya)');
    expect(d).toMatchObject({ severity: 'warning', code: 'direction' });
    expect(d.message).toMatch(/maya.*argument 2.*argument 1/s);
    expect(diagnose('reports_to(liam, maya)', false)).toEqual([]);
    // closure predicates are checked against their base relation
    const [c] = diagnose('waits_on_plus(vendor, atlas)');
    expect(c.code).toBe('direction');
  });

  it('checks literals inside authored rule bodies, scoping singleton variables per rule', () => {
    const [d] = diagnose('q(M) :- reports_to(Maya, M).', false);
    expect(d).toMatchObject({ code: 'capitalized_constant' });
    expect(diagnose('q(M) :- report_to(maya, M).')[0]).toMatchObject({
      code: 'unknown_predicate',
      severity: 'warning',
    });
    // M is used in head and body: not a singleton
    expect(diagnose('q(M) :- reports_to(maya, M).', false)).toEqual([]);
  });

  it('treats caller-supplied known relations (e.g. empty tables) as known', () => {
    const program = parseQueryProgram(
      'q(X) :- reports_to(X, _), \\+ suspended(X).',
    );
    expect(
      diagnoseQuery([...FACTS, ...program.clauses], program.query, {
        authored: program.clauses,
        knownPredicates: ['suspended/1'],
      }),
    ).toEqual([]);
  });

  it('rejects a query whose only variables are wildcards, since its rows carry no values', () => {
    const [d] = diagnose('waits_on(_, atlas)', false);
    expect(d).toMatchObject({ severity: 'error', code: 'wildcard_only' });
    expect(d.message).toMatch(/named variable.*waits_on\(X, atlas\)/s);
    // a fully ground goal is a yes/no question and is fine
    expect(diagnose('waits_on(vendor, atlas)', false)).toEqual([]);
  });

  it('stays quiet for a well-formed query with an honestly empty answer', () => {
    // both constants have been seen in exactly these positions; the answer is simply no
    expect(diagnose('status(atlas, active)')).toEqual([]);
    expect(
      diagnose('prefers_meeting(maya, W), status(beacon, blocked)'),
    ).toEqual([]);
  });
});

describe('emptyResultFeedback: why a join came back empty', () => {
  const feedback = (query: string) => {
    const program = parseQueryProgram(query);
    return emptyResultFeedback([...FACTS, ...program.clauses], program.query, {
      authored: program.clauses,
    });
  };

  it('reports how many rows each body goal matches on its own when the join is empty', () => {
    // reversed argument order in one goal: each goal matches alone, the join does not
    const text = feedback('status(P, blocked), reports_to(P, maya)');
    expect(text).toMatch(/status\(P, blocked\) alone matches 1 row/);
    expect(text).toMatch(/reports_to\(P, maya\) alone matches 0 rows/);
    expect(text).toMatch(/argument positions/);
  });

  it('folds the warning diagnostics in ahead of the per-goal counts', () => {
    const text = feedback('statuses(P, S)');
    expect(text).toMatch(/unknown predicate statuses\/2/);
  });

  it('is empty for a single goal that simply has no matching fact', () => {
    // nothing to say beyond the closed-world "no"
    expect(feedback('status(zephyr, S)')).toBe('');
  });

  it('suggests swapping a shared variable to the argument position that yields rows', () => {
    // M sits in the report slot; every report works on something, so the
    // negation empties the result. In the manager slot, ava survives.
    const text = feedback('reports_to(M, _), \\+ works_on(M, _)');
    expect(text).toMatch(/reports_to\(M, _\) alone matches 2 rows/);
    expect(text).toMatch(
      /Did you mean reports_to\(_, M\)\? That returns 1 row/,
    );
  });

  it('suggests joining through a different column when the shared variable sits in the wrong one', () => {
    // P is joined to status by the person column; the project is the third argument
    const text = feedback('promised(_, P, _), status(P, blocked)');
    expect(text).toMatch(
      /Did you mean promised\(_, P, V1\), status\(V1, blocked\)\? That returns 1 row/,
    );
  });

  it('counts goals through model-authored rules', () => {
    const text = feedback(
      'q(P) :- status(P, blocked), reports_to(P, maya).\n?- q(P).',
    );
    expect(text).toMatch(/status\(P, blocked\) alone matches 1 row/);
    expect(text).toMatch(/reports_to\(P, maya\) alone matches 0 rows/);
  });
});
