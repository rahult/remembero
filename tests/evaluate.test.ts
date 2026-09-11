import { describe, it, expect } from 'vitest';
import {
  evaluate,
  evaluateQuerySpec,
  evaluateQuerySpecWithProof,
  evaluateWithProof,
  parseProgram,
  parseQuery,
  parseQuerySpec,
  serializeTerm,
  EngineLimitError,
  EngineSafetyError,
  type EvaluateOptions,
  type Bindings,
  type Clause,
  type Goal,
  type ScalarExpression,
  DEFAULT_MAX_PROOF_ENUMERATION_STEPS,
  DEFAULT_MAX_PROOFS_PER_ROW,
  isIntegrityConstraint,
  materialize,
  materializeWithProof,
  literalMatches,
  MAX_PROOF_ENUMERATION_STEPS,
  MAX_PROOFS_PER_ROW,
} from '../src/engine/index.js';

/** Render bindings as sorted "X=a Y=b" rows for order-independent assertions. */
function rows(bindings: Bindings[]): string[] {
  return bindings
    .map((b) =>
      Object.entries(b)
        .map(([name, term]) => `${name}=${serializeTerm(term)}`)
        .sort()
        .join(' '),
    )
    .sort();
}

function run(program: string, query: string, options?: EvaluateOptions) {
  return evaluate(parseProgram(program), parseQuery(query), options);
}

function explain(program: string, query: string, options?: EvaluateOptions) {
  return evaluateWithProof(parseProgram(program), parseQuery(query), options);
}

function runSpec(program: string, query: string, options?: EvaluateOptions) {
  return evaluateQuerySpec(
    parseProgram(program),
    parseQuerySpec(query),
    options,
  );
}

function explainSpec(
  program: string,
  query: string,
  options?: EvaluateOptions,
) {
  return evaluateQuerySpecWithProof(
    parseProgram(program),
    parseQuerySpec(query),
    options,
  );
}

describe('evaluate: facts and joins', () => {
  const db = `
    works_at(rahul, acme).
    works_at(maya, acme).
    works_at(chen, initech).
    lives_in(rahul, sydney).
    lives_in(maya, melbourne).
  `;

  it('answers a ground query with one empty binding when true', () => {
    expect(run(db, 'works_at(rahul, acme)')).toEqual([{}]);
  });

  it('answers a false ground query with no bindings', () => {
    expect(run(db, 'works_at(rahul, initech)')).toEqual([]);
  });

  it('binds variables against facts', () => {
    expect(rows(run(db, 'works_at(X, acme)'))).toEqual(['X=maya', 'X=rahul']);
  });

  it('joins conjunctive goals on shared variables', () => {
    expect(rows(run(db, 'works_at(X, acme), lives_in(X, sydney)'))).toEqual([
      'X=rahul',
    ]);
  });

  it('supports wildcards without binding them', () => {
    expect(run(db, 'works_at(rahul, _)')).toEqual([{}]);
    expect(rows(run(db, 'works_at(X, _)'))).toEqual([
      'X=chen',
      'X=maya',
      'X=rahul',
    ]);
  });

  it('handles zero-arity predicates', () => {
    expect(run('raining.', 'raining')).toEqual([{}]);
    expect(run('raining.', 'sunny')).toEqual([]);
  });

  it('ignores integrity constraints during ordinary evaluation and materialization', () => {
    const program = `
      employee(alice).
      employee(bob).
      age(alice, 30).
      age(bob, 17).
      :- employee(X), age(X, A), A < 18, \\+ guardian_present(X).
    `;

    const clauses = parseProgram(program);
    expect(clauses.some(isIntegrityConstraint)).toBe(true);
    expect(rows(evaluate(clauses, parseQuery('employee(X)')))).toEqual([
      'X=alice',
      'X=bob',
    ]);
    expect(materializeWithProof(clauses).map((fact) => fact.predicate)).toEqual(
      ['employee', 'employee', 'age', 'age'],
    );
  });
});

describe('evaluate: explicit relational projection', () => {
  const db = `
    edge(a, x1).
    edge(a, x2).
    edge(x1, z).
    edge(x2, z).
    edge(x2, y).
  `;

  it('returns only selected variables in selected order and deduplicates projected rows', () => {
    const result = runSpec(db, 'select End where edge(a, Mid), edge(Mid, End)');
    expect(result).toEqual([
      { End: { type: 'atom', value: 'z' } },
      { End: { type: 'atom', value: 'y' } },
    ]);
    expect(
      runSpec(db, 'select End, Mid where edge(a, Mid), edge(Mid, End)')[0],
    ).toEqual({
      End: { type: 'atom', value: 'z' },
      Mid: { type: 'atom', value: 'x1' },
    });
  });

  it('applies row limits after projection rather than helper-variable expansion', () => {
    expect(
      runSpec(db, 'select End where edge(a, Mid), edge(Mid, End)', {
        maxRows: 1,
      }),
    ).toEqual([{ End: { type: 'atom', value: 'z' } }]);
  });

  it('merges complete alternative proof vectors for one projected row', () => {
    const result = explainSpec(
      db,
      'select End where edge(a, Mid), edge(Mid, End)',
      { maxProofsPerRow: 2 },
    );
    expect(result[0].bindings).toEqual({ End: { type: 'atom', value: 'z' } });
    expect(result[0].proofs).toHaveLength(2);
    expect(result[0].alternativeProofs).toHaveLength(1);
    expect(result[0].alternativeProofs?.[0]).toHaveLength(2);
  });
});

describe('evaluate: rules', () => {
  it('derives via a non-recursive rule with a comparison', () => {
    const db = `
      works_at(rahul, acme).
      works_at(maya, acme).
      works_at(chen, initech).
      colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
    `;
    expect(rows(run(db, 'colleague(rahul, Who)'))).toEqual(['Who=maya']);
  });

  it('computes transitive closure (ancestor)', () => {
    const db = `
      parent(alice, bob).
      parent(bob, carol).
      parent(carol, dan).
      ancestor(X, Y) :- parent(X, Y).
      ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    `;
    expect(rows(run(db, 'ancestor(alice, X)'))).toEqual([
      'X=bob',
      'X=carol',
      'X=dan',
    ]);
    expect(rows(run(db, 'ancestor(X, dan)'))).toEqual([
      'X=alice',
      'X=bob',
      'X=carol',
    ]);
  });

  it('computes same-generation', () => {
    const db = `
      person(alice). person(bob). person(betty). person(carol). person(chad).
      parent(alice, bob).
      parent(alice, betty).
      parent(bob, carol).
      parent(betty, chad).
      sg(X, X) :- person(X).
      sg(X, Y) :- parent(PX, X), parent(PY, Y), sg(PX, PY).
    `;
    expect(rows(run(db, 'sg(carol, Y), carol != Y'))).toEqual(['Y=chad']);
  });

  it('deduplicates derived facts across multiple derivation paths', () => {
    const db = `
      edge(a, b). edge(b, d). edge(a, c). edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;
    expect(rows(run(db, 'path(a, X)'))).toEqual(['X=b', 'X=c', 'X=d']);
  });
});

describe('evaluate: deterministic relation indexing', () => {
  const program = `
    selected(person_97).
    selected(person_98).
    selected(person_99).
    ${Array.from(
      { length: 100 },
      (_, index) => `related(person_${index}, topic_${index % 7}).`,
    ).join('\n')}
    blocked(person_98).
    relevant(X, Y) :- selected(X), related(X, Y), \\+ blocked(X).
  `;

  it('preserves byte-identical rows and proofs when the index is disabled', () => {
    const clauses = parseProgram(program);
    const query = parseQuery('relevant(X, Y)');
    const indexed = evaluateWithProof(clauses, query, {
      maxProofsPerRow: 4,
      relationIndex: 'auto',
    });
    const scanned = evaluateWithProof(clauses, query, {
      maxProofsPerRow: 4,
      relationIndex: 'off',
    });

    expect(JSON.stringify(indexed)).toBe(JSON.stringify(scanned));
  });

  it('preserves recursive first-witness and aggregate proof order', () => {
    const recursive = parseProgram(`
      edge(a, b). edge(b, d). edge(a, c). edge(c, d). edge(d, e).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `);
    const relational = parseQuery('path(a, Y)');
    const aggregate = parseQuerySpec('count(*) as Count where path(a, Y)');

    expect(
      JSON.stringify(
        evaluateWithProof(recursive, relational, {
          relationIndex: 'auto',
          maxProofsPerRow: 4,
        }),
      ),
    ).toBe(
      JSON.stringify(
        evaluateWithProof(recursive, relational, {
          relationIndex: 'off',
          maxProofsPerRow: 4,
        }),
      ),
    );
    expect(
      JSON.stringify(
        evaluateQuerySpecWithProof(recursive, aggregate, {
          relationIndex: 'auto',
        }),
      ),
    ).toBe(
      JSON.stringify(
        evaluateQuerySpecWithProof(recursive, aggregate, {
          relationIndex: 'off',
        }),
      ),
    );
  });

  it('profiles a bounded-first-argument join without relying on wall-clock timing', () => {
    const clauses = parseProgram(program);
    const query = parseQuery('relevant(X, Y)');
    const indexedMetrics = {
      relationLookups: 0,
      indexedRelationLookups: 0,
      indexFactsProcessed: 0,
      candidateFactsVisited: 0,
    };
    const scannedMetrics = { ...indexedMetrics };

    const indexed = evaluate(clauses, query, {
      relationIndex: 'auto',
      metrics: indexedMetrics,
    });
    const scanned = evaluate(clauses, query, {
      relationIndex: 'off',
      metrics: scannedMetrics,
    });

    expect(indexed).toEqual(scanned);
    expect(indexedMetrics.indexedRelationLookups).toBeGreaterThan(0);
    expect(indexedMetrics.candidateFactsVisited * 10).toBeLessThan(
      scannedMetrics.candidateFactsVisited,
    );
  });

  it('does not build an index for an unbound full-relation scan', () => {
    const facts = Array.from(
      { length: 100 },
      (_, index) => `related(person_${index}, topic_${index % 7}).`,
    ).join('\n');
    const metrics = {
      relationLookups: 0,
      indexedRelationLookups: 0,
      indexFactsProcessed: 0,
      candidateFactsVisited: 0,
    };

    expect(run(facts, 'related(X, Y)', { metrics })).toHaveLength(100);
    expect(metrics.indexedRelationLookups).toBe(0);
    expect(metrics.indexFactsProcessed).toBe(0);
    expect(metrics.candidateFactsVisited).toBe(100);
  });

  it('resets caller-supplied metrics for each public evaluation', () => {
    const metrics = {
      relationLookups: 99,
      indexedRelationLookups: 99,
      indexFactsProcessed: 99,
      candidateFactsVisited: 99,
    };

    expect(run('fact(a).', 'fact(X)', { metrics })).toHaveLength(1);
    expect(metrics).toEqual({
      relationLookups: 1,
      indexedRelationLookups: 0,
      indexFactsProcessed: 0,
      candidateFactsVisited: 1,
    });
    expect(
      runSpec('fact(a).', 'count(*) as Count where fact(X)', {
        maxRows: 0,
        metrics,
      }),
    ).toEqual([]);
    expect(metrics).toEqual({
      relationLookups: 0,
      indexedRelationLookups: 0,
      indexFactsProcessed: 0,
      candidateFactsVisited: 0,
    });
  });

  it('keeps numeric and atom first-argument buckets type-distinct', () => {
    const metrics = {
      relationLookups: 0,
      indexedRelationLookups: 0,
      indexFactsProcessed: 0,
      candidateFactsVisited: 0,
    };

    expect(
      rows(
        run("value(1, numeric). value('1', atom).", 'value(1, X)', { metrics }),
      ),
    ).toEqual(['X=numeric']);
    expect(metrics.indexFactsProcessed).toBe(2);
    expect(metrics.candidateFactsVisited).toBe(1);
  });

  it('rejects unknown relation index modes instead of silently changing execution', () => {
    expect(() =>
      run('fact(a).', 'fact(X)', {
        relationIndex: 'sometimes',
      } as unknown as EvaluateOptions),
    ).toThrow(EngineSafetyError);
  });
});

describe('evaluate: comparison builtins', () => {
  const db = `
    age(rahul, 38).
    age(maya, 29).
    age(kid, 11).
  `;

  it('filters with numeric comparisons', () => {
    expect(rows(run(db, 'age(X, A), A >= 29'))).toEqual([
      'A=29 X=maya',
      'A=38 X=rahul',
    ]);
    expect(rows(run(db, 'age(X, A), A < 18'))).toEqual(['A=11 X=kid']);
  });

  it('compares atoms lexicographically', () => {
    expect(rows(run('name(a). name(b).', 'name(X), X > a'))).toEqual(['X=b']);
  });

  it('fails mixed-type ordered comparisons instead of throwing', () => {
    expect(run(db, 'age(rahul, A), A > banana')).toEqual([]);
  });

  it('supports equality on ground terms', () => {
    expect(rows(run(db, 'age(X, A), A = 38'))).toEqual(['A=38 X=rahul']);
  });

  it('evaluates arithmetic with standard precedence and grouping', () => {
    expect(rows(run(db, 'age(X, A), A > 10 + 3 * 6'))).toEqual([
      'A=29 X=maya',
      'A=38 X=rahul',
    ]);
    expect(rows(run(db, 'age(X, A), A > (10 + 3) * 2'))).toEqual([
      'A=29 X=maya',
      'A=38 X=rahul',
    ]);
    expect(rows(run(db, 'age(X, A), A > (10 + 5) * 2'))).toEqual([
      'A=38 X=rahul',
    ]);
  });

  it('evaluates subtraction and division left-associatively on either side', () => {
    const values = 'value(five, 5). value(two, 2). value(negative, -5).';
    expect(rows(run(values, 'value(X, V), V = 10 - 3 - 2'))).toEqual([
      'V=5 X=five',
    ]);
    expect(rows(run(values, 'value(X, V), 20 / 2 / 2 = V'))).toEqual([
      'V=5 X=five',
    ]);
    expect(rows(run(values, 'value(X, V), -V = 5'))).toEqual([
      'V=-5 X=negative',
    ]);
  });

  it('uses bound variables from multiple relations in arithmetic filters', () => {
    const program = `
      score(alice, 20). score(bob, 14). baseline(team, 10).
      ahead(X) :- score(X, S), baseline(team, B), S > B + 5.
    `;
    expect(rows(run(program, 'ahead(X)'))).toEqual(['X=alice']);
    expect(
      rows(run(program, 'score(X, S), baseline(team, B), S - B >= 10')),
    ).toEqual(['B=10 S=20 X=alice']);
  });

  it('fails closed on non-numeric, zero-divisor, and non-finite arithmetic', () => {
    expect(() => run('value(x, banana).', 'value(x, V), V + 1 > 0')).toThrow(
      EngineSafetyError,
    );
    expect(() => run('value(x, 1).', 'value(x, V), V / 0 > 0')).toThrow(
      /division by zero/i,
    );
    const huge = '9'.repeat(200);
    expect(() => run(`value(x, ${huge}).`, 'value(x, V), V * V > 0')).toThrow(
      /non-finite/i,
    );
  });

  it('revalidates hand-built arithmetic ASTs at the evaluator boundary', () => {
    const ungrounded: Goal[] = [
      {
        op: '>',
        left: {
          kind: 'binary',
          op: '+',
          left: { type: 'var', name: 'Missing' },
          right: { type: 'num', value: 1 },
        },
        right: { type: 'num', value: 0 },
      },
    ];
    expect(() => evaluate([], ungrounded)).toThrow(/not grounded/i);

    let deep: ScalarExpression = { type: 'num', value: 1 };
    for (let index = 0; index < 65; index++) {
      deep = { kind: 'unary', op: '-', operand: deep };
    }
    expect(() =>
      evaluate(
        [],
        [{ op: '=', left: deep, right: { type: 'num', value: -1 } }],
      ),
    ).toThrow(EngineLimitError);

    for (const value of [Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() =>
        evaluate(
          [],
          [
            {
              op: '>',
              left: { type: 'num', value },
              right: { type: 'num', value: 0 },
            },
          ],
        ),
      ).toThrow(/finite/i);
    }

    const invalidFact: Clause = {
      head: { predicate: 'value', args: [{ type: 'num', value: Number.NaN }] },
      body: [],
    };
    expect(() => evaluate([invalidFact], parseQuery('value(_)'))).toThrow(
      /finite/i,
    );
    expect(() =>
      serializeTerm({ type: 'num', value: Number.POSITIVE_INFINITY }),
    ).toThrow(/finite/i);
    expect(() =>
      literalMatches(
        { predicate: 'value', args: [{ type: 'wildcard' }] },
        { predicate: 'value', args: [{ type: 'num', value: Number.NaN }] },
      ),
    ).toThrow(/finite/i);
  });

  it('keeps arithmetic filters out of derivation proof nodes', () => {
    const program = `
      age(alice, 30). threshold(adult, 18).
      adult(X) :- age(X, A), threshold(adult, T), A >= T + 0.
    `;
    expect(explain(program, 'adult(alice)')).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'adult',
            values: ['alice'],
            rule: 1,
            because: [
              { predicate: 'age', values: ['alice', 30] },
              { predicate: 'threshold', values: ['adult', 18] },
            ],
          },
        ],
      },
    ]);
  });
});

describe('evaluate: stratified negation', () => {
  const employment = `
    employee(alice).
    employee(bob).
    employee(carol).
    suspended(carol).
    desk(alice, 101).
    eligible(X) :- employee(X), \\+ suspended(X).
    employee_without_desk(X) :- employee(X), \\+ desk(X, _).
  `;

  it('filters on completed lower-stratum relations', () => {
    expect(rows(run(employment, 'eligible(X)'))).toEqual(['X=alice', 'X=bob']);
    expect(rows(run(employment, 'employee_without_desk(X)'))).toEqual([
      'X=bob',
      'X=carol',
    ]);
  });

  it('answers ground negative queries under the closed-world assumption', () => {
    expect(run(employment, '\\+ suspended(alice)')).toEqual([{}]);
    expect(run(employment, '\\+ suspended(carol)')).toEqual([]);
  });

  it('supports ground rules whose body is only an absence check', () => {
    expect(run('safe :- \\+ outage.', 'safe')).toEqual([{}]);
    expect(run('outage. safe :- \\+ outage.', 'safe')).toEqual([]);
    expect(explain('safe :- \\+ outage.', 'safe')).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'safe',
            values: [],
            rule: 1,
            because: [
              { negated: true, predicate: 'outage', pattern: [], stratum: 0 },
            ],
          },
        ],
      },
    ]);
  });

  it('combines same-stratum recursion with a lower-stratum negative filter', () => {
    const db = `
      edge(a, b). edge(b, c). edge(c, d). blocked(d).
      reachable(X, Y) :- edge(X, Y), \\+ blocked(Y).
      reachable(X, Y) :- edge(X, Z), reachable(Z, Y), \\+ blocked(Y).
    `;
    expect(rows(run(db, 'reachable(a, Y)'))).toEqual(['Y=b', 'Y=c']);
    expect(run(db, 'reachable(a, d)')).toEqual([]);
  });

  it('emits an atomic absence proof rather than inventing source support', () => {
    expect(explain(employment, 'eligible(bob)')).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'eligible',
            values: ['bob'],
            rule: 1,
            because: [
              { predicate: 'employee', values: ['bob'] },
              {
                negated: true,
                predicate: 'suspended',
                pattern: ['bob'],
                stratum: 0,
              },
            ],
          },
        ],
      },
    ]);
    expect(
      explain(employment, 'employee_without_desk(bob)')[0].proofs[0],
    ).toMatchObject({
      predicate: 'employee_without_desk',
      because: [
        { predicate: 'employee', values: ['bob'] },
        {
          negated: true,
          predicate: 'desk',
          pattern: ['bob', null],
          stratum: 0,
        },
      ],
    });
  });

  it('evaluates successive negative dependencies in completed strata', () => {
    const db = `
      person(alice). person(bob). person(carol).
      suspended(carol). retired(bob).
      active(X) :- person(X), \\+ suspended(X).
      billable(X) :- active(X), \\+ retired(X).
      mention(X) :- billable(X).
    `;
    expect(rows(run(db, 'mention(X)'))).toEqual(['X=alice']);
    const proof = explain(db, 'mention(alice)')[0].proofs[0];
    expect(JSON.stringify(proof)).toContain('"predicate":"suspended"');
    expect(JSON.stringify(proof)).toContain('"predicate":"retired"');
  });

  it('counts absence nodes against the global proof budget', () => {
    expect(() =>
      explain(employment, 'eligible(bob)', { maxProofNodes: 2 }),
    ).toThrow(EngineLimitError);
  });
});

describe('evaluate: limits', () => {
  it('throws EngineLimitError when base facts exceed maxFacts', () => {
    const db = `
      n(1). n(2). n(3).
    `;
    expect(() => run(db, 'n(X)', { maxFacts: 2 })).toThrow(EngineLimitError);
  });

  it('throws EngineLimitError when derived facts exceed maxFacts', () => {
    const db = `
      n(1). n(2). n(3). n(4). n(5).
      pair(X, Y) :- n(X), n(Y).
    `;
    expect(() => run(db, 'pair(X, Y)', { maxFacts: 10 })).toThrow(
      EngineLimitError,
    );
  });
});

describe('evaluate: reusable aggregate rules', () => {
  const grouped = `
    member(red, alice). member(red, bob). member(blue, carol).
    score(red, alice, 10). score(red, bob, 15). score(blue, carol, 7).
    team_size(Team, Count) :- count(*) as Count where member(Team, Person).
    team_total(Team, Total) :- sum(Points) as Total where score(Team, Person, Points).
    team_min(Team, Minimum) :- min(Points) as Minimum where score(Team, Person, Points).
    team_max(Team, Maximum) :- max(Points) as Maximum where score(Team, Person, Points).
    large_team(Team) :- team_size(Team, Count), Count >= 2.
    largest_team_size(Maximum) :- max(Count) as Maximum where team_size(Team, Count).
  `;

  it('derives one deterministic aggregate fact per group for downstream rules', () => {
    expect(rows(run(grouped, 'team_size(Team, Count)'))).toEqual([
      'Count=1 Team=blue',
      'Count=2 Team=red',
    ]);
    expect(rows(run(grouped, 'team_total(Team, Total)'))).toEqual([
      'Team=blue Total=7',
      'Team=red Total=25',
    ]);
    expect(rows(run(grouped, 'team_min(red, Minimum)'))).toEqual([
      'Minimum=10',
    ]);
    expect(rows(run(grouped, 'team_max(red, Maximum)'))).toEqual([
      'Maximum=15',
    ]);
    expect(rows(run(grouped, 'large_team(Team)'))).toEqual(['Team=red']);
    expect(rows(run(grouped, 'largest_team_size(Maximum)'))).toEqual([
      'Maximum=2',
    ]);
    expect(run(grouped, 'team_size(Team, Count)')).toEqual(
      run(grouped, 'team_size(Team, Count)'),
    );
  });

  it('supports deterministic aggregate strata over aggregate-derived relations', () => {
    const proof = explain(grouped, 'largest_team_size(Maximum)')[0].proofs[0];
    expect(proof).toMatchObject({
      predicate: 'largest_team_size',
      aggregate: {
        aggregated: true,
        op: 'max',
        value: 2,
        witnessPositions: [0],
        contributors: [
          {
            bindings: { Team: { value: 'red' }, Count: { value: 2 } },
            proofs: [{ predicate: 'team_size', aggregate: { value: 2 } }],
          },
          {
            bindings: { Team: { value: 'blue' }, Count: { value: 1 } },
            proofs: [{ predicate: 'team_size', aggregate: { value: 1 } }],
          },
        ],
      },
    });
  });

  it('preserves aggregate-rule rows and nested proofs with indexing disabled', () => {
    expect(
      explain(grouped, 'largest_team_size(Maximum)', { relationIndex: 'auto' }),
    ).toEqual(
      explain(grouped, 'largest_team_size(Maximum)', { relationIndex: 'off' }),
    );
  });

  it('lets terminal aggregates reduce aggregate-derived relations with nested evidence', () => {
    const result = explainSpec(
      grouped,
      'count(*) as TeamCount where team_size(Team, Size)',
    );
    expect(result).toMatchObject([
      {
        bindings: { TeamCount: { value: 2 } },
        proofs: [
          {
            aggregated: true,
            op: 'count',
            value: 2,
            contributors: [
              {
                bindings: { Team: { value: 'red' }, Size: { value: 2 } },
                proofs: [{ predicate: 'team_size', aggregate: { value: 2 } }],
              },
              {
                bindings: { Team: { value: 'blue' }, Size: { value: 1 } },
                proofs: [{ predicate: 'team_size', aggregate: { value: 1 } }],
              },
            ],
          },
        ],
      },
    ]);
  });

  it('derives global count zero but no empty grouped or scalar reduction', () => {
    const program = `
      global_count(Count) :- count(*) as Count where missing(Item).
      grouped_count(Group, Count) :- count(*) as Count where missing(Group, Item).
      global_sum(Total) :- sum(Value) as Total where missing_value(Value).
    `;
    expect(rows(run(program, 'global_count(Count)'))).toEqual(['Count=0']);
    expect(run(program, 'grouped_count(Group, Count)')).toEqual([]);
    expect(run(program, 'global_sum(Total)')).toEqual([]);
  });

  it('aggregates completed derived and negated input strata', () => {
    const program = `
      employee(alice, red). employee(bob, red). employee(carol, blue).
      suspended(bob).
      eligible(Person, Team) :- employee(Person, Team), \\+ suspended(Person).
      eligible_count(Team, Count) :- count(*) as Count where eligible(Person, Team).
    `;
    expect(rows(run(program, 'eligible_count(Team, Count)'))).toEqual([
      'Count=1 Team=blue',
      'Count=1 Team=red',
    ]);
  });

  it('nests bounded aggregate contributor evidence under the derived claim', () => {
    const result = explain(grouped, 'team_size(red, Count)');
    expect(result).toEqual([
      {
        bindings: { Count: { type: 'num', value: 2 } },
        proofs: [
          {
            predicate: 'team_size',
            values: ['red', 2],
            rule: 1,
            aggregate: {
              aggregated: true,
              op: 'count',
              input: '*',
              as: 'Count',
              value: 2,
              contributors: [
                {
                  bindings: {
                    Team: { type: 'atom', value: 'red' },
                    Person: { type: 'atom', value: 'alice' },
                  },
                  proofs: [{ predicate: 'member', values: ['red', 'alice'] }],
                },
                {
                  bindings: {
                    Team: { type: 'atom', value: 'red' },
                    Person: { type: 'atom', value: 'bob' },
                  },
                  proofs: [{ predicate: 'member', values: ['red', 'bob'] }],
                },
              ],
            },
          },
        ],
      },
    ]);
    expect(
      materializeWithProof(parseProgram(grouped)).find(
        (fact) => fact.predicate === 'team_size' && fact.values[0] === 'red',
      )?.proof.aggregate,
    ).toMatchObject({ aggregated: true, value: 2 });
  });

  it('fails closed on aggregate rule input, proof, and alternative-proof bounds', () => {
    const program = `
      item(a). item(b). item(c).
      item_count(Count) :- count(*) as Count where item(Item).
    `;
    expect(() =>
      run(program, 'item_count(Count)', { maxAggregateRows: 2 }),
    ).toThrow(/aggregate input exceeded 2/i);
    expect(() =>
      explain(program, 'item_count(Count)', { maxAggregateProofRows: 2 }),
    ).toThrow(/aggregate proof exceeded 2 contributor rows/i);
    expect(() =>
      explain(program, 'item_count(Count)', { maxProofsPerRow: 2 }),
    ).toThrow(/alternative proofs through aggregate-derived rules/i);
  });

  it('fails closed when a grouped sum receives non-numeric input', () => {
    expect(() =>
      run(
        `value(team, one).
         total(Group, Sum) :- sum(Value) as Sum where value(Group, Value).`,
        'total(Group, Sum)',
      ),
    ).toThrow(/numeric input/i);
  });

  it('validates directly constructed aggregate clauses before evaluation', () => {
    const output = { type: 'var' as const, name: 'Count' };
    const negationOnly: Clause = {
      head: { predicate: 'bad_count', args: [output] },
      body: [
        {
          not: {
            predicate: 'missing',
            args: [{ type: 'atom', value: 'item' }],
          },
        },
      ],
      aggregate: { op: 'count', input: '*', as: 'Count' },
    };
    expect(() =>
      evaluate([negationOnly], parseQuery('bad_count(Count)')),
    ).toThrow(/positive relation/i);

    const outputReused: Clause = {
      head: { predicate: 'bad_count', args: [output] },
      body: [{ predicate: 'item', args: [{ type: 'var', name: 'Count' }] }],
      aggregate: { op: 'count', input: '*', as: 'Count' },
    };
    expect(() =>
      evaluate(
        [parseProgram('item(a).')[0], outputReused],
        parseQuery('bad_count(Count)'),
      ),
    ).toThrow(/fresh variable/i);

    const unboundGroup: Clause = {
      head: {
        predicate: 'bad_group',
        args: [{ type: 'var', name: 'Group' }, output],
      },
      body: [{ predicate: 'item', args: [{ type: 'var', name: 'Person' }] }],
      aggregate: { op: 'count', input: '*', as: 'Count' },
    };
    expect(() =>
      evaluate(
        [parseProgram('item(a).')[0], unboundGroup],
        parseQuery('bad_group(Group, Count)'),
      ),
    ).toThrow(/Group.*positive aggregate relation/i);
  });
});

describe('evaluate: scalar query aggregation', () => {
  const employment = `
    works_at(alice, acme).
    works_at(bob, acme).
    works_at(carol, initech).
    suspended(bob).
  `;

  it('counts complete deduplicated result rows, including zero', () => {
    expect(
      runSpec(employment, 'count(*) as Count where works_at(Person, acme)'),
    ).toEqual([{ Count: { type: 'num', value: 2 } }]);
    expect(
      runSpec(employment, 'count(*) as Count where works_at(alice, acme)'),
    ).toEqual([{ Count: { type: 'num', value: 1 } }]);
    expect(
      runSpec(employment, 'count(*) as Count where works_at(Person, nowhere)'),
    ).toEqual([{ Count: { type: 'num', value: 0 } }]);
    expect(
      runSpec(
        employment,
        'count(*) as Count where works_at(Person, _), \\+ suspended(Person)',
      ),
    ).toEqual([{ Count: { type: 'num', value: 2 } }]);
  });

  it('aggregates recursive query results rather than derivation multiplicity', () => {
    const graph = `
      edge(a, b). edge(b, c). edge(a, c). edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;
    expect(
      runSpec(graph, 'count(*) as Count where path(a, Descendant)'),
    ).toEqual([{ Count: { type: 'num', value: 3 } }]);
    expect(
      runSpec(
        'person(alice). tag(alice, one). tag(alice, two).',
        'count(*) as Count where person(Person), tag(Person, _)',
      ),
    ).toEqual([{ Count: { type: 'num', value: 2 } }]);
    expect(
      runSpec(employment, 'count(*) as Count where works_at(_, acme)'),
    ).toEqual([{ Count: { type: 'num', value: 2 } }]);
  });

  it('applies arithmetic filters before exact aggregation', () => {
    const scores = 'score(alice, 20). score(bob, 14). baseline(team, 10).';
    expect(
      runSpec(
        scores,
        'count(*) as Count where score(Person, Points), baseline(team, Base), Points > Base + 5',
      ),
    ).toEqual([{ Count: { type: 'num', value: 1 } }]);
  });

  it('computes sum over numeric rows and returns no row for empty input', () => {
    const scores = 'score(alice, 1.5). score(bob, -2). score(carol, 4).';
    expect(
      runSpec(scores, 'sum(Points) as Total where score(Player, Points)'),
    ).toEqual([{ Total: { type: 'num', value: 3.5 } }]);
    expect(
      runSpec(scores, 'sum(Points) as Total where score(nobody, Points)'),
    ).toEqual([]);
  });

  it('computes numeric and atom extrema with deterministic tie positions', () => {
    const values =
      'score(a, 2). score(b, 1). score(c, 1). name(zoe). name(amy).';
    expect(
      runSpec(values, 'min(Value) as Minimum where score(_, Value)'),
    ).toEqual([{ Minimum: { type: 'num', value: 1 } }]);
    expect(
      runSpec(values, 'max(Value) as Maximum where score(_, Value)'),
    ).toEqual([{ Maximum: { type: 'num', value: 2 } }]);
    expect(runSpec(values, 'min(Name) as First where name(Name)')).toEqual([
      { First: { type: 'atom', value: 'amy' } },
    ]);

    const proof = explainSpec(
      values,
      'min(Value) as Minimum where score(Person, Value)',
    );
    expect(proof[0].proofs[0]).toMatchObject({
      aggregated: true,
      op: 'min',
      input: 'Value',
      as: 'Minimum',
      value: 1,
      witnessPositions: [1, 2],
    });
  });

  it('fails closed for invalid scalar domains and non-finite sums', () => {
    expect(() =>
      runSpec(
        'value(1). value(one).',
        'min(Value) as Minimum where value(Value)',
      ),
    ).toThrow(EngineSafetyError);
    expect(() =>
      runSpec('value(one).', 'sum(Value) as Total where value(Value)'),
    ).toThrow(EngineSafetyError);
    const huge = '9'.repeat(307);
    const overflowing = Array.from(
      { length: 20 },
      (_, index) => `value(${index}, ${huge}).`,
    ).join(' ');
    expect(() =>
      runSpec(overflowing, 'sum(Value) as Total where value(Id, Value)'),
    ).toThrow(EngineSafetyError);
  });

  it('does not silently reuse maxRows as the aggregate input cap', () => {
    const facts = Array.from(
      { length: 1005 },
      (_, index) => `item(${index}).`,
    ).join(' ');
    expect(
      runSpec(facts, 'count(*) as Count where item(Item)', { maxRows: 1 }),
    ).toEqual([{ Count: { type: 'num', value: 1005 } }]);
  });

  it('fails closed when aggregate input exceeds its dedicated cap', () => {
    expect(() =>
      runSpec(
        'item(1). item(2). item(3).',
        'count(*) as Count where item(Item)',
        {
          maxAggregateRows: 2,
        },
      ),
    ).toThrow(/aggregate input exceeded 2/i);
  });

  it('emits one bounded aggregate proof with ordered contributor evidence', () => {
    const result = explainSpec(
      employment,
      'count(*) as Count where works_at(Person, acme)',
    );
    expect(result).toEqual([
      {
        bindings: { Count: { type: 'num', value: 2 } },
        proofs: [
          {
            aggregated: true,
            op: 'count',
            input: '*',
            as: 'Count',
            value: 2,
            contributors: [
              {
                bindings: { Person: { type: 'atom', value: 'alice' } },
                proofs: [{ predicate: 'works_at', values: ['alice', 'acme'] }],
              },
              {
                bindings: { Person: { type: 'atom', value: 'bob' } },
                proofs: [{ predicate: 'works_at', values: ['bob', 'acme'] }],
              },
            ],
          },
        ],
      },
    ]);
    expect(() =>
      explainSpec(
        employment,
        'count(*) as Count where works_at(Person, acme)',
        {
          maxProofNodes: 2,
        },
      ),
    ).toThrow(EngineLimitError);
  });

  it('separates exact aggregate evaluation from the smaller explanation cap', () => {
    const facts = Array.from(
      { length: 257 },
      (_, index) => `item(${index}).`,
    ).join(' ');
    const query = 'count(*) as Count where item(Item)';
    expect(runSpec(facts, query)).toEqual([
      { Count: { type: 'num', value: 257 } },
    ]);
    expect(() => explainSpec(facts, query)).toThrow(
      /aggregate proof exceeded 256 contributor rows/i,
    );
    expect(
      explainSpec(facts, query, { maxAggregateProofRows: 257 })[0].proofs[0],
    ).toMatchObject({ aggregated: true, value: 257 });
  });

  it('rejects aggregate alternative-proof enumeration explicitly', () => {
    const query = 'count(*) as Count where item(Item)';
    expect(runSpec('item(1). item(2).', query, { maxProofsPerRow: 2 })).toEqual(
      [{ Count: { type: 'num', value: 2 } }],
    );
    expect(() =>
      explainSpec('item(1). item(2).', query, { maxProofsPerRow: 2 }),
    ).toThrow(/alternative proofs are relational-only/i);
  });
});

describe('evaluateWithProof', () => {
  it('keeps rule numbering and witnesses unchanged when integrity constraints are present', () => {
    const withoutConstraint = `
      edge(a, b).
      path(X, Y) :- edge(X, Y).
    `;
    const withConstraint = `
      edge(a, b).
      :- edge(X, Y), X != Y, \\+ blocked(Y).
      path(X, Y) :- edge(X, Y).
    `;

    expect(explain(withConstraint, 'path(a, b)')).toEqual(
      explain(withoutConstraint, 'path(a, b)'),
    );
  });

  it('keeps default proof behavior unchanged when alternative enumeration is off', () => {
    const result = explain('works_at(rahul, acme).', 'works_at(rahul, acme)');
    expect(result).toEqual([
      {
        bindings: {},
        proofs: [{ predicate: 'works_at', values: ['rahul', 'acme'] }],
      },
    ]);
    expect(result[0]).not.toHaveProperty('alternativeProofs');
    expect(DEFAULT_MAX_PROOFS_PER_ROW).toBe(1);
    expect(DEFAULT_MAX_PROOF_ENUMERATION_STEPS).toBe(100_000);
  });

  it('returns a leaf proof for a base fact query', () => {
    expect(explain('works_at(rahul, acme).', 'works_at(rahul, acme)')).toEqual([
      {
        bindings: {},
        proofs: [{ predicate: 'works_at', values: ['rahul', 'acme'] }],
      },
    ]);
  });

  it('returns ordered proofs for multi-goal queries', () => {
    const db = `
      works_at(rahul, acme).
      lives_in(rahul, sydney).
    `;

    expect(explain(db, 'works_at(X, acme), lives_in(X, sydney)')).toEqual([
      {
        bindings: { X: { type: 'atom', value: 'rahul' } },
        proofs: [
          { predicate: 'works_at', values: ['rahul', 'acme'] },
          { predicate: 'lives_in', values: ['rahul', 'sydney'] },
        ],
      },
    ]);
  });

  it('enumerates distinct alternative proofs from multiple matching rules', () => {
    const db = `
      direct(a, d).
      edge(a, b).
      edge(b, d).
      reach(X, Y) :- direct(X, Y).
      reach(X, Y) :- edge(X, Z), edge(Z, Y).
    `;

    expect(explain(db, 'reach(a, d)', { maxProofsPerRow: 2 })).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'reach',
            values: ['a', 'd'],
            rule: 1,
            because: [{ predicate: 'direct', values: ['a', 'd'] }],
          },
        ],
        alternativeProofs: [
          [
            {
              predicate: 'reach',
              values: ['a', 'd'],
              rule: 2,
              because: [
                { predicate: 'edge', values: ['a', 'b'] },
                { predicate: 'edge', values: ['b', 'd'] },
              ],
            },
          ],
        ],
      },
    ]);
  });

  it('returns a joined rule proof', () => {
    const db = `
      works_at(rahul, acme).
      works_at(maya, acme).
      colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
    `;

    expect(explain(db, 'colleague(rahul, maya)')).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'colleague',
            values: ['rahul', 'maya'],
            rule: 1,
            because: [
              { predicate: 'works_at', values: ['rahul', 'acme'] },
              { predicate: 'works_at', values: ['maya', 'acme'] },
            ],
          },
        ],
      },
    ]);
  });

  it('enumerates recursive multi-path witnesses in deterministic order', () => {
    const db = `
      edge(a, b).
      edge(b, d).
      edge(a, c).
      edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    expect(explain(db, 'path(a, d)', { maxProofsPerRow: 2 })).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'path',
            values: ['a', 'd'],
            rule: 2,
            because: [
              { predicate: 'edge', values: ['a', 'b'] },
              {
                predicate: 'path',
                values: ['b', 'd'],
                rule: 1,
                because: [{ predicate: 'edge', values: ['b', 'd'] }],
              },
            ],
          },
        ],
        alternativeProofs: [
          [
            {
              predicate: 'path',
              values: ['a', 'd'],
              rule: 2,
              because: [
                { predicate: 'edge', values: ['a', 'c'] },
                {
                  predicate: 'path',
                  values: ['c', 'd'],
                  rule: 1,
                  because: [{ predicate: 'edge', values: ['c', 'd'] }],
                },
              ],
            },
          ],
        ],
      },
    ]);
  });

  it('returns a recursive proof tree that matches the SQLite structure', () => {
    const db = `
      edge(a, b).
      edge(b, c).
      edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    expect(explain(db, 'path(a, d)')).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'path',
            values: ['a', 'd'],
            rule: 2,
            because: [
              { predicate: 'edge', values: ['a', 'b'] },
              {
                predicate: 'path',
                values: ['b', 'd'],
                rule: 2,
                because: [
                  { predicate: 'edge', values: ['b', 'c'] },
                  {
                    predicate: 'path',
                    values: ['c', 'd'],
                    rule: 1,
                    because: [{ predicate: 'edge', values: ['c', 'd'] }],
                  },
                ],
              },
            ],
          },
        ],
      },
    ]);
  });

  it('keeps the deterministic first witness across repeated runs', () => {
    const db = `
      edge(a, b).
      edge(b, d).
      edge(a, c).
      edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    const runs = Array.from({ length: 5 }, () => explain(db, 'path(a, d)'));
    expect(new Set(runs.map((run) => JSON.stringify(run))).size).toBe(1);
    expect(runs[0]).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'path',
            values: ['a', 'd'],
            rule: 2,
            because: [
              { predicate: 'edge', values: ['a', 'b'] },
              {
                predicate: 'path',
                values: ['b', 'd'],
                rule: 1,
                because: [{ predicate: 'edge', values: ['b', 'd'] }],
              },
            ],
          },
        ],
      },
    ]);
  });

  it('keeps the first rule as witness when duplicate rules derive the same tuple', () => {
    const db = `
      base(a).
      pick(X) :- base(X).
      pick(X) :- base(X).
    `;
    expect(explain(db, 'pick(a)')[0].proofs[0]).toMatchObject({ rule: 1 });
  });

  it('deduplicates hidden-variable witnesses and identical duplicate-rule proofs structurally', () => {
    const wildcard = `
      person(alice).
      tag(alice, one).
      tag(alice, two).
    `;
    expect(
      explain(wildcard, 'person(Person), tag(Person, _)', {
        maxProofsPerRow: 2,
      }),
    ).toEqual([
      {
        bindings: { Person: { type: 'atom', value: 'alice' } },
        proofs: [
          { predicate: 'person', values: ['alice'] },
          { predicate: 'tag', values: ['alice', 'one'] },
        ],
        alternativeProofs: [
          [
            { predicate: 'person', values: ['alice'] },
            { predicate: 'tag', values: ['alice', 'two'] },
          ],
        ],
      },
    ]);

    const duplicateRules = `
      base(a).
      pick(X) :- base(X).
      pick(X) :- base(X).
    `;
    const duplicate = explain(duplicateRules, 'pick(a)', {
      maxProofsPerRow: 2,
    });
    expect(duplicate).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'pick',
            values: ['a'],
            rule: 1,
            because: [{ predicate: 'base', values: ['a'] }],
          },
        ],
      },
    ]);
    expect(duplicate[0]).not.toHaveProperty('alternativeProofs');
  });

  it('keeps semantically distinct rules even when they have the same supporting facts', () => {
    const db = `
      base(a).
      pick(X) :- base(X), 1 = 1.
      pick(X) :- base(X), 2 = 2.
    `;

    const result = explain(db, 'pick(a)', { maxProofsPerRow: 2 });
    expect(result[0].proofs[0]).toMatchObject({ rule: 1 });
    expect(result[0].alternativeProofs).toEqual([
      [expect.objectContaining({ predicate: 'pick', rule: 2 })],
    ]);
  });

  it('avoids cyclic self-support while still returning the acyclic witness', () => {
    const db = `
      seed(a).
      loop(X) :- seed(X).
      loop(X) :- loop(X).
    `;

    expect(explain(db, 'loop(a)', { maxProofsPerRow: 2 })).toEqual([
      {
        bindings: {},
        proofs: [
          {
            predicate: 'loop',
            values: ['a'],
            rule: 1,
            because: [{ predicate: 'seed', values: ['a'] }],
          },
        ],
      },
    ]);
  });

  it('fails closed when a row has more proofs than the requested cap', () => {
    const db = `
      edge(a, b).
      edge(b, d).
      edge(a, c).
      edge(c, d).
      edge(a, e).
      edge(e, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    expect(() => explain(db, 'path(a, d)', { maxProofsPerRow: 2 })).toThrow(
      /proof alternatives exceeded maxProofsPerRow 2/i,
    );
  });

  it('validates alternative-proof options and the exported hard caps', () => {
    expect(MAX_PROOFS_PER_ROW).toBe(16);
    expect(MAX_PROOF_ENUMERATION_STEPS).toBe(1_000_000);
    expect(() =>
      explain('fact(a).', 'fact(a)', { maxProofsPerRow: 0 }),
    ).toThrow(/maxProofsPerRow/i);
    expect(() =>
      explain('fact(a).', 'fact(a)', {
        maxProofsPerRow: MAX_PROOFS_PER_ROW + 1,
      }),
    ).toThrow(/maxProofsPerRow/i);
    expect(() =>
      explain('fact(a).', 'fact(a)', { maxProofEnumerationSteps: 0 }),
    ).toThrow(/maxProofEnumerationSteps/i);
    expect(() =>
      explain('fact(a).', 'fact(a)', {
        maxProofEnumerationSteps: MAX_PROOF_ENUMERATION_STEPS + 1,
      }),
    ).toThrow(/maxProofEnumerationSteps/i);
  });

  it('fails closed when proof enumeration work exceeds the configured cap', () => {
    const db = `
      person(alice).
      tag(alice, one).
      tag(alice, two).
    `;

    expect(() =>
      explain(db, 'person(Person), tag(Person, _)', {
        maxProofsPerRow: 2,
        maxProofEnumerationSteps: 1,
      }),
    ).toThrow(/proof enumeration exceeded 1 steps/i);
  });

  it('enforces the proof depth cap during serialization', () => {
    const db = `
      edge(a, b).
      edge(b, c).
      edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    expect(() => explain(db, 'path(a, d)', { maxProofDepth: 2 })).toThrow(
      EngineLimitError,
    );
  });

  it('shares proof depth and node budgets across primary and alternative witnesses', () => {
    const wildcard = `
      person(alice).
      tag(alice, one).
      tag(alice, two).
    `;
    expect(() =>
      explain(wildcard, 'person(Person), tag(Person, _)', {
        maxProofsPerRow: 2,
        maxProofNodes: 3,
      }),
    ).toThrow(EngineLimitError);

    const recursive = `
      edge(a, b).
      edge(b, d).
      edge(a, c).
      edge(c, d).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;
    expect(() =>
      explain(recursive, 'path(a, d)', {
        maxProofsPerRow: 2,
        maxProofDepth: 2,
      }),
    ).toThrow(EngineLimitError);
  });

  it('enforces the proof node cap for a single joined proof', () => {
    const db = `
      works_at(rahul, acme).
      works_at(maya, acme).
      colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
    `;

    expect(() =>
      explain(db, 'colleague(rahul, maya)', { maxProofNodes: 2 }),
    ).toThrow(EngineLimitError);
  });

  it('enforces the proof node cap across emitted rows', () => {
    const db = `
      works_at(rahul, acme).
      works_at(maya, acme).
    `;

    expect(() =>
      explain(db, 'works_at(X, acme)', { maxProofNodes: 1 }),
    ).toThrow(EngineLimitError);
  });
});

describe('materializeWithProof', () => {
  it('can materialize the same bounded fixpoint without serializing proofs', () => {
    const program = parseProgram(
      'edge(a, b). edge(b, c). path(X, Y) :- edge(X, Y).',
    );
    expect(materialize(program)).toEqual(
      materializeWithProof(program).map(({ proof: _proof, ...fact }) => fact),
    );
    expect(() => materialize(program, { maxFacts: 2 })).toThrow(
      EngineLimitError,
    );
  });

  it('returns base and derived facts with proofs', () => {
    const db = `
      edge(a, b).
      edge(b, c).
      path(X, Y) :- edge(X, Y).
      path(X, Y) :- edge(X, Z), path(Z, Y).
    `;

    const facts = materializeWithProof(parseProgram(db));
    const byKey = new Map(
      facts.map(
        (fact) =>
          [`${fact.predicate}(${fact.values.join(',')})`, fact] as const,
      ),
    );

    expect(byKey.get('edge(a,b)')).toEqual({
      predicate: 'edge',
      values: ['a', 'b'],
      derived: false,
      proof: { predicate: 'edge', values: ['a', 'b'] },
    });
    expect(byKey.get('path(a,c)')).toEqual({
      predicate: 'path',
      values: ['a', 'c'],
      derived: true,
      proof: {
        predicate: 'path',
        values: ['a', 'c'],
        rule: 2,
        because: [
          { predicate: 'edge', values: ['a', 'b'] },
          {
            predicate: 'path',
            values: ['b', 'c'],
            rule: 1,
            because: [{ predicate: 'edge', values: ['b', 'c'] }],
          },
        ],
      },
    });
  });

  it('enforces the proof node cap across materialized facts', () => {
    const db = `
      edge(a, b).
      edge(b, c).
    `;

    expect(() =>
      materializeWithProof(parseProgram(db), { maxProofNodes: 1 }),
    ).toThrow(EngineLimitError);
  });
});

describe('candidate visit budget', () => {
  // a rule whose body cross-joins wildcard goals derives few facts but visits
  // a Cartesian product of candidates; the budget stops it before it grinds
  const facts = ['learned', 'attended', 'wants', 'prefers']
    .flatMap((p) =>
      Array.from({ length: 60 }, (_, i) => `${p}(user, a${i}, b${i}, c${i}).`),
    )
    .join('\n');
  const program = `${facts}
    q(X) :- learned(user, X, _, _), attended(user, _, _, _), wants(user, _, _, _), prefers(user, _, _, _), learned(user, _, _, _), attended(user, _, _, _).`;

  it('throws EngineLimitError quickly when a rule body visits more candidates than allowed', () => {
    const started = Date.now();
    expect(() =>
      evaluateQuerySpec(parseProgram(program), parseQuerySpec('?- q(X).'), {
        maxCandidateVisits: 100_000,
      }),
    ).toThrow(EngineLimitError);
    expect(Date.now() - started).toBeLessThan(2_000);
  });

  it('has a finite default so no caller can hang the evaluator', () => {
    const started = Date.now();
    expect(() =>
      evaluateQuerySpec(parseProgram(program), parseQuerySpec('?- q(X).')),
    ).toThrow(EngineLimitError);
    expect(Date.now() - started).toBeLessThan(30_000);
  });
});
