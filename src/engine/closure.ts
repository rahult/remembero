/**
 * On-demand transitive closure predicates.
 *
 * Any binary predicate `p` also answers `p_plus(X, Y)`: Y is reachable from X
 * in one or more `p` hops. The two closure rules are synthesized only when a
 * program or query references `p_plus/2` without defining it, so authors
 * (including small language models) never write recursive rules for chains.
 * A user-defined `p_plus` always wins and suppresses synthesis.
 */
import {
  type Clause,
  type Goal,
  type Literal,
  type OrdinaryClause,
  type Term,
  isComparison,
  isNegation,
  predKey,
} from './ast.js';

export const CLOSURE_SUFFIX = '_plus';

/**
 * Decide whether `literal` is a closure reference the engine should satisfy
 * by synthesis, and if so return the base predicate name it closes over.
 *
 * `definedPredicates` holds the `predicate/arity` keys of every clause head
 * already in the program (facts and rules alike).
 *
 * Policy:
 *   - binary literals only (chains are edges);
 *   - the name must end in `_plus` with a non-empty base before the suffix;
 *   - a program that already defines `p_plus/2` itself wins;
 *   - the base may not itself be a closure name: `a_plus_plus` is refused so a
 *     doubled suffix (the likeliest small-model typo) yields no rows rather
 *     than silently working.
 */
export function closureBasePredicate(
  literal: Literal,
  definedPredicates: ReadonlySet<string>,
): string | undefined {
  if (literal.args.length !== 2) return undefined;
  if (!literal.predicate.endsWith(CLOSURE_SUFFIX)) return undefined;
  const base = literal.predicate.slice(0, -CLOSURE_SUFFIX.length);
  if (base.length === 0 || base.endsWith(CLOSURE_SUFFIX)) return undefined;
  if (definedPredicates.has(predKey(literal))) return undefined;
  return base;
}

/** The two rules that define `base_plus` as one-or-more hops of `base`. */
export function closureClauses(base: string): OrdinaryClause[] {
  const plus = `${base}${CLOSURE_SUFFIX}`;
  const X = { type: 'var', name: 'X' } as const;
  const Y = { type: 'var', name: 'Y' } as const;
  const Z = { type: 'var', name: 'Z' } as const;
  return [
    {
      head: { predicate: plus, args: [X, Y] },
      body: [{ predicate: base, args: [X, Y] }],
    },
    {
      head: { predicate: plus, args: [X, Y] },
      body: [
        { predicate: base, args: [X, Z] },
        { predicate: plus, args: [Z, Y] },
      ],
    },
  ];
}

/**
 * Seeded closure for a reference anchored at one end. With `c` in the first
 * position the rules derive only `base_plus(c, Y)`; with `c` second, only
 * `base_plus(X, c)`. The head keeps the closure predicate name, so proofs and
 * the rule catalog read exactly as for the full closure, while the derived
 * relation stays linear in the reachable slice instead of quadratic in the
 * graph.
 */
export function seededClosureClauses(
  base: string,
  anchor: Term,
  position: 0 | 1,
): OrdinaryClause[] {
  const plus = `${base}${CLOSURE_SUFFIX}`;
  const X = { type: 'var', name: 'X' } as const;
  const Y = { type: 'var', name: 'Y' } as const;
  const Z = { type: 'var', name: 'Z' } as const;
  if (position === 0) {
    return [
      {
        head: { predicate: plus, args: [anchor, Y] },
        body: [{ predicate: base, args: [anchor, Y] }],
      },
      {
        head: { predicate: plus, args: [anchor, Y] },
        body: [
          { predicate: plus, args: [anchor, Z] },
          { predicate: base, args: [Z, Y] },
        ],
      },
    ];
  }
  return [
    {
      head: { predicate: plus, args: [X, anchor] },
      body: [{ predicate: base, args: [X, anchor] }],
    },
    {
      head: { predicate: plus, args: [X, anchor] },
      body: [
        { predicate: base, args: [X, Z] },
        { predicate: plus, args: [Z, anchor] },
      ],
    },
  ];
}

function literalOf(goal: Goal): Literal | undefined {
  if (isComparison(goal)) return undefined;
  return isNegation(goal) ? goal.not : goal;
}

function isGroundTerm(term: Term): boolean {
  return term.type === 'atom' || term.type === 'num';
}

/**
 * Return `clauses` plus synthesized closure rules for every undefined
 * `p_plus/2` referenced in a rule body or in `goals`. References anchored at
 * one end get seeded rules for that anchor; references with two variables get
 * the full closure. Programs that reference no closure predicate are returned
 * unchanged (same array identity).
 */
export function expandClosurePredicates(
  clauses: Clause[],
  goals: readonly Goal[] = [],
): Clause[] {
  const defined = new Set(clauses.map((clause) => predKey(clause.head)));
  const full = new Set<string>();
  const seeded = new Map<
    string,
    { base: string; anchor: Term; position: 0 | 1 }
  >();
  const consider = (goal: Goal) => {
    const literal = literalOf(goal);
    if (literal === undefined) return;
    const base = closureBasePredicate(literal, defined);
    if (base === undefined) return;
    const [first, second] = literal.args;
    if (isGroundTerm(first)) {
      seeded.set(`${base}|0|${String((first as { value: unknown }).value)}`, {
        base,
        anchor: first,
        position: 0,
      });
    } else if (isGroundTerm(second)) {
      seeded.set(`${base}|1|${String((second as { value: unknown }).value)}`, {
        base,
        anchor: second,
        position: 1,
      });
    } else {
      full.add(base);
    }
  };
  for (const clause of clauses) for (const goal of clause.body) consider(goal);
  for (const goal of goals) consider(goal);
  if (full.size === 0 && seeded.size === 0) return clauses;
  const synthesized: Clause[] = [...full].sort().flatMap(closureClauses);
  for (const key of [...seeded.keys()].sort()) {
    const { base, anchor, position } = seeded.get(key)!;
    if (full.has(base)) continue; // the full closure already covers this anchor
    synthesized.push(...seededClosureClauses(base, anchor, position));
  }
  return [...clauses, ...synthesized];
}
