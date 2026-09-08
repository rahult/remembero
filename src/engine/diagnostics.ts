/**
 * Query diagnostics: turn the engine's silent empties into messages a model
 * (or a person) can act on.
 *
 * The evaluator answers an unknown predicate, a wrong arity, a reversed
 * argument order or a capitalized constant with `[]` or a plausible superset.
 * For a small model authoring queries, "empty" and "wrong" are then the same
 * thing, and the yes/no idiom reads empty as "no". These checks run over the
 * query goals against the clauses in view and report:
 *
 *   - warning unknown_predicate       (empty result, positive goal only) no facts, no rules, not a closure
 *   - error   closure_arity           p_plus with arity != 2
 *   - error   closure_double_suffix   a_plus_plus
 *   - error   capitalized_constant    singleton variable whose lowercase form is a known atom in that slot
 *   - warning direction               (empty result only) a constant never seen in that argument
 *                                     position but seen in another
 */
import {
  type Clause,
  type Goal,
  type Literal,
  type QuerySpec,
  type Term,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  predKey,
} from './ast.js';
import { CLOSURE_SUFFIX } from './closure.js';

export interface QueryDiagnostic {
  severity: 'error' | 'warning';
  code:
    | 'unknown_predicate'
    | 'closure_arity'
    | 'closure_double_suffix'
    | 'capitalized_constant'
    | 'wildcard_only'
    | 'direction';
  message: string;
}

export interface DiagnoseOptions {
  /** True when the query returned no rows; enables the direction hint. */
  emptyResult?: boolean;
  /** Model-authored rules whose bodies should be checked too (each rule is one variable scope). */
  authored?: Clause[];
  /** Extra predicate/arity keys the caller knows exist (e.g. empty SQLite tables). */
  knownPredicates?: Iterable<string>;
}

interface Known {
  /** predicate/arity keys of every fact and rule head. */
  keys: Set<string>;
  /** predicate -> arities seen */
  arities: Map<string, Set<number>>;
  /** predicate/arity -> per-argument-position set of atom values (facts only). */
  slots: Map<string, Array<Set<string>>>;
}

function collectKnown(clauses: Clause[]): Known {
  const keys = new Set<string>();
  const arities = new Map<string, Set<number>>();
  const slots = new Map<string, Array<Set<string>>>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) continue;
    const key = predKey(clause.head);
    keys.add(key);
    const set = arities.get(clause.head.predicate) ?? new Set<number>();
    set.add(clause.head.args.length);
    arities.set(clause.head.predicate, set);
    if (clause.body.length === 0) {
      const positions =
        slots.get(key) ?? clause.head.args.map(() => new Set<string>());
      clause.head.args.forEach((term, index) => {
        if (term.type === 'atom' || term.type === 'num') {
          positions[index].add(String(term.value));
        }
      });
      slots.set(key, positions);
    }
  }
  return { keys, arities, slots };
}

interface ScopedLiteral {
  literal: Literal;
  negated: boolean;
}

function literalsOf(goals: readonly Goal[]): ScopedLiteral[] {
  const literals: ScopedLiteral[] = [];
  for (const goal of goals) {
    if (isComparison(goal)) continue;
    literals.push(
      isNegation(goal)
        ? { literal: goal.not, negated: true }
        : { literal: goal, negated: false },
    );
  }
  return literals;
}

/** A variable scope: the literals to check plus every place a variable occurs in that scope. */
interface Scope {
  literals: ScopedLiteral[];
  occurrences: Map<string, number>;
}

function scopeOf(
  literals: ScopedLiteral[],
  extraTerms: readonly Term[] = [],
): Scope {
  const occurrences = new Map<string, number>();
  const count = (term: Term) => {
    if (term.type === 'var') {
      occurrences.set(term.name, (occurrences.get(term.name) ?? 0) + 1);
    }
  };
  for (const { literal } of literals) literal.args.forEach(count);
  extraTerms.forEach(count);
  return { literals, occurrences };
}

function editDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [
    i,
    ...Array<number>(b.length).fill(0),
  ]);
  for (let j = 1; j <= b.length; j += 1) dp[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
  }
  return dp[a.length][b.length];
}

function nearMisses(name: string, known: Known): string[] {
  const candidates = new Set<string>();
  for (const [predicate, arities] of known.arities) {
    const close =
      predicate === name ||
      editDistance(predicate, name) <= 2 ||
      predicate.startsWith(name) ||
      name.startsWith(predicate);
    if (!close) continue;
    for (const arity of arities) candidates.add(`${predicate}/${arity}`);
  }
  return [...candidates].sort().slice(0, 5);
}

/** Base predicate for a closure reference, with the validation the engine applies. */
function closureBase(literal: Literal): {
  base?: string;
  error?: QueryDiagnostic;
} {
  const { predicate, args } = literal;
  if (!predicate.endsWith(CLOSURE_SUFFIX)) return {};
  const base = predicate.slice(0, -CLOSURE_SUFFIX.length);
  if (base.endsWith(CLOSURE_SUFFIX)) {
    return {
      error: {
        severity: 'error',
        code: 'closure_double_suffix',
        message: `${predicate}/${args.length}: the closure suffix appears twice. Did you mean ${base}(...)?`,
      },
    };
  }
  if (args.length !== 2) {
    return {
      error: {
        severity: 'error',
        code: 'closure_arity',
        message: `${predicate}/${args.length}: closure predicates are binary; ${base} answers ${base}_plus(X, Y) only.`,
      },
    };
  }
  return { base };
}

function renderCall(
  literal: Literal,
  replaceIndex: number,
  replacement: string,
): string {
  const args = literal.args.map((term, index) =>
    index === replaceIndex ? replacement : termText(term),
  );
  return `${literal.predicate}(${args.join(', ')})`;
}

function termText(term: Term): string {
  switch (term.type) {
    case 'var':
      return term.name;
    case 'wildcard':
      return '_';
    default:
      return String(term.value);
  }
}

export function diagnoseQuery(
  clauses: Clause[],
  query: QuerySpec,
  options: DiagnoseOptions = {},
): QueryDiagnostic[] {
  const known = collectKnown(clauses);
  for (const key of options.knownPredicates ?? []) known.keys.add(key);
  const out: QueryDiagnostic[] = [];
  const seen = new Set<string>();
  const push = (diagnostic: QueryDiagnostic) => {
    const key = `${diagnostic.code}|${diagnostic.message}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(diagnostic);
  };

  // A relational query whose only variables are wildcards answers with rows
  // that carry no values: neither a readable answer nor a yes/no question.
  if (query.kind === 'relational') {
    const goals = literalsOf(query.goals);
    const hasNamed = goals.some(({ literal }) =>
      literal.args.some((t) => t.type === 'var'),
    );
    const wildcardOnly =
      !hasNamed &&
      goals.some(({ literal }) =>
        literal.args.some((t) => t.type === 'wildcard'),
      );
    if (wildcardOnly) {
      const first = goals.find(({ literal }) =>
        literal.args.some((t) => t.type === 'wildcard'),
      )!.literal;
      let n = 0;
      const fixed = `${first.predicate}(${first.args
        .map((t) =>
          t.type === 'wildcard' ? (n++ === 0 ? 'X' : `X${n}`) : termText(t),
        )
        .join(', ')})`;
      push({
        severity: 'error',
        code: 'wildcard_only',
        message: `the query has no named variable to return; _ discards the value. Use a named variable, e.g. ${fixed}`,
      });
    }
  }

  // The query goals are one scope; each authored rule body is another, with the
  // rule's head variables counted so a head-and-body variable is not a singleton.
  const scopes: Scope[] = [scopeOf(literalsOf(query.goals))];
  for (const clause of options.authored ?? []) {
    if (isIntegrityConstraint(clause) || clause.body.length === 0) continue;
    scopes.push(scopeOf(literalsOf(clause.body), clause.head.args));
  }

  for (const scope of scopes) {
    for (const { literal, negated } of scope.literals) {
      const key = predKey(literal);
      const closure = closureBase(literal);
      if (closure.error) {
        push(closure.error);
        continue;
      }
      // the relation whose facts we can inspect: the literal itself, or a closure's base
      const factKey = closure.base ? `${closure.base}/2` : key;
      const isClosureOfKnownBase =
        closure.base !== undefined && known.keys.has(`${closure.base}/2`);
      if (!known.keys.has(key) && !isClosureOfKnownBase) {
        // A positive goal over an unknown predicate is an honest closed-world
        // "no", so this is a hint on an empty result, not an error; a negated
        // unknown ("nothing is suspended") is the normal case and stays silent.
        if (options.emptyResult === true && !negated) {
          const misses = nearMisses(closure.base ?? literal.predicate, known);
          push({
            severity: 'warning',
            code: 'unknown_predicate',
            message:
              `unknown predicate ${key}` +
              (misses.length > 0
                ? `; known: ${misses.join(', ')}`
                : '; no similar predicate is stored'),
          });
        }
        continue;
      }
      const positions = known.slots.get(factKey);
      if (!positions) continue;

      literal.args.forEach((term, index) => {
        if (term.type === 'var' && scope.occurrences.get(term.name) === 1) {
          const lowered = term.name.toLowerCase();
          if (positions[index]?.has(lowered)) {
            push({
              severity: 'error',
              code: 'capitalized_constant',
              message:
                `'${term.name}' is a variable (uppercase), but the constant ${lowered} exists in ` +
                `argument ${index + 1} of ${factKey}. Did you mean ${renderCall(literal, index, lowered)}?`,
            });
          }
        }
        if (
          options.emptyResult === true &&
          (term.type === 'atom' || term.type === 'num')
        ) {
          const value = String(term.value);
          if (positions[index]?.has(value)) return;
          const elsewhere = positions.findIndex(
            (set, j) => j !== index && set.has(value),
          );
          if (elsewhere < 0) return;
          const swapped = literal.args.map((t, j) =>
            j === index
              ? termText(literal.args[elsewhere])
              : j === elsewhere
                ? value
                : termText(t),
          );
          push({
            severity: 'warning',
            code: 'direction',
            message:
              `${value} never appears in argument ${index + 1} of ${factKey}; it appears in ` +
              `argument ${elsewhere + 1}. Did you mean ${literal.predicate}(${swapped.join(', ')})?`,
          });
        }
      });
    }
  }
  return out;
}
