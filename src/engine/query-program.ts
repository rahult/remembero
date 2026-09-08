/**
 * One normalizer for every query surface (MCP `query`, CLI, SQLite bridge).
 *
 * Accepts three spellings and turns each into authored clauses plus a query:
 *   - a goal list, with or without `?-` and a trailing period:
 *       `reports_to(maya, M)`   `?- reports_to(maya, M).`
 *   - an aggregate or projected query spec:
 *       `count(*) as N where reports_to(X, Y)`
 *   - a rule program. The query target is the unique *sink* rule — the head no
 *     other rule's body references — because authors (human and model) write
 *     helpers first and the answer last. An explicit `?- goal.` as the final
 *     line overrides; more than one sink is an error naming the fix.
 *
 * A query with no variables is `ground`: its answer is one boolean row rather
 * than the unreadable `[{}]` / `[]` pair.
 */
import {
  type Clause,
  type Goal,
  type QuerySpec,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  predKey,
  serializeTerm,
} from './ast.js';
import { ParseError } from './lexer.js';
import { parseProgram, parseQuerySpec } from './parser.js';

export interface QueryProgram {
  /** Authored rules to evaluate alongside the knowledge base (may be empty). */
  clauses: Clause[];
  query: QuerySpec;
  /** No variables in the query: answer as a single boolean row. */
  ground: boolean;
  /** How the target was chosen. */
  target: 'goals' | 'explicit' | 'sink';
}

function isGround(query: QuerySpec): boolean {
  if (query.kind !== 'relational') return false;
  return query.goals.every((goal) => {
    if (isComparison(goal)) return true;
    const literal = isNegation(goal) ? goal.not : goal;
    return literal.args.every(
      (term) => term.type === 'atom' || term.type === 'num',
    );
  });
}

/** Split `rules... ?- goal.` into its two halves; the `?-` line must come last. */
function splitExplicitQuery(
  input: string,
): { program: string; query: string } | undefined {
  const index = input.indexOf('?-');
  if (index < 0) return undefined;
  return { program: input.slice(0, index), query: input.slice(index) };
}

function sinkTarget(clauses: Clause[]): Goal[] {
  const rules = clauses.filter(
    (clause) => !isIntegrityConstraint(clause) && clause.body.length > 0,
  );
  if (rules.length === 0) {
    throw new ParseError(
      clauses.some(isIntegrityConstraint)
        ? 'an integrity constraint is a policy, not a query'
        : 'a program of facts is not a query; ask a goal such as p(X) or write a rule',
    );
  }
  // A predicate referenced only by its own rules (direct recursion) is still a
  // sink; only references from *other* rules disqualify it.
  const referenced = new Set<string>();
  for (const rule of rules) {
    const own = predKey(rule.head);
    for (const goal of rule.body) {
      if (isComparison(goal)) continue;
      const key = predKey(isNegation(goal) ? goal.not : goal);
      if (key !== own) referenced.add(key);
    }
  }
  const sinks = new Map<string, Clause>();
  for (const rule of rules) {
    const key = predKey(rule.head);
    if (!referenced.has(key) && !sinks.has(key)) sinks.set(key, rule);
  }
  if (sinks.size === 1) {
    const [rule] = sinks.values();
    return [rule.head];
  }
  if (sinks.size === 0) {
    throw new ParseError(
      'every rule feeds another rule, so no answer predicate stands out; add a final ?- goal. line',
    );
  }
  const names = [...sinks.keys()];
  const [firstKey, firstRule] = [...sinks.entries()][0];
  const example = `?- ${firstRule.head.predicate}(${firstRule.head.args
    .map((term) => (term.type === 'var' ? term.name : serializeTerm(term)))
    .join(', ')}).`;
  void firstKey;
  throw new ParseError(
    `program derives ${names.slice(0, -1).join(', ')} and ${names.at(-1)}; ` +
      `add a final line to choose the answer, e.g. ${example}`,
  );
}

export function parseQueryProgram(input: string): QueryProgram {
  const trimmed = input.trim();
  if (trimmed.length === 0) throw new ParseError('empty query');
  const hasRule = trimmed.includes(':-');
  const explicit = hasRule ? splitExplicitQuery(trimmed) : undefined;

  if (!hasRule) {
    const query = parseQuerySpec(trimmed);
    return { clauses: [], query, ground: isGround(query), target: 'goals' };
  }

  if (explicit !== undefined) {
    const clauses = parseProgram(explicit.program);
    const query = parseQuerySpec(explicit.query);
    return { clauses, query, ground: isGround(query), target: 'explicit' };
  }

  const clauses = parseProgram(trimmed);
  const goals = sinkTarget(clauses);
  const query: QuerySpec = { kind: 'relational', goals };
  return { clauses, query, ground: isGround(query), target: 'sink' };
}
