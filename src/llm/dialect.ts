/**
 * The query dialect the writer adapter is trained on, shared by the training
 * data generator, the LongMemEval engine-recall harness and the product's
 * `dialect` recall variant. One card, one schema listing shape, so the model
 * sees at recall time exactly what it saw in training.
 */
import {
  isIntegrityConstraint,
  predKey,
  serializeClause,
  type Clause,
} from '../engine/index.js';

export const DIALECT_CARD = `Write ONE Datalog program that answers the question. Reply with ONLY the program.
- Rule shape: q(A, B) :- predicate(A), other(B, A).   The answer is the rule no other rule uses.
- Variables start uppercase (X, Person). Constants are lowercase exactly as listed.
- _ is a wildcard. \\+ predicate(X) means "no such fact". Comparisons: A != B, X = value.
- Any binary predicate p also answers p_plus(X, Y): Y is reachable from X in ONE OR MORE hops.
  Use p_plus for chains ("above", "below", "ultimately", "directly or transitively"). NEVER write recursive rules.
- Yes/no: ask the ground goal: ?- p_plus(x, target).   Answers yes = true or yes = false.
- Counting: count(*) as N where predicate(X, value)`;

const PLUMBING = /^(longmem_|rembero_)/;
const MAX_LISTING_PREDICATES = 120;
const MAX_SAMPLES = 3;

/**
 * A live store has no argument names, so each predicate is listed with
 * placeholders and up to three sample facts, most-used predicates first.
 * Plumbing and metadata predicates are left out.
 */
export function dialectSchemaListing(
  clauses: readonly Clause[],
  allowed?: ReadonlySet<string>,
): string {
  const groups = new Map<
    string,
    { arity: number; facts: string[]; count: number }
  >();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) continue;
    if (PLUMBING.test(clause.head.predicate)) continue;
    const key = predKey(clause.head);
    if (allowed !== undefined && !allowed.has(key)) continue;
    const entry = groups.get(key) ?? {
      arity: clause.head.args.length,
      facts: [],
      count: 0,
    };
    entry.count += 1;
    if (clause.body.length === 0 && entry.facts.length < MAX_SAMPLES) {
      entry.facts.push(serializeClause(clause));
    }
    groups.set(key, entry);
  }
  return [...groups.entries()]
    .sort((a, b) => b[1].count - a[1].count || a[0].localeCompare(b[0]))
    .slice(0, MAX_LISTING_PREDICATES)
    .map(([key, entry]) => {
      const name = key.slice(0, key.lastIndexOf('/'));
      const placeholders = Array.from(
        { length: entry.arity },
        (_, i) => `A${i + 1}`,
      );
      const examples =
        entry.facts.length > 0 ? `   e.g. ${entry.facts.join(' ')}` : '';
      return `${name}(${placeholders.join(', ')})${examples}`;
    })
    .join('\n');
}

export function dialectQuerySystemPrompt(
  clauses: readonly Clause[],
  allowed?: ReadonlySet<string>,
): string {
  return `You query a knowledge base with these predicates:\n${dialectSchemaListing(clauses, allowed)}\n\n${DIALECT_CARD}`;
}

/**
 * Small models mix the two dialects: a rule prefixed with `?- ` ("?- q(X) :- p(X)."),
 * or a bare query wrapped in code fences. Normalize before parsing.
 */
export function normalizeDialectResponse(response: string): string {
  const unfenced = response
    .replace(/^```[a-z]*\n?/gim, '')
    .replace(/```\s*$/gm, '')
    .trim();
  return unfenced
    .split('\n')
    .map((line) => {
      const trimmed = line.trim();
      return /^\?-\s*.+:-/.test(trimmed)
        ? trimmed.replace(/^\?-\s*/, '')
        : line;
    })
    .join('\n');
}
