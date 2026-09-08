/**
 * Deterministic guards around LLM extraction. Each one removes an instruction a
 * small model gets wrong, by fixing the output locally or by rejecting it with
 * a message the retry loop can act on:
 *
 *   - normalizeExtractionOutput  list markers, missing periods, blank lines
 *   - rewriteSelfAtoms           "I" / "me" / "the_user" -> the configured self atom
 *   - assertGroundedConstants    every new constant must appear in the input text
 *   - predicate aliases          rembero_predicate_alias(from, to) rewrites predicates
 *   - assertKnownVocabulary      closed mode: unknown predicates are rejected
 *
 * Evidence for each: docs/research/EXTRACTION-BENCH.md.
 */
import {
  type Clause,
  type Goal,
  type Term,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  predKey,
} from '../engine/index.js';

// ---- output normalization ---------------------------------------------------------

const LIST_MARKER = /^\s*(?:[-*•]|\d+[.)])\s+/;

/**
 * Split a model reply into clause lines the parser can read: fences and list
 * markers removed, blank lines dropped, a period added to a line that ends in a
 * closing parenthesis. Prose lines pass through untouched so the parser still
 * rejects them and the retry loop reports the real problem.
 */
/** `... )` followed by whitespace and the start of another fact `name(`: split point between facts on one line. */
const FACT_BOUNDARY = /(?<=\))\.?\s+(?=[a-z][a-z0-9_]*\s*\()/g;

export function normalizeExtractionOutput(response: string): string[] {
  const lines: string[] = [];
  for (const raw of response.split('\n')) {
    // "assert fact." is a keyword small models invent by analogy with "retract"
    const stripped = raw
      .replace(LIST_MARKER, '')
      .trim()
      .replace(/^assert\s+/i, '');
    if (
      stripped === '' ||
      stripped === '```' ||
      /^```[a-zA-Z]*$/.test(stripped)
    )
      continue;
    // a prose label ("Here are the facts:", "Output:") has no parenthesis: drop it
    if (!stripped.includes('(') && !/^retract\b/.test(stripped)) continue;
    // several facts on one line without periods: split at fact boundaries,
    // except inside rules (":-"), which legitimately chain literals
    const pieces = stripped.includes(':-')
      ? [stripped]
      : stripped.split(FACT_BOUNDARY);
    for (let piece of pieces) {
      piece = piece.trim();
      if (piece === '') continue;
      if (/\)\s*$/.test(piece) && !/\.\s*$/.test(piece)) piece = `${piece}.`;
      if (/^retract\s+/.test(piece) && !/\.\s*$/.test(piece))
        piece = `${piece}.`;
      lines.push(piece);
    }
  }
  return lines;
}

// ---- self atom -------------------------------------------------------------------

/** Atoms models produce for the speaker; rewritten to the configured self atom. */
export const SELF_ATOM_SYNONYMS: readonly string[] = [
  'i',
  'me',
  'my',
  'myself',
  'self',
  'user',
  'the_user',
  'you',
  'speaker',
];

function rewriteTerm(
  term: Term,
  selfAtom: string,
  synonyms: ReadonlySet<string>,
): Term {
  if (
    term.type === 'atom' &&
    synonyms.has(term.value) &&
    term.value !== selfAtom
  ) {
    return { type: 'atom', value: selfAtom };
  }
  return term;
}

function rewriteGoal(
  goal: Goal,
  selfAtom: string,
  synonyms: ReadonlySet<string>,
): Goal {
  if (isComparison(goal)) return goal;
  if (isNegation(goal)) {
    return {
      ...goal,
      not: {
        ...goal.not,
        args: goal.not.args.map((t) => rewriteTerm(t, selfAtom, synonyms)),
      },
    };
  }
  return {
    ...goal,
    args: goal.args.map((t) => rewriteTerm(t, selfAtom, synonyms)),
  };
}

/** Rewrite bare first-person atoms to `selfAtom`. Quoted names such as 'Me' are atoms with a capital and are left alone. */
export function rewriteSelfAtoms(
  clauses: Clause[],
  selfAtom: string,
): Clause[] {
  const synonyms = new Set(SELF_ATOM_SYNONYMS);
  return clauses.map((clause) => {
    if (isIntegrityConstraint(clause)) return clause;
    return {
      ...clause,
      head: {
        ...clause.head,
        args: clause.head.args.map((t) => rewriteTerm(t, selfAtom, synonyms)),
      },
      body: clause.body.map((goal) => rewriteGoal(goal, selfAtom, synonyms)),
    } as Clause;
  });
}

/** Rewrite self atoms inside retraction patterns (goal lists). */
export function rewriteSelfAtomsInGoals(
  patterns: Goal[][],
  selfAtom: string,
): Goal[][] {
  const synonyms = new Set(SELF_ATOM_SYNONYMS);
  return patterns.map((goals) =>
    goals.map((goal) => rewriteGoal(goal, selfAtom, synonyms)),
  );
}

// ---- constant grounding ------------------------------------------------------------

function looseTokens(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .split(' ')
      .filter((token) => token.length > 0),
  );
}

/** A constant is grounded when each of its word parts appears in the input, joined or separately. */
function constantGrounded(
  value: string,
  input: string,
  tokens: ReadonlySet<string>,
): boolean {
  const lowered = value.toLowerCase();
  const compact = lowered.replace(/[^a-z0-9]+/g, '');
  const inputCompact = input.toLowerCase().replace(/[^a-z0-9]+/g, '');
  if (compact.length > 0 && inputCompact.includes(compact)) return true;
  const parts = lowered.split(/[^a-z0-9]+/).filter((part) => part.length > 0);
  return parts.length > 0 && parts.every((part) => tokens.has(part));
}

export interface GroundingOptions {
  /** Constants already in the store (values the model may legitimately reuse). */
  known: ReadonlySet<string>;
  selfAtom: string;
}

/**
 * Every constant in an added fact must appear in the input (loosely: case,
 * quotes, underscores and hyphens ignored), already exist in the store, or be
 * the self atom. Numbers not present in the input are rejected too: a model
 * that turns "turns 40 in 2027" into birth_year 1987 is inferring, not storing.
 */
export function assertGroundedConstants(
  clauses: Clause[],
  input: string,
  options: GroundingOptions,
): void {
  const tokens = looseTokens(input);
  const offenders: string[] = [];
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause) || clause.body.length > 0) continue;
    for (const term of clause.head.args) {
      if (term.type !== 'atom' && term.type !== 'num') continue;
      const value = String(term.value);
      if (value === options.selfAtom || options.known.has(value)) continue;
      if (!constantGrounded(value, input, tokens)) offenders.push(value);
    }
  }
  if (offenders.length > 0) {
    const unique = [...new Set(offenders)];
    throw new Error(
      `constant${unique.length > 1 ? 's' : ''} ${unique.map((v) => `'${v}'`).join(', ')} ${unique.length > 1 ? 'are' : 'is'} not in the input; store only values the text states, do not infer or rename them`,
    );
  }
}

// ---- vocabulary -------------------------------------------------------------------------

export const PREDICATE_ALIAS_PREDICATE = 'rembero_predicate_alias';

/** `rembero_predicate_alias(from, to).` declarations in the store, as from -> to. */
export function predicateAliasesFrom(
  clauses: readonly Clause[],
): Map<string, string> {
  const aliases = new Map<string, string>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause) || clause.body.length > 0) continue;
    if (
      clause.head.predicate !== PREDICATE_ALIAS_PREDICATE ||
      clause.head.args.length !== 2
    )
      continue;
    const [from, to] = clause.head.args;
    if (from.type === 'atom' && to.type === 'atom')
      aliases.set(from.value, to.value);
  }
  return aliases;
}

export function applyPredicateAliases(
  clauses: Clause[],
  aliases: ReadonlyMap<string, string>,
): Clause[] {
  if (aliases.size === 0) return clauses;
  const rename = (name: string) => aliases.get(name) ?? name;
  return clauses.map((clause) => {
    if (isIntegrityConstraint(clause)) return clause;
    return {
      ...clause,
      head: { ...clause.head, predicate: rename(clause.head.predicate) },
      body: clause.body.map((goal) => {
        if (isComparison(goal)) return goal;
        if (isNegation(goal))
          return {
            ...goal,
            not: { ...goal.not, predicate: rename(goal.not.predicate) },
          };
        return { ...goal, predicate: rename(goal.predicate) };
      }),
    } as Clause;
  });
}

export function applyPredicateAliasesToGoals(
  patterns: Goal[][],
  aliases: ReadonlyMap<string, string>,
): Goal[][] {
  if (aliases.size === 0) return patterns;
  const rename = (name: string) => aliases.get(name) ?? name;
  return patterns.map((goals) =>
    goals.map((goal) => {
      if (isComparison(goal)) return goal;
      if (isNegation(goal))
        return {
          ...goal,
          not: { ...goal.not, predicate: rename(goal.not.predicate) },
        };
      return { ...goal, predicate: rename(goal.predicate) };
    }),
  );
}

/** Closed vocabulary: every added fact's predicate/arity must already be in the schema. */
export function assertKnownVocabulary(
  clauses: Clause[],
  known: ReadonlySet<string>,
): void {
  const unknown = new Set<string>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) continue;
    const key = predKey(clause.head);
    if (!known.has(key)) unknown.add(key);
  }
  if (unknown.size > 0) {
    const shown = [...known].sort().slice(0, 40);
    throw new Error(
      `unknown predicate${unknown.size > 1 ? 's' : ''} ${[...unknown].join(', ')}; the vocabulary is closed. Use one of: ${shown.join(', ')}${known.size > shown.length ? ', ...' : ''}`,
    );
  }
}
