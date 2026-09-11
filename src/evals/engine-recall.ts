/**
 * Engine recall for the LongMemEval harness: the writer model authors a Datalog
 * program in its training dialect over the facts remembered from a question's
 * whole history, the engine runs it, and the rows (with the dates of the facts
 * that produced them) go to the reader ahead of the chats.
 *
 * This is the moonshot composition measured in isolation: counting, chaining and
 * latest-value questions answered by the engine from extracted facts, the reader
 * left to read. The prompt is the one the writer was trained on (the same card
 * the query training data uses), with the store's predicates listed with
 * argument placeholders and up to three sample facts each, since a live store
 * has no argument names.
 */
import {
  canonicalKey,
  emptyResultFeedback,
  evaluateQuerySpec,
  goalVariables,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  parseQueryProgram,
  predKey,
  serializeClause,
  serializeTerm,
  type Bindings,
  type Clause,
  type Goal,
  type Literal,
  type QuerySpec,
} from '../engine/index.js';
import type { ChatMessage } from '../llm/client.js';
import type { MemoryStore, MemorySource } from '../store/store.js';
import { DIALECT_CARD, dialectSchemaListing } from '../llm/dialect.js';

export interface EngineRecallOutcome {
  status: 'answered' | 'empty' | 'unparsable' | 'error';
  program: string | null;
  rows: number;
  /** Rendered rows with the dates of the facts behind them; empty unless answered. */
  rendered: string;
  attempts: number;
  error?: string;
}

const MAX_ROWS_RENDERED = 40;

export { dialectSchemaListing };

export function engineRecallMessages(
  clauses: readonly Clause[],
  question: string,
): ChatMessage[] {
  return [
    {
      role: 'system',
      content: `You query a knowledge base with these predicates:\n${dialectSchemaListing(clauses)}\n\n${DIALECT_CARD}`,
    },
    { role: 'user', content: question },
  ];
}

const MAX_CROSS_PRODUCT = 200_000;

/**
 * Upper bound on the rows a goal list can produce before any join narrows it:
 * each positive goal that shares no variable with the goals before it
 * multiplies by its relation's clause count. A model that lists relations
 * side by side with wildcards writes a Cartesian product the evaluator would
 * grind through; this catches it before evaluation.
 */
export function crossProductEstimate(
  clauses: readonly Clause[],
  query: QuerySpec,
  authored: readonly Clause[],
): number {
  const counts = new Map<string, number>();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) continue;
    const key = predKey(clause.head);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  // the body to estimate: the query's goals, or the sink rule's body when the
  // query just asks for an authored head
  let goals: Goal[] = query.goals;
  const first = goals.find(
    (g): g is Literal => !isComparison(g) && !isNegation(g),
  );
  if (first !== undefined && goals.length === 1) {
    const rule = authored.find(
      (c) => !isIntegrityConstraint(c) && predKey(c.head) === predKey(first),
    );
    if (rule !== undefined && rule.body.length > 0) goals = rule.body;
  }
  let estimate = 1;
  const bound = new Set<string>();
  for (const goal of goals) {
    if (isComparison(goal) || isNegation(goal)) continue;
    const vars = goalVariables(goal);
    const shares = [...vars].some((v) => bound.has(v));
    if (!shares) estimate *= Math.max(1, counts.get(predKey(goal)) ?? 1);
    for (const v of vars) bound.add(v);
  }
  return estimate;
}

function stripFences(text: string): string {
  return text
    .replace(/^```[a-z]*\n?/gim, '')
    .replace(/```\s*$/gm, '')
    .trim();
}

/** Rows as `X = a, Y = b` lines, each followed by the dates of the facts that mention its constants. */
export function renderEngineRows(
  rows: readonly Bindings[],
  clauses: readonly Clause[],
  sources: Map<string, MemorySource[]>,
): string {
  const dateOf = new Map<string, Set<string>>();
  for (const clause of clauses) {
    if (clause.body.length > 0 || isIntegrityConstraint(clause)) continue;
    const ts = sources.get(canonicalKey(clause))?.[0]?.ts;
    if (ts === undefined) continue;
    for (const term of clause.head.args) {
      if (term.type !== 'atom' && term.type !== 'num') continue;
      const value = String(term.value).toLowerCase();
      const set = dateOf.get(value) ?? new Set<string>();
      set.add(ts.slice(0, 10));
      dateOf.set(value, set);
    }
  }
  const lines = rows.slice(0, MAX_ROWS_RENDERED).map((row) => {
    const parts = Object.entries(row).map(
      ([name, term]) => `${name} = ${serializeTerm(term)}`,
    );
    const dates = new Set<string>();
    for (const term of Object.values(row)) {
      if (term.type !== 'atom' && term.type !== 'num') continue;
      for (const d of dateOf.get(String(term.value).toLowerCase()) ?? [])
        dates.add(d);
    }
    const when =
      dates.size > 0 ? `   (stated ${[...dates].sort().join(', ')})` : '';
    return `- ${parts.length > 0 ? parts.join(', ') : 'yes'}${when}`;
  });
  const more =
    rows.length > MAX_ROWS_RENDERED
      ? `\n- … ${rows.length - MAX_ROWS_RENDERED} more rows`
      : '';
  return `Rows (${rows.length}):\n${lines.join('\n')}${more}`;
}

const EVALUATE_BOUNDS = {
  // the same bounds the SQLite bridge applies: a cross product of wildcard
  // goals must fail fast, not grind the evaluator
  maxFacts: 50_000,
  maxIterations: 1_000,
  maxRows: 10_001,
  maxAggregateRows: 10_000,
};

/**
 * Run a parsed program and render its result. A count is rendered with the
 * items it counted (the reader can verify a list, not a bare number) and a
 * count of nothing is an empty result, not a row that says zero.
 */
export function runProgram(
  clauses: readonly Clause[],
  sources: Map<string, MemorySource[]>,
  normalized: ReturnType<typeof parseQueryProgram>,
): { rows: number; rendered: string } {
  const all = [...clauses, ...normalized.clauses];
  const query = normalized.query;
  if (query.kind === 'aggregate' && query.op === 'count') {
    const items = evaluateQuerySpec(
      all,
      { kind: 'relational', goals: query.goals },
      EVALUATE_BOUNDS,
    );
    if (items.length === 0) return { rows: 0, rendered: '' };
    const rendered = renderEngineRows(items, clauses, sources).replace(
      /^Rows \((\d+)\):/,
      (_, n: string) => `Count: ${n}. The counted items:`,
    );
    return { rows: items.length, rendered };
  }
  const rows = evaluateQuerySpec(all, query, EVALUATE_BOUNDS);
  if (rows.length === 0) return { rows: 0, rendered: '' };
  return {
    rows: rows.length,
    rendered: renderEngineRows(rows, clauses, sources),
  };
}

/**
 * Author, validate, run, and on an empty or unparsable result give the writer one
 * repair turn with the engine's feedback.
 */
export async function engineRecall(
  store: MemoryStore,
  namespace: string,
  question: string,
  writer: { complete(messages: ChatMessage[]): Promise<string> },
): Promise<EngineRecallOutcome> {
  const clauses = store.clausesFor([namespace]);
  const sources = store.sourcesFor([namespace]);
  const messages = engineRecallMessages(clauses, question);
  let program: string | null = null;
  let attempts = 0;
  let lastError: string | undefined;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    attempts += 1;
    const response = stripFences(await writer.complete(messages));
    let normalized;
    try {
      normalized = parseQueryProgram(response);
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: `That program failed to parse: ${lastError}. Reply with ONLY a corrected program.`,
      });
      continue;
    }
    program = response;
    const estimate = crossProductEstimate(
      clauses,
      normalized.query,
      normalized.clauses,
    );
    if (estimate > MAX_CROSS_PRODUCT) {
      lastError = `the goals form a cross product of about ${estimate.toExponential(1)} rows`;
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: `That program was refused: ${lastError}. Goals must share variables so the engine can join them, and one relation is usually enough for a count. Reply with ONLY a corrected program.`,
      });
      continue;
    }
    let result: { rows: number; rendered: string };
    try {
      result = runProgram(clauses, sources, normalized);
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: `That program failed with: ${lastError}. Reply with ONLY a corrected program.`,
      });
      continue;
    }
    if (result.rows > 0) {
      return {
        status: 'answered',
        program,
        rows: result.rows,
        rendered: result.rendered,
        attempts,
      };
    }
    if (attempt === 0) {
      const feedback = emptyResultFeedback(
        [...clauses, ...normalized.clauses],
        normalized.query,
        {
          authored: normalized.clauses,
        },
      );
      if (feedback.length === 0) break;
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: `That query ran but returned no rows. ${feedback} If the query correctly expresses the question, repeat it unchanged; otherwise reply with ONLY a corrected query.`,
      });
      continue;
    }
  }
  const error = lastError === undefined ? {} : { error: lastError };
  if (program === null) {
    return {
      status: 'unparsable',
      program: null,
      rows: 0,
      rendered: '',
      attempts,
      ...error,
    };
  }
  return {
    status: 'empty',
    program,
    rows: 0,
    rendered: '',
    attempts,
    ...error,
  };
}
