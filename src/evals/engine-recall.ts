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
  isIntegrityConstraint,
  parseQueryProgram,
  predKey,
  serializeClause,
  serializeTerm,
  type Bindings,
  type Clause,
} from '../engine/index.js';
import type { ChatMessage } from '../llm/client.js';
import type { MemoryStore, MemorySource } from '../store/store.js';
import { DIALECT_CARD } from '../training/export.js';

export interface EngineRecallOutcome {
  status: 'answered' | 'empty' | 'unparsable' | 'error';
  program: string | null;
  rows: number;
  /** Rendered rows with the dates of the facts behind them; empty unless answered. */
  rendered: string;
  attempts: number;
  error?: string;
}

const PLUMBING = /^(longmem_|rembero_)/;
const MAX_LISTING_PREDICATES = 120;
const MAX_SAMPLES = 3;
const MAX_ROWS_RENDERED = 40;

/** The store's predicates as `name(A1, A2)` lines with sample facts, plumbing excluded. */
export function dialectSchemaListing(clauses: readonly Clause[]): string {
  const samples = new Map<
    string,
    { arity: number; facts: string[]; count: number }
  >();
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause)) continue;
    if (PLUMBING.test(clause.head.predicate)) continue;
    const key = predKey(clause.head);
    const entry = samples.get(key) ?? {
      arity: clause.head.args.length,
      facts: [],
      count: 0,
    };
    entry.count += 1;
    if (clause.body.length === 0 && entry.facts.length < MAX_SAMPLES) {
      entry.facts.push(serializeClause(clause));
    }
    samples.set(key, entry);
  }
  return [...samples.entries()]
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
    let rows: Bindings[];
    try {
      rows = evaluateQuerySpec(
        [...clauses, ...normalized.clauses],
        normalized.query,
        // the same bounds the SQLite bridge applies: a cross product of wildcard
        // goals must fail fast, not grind the evaluator
        {
          maxFacts: 50_000,
          maxIterations: 1_000,
          maxRows: 10_001,
          maxAggregateRows: 10_000,
        },
      );
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: `That program failed with: ${lastError}. Reply with ONLY a corrected program.`,
      });
      continue;
    }
    if (rows.length > 0) {
      return {
        status: 'answered',
        program,
        rows: rows.length,
        rendered: renderEngineRows(rows, clauses, sources),
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
