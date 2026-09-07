/**
 * Execution verification for training candidates. A candidate becomes an
 * Example only if its program runs on its world through the real engine and
 * the answer has the shape the template promised. Because the engine is
 * deterministic, every kept example is a zero-noise label.
 */
import {
  type Bindings,
  type Clause,
  evaluateQuerySpec,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  parseProgram,
  parseQuerySpec,
  serializeTerm,
} from '../engine/index.js';
import type { Candidate } from './templates.js';
import { worldClauses, type World } from './worlds.js';

export interface Example extends Candidate {
  /** Sorted "X=a Y=b" rows; empty for yes/no-false examples. */
  answer: string[];
}

export interface Rejection {
  candidate: Candidate;
  reason: string;
}

export const MAX_ANSWER_ROWS = 30;

export function isRejection(value: Example | Rejection): value is Rejection {
  return 'reason' in value;
}

/** Rule programs answer their first head; bare queries (including aggregates) run as query specs. */
export function executeProgram(clauses: Clause[], program: string): Bindings[] {
  const options = { maxRows: MAX_ANSWER_ROWS + 1 };
  if (program.includes(':-')) {
    const rules = parseProgram(program);
    const head = rules[0]?.head;
    if (!head) throw new Error('program has no rule');
    return evaluateQuerySpec(
      [...clauses, ...rules],
      { kind: 'relational', goals: [head] },
      options,
    );
  }
  return evaluateQuerySpec(clauses, parseQuerySpec(program), options);
}

/** Throw when any authored rule reaches its own head through the authored rules. */
export function assertRecursionFree(program: string): void {
  if (!program.includes(':-')) return;
  const rules = parseProgram(program).filter((c) => !isIntegrityConstraint(c));
  const deps = new Map<string, Set<string>>();
  for (const rule of rules) {
    const set = deps.get(rule.head.predicate) ?? new Set<string>();
    for (const goal of rule.body) {
      if (isComparison(goal)) continue;
      set.add(isNegation(goal) ? goal.not.predicate : goal.predicate);
    }
    deps.set(rule.head.predicate, set);
  }
  const visiting = new Set<string>();
  const done = new Set<string>();
  const visit = (node: string, path: string[]) => {
    if (done.has(node)) return;
    if (visiting.has(node)) {
      throw new Error(`recursive rule: ${[...path, node].join(' -> ')}`);
    }
    visiting.add(node);
    for (const next of deps.get(node) ?? []) {
      if (deps.has(next)) visit(next, [...path, node]);
    }
    visiting.delete(node);
    done.add(node);
  };
  for (const head of deps.keys()) visit(head, []);
}

function rowStrings(bindings: Bindings[]): string[] {
  return bindings
    .map((b) =>
      Object.entries(b)
        .map(([k, t]) => `${k}=${serializeTerm(t)}`)
        .sort()
        .join(' '),
    )
    .sort();
}

export function verifyCandidate(
  world: World,
  candidate: Candidate,
): Example | Rejection {
  const reject = (reason: string): Rejection => ({ candidate, reason });
  const clauses = worldClauses(world);
  let rows: Bindings[];
  try {
    assertRecursionFree(candidate.program);
    rows = executeProgram(clauses, candidate.program);
  } catch (error) {
    return reject(error instanceof Error ? error.message : String(error));
  }
  if (rows.length > MAX_ANSWER_ROWS)
    return reject(`more than ${MAX_ANSWER_ROWS} rows`);
  if (candidate.expectEmpty && rows.length > 0)
    return reject('expected empty answer');
  if (!candidate.expectEmpty && rows.length === 0)
    return reject('empty answer');
  if (candidate.requiresClosure && !candidate.expectEmpty) {
    const oneHop = candidate.program.replaceAll('_plus(', '(');
    try {
      const hopRows = executeProgram(clauses, oneHop);
      if (rowStrings(hopRows).join('|') === rowStrings(rows).join('|')) {
        return reject('closure not required: one-hop answer identical');
      }
    } catch {
      // if the one-hop form does not run, the closure form is certainly needed
    }
  }
  return { ...candidate, answer: rowStrings(rows) };
}

export function verifyAll(
  world: World,
  candidates: Candidate[],
): { examples: Example[]; rejections: Rejection[] } {
  const examples: Example[] = [];
  const rejections: Rejection[] = [];
  for (const candidate of candidates) {
    const result = verifyCandidate(world, candidate);
    if (isRejection(result)) rejections.push(result);
    else examples.push(result);
  }
  return { examples, rejections };
}

/** A template rejected more than `maxRate` of its candidates is a template bug, not data to skip. */
export function assertRejectionRates(
  candidates: Candidate[],
  rejections: Rejection[],
  maxRate = 0.5,
): void {
  const total = new Map<string, number>();
  const rejected = new Map<string, number>();
  for (const c of candidates)
    total.set(c.template, (total.get(c.template) ?? 0) + 1);
  for (const r of rejections) {
    rejected.set(
      r.candidate.template,
      (rejected.get(r.candidate.template) ?? 0) + 1,
    );
  }
  const offenders = [...total].filter(
    ([template, n]) => (rejected.get(template) ?? 0) / n > maxRate,
  );
  if (offenders.length > 0) {
    throw new Error(
      `rejection rate above ${maxRate} for templates: ${offenders
        .map(([t]) => t)
        .join(', ')} — fix the template, do not skip`,
    );
  }
}
