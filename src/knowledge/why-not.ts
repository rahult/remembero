import { createHash } from 'node:crypto';
import {
  type Bindings,
  type Clause,
  type Comparison,
  type EvaluateOptions,
  type Goal,
  type Literal,
  type MaterializedFact,
  type QuerySpec,
  type ScalarExpression,
  type Term,
  EngineLimitError,
  EngineSafetyError,
  comparisonMatches,
  isAggregateRule,
  isArithmeticExpression,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  materialize,
  parseQueryProgram,
  predKey,
  serializeClause,
  serializeGoal,
  serializeQuerySpec,
  serializeTerm,
} from '../engine/index.js';
import type { MemorySource } from '../store/store.js';
import { explainKnowledge, type ExplainKnowledgeResult } from './graph.js';
import {
  canonicalizeKnowledge,
  literalKnowledge,
  type EntityIdentityMode,
} from './identity.js';
import type { TrustViewMode } from './trust.js';

export const DEFAULT_MAX_WHY_NOT_FAILURES = 32;
export const MAX_WHY_NOT_FAILURES = 128;
export const DEFAULT_MAX_WHY_NOT_DEPTH = 8;
export const MAX_WHY_NOT_DEPTH = 32;
export const DEFAULT_MAX_WHY_NOT_CANDIDATES = 4;
export const MAX_WHY_NOT_CANDIDATES = 16;
export const DEFAULT_MAX_WHY_NOT_EVIDENCE = 16;
export const MAX_WHY_NOT_EVIDENCE = 64;
export const DEFAULT_WHY_NOT_SUMMARY_FAILURES = 3;
export const MAX_WHY_NOT_SUMMARY_FAILURES = 8;

export type WhyNotReason =
  | 'missing_fact'
  | 'rules_blocked'
  | 'negated_fact_present'
  | 'comparison_false'
  | 'recursive_cycle'
  | 'rule_output_mismatch'
  | 'aggregate_result_mismatch';

export interface WhyNotObservedFact {
  id: string;
  fact: string;
  explanation: ExplainKnowledgeResult;
}

export interface WhyNotRuleAttempt {
  id: string;
  rule: number;
  clause: string;
  aggregate: boolean;
  failures: WhyNotFailure[];
}

export interface WhyNotFailure {
  id: string;
  reason: WhyNotReason;
  goal: string;
  bindings: Record<string, string>;
  nearby: WhyNotObservedFact[];
  rules: WhyNotRuleAttempt[];
}

export type WhyNotGraphNode =
  | { id: string; kind: 'query'; query: string; status: WhyNotStatus }
  | {
      id: string;
      kind: 'failure';
      reason: WhyNotReason;
      goal: string;
      bindings: Record<string, string>;
    }
  | {
      id: string;
      kind: 'rule';
      rule: number;
      clause: string;
      aggregate: boolean;
    }
  | { id: string; kind: 'observed'; fact: string };

export interface WhyNotGraphEdge {
  id: string;
  kind: 'fails_at' | 'attempts' | 'blocked_by' | 'observed';
  from: string;
  to: string;
}

export interface WhyNotGraph {
  nodes: WhyNotGraphNode[];
  edges: WhyNotGraphEdge[];
}

export type WhyNotStatus = 'satisfied' | 'blocked';

export interface ExplainWhyNotOptions extends Omit<EvaluateOptions, 'metrics'> {
  entityIdentity?: EntityIdentityMode;
  trustMode?: TrustViewMode;
  maxFailures?: number;
  maxDiagnosticDepth?: number;
  maxCandidatesPerFailure?: number;
  maxEvidenceFacts?: number;
}

export interface ExplainWhyNotResult {
  status: WhyNotStatus;
  query: string;
  evaluatedQuery: string;
  summary: string;
  explanation: ExplainKnowledgeResult;
  failures: WhyNotFailure[];
  graph: WhyNotGraph;
  trustMode?: TrustViewMode;
}

function leafFailures(failures: WhyNotFailure[]): WhyNotFailure[] {
  const result: WhyNotFailure[] = [];
  const seen = new Set<string>();
  const visit = (failure: WhyNotFailure) => {
    if (failure.reason === 'rules_blocked') {
      const nested = failure.rules.flatMap((rule) => rule.failures);
      if (nested.length > 0) {
        for (const child of nested) visit(child);
        return;
      }
    }
    if (seen.has(failure.id)) return;
    seen.add(failure.id);
    result.push(failure);
  };
  for (const failure of failures) visit(failure);
  return result;
}

function failureSummary(failure: WhyNotFailure): string {
  switch (failure.reason) {
    case 'missing_fact':
      return `Required fact ${failure.goal} is missing`;
    case 'negated_fact_present': {
      const observed = failure.nearby.map(({ fact }) => fact).join(', ');
      return observed.length === 0
        ? `Negated goal ${failure.goal} is blocked by a present fact`
        : `Negated goal ${failure.goal} is blocked by ${observed}`;
    }
    case 'comparison_false':
      return `Comparison ${failure.goal} is false`;
    case 'recursive_cycle':
      return `Rule diagnosis cycles back to ${failure.goal}`;
    case 'rule_output_mismatch':
      return `A rule body succeeds but its output does not match ${failure.goal}`;
    case 'aggregate_result_mismatch':
      return `The aggregate result does not match ${failure.goal}`;
    case 'rules_blocked':
      return `Every rule for ${failure.goal} is blocked`;
  }
}

/** Produce a local, source-text-free summary from complete deterministic blockers. */
export function summarizeWhyNot(
  result: Pick<
    ExplainWhyNotResult,
    'status' | 'evaluatedQuery' | 'failures' | 'explanation'
  >,
  maxFailures = DEFAULT_WHY_NOT_SUMMARY_FAILURES,
): string {
  if (
    !Number.isSafeInteger(maxFailures) ||
    maxFailures < 1 ||
    maxFailures > MAX_WHY_NOT_SUMMARY_FAILURES
  ) {
    throw new EngineSafetyError(
      `why-not summary failures must be from 1 to ${MAX_WHY_NOT_SUMMARY_FAILURES}`,
    );
  }
  if (result.status === 'satisfied') {
    const count = result.explanation.rows.length;
    return `The query ${result.evaluatedQuery} is satisfied by ${count} result row${
      count === 1 ? '' : 's'
    }.`;
  }
  const failures = leafFailures(result.failures);
  const selected = failures.slice(0, maxFailures).map(failureSummary);
  const remainder = failures.length - selected.length;
  const detail =
    selected.length === 0
      ? 'No grounded repairable blocker was available.'
      : `${selected.join('; ')}${
          remainder > 0
            ? `; plus ${remainder} additional blocker${remainder === 1 ? '' : 's'}`
            : ''
        }.`;
  return `No stored result matches ${result.evaluatedQuery}. ${detail}`;
}

interface RuleDefinition {
  clause: Clause;
  number: number;
}

interface DiagnosticLimits {
  maxFailures: number;
  maxDepth: number;
  maxCandidates: number;
  maxEvidence: number;
}

interface DiagnosticContext {
  originalClauses: Clause[];
  originalSources: Map<string, MemorySource[]>;
  explanationOptions: Omit<
    ExplainWhyNotOptions,
    | 'maxFailures'
    | 'maxDiagnosticDepth'
    | 'maxCandidatesPerFailure'
    | 'maxEvidenceFacts'
  >;
  factsByPredicate: Map<string, MaterializedFact[]>;
  rulesByPredicate: Map<string, RuleDefinition[]>;
  limits: DiagnosticLimits;
  failures: number;
  evidenceByFact: Map<string, WhyNotObservedFact>;
}

function boundedOption(
  value: number | undefined,
  fallback: number,
  maximum: number,
  label: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1 || resolved > maximum) {
    throw new EngineSafetyError(`${label} must be from 1 to ${maximum}`);
  }
  return resolved;
}

function stableId(prefix: string, value: unknown): string {
  return `${prefix}:${createHash('sha256')
    .update(JSON.stringify(value))
    .digest('hex')}`;
}

function termFromValue(value: string | number): Term {
  return typeof value === 'number'
    ? { type: 'num', value }
    : { type: 'atom', value };
}

function termEqual(left: Term, right: Term): boolean {
  return (
    (left.type === 'atom' &&
      right.type === 'atom' &&
      left.value === right.value) ||
    (left.type === 'num' && right.type === 'num' && left.value === right.value)
  );
}

function resolveTerm(term: Term, bindings: Bindings): Term {
  return term.type === 'var' ? (bindings[term.name] ?? term) : term;
}

function substituteExpression(
  expression: ScalarExpression,
  bindings: Bindings,
): ScalarExpression {
  if (!isArithmeticExpression(expression))
    return resolveTerm(expression, bindings);
  if (expression.kind === 'unary') {
    return {
      ...expression,
      operand: substituteExpression(expression.operand, bindings),
    };
  }
  return {
    ...expression,
    left: substituteExpression(expression.left, bindings),
    right: substituteExpression(expression.right, bindings),
  };
}

function substituteLiteral(literal: Literal, bindings: Bindings): Literal {
  return {
    predicate: literal.predicate,
    args: literal.args.map((term) => resolveTerm(term, bindings)),
  };
}

function substituteGoal(goal: Goal, bindings: Bindings): Goal {
  if (isComparison(goal)) {
    return {
      op: goal.op,
      left: substituteExpression(goal.left, bindings),
      right: substituteExpression(goal.right, bindings),
    };
  }
  if (isNegation(goal)) {
    return { not: substituteLiteral(goal.not, bindings) };
  }
  return substituteLiteral(goal, bindings);
}

function bindingsJson(bindings: Bindings): Record<string, string> {
  return Object.fromEntries(
    Object.entries(bindings).map(([name, term]) => [name, serializeTerm(term)]),
  );
}

function bindingKey(bindings: Bindings): string {
  return JSON.stringify(
    Object.entries(bindings)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, term]) => [name, serializeTerm(term)]),
  );
}

function groundLiteral(fact: MaterializedFact): Literal {
  return {
    predicate: fact.predicate,
    args: fact.values.map(termFromValue),
  };
}

function matchLiteral(
  pattern: Literal,
  fact: MaterializedFact,
  bindings: Bindings,
): Bindings | undefined {
  if (
    pattern.predicate !== fact.predicate ||
    pattern.args.length !== fact.values.length
  ) {
    return undefined;
  }
  let next = bindings;
  for (const [index, requested] of pattern.args.entries()) {
    const actual = termFromValue(fact.values[index]);
    const resolved = resolveTerm(requested, next);
    if (resolved.type === 'wildcard') continue;
    if (resolved.type === 'var') {
      if (next === bindings) next = { ...bindings };
      next[resolved.name] = actual;
    } else if (!termEqual(resolved, actual)) {
      return undefined;
    }
  }
  return next;
}

function relation(
  context: DiagnosticContext,
  literal: Literal,
): MaterializedFact[] {
  return context.factsByPredicate.get(predKey(literal)) ?? [];
}

function deduplicateBindings(
  values: Bindings[],
  context: DiagnosticContext,
): Bindings[] {
  const seen = new Set<string>();
  const result: Bindings[] = [];
  for (const value of values) {
    const key = bindingKey(value);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(value);
    if (result.length > context.limits.maxFailures) {
      throw new EngineLimitError(
        `why-not diagnostic frontier exceeded ${context.limits.maxFailures} bindings`,
      );
    }
  }
  return result;
}

function advanceGoal(
  context: DiagnosticContext,
  goal: Goal,
  bindings: Bindings,
): Bindings[] {
  if (isComparison(goal)) {
    return comparisonMatches(goal, bindings) ? [bindings] : [];
  }
  if (isNegation(goal)) {
    const literal = substituteLiteral(goal.not, bindings);
    const unbound = literal.args.find((term) => term.type === 'var');
    if (unbound?.type === 'var') {
      throw new EngineSafetyError(
        `negated variable ${unbound.name} is not bound by an earlier positive relation`,
      );
    }
    return relation(context, literal).some(
      (fact) => matchLiteral(literal, fact, {}) !== undefined,
    )
      ? []
      : [bindings];
  }
  return relation(context, goal).flatMap((fact) => {
    const matched = matchLiteral(goal, fact, bindings);
    return matched === undefined ? [] : [matched];
  });
}

function evidenceFor(
  context: DiagnosticContext,
  fact: MaterializedFact,
): WhyNotObservedFact {
  const literal = groundLiteral(fact);
  const serialized = `${serializeGoal(literal)}.`;
  const existing = context.evidenceByFact.get(serialized);
  if (existing !== undefined) return existing;
  if (context.evidenceByFact.size >= context.limits.maxEvidence) {
    throw new EngineLimitError(
      `why-not evidence exceeded ${context.limits.maxEvidence} facts`,
    );
  }
  const explanation = explainKnowledge(
    context.originalClauses,
    serializeGoal(literal),
    context.originalSources,
    {
      ...context.explanationOptions,
      maxRows: 1,
    },
  );
  const observed: WhyNotObservedFact = {
    id: stableId('observed', serialized),
    fact: serialized,
    explanation,
  };
  context.evidenceByFact.set(serialized, observed);
  return observed;
}

function nearbyFacts(
  context: DiagnosticContext,
  literal: Literal,
  bindings: Bindings,
  exactOnly = false,
): WhyNotObservedFact[] {
  const resolved = substituteLiteral(literal, bindings);
  const known = resolved.args.map((term) =>
    term.type === 'atom' || term.type === 'num' ? term : undefined,
  );
  const ranked = relation(context, resolved)
    .map((fact, index) => ({
      fact,
      index,
      exact: matchLiteral(resolved, fact, {}) !== undefined,
      score: fact.values.reduce<number>(
        (total, value, position) =>
          known[position] !== undefined &&
          termEqual(known[position]!, termFromValue(value))
            ? total + 1
            : total,
        0,
      ),
    }))
    .filter(({ exact }) => !exactOnly || exact)
    .sort(
      (left, right) =>
        Number(right.exact) - Number(left.exact) ||
        right.score - left.score ||
        left.index - right.index,
    )
    .slice(0, context.limits.maxCandidates);
  return ranked.map(({ fact }) => evidenceFor(context, fact));
}

function newFailure(
  context: DiagnosticContext,
  reason: WhyNotReason,
  goal: Goal,
  bindings: Bindings,
  path: Array<string | number>,
  nearby: WhyNotObservedFact[] = [],
  rules: WhyNotRuleAttempt[] = [],
): WhyNotFailure {
  if (++context.failures > context.limits.maxFailures) {
    throw new EngineLimitError(
      `why-not explanation exceeded ${context.limits.maxFailures} failures`,
    );
  }
  const serializedGoal = serializeGoal(substituteGoal(goal, bindings));
  return {
    id: stableId('failure', [
      path,
      reason,
      serializedGoal,
      bindingsJson(bindings),
    ]),
    reason,
    goal: serializedGoal,
    bindings: bindingsJson(bindings),
    nearby,
    rules,
  };
}

function seedRuleBindings(
  rule: Clause,
  requested: Literal,
  outerBindings: Bindings,
): Bindings | undefined {
  const resolved = substituteLiteral(requested, outerBindings);
  const bindings: Bindings = {};
  const aggregateOutput = isAggregateRule(rule) ? rule.aggregate.as : undefined;
  for (const [index, headTerm] of rule.head.args.entries()) {
    const wanted = resolved.args[index];
    if (wanted === undefined) return undefined;
    if (headTerm.type === 'var') {
      if (headTerm.name === aggregateOutput) continue;
      if (wanted.type !== 'atom' && wanted.type !== 'num') continue;
      const prior = bindings[headTerm.name];
      if (prior === undefined) bindings[headTerm.name] = wanted;
      else if (!termEqual(prior, wanted)) return undefined;
      continue;
    }
    if (
      (wanted.type === 'atom' || wanted.type === 'num') &&
      !termEqual(headTerm, wanted)
    ) {
      return undefined;
    }
  }
  return bindings;
}

function diagnoseSequence(
  context: DiagnosticContext,
  goals: Goal[],
  seeds: Bindings[],
  depth: number,
  path: Array<string | number>,
  visited: ReadonlySet<string>,
): { solutions: Bindings[]; failures: WhyNotFailure[] } {
  let frontier = seeds;
  const failures: WhyNotFailure[] = [];
  for (const [goalIndex, goal] of goals.entries()) {
    const next: Bindings[] = [];
    for (const [bindingIndex, bindings] of frontier.entries()) {
      const advanced = advanceGoal(context, goal, bindings);
      if (advanced.length === 0) {
        failures.push(
          diagnoseGoal(
            context,
            goal,
            bindings,
            depth,
            [...path, goalIndex, bindingIndex],
            visited,
          ),
        );
      } else {
        next.push(...advanced);
      }
    }
    frontier = deduplicateBindings(next, context);
    if (frontier.length === 0) break;
  }
  return { solutions: frontier, failures };
}

function diagnoseRule(
  context: DiagnosticContext,
  definition: RuleDefinition,
  requested: Literal,
  outerBindings: Bindings,
  depth: number,
  path: Array<string | number>,
  visited: ReadonlySet<string>,
): WhyNotRuleAttempt | undefined {
  const seeded = seedRuleBindings(definition.clause, requested, outerBindings);
  if (seeded === undefined) return undefined;
  const result = diagnoseSequence(
    context,
    definition.clause.body,
    [seeded],
    depth + 1,
    [...path, 'rule', definition.number],
    visited,
  );
  const aggregate = isAggregateRule(definition.clause)
    ? definition.clause.aggregate
    : undefined;
  const globalCount =
    aggregate?.op === 'count' &&
    !definition.clause.head.args.some(
      (term) => term.type === 'var' && term.name !== aggregate.as,
    );
  const failures =
    result.solutions.length === 0 && !globalCount
      ? result.failures
      : [
          newFailure(
            context,
            isAggregateRule(definition.clause)
              ? 'aggregate_result_mismatch'
              : 'rule_output_mismatch',
            requested,
            outerBindings,
            [...path, 'rule', definition.number, 'output'],
          ),
        ];
  return {
    id: stableId('rule-attempt', [
      path,
      definition.number,
      serializeClause(definition.clause),
    ]),
    rule: definition.number,
    clause: serializeClause(definition.clause),
    aggregate: isAggregateRule(definition.clause),
    failures,
  };
}

function diagnoseGoal(
  context: DiagnosticContext,
  goal: Goal,
  bindings: Bindings,
  depth: number,
  path: Array<string | number>,
  visited: ReadonlySet<string>,
): WhyNotFailure {
  if (depth > context.limits.maxDepth) {
    throw new EngineLimitError(
      `why-not explanation exceeded depth ${context.limits.maxDepth}`,
    );
  }
  if (isComparison(goal)) {
    return newFailure(context, 'comparison_false', goal, bindings, path);
  }
  if (isNegation(goal)) {
    return newFailure(
      context,
      'negated_fact_present',
      goal,
      bindings,
      path,
      nearbyFacts(context, goal.not, bindings, true),
    );
  }

  const resolved = substituteLiteral(goal, bindings);
  const signature = serializeGoal(resolved);
  const nearby = nearbyFacts(context, goal, bindings);
  if (visited.has(signature)) {
    return newFailure(context, 'recursive_cycle', goal, bindings, path, nearby);
  }
  const definitions = context.rulesByPredicate.get(predKey(goal)) ?? [];
  if (definitions.length === 0) {
    return newFailure(context, 'missing_fact', goal, bindings, path, nearby);
  }
  const nestedVisited = new Set(visited);
  nestedVisited.add(signature);
  const attempts = definitions.flatMap((definition) => {
    const attempt = diagnoseRule(
      context,
      definition,
      goal,
      bindings,
      depth,
      path,
      nestedVisited,
    );
    return attempt === undefined ? [] : [attempt];
  });
  if (attempts.length === 0) {
    return newFailure(context, 'missing_fact', goal, bindings, path, nearby);
  }
  return newFailure(
    context,
    'rules_blocked',
    goal,
    bindings,
    path,
    nearby,
    attempts,
  );
}

function rulesByPredicate(clauses: Clause[]): Map<string, RuleDefinition[]> {
  const result = new Map<string, RuleDefinition[]>();
  let ruleNumber = 0;
  for (const clause of clauses) {
    if (isIntegrityConstraint(clause) || clause.body.length === 0) continue;
    ruleNumber += 1;
    const key = predKey(clause.head);
    const rules = result.get(key) ?? [];
    rules.push({ clause, number: ruleNumber });
    result.set(key, rules);
  }
  return result;
}

function graphFor(
  query: string,
  status: WhyNotStatus,
  failures: WhyNotFailure[],
): WhyNotGraph {
  const queryId = stableId('query', query);
  const nodes = new Map<string, WhyNotGraphNode>([
    [queryId, { id: queryId, kind: 'query', query, status }],
  ]);
  const edges = new Map<string, WhyNotGraphEdge>();
  const addEdge = (kind: WhyNotGraphEdge['kind'], from: string, to: string) => {
    const id = stableId('why-not-edge', [kind, from, to]);
    edges.set(id, { id, kind, from, to });
  };
  const visitFailure = (
    failure: WhyNotFailure,
    parent: string,
    edgeKind: 'fails_at' | 'blocked_by',
  ) => {
    nodes.set(failure.id, {
      id: failure.id,
      kind: 'failure',
      reason: failure.reason,
      goal: failure.goal,
      bindings: failure.bindings,
    });
    addEdge(edgeKind, parent, failure.id);
    for (const observed of failure.nearby) {
      nodes.set(observed.id, {
        id: observed.id,
        kind: 'observed',
        fact: observed.fact,
      });
      addEdge('observed', failure.id, observed.id);
    }
    for (const rule of failure.rules) {
      nodes.set(rule.id, {
        id: rule.id,
        kind: 'rule',
        rule: rule.rule,
        clause: rule.clause,
        aggregate: rule.aggregate,
      });
      addEdge('attempts', failure.id, rule.id);
      for (const nested of rule.failures)
        visitFailure(nested, rule.id, 'blocked_by');
    }
  };
  for (const failure of failures) visitFailure(failure, queryId, 'fails_at');
  return {
    nodes: [...nodes.values()].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
    edges: [...edges.values()].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
  };
}

function queryGoals(query: QuerySpec): Goal[] {
  return query.goals;
}

/** Explain a failed relational or aggregate query through bounded deterministic blockers. */
export function explainWhyNot(
  clauses: Clause[],
  query: string,
  sourceIndex: Map<string, MemorySource[]> = new Map(),
  options: ExplainWhyNotOptions = {},
): ExplainWhyNotResult {
  const {
    entityIdentity,
    trustMode,
    maxFailures: requestedFailures,
    maxDiagnosticDepth: requestedDepth,
    maxCandidatesPerFailure: requestedCandidates,
    maxEvidenceFacts: requestedEvidence,
    ...evaluateOptions
  } = options;
  const limits: DiagnosticLimits = {
    maxFailures: boundedOption(
      requestedFailures,
      DEFAULT_MAX_WHY_NOT_FAILURES,
      MAX_WHY_NOT_FAILURES,
      'maxFailures',
    ),
    maxDepth: boundedOption(
      requestedDepth,
      DEFAULT_MAX_WHY_NOT_DEPTH,
      MAX_WHY_NOT_DEPTH,
      'maxDiagnosticDepth',
    ),
    maxCandidates: boundedOption(
      requestedCandidates,
      DEFAULT_MAX_WHY_NOT_CANDIDATES,
      MAX_WHY_NOT_CANDIDATES,
      'maxCandidatesPerFailure',
    ),
    maxEvidence: boundedOption(
      requestedEvidence,
      DEFAULT_MAX_WHY_NOT_EVIDENCE,
      MAX_WHY_NOT_EVIDENCE,
      'maxEvidenceFacts',
    ),
  };
  // Same normalizer as the query tool and explainKnowledge: a goal list, a `?-`
  // query, or a rule program whose sink rule (or explicit `?-` line) is the target.
  // Authored rules join the clauses so their heads can be diagnosed like any rule.
  const program = parseQueryProgram(query);
  const clausesWithAuthored =
    program.clauses.length === 0 ? clauses : [...clauses, ...program.clauses];
  const view =
    entityIdentity === 'canonical'
      ? canonicalizeKnowledge(clausesWithAuthored, sourceIndex, trustMode)
      : literalKnowledge(clausesWithAuthored, sourceIndex, trustMode);
  const parsed = program.query;
  const evaluated =
    entityIdentity === 'canonical'
      ? view.resolver.canonicalizeQuery(parsed).query
      : parsed;
  const explanationOptions = {
    ...evaluateOptions,
    ...(entityIdentity === undefined ? {} : { entityIdentity }),
    ...(trustMode === undefined ? {} : { trustMode }),
  };
  const explanation = explainKnowledge(
    clauses,
    query,
    sourceIndex,
    explanationOptions,
  );
  const evaluatedQuery = serializeQuerySpec(evaluated);
  clauses = clausesWithAuthored;
  if (explanation.rows.length > 0) {
    const summaryInput = {
      status: 'satisfied' as const,
      evaluatedQuery,
      explanation,
      failures: [],
    };
    return {
      status: 'satisfied',
      query,
      evaluatedQuery,
      summary: summarizeWhyNot(summaryInput),
      explanation,
      failures: [],
      graph: graphFor(evaluatedQuery, 'satisfied', []),
      ...(trustMode === undefined || trustMode === 'accepted'
        ? {}
        : { trustMode }),
    };
  }

  const materialized = materialize(view.clauses, evaluateOptions);
  const factsByPredicate = new Map<string, MaterializedFact[]>();
  for (const fact of materialized) {
    const key = `${fact.predicate}/${fact.values.length}`;
    const values = factsByPredicate.get(key) ?? [];
    values.push(fact);
    factsByPredicate.set(key, values);
  }
  const context: DiagnosticContext = {
    originalClauses: clauses,
    originalSources: sourceIndex,
    explanationOptions,
    factsByPredicate,
    rulesByPredicate: rulesByPredicate(view.clauses),
    limits,
    failures: 0,
    evidenceByFact: new Map(),
  };
  const diagnosed = diagnoseSequence(
    context,
    queryGoals(evaluated),
    [{}],
    1,
    ['query'],
    new Set(),
  );
  if (diagnosed.solutions.length > 0) {
    throw new Error(
      'why-not diagnostic disagreed with deterministic query evaluation',
    );
  }
  const summaryInput = {
    status: 'blocked' as const,
    evaluatedQuery,
    explanation,
    failures: diagnosed.failures,
  };
  return {
    status: 'blocked',
    query,
    evaluatedQuery,
    summary: summarizeWhyNot(summaryInput),
    explanation,
    failures: diagnosed.failures,
    graph: graphFor(evaluatedQuery, 'blocked', diagnosed.failures),
    ...(trustMode === undefined || trustMode === 'accepted'
      ? {}
      : { trustMode }),
  };
}
