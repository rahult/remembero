import {
  type Clause,
  type CmpOp,
  type Comparison,
  type Goal,
  type Literal,
  type AggregateOperator,
  type AggregateQuerySpec,
  type AggregateRuleClause,
  type AggregateRuleSpec,
  type QuerySpec,
  type ScalarExpression,
  type Term,
  MAX_ARITHMETIC_EXPRESSION_DEPTH,
  MAX_ARITHMETIC_EXPRESSION_NODES,
  isArithmeticExpression,
  isAggregateRule,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  canonicalKey,
  predKey,
} from './ast.js';
import { expandClosurePredicates } from './closure.js';
import { stratifyProgram, type StratifiedRule } from './stratify.js';

export type Bindings = Record<string, Term>;

export interface DerivationProof {
  predicate: string;
  values: (string | number)[];
  rule?: number;
  because?: ProofStep[];
  /** Exact grouped reduction used to derive this reusable relation fact. */
  aggregate?: AggregateProof;
}

export interface AbsenceProof {
  negated: true;
  predicate: string;
  /** Grounded arguments; null represents an existential wildcard. */
  pattern: (string | number | null)[];
  stratum: number;
}

export type ProofStep = DerivationProof | AbsenceProof;

export interface AggregateContribution {
  bindings: Bindings;
  proofs: ProofStep[];
}

export interface AggregateProof {
  aggregated: true;
  op: AggregateOperator;
  input: '*' | string;
  as: string;
  value: string | number;
  contributors: AggregateContribution[];
  /** Contributor indexes equal to the selected min/max value, in stable query order. */
  witnessPositions?: number[];
}

export type QueryProof = ProofStep | AggregateProof;

export interface ExplainedBindings {
  bindings: Bindings;
  proofs: ProofStep[];
  alternativeProofs?: ProofStep[][];
}

export interface ExplainedQueryBindings {
  bindings: Bindings;
  proofs: QueryProof[];
  alternativeProofs?: ProofStep[][];
}

export interface MaterializedFactWithProof {
  predicate: string;
  values: (string | number)[];
  derived: boolean;
  proof: DerivationProof;
}

export interface MaterializedFact {
  predicate: string;
  values: (string | number)[];
  derived: boolean;
}

export class EngineLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EngineLimitError';
  }
}

export class EngineSafetyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EngineSafetyError';
  }
}

export interface EvaluateOptions {
  maxFacts?: number;
  maxIterations?: number;
  maxRows?: number;
  /** Maximum candidate rows inspected before an exact aggregate fails closed. */
  maxAggregateRows?: number;
  /** Maximum contributor rows retained in an aggregate explanation. */
  maxAggregateProofRows?: number;
  maxProofDepth?: number;
  maxProofNodes?: number;
  maxProofsPerRow?: number;
  maxProofEnumerationSteps?: number;
  /** Use deterministic per-relation lookup indexes, or disable them for profiling. */
  relationIndex?: 'auto' | 'off';
  /** Optional deterministic counters reset and populated by each evaluation call. */
  metrics?: EvaluationMetrics;
}

export interface EvaluationMetrics {
  relationLookups: number;
  indexedRelationLookups: number;
  indexFactsProcessed: number;
  candidateFactsVisited: number;
}

export const DEFAULT_MAX_PROOFS_PER_ROW = 1;
export const MAX_PROOFS_PER_ROW = 16;
export const DEFAULT_MAX_PROOF_ENUMERATION_STEPS = 100_000;
export const MAX_PROOF_ENUMERATION_STEPS = 1_000_000;

function assertAggregateProofRowLimit(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new EngineSafetyError(
      'maxAggregateProofRows must be a non-negative safe integer'
    );
  }
}

interface FactEntry {
  predicate: string;
  tuple: Term[];
  derived: boolean;
  rule?: number;
  because?: ProofRef[];
  aggregate?: AggregateDerivationRef;
}

interface AggregateDerivationRef {
  spec: AggregateRuleSpec;
  result: AggregateResultRef;
}

type ProofRef = FactEntry | AbsenceProof;

interface GoalSolution {
  env: Bindings;
  proofs: ProofRef[];
}

interface QueryRowRef {
  bindings: Bindings;
  proofs: ProofRef[];
}

interface AlternativeProofOptions {
  maxProofsPerRow: number;
  maxProofEnumerationSteps: number;
}

/** Ground tuples and an insertion-ordered first-argument index per predicate. */
interface Relation {
  tuples: Map<string, FactEntry>;
  byFirstArgument?: Map<string, FactEntry[]>;
}
type Database = Map<string, Relation>;

interface RelationLookupContext {
  indexed: boolean;
  metrics?: EvaluationMetrics;
}

const litKey = (predicate: string, arity: number) => `${predicate}/${arity}`;

function resolveAlternativeProofOptions(
  options: EvaluateOptions
): AlternativeProofOptions {
  const {
    maxProofsPerRow = DEFAULT_MAX_PROOFS_PER_ROW,
    maxProofEnumerationSteps = DEFAULT_MAX_PROOF_ENUMERATION_STEPS,
  } = options;
  if (!Number.isSafeInteger(maxProofsPerRow) || maxProofsPerRow < 1) {
    throw new EngineSafetyError('maxProofsPerRow must be a positive safe integer');
  }
  if (maxProofsPerRow > MAX_PROOFS_PER_ROW) {
    throw new EngineSafetyError(
      `maxProofsPerRow must be at most ${MAX_PROOFS_PER_ROW}`
    );
  }
  if (
    !Number.isSafeInteger(maxProofEnumerationSteps) ||
    maxProofEnumerationSteps < 1
  ) {
    throw new EngineSafetyError(
      'maxProofEnumerationSteps must be a positive safe integer'
    );
  }
  if (maxProofEnumerationSteps > MAX_PROOF_ENUMERATION_STEPS) {
    throw new EngineSafetyError(
      `maxProofEnumerationSteps must be at most ${MAX_PROOF_ENUMERATION_STEPS}`
    );
  }
  return { maxProofsPerRow, maxProofEnumerationSteps };
}

function keyPart(term: Term): readonly [string, string | number] {
  switch (term.type) {
    case 'atom':
      return ['atom', term.value];
    case 'num':
      return ['num', term.value];
    case 'var':
      return ['var', term.name];
    case 'wildcard':
      return ['wildcard', '_'];
  }
}

function tupleKey(args: Term[]): string {
  return JSON.stringify(args.map(keyPart));
}

function firstArgumentKey(term: Term): string | undefined {
  if (term.type !== 'atom' && term.type !== 'num') return undefined;
  return JSON.stringify(keyPart(term));
}

function relationLookupContext(options: EvaluateOptions): RelationLookupContext {
  const mode = options.relationIndex ?? 'auto';
  if (mode !== 'auto' && mode !== 'off') {
    throw new EngineSafetyError("relationIndex must be 'auto' or 'off'");
  }
  if (options.metrics !== undefined) {
    options.metrics.relationLookups = 0;
    options.metrics.indexedRelationLookups = 0;
    options.metrics.indexFactsProcessed = 0;
    options.metrics.candidateFactsVisited = 0;
  }
  return { indexed: mode === 'auto', metrics: options.metrics };
}

function assertFiniteNumericTerm(term: Term): void {
  if (term.type === 'num' && !Number.isFinite(term.value)) {
    throw new EngineSafetyError('numeric terms must be finite');
  }
}

function assertExpressionSafety(
  expression: ScalarExpression,
  budget: { nodes: number } = { nodes: 0 }
): void {
  const pending: Array<{ expression: ScalarExpression; depth: number }> = [
    { expression, depth: 1 },
  ];
  while (pending.length > 0) {
    const current = pending.pop()!;
    if (current.depth > MAX_ARITHMETIC_EXPRESSION_DEPTH) {
      throw new EngineLimitError(
        `arithmetic expression exceeded depth ${MAX_ARITHMETIC_EXPRESSION_DEPTH}`
      );
    }
    if (++budget.nodes > MAX_ARITHMETIC_EXPRESSION_NODES) {
      throw new EngineLimitError(
        `arithmetic expression exceeded ${MAX_ARITHMETIC_EXPRESSION_NODES} nodes`
      );
    }
    if (!isArithmeticExpression(current.expression)) {
      assertFiniteNumericTerm(current.expression);
      continue;
    }
    if (current.expression.kind === 'unary') {
      pending.push({
        expression: current.expression.operand,
        depth: current.depth + 1,
      });
    } else {
      pending.push(
        { expression: current.expression.right, depth: current.depth + 1 },
        { expression: current.expression.left, depth: current.depth + 1 }
      );
    }
  }
}

function assertGoalNumericSafety(goal: Goal): void {
  if (isComparison(goal)) {
    const budget = { nodes: 0 };
    assertExpressionSafety(goal.left, budget);
    assertExpressionSafety(goal.right, budget);
    return;
  }
  for (const term of (isNegation(goal) ? goal.not.args : goal.args)) {
    assertFiniteNumericTerm(term);
  }
}

function assertGoalsNumericSafety(goals: Goal[]): void {
  for (const goal of goals) assertGoalNumericSafety(goal);
}

function termEq(a: Term, b: Term): boolean {
  assertFiniteNumericTerm(a);
  assertFiniteNumericTerm(b);
  if (a.type === 'num' && b.type === 'num') return a.value === b.value;
  if (a.type === 'atom' && b.type === 'atom') return a.value === b.value;
  return false;
}

function ordered(op: CmpOp, cmp: number): boolean {
  switch (op) {
    case '<':
      return cmp < 0;
    case '>':
      return cmp > 0;
    case '<=':
      return cmp <= 0;
    case '>=':
      return cmp >= 0;
    default:
      return false;
  }
}

function comparisonHolds(op: CmpOp, left: Term, right: Term): boolean {
  assertFiniteNumericTerm(left);
  assertFiniteNumericTerm(right);
  if (op === '=') return termEq(left, right);
  if (op === '!=') {
    // != only meaningfully compares two ground terms of the same type
    if (left.type === 'num' && right.type === 'num') return left.value !== right.value;
    if (left.type === 'atom' && right.type === 'atom') return left.value !== right.value;
    return true;
  }
  if (left.type === 'num' && right.type === 'num') {
    return ordered(op, left.value === right.value ? 0 : left.value < right.value ? -1 : 1);
  }
  if (left.type === 'atom' && right.type === 'atom') {
    return ordered(op, left.value === right.value ? 0 : left.value < right.value ? -1 : 1);
  }
  return false; // mixed types: goal fails rather than throwing
}

function resolve(term: Term, env: Bindings): Term {
  return term.type === 'var' ? (env[term.name] ?? term) : term;
}

interface ExpressionEvaluationBudget {
  nodes: number;
}

interface EvaluatedExpression {
  term: Term;
  arithmetic: boolean;
}

function numericArithmeticOperand(term: Term): number {
  if (term.type === 'var') {
    throw new EngineSafetyError(`arithmetic variable ${term.name} is not grounded`);
  }
  if (term.type === 'wildcard') {
    throw new EngineSafetyError('arithmetic expressions may not contain wildcards');
  }
  if (term.type !== 'num') {
    throw new EngineSafetyError('arithmetic expressions require numeric operands');
  }
  assertFiniteNumericTerm(term);
  return term.value;
}

function evaluateScalarExpression(
  expression: ScalarExpression,
  env: Bindings,
  budget: ExpressionEvaluationBudget,
  depth = 1
): EvaluatedExpression {
  if (depth > MAX_ARITHMETIC_EXPRESSION_DEPTH) {
    throw new EngineLimitError(
      `arithmetic expression exceeded depth ${MAX_ARITHMETIC_EXPRESSION_DEPTH}`
    );
  }
  if (++budget.nodes > MAX_ARITHMETIC_EXPRESSION_NODES) {
    throw new EngineLimitError(
      `arithmetic expression exceeded ${MAX_ARITHMETIC_EXPRESSION_NODES} nodes`
    );
  }
  if (!isArithmeticExpression(expression)) {
    return { term: resolve(expression, env), arithmetic: false };
  }
  if (expression.kind === 'unary') {
    const operand = evaluateScalarExpression(expression.operand, env, budget, depth + 1);
    const numeric = numericArithmeticOperand(operand.term);
    const result = expression.op === '-' ? -numeric : numeric;
    if (!Number.isFinite(result)) {
      throw new EngineSafetyError('arithmetic expression produced a non-finite result');
    }
    return {
      term: { type: 'num', value: result === 0 ? 0 : result },
      arithmetic: true,
    };
  }

  const left = evaluateScalarExpression(expression.left, env, budget, depth + 1);
  const right = evaluateScalarExpression(expression.right, env, budget, depth + 1);
  const leftValue = numericArithmeticOperand(left.term);
  const rightValue = numericArithmeticOperand(right.term);
  if (expression.op === '/' && rightValue === 0) {
    throw new EngineSafetyError('arithmetic division by zero');
  }
  let result: number;
  switch (expression.op) {
    case '+':
      result = leftValue + rightValue;
      break;
    case '-':
      result = leftValue - rightValue;
      break;
    case '*':
      result = leftValue * rightValue;
      break;
    case '/':
      result = leftValue / rightValue;
      break;
  }
  if (!Number.isFinite(result)) {
    throw new EngineSafetyError('arithmetic expression produced a non-finite result');
  }
  return {
    term: { type: 'num', value: result === 0 ? 0 : result },
    arithmetic: true,
  };
}

function matchArgs(args: Term[], tuple: Term[], env: Bindings): Bindings | null {
  let extended = env;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.type === 'wildcard') continue;
    if (arg.type === 'var') {
      const bound = extended[arg.name];
      if (bound === undefined) {
        if (extended === env) extended = { ...env };
        extended[arg.name] = tuple[i];
      } else if (!termEq(bound, tuple[i])) {
        return null;
      }
    } else if (!termEq(arg, tuple[i])) {
      return null;
    }
  }
  return extended;
}

function checkComparison(goal: Comparison, env: Bindings): boolean {
  const budget: ExpressionEvaluationBudget = { nodes: 0 };
  const leftExpression = evaluateScalarExpression(goal.left, env, budget);
  const rightExpression = evaluateScalarExpression(goal.right, env, budget);
  const left = leftExpression.term;
  const right = rightExpression.term;
  if (left.type === 'var' || left.type === 'wildcard') return false;
  if (right.type === 'var' || right.type === 'wildcard') return false;
  if (
    (leftExpression.arithmetic || rightExpression.arithmetic) &&
    (left.type !== 'num' || right.type !== 'num')
  ) {
    throw new EngineSafetyError(
      'arithmetic comparisons require numeric values on both sides'
    );
  }
  return comparisonHolds(goal.op, left, right);
}

/** Evaluate one comparison against an explicit binding environment with engine-identical safety. */
export function comparisonMatches(
  goal: Comparison,
  bindings: Bindings = {}
): boolean {
  assertGoalNumericSafety(goal);
  return checkComparison(goal, bindings);
}

function relationCandidates(
  relation: Relation,
  args: Term[],
  env: Bindings,
  lookup: RelationLookupContext
): Iterable<FactEntry> {
  if (lookup.metrics) ++lookup.metrics.relationLookups;
  if (lookup.indexed && args.length > 0) {
    const key = firstArgumentKey(resolve(args[0], env));
    if (key !== undefined) {
      if (lookup.metrics) ++lookup.metrics.indexedRelationLookups;
      if (relation.byFirstArgument === undefined) {
        relation.byFirstArgument = new Map();
        for (const entry of relation.tuples.values()) {
          if (lookup.metrics) ++lookup.metrics.indexFactsProcessed;
          const entryKey = firstArgumentKey(entry.tuple[0]);
          if (entryKey === undefined) continue;
          const bucket = relation.byFirstArgument.get(entryKey);
          if (bucket) bucket.push(entry);
          else relation.byFirstArgument.set(entryKey, [entry]);
        }
      }
      return relation.byFirstArgument.get(key) ?? [];
    }
  }
  return relation.tuples.values();
}

/** Walk goals left-to-right, extending env; `source` picks which relation a goal reads. */
function* solveGoals(
  goals: Goal[],
  index: number,
  env: Bindings,
  proofs: ProofRef[],
  source: (goalIndex: number, key: string) => Relation | undefined,
  stratumOf: (key: string) => number,
  lookup: RelationLookupContext
): Generator<GoalSolution> {
  if (index === goals.length) {
    yield { env, proofs };
    return;
  }
  const goal = goals[index];
  if (isComparison(goal)) {
    if (checkComparison(goal, env)) {
      yield* solveGoals(goals, index + 1, env, proofs, source, stratumOf, lookup);
    }
    return;
  }
  if (isNegation(goal)) {
    const key = litKey(goal.not.predicate, goal.not.args.length);
    const resolved = goal.not.args.map((term): Term => {
      const value = resolve(term, env);
      if (value.type === 'var') {
        throw new EngineSafetyError(
          `negated variable ${value.name} is not bound by an earlier positive relation`
        );
      }
      return value;
    });
    const relation = source(index, key);
    if (relation) {
      for (const entry of relationCandidates(relation, resolved, {}, lookup)) {
        if (lookup.metrics) ++lookup.metrics.candidateFactsVisited;
        if (matchArgs(resolved, entry.tuple, {}) !== null) return;
      }
    }
    const absence: AbsenceProof = {
      negated: true,
      predicate: goal.not.predicate,
      pattern: resolved.map((term) =>
        term.type === 'wildcard' ? null : groundValue(term)
      ),
      stratum: stratumOf(key),
    };
    yield* solveGoals(
      goals,
      index + 1,
      env,
      [...proofs, absence],
      source,
      stratumOf,
      lookup
    );
    return;
  }
  const relation = source(index, litKey(goal.predicate, goal.args.length));
  if (!relation) return;
  for (const entry of relationCandidates(relation, goal.args, env, lookup)) {
    if (lookup.metrics) ++lookup.metrics.candidateFactsVisited;
    const extended = matchArgs(goal.args, entry.tuple, env);
    if (extended) {
      yield* solveGoals(
        goals,
        index + 1,
        extended,
        [...proofs, entry],
        source,
        stratumOf,
        lookup
      );
    }
  }
}

/** Does a ground literal match a pattern that may contain variables/wildcards? */
export function literalMatches(pattern: Literal, fact: Literal): boolean {
  if (pattern.predicate !== fact.predicate) return false;
  if (pattern.args.length !== fact.args.length) return false;
  for (const term of pattern.args) assertFiniteNumericTerm(term);
  for (const term of fact.args) assertFiniteNumericTerm(term);
  return matchArgs(pattern.args, fact.args, {}) !== null;
}

function substituteHead(head: Literal, env: Bindings): Term[] {
  return head.args.map((arg) => resolve(arg, env));
}

function addTuple(db: Database, key: string, entry: FactEntry): boolean {
  let relation = db.get(key);
  if (!relation) {
    relation = { tuples: new Map() };
    db.set(key, relation);
  }
  const tk = tupleKey(entry.tuple);
  if (relation.tuples.has(tk)) return false;
  relation.tuples.set(tk, entry);
  const index = relation.byFirstArgument;
  if (index !== undefined && entry.tuple.length > 0) {
    const firstKey = firstArgumentKey(entry.tuple[0]);
    if (firstKey === undefined) return true;
    const bucket = index.get(firstKey);
    if (bucket) bucket.push(entry);
    else index.set(firstKey, [entry]);
  }
  return true;
}

function groundValue(term: Term): string | number {
  if (term.type === 'atom') return term.value;
  if (term.type === 'num') {
    assertFiniteNumericTerm(term);
    return term.value;
  }
  throw new Error('proof serialization requires grounded facts');
}

interface ProofBudget {
  maxNodes: number;
  emittedNodes: number;
  maxAggregateProofRows: number;
}

function isAbsenceProof(proof: ProofRef): proof is AbsenceProof {
  return 'negated' in proof;
}

function serializeProof(
  entry: FactEntry,
  maxProofDepth: number,
  budget: ProofBudget,
  depth?: number
): DerivationProof;
function serializeProof(
  entry: AbsenceProof,
  maxProofDepth: number,
  budget: ProofBudget,
  depth?: number
): AbsenceProof;
function serializeProof(
  entry: ProofRef,
  maxProofDepth: number,
  budget: ProofBudget,
  depth?: number
): ProofStep;
function serializeProof(
  entry: ProofRef,
  maxProofDepth: number,
  budget: ProofBudget,
  depth = 1
): ProofStep {
  if (depth > maxProofDepth) {
    throw new EngineLimitError(`proof exceeded max depth ${maxProofDepth}`);
  }
  if (++budget.emittedNodes > budget.maxNodes) {
    throw new EngineLimitError(`proof exceeded max nodes ${budget.maxNodes}`);
  }
  if (isAbsenceProof(entry)) return { ...entry, pattern: [...entry.pattern] };
  const proof: DerivationProof = {
    predicate: entry.predicate,
    values: entry.tuple.map(groundValue),
  };
  if (entry.rule !== undefined) proof.rule = entry.rule;
  if (entry.because && entry.because.length > 0) {
    proof.because = entry.because.map((child) =>
      serializeProof(child, maxProofDepth, budget, depth + 1)
    );
  }
  if (entry.aggregate !== undefined) {
    proof.aggregate = serializeAggregateProof(
      entry.aggregate.spec,
      entry.aggregate.result,
      maxProofDepth,
      budget,
      depth + 1
    );
  }
  return proof;
}

function cloneProofStep(
  proof: ProofStep,
  maxProofDepth: number,
  budget: ProofBudget,
  depth = 1
): ProofStep {
  if (depth > maxProofDepth) {
    throw new EngineLimitError(`proof exceeded max depth ${maxProofDepth}`);
  }
  if (++budget.emittedNodes > budget.maxNodes) {
    throw new EngineLimitError(`proof exceeded max nodes ${budget.maxNodes}`);
  }
  if ('negated' in proof) return { ...proof, pattern: [...proof.pattern] };
  return {
    predicate: proof.predicate,
    values: [...proof.values],
    ...(proof.rule === undefined ? {} : { rule: proof.rule }),
    ...(proof.because === undefined
      ? {}
      : {
          because: proof.because.map((child) =>
            cloneProofStep(child, maxProofDepth, budget, depth + 1)
          ),
        }),
    ...(proof.aggregate === undefined
      ? {}
      : {
          aggregate: cloneAggregateProof(
            proof.aggregate,
            maxProofDepth,
            budget,
            depth + 1
          ),
        }),
  };
}

function cloneAggregateProof(
  proof: AggregateProof,
  maxProofDepth: number,
  budget: ProofBudget,
  depth: number
): AggregateProof {
  if (depth > maxProofDepth) {
    throw new EngineLimitError(`proof exceeded max depth ${maxProofDepth}`);
  }
  if (++budget.emittedNodes > budget.maxNodes) {
    throw new EngineLimitError(`proof exceeded max nodes ${budget.maxNodes}`);
  }
  if (proof.contributors.length > budget.maxAggregateProofRows) {
    throw new EngineLimitError(
      `aggregate proof exceeded ${budget.maxAggregateProofRows} contributor rows`
    );
  }
  return {
    aggregated: true,
    op: proof.op,
    input: proof.input,
    as: proof.as,
    value: proof.value,
    contributors: proof.contributors.map((contributor) => ({
      bindings: { ...contributor.bindings },
      proofs: contributor.proofs.map((child) =>
        cloneProofStep(child, maxProofDepth, budget, depth + 1)
      ),
    })),
    ...(proof.witnessPositions === undefined
      ? {}
      : { witnessPositions: [...proof.witnessPositions] }),
  };
}

function rowKey(bindings: Bindings): string {
  return JSON.stringify(
    Object.entries(bindings)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, term]) => [name, keyPart(term)])
  );
}

interface DerivedDatabase {
  db: Database;
  predicateStrata: Map<string, number>;
  rulesByPredicate: Map<string, StratifiedRule[]>;
  ruleIdentity: Map<number, string>;
}

interface EnumerationBudget {
  maxSteps: number;
  steps: number;
}

interface EnumerationContext {
  db: Database;
  predicateStrata: Map<string, number>;
  rulesByPredicate: Map<string, StratifiedRule[]>;
  ruleIdentity: Map<number, string>;
  budget: EnumerationBudget;
  maxProofDepth: number;
  maxProofNodes: number;
  maxAggregateProofRows: number;
  lookup: RelationLookupContext;
}

interface ProofRowAccumulator {
  bindings: Bindings;
  proofs: ProofStep[][];
  proofKeys: Set<string>;
}

function consumeEnumerationStep(budget: EnumerationBudget): void {
  if (++budget.steps > budget.maxSteps) {
    throw new EngineLimitError(`proof enumeration exceeded ${budget.maxSteps} steps`);
  }
}

function factRefKey(entry: FactEntry): string {
  return `${entry.predicate}:${tupleKey(entry.tuple)}`;
}

function structuralAggregateProofKey(
  proof: AggregateProof,
  ruleIdentity: ReadonlyMap<number, string>
): unknown {
  return {
    aggregated: true,
    op: proof.op,
    input: proof.input,
    as: proof.as,
    value: proof.value,
    contributors: proof.contributors.map((contributor) => ({
      bindings: Object.entries(contributor.bindings).sort(([left], [right]) =>
        left.localeCompare(right)
      ),
      proofs: contributor.proofs.map((child) =>
        structuralProofKey(child, ruleIdentity)
      ),
    })),
    witnessPositions: proof.witnessPositions,
  };
}

function structuralProofKey(
  proof: ProofStep,
  ruleIdentity: ReadonlyMap<number, string>
): string {
  if ('negated' in proof) {
    return JSON.stringify({
      negated: true,
      predicate: proof.predicate,
      pattern: proof.pattern,
      stratum: proof.stratum,
    });
  }
  return JSON.stringify({
    predicate: proof.predicate,
    values: proof.values,
    rule:
      proof.rule === undefined
        ? undefined
        : (ruleIdentity.get(proof.rule) ?? `rule:${proof.rule}`),
    because: proof.because?.map((child) => structuralProofKey(child, ruleIdentity)),
    aggregate:
      proof.aggregate === undefined
        ? undefined
        : structuralAggregateProofKey(proof.aggregate, ruleIdentity),
  });
}

function proofVectorKey(
  proofs: ProofStep[],
  ruleIdentity: ReadonlyMap<number, string>
): string {
  return JSON.stringify(proofs.map((proof) => structuralProofKey(proof, ruleIdentity)));
}

interface AggregateRuleGroup {
  env: Bindings;
  rows: QueryRowRef[];
}

function expressionVariables(expression: ScalarExpression): string[] {
  if (!isArithmeticExpression(expression)) {
    return expression.type === 'var' ? [expression.name] : [];
  }
  return expression.kind === 'unary'
    ? expressionVariables(expression.operand)
    : [
        ...expressionVariables(expression.left),
        ...expressionVariables(expression.right),
      ];
}

function aggregateGoalVariables(goal: Goal): string[] {
  if (isComparison(goal)) {
    return [
      ...expressionVariables(goal.left),
      ...expressionVariables(goal.right),
    ];
  }
  const literal = isNegation(goal) ? goal.not : goal;
  return literal.args.flatMap((term) =>
    term.type === 'var' ? [term.name] : []
  );
}

function assertAggregateRuleSafety(rule: AggregateRuleClause): void {
  const operator: unknown = rule.aggregate.op;
  if (
    operator !== 'count' &&
    operator !== 'sum' &&
    operator !== 'min' &&
    operator !== 'max'
  ) {
    throw new EngineSafetyError(
      `unsupported aggregate rule operator '${String(rule.aggregate.op)}'`
    );
  }
  if (rule.aggregate.op === 'count' && rule.aggregate.input !== '*') {
    throw new EngineSafetyError('count aggregation must use count(*)');
  }
  if (rule.aggregate.op !== 'count' && rule.aggregate.input === '*') {
    throw new EngineSafetyError(
      `${rule.aggregate.op} aggregate input must be a variable`
    );
  }
  if (rule.head.args.some((term) => term.type === 'wildcard')) {
    throw new EngineSafetyError('rule heads may not contain wildcards');
  }

  const bound = new Set<string>();
  const allBodyVariables = new Set<string>();
  let positiveRelations = 0;
  for (const goal of rule.body) {
    const variables = aggregateGoalVariables(goal);
    for (const variable of variables) allBodyVariables.add(variable);
    if (!isComparison(goal) && !isNegation(goal)) {
      positiveRelations++;
      for (const variable of variables) bound.add(variable);
      continue;
    }
    for (const variable of variables) {
      if (!bound.has(variable)) {
        throw new EngineSafetyError(
          `range restriction violated: variable ${variable} must be bound by an earlier positive aggregate relation`
        );
      }
    }
  }
  if (positiveRelations === 0) {
    throw new EngineSafetyError('aggregate rules require at least one positive relation');
  }
  if (
    rule.aggregate.input !== '*' &&
    !bound.has(rule.aggregate.input)
  ) {
    throw new EngineSafetyError(
      `aggregate input ${rule.aggregate.input} must be bound by a positive relation`
    );
  }
  if (allBodyVariables.has(rule.aggregate.as)) {
    throw new EngineSafetyError(
      `aggregate output ${rule.aggregate.as} must be a fresh variable`
    );
  }

  const headVariables = rule.head.args.flatMap((term) =>
    term.type === 'var' ? [term.name] : []
  );
  if (
    headVariables.filter((name) => name === rule.aggregate.as).length !== 1
  ) {
    throw new EngineSafetyError(
      `aggregate output ${rule.aggregate.as} must appear exactly once in the rule head`
    );
  }
  for (const variable of headVariables) {
    if (variable === rule.aggregate.as) continue;
    if (!bound.has(variable)) {
      throw new EngineSafetyError(
        `range restriction violated: variable ${variable} does not appear in any positive aggregate relation`
      );
    }
  }
}

function deriveAggregateRuleEntries(
  rule: AggregateRuleClause,
  ruleNumber: number,
  db: Database,
  predicateStrata: ReadonlyMap<string, number>,
  lookup: RelationLookupContext,
  maxAggregateRows: number
): FactEntry[] {
  if (!Number.isSafeInteger(maxAggregateRows) || maxAggregateRows < 0) {
    throw new EngineSafetyError('maxAggregateRows must be a non-negative safe integer');
  }
  const outputPositions = rule.head.args.flatMap((term, position) =>
    term.type === 'var' && term.name === rule.aggregate.as ? [position] : []
  );
  if (outputPositions.length !== 1) {
    throw new EngineSafetyError(
      `aggregate output ${rule.aggregate.as} must appear exactly once in the rule head`
    );
  }
  const outputPosition = outputPositions[0];
  const groupHeadTerms = rule.head.args.filter(
    (_term, position) => position !== outputPosition
  );
  const hasGroupVariables = groupHeadTerms.some((term) => term.type === 'var');
  const groups = new Map<string, AggregateRuleGroup>();
  if (!hasGroupVariables) {
    const tuple = groupHeadTerms.map((term) => resolve(term, {}));
    for (const term of tuple) groundValue(term);
    groups.set(tupleKey(tuple), { env: {}, rows: [] });
  }

  const fromDb = (_goalIndex: number, key: string) => db.get(key);
  const stratumOf = (key: string) => predicateStrata.get(key) ?? 0;
  let inspected = 0;
  for (const solution of solveGoals(
    rule.body,
    0,
    {},
    [],
    fromDb,
    stratumOf,
    lookup
  )) {
    if (++inspected > maxAggregateRows) {
      throw new EngineLimitError(`aggregate input exceeded ${maxAggregateRows} rows`);
    }
    const tuple = groupHeadTerms.map((term) => resolve(term, solution.env));
    for (const term of tuple) groundValue(term);
    const key = tupleKey(tuple);
    let group = groups.get(key);
    if (group === undefined) {
      group = { env: { ...solution.env }, rows: [] };
      groups.set(key, group);
    }
    group.rows.push({
      bindings: { ...solution.env },
      proofs: [...solution.proofs],
    });
  }

  const entries: FactEntry[] = [];
  for (const group of groups.values()) {
    const result = aggregateValue(rule.aggregate, group.rows);
    if (result === null) continue;
    const tuple = rule.head.args.map((term, position) =>
      position === outputPosition ? result.value : resolve(term, group.env)
    );
    for (const term of tuple) groundValue(term);
    entries.push({
      predicate: rule.head.predicate,
      tuple,
      derived: true,
      rule: ruleNumber,
      aggregate: { spec: rule.aggregate, result },
    });
  }
  return entries;
}

function deriveDatabase(
  authoredClauses: Clause[],
  goals: readonly Goal[],
  options: EvaluateOptions,
  lookup: RelationLookupContext,
  collectAlternativeRules = false
): DerivedDatabase {
  const {
    maxFacts = 100_000,
    maxIterations = 10_000,
    maxAggregateRows = 100_000,
  } = options;
  const clauses = expandClosurePredicates(authoredClauses, goals);
  const ordinaryClauses = clauses.filter((clause) => !isIntegrityConstraint(clause));
  for (const clause of ordinaryClauses) {
    if (isAggregateRule(clause)) assertAggregateRuleSafety(clause);
    for (const term of clause.head.args) assertFiniteNumericTerm(term);
    assertGoalsNumericSafety(clause.body);
  }
  const facts = ordinaryClauses.filter((c) => c.body.length === 0);
  const stratified = stratifyProgram(ordinaryClauses);
  const rulesByPredicate = new Map<string, StratifiedRule[]>();
  const ruleIdentity = new Map<number, string>();
  if (collectAlternativeRules) {
    for (const rules of stratified.strata) {
      for (const rule of rules) {
        const key = predKey(rule.clause.head);
        const ordered = rulesByPredicate.get(key);
        if (ordered) {
          ordered.push(rule);
        } else {
          rulesByPredicate.set(key, [rule]);
        }
        ruleIdentity.set(rule.ruleNumber, canonicalKey(rule.clause));
      }
    }
  }

  const db: Database = new Map();
  let totalFacts = 0;

  for (const fact of facts) {
    const key = litKey(fact.head.predicate, fact.head.args.length);
    const entry: FactEntry = {
      predicate: fact.head.predicate,
      tuple: fact.head.args,
      derived: false,
    };
    if (addTuple(db, key, entry)) {
      if (++totalFacts > maxFacts) {
        throw new EngineLimitError(`derivation exceeded ${maxFacts} facts`);
      }
    }
  }

  let iterations = 0;
  const stratumOf = (key: string) => stratified.predicateStrata.get(key) ?? 0;
  for (const rules of stratified.strata) {
    if (rules.length === 0) continue;
    let delta: Database = new Map(db);
    let firstRound = true;
    while (firstRound || delta.size > 0) {
      if (++iterations > maxIterations) {
        throw new EngineLimitError(`derivation exceeded ${maxIterations} iterations`);
      }
      const newDelta: Database = new Map();
      for (const { clause: rule, ruleNumber } of rules) {
        const headKey = litKey(rule.head.predicate, rule.head.args.length);
        if (isAggregateRule(rule)) {
          if (!firstRound) continue;
          for (const entry of deriveAggregateRuleEntries(
            rule,
            ruleNumber,
            db,
            stratified.predicateStrata,
            lookup,
            maxAggregateRows
          )) {
            if (addTuple(db, headKey, entry)) {
              addTuple(newDelta, headKey, entry);
              if (++totalFacts > maxFacts) {
                throw new EngineLimitError(`derivation exceeded ${maxFacts} facts`);
              }
            }
          }
          continue;
        }
        const positiveIndexes = rule.body
          .map((goal, i) => (isComparison(goal) || isNegation(goal) ? -1 : i))
          .filter((i) => i >= 0);
        const deltaPositions =
          positiveIndexes.length > 0 ? positiveIndexes : firstRound ? [-1] : [];
        for (const deltaPos of deltaPositions) {
          const source = (goalIndex: number, key: string) =>
            deltaPos >= 0 && goalIndex === deltaPos ? delta.get(key) : db.get(key);
          for (const solution of solveGoals(
            rule.body,
            0,
            {},
            [],
            source,
            stratumOf,
            lookup
          )) {
            const tuple = substituteHead(rule.head, solution.env);
            const entry: FactEntry = {
              predicate: rule.head.predicate,
              tuple,
              derived: true,
              rule: ruleNumber,
              because: solution.proofs.length > 0 ? solution.proofs : undefined,
            };
            if (addTuple(db, headKey, entry)) {
              addTuple(newDelta, headKey, entry);
              if (++totalFacts > maxFacts) {
                throw new EngineLimitError(`derivation exceeded ${maxFacts} facts`);
              }
            }
          }
        }
      }
      delta = newDelta;
      firstRound = false;
    }
  }

  return {
    db,
    predicateStrata: stratified.predicateStrata,
    rulesByPredicate,
    ruleIdentity,
  };
}

function queryBindingsWithProofRefs(
  db: Database,
  query: Goal[],
  maxRows: number,
  predicateStrata: Map<string, number>,
  lookup: RelationLookupContext,
  project?: readonly string[]
): QueryRowRef[] {
  const results: QueryRowRef[] = [];
  const seen = new Set<string>();
  const fromDb = (_: number, key: string) => db.get(key);
  const stratumOf = (key: string) => predicateStrata.get(key) ?? 0;

  for (const solution of solveGoals(query, 0, {}, [], fromDb, stratumOf, lookup)) {
    const bindings = projectedBindings(solution.env, project);
    const key = rowKey(bindings);
    if (seen.has(key)) continue;
    seen.add(key);
    results.push({ bindings, proofs: solution.proofs });
    if (results.length >= maxRows) break;
  }

  return results;
}

function projectedBindings(
  bindings: Bindings,
  project: readonly string[] | undefined
): Bindings {
  if (project === undefined) return { ...bindings };
  const selected: Bindings = {};
  for (const name of project) {
    const value = bindings[name];
    if (value === undefined) {
      throw new EngineSafetyError(`projected variable ${name} was not bound`);
    }
    selected[name] = value;
  }
  return selected;
}

function aggregateInputRows(
  db: Database,
  query: Goal[],
  maxAggregateRows: number,
  predicateStrata: Map<string, number>,
  lookup: RelationLookupContext
): QueryRowRef[] {
  if (!Number.isSafeInteger(maxAggregateRows) || maxAggregateRows < 0) {
    throw new EngineSafetyError('maxAggregateRows must be a non-negative safe integer');
  }
  const results: QueryRowRef[] = [];
  const fromDb = (_: number, key: string) => db.get(key);
  const stratumOf = (key: string) => predicateStrata.get(key) ?? 0;
  let inspected = 0;

  for (const solution of solveGoals(query, 0, {}, [], fromDb, stratumOf, lookup)) {
    if (++inspected > maxAggregateRows) {
      throw new EngineLimitError(`aggregate input exceeded ${maxAggregateRows} rows`);
    }
    const bindings: Bindings = {};
    for (const [name, term] of Object.entries(solution.env)) bindings[name] = term;
    results.push({ bindings, proofs: solution.proofs });
  }

  return results;
}

interface AggregateResultRef {
  bindings: Bindings;
  value: Term;
  contributors: QueryRowRef[];
  witnessPositions?: number[];
}

function aggregateValue(
  query: AggregateRuleSpec,
  rows: QueryRowRef[]
): AggregateResultRef | null {
  if (query.op === 'count') {
    const value: Term = { type: 'num', value: rows.length };
    return { bindings: { [query.as]: value }, value, contributors: rows };
  }
  if (rows.length === 0) return null;

  const values = rows.map(({ bindings }) => {
    const value = bindings[query.input];
    if (value === undefined || value.type === 'var' || value.type === 'wildcard') {
      throw new EngineSafetyError(
        `aggregate input ${query.input} was not grounded by the query`
      );
    }
    return value;
  });

  let value: Term;
  if (query.op === 'sum') {
    if (values.some((term) => term.type !== 'num')) {
      throw new EngineSafetyError('sum aggregation requires numeric input values');
    }
    const sum = values.reduce((total, term) => total + (term as Term & { type: 'num' }).value, 0);
    if (!Number.isFinite(sum)) {
      throw new EngineSafetyError('sum aggregation produced a non-finite result');
    }
    value = { type: 'num', value: sum };
  } else {
    const type = values[0].type;
    if (values.some((term) => term.type !== type)) {
      throw new EngineSafetyError(`${query.op} aggregation requires one scalar type`);
    }
    value = values[0];
    for (const candidate of values.slice(1)) {
      if (comparisonHolds(query.op === 'min' ? '<' : '>', candidate, value)) {
        value = candidate;
      }
    }
  }

  const witnessPositions =
    query.op === 'min' || query.op === 'max'
      ? values.flatMap((candidate, index) => (termEq(candidate, value) ? [index] : []))
      : undefined;
  return {
    bindings: { [query.as]: value },
    value,
    contributors: rows,
    ...(witnessPositions === undefined ? {} : { witnessPositions }),
  };
}

function serializeAggregateProof(
  query: AggregateRuleSpec,
  result: AggregateResultRef,
  maxProofDepth: number,
  budget: ProofBudget,
  depth = 1
): AggregateProof {
  if (depth > maxProofDepth) {
    throw new EngineLimitError(`proof exceeded max depth ${maxProofDepth}`);
  }
  if (++budget.emittedNodes > budget.maxNodes) {
    throw new EngineLimitError(`proof exceeded max nodes ${budget.maxNodes}`);
  }
  if (result.contributors.length > budget.maxAggregateProofRows) {
    throw new EngineLimitError(
      `aggregate proof exceeded ${budget.maxAggregateProofRows} contributor rows`
    );
  }
  return {
    aggregated: true,
    op: query.op,
    input: query.input,
    as: query.as,
    value: groundValue(result.value),
    contributors: result.contributors.map(({ bindings, proofs }) => ({
      bindings: { ...bindings },
      proofs: proofs.map((proof) =>
        serializeProof(proof, maxProofDepth, budget, depth + 1)
      ),
    })),
    ...(result.witnessPositions === undefined
      ? {}
      : { witnessPositions: [...result.witnessPositions] }),
  };
}

function serializeProofForEnumeration(
  entry: ProofRef,
  context: EnumerationContext
): ProofStep {
  return serializeProof(
    entry,
    context.maxProofDepth,
    {
      maxNodes: context.maxProofNodes,
      emittedNodes: 0,
      maxAggregateProofRows: context.maxAggregateProofRows,
    }
  );
}

function addProofVector(
  row: ProofRowAccumulator,
  proofs: ProofStep[],
  maxProofsPerRow: number,
  ruleIdentity: ReadonlyMap<number, string>
): void {
  const key = proofVectorKey(proofs, ruleIdentity);
  if (row.proofKeys.has(key)) return;
  if (row.proofs.length >= maxProofsPerRow) {
    throw new EngineLimitError(
      `proof alternatives exceeded maxProofsPerRow ${maxProofsPerRow}`
    );
  }
  row.proofKeys.add(key);
  row.proofs.push(proofs);
}

function enumerateProofVectors(
  proofs: ProofRef[],
  context: EnumerationContext,
  trail: ReadonlySet<string>,
  depth: number,
  onProofs: (proofs: ProofStep[]) => boolean | void
): boolean {
  const current: ProofStep[] = [];

  const visit = (index: number): boolean => {
    if (index >= proofs.length) {
      consumeEnumerationStep(context.budget);
      return onProofs([...current]) === true;
    }
    return enumerateProofChoices(proofs[index], context, trail, depth, (proof) => {
      current.push(proof);
      try {
        return visit(index + 1);
      } finally {
        current.pop();
      }
    });
  };

  return visit(0);
}

function enumerateProofChoices(
  proof: ProofRef,
  context: EnumerationContext,
  trail: ReadonlySet<string>,
  depth: number,
  onProof: (proof: ProofStep) => boolean | void
): boolean {
  if (depth > context.maxProofDepth) {
    throw new EngineLimitError(`proof exceeded max depth ${context.maxProofDepth}`);
  }
  consumeEnumerationStep(context.budget);
  if (isAbsenceProof(proof)) {
    return onProof({ ...proof, pattern: [...proof.pattern] }) === true;
  }

  const currentKey = factRefKey(proof);
  if (trail.has(currentKey)) return false;
  const nextTrail = new Set(trail);
  nextTrail.add(currentKey);
  const seen = new Set<string>();

  const primary = serializeProofForEnumeration(proof, context);
  seen.add(structuralProofKey(primary, context.ruleIdentity));
  if (onProof(primary) === true) return true;

  const rules =
    context.rulesByPredicate.get(litKey(proof.predicate, proof.tuple.length)) ?? [];
  if (rules.length === 0) return false;
  if (rules.some(({ clause }) => isAggregateRule(clause))) {
    throw new EngineSafetyError(
      'alternative proofs through aggregate-derived rules are not supported'
    );
  }

  const tupleValues = proof.tuple.map(groundValue);
  const fromDb = (_: number, key: string) => context.db.get(key);
  const stratumOf = (key: string) => context.predicateStrata.get(key) ?? 0;

  for (const { clause, ruleNumber } of rules) {
    consumeEnumerationStep(context.budget);
    const seeded = matchArgs(clause.head.args, proof.tuple, {});
    if (seeded === null) continue;
    for (const solution of solveGoals(
      clause.body,
      0,
      seeded,
      [],
      fromDb,
      stratumOf,
      context.lookup
    )) {
      consumeEnumerationStep(context.budget);
      const stopped = enumerateProofVectors(
        solution.proofs,
        context,
        nextTrail,
        depth + 1,
        (because) => {
          const derived: DerivationProof = {
            predicate: proof.predicate,
            values: tupleValues,
            rule: ruleNumber,
            ...(because.length === 0 ? {} : { because }),
          };
          const key = structuralProofKey(derived, context.ruleIdentity);
          if (seen.has(key)) return false;
          seen.add(key);
          return onProof(derived);
        }
      );
      if (stopped) return true;
    }
  }

  return false;
}

function queryBindingsWithAlternativeProofs(
  db: Database,
  query: Goal[],
  maxRows: number,
  predicateStrata: Map<string, number>,
  rulesByPredicate: Map<string, StratifiedRule[]>,
  ruleIdentity: Map<number, string>,
  alternativeOptions: AlternativeProofOptions,
  maxProofDepth: number,
  maxProofNodes: number,
  maxAggregateProofRows: number,
  lookup: RelationLookupContext,
  project?: readonly string[]
): ProofRowAccumulator[] {
  if (maxRows <= 0) return [];
  const rows: ProofRowAccumulator[] = [];
  const rowByKey = new Map<string, ProofRowAccumulator>();
  const fromDb = (_: number, key: string) => db.get(key);
  const stratumOf = (key: string) => predicateStrata.get(key) ?? 0;
  const context: EnumerationContext = {
    db,
    predicateStrata,
    rulesByPredicate,
    ruleIdentity,
    budget: {
      maxSteps: alternativeOptions.maxProofEnumerationSteps,
      steps: 0,
    },
    maxProofDepth,
    maxProofNodes,
    maxAggregateProofRows,
    lookup,
  };

  for (const solution of solveGoals(query, 0, {}, [], fromDb, stratumOf, lookup)) {
    consumeEnumerationStep(context.budget);
    const bindings = projectedBindings(solution.env, project);
    const key = rowKey(bindings);
    let row = rowByKey.get(key);
    if (!row) {
      if (rows.length >= maxRows) continue;
      row = { bindings, proofs: [], proofKeys: new Set() };
      rowByKey.set(key, row);
      rows.push(row);
    }
    enumerateProofVectors(solution.proofs, context, new Set(), 1, (proofs) => {
      addProofVector(
        row,
        proofs,
        alternativeOptions.maxProofsPerRow,
        context.ruleIdentity
      );
      return false;
    });
  }

  return rows;
}

/**
 * Semi-naive bottom-up evaluation: derive the fixpoint of all rules over the
 * facts, then answer the query conjunction against the resulting database.
 */
export function evaluate(
  clauses: Clause[],
  query: Goal[],
  options: EvaluateOptions = {}
): Bindings[] {
  return evaluateRelational(clauses, query, undefined, options);
}

function evaluateRelational(
  clauses: Clause[],
  query: Goal[],
  project: readonly string[] | undefined,
  options: EvaluateOptions
): Bindings[] {
  const { maxRows = 1000 } = options;
  const lookup = relationLookupContext(options);
  assertGoalsNumericSafety(query);
  const { db, predicateStrata } = deriveDatabase(clauses, query, options, lookup);
  return queryBindingsWithProofRefs(
    db,
    query,
    maxRows,
    predicateStrata,
    lookup,
    project
  ).map(({ bindings }) => bindings);
}

export function evaluateWithProof(
  clauses: Clause[],
  query: Goal[],
  options: EvaluateOptions = {}
): ExplainedBindings[] {
  return evaluateRelationalWithProof(clauses, query, undefined, options);
}

function evaluateRelationalWithProof(
  clauses: Clause[],
  query: Goal[],
  project: readonly string[] | undefined,
  options: EvaluateOptions
): ExplainedBindings[] {
  const alternativeOptions = resolveAlternativeProofOptions(options);
  const {
    maxRows = 1000,
    maxProofDepth = 128,
    maxProofNodes = 100_000,
    maxAggregateProofRows = 256,
  } = options;
  assertAggregateProofRowLimit(maxAggregateProofRows);
  const lookup = relationLookupContext(options);
  assertGoalsNumericSafety(query);
  const { db, predicateStrata, rulesByPredicate, ruleIdentity } = deriveDatabase(
    clauses,
    query,
    options,
    lookup,
    alternativeOptions.maxProofsPerRow > DEFAULT_MAX_PROOFS_PER_ROW
  );
  const proofBudget: ProofBudget = {
    maxNodes: maxProofNodes,
    emittedNodes: 0,
    maxAggregateProofRows,
  };
  if (alternativeOptions.maxProofsPerRow === DEFAULT_MAX_PROOFS_PER_ROW) {
    return queryBindingsWithProofRefs(
      db,
      query,
      maxRows,
      predicateStrata,
      lookup,
      project
    ).map(
      ({ bindings, proofs }) => ({
        bindings,
        proofs: proofs.map((proof) => serializeProof(proof, maxProofDepth, proofBudget)),
      })
    );
  }
  return queryBindingsWithAlternativeProofs(
    db,
    query,
    maxRows,
    predicateStrata,
    rulesByPredicate,
    ruleIdentity,
    alternativeOptions,
    maxProofDepth,
    maxProofNodes,
    maxAggregateProofRows,
    lookup,
    project
  ).map(({ bindings, proofs }) => ({
    bindings,
    proofs: proofs[0].map((proof) => cloneProofStep(proof, maxProofDepth, proofBudget)),
    ...(proofs.length > 1
      ? {
          alternativeProofs: proofs
            .slice(1)
            .map((proofVector) =>
              proofVector.map((proof) => cloneProofStep(proof, maxProofDepth, proofBudget))
            ),
        }
      : {}),
  }));
}

/** Evaluate either a relational query or one exact scalar reduction over its full result. */
export function evaluateQuerySpec(
  clauses: Clause[],
  query: QuerySpec,
  options: EvaluateOptions = {}
): Bindings[] {
  if (query.kind === 'relational') {
    return evaluateRelational(clauses, query.goals, query.project, options);
  }
  const { maxRows = 1000, maxAggregateRows = 100_000 } = options;
  const lookup = relationLookupContext(options);
  if (maxRows < 1) return [];
  assertGoalsNumericSafety(query.goals);
  const { db, predicateStrata } = deriveDatabase(clauses, query.goals, options, lookup);
  const rows = aggregateInputRows(
    db,
    query.goals,
    maxAggregateRows,
    predicateStrata,
    lookup
  );
  const result = aggregateValue(query, rows);
  return result === null ? [] : [result.bindings];
}

/** Aggregate-aware explanation path; contributors retain their ordered relational proofs. */
export function evaluateQuerySpecWithProof(
  clauses: Clause[],
  query: QuerySpec,
  options: EvaluateOptions = {}
): ExplainedQueryBindings[] {
  const alternativeOptions = resolveAlternativeProofOptions(options);
  if (query.kind === 'relational') {
    return evaluateRelationalWithProof(
      clauses,
      query.goals,
      query.project,
      options
    );
  }
  if (alternativeOptions.maxProofsPerRow > DEFAULT_MAX_PROOFS_PER_ROW) {
    throw new EngineSafetyError('alternative proofs are relational-only');
  }
  const {
    maxRows = 1000,
    maxAggregateRows = 100_000,
    maxAggregateProofRows = 256,
    maxProofDepth = 128,
    maxProofNodes = 100_000,
  } = options;
  const lookup = relationLookupContext(options);
  if (maxRows < 1) return [];
  assertGoalsNumericSafety(query.goals);
  const { db, predicateStrata } = deriveDatabase(clauses, query.goals, options, lookup);
  const rows = aggregateInputRows(
    db,
    query.goals,
    maxAggregateRows,
    predicateStrata,
    lookup
  );
  const result = aggregateValue(query, rows);
  if (result === null) return [];
  assertAggregateProofRowLimit(maxAggregateProofRows);
  if (result.contributors.length > maxAggregateProofRows) {
    throw new EngineLimitError(
      `aggregate proof exceeded ${maxAggregateProofRows} contributor rows`
    );
  }
  const proofBudget: ProofBudget = {
    maxNodes: maxProofNodes,
    emittedNodes: 0,
    maxAggregateProofRows,
  };
  return [
    {
      bindings: result.bindings,
      proofs: [serializeAggregateProof(query, result, maxProofDepth, proofBudget)],
    },
  ];
}

export function materializeWithProof(
  clauses: Clause[],
  options: EvaluateOptions = {}
): MaterializedFactWithProof[] {
  const {
    maxProofDepth = 128,
    maxProofNodes = 100_000,
    maxAggregateProofRows = 256,
  } = options;
  assertAggregateProofRowLimit(maxAggregateProofRows);
  const lookup = relationLookupContext(options);
  const { db } = deriveDatabase(clauses, [], options, lookup);
  const facts: MaterializedFactWithProof[] = [];
  const proofBudget: ProofBudget = {
    maxNodes: maxProofNodes,
    emittedNodes: 0,
    maxAggregateProofRows,
  };

  for (const relation of db.values()) {
    for (const entry of relation.tuples.values()) {
      facts.push({
        predicate: entry.predicate,
        values: entry.tuple.map(groundValue),
        derived: entry.derived,
        proof: serializeProof(entry, maxProofDepth, proofBudget),
      });
    }
  }

  return facts;
}

/** Materialize the bounded fixpoint without paying proof serialization cost. */
export function materialize(
  clauses: Clause[],
  options: EvaluateOptions = {}
): MaterializedFact[] {
  const lookup = relationLookupContext(options);
  const { db } = deriveDatabase(clauses, [], options, lookup);
  const facts: MaterializedFact[] = [];
  for (const relation of db.values()) {
    for (const entry of relation.tuples.values()) {
      facts.push({
        predicate: entry.predicate,
        values: entry.tuple.map(groundValue),
        derived: entry.derived,
      });
    }
  }
  return facts;
}
