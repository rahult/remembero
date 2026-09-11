import {
  isArithmeticExpression,
  isComparison,
  isNegation,
  parseQuerySpec,
  parseQueryProgram,
  type Goal,
  type ScalarExpression,
  type Term,
} from '../engine/index.js';
import type { QueryPromptVariant } from '../llm/prompts.js';
import type { RecallStatus } from '../llm/pipeline.js';
import type { TrustViewMode } from '../knowledge/trust.js';
import type { LlmUsageTotals } from '../llm/client.js';

export const RECALL_EVAL_DISTRACTOR_COUNT = 100;
const RECALL_EVAL_DISTRACTORS = Array.from(
  { length: RECALL_EVAL_DISTRACTOR_COUNT },
  (_, index) =>
    `eval_noise_${String(index).padStart(3, '0')}(subject_${index}, value_${index}).`,
).join('\n');

export const RECALL_EVAL_PROGRAM = `
works_at(ava, northwind).
works_at(ben, contoso).
works_at(cy, globex).
works_at(rahul, acme).
works_at(mira, acme).
works_at(chen, initech).
lives_in(ava, perth).
lives_in(ben, adelaide).
lives_in(cy, brisbane).
lives_in(rahul, melbourne).
lives_in(mira, sydney).
lives_in(chen, singapore).
lives_in(dr_chen, 'New York').
dentist(ava, dr_lee).
dentist(ben, dr_patel).
dentist(cy, dr_ng).
dentist(rahul, dr_chen).
uses_language(orchard, go).
uses_language(beacon, python).
uses_language(cedar, java).
uses_language(atlas, rust).
uses_language(kiln, typescript).
project_owner(orchard, ava).
project_owner(beacon, ben).
project_owner(cedar, cy).
project_owner(atlas, rahul).
likes_cuisine(ava, italian).
likes_cuisine(ben, mexican).
likes_cuisine(cy, vietnamese).
likes_cuisine(rahul, thai).
birth_year(ava, 1991).
birth_year(ben, 1992).
birth_year(cy, 1996).
birth_year(rahul, 1985).
birth_year(mira, 1994).
birth_year(chen, 1978).
parent(uma, victor).
parent(wendy, xavier).
parent(yara, zane).
parent(alice, bob).
parent(bob, carol).
parent(carol, dan).
colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.
company_size(Company, Count) :- count(*) as Count where works_at(Person, Company).
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
${RECALL_EVAL_DISTRACTORS}
rembero_tentative('favorite_color(rahul, blue).').
`;

export type ExpectedQueryDecision = 'required' | 'unanswerable';

export interface RecallEvalCase {
  id: string;
  question: string;
  expectedQuery: ExpectedQueryDecision;
  /** Values in each expected binding row. Variable names are intentionally ignored. */
  expectedRows: string[][];
  tags: string[];
  trustMode?: TrustViewMode;
}

export const RECALL_EVAL_CASES: RecallEvalCase[] = [
  {
    id: 'direct_employer',
    question: 'Where does Rahul work?',
    expectedQuery: 'required',
    expectedRows: [['acme']],
    tags: ['direct', 'single-answer'],
  },
  {
    id: 'lexical_employer',
    question: 'Who is Mira employed by?',
    expectedQuery: 'required',
    expectedRows: [['acme']],
    tags: ['direct', 'lexical'],
  },
  {
    id: 'multiple_workers',
    question: 'Who works at Acme?',
    expectedQuery: 'required',
    expectedRows: [['rahul'], ['mira']],
    tags: ['direct', 'multi-answer'],
  },
  {
    id: 'reusable_company_size',
    question: 'How many people work at Acme?',
    expectedQuery: 'required',
    expectedRows: [['2']],
    tags: ['aggregate-rule', 'derived', 'single-answer'],
  },
  {
    id: 'derived_colleague',
    question: "Who are Rahul's colleagues?",
    expectedQuery: 'required',
    expectedRows: [['mira']],
    tags: ['derived', 'inequality'],
  },
  {
    id: 'conjunctive_location',
    question: 'Who both works at Acme and lives in Melbourne?',
    expectedQuery: 'required',
    expectedRows: [['rahul']],
    tags: ['join', 'constraint'],
  },
  {
    id: 'dentist_city_join',
    question: "Which city does Rahul's dentist live in?",
    expectedQuery: 'required',
    expectedRows: [["'New York'"]],
    tags: ['join', 'quoted-atom', 'projected-answer'],
  },
  {
    id: 'project_language',
    question: 'Which language does the Atlas project use?',
    expectedQuery: 'required',
    expectedRows: [['rust']],
    tags: ['direct', 'lexical'],
  },
  {
    id: 'project_owner',
    question: 'Which project does Rahul own?',
    expectedQuery: 'required',
    expectedRows: [['atlas']],
    tags: ['direct', 'inverse-direction'],
  },
  {
    id: 'confusable_atlas_owner',
    question: 'Who owns Atlas?',
    expectedQuery: 'required',
    expectedRows: [['rahul']],
    tags: ['direct', 'confusable-nonempty', 'owner-vs-language'],
  },
  {
    id: 'confusable_atlas_language',
    question: 'What language does Atlas use?',
    expectedQuery: 'required',
    expectedRows: [['rust']],
    tags: ['direct', 'confusable-nonempty', 'language-vs-owner'],
  },
  {
    id: 'confusable_rahul_location',
    question: 'Where does Rahul live?',
    expectedQuery: 'required',
    expectedRows: [['melbourne']],
    tags: ['direct', 'confusable-nonempty', 'home-vs-work'],
  },
  {
    id: 'confusable_rahul_employer',
    question: 'Who employs Rahul?',
    expectedQuery: 'required',
    expectedRows: [['acme']],
    tags: ['direct', 'confusable-nonempty', 'work-vs-home'],
  },
  {
    id: 'recursive_ancestor',
    question: "Who are Alice's descendants?",
    expectedQuery: 'required',
    expectedRows: [['bob'], ['carol'], ['dan']],
    tags: ['recursive', 'multi-answer'],
  },
  {
    id: 'grandparent',
    question: "Who is Alice's grandchild?",
    expectedQuery: 'required',
    expectedRows: [['carol']],
    tags: ['derived', 'join'],
  },
  {
    id: 'numeric_comparison',
    question: 'Who was born before 1990?',
    expectedQuery: 'required',
    expectedRows: [['chen'], ['rahul']],
    tags: ['comparison', 'multi-answer', 'projected-answer'],
  },
  {
    id: 'arithmetic_offset',
    question: 'Who was born more than 5 years before Mira?',
    expectedQuery: 'required',
    expectedRows: [['chen'], ['rahul']],
    tags: [
      'comparison',
      'arithmetic',
      'join',
      'multi-answer',
      'projected-answer',
    ],
  },
  {
    id: 'ground_true',
    question: 'Does Chen work at Initech?',
    expectedQuery: 'required',
    expectedRows: [[]],
    tags: ['boolean', 'positive'],
  },
  {
    id: 'ground_false',
    question: 'Does Mira work at Initech?',
    expectedQuery: 'required',
    expectedRows: [],
    tags: ['boolean', 'negative'],
  },
  {
    id: 'ground_true_language',
    question: 'Does the Atlas project use Rust?',
    expectedQuery: 'required',
    expectedRows: [[]],
    tags: ['boolean', 'positive', 'prompt-generalization'],
  },
  {
    id: 'ground_false_language',
    question: 'Does the Kiln project use Rust?',
    expectedQuery: 'required',
    expectedRows: [],
    tags: ['boolean', 'negative', 'prompt-generalization'],
  },
  {
    id: 'missing_entity',
    question: 'Where does Noor work?',
    expectedQuery: 'required',
    expectedRows: [],
    tags: ['empty', 'sample-leakage'],
  },
  {
    id: 'missing_location',
    question: 'Where does Priya live?',
    expectedQuery: 'required',
    expectedRows: [],
    tags: ['empty', 'prompt-generalization'],
  },
  {
    id: 'unexpressible_reason',
    question: 'Why does Rahul work at Acme?',
    expectedQuery: 'unanswerable',
    expectedRows: [],
    tags: ['unanswerable', 'related-schema'],
  },
  {
    id: 'unknown_attribute',
    question: "What is Rahul's favorite color?",
    expectedQuery: 'unanswerable',
    expectedRows: [],
    tags: ['unanswerable', 'unknown-predicate'],
  },
  {
    id: 'tentative_preference',
    question: "What might Rahul's favorite color be?",
    expectedQuery: 'required',
    expectedRows: [['blue']],
    tags: ['tentative', 'direct', 'single-answer'],
    trustMode: 'include_tentative',
  },
];

export interface RecallEvalObservation {
  case: RecallEvalCase;
  model: string;
  variant: QueryPromptVariant;
  status: RecallStatus;
  query: string | null;
  actualRows: string[][];
  llmCalls: number;
  usage: LlmUsageTotals;
  durationMs: number;
  error?: string;
}

export interface RecallEvalScore {
  cases: number;
  accuracy: number;
  answerabilityAccuracy: number;
  precision: number;
  recall: number;
  f1: number;
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  errors: number;
  schemaBudgetExhaustions: number;
  llmCalls: number;
  usageResponses: number;
  costResponses: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedPromptTokens: number;
  reasoningTokens: number;
  costUsd: number;
  durationMs: number;
}

function rowKey(row: readonly string[]): string {
  return JSON.stringify(row);
}

function sameSet(left: Set<string>, right: Set<string>): boolean {
  return (
    left.size === right.size && [...left].every((value) => right.has(value))
  );
}

function visitTerm(term: Term, order: string[], seen: Set<string>): void {
  if (term.type === 'var' && !seen.has(term.name)) {
    seen.add(term.name);
    order.push(term.name);
  }
}

function visitExpression(
  expression: ScalarExpression,
  order: string[],
  seen: Set<string>,
): void {
  if (!isArithmeticExpression(expression)) {
    visitTerm(expression, order, seen);
    return;
  }
  if (expression.kind === 'unary') {
    visitExpression(expression.operand, order, seen);
    return;
  }
  visitExpression(expression.left, order, seen);
  visitExpression(expression.right, order, seen);
}

function visitGoal(goal: Goal, order: string[], seen: Set<string>): void {
  if (isComparison(goal)) {
    visitExpression(goal.left, order, seen);
    visitExpression(goal.right, order, seen);
  } else {
    for (const term of isNegation(goal) ? goal.not.args : goal.args) {
      visitTerm(term, order, seen);
    }
  }
}

export function bindingRows(
  bindings: Record<string, string>[],
  query: string,
): string[][] {
  // the dialect variant records a whole program; its query is the last `?-` line, and the
  // answer columns are the sink rule's head variables in order
  const normalized = parseQueryProgram(query);
  const spec = normalized.query;
  if (normalized.clauses.length > 0 && spec.kind === 'relational') {
    const [first] = spec.goals;
    if (first !== undefined && !isNegation(first) && !isComparison(first)) {
      const order = first.args.flatMap((t) =>
        t.type === 'var' ? [t.name] : [],
      );
      return bindings.map((binding) =>
        order
          .filter((variable) => variable in binding)
          .map((variable) => binding[variable]),
      );
    }
  }
  if (spec.kind === 'aggregate') {
    return bindings.map((binding) =>
      spec.as in binding ? [binding[spec.as]] : [],
    );
  }
  if (spec.project !== undefined) {
    const project = spec.project;
    return bindings.map((binding) =>
      project
        .filter((variable) => variable in binding)
        .map((variable) => binding[variable]),
    );
  }
  const order: string[] = [];
  const seen = new Set<string>();
  for (const goal of spec.goals) visitGoal(goal, order, seen);
  return bindings.map((binding) =>
    order
      .filter((variable) => variable in binding)
      .map((variable) => binding[variable]),
  );
}

export function observationIsCorrect(
  observation: RecallEvalObservation,
): boolean {
  if (observation.error || observation.status === 'schema_budget_exhausted')
    return false;
  const expectedQuery = observation.case.expectedQuery === 'required';
  if ((observation.query !== null) !== expectedQuery) return false;
  return sameSet(
    new Set(observation.case.expectedRows.map(rowKey)),
    new Set(observation.actualRows.map(rowKey)),
  );
}

export function scoreRecallEval(
  observations: RecallEvalObservation[],
): RecallEvalScore {
  let truePositives = 0;
  let falsePositives = 0;
  let falseNegatives = 0;
  let exact = 0;
  let answerabilityCorrect = 0;

  for (const observation of observations) {
    const expected = new Set(observation.case.expectedRows.map(rowKey));
    const actual = new Set(observation.actualRows.map(rowKey));
    for (const row of actual) {
      if (expected.has(row)) truePositives++;
      else falsePositives++;
    }
    for (const row of expected) {
      if (!actual.has(row)) falseNegatives++;
    }
    if (observationIsCorrect(observation)) exact++;
    const expectedQuery = observation.case.expectedQuery === 'required';
    if (
      !observation.error &&
      observation.status !== 'schema_budget_exhausted' &&
      (observation.query !== null) === expectedQuery
    ) {
      answerabilityCorrect++;
    }
  }

  const precision =
    truePositives + falsePositives === 0
      ? 1
      : truePositives / (truePositives + falsePositives);
  const recall =
    truePositives + falseNegatives === 0
      ? 1
      : truePositives / (truePositives + falseNegatives);
  const f1 =
    precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);
  const cases = observations.length;
  return {
    cases,
    accuracy: cases === 0 ? 0 : exact / cases,
    answerabilityAccuracy: cases === 0 ? 0 : answerabilityCorrect / cases,
    precision,
    recall,
    f1,
    truePositives,
    falsePositives,
    falseNegatives,
    errors: observations.filter(
      (observation) => observation.error !== undefined,
    ).length,
    schemaBudgetExhaustions: observations.filter(
      (observation) => observation.status === 'schema_budget_exhausted',
    ).length,
    llmCalls: observations.reduce(
      (total, observation) => total + observation.llmCalls,
      0,
    ),
    usageResponses: observations.reduce(
      (total, observation) => total + observation.usage.usageResponses,
      0,
    ),
    costResponses: observations.reduce(
      (total, observation) => total + observation.usage.costResponses,
      0,
    ),
    promptTokens: observations.reduce(
      (total, observation) => total + observation.usage.promptTokens,
      0,
    ),
    completionTokens: observations.reduce(
      (total, observation) => total + observation.usage.completionTokens,
      0,
    ),
    totalTokens: observations.reduce(
      (total, observation) => total + observation.usage.totalTokens,
      0,
    ),
    cachedPromptTokens: observations.reduce(
      (total, observation) => total + observation.usage.cachedPromptTokens,
      0,
    ),
    reasoningTokens: observations.reduce(
      (total, observation) => total + observation.usage.reasoningTokens,
      0,
    ),
    costUsd: observations.reduce(
      (total, observation) => total + observation.usage.costUsd,
      0,
    ),
    durationMs: observations.reduce(
      (total, observation) => total + observation.durationMs,
      0,
    ),
  };
}
