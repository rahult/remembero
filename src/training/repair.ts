/**
 * Repair-turn training examples for the query dialect.
 *
 * The engine explains an empty result before the model's retry
 * (`emptyResultFeedback`: per-goal match counts and an argument-swap that would
 * return rows). Gemma 4 E2B r16 ignored that feedback on the agent-boundary
 * benchmark because it had never seen a repair turn: it took the "repeat it
 * unchanged if correct" branch every time. These conversations teach both
 * branches: a mutated program (one shared variable moved to the wrong argument
 * position) that runs but matches nothing, followed by the feedback and the
 * verified program; and a correct program whose empty answer is the right
 * answer, followed by feedback and the same program repeated.
 */
import {
  emptyResultFeedback,
  goalVariables,
  parseProgram,
  parseQueryProgram,
  serializeClause,
  serializeGoal,
  type Goal,
  type Literal,
  isComparison,
  isNegation,
} from '../engine/index.js';
import { type Conversation, systemPrompt } from './export.js';
import type { Rng } from './rng.js';
import { executeProgram, type Example } from './verify.js';
import { worldClauses, type World } from './worlds.js';

export interface EmptyMutation {
  /** A program that parses, runs against the world and returns no rows. */
  program: string;
  /** What the engine says about it; never empty. */
  feedback: string;
}

function canonical(program: string): string {
  if (!program.includes(':-')) return program.replace(/\s+/g, ' ').trim();
  return parseProgram(program).map(serializeClause).join('\n');
}

/**
 * Move one shared variable of a body literal to another argument position.
 * Returns the mutated program text, or undefined when the program has no
 * literal with a variable shared across goals.
 */
function swapCandidates(
  bodyOwner: { body: Goal[] } | undefined,
  render: (goals: Goal[]) => string,
): string[] {
  if (bodyOwner === undefined) return [];
  const goals = bodyOwner.body;
  const out: string[] = [];
  goals.forEach((goal, gi) => {
    if (isComparison(goal) || isNegation(goal)) return;
    goal.args.forEach((term, i) => {
      if (term.type !== 'var') return;
      const shared = goals.some(
        (g, j) => j !== gi && goalVariables(g).has(term.name),
      );
      if (!shared) return;
      goal.args.forEach((other, j) => {
        if (j === i) return;
        if (other.type === 'var' && other.name === term.name) return;
        const args = goal.args.slice();
        args[i] = other;
        args[j] = term;
        const swapped: Literal = { predicate: goal.predicate, args };
        out.push(render(goals.map((g, k) => (k === gi ? swapped : g))));
      });
    });
  });
  return out;
}

export function mutateToEmpty(
  world: World,
  example: Example,
  rng: Rng,
): EmptyMutation | undefined {
  if (example.expectEmpty) return undefined;
  const program = example.program;
  if (/\bcount\(|\bsum\(|\bmin\(|\bmax\(|\bavg\(| as [A-Z]/.test(program))
    return undefined;
  let normalized;
  try {
    normalized = parseQueryProgram(program);
  } catch {
    return undefined;
  }
  const worldFacts = worldClauses(world);
  let candidates: string[];
  if (normalized.clauses.length > 0) {
    // rule program: mutate the sink rule's body, keep the other rules as written
    const sink =
      normalized.clauses.find(
        (c) =>
          c.head.predicate === (normalized.query.goals[0] as Literal).predicate,
      ) ?? normalized.clauses[normalized.clauses.length - 1];
    if (sink.body.length < 2) return undefined;
    candidates = swapCandidates(sink, (goals) =>
      normalized.clauses
        .map((c) =>
          c === sink
            ? serializeClause({ ...c, body: goals })
            : serializeClause(c),
        )
        .join('\n'),
    );
  } else {
    if (
      normalized.query.kind !== 'relational' ||
      normalized.query.goals.length < 2
    )
      return undefined;
    const goals = normalized.query.goals;
    candidates = swapCandidates(
      { body: goals },
      (mutated) => `?- ${mutated.map(serializeGoal).join(', ')}.`,
    );
  }
  for (const candidate of rng.shuffle(candidates)) {
    let rows;
    try {
      rows = executeProgram(worldFacts, candidate);
    } catch {
      continue;
    }
    if (rows.length !== 0) continue;
    const mutated = parseQueryProgram(candidate);
    const feedback = emptyResultFeedback(
      [...worldFacts, ...mutated.clauses],
      mutated.query,
      { authored: mutated.clauses },
    );
    if (feedback.length === 0) continue;
    return { program: canonical(candidate), feedback };
  }
  return undefined;
}

/** The two phrasings the product and the benchmark harness use for the repair turn. */
export function repairPrompt(
  queryText: string,
  feedback: string,
  variant: 0 | 1,
): string {
  const detail = feedback.length > 0 ? ` ${feedback}` : '';
  return variant === 0
    ? `The query ${queryText} returned no results.${detail} If it correctly expresses the question, repeat it unchanged: an empty result is valid evidence that no stored fact matches. Try ONE alternative only if the first query mistranslated the question.`
    : `That query ran but returned no rows.${detail} If the query correctly expresses the question, repeat it unchanged; otherwise reply with ONLY a corrected query.`;
}

function queryLine(program: string): string {
  const lines = program.split('\n');
  return lines[lines.length - 1].replace(/^\?-\s*/, '').replace(/\.$/, '');
}

/**
 * A repair conversation for the example, or undefined when none can be built:
 * for an answerable example, the mutated empty program then the fix; for a
 * yes/no-false example, the correct program repeated after the empty result.
 */
export function toRepairConversation(
  world: World,
  example: Example,
  rng: Rng,
): Conversation | undefined {
  const gold = canonical(example.program);
  const variant = rng.int(2) as 0 | 1;
  if (example.expectEmpty) {
    let feedback = '';
    try {
      const normalized = parseQueryProgram(example.program);
      feedback = emptyResultFeedback(
        [...worldClauses(world), ...normalized.clauses],
        normalized.query,
        { authored: normalized.clauses },
      );
    } catch {
      return undefined;
    }
    return {
      messages: [
        { role: 'system', content: systemPrompt(world) },
        { role: 'user', content: example.question },
        { role: 'assistant', content: gold },
        {
          role: 'user',
          content: repairPrompt(queryLine(gold), feedback, variant),
        },
        { role: 'assistant', content: gold },
      ],
    };
  }
  const mutated = mutateToEmpty(world, example, rng);
  if (mutated === undefined) return undefined;
  return {
    messages: [
      { role: 'system', content: systemPrompt(world) },
      { role: 'user', content: example.question },
      { role: 'assistant', content: mutated.program },
      {
        role: 'user',
        content: repairPrompt(
          queryLine(mutated.program),
          mutated.feedback,
          variant,
        ),
      },
      { role: 'assistant', content: gold },
    ],
  };
}
