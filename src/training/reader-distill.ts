/**
 * Reader distillation data (reader v3).
 *
 * Reader v1 and v2 learned the generator's answer shapes without the reading:
 * a count that misses items phrased differently across raw sessions, a
 * knowledge update answered with the earlier value. Deterministic gold from
 * labelled facts teaches the form of an answer, not how to find it in fifteen
 * raw sessions. So v3 distils a reader that does read: a teacher writes a
 * LongMemEval-style question over an assembled haystack of real sessions, then
 * answers it through the exact prompt the evaluation gives a reader, and the
 * student trains on that answer. The labelled facts stay available as a check.
 *
 * Contamination: the session pool is the 3,400 haystack sessions that are
 * evidence for no LongMemEval question; questions are written fresh over them.
 */
import {
  buildLongMemEvalAnswerContext,
  finalAnswerLine,
} from '../evals/longmemeval-answer.js';
import {
  THINKING_TYPES,
  contractBuilderArgs,
  contractFromEnv,
  readingFor,
  type ReaderContract,
} from '../evals/reader-contract.js';
import type { LongMemEvalInstance } from '../evals/longmemeval.js';
import type { Conversation } from './export.js';
import type { LabelledSession } from './reader-data.js';
import { createRng, type Rng } from './rng.js';

export type DistillType =
  | 'single-session-user'
  | 'single-session-assistant'
  | 'single-session-preference'
  | 'multi-session'
  | 'temporal-reasoning'
  | 'knowledge-update'
  | 'abstention';

/** Roughly LongMemEval's mix with abstention added; multi-session and temporal weighted up. */
const TYPE_WEIGHTS: Array<[DistillType, number]> = [
  ['multi-session', 24],
  ['temporal-reasoning', 20],
  ['knowledge-update', 14],
  ['single-session-user', 12],
  ['single-session-assistant', 8],
  ['single-session-preference', 8],
  ['abstention', 14],
];

/**
 * The two seeds of a distill run. `--seed` draws questions and haystacks; `--split-seed`
 * (default: `--seed`) orders the session pool that `--train-count` cuts into train and
 * held-out sessions, so a fresh distill can draw new questions over an earlier run's split.
 */
export function distillSeeds(argv: readonly string[]): {
  seed: number;
  splitSeed: number;
} {
  const read = (name: string, fallback: string): number => {
    const index = argv.indexOf(name);
    const raw =
      index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
    const value = Number(raw);
    if (!Number.isInteger(value))
      throw new Error(`${name} must be an integer, got ${raw}`);
    return value;
  };
  const seed = read('--seed', '7');
  return { seed, splitSeed: read('--split-seed', String(seed)) };
}

/**
 * Train and held-out session pools: sessions in id order, shuffled by the split seed, kept
 * sessions only, cut at `trainCount`. The order (shuffle, then filter) is the one earlier
 * distill runs used with their `--seed`, so `--split-seed 7` reproduces a `--seed 7` split.
 */
export function splitDistillPools<T extends { id: string }>(
  sessions: readonly T[],
  splitSeed: number,
  trainCount: number,
  keep: (session: T) => boolean = () => true,
): { train: T[]; held: T[] } {
  const ordered = createRng(splitSeed)
    .shuffle([...sessions].sort((a, b) => a.id.localeCompare(b.id)))
    .filter(keep);
  return {
    train: ordered.slice(0, trainCount),
    held: ordered.slice(trainCount),
  };
}

/** What a distill output directory is pinned to in `run.json`; a resume must match it. */
export interface DistillRunIdentity {
  contract: string;
  teacher: string;
  seed: number;
  splitSeed: number;
  typeWeights: string;
  trainCount: number;
  labels: string;
  /** The teacher's sampling knobs, recorded only when the run moved them off the defaults. */
  temperature?: number;
  timeoutMs?: number;
}

export function distillRunIdentity(
  identity: DistillRunIdentity,
): DistillRunIdentity {
  const {
    contract,
    teacher,
    seed,
    splitSeed,
    typeWeights,
    trainCount,
    labels,
    temperature,
    timeoutMs,
  } = identity;
  return {
    contract,
    teacher,
    seed,
    splitSeed,
    typeWeights,
    trainCount,
    labels,
    ...(temperature === undefined ? {} : { temperature }),
    ...(timeoutMs === undefined ? {} : { timeoutMs }),
  };
}

export function pickType(
  rng: Rng,
  weights: ReadonlyArray<[DistillType, number]> = TYPE_WEIGHTS,
): DistillType {
  const total = weights.reduce((a, [, w]) => a + w, 0);
  let roll = rng.next() * total;
  for (const [type, weight] of weights) {
    roll -= weight;
    if (roll <= 0) return type;
  }
  return weights[weights.length - 1][0];
}

/** "multi-session=40,temporal-reasoning=35" → weight pairs; unknown types are an error. */
export function parseTypeWeights(spec: string): Array<[DistillType, number]> {
  const known = new Set(TYPE_WEIGHTS.map(([t]) => t));
  return spec.split(',').map((part) => {
    const [type, weight] = part.split('=').map((x) => x.trim());
    if (!known.has(type as DistillType) || !Number.isFinite(Number(weight))) {
      throw new Error(`bad type weight: ${part}`);
    }
    return [type as DistillType, Number(weight)];
  });
}

/**
 * Sessions grouped by a self-fact predicate they carry, for seeding haystacks: a
 * multi-session question needs several sessions about the same kind of thing, a
 * knowledge update needs two sessions where the same predicate has different values.
 */
export function predicateGroups(
  pool: readonly LabelledSession[],
  selfAtom = 'user',
): Map<string, Array<{ session: LabelledSession; value: string }>> {
  const groups = new Map<
    string,
    Array<{ session: LabelledSession; value: string }>
  >();
  for (const session of pool) {
    const seen = new Set<string>();
    for (const fact of session.facts) {
      const match = /^([a-z][a-z0-9_]*)\(([^,()]+),\s*([^()]+)\)\.$/.exec(
        fact.trim(),
      );
      if (match === null || match[2].trim() !== selfAtom || seen.has(match[1]))
        continue;
      seen.add(match[1]);
      const list = groups.get(match[1]) ?? [];
      list.push({ session, value: match[3].trim() });
      groups.set(match[1], list);
    }
  }
  return groups;
}

export interface Haystack {
  sessions: LabelledSession[];
  /** ISO day after the latest session. */
  questionDate: string;
}

function addDaysIso(iso: string, days: number): string {
  const base = Date.UTC(
    Number(iso.slice(0, 4)),
    Number(iso.slice(5, 7)) - 1,
    Number(iso.slice(8, 10)),
  );
  return new Date(base + days * 86_400_000).toISOString().slice(0, 10);
}

export interface AssembleOptions {
  /** Seed the haystack with material for the type (multi-session or knowledge-update). */
  seed?: DistillType;
  groups?: ReturnType<typeof predicateGroups>;
}

/**
 * 5 to 15 sessions, date-ordered, question date 7 to 120 days after the latest. Seeded
 * for multi-session (three to five sessions sharing a predicate) or knowledge-update (two
 * sessions where a predicate's value changed), filled to size with random sessions.
 */
export function assembleHaystack(
  pool: readonly LabelledSession[],
  rng: Rng,
  options: AssembleOptions = {},
): Haystack {
  const size = 5 + rng.int(11);
  const seeded: LabelledSession[] = [];
  if (options.groups !== undefined && options.seed !== undefined) {
    const candidates = [...options.groups.entries()];
    if (options.seed === 'multi-session') {
      const eligible = candidates.filter(
        ([, list]) => new Set(list.map((e) => e.session.id)).size >= 3,
      );
      if (eligible.length > 0) {
        const [, list] = rng.pick(eligible);
        const distinct = [
          ...new Map(list.map((e) => [e.session.id, e.session])).values(),
        ];
        seeded.push(...rng.shuffle(distinct).slice(0, 3 + rng.int(3)));
      }
    } else if (options.seed === 'knowledge-update') {
      const eligible = candidates.filter(
        ([, list]) => new Set(list.map((e) => e.value)).size >= 2,
      );
      if (eligible.length > 0) {
        const [, list] = rng.pick(eligible);
        const first = rng.pick(list);
        const second = list.find(
          (e) => e.value !== first.value && e.session.id !== first.session.id,
        );
        if (second !== undefined) seeded.push(first.session, second.session);
      }
    }
  }
  const seededIds = new Set(seeded.map((s) => s.id));
  const filler = rng
    .shuffle(pool.filter((s) => !seededIds.has(s.id)))
    .slice(0, Math.max(0, Math.min(size, pool.length) - seeded.length));
  const sessions = [...seeded, ...filler].sort((a, b) =>
    a.date.localeCompare(b.date),
  );
  const latest = sessions[sessions.length - 1]?.date ?? '2023-01-01';
  return { sessions, questionDate: addDaysIso(latest, 7 + rng.int(114)) };
}

const TYPE_GUIDANCE: Record<DistillType, string> = {
  'single-session-user':
    'a question about a personal detail the user stated in exactly one session (a fact, a preference, a plan, a possession)',
  'single-session-assistant':
    'a question about something the assistant told the user in exactly one session (a recommendation, an explanation, a piece of advice the assistant gave), phrased as "what did you suggest/tell me about…"',
  'single-session-preference':
    'a request for a recommendation or advice where a good answer must use a preference or circumstance the user stated in one session (e.g. "recommend me a camera lens" when the user mentioned their camera body)',
  'multi-session':
    'a question that needs information from several sessions combined: a count ("how many … have I …"), a list ("which … have I mentioned"), or a comparison across sessions',
  'temporal-reasoning':
    'a question that needs time arithmetic against the session dates: "how long ago did I…", "how many weeks between … and …", "what did I do the week before …", "which came first"',
  'knowledge-update':
    'a question about something the user stated in one session and then changed or updated in a later session, asking for the current value',
  abstention:
    'a natural question about the user that the sessions do NOT answer: it must sound like the other questions and be about the kind of thing these chats discuss, but the information must be absent',
};

/** The teacher sees the sessions numbered and dated, and writes one question of the type. */
export function questionWriterPrompt(
  haystack: Haystack,
  type: DistillType,
): string {
  const rendered = haystack.sessions
    .map(
      (s, i) =>
        `### Session ${i + 1} (${s.date})\n${s.transcript.slice(0, 6_000)}`,
    )
    .join('\n\n');
  return `You are building an evaluation of an assistant's long-term memory. Below are chat sessions between a user and an assistant, numbered and dated. Today is ${haystack.questionDate}.

Write ONE question the user might ask the assistant today that tests memory of these sessions. The question must be: ${TYPE_GUIDANCE[type]}. Write it the way a real person types to their assistant (first person, casual, no session numbers, no quoting). ${
    type === 'multi-session'
      ? 'It must require at least two different sessions to answer. '
      : ''
  }${type === 'abstention' ? 'Do not make it answerable from the sessions. ' : ''}Reply with JSON only: {"question": "...", "evidence": [session numbers the answer depends on, empty for an unanswerable question]}.

${rendered}`;
}

export function parseQuestionReply(
  reply: string,
): { question: string; evidence: number[] } | undefined {
  const match = /\{[\s\S]*\}/.exec(reply);
  if (match === null) return undefined;
  try {
    const parsed = JSON.parse(match[0]) as {
      question?: unknown;
      evidence?: unknown;
    };
    if (
      typeof parsed.question !== 'string' ||
      parsed.question.trim().length < 8
    )
      return undefined;
    const evidence = Array.isArray(parsed.evidence)
      ? parsed.evidence.filter((n): n is number => Number.isInteger(n))
      : [];
    return { question: parsed.question.trim(), evidence };
  } catch {
    return undefined;
  }
}

/** A reply that abstains: says the history does not answer the question. */
export const ABSTAINS =
  /(does not|doesn't|don't|do not|no) (say|mention|know|have|contain|record|information|indicate)|not (mentioned|recorded|in (the|your) history)|i do not know|i don't know|no information/i;

/**
 * Whether the row is rendered with the notes prompt: a thinking type, or an abstention
 * question, which under thinking renders as multi-session (LongMemEval's abstention
 * questions keep their aggregation type, so the harness reads them with notes).
 */
export function thinksOn(thinking: boolean, type: string): boolean {
  return thinking && (THINKING_TYPES.has(type) || type === 'abstention');
}

/** The text the evaluation judges: the final answer line on a notes-rendered row, else the reply. */
export function completionAnswer(
  reply: string,
  thinking: boolean,
  type: string,
): string {
  return thinksOn(thinking, type) ? finalAnswerLine(reply) : reply.trim();
}

/** A last "Answer:" marker with text after it, as finalAnswerLine reads one. */
export function hasAnswerLine(reply: string): boolean {
  const index = reply.lastIndexOf('Answer:');
  return index >= 0 && reply.slice(index + 'Answer:'.length).trim() !== '';
}

/**
 * An abstention example is kept only when the reader abstained; any other type
 * is dropped when the reader abstained (the teacher could not answer its own
 * question, so the example teaches nothing) or answered nothing at all. Under the
 * thinking step a notes-rendered row (a thinking type or abstention) must end in an
 * answer line, and only that line is tested: notes may well say what the history does
 * not mention.
 */
export function acceptDistilled(
  type: DistillType,
  answer: string,
  thinking = false,
): boolean {
  if (answer.trim().length === 0) return false;
  if (thinksOn(thinking, type) && !hasAnswerLine(answer)) return false;
  const abstained = ABSTAINS.test(completionAnswer(answer, thinking, type));
  return type === 'abstention' ? abstained : !abstained;
}

/** The exact reader prompt the evaluation builds, over the haystack, for the question. */
export function readerMessages(
  haystack: Haystack,
  question: string,
  type: DistillType,
  contract: ReaderContract = contractFromEnv(),
): Conversation['messages'] {
  // distillation orders a haystack by date, evaluation takes the top lexical ranks: the full
  // tier would go to different sessions in the two, so tiers wait for rank-ordered haystacks
  if (contract.fullSessions !== null)
    throw new Error(
      'tiered contracts need rank-ordered haystacks; see plan Task 5',
    );
  // abstention renders as an aggregation question under thinking, as LongMemEval's do; a
  // direct contract keeps single-session-user so its prompts stay byte-identical
  const instanceType =
    type === 'abstention'
      ? contract.thinking
        ? 'multi-session'
        : 'single-session-user'
      : type;
  const instance = {
    question_id: `distill-${type}`,
    question_type: instanceType,
    question,
    question_date: `${haystack.questionDate.replace(/-/g, '/')} (Sat) 09:00`,
    answer: '',
    haystack_session_ids: [],
    haystack_dates: [],
    haystack_sessions: [],
    answer_session_ids: [],
  } as unknown as LongMemEvalInstance;
  const context = buildLongMemEvalAnswerContext(
    instance,
    haystack.sessions.map((s) => ({
      opId: s.id,
      ts: `${s.date}T09:00:00.000Z`,
      text: s.transcript,
      facts: s.facts,
    })),
    contract.contextBytes,
    [],
    readingFor(contract, instanceType),
    undefined,
    ...contractBuilderArgs(contract),
  );
  return context.messages as Conversation['messages'];
}

export interface DistilledExample {
  type: DistillType;
  question: string;
  questionDate: string;
  sessionIds: string[];
  evidence: number[];
  answer: string;
  messages: Conversation['messages'];
}

export function toDistilledConversation(
  example: DistilledExample,
): Conversation {
  return {
    messages: [
      ...example.messages,
      { role: 'assistant', content: example.answer },
    ],
  };
}
