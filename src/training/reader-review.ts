/**
 * Data review: what a trained reader got wrong, turned into the next reader's data recipe.
 *
 * Two views of the same reader. A LongMemEval result gives per-type accuracy (and, against
 * a baseline run, the deltas) and its misses; its observations store the judged text but
 * neither the question nor the gold answer, so only the classes that need neither are
 * counted there (format from the reading and token cap, the two abstention classes from
 * the question's abstention flag) and every other miss is `unclassified`. A mine directory
 * (reader-mine.ts) stores the question, the teacher's judged text and the student's whole
 * reply for every row, so its misses get every class; its per-type miss counts set the next
 * distillation's type weights and its miss rate the share of misses composed into training.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { finalAnswerLine } from '../evals/longmemeval-answer.js';
import {
  DEFAULT_ABSTRACT_BYTES,
  THINKING_TYPES,
  contractId,
} from '../evals/reader-contract.js';
import { ABSTAINS, hasAnswerLine, thinksOn } from './reader-distill.js';
import type { MineResult } from './reader-mine.js';

export const MISS_CLASSES = [
  'format',
  'false-abstain',
  'missed-abstain',
  'count',
  'date-arithmetic',
  'other',
] as const;
export type MissClass = (typeof MISS_CLASSES)[number];

export interface MissRow {
  type: string;
  question: string;
  /** The teacher's (or gold) judged text. */
  expected: string;
  /** The reader's judged text. */
  hypothesis: string;
  /** The reader's whole reply. */
  reply: string;
  /** Whether the reader thinks on notes-rendered types (its contract's thinking flag). */
  thinking?: boolean;
  /** The reader's completion reached its token cap. */
  cutAtLimit?: boolean;
}

const NUMBER =
  /\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b/i;
const DATE =
  /\d|\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i;

/** A thinking-type reply that is malformed as an answer: no Answer line, a bold marker, or cut. */
function formatMiss(
  thinking: boolean,
  type: string,
  reply: { hasAnswerLine: boolean; answer: string; cutAtLimit: boolean },
): boolean {
  return (
    thinksOn(thinking, type) &&
    (!reply.hasAnswerLine || reply.answer.startsWith('*') || reply.cutAtLimit)
  );
}

/** Checked in order: format, false-abstain, missed-abstain, count, date-arithmetic, other. */
export function classifyMiss(row: MissRow): MissClass {
  if (
    formatMiss(row.thinking === true, row.type, {
      hasAnswerLine: hasAnswerLine(row.reply),
      answer: finalAnswerLine(row.reply),
      cutAtLimit: row.cutAtLimit === true,
    })
  )
    return 'format';
  const hypothesisAbstains = ABSTAINS.test(row.hypothesis);
  const expectedAbstains = ABSTAINS.test(row.expected);
  if (hypothesisAbstains && !expectedAbstains) return 'false-abstain';
  if (expectedAbstains && !hypothesisAbstains) return 'missed-abstain';
  if (
    /\bhow many\b/i.test(row.question) &&
    NUMBER.test(row.expected) &&
    NUMBER.test(row.hypothesis)
  )
    return 'count';
  if (
    row.type === 'temporal-reasoning' &&
    DATE.test(row.expected) &&
    DATE.test(row.hypothesis)
  )
    return 'date-arithmetic';
  return 'other';
}

/** The fields of a LongMemEval observation the review reads. */
export interface ReviewObservation {
  questionId?: string;
  questionType: string;
  abstention?: boolean;
  status?: string;
  correct?: boolean;
  hypothesis: string | null;
  readerUsage?: { completionTokens?: number | null } | null;
}

export interface ReviewSettings {
  readingStrategy?: string;
  readerMaxTokens?: number | null;
}

export type ObservationClass =
  'format' | 'false-abstain' | 'missed-abstain' | 'unclassified';

/** The classes a LongMemEval observation cannot be given: they need the question or gold answer. */
export const OBSERVATION_UNAVAILABLE: MissClass[] = [
  'count',
  'date-arithmetic',
  'other',
];

/**
 * A LongMemEval miss by what its observation stores. Under notes the stored hypothesis is
 * already the final answer line, so a missing Answer line cannot be seen; the token cap
 * and a bold marker can. The abstention flag says whether the gold answer abstains.
 */
export function classifyObservation(
  observation: ReviewObservation,
  settings: ReviewSettings,
): ObservationClass {
  const hypothesis = observation.hypothesis ?? '';
  const tokens = observation.readerUsage?.completionTokens;
  const cap = settings.readerMaxTokens;
  const thinking =
    settings.readingStrategy === 'notes' &&
    THINKING_TYPES.has(observation.questionType);
  if (
    formatMiss(thinking, observation.questionType, {
      hasAnswerLine: true,
      answer: hypothesis,
      cutAtLimit:
        typeof tokens === 'number' && typeof cap === 'number' && tokens >= cap,
    })
  )
    return 'format';
  const abstains = ABSTAINS.test(hypothesis);
  if (observation.abstention === false && abstains) return 'false-abstain';
  if (observation.abstention === true && !abstains) return 'missed-abstain';
  return 'unclassified';
}

export interface TypeMissCounts {
  asked: number;
  misses: number;
}

export const WEIGHT_FLOOR = 0.05;

/**
 * Each asked type's share of the misses, floored at 5% so no type drops out of the mix,
 * renormalised to sum 1; `spec` is the distiller's `--type-weights` in whole percentages,
 * heaviest first.
 */
export function recommendTypeWeights(byType: Record<string, TypeMissCounts>): {
  spec: string;
  weights: Record<string, number>;
} {
  const asked = Object.entries(byType).filter(([, counts]) => counts.asked > 0);
  const total = asked.reduce((sum, [, counts]) => sum + counts.misses, 0);
  const floored = asked.map(
    ([type, counts]) =>
      [
        type,
        Math.max(total === 0 ? 0 : counts.misses / total, WEIGHT_FLOOR),
      ] as const,
  );
  const sum = floored.reduce((acc, [, weight]) => acc + weight, 0);
  const weights = Object.fromEntries(
    floored.map(([type, weight]) => [type, weight / sum]),
  );
  const spec = Object.entries(weights)
    .map(([type, weight]) => [type, Math.round(weight * 100)] as const)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([type, percent]) => `${type}=${percent}`)
    .join(',');
  return { spec, weights };
}

export const MISS_SHARE = 0.25;
export const RAISED_MISS_SHARE = 0.35;
export const MISS_RATE_RAISE = 0.4;

/** The share of composed rows taken from misses: raised when the student misses often. */
export function recommendMissShare(asked: number, misses: number): number {
  return asked > 0 && misses / asked > MISS_RATE_RAISE
    ? RAISED_MISS_SHARE
    : MISS_SHARE;
}

export interface TypeAccuracy {
  questions: number;
  correct: number;
  accuracy?: number;
}

export const DROP_MARGIN = 5;

/**
 * Types whose thinking run answered more than five fewer questions than its direct
 * baseline. The caller pairs the runs; this only compares correct counts.
 */
export function dropTypesFromThinking(
  run: Record<string, TypeAccuracy>,
  baseline: Record<string, TypeAccuracy>,
): string[] {
  return Object.keys(run)
    .filter(
      (type) =>
        baseline[type] !== undefined &&
        baseline[type]!.correct - run[type]!.correct > DROP_MARGIN,
    )
    .sort();
}

interface LongMemEvalResult {
  settings?: ReviewSettings & Record<string, unknown>;
  contextBytes?: number;
  byQuestionType?: Record<string, TypeAccuracy>;
  observations?: ReviewObservation[];
}

function readJson<T>(path: string): T {
  if (!existsSync(path)) throw new Error(`${path} does not exist`);
  return JSON.parse(readFileSync(path, 'utf8')) as T;
}

/** The contract id a LongMemEval run read under, from its recorded settings; null when unrecorded. */
export function runContractId(result: LongMemEvalResult): string | null {
  const s = result.settings;
  if (s === undefined || typeof result.contextBytes !== 'number') return null;
  const fullSessions =
    typeof s.fullSessions === 'number' ? s.fullSessions : null;
  return contractId({
    dateDistances: s.dateDistances === true,
    computedNotes: s.computedNotes === true,
    focusedBudget: s.focusedBudget === true,
    structuredEvidence: s.structuredEvidence === true,
    contextBytes: result.contextBytes,
    fullSessions,
    abstractBytes:
      typeof s.abstractBytes === 'number'
        ? s.abstractBytes
        : DEFAULT_ABSTRACT_BYTES,
    thinking: s.readingStrategy === 'notes',
  });
}

function accuracyOf(entry: TypeAccuracy): Required<TypeAccuracy> {
  return {
    questions: entry.questions,
    correct: entry.correct,
    accuracy:
      typeof entry.accuracy === 'number'
        ? entry.accuracy
        : entry.questions === 0
          ? 0
          : entry.correct / entry.questions,
  };
}

export interface ObservationClasses {
  misses: number;
  classes: Record<ObservationClass, number>;
  byType: Record<string, Partial<Record<ObservationClass, number>>>;
  unavailable: MissClass[];
}

/** Class counts over the judged misses of a LongMemEval run. */
export function observationClasses(
  result: LongMemEvalResult,
): ObservationClasses {
  const summary: ObservationClasses = {
    misses: 0,
    classes: {
      format: 0,
      'false-abstain': 0,
      'missed-abstain': 0,
      unclassified: 0,
    },
    byType: {},
    unavailable: OBSERVATION_UNAVAILABLE,
  };
  for (const observation of result.observations ?? []) {
    if (observation.status !== 'judged' || observation.correct !== false)
      continue;
    const kind = classifyObservation(observation, result.settings ?? {});
    summary.misses += 1;
    summary.classes[kind] += 1;
    const type = (summary.byType[observation.questionType] ??= {});
    type[kind] = (type[kind] ?? 0) + 1;
  }
  return summary;
}

/** results.jsonl of a mine directory, one row per (file, index), the last written kept. */
export function readMineResults(minedDir: string): MineResult[] {
  const path = join(minedDir, 'results.jsonl');
  if (!existsSync(path))
    throw new Error(`${path} does not exist; review reads a mine directory`);
  const rows = new Map<string, MineResult>();
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as MineResult;
    rows.set(`${row.file}#${row.index}`, row);
  }
  return [...rows.values()];
}

function contractIdOf(value: unknown): string | null {
  return value !== null &&
    typeof value === 'object' &&
    typeof (value as { id?: unknown }).id === 'string'
    ? (value as { id: string }).id
    : null;
}

export interface ReviewOptions {
  run: string;
  baseline?: string;
  mined: string;
  now?: Date;
}

export function reviewReader(options: ReviewOptions) {
  const run = readJson<LongMemEvalResult>(options.run);
  const baseline =
    options.baseline === undefined
      ? undefined
      : readJson<LongMemEvalResult>(options.baseline);
  const minedManifest = readJson<{
    contract?: unknown;
    studentContract?: { id?: unknown; thinking?: unknown };
  }>(join(options.mined, 'manifest.json'));
  const studentThinking = minedManifest.studentContract?.thinking === true;

  const accuracy: Record<
    string,
    Required<TypeAccuracy> & {
      baseline?: Required<TypeAccuracy>;
      delta?: { correct: number; accuracy: number };
    }
  > = {};
  for (const [type, entry] of Object.entries(run.byQuestionType ?? {})) {
    const own = accuracyOf(entry);
    const paired = baseline?.byQuestionType?.[type];
    accuracy[type] =
      paired === undefined
        ? own
        : {
            ...own,
            baseline: accuracyOf(paired),
            delta: {
              correct: own.correct - paired.correct,
              accuracy: own.accuracy - accuracyOf(paired).accuracy,
            },
          };
  }

  const results = readMineResults(options.mined);
  const classes = Object.fromEntries(
    MISS_CLASSES.map((kind) => [kind, 0]),
  ) as Record<MissClass, number>;
  const classesByType: Record<string, Partial<Record<MissClass, number>>> = {};
  const byType: Record<
    string,
    { asked: number; correct: number; misses: number }
  > = {};
  let conversationsAsked = 0;
  let conversationsMisses = 0;
  for (const row of results) {
    const counts = (byType[row.type] ??= { asked: 0, correct: 0, misses: 0 });
    counts.asked += 1;
    if (row.file === 'conversations.jsonl') conversationsAsked += 1;
    if (row.correct) {
      counts.correct += 1;
      continue;
    }
    counts.misses += 1;
    if (row.file === 'conversations.jsonl') conversationsMisses += 1;
    const kind = classifyMiss({ ...row, thinking: studentThinking });
    classes[kind] += 1;
    const typeClasses = (classesByType[row.type] ??= {});
    typeClasses[kind] = (typeClasses[kind] ?? 0) + 1;
  }
  const weights = recommendTypeWeights(byType);

  return {
    generatedAt: (options.now ?? new Date()).toISOString(),
    inputs: {
      run: { path: options.run, contract: runContractId(run) },
      baseline:
        options.baseline === undefined || baseline === undefined
          ? null
          : { path: options.baseline, contract: runContractId(baseline) },
      mined: {
        path: options.mined,
        contract: contractIdOf(minedManifest.contract),
        studentContract: contractIdOf(minedManifest.studentContract),
      },
    },
    accuracy,
    longMemEval: {
      run: observationClasses(run),
      baseline: baseline === undefined ? null : observationClasses(baseline),
    },
    mined: {
      byType,
      classes,
      classesByType,
      conversations: { asked: conversationsAsked, misses: conversationsMisses },
      missRate:
        conversationsAsked === 0
          ? null
          : conversationsMisses / conversationsAsked,
    },
    recommend: {
      typeWeights: weights.spec,
      typeWeightsRaw: weights.weights,
      missShare: recommendMissShare(conversationsAsked, conversationsMisses),
      dropTypesFromThinking:
        baseline === undefined
          ? []
          : dropTypesFromThinking(
              run.byQuestionType ?? {},
              baseline.byQuestionType ?? {},
            ),
    },
  };
}

export type Review = ReturnType<typeof reviewReader>;
