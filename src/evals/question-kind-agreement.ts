#!/usr/bin/env node
/**
 * How well does a label-free classification of a question agree with LongMemEval's
 * `question_type` labels? One precision/recall pair per flag of `knowledge/question-kind.ts`,
 * with the confusion counts, plus the union that drives the Notes-then-Answer reading
 * (aggregation, temporal or update — the labels the harness reads for it today).
 *
 * Reads only questions and labels, never answers or haystacks.
 *
 *   node dist/evals/question-kind-agreement.js [--data <path>] [--source text|typesafe] [--list]
 *
 * `--source typesafe` makes one live Jev request per question (cached under .cache/typesafe/)
 * and needs a key in TYPESAFE_AI_API_KEY; `text` is deterministic and offline.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  QUESTION_KIND_FLAGS,
  QUESTION_KIND_LABELS,
  questionKindFromLabel,
  questionKindFromText,
  type QuestionKind,
} from '../knowledge/question-kind.js';
import {
  DEFAULT_RERANK_CONCURRENCY,
  DEFAULT_RERANK_KEY_ENV,
  DEFAULT_TYPESAFE_MODEL,
  createLimiter,
  typesafeNouls,
} from './typesafe-rerank.js';
import { typesafeQuestionKind } from './typesafe-question-kind.js';

/** The confusion of one flag against the label it stands for. */
export interface FlagAgreement {
  /** The LongMemEval question type this flag stands for. */
  label: string;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  /** Labelled questions the classification missed. */
  misses: string[];
  /** Questions of another type the classification claims. */
  falseAlarms: Array<{ questionType: string; question: string }>;
}

export interface QuestionKindAgreement {
  total: number;
  flags: Record<string, FlagAgreement>;
  /** aggregation OR temporal OR update against the three types read for the notes reading. */
  notesReading: FlagAgreement;
}

function emptyAgreement(label: string): FlagAgreement {
  return {
    label,
    truePositives: 0,
    falsePositives: 0,
    trueNegatives: 0,
    falseNegatives: 0,
    precision: 0,
    recall: 0,
    misses: [],
    falseAlarms: [],
  };
}

function record(
  agreement: FlagAgreement,
  predicted: boolean,
  actual: boolean,
  instance: { question: string; question_type: string },
): void {
  if (predicted && actual) agreement.truePositives += 1;
  else if (predicted) {
    agreement.falsePositives += 1;
    agreement.falseAlarms.push({
      questionType: instance.question_type,
      question: instance.question,
    });
  } else if (actual) {
    agreement.falseNegatives += 1;
    agreement.misses.push(instance.question);
  } else agreement.trueNegatives += 1;
}

function close(agreement: FlagAgreement): void {
  const predicted = agreement.truePositives + agreement.falsePositives;
  const actual = agreement.truePositives + agreement.falseNegatives;
  agreement.precision = predicted === 0 ? 0 : agreement.truePositives / predicted;
  agreement.recall = actual === 0 ? 0 : agreement.truePositives / actual;
}

/** Every flag's agreement with the labels, for whatever classification `kindFor` applies. */
export function questionKindAgreement(
  instances: ReadonlyArray<{ question: string; question_type: string }>,
  kinds: ReadonlyArray<QuestionKind>,
): QuestionKindAgreement {
  if (kinds.length !== instances.length) {
    throw new Error('one classified kind per instance is required');
  }
  const flags = Object.fromEntries(
    QUESTION_KIND_FLAGS.map((flag) => [
      flag,
      emptyAgreement(QUESTION_KIND_LABELS[flag]),
    ]),
  );
  const notesReading = emptyAgreement(
    `${QUESTION_KIND_LABELS.aggregation}|${QUESTION_KIND_LABELS.temporal}|${QUESTION_KIND_LABELS.update}`,
  );
  for (const [index, instance] of instances.entries()) {
    const predicted = kinds[index]!;
    const labelled = questionKindFromLabel(instance.question_type);
    for (const flag of QUESTION_KIND_FLAGS) {
      record(flags[flag]!, predicted[flag], labelled[flag], instance);
    }
    record(
      notesReading,
      predicted.aggregation || predicted.temporal || predicted.update,
      labelled.aggregation || labelled.temporal || labelled.update,
      instance,
    );
  }
  for (const agreement of [...Object.values(flags), notesReading]) {
    close(agreement);
  }
  return { total: instances.length, flags, notesReading };
}

export function formatQuestionKindAgreement(
  report: QuestionKindAgreement,
  source: string,
  list: boolean,
): string {
  const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const row = (name: string, a: FlagAgreement) =>
    `| ${name} | ${a.label} | ${percent(a.precision)} | ${percent(a.recall)} | ${a.truePositives} | ${a.falsePositives} | ${a.falseNegatives} | ${a.trueNegatives} |`;
  const lines = [
    `questions: ${report.total}, classification: ${source}`,
    '',
    '| flag | label it stands for | precision | recall | tp | fp | fn | tn |',
    '|---|---|---:|---:|---:|---:|---:|---:|',
    ...QUESTION_KIND_FLAGS.map((flag) => row(flag, report.flags[flag]!)),
    row('notes reading', report.notesReading),
  ];
  if (list) {
    for (const flag of QUESTION_KIND_FLAGS) {
      const agreement = report.flags[flag]!;
      lines.push('', `${flag} misses:`);
      lines.push(...agreement.misses.map((question) => `  - ${question}`));
      lines.push(`${flag} false alarms:`);
      lines.push(
        ...agreement.falseAlarms.map(
          ({ questionType, question }) => `  - [${questionType}] ${question}`,
        ),
      );
    }
  }
  return lines.join('\n');
}

async function main(argv: string[]): Promise<void> {
  let data = '.cache/longmemeval/longmemeval_s_cleaned.json';
  let source: 'text' | 'typesafe' = 'text';
  let list = false;
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--data') data = argv[++index] ?? data;
    else if (arg === '--source') {
      const value = argv[++index];
      if (value !== 'text' && value !== 'typesafe') {
        throw new Error('--source must be text or typesafe');
      }
      source = value;
    } else if (arg === '--list') list = true;
    else throw new Error(`unknown option: ${arg}`);
  }
  const instances = JSON.parse(readFileSync(resolve(data), 'utf8')) as Array<{
    question: string;
    question_type: string;
  }>;
  let kinds: QuestionKind[];
  if (source === 'text') {
    kinds = instances.map(({ question }) => questionKindFromText(question));
  } else {
    const apiKey = process.env[DEFAULT_RERANK_KEY_ENV];
    if (apiKey === undefined || apiKey.trim() === '') {
      throw new Error(
        `--source typesafe needs a key in the environment variable ${DEFAULT_RERANK_KEY_ENV}`,
      );
    }
    const limiter = createLimiter(DEFAULT_RERANK_CONCURRENCY);
    const nouls = (state: unknown, questions: Parameters<typeof typesafeNouls>[1]) =>
      typesafeNouls(state, questions, {
        apiKey,
        model: DEFAULT_TYPESAFE_MODEL,
        limiter,
      });
    let failures = 0;
    kinds = await Promise.all(
      instances.map(async ({ question }) => {
        try {
          return (await typesafeQuestionKind(question, nouls)).kind;
        } catch {
          // the harness falls back to the text rules, so the report measures what it would use
          failures += 1;
          return questionKindFromText(question);
        }
      }),
    );
    if (failures > 0) {
      process.stderr.write(
        `${failures} of ${instances.length} classifications failed and fell back to the text rules\n`,
      );
    }
  }
  console.log(
    formatQuestionKindAgreement(
      questionKindAgreement(instances, kinds),
      source,
      list,
    ),
  );
}

if (
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1]
) {
  await main(process.argv.slice(2));
}
