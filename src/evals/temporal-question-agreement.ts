#!/usr/bin/env node
/**
 * How well does the text-only temporal-question detector agree with LongMemEval's
 * `question_type` labels? Reads only questions and labels, never answers.
 *
 *   node dist/evals/temporal-question-agreement.js [--data <path>] [--list]
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isTemporalQuestion } from '../knowledge/temporal-question.js';

export const TEMPORAL_QUESTION_TYPE = 'temporal-reasoning';

export interface TemporalQuestionAgreement {
  total: number;
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  /** Per labelled type: how many questions, and how many the detector calls temporal. */
  byType: Record<string, { total: number; temporal: number }>;
  /** Temporal-reasoning questions the detector missed (they would go to the turn unit). */
  misses: string[];
  /** Other types the detector calls temporal (they lose the turn unit). */
  falseAlarms: Array<{ questionType: string; question: string }>;
}

export function temporalQuestionAgreement(
  instances: ReadonlyArray<{ question: string; question_type: string }>,
): TemporalQuestionAgreement {
  const byType: Record<string, { total: number; temporal: number }> = {};
  const misses: string[] = [];
  const falseAlarms: Array<{ questionType: string; question: string }> = [];
  let truePositives = 0;
  for (const { question, question_type: questionType } of instances) {
    const temporal = isTemporalQuestion(question);
    const row = (byType[questionType] ??= { total: 0, temporal: 0 });
    row.total += 1;
    if (temporal) row.temporal += 1;
    const labelled = questionType === TEMPORAL_QUESTION_TYPE;
    if (temporal && labelled) truePositives += 1;
    else if (labelled) misses.push(question);
    else if (temporal) falseAlarms.push({ questionType, question });
  }
  const predicted = truePositives + falseAlarms.length;
  const actual = truePositives + misses.length;
  return {
    total: instances.length,
    truePositives,
    falsePositives: falseAlarms.length,
    falseNegatives: misses.length,
    precision: predicted === 0 ? 0 : truePositives / predicted,
    recall: actual === 0 ? 0 : truePositives / actual,
    byType: Object.fromEntries(
      Object.entries(byType).sort(([a], [b]) => a.localeCompare(b)),
    ),
    misses,
    falseAlarms,
  };
}

export function formatTemporalQuestionAgreement(
  report: TemporalQuestionAgreement,
  list: boolean,
): string {
  const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const lines = [
    `questions: ${report.total}`,
    `temporal-reasoning: precision ${percent(report.precision)} (${report.truePositives}/${report.truePositives + report.falsePositives}), recall ${percent(report.recall)} (${report.truePositives}/${report.truePositives + report.falseNegatives})`,
    '',
    '| question type | questions | detected temporal | share |',
    '|---|---:|---:|---:|',
    ...Object.entries(report.byType).map(
      ([type, row]) =>
        `| ${type} | ${row.total} | ${row.temporal} | ${percent(row.temporal / row.total)} |`,
    ),
  ];
  if (list) {
    lines.push('', 'missed temporal-reasoning:');
    lines.push(...report.misses.map((q) => `  - ${q}`));
    lines.push('', 'false alarms:');
    lines.push(
      ...report.falseAlarms.map((f) => `  - [${f.questionType}] ${f.question}`),
    );
  }
  return lines.join('\n');
}

function main(argv: string[]): void {
  let data = '.cache/longmemeval/longmemeval_s_cleaned.json';
  let list = false;
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--data') data = argv[++index] ?? data;
    else if (arg === '--list') list = true;
    else throw new Error(`unknown option: ${arg}`);
  }
  const instances = JSON.parse(readFileSync(resolve(data), 'utf8')) as Array<{
    question: string;
    question_type: string;
  }>;
  console.log(
    formatTemporalQuestionAgreement(temporalQuestionAgreement(instances), list),
  );
}

if (
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1]
) {
  main(process.argv.slice(2));
}
