#!/usr/bin/env node
/**
 * Print the comparison table across arms, straight from the result files.
 *
 * Usage: node benchmarks/document-recall/summarise.mjs results/docrecall-*.json
 *
 * Each row is one tier of one arm. Accuracy is the benchmark's own rule; "judge" is the optional
 * second opinion; "hit" and "recall" are the page-level retrieval numbers the benchmark itself
 * does not measure. Arms are printed in the order given, so pass the oracle run last.
 */

import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('pass one or more results/docrecall-*.json files');
  process.exit(1);
}

const percent = (value) => (value === undefined || Number.isNaN(value) ? '-' : `${(value * 100).toFixed(1)}%`);

console.log(
  'arm                        | tier  | depth | Qs | rule   | judge  | page hit | page recall | win prec | err',
);
console.log(
  '-------------------------- | ----- | ----- | -- | ------ | ------ | -------- | ----------- | -------- | ---',
);
for (const file of files) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const arm = basename(file, '.json').replace(/^docrecall-/, '');
  const depth = run.settings?.oraclePages === true ? 'oracle' : String(run.settings?.topK ?? '?');
  for (const tier of run.tiers ?? []) {
    const summary = tier.summary;
    console.log(
      [
        arm.padEnd(26).slice(0, 26),
        tier.tier.padEnd(5),
        depth.padStart(5),
        String(summary.questions).padStart(2),
        percent(summary.accuracy).padStart(6),
        percent(summary.judgedAccuracy).padStart(6),
        percent(summary.evidenceHitRate).padStart(8),
        percent(summary.meanPageRecall).padStart(11),
        percent(summary.meanWindowPrecision).padStart(8),
        String((tier.errors ?? []).length).padStart(3),
      ].join(' | '),
    );
  }
}
