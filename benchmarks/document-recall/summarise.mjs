#!/usr/bin/env node
/**
 * The comparison table across arms, read straight from the result files, on three lenses:
 * accuracy, cost and speed.
 *
 * Usage: node benchmarks/document-recall/summarise.mjs results/docrecall-*.json
 *
 * Cost is what the endpoint reported (OpenRouter does) or, for a run that recorded only tokens,
 * the tokens priced from PRICES below. A local reader is priced at zero marginal cost: the Mac is
 * already paid for, and its electricity for one ~20 s answer is a few hundredths of a cent.
 * "$/right" divides by judge-correct answers (the rule is too strict to price usefulness with).
 */

import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

/** Dollars per million tokens [input, output], used only when a run did not record its cost. */
const PRICES = {
  // DeepSeek API, deepseek-chat served by V4.1 Flash, cache-miss off-peak rate (the runs were
  // 13:40-14:30 UTC, off-peak); api-docs.deepseek.com/quick_start/pricing, read 2026-09-23
  'deepseek-chat': [0.15, 0.6],
  // local llama-server on the Mac: zero marginal cost
  'rembero-reader-v7': [0, 0],
};

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('pass one or more results/docrecall-*.json files');
  process.exit(1);
}

const pct = (value) => (value === undefined || Number.isNaN(value) ? '-' : `${(value * 100).toFixed(1)}%`);
const usd = (value) =>
  value === undefined ? '-' : value === 0 ? '$0' : value < 0.001 ? `$${value.toFixed(5)}` : `$${value.toFixed(4)}`;
const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;

console.log(
  'arm                  | tier  | depth  | Qs | rule   | judge  | page hit | $/answer | $/right  | p50    | p95    | rank   | read',
);
console.log(
  '-------------------- | ----- | ------ | -- | ------ | ------ | -------- | -------- | -------- | ------ | ------ | ------ | ------',
);
for (const file of files) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const arm = basename(file, '.json').replace(/^docrecall-/, '');
  const depth = run.settings?.oraclePages === true ? 'oracle' : String(run.settings?.topK ?? '?');
  const price = PRICES[run.reader];
  for (const tier of run.tiers ?? []) {
    const outcomes = tier.outcomes ?? [];
    const cost = outcomes.map((o) => {
      if (typeof o.readerCostUsd === 'number') return o.readerCostUsd;
      if (price === undefined || o.readerTokens === undefined) return undefined;
      return (o.readerTokens.prompt * price[0] + o.readerTokens.completion * price[1]) / 1e6;
    });
    const known = cost.every((c) => c !== undefined) && cost.length > 0;
    const total = known ? cost.reduce((a, b) => a + b, 0) : undefined;
    const right = outcomes.filter((o) => (o.judged ?? o.correct) === true).length;
    const latency = outcomes.map((o) => o.latencyMs).sort((a, b) => a - b);
    const q = (p) => (latency.length === 0 ? 0 : latency[Math.min(latency.length - 1, Math.floor(p * latency.length))]);
    const mean = (xs) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
    const hasSplit = outcomes.some((o) => typeof o.retrievalMs === 'number');
    console.log(
      [
        arm.padEnd(20).slice(0, 20),
        tier.tier.padEnd(5),
        depth.padStart(6),
        String(outcomes.length).padStart(2),
        pct(tier.summary.accuracy).padStart(6),
        pct(tier.summary.judgedAccuracy).padStart(6),
        pct(tier.summary.evidenceHitRate).padStart(8),
        usd(total === undefined ? undefined : total / outcomes.length).padStart(8),
        usd(total === undefined || right === 0 ? undefined : total / right).padStart(8),
        secs(q(0.5)).padStart(6),
        secs(q(0.95)).padStart(6),
        (hasSplit ? secs(mean(outcomes.map((o) => o.retrievalMs ?? 0))) : '-').padStart(6),
        (hasSplit ? secs(mean(outcomes.map((o) => o.readerMs ?? 0))) : '-').padStart(6),
      ].join(' | '),
    );
  }
}
