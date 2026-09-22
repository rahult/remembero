#!/usr/bin/env node
/**
 * Replay routing policies offline from saved runs: no model is called.
 *
 * A cascade answers with a cheap arm first and escalates a question to a stronger arm only when
 * the cheap one declines ("I do not know" and friends — the `abstained` flag the harness saved).
 * Because every arm answered every question, the cascade's accuracy, cost and time can be read
 * straight off the two runs' per-question records.
 *
 * Usage: node benchmarks/document-recall/cascade.mjs CHEAP.json STRONG.json [label]
 * Correctness is the judge's verdict. Cost: recorded dollars, else tokens x PRICES.
 */

import { readFileSync } from 'node:fs';

const PRICES = { 'deepseek-chat': [0.15, 0.6], 'rembero-reader-v7': [0, 0] };

function load(file) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const price = PRICES[run.reader];
  const byId = new Map();
  for (const tier of run.tiers) {
    for (const o of tier.outcomes) {
      const cost =
        typeof o.readerCostUsd === 'number'
          ? o.readerCostUsd
          : price === undefined
            ? NaN
            : (o.readerTokens.prompt * price[0] + o.readerTokens.completion * price[1]) / 1e6;
      byId.set(o.id, { tier: tier.tier, right: o.judged === true, abstained: o.abstained, gold: o.gold, cost, ms: o.latencyMs });
    }
  }
  return { reader: run.reader, byId };
}

const [cheapFile, strongFile, label = 'cascade'] = process.argv.slice(2);
const cheap = load(cheapFile);
const strong = load(strongFile);

const rows = {};
for (const [id, c] of cheap.byId) {
  const s = strong.byId.get(id);
  if (s === undefined) continue;
  const row = (rows[c.tier] ??= { n: 0, cheapRight: 0, strongRight: 0, cascadeRight: 0, escalated: 0, cheapCost: 0, strongCost: 0, cascadeCost: 0, cheapMs: 0, strongMs: 0, cascadeMs: 0, falseAnswers: 0, unanswerable: 0 });
  const escalate = c.abstained;
  row.n += 1;
  row.cheapRight += c.right ? 1 : 0;
  row.strongRight += s.right ? 1 : 0;
  row.cascadeRight += (escalate ? s.right : c.right) ? 1 : 0;
  row.escalated += escalate ? 1 : 0;
  row.cheapCost += c.cost;
  row.strongCost += s.cost;
  row.cascadeCost += c.cost + (escalate ? s.cost : 0);
  row.cheapMs += c.ms;
  row.strongMs += s.ms;
  row.cascadeMs += c.ms + (escalate ? s.ms : 0);
  if (c.gold === null) {
    row.unanswerable += 1;
    // a false answer: the final responder answered a question the document cannot answer
    const final = escalate ? s : c;
    if (!final.abstained) row.falseAnswers += 1;
  }
}

const pct = (a, b) => `${((100 * a) / b).toFixed(1)}%`;
console.log(`${label}: ${cheap.reader} first, escalate to ${strong.reader} when it declines`);
console.log('tier  | Qs | cheap  | strong | cascade | escalated | $/Q cheap | $/Q strong | $/Q cascade | s/Q cascade | false ans');
for (const [tier, r] of Object.entries(rows)) {
  console.log(
    [
      tier.padEnd(5),
      String(r.n).padStart(2),
      pct(r.cheapRight, r.n).padStart(6),
      pct(r.strongRight, r.n).padStart(6),
      pct(r.cascadeRight, r.n).padStart(7),
      pct(r.escalated, r.n).padStart(9),
      `$${(r.cheapCost / r.n).toFixed(4)}`.padStart(9),
      `$${(r.strongCost / r.n).toFixed(4)}`.padStart(10),
      `$${(r.cascadeCost / r.n).toFixed(4)}`.padStart(11),
      `${(r.cascadeMs / r.n / 1000).toFixed(1)}s`.padStart(11),
      `${r.falseAnswers}/${r.unanswerable}`.padStart(9),
    ].join(' | '),
  );
}
