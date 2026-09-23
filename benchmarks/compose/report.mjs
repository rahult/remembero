#!/usr/bin/env node
/**
 * Compose results by tier and family, rescored from the saved answers against the spec's gold —
 * so a scorer change never needs a paid reader run again.
 *
 * Usage: node benchmarks/compose/report.mjs [--spec benchmarks/compose/test/spec.json] results/compose-*.json
 * Outcomes: right, WRONG (confidently wrong), partial, declined.
 */
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import { scoreCompose } from '../../dist/compose/score.js';

const argv = process.argv.slice(2);
const specAt = argv.indexOf('--spec');
const specPath = specAt >= 0 ? argv.splice(specAt, 2)[1] : 'benchmarks/compose/test/spec.json';
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
const goldOf = new Map(spec.sources.flatMap((s) => s.questions.map((q) => [q.id, q])));

const pct = (a, n) => (n === 0 ? '   -' : `${((100 * a) / n).toFixed(0).padStart(3)}%`);
for (const file of argv) {
  const run = JSON.parse(readFileSync(file, 'utf8'));
  const depth = run.settings?.oraclePages ? 'oracle' : run.settings?.topK;
  console.log(`\n${basename(file, '.json')}  (reader ${run.reader}, ranker ${run.settings?.ranker ?? 'product'}, depth ${depth})`);
  const families = spec.sources[0].questions.map((q) => q.datasetKind).filter((f, i, all) => all.indexOf(f) === i);
  console.log(`${'tier'.padEnd(6)} ${'n'.padStart(3)}  right  WRONG  part  decl  hit  | ${families.map((f) => f.slice(0, 7).padStart(7)).join(' ')}   (% right)`);
  for (const tier of run.tiers) {
    const scored = tier.outcomes.map((o) => {
      const q = goldOf.get(o.id);
      return { ...o, family: q?.datasetKind ?? o.family, outcome: q?.compose ? scoreCompose(o.answer, q.compose) : o.correct ? 'correct' : 'wrong' };
    });
    const n = scored.length;
    const c = (k) => scored.filter((o) => o.outcome === k).length;
    const answerable = scored.filter((o) => o.gold !== null);
    const perFamily = families.map((f) => {
      const fo = scored.filter((o) => o.family === f);
      return pct(fo.filter((o) => o.outcome === 'correct').length, fo.length).padStart(7);
    });
    console.log(
      `${tier.tier.padEnd(6)} ${String(n).padStart(3)}  ${pct(c('correct'), n)}  ${pct(c('wrong'), n)}  ${pct(c('partial'), n)}  ${pct(c('declined'), n)}  ${pct(answerable.filter((o) => o.retrieval.evidenceHit).length, answerable.length)} | ${perFamily.join(' ')}`,
    );
  }
}
