/**
 * `node dist/evals/run-document-taxonomy.js --arm results/A.json --oracle results/A-oracle.json`
 *
 * Every judged-wrong answer of an arm gets a cause; every question gets a reasoning shape. The
 * shapes are cached and shared across arms (a question's shape does not depend on who answered).
 * Labeller: deepseek-chat, on the benchmark's verbatim evidence quotes.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { OpenRouterClient } from '../llm/client.js';
import {
  buildShapePrompt,
  buildTaxonomyPrompt,
  deterministicCause,
  parseShape,
  parseTaxonomyReply,
  type MissCause,
  type Shape,
} from './document-miss-taxonomy.js';
import { mapConcurrent } from './map-concurrent.js';

interface Outcome {
  id: string;
  question: string;
  gold: string | null;
  answer: string;
  judged?: boolean;
  correct: boolean;
  evidencePages: number[];
  retrieval: { evidenceHit: boolean; pageRecall: number };
}

const SHAPE_CACHE = '.cache/document-recall/shapes.json';

function outcomesOf(path: string): Map<string, Outcome & { tier: string }> {
  const run = JSON.parse(readFileSync(path, 'utf8')) as { tiers: Array<{ tier: string; outcomes: Outcome[] }> };
  return new Map(run.tiers.flatMap((tier) => tier.outcomes.map((o) => [o.id, { ...o, tier: tier.tier }] as const)));
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string) => {
    const at = argv.indexOf(name);
    return at >= 0 ? argv[at + 1] : undefined;
  };
  const armPath = flag('--arm');
  if (armPath === undefined) throw new Error('--arm results/X.json is required');
  const oraclePath = flag('--oracle');
  const output = flag('--output');

  const client = new OpenRouterClient({
    model: 'deepseek-chat',
    apiKey: process.env.DEEPSEEK_API_KEY ?? '',
    baseUrl: 'https://api.deepseek.com/v1',
  });
  const quotes = new Map<string, string[]>();
  for (const line of readFileSync('.cache/xl-docbench/qa_single_doc.jsonl', 'utf8').trim().split('\n')) {
    const row = JSON.parse(line) as { question_id: string; document: { evidence_items?: Array<{ quote?: string }> } };
    quotes.set(row.question_id, (row.document.evidence_items ?? []).map((item) => item.quote ?? '').filter(Boolean));
  }

  const arm = outcomesOf(armPath);
  const oracle = oraclePath === undefined ? undefined : outcomesOf(oraclePath);
  const shapes: Record<string, Shape> = existsSync(SHAPE_CACHE) ? JSON.parse(readFileSync(SHAPE_CACHE, 'utf8')) : {};

  const all = [...arm.values()];
  await mapConcurrent(all.filter((o) => shapes[o.id] === undefined), 8, async (o) => {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const reply = await client.completeWithUsage([{ role: 'user', content: buildShapePrompt(o.question) }]);
        shapes[o.id] = parseShape(reply.content);
        return;
      } catch {
        // a malformed label is retried; three failures leave the question unshaped
      }
    }
  });
  mkdirSync(dirname(SHAPE_CACHE), { recursive: true });
  writeFileSync(SHAPE_CACHE, JSON.stringify(shapes, null, 1));

  const judged = (o: Outcome) => o.judged ?? o.correct;
  const misses = all.filter((o) => !judged(o));
  const labelled = await mapConcurrent(misses, 8, async (o) => {
    const oracleOutcome = oracle?.get(o.id);
    const input = {
      id: o.id,
      question: o.question,
      gold: o.gold,
      answer: o.answer,
      evidencePages: o.evidencePages,
      pageRecall: o.retrieval.pageRecall,
      evidenceHit: o.retrieval.evidenceHit,
      ...(oracleOutcome === undefined ? {} : { oracleCorrect: judged(oracleOutcome) }),
    };
    const fixed = deterministicCause(input);
    if (fixed !== undefined) return { id: o.id, tier: o.tier, cause: fixed as MissCause };
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const reply = await client.completeWithUsage([
          { role: 'user', content: buildTaxonomyPrompt(input, quotes.get(o.id) ?? []) },
        ]);
        return { id: o.id, tier: o.tier, cause: parseTaxonomyReply(reply.content).cause as MissCause };
      } catch {
        // retry a malformed reply
      }
    }
    return { id: o.id, tier: o.tier, cause: 'unlabelled' as MissCause };
  });

  const count = <T extends string>(values: T[]) => {
    const counts: Record<string, number> = {};
    for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  };
  const right = all.length - misses.length;
  console.log(`${armPath}: ${all.length} questions, ${right} right (${((100 * right) / all.length).toFixed(1)}%), ${misses.length} missed`);
  console.log('\nmisses by cause:');
  for (const [cause, n] of count(labelled.map((l) => l.cause))) {
    console.log(`  ${cause.padEnd(32)} ${String(n).padStart(3)}  ${((100 * n) / misses.length).toFixed(0).padStart(3)}% of misses`);
  }
  console.log('\naccuracy by shape:');
  const byShape = new Map<string, { n: number; right: number }>();
  for (const o of all) {
    const shape = shapes[o.id] ?? 'unshaped';
    const entry = byShape.get(shape) ?? { n: 0, right: 0 };
    entry.n += 1;
    entry.right += judged(o) ? 1 : 0;
    byShape.set(shape, entry);
  }
  for (const [shape, { n, right: r }] of [...byShape.entries()].sort((a, b) => b[1].n - a[1].n)) {
    console.log(`  ${shape.padEnd(18)} ${String(r).padStart(3)}/${String(n).padEnd(3)} ${((100 * r) / n).toFixed(0).padStart(3)}%`);
  }
  if (output !== undefined) {
    mkdirSync(dirname(resolve(output)), { recursive: true });
    writeFileSync(output, `${JSON.stringify({ arm: armPath, oracle: oraclePath ?? null, misses: labelled.map((l) => ({ ...l, shape: shapes[l.id] })) }, null, 1)}\n`);
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
