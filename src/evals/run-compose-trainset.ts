/**
 * `node dist/evals/run-compose-trainset.js [--worlds 250] [--paraphrase 3000]` — training data for
 * a local engine model: one small model that both extracts a page into the schema and turns a
 * question into a plan, with exactly the prompts the evaluation uses.
 *
 * Every target is exact by construction: extraction targets come from the renderer (which knows
 * each line's facts and how it wrote them), plan targets from the question generator. Train rows
 * use train-split worlds (disjoint name pool); held-out rows use dev-split worlds; no test world,
 * and no test filler document, is touched.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { OpenRouterClient } from '../llm/client.js';
import { EXTRACTION_PROMPT } from '../compose/extract.js';
import { PARAPHRASE_PROMPT, paraphraseProblem } from '../compose/paraphrase.js';
import { PLANNER_PROMPT } from '../compose/planner.js';
import { generateQuestions, type ComposeQuestion } from '../compose/questions.js';
import { paginate, renderWorld, type SchemaFact } from '../compose/render.js';
import { Rng } from '../compose/rng.js';
import { generateWorld, type Split } from '../compose/world.js';
import { pagesFromTextFile } from './document-corpus.js';
import { mapConcurrent } from './map-concurrent.js';

type Message = { role: 'system' | 'user' | 'assistant'; content: string };
type Row = { messages: Message[] };

const SECOND_PASS = 'Read every line of this page, including tables and continued registers, and extract every record it states.\n\n';

function quote(text: string): string {
  return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

export function factLine(fact: SchemaFact): string {
  return `${fact.predicate}(${fact.args.map((a) => (typeof a === 'number' ? String(a) : quote(a))).join(', ')}).`;
}

/** The plan the planner should produce for a generated question. */
export function goldPlan(q: ComposeQuestion): { shape: string; params: Record<string, string | number> } {
  const p = q.params;
  switch (q.family) {
    case 'needle': return { shape: 'value_at_signing', params: { contract: p.contract! } };
    case 'validity': return { shape: 'role_holder_on_date', params: { role: p.role!, date: p.date!, organisation: p.organisation! } };
    case 'identity': return { shape: 'approver_of_contract', params: { contract: p.contract! } };
    case 'supersession': return { shape: 'value_on_date', params: { contract: p.contract!, date: p.date! } };
    case 'comparison': return { shape: 'higher_value_on_date', params: { contractA: p.contractA!, contractB: p.contractB!, date: p.date! } };
    case 'aggregate':
      return p.standard !== undefined
        ? { shape: 'count_incidents_without_certificate', params: { standard: p.standard, organisation: p.organisation! } }
        : { shape: 'count_incidents_for_supplier', params: { supplier: p.supplier! } };
    case 'absence': return { shape: 'suppliers_without_certificate_on_date', params: { standard: p.standard!, date: p.date!, organisation: p.organisation! } };
    case 'authority': return { shape: 'approval_within_authority', params: { contract: p.contract! } };
    case 'unanswerable':
      return p.role !== undefined
        ? { shape: 'role_holder_on_date', params: { role: p.role, date: p.date!, organisation: p.organisation! } }
        : { shape: 'value_at_signing', params: { contract: p.contract! } };
  }
}

function extractionRows(split: Split, seeds: number[], fillerDocs: string[], rng: Rng): Row[] {
  const rows: Row[] = [];
  for (const seed of seeds) {
    for (const page of paginate(renderWorld(generateWorld(seed, split)))) {
      const target = page.facts.length === 0 ? '% nothing' : page.facts.map(factLine).join('\n');
      // most rows use the first-pass prompt; some the second-pass nudge, as the evaluation does
      const prefix = rng.chance(0.25) ? SECOND_PASS : '';
      rows.push({
        messages: [
          { role: 'system', content: EXTRACTION_PROMPT },
          { role: 'user', content: prefix + page.text.slice(0, 12_000) },
          { role: 'assistant', content: target },
        ],
      });
    }
  }
  // filler: real document pages that state no schema facts; about one for every three world pages
  const filler = fillerDocs.flatMap((doc) => pagesFromTextFile(`.cache/document-recall/${doc}.pages.txt`).map((p) => p.text));
  const usable = filler.filter((t) => t.trim().length > 300);
  const want = Math.round(rows.length / 3);
  for (const text of rng.sample(usable, Math.min(want, usable.length))) {
    rows.push({
      messages: [
        { role: 'system', content: EXTRACTION_PROMPT },
        { role: 'user', content: (rng.chance(0.25) ? SECOND_PASS : '') + text.slice(0, 6_000) },
        { role: 'assistant', content: '% nothing' },
      ],
    });
  }
  return rows;
}

async function plannerRows(split: Split, seeds: number[], paraphraseBudget: number, rng: Rng, client: OpenRouterClient): Promise<Row[]> {
  const questions = seeds.flatMap((seed) => generateQuestions(generateWorld(seed, split)));
  const cachePath = `.cache/compose/paraphrase.trainset.json`;
  const cache: Record<string, string> = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, 'utf8')) : {};
  const chosen = rng.sample(questions, Math.min(paraphraseBudget, questions.length));
  await mapConcurrent(chosen.filter((q) => cache[q.question] === undefined), 16, async (q) => {
    try {
      const reply = await client.completeWithUsage([
        { role: 'system', content: PARAPHRASE_PROMPT },
        { role: 'user', content: q.question },
      ]);
      cache[q.question] = reply.content.trim();
    } catch {
      // a failed paraphrase leaves the templated question in the set
    }
  });
  mkdirSync('.cache/compose', { recursive: true });
  writeFileSync(cachePath, JSON.stringify(cache, null, 1));
  const paraphrased = new Set(chosen.map((q) => q.question));
  return questions.map((q) => {
    const candidate = paraphrased.has(q.question) ? cache[q.question] : undefined;
    const text = candidate !== undefined && paraphraseProblem(q, candidate) === undefined ? candidate : q.question;
    return {
      messages: [
        { role: 'system', content: PLANNER_PROMPT },
        { role: 'user', content: text },
        { role: 'assistant', content: JSON.stringify(goldPlan(q)) },
      ],
    };
  });
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string, fallback: string) => {
    const at = argv.indexOf(name);
    return at >= 0 && argv[at + 1] !== undefined ? argv[at + 1]! : fallback;
  };
  const worlds = Number(flag('--worlds', '250'));
  const paraphraseBudget = Number(flag('--paraphrase', '3000'));
  const out = flag('--out', 'data/compose-engine');
  const rng = new Rng(20260924);
  const client = new OpenRouterClient({ model: 'deepseek-chat', apiKey: process.env.DEEPSEEK_API_KEY ?? '', baseUrl: 'https://api.deepseek.com/v1', temperature: 0.7 });

  const trainSeeds = Array.from({ length: worlds }, (_u, i) => 10_000 + i);
  const heldSeeds = Array.from({ length: 12 }, (_u, i) => 20_000 + i);
  // filler documents of the train and dev splits only; test filler stays unseen
  const train = [
    ...extractionRows('train', trainSeeds, ['doc_000175', 'doc_000029', 'doc_000160', 'doc_000156', 'doc_000251'], rng),
    ...(await plannerRows('train', trainSeeds, paraphraseBudget, rng, client)),
  ];
  const held = [
    ...extractionRows('dev', heldSeeds, ['doc_000150', 'doc_000324'], rng),
    ...(await plannerRows('dev', heldSeeds, 150, rng, client)),
  ];
  mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/conversations.jsonl`, `${rng.shuffle(train).map((r) => JSON.stringify(r)).join('\n')}\n`);
  writeFileSync(`${out}/heldout.jsonl`, `${held.map((r) => JSON.stringify(r)).join('\n')}\n`);
  const kinds = (rows: Row[]) => {
    const extract = rows.filter((r) => r.messages[0]!.content === EXTRACTION_PROMPT);
    return `${extract.length} extraction (${extract.filter((r) => r.messages[2]!.content === '% nothing').length} nothing), ${rows.length - extract.length} planning`;
  };
  console.log(`train ${train.length}: ${kinds(train)}`);
  console.log(`heldout ${held.length}: ${kinds(held)}`);
  console.log(`wrote ${out}/conversations.jsonl and heldout.jsonl`);
}

if (process.argv[1]?.endsWith('run-compose-trainset.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
