/**
 * `npm run compose:engine` — the Proved path on Compose: extract every page of each haystack into
 * the schema, drop what the page does not support, let the engine answer, score it like any
 * reader. Also measures extraction itself against the world's true facts.
 *
 * Extraction is cached by page text, so the 100, 500 and 1000-page tiers (which share filler and
 * world pages) pay for each page once.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { OpenRouterClient } from '../llm/client.js';
import { EngineAnswerer } from '../compose/engine-answer.js';
import { EXTRACTION_PROMPT, normaliseName, parseSchemaFacts, repairFacts, unsupportedReason, type Fact } from '../compose/extract.js';
import type { ComposeGold } from '../compose/questions.js';
import { worldAsSchemaFacts } from '../compose/schema-truth.js';
import { scoreCompose, type ComposeOutcome } from '../compose/score.js';
import { engineCall, parsePlan, planProblem, PLANNER_PROMPT } from '../compose/planner.js';
import { generateWorld, ORGANISATIONS } from '../compose/world.js';
import { pagesFromTextFile } from './document-corpus.js';
import { mapConcurrent } from './map-concurrent.js';

interface SpecQuestion {
  id: string;
  question: string;
  datasetKind: string;
  compose: ComposeGold;
  params: Record<string, string | number>;
}

const key = (f: Fact) => `${f.predicate}(${f.args.map((a) => (typeof a === 'string' ? normaliseName(a) : a)).join('|')})`;

/**
 * Compare extracted facts with the truth person-for-person: a resignation written "Tobias Brennan"
 * on one page and "Mx Brennan" on another is the same fact, so names are resolved to the world's
 * full names (full name, surname, or initial and surname) before comparing.
 */
function canonicalKey(fact: Fact, people: Map<string, string>): string {
  const resolve = (value: string | number) => {
    if (typeof value !== 'string') return value;
    const name = normaliseName(value);
    return people.get(name) ?? name;
  };
  return key({ predicate: fact.predicate, args: fact.args.map(resolve) });
}

function peopleIndex(seed: number, split: 'train' | 'dev' | 'test'): Map<string, string> {
  const index = new Map<string, string>();
  for (const p of generateWorld(seed, split).people) {
    const full = normaliseName(`${p.first} ${p.last}`);
    for (const alias of [full, normaliseName(p.last), normaliseName(`${p.first[0]}. ${p.last}`)]) index.set(alias, full);
  }
  return index;
}

async function main() {
  const argv = process.argv.slice(2);
  const flag = (name: string, fallback: string) => {
    const at = argv.indexOf(name);
    return at >= 0 && argv[at + 1] !== undefined ? argv[at + 1]! : fallback;
  };
  const specPath = flag('--spec', 'benchmarks/compose/test/spec.json');
  const model = flag('--model', 'deepseek-chat');
  const price = flag('--price', '0.15,0.6').split(',').map(Number);
  const output = flag('--output', 'results/compose-engine-deepseek.json');
  // oracle: the question generator's parameters; llm: a model reads the question text
  const planner = flag('--planner', 'oracle');
  // extraction passes over record-like pages: 1 (single call) or 2 (second independent pass, merged)
  const passes = Number(flag('--passes', '2'));
  const plannerModel = flag('--planner-model', 'deepseek-chat');
  const plannerClient = new OpenRouterClient({
    model: plannerModel,
    apiKey: flag('--planner-api-key', process.env.DEEPSEEK_API_KEY ?? ''),
    baseUrl: flag('--planner-base-url', 'https://api.deepseek.com/v1'),
    timeoutMs: 120_000,
  });
  const planVersion = createHash('sha256').update(PLANNER_PROMPT).digest('hex').slice(0, 10);
  const planCachePath = `.cache/compose/plans.${plannerModel.replace(/[^a-z0-9.-]+/gi, '_')}.${planVersion}.json`;
  const planCache: Record<string, string> = existsSync(planCachePath) ? JSON.parse(readFileSync(planCachePath, 'utf8')) : {};
  const client = new OpenRouterClient({
    model,
    apiKey: flag('--api-key', process.env.DEEPSEEK_API_KEY ?? ''),
    baseUrl: flag('--base-url', 'https://api.deepseek.com/v1'),
    timeoutMs: 120_000,
  });
  const spec = JSON.parse(readFileSync(specPath, 'utf8')) as {
    name: string;
    tiers: Array<{ name: string; documents: string[] }>;
    sources: Array<{ id: string; text: string; questions: SpecQuestion[] }>;
  };

  // keyed by prompt too: a schema change must never reuse replies written for the old schema
  const promptVersion = createHash('sha256').update(EXTRACTION_PROMPT).digest('hex').slice(0, 10);
  const cachePath = `.cache/compose/extract.${model.replace(/[^a-z0-9.-]+/gi, '_')}.${promptVersion}.json`;
  const cache: Record<string, string> = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, 'utf8')) : {};
  let promptTokens = 0;
  let completionTokens = 0;
  let calls = 0;

  const report: unknown[] = [];
  console.log('tier  | world | pages | kept facts | ungrounded | fact precision | fact recall | right | WRONG | part | decl');
  for (const tier of spec.tiers) {
    for (const documentId of tier.documents) {
      const source = spec.sources.find((s) => s.id === documentId)!;
      const seed = Number(/-w(\d+)-/.exec(documentId)![1]);
      const split = (/^compose-(train|dev|test)(?:v1)?-/.exec(documentId)?.[1] ?? 'test') as 'train' | 'dev' | 'test';
      const pages = pagesFromTextFile(source.text);

      // a page dense with dates is likely a record page: it gets a second, independent pass,
      // because one extraction call can answer "% nothing" for a register it read correctly the
      // run before, and a lost register becomes a confidently wrong answer downstream
      const dense = (text: string) => (text.match(/\b(19|20)\d{2}\b/g) ?? []).length >= 3;
      const jobs = pages.flatMap((page) => {
        const hash = createHash('sha256').update(page.text).digest('hex');
        return [
          { page, key: hash, second: false },
          ...(passes >= 2 && dense(page.text) ? [{ page, key: `${hash}:2`, second: true }] : []),
        ];
      });
      await mapConcurrent(jobs.filter((job) => cache[job.key] === undefined), 24, async (job) => {
        const reply = await client.completeWithUsage(
          [
            { role: 'system', content: EXTRACTION_PROMPT },
            {
              role: 'user',
              content: job.second
                ? `Read every line of this page, including tables and continued registers, and extract every record it states.\n\n${job.page.text.slice(0, 12_000)}`
                : job.page.text.slice(0, 12_000),
            },
          ],
          { maxTokens: 2_048 },
        );
        promptTokens += reply.usage?.promptTokens ?? 0;
        completionTokens += reply.usage?.completionTokens ?? 0;
        calls += 1;
        cache[job.key] = reply.content;
      });
      mkdirSync('.cache/compose', { recursive: true });
      writeFileSync(cachePath, JSON.stringify(cache));

      const grounded: Fact[] = [];
      const dropped: Array<{ page: number; fact: string; reason: string }> = [];
      for (const page of pages) {
        const hash = createHash('sha256').update(page.text).digest('hex');
        const replies = [cache[hash] ?? '', ...(passes >= 2 ? [cache[`${hash}:2`] ?? ''] : [])];
        const seen = new Set<string>();
        for (const fact of replies.flatMap(parseSchemaFacts)) {
          if (seen.has(key(fact))) continue;
          seen.add(key(fact));
          const reason = unsupportedReason(fact, page.text);
          if (reason === undefined) grounded.push(fact);
          else dropped.push({ page: page.page, fact: key(fact), reason });
        }
      }
      const { kept, repaired, rejected } = repairFacts(grounded);

      const people = peopleIndex(seed, split);
      const truth = new Set(worldAsSchemaFacts(generateWorld(seed, split)).map((f) => canonicalKey(f, people)));
      // facts of the sister organisations are true too: count them for precision, not for recall
      const sisterTruth = new Set<string>();
      if (/v1-|-v1/.test(documentId) || /v1/.test(documentId)) {
        const sisterPool = split === 'train' ? 'dev' : 'train';
        const main = generateWorld(seed, split);
        const refs = new Set(main.contracts.map((c) => c.ref));
        for (let k = 0; k < 2; k += 1) {
          const sister = generateWorld(seed * 10 + 7_000 + k, sisterPool, {
            organisation: ORGANISATIONS.filter((o) => o !== main.organisation)[(seed + k) % (ORGANISATIONS.length - 1)]!,
            avoidRefs: refs,
          });
          const sisterPeople = new Map<string, string>();
          for (const p of sister.people) {
            const full = normaliseName(`${p.first} ${p.last}`);
            for (const alias of [full, normaliseName(p.last), normaliseName(`${p.first[0]}. ${p.last}`)]) sisterPeople.set(alias, full);
          }
          for (const f of worldAsSchemaFacts(sister)) sisterTruth.add(canonicalKey(f, sisterPeople));
        }
      }
      const keptKeys = new Set(kept.map((f) => canonicalKey(f, people)));
      const truePositives = [...keptKeys].filter((k) => truth.has(k)).length;
      const supported = [...keptKeys].filter((k) => truth.has(k) || sisterTruth.has(k)).length;

      const days = source.questions.map((q) => q.params.date).filter((d): d is number => typeof d === 'number');
      const engine = new EngineAnswerer(kept, days);
      if (planner === 'llm') {
        await mapConcurrent(source.questions.filter((q) => planCache[q.question] === undefined), 8, async (q) => {
          const reply = await plannerClient.completeWithUsage([
            { role: 'system', content: PLANNER_PROMPT },
            { role: 'user', content: q.question },
          ]);
          planCache[q.question] = reply.content;
        });
        writeFileSync(planCachePath, JSON.stringify(planCache, null, 1));
      }
      const outcomes = source.questions.map((q) => {
        let call = { family: q.datasetKind, params: q.params };
        let planIssue: string | undefined;
        if (planner === 'llm') {
          const plan = parsePlan(planCache[q.question] ?? '');
          planIssue = planProblem(q.question, plan);
          call = planIssue === undefined ? engineCall(plan) : { family: 'none', params: {} };
        }
        let answer: string;
        let engineError: string | undefined;
        try {
          answer = engine.answer(call.family, call.params);
        } catch (error) {
          // an evaluation the engine cannot finish is not an answer: Unknown, with the reason kept
          answer = 'unknown';
          engineError = error instanceof Error ? error.message : String(error);
        }
        return {
          id: q.id,
          family: q.datasetKind,
          question: q.question,
          gold: q.compose.display,
          plan: planner === 'llm' ? call : undefined,
          planIssue,
          engineError,
          answer,
          outcome: scoreCompose(answer, q.compose),
        };
      });
      const count = (o: ComposeOutcome) => outcomes.filter((x) => x.outcome === o).length;
      const pct = (n: number, d: number) => `${((100 * n) / d).toFixed(1)}%`;
      console.log(
        [
          tier.name.padEnd(5),
          String(seed).padStart(5),
          String(pages.length).padStart(5),
          String(keptKeys.size).padStart(10),
          `${dropped.length}+${rejected.length}`.padStart(10),
          pct(supported, keptKeys.size).padStart(14),
          pct(truePositives, truth.size).padStart(11),
          pct(count('correct'), outcomes.length).padStart(5),
          pct(count('wrong'), outcomes.length).padStart(5),
          pct(count('partial'), outcomes.length).padStart(4),
          pct(count('declined'), outcomes.length).padStart(4),
        ].join(' | '),
      );
      report.push({
        tier: tier.name,
        world: seed,
        pages: pages.length,
        extraction: {
          kept: keptKeys.size,
          dropped: dropped.length,
          repaired,
          rejectedByTypeCheck: rejected.map(key),
          truth: truth.size,
          truePositives,
          falseFacts: [...keptKeys].filter((k) => !truth.has(k) && !sisterTruth.has(k)),
          missedFacts: [...truth].filter((k) => !keptKeys.has(k)),
          droppedSample: dropped.slice(0, 40),
        },
        outcomes,
      });
    }
  }
  const cost = (promptTokens * price[0]! + completionTokens * price[1]!) / 1e6;
  console.log(`\nextraction: ${calls} new calls, ${promptTokens} + ${completionTokens} tokens, $${cost.toFixed(2)}`);
  writeFileSync(output, `${JSON.stringify({ model, passes, planner, plannerModel: planner === 'llm' ? plannerModel : null, spec: specPath, ranAt: new Date().toISOString(), costUsd: cost, runs: report }, null, 1)}\n`);
  console.log(`wrote ${output}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
