/**
 * `npm run compose:build` — generate Compose worlds, write them into haystacks of 100, 500 and
 * 1000 pages, and emit a spec the page-scale harness (`eval:doc-recall`) runs unchanged.
 *
 * Also writes, per world, the Datalog program the documents were written from and each question's
 * engine query — the audit trail that makes every label checkable.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { OpenRouterClient } from '../llm/client.js';
import { PARAPHRASE_PROMPT, paraphraseProblem } from '../compose/paraphrase.js';
import { LineRewriter } from '../compose/reword.js';
import type { Section } from '../compose/render.js';
import { mapConcurrent } from './map-concurrent.js';
import { pagesFromTextFile } from './document-corpus.js';
import { WorldProgram, RULES, worldFacts } from '../compose/program.js';
import { generateQuestions } from '../compose/questions.js';
import { paginate, renderWorld } from '../compose/render.js';
import { buildHaystack, cleanFiller, relativeDepths, worldVocabulary } from '../compose/haystack.js';
import { generateWorld, ORGANISATIONS, type Split } from '../compose/world.js';

const TIERS = [100, 500, 1000];

/** Filler documents per split, drawn from the XL-DocBench pages already on disk; disjoint. */
const FILLER: Record<Split, string[][]> = {
  test: [
    ['doc_000209', 'doc_000267'],
    ['doc_000120', 'doc_000181', 'doc_000179'],
  ],
  dev: [['doc_000032', 'doc_000189'], ['doc_000189', 'doc_000032']],
  train: [['doc_000175', 'doc_000029', 'doc_000160'], ['doc_000156', 'doc_000251', 'doc_000150', 'doc_000324', 'doc_000220', 'doc_000029']],
};

const SEEDS: Record<Split, number[]> = { test: [101, 102], dev: [201, 202], train: [301, 302] };

/** v1: sister organisations hidden in the filler, and paraphrased questions. */
const V1_SISTERS = 2;

async function main() {
  const split = (process.argv[2] ?? 'test') as Split;
  const v3 = process.argv.includes('--v3');
  const v2 = v3 || process.argv.includes('--v2');
  const v1 = v2 || process.argv.includes('--v1');
  // --seeds 103,104 --name holdout-v1: fresh worlds, generated after the pipeline was last changed
  const seedsAt = process.argv.indexOf('--seeds');
  const nameAt = process.argv.indexOf('--name');
  const seeds = seedsAt >= 0 ? process.argv[seedsAt + 1]!.split(',').map(Number) : SEEDS[split];
  const outDir = `benchmarks/compose/${nameAt >= 0 ? process.argv[nameAt + 1] : `${split}${v3 ? '-v3' : v2 ? '-v2' : v1 ? '-v1' : ''}`}`;
  const paraphraseCachePath = '.cache/compose/paraphrase.deepseek.json';
  const paraphrases: Record<string, string> = existsSync(paraphraseCachePath) ? JSON.parse(readFileSync(paraphraseCachePath, 'utf8')) : {};
  const client = new OpenRouterClient({
    model: 'deepseek-chat',
    apiKey: process.env.DEEPSEEK_API_KEY ?? '',
    baseUrl: 'https://api.deepseek.com/v1',
    temperature: 0.7,
  });
  let kept = 0;
  let rejected = 0;
  const rewriter = new LineRewriter(client);
  /** v2: prose lines reworded; a rewrite is kept only if every fact of the line survives on it. */
  const reword = async (sections: Section[]): Promise<Section[]> => {
    if (!v2) return sections;
    await rewriter.prepare([sections]);
    return rewriter.apply(sections);
  };
  const cacheDir = '.cache/compose';
  mkdirSync(outDir, { recursive: true });
  mkdirSync(cacheDir, { recursive: true });

  const sources: unknown[] = [];
  const tiers = TIERS.map((pages) => ({ name: `${pages}p`, documents: [] as string[] }));
  for (const [index, seed] of seeds.entries()) {
    const world = generateWorld(seed, split, v3 ? { adversarial: true } : {});
    const mainPages = paginate(await reword(renderWorld(world)));
    const questions = generateQuestions(world);

    // sister organisations: same documents, role titles and formats; people, suppliers and sites
    // drawn from another split's name pool so no answer becomes ambiguous
    const sisterPool: Split = split === 'train' ? 'dev' : 'train';
    const refs = new Set(world.contracts.map((c) => c.ref));
    const sisters = v1
      ? Array.from({ length: V1_SISTERS }, (_unused, k) =>
          generateWorld(seed * 10 + 7_000 + k, sisterPool, {
            organisation: ORGANISATIONS.filter((o) => o !== world.organisation)[(seed + k) % (ORGANISATIONS.length - 1)]!,
            avoidRefs: refs,
          }),
        )
      : [];
    const sisterPages: Array<{ text: string; keys: string[]; facts: import('../compose/render.js').SchemaFact[] }> = [];
    for (const sister of sisters) {
      for (const page of paginate(await reword(renderWorld(sister)))) sisterPages.push({ text: page.text, keys: [], facts: page.facts });
    }
    const worldPages = [...mainPages, ...sisterPages];

    if (v1) {
      await mapConcurrent(questions.filter((q) => paraphrases[q.question] === undefined), 8, async (q) => {
        const reply = await client.completeWithUsage([
          { role: 'system', content: PARAPHRASE_PROMPT },
          { role: 'user', content: q.question },
        ]);
        paraphrases[q.question] = reply.content.trim();
      });
      mkdirSync('.cache/compose', { recursive: true });
      writeFileSync(paraphraseCachePath, JSON.stringify(paraphrases, null, 1));
      for (const q of questions) {
        const candidate = paraphrases[q.question]!;
        if (paraphraseProblem(q, candidate) === undefined) {
          q.question = candidate;
          kept += 1;
        } else {
          rejected += 1;
        }
      }
    }
    const fillerRaw = FILLER[split][index % FILLER[split].length]!.flatMap((doc) =>
      pagesFromTextFile(`.cache/document-recall/${doc}.pages.txt`).map((p) => p.text),
    );
    const filler = cleanFiller(fillerRaw, [...worldVocabulary(world), ...sisters.flatMap(worldVocabulary)]);
    // main pages keep document order among themselves; sister pages get depths of their own
    const depths = [...relativeDepths(mainPages.length, seed), ...relativeDepths(sisterPages.length, seed + 1)];

    // the audit trail: facts, rules and each question's engine query
    writeFileSync(`${outDir}/w${seed}.dl`, `% Compose world ${seed} (${split}), ${world.organisation}\n${worldFacts(world).join('\n')}\n${RULES}`);
    writeFileSync(
      `${outDir}/w${seed}.questions.jsonl`,
      `${questions.map((q) => JSON.stringify({ id: q.id, family: q.family, hops: q.hops, question: q.question, gold: q.gold.display, datalog: q.datalog })).join('\n')}\n`,
    );
    void WorldProgram;

    for (const [t, total] of TIERS.entries()) {
      const hay = buildHaystack(worldPages, filler, total, depths);
      const id = `compose-${split}${v3 ? 'v3' : v2 ? 'v2' : v1 ? 'v1' : ''}-w${seed}-${total}p`;
      const textPath = `${cacheDir}/${id}.pages.txt`;
      writeFileSync(textPath, hay.pages.join('\f'));
      tiers[t]!.documents.push(id);
      sources.push({
        id,
        title: `${world.organisation} records (Compose world ${seed}) in ${total} pages`,
        sourceUrl: 'generated',
        text: textPath,
        labelPages: total,
        questions: questions.map((q) => ({
          id: `${q.id}-${total}p`,
          question: q.question,
          answer: q.gold.kind === 'unknown' ? null : q.gold.display,
          evidencePages: [...new Set(q.evidence.flatMap((key) => hay.pagesOf.get(key) ?? []))].sort((a, b) => a - b),
          datasetKind: q.family,
          answerFormat: 'Compose',
          hops: q.hops,
          compose: q.gold,
          params: q.params,
        })),
      });
      console.log(
        `${id}: ${hay.pages.length} pages (${worldPages.length} world pages at ${hay.worldPageNumbers.slice(0, 4).join(', ')}…), ` +
          `${questions.length} questions, filler ${filler.length}/${fillerRaw.length} pages kept`,
      );
    }
  }
  if (v1) console.log(`paraphrases: ${kept} kept, ${rejected} rejected (original question used)`);
  if (v2) console.log(`reworded lines: ${rewriter.stats.kept} kept, ${rewriter.stats.rejected} rejected (original line used)`);
  writeFileSync(
    `${outDir}/spec.json`,
    `${JSON.stringify({ name: `compose-${split}${v3 ? '-v3' : v2 ? '-v2' : v1 ? '-v1' : ''}`, labels: 'Compose: generated worlds, gold computed by the Datalog engine', tiers, sources }, null, 1)}\n`,
  );
  console.log(`wrote ${outDir}/spec.json`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
