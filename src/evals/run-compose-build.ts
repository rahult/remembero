/**
 * `npm run compose:build` — generate Compose worlds, write them into haystacks of 100, 500 and
 * 1000 pages, and emit a spec the page-scale harness (`eval:doc-recall`) runs unchanged.
 *
 * Also writes, per world, the Datalog program the documents were written from and each question's
 * engine query — the audit trail that makes every label checkable.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { pagesFromTextFile } from './document-corpus.js';
import { WorldProgram, RULES, worldFacts } from '../compose/program.js';
import { generateQuestions } from '../compose/questions.js';
import { paginate, renderWorld } from '../compose/render.js';
import { buildHaystack, cleanFiller, relativeDepths, worldVocabulary } from '../compose/haystack.js';
import { generateWorld, type Split } from '../compose/world.js';

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

function main() {
  const split = (process.argv[2] ?? 'test') as Split;
  const outDir = `benchmarks/compose/${split}`;
  const cacheDir = '.cache/compose';
  mkdirSync(outDir, { recursive: true });
  mkdirSync(cacheDir, { recursive: true });

  const sources: unknown[] = [];
  const tiers = TIERS.map((pages) => ({ name: `${pages}p`, documents: [] as string[] }));
  SEEDS[split].forEach((seed, index) => {
    const world = generateWorld(seed, split);
    const worldPages = paginate(renderWorld(world));
    const questions = generateQuestions(world);
    const fillerRaw = FILLER[split][index % FILLER[split].length]!.flatMap((doc) =>
      pagesFromTextFile(`.cache/document-recall/${doc}.pages.txt`).map((p) => p.text),
    );
    const filler = cleanFiller(fillerRaw, worldVocabulary(world));
    const depths = relativeDepths(worldPages.length, seed);

    // the audit trail: facts, rules and each question's engine query
    writeFileSync(`${outDir}/w${seed}.dl`, `% Compose world ${seed} (${split}), ${world.organisation}\n${worldFacts(world).join('\n')}\n${RULES}`);
    writeFileSync(
      `${outDir}/w${seed}.questions.jsonl`,
      `${questions.map((q) => JSON.stringify({ id: q.id, family: q.family, hops: q.hops, question: q.question, gold: q.gold.display, datalog: q.datalog })).join('\n')}\n`,
    );
    void WorldProgram;

    for (const [t, total] of TIERS.entries()) {
      const hay = buildHaystack(worldPages, filler, total, depths);
      const id = `compose-${split}-w${seed}-${total}p`;
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
  });
  writeFileSync(
    `${outDir}/spec.json`,
    `${JSON.stringify({ name: `compose-${split}`, labels: 'Compose: generated worlds, gold computed by the Datalog engine', tiers, sources }, null, 1)}\n`,
  );
  console.log(`wrote ${outDir}/spec.json`);
}

main();
