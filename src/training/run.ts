/**
 * Generate execution-verified query-dialect training data.
 *
 *   npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training
 *   npm run train:data -- --no-paraphrase          # zero-cost dry run
 *
 * Output: conversations.jsonl (train), heldout.jsonl (whole held-out worlds),
 * manifest.json (counts, rejections, paraphrase settings).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadEnv } from '../env.js';
import { clientFromEnv } from '../llm/client.js';
import { chooseHeldoutWorlds, exportDataset, type Manifest } from './export.js';
import {
  createLlmParaphraser,
  PARAPHRASE_MODEL,
  paraphraseExamples,
  type Paraphraser,
} from './paraphrase.js';
import { createRng } from './rng.js';
import { generateCandidates } from './templates.js';
import {
  assertRejectionRates,
  verifyAll,
  type Example,
  type Rejection,
} from './verify.js';
import { generateWorld } from './worlds.js';

export interface RunOptions {
  /** Target number of verified templated examples before paraphrasing (ignored when rounds is set). */
  examples: number;
  /** Draw exactly this many candidate rounds per world; keeps data size comparable across runs. */
  rounds?: number;
  worlds: number;
  paraphrases: number;
  seed: number;
  out: string;
  paraphrase: boolean;
}

const MAX_ROUNDS = 50;

export async function generateTrainingData(
  options: RunOptions,
  paraphraser?: Paraphraser,
): Promise<Manifest> {
  const worlds = Array.from({ length: options.worlds }, (_, i) =>
    generateWorld(options.seed * 1000 + i + 1),
  );
  const examples: Example[] = [];
  const rejections: Rejection[] = [];
  const seen = new Set<string>();
  // Draw candidates with a fresh rng each round (new anchors and phrasings)
  // until the target is met; identical (world, question, program) triples are dropped.
  const roundLimit = options.rounds ?? MAX_ROUNDS;
  let roundsDrawn = 0;
  for (
    let round = 0;
    round < roundLimit &&
    (options.rounds !== undefined || examples.length < options.examples);
    round += 1
  ) {
    roundsDrawn = round + 1;
    let added = 0;
    const roundCandidates = [];
    const roundRejections: Rejection[] = [];
    for (const world of worlds) {
      const candidates = generateCandidates(
        world,
        createRng(options.seed * 7919 + world.seed * 31 + round),
      );
      const verified = verifyAll(world, candidates);
      roundCandidates.push(...candidates);
      roundRejections.push(...verified.rejections);
      rejections.push(...verified.rejections);
      for (const example of verified.examples) {
        const key = `${example.world}\n${example.question}\n${example.program}`;
        if (seen.has(key)) continue;
        seen.add(key);
        examples.push(example);
        added += 1;
      }
    }
    // judged across all worlds: one world's single rejected candidate is not a template bug
    if (round === 0) assertRejectionRates(roundCandidates, roundRejections);
    if (added === 0 && options.rounds === undefined) break; // templates exhausted
  }

  const heldout = chooseHeldoutWorlds(worlds);
  let final = examples;
  let model: string | null = null;
  if (options.paraphrase && options.paraphrases > 0) {
    loadEnv();
    const active =
      paraphraser ??
      createLlmParaphraser(
        clientFromEnv({ ...process.env, LLM_MODEL: PARAPHRASE_MODEL }),
        join(options.out, 'cache'),
      );
    final = await paraphraseExamples(examples, active, options.paraphrases);
    model = PARAPHRASE_MODEL;
  }

  const {
    train,
    heldout: held,
    manifest,
  } = exportDataset({
    worlds,
    examples: final,
    heldoutWorldIds: heldout,
    rejections,
    seed: options.seed,
    paraphraseModel: model,
    paraphrasesPerExample: model ? options.paraphrases : 0,
    rounds: roundsDrawn,
  });
  mkdirSync(options.out, { recursive: true });
  writeFileSync(join(options.out, 'conversations.jsonl'), train);
  writeFileSync(join(options.out, 'heldout.jsonl'), held);
  writeFileSync(
    join(options.out, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  return manifest;
}

function flag(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] !== undefined
    ? process.argv[index + 1]
    : fallback;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];

if (invokedDirectly) {
  const manifest = await generateTrainingData({
    examples: Number(flag('--examples', '3000')),
    ...(process.argv.includes('--rounds')
      ? { rounds: Number(flag('--rounds', '4')) }
      : {}),
    worlds: Number(flag('--worlds', '60')),
    paraphrases: Number(flag('--paraphrases', '3')),
    seed: Number(flag('--seed', '7')),
    out: flag('--out', 'data/training'),
    paraphrase: !process.argv.includes('--no-paraphrase'),
  });
  console.log(JSON.stringify(manifest, null, 2));
}
