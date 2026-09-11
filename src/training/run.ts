/**
 * Generate execution-verified query-dialect training data.
 *
 *   npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training
 *   npm run train:data -- --no-paraphrase          # zero-cost dry run
 *   npm run train:data -- --repair-share 0.2       # add repair-turn conversations for 20% of query examples
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
import { generateWorld, type World } from './worlds.js';
import {
  generateExtractionExamples,
  renderRequestText,
  toExtractionConversation,
  type ExtractionExample,
  type Renderer,
} from './extraction-data.js';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import type { LlmClient } from '../llm/client.js';

export type TrainingTask = 'query' | 'extraction';

export interface RunOptions {
  /** Which task families to generate (default: both). */
  tasks?: TrainingTask[];
  /** Self atom used in extraction examples (default 'user'). */
  selfAtom?: string;
  /** Extraction examples attempted per kind per world (default 2; each is one rendering call). */
  extractionPerKind?: number;
  /** Share of query examples that also get a repair-turn conversation (default 0). */
  repairShare?: number;
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

/** Luna renders facts to text; results are cached by request so reruns are free. */
export function createLlmRenderer(
  client: LlmClient,
  cacheDir: string,
): Renderer {
  mkdirSync(cacheDir, { recursive: true });
  return async (request) => {
    const instruction = renderRequestText(request);
    const key = createHash('sha256')
      .update(`${PARAPHRASE_MODEL}\n${instruction}`)
      .digest('hex');
    const path = join(cacheDir, `render-${key}.txt`);
    if (existsSync(path)) return readFileSync(path, 'utf8');
    const text = (
      await client.complete([{ role: 'user', content: instruction }])
    ).trim();
    writeFileSync(path, text);
    return text;
  };
}

async function extractionLines(
  worlds: World[],
  options: RunOptions,
  renderer: Renderer | undefined,
  heldout: Set<string>,
): Promise<{
  train: string[];
  heldout: string[];
  count: number;
  byKind: Record<string, number>;
}> {
  const selfAtom = options.selfAtom ?? 'user';
  const train: string[] = [];
  const held: string[] = [];
  const byKind: Record<string, number> = {};
  if (renderer === undefined) return { train, heldout: held, count: 0, byKind };
  // Render worlds eight at a time; within a world rendering stays sequential
  // (each example is one call) and output order is fixed by world order.
  const perWorld: ExtractionExample[][] = new Array(worlds.length);
  const batch = 8;
  for (let start = 0; start < worlds.length; start += batch) {
    await Promise.all(
      worlds.slice(start, start + batch).map(async (world, offset) => {
        perWorld[start + offset] = await generateExtractionExamples(
          world,
          createRng(options.seed * 104729 + world.seed),
          renderer,
          { selfAtom, perKind: options.extractionPerKind ?? 2 },
        );
      }),
    );
  }
  worlds.forEach((world, index) => {
    for (const example of perWorld[index] ?? []) {
      byKind[example.kind] = (byKind[example.kind] ?? 0) + 1;
      const line = JSON.stringify(
        toExtractionConversation(world, example, selfAtom),
      );
      (heldout.has(world.id) ? held : train).push(line);
    }
  });
  return { train, heldout: held, count: train.length + held.length, byKind };
}

export async function generateTrainingData(
  options: RunOptions,
  paraphraser?: Paraphraser,
  extractionRenderer?: Renderer,
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
    ...(options.repairShare === undefined
      ? {}
      : { repairShare: options.repairShare }),
  });
  const tasks = options.tasks ?? ['query', 'extraction'];
  let trainText = tasks.includes('query') ? train : '';
  let heldText = tasks.includes('query') ? held : '';
  let extraction: Awaited<ReturnType<typeof extractionLines>> | undefined;
  if (tasks.includes('extraction')) {
    let renderer = extractionRenderer;
    if (renderer === undefined && options.paraphrase) {
      loadEnv();
      renderer = createLlmRenderer(
        clientFromEnv({ ...process.env, LLM_MODEL: PARAPHRASE_MODEL }),
        join(options.out, 'cache'),
      );
    }
    extraction = await extractionLines(worlds, options, renderer, heldout);
    trainText += extraction.train.map((l) => `${l}\n`).join('');
    heldText += extraction.heldout.map((l) => `${l}\n`).join('');
  }
  const finalManifest: Manifest = {
    ...manifest,
    train: trainText.split('\n').filter(Boolean).length,
    heldout: heldText.split('\n').filter(Boolean).length,
    tasks,
    ...(extraction === undefined
      ? {}
      : { extraction: { count: extraction.count, byKind: extraction.byKind } }),
  };
  mkdirSync(options.out, { recursive: true });
  writeFileSync(join(options.out, 'conversations.jsonl'), trainText);
  writeFileSync(join(options.out, 'heldout.jsonl'), heldText);
  writeFileSync(
    join(options.out, 'manifest.json'),
    `${JSON.stringify(finalManifest, null, 2)}\n`,
  );
  return finalManifest;
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
    tasks: flag('--tasks', 'query,extraction').split(',') as TrainingTask[],
    selfAtom: flag('--self', 'user'),
    extractionPerKind: Number(flag('--extraction-per-kind', '2')),
    repairShare: Number(flag('--repair-share', '0')),
  });
  console.log(JSON.stringify(manifest, null, 2));
}
