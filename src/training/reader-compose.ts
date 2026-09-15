/**
 * Training-data composer: a base distilled directory with the student's mined misses
 * mixed in at a set share.
 *
 * Every source must be rendered under one contract: a miss row is the teacher's line as
 * written, so mixing it with base rows of another contract would train two prompts at
 * once. Misses mined from `heldout.jsonl` never enter training rows, and a miss written
 * twice (a resumed mine) counts once. Row counts come from the files, never from a
 * manifest. The seed fixes which base rows are kept, the order misses are cycled in, and
 * the final order; meta twins stay line-aligned and each composed meta row names its source.
 */
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { join, resolve } from 'node:path';
import { readDistilledFile } from './reader-mine.js';
import type { DistilledMetaRow } from './reader-rerender.js';
import { createRng } from './rng.js';

export interface ComposeOptions {
  base: string;
  misses: string;
  out: string;
  /** The heldout source directory; the base directory when absent. */
  heldout?: string;
  /** The share of composed rows taken from misses, 0 to 1. */
  share: number;
  rows: number;
  seed: number;
  now?: Date;
}

type Contract = Record<string, unknown> & { id: string };

function manifestContract(dir: string): Contract | undefined {
  const path = join(dir, 'manifest.json');
  if (!existsSync(path)) return undefined;
  const contract = (
    JSON.parse(readFileSync(path, 'utf8')) as { contract?: unknown }
  ).contract;
  return contract !== null &&
    typeof contract === 'object' &&
    typeof (contract as { id?: unknown }).id === 'string'
    ? (contract as Contract)
    : undefined;
}

function typeCounts(
  rows: ReadonlyArray<{ type: string }>,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) counts[row.type] = (counts[row.type] ?? 0) + 1;
  return counts;
}

type MissMetaRow = DistilledMetaRow & { file?: unknown; index?: unknown };

export function composeTraining(options: ComposeOptions) {
  const { base, misses, out, share, rows, seed } = options;
  const heldout = options.heldout ?? base;
  if (!Number.isInteger(rows) || rows <= 0)
    throw new Error(`--rows must be a positive integer, got ${rows}`);
  if (!Number.isFinite(share) || share < 0 || share > 1)
    throw new Error(`--miss-share must be from 0 to 1, got ${share}`);
  if (!Number.isInteger(seed))
    throw new Error(`--seed must be an integer, got ${seed}`);
  for (const [name, dir] of [
    ['base', base],
    ['misses', misses],
    ['heldout', heldout],
  ] as const)
    if (resolve(dir) === resolve(out))
      throw new Error(`--out must not be the ${name} directory`);
  if (existsSync(join(out, 'conversations.jsonl')))
    throw new Error(
      `${join(out, 'conversations.jsonl')} already exists; compose writes a fresh directory`,
    );

  // one contract across every source, or nothing is written
  const sources = [
    ['base', base, manifestContract(base)],
    ['misses', misses, manifestContract(misses)],
    ['heldout', heldout, manifestContract(heldout)],
  ] as const;
  const ids = sources.map(([, , contract]) => contract?.id);
  if (ids.some((id) => id === undefined) || new Set(ids).size !== 1)
    throw new Error(
      `compose needs every source under one contract: ${sources
        .map(
          ([name, dir, contract]) =>
            `${name} ${dir} has ${contract === undefined ? 'no contract' : contract.id}`,
        )
        .join('; ')}`,
    );
  const contract = sources[0][2]!;

  const baseFile = readDistilledFile(base, 'conversations.jsonl');
  const missFile = readDistilledFile(misses, 'misses.jsonl');
  const heldoutFile = readDistilledFile(heldout, 'heldout.jsonl');

  const seen = new Set<string>();
  let heldoutDropped = 0;
  let duplicatesDropped = 0;
  const usable: Array<{ line: string; meta: MissMetaRow }> = [];
  missFile.rows.forEach((meta: MissMetaRow, i) => {
    if (typeof meta.file !== 'string' || !Number.isInteger(meta.index))
      throw new Error(
        `${join(misses, 'misses.jsonl.meta.jsonl')} row ${i} has no source file and index`,
      );
    if (meta.file === 'heldout.jsonl') {
      heldoutDropped += 1;
      return;
    }
    const key = `${meta.file}#${String(meta.index)}`;
    if (seen.has(key)) {
      duplicatesDropped += 1;
      return;
    }
    seen.add(key);
    usable.push({ line: missFile.lines[i]!, meta });
  });

  const rng = createRng(seed);
  const baseTarget = Math.round(rows * (1 - share));
  const baseIndices = baseFile.rows.map((_, i) => i);
  const keptIndices =
    baseIndices.length > baseTarget
      ? rng.shuffle(baseIndices).slice(0, baseTarget)
      : baseIndices;
  const missCount = rows - keptIndices.length;
  if (missCount > 0 && usable.length === 0)
    throw new Error(
      `compose needs ${missCount} miss rows but ${misses} has no usable misses`,
    );
  const missOrder = rng.shuffle(usable);
  const composed = [
    ...keptIndices.map((i) => ({
      line: baseFile.lines[i]!,
      meta: { ...baseFile.rows[i]!, source: 'base' as const },
    })),
    ...Array.from({ length: missCount }, (_, i) => {
      const miss = missOrder[i % missOrder.length]!;
      return {
        line: miss.line,
        meta: { ...miss.meta, source: 'miss' as const },
      };
    }),
  ];
  const ordered = rng.shuffle(composed);

  mkdirSync(out, { recursive: true });
  writeFileSync(
    join(out, 'conversations.jsonl'),
    ordered.map((row) => `${row.line}\n`).join(''),
  );
  writeFileSync(
    join(out, 'conversations.jsonl.meta.jsonl'),
    ordered.map((row) => `${JSON.stringify(row.meta)}\n`).join(''),
  );
  copyFileSync(join(heldout, 'heldout.jsonl'), join(out, 'heldout.jsonl'));
  copyFileSync(
    join(heldout, 'heldout.jsonl.meta.jsonl'),
    join(out, 'heldout.jsonl.meta.jsonl'),
  );

  const realisedShare = missCount / rows;
  if (Math.abs(realisedShare - share) > 0.01)
    console.error(
      `compose: the realised miss share is ${realisedShare} (${missCount} of ${rows} rows), not the ${share} asked for; the base has ${baseFile.rows.length} rows`,
    );
  const metaOf = (source: 'base' | 'miss') =>
    composed.filter((row) => row.meta.source === source).map((row) => row.meta);
  const manifest = {
    sources: {
      base: { path: base, contract: contract.id, rows: baseFile.rows.length },
      misses: {
        path: misses,
        contract: contract.id,
        rows: missFile.rows.length,
        heldoutDropped,
        duplicatesDropped,
        usable: usable.length,
      },
      heldout: { path: heldout, contract: contract.id },
    },
    share,
    realisedShare,
    rows,
    train: { base: keptIndices.length, miss: missCount },
    heldout: heldoutFile.rows.length,
    byType: {
      base: typeCounts(metaOf('base')),
      miss: typeCounts(metaOf('miss')),
    },
    seed,
    contract,
    generatedAt: (options.now ?? new Date()).toISOString(),
  };
  writeFileSync(
    join(out, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  return manifest;
}

export type ComposeManifest = ReturnType<typeof composeTraining>;
