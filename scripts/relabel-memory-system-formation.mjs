#!/usr/bin/env node
// One-off: memory-system rows recorded `"formation": "durable-raw-session-facts"` because the
// label was computed from --formation, which the --memory-system seam forces to `raw`. On a
// LangGraph or BM25 row that claims a Remembero formation that never ran. The runner now writes
// `memory-system:<id>` (src/evals/longmemeval-answer.ts); this rewrites the artifacts that were
// produced before that fix, so they do not have to be re-run to stop lying.
//
// Only the top-level `formation` string is touched. Everything else — every observation, every
// number — is left byte for byte as it was, so the edit is a one-line diff per file.
//
// Usage: node scripts/relabel-memory-system-formation.mjs [--check] [file…]
//   --check  report what would change and exit non-zero if anything would, writing nothing

import { readFileSync, writeFileSync } from 'node:fs';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

const RESULTS_DIR = 'docs/research/results';
const argv = process.argv.slice(2);
const check = argv.includes('--check');
const explicit = argv.filter((value) => !value.startsWith('--'));

const files =
  explicit.length > 0
    ? explicit
    : readdirSync(RESULTS_DIR)
        .filter((name) => name.endsWith('.json'))
        .sort()
        .map((name) => join(RESULTS_DIR, name));

// top-level keys sit at exactly two spaces; observation fields are nested deeper
const TOP_LEVEL_FORMATION = /^ {2}"formation": "(?<label>[^"]*)",$/m;

let changed = 0;
let skipped = 0;
for (const file of files) {
  const text = readFileSync(file, 'utf8');
  let run;
  try {
    run = JSON.parse(text);
  } catch {
    console.log(`skip  ${file} (not JSON)`);
    skipped += 1;
    continue;
  }
  const memorySystem = run?.settings?.memorySystem;
  if (typeof memorySystem !== 'string' || memorySystem.length === 0) {
    skipped += 1;
    continue; // a stock Remembero run: its formation label is the truth
  }
  const wanted = `memory-system:${memorySystem}`;
  if (run.formation === wanted) {
    console.log(`ok    ${file} (already ${wanted})`);
    skipped += 1;
    continue;
  }
  if (run.retrieval !== wanted) {
    throw new Error(
      `${file}: retrieval is ${String(run.retrieval)}, expected ${wanted} — refusing to guess`,
    );
  }
  const match = TOP_LEVEL_FORMATION.exec(text);
  if (match === null || match.groups?.label !== run.formation) {
    throw new Error(`${file}: could not locate the top-level formation line`);
  }
  const updated = text.replace(
    TOP_LEVEL_FORMATION,
    `  "formation": ${JSON.stringify(wanted)},`,
  );
  const reparsed = JSON.parse(updated);
  if (reparsed.formation !== wanted) {
    throw new Error(`${file}: rewrite did not take`);
  }
  console.log(`fix   ${file}: ${run.formation} -> ${wanted}`);
  changed += 1;
  if (!check) writeFileSync(file, updated, 'utf8');
}

console.log(
  `${check ? 'would change' : 'changed'} ${changed} file(s); left ${skipped} alone`,
);
if (check && changed > 0) process.exit(1);
