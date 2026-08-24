#!/usr/bin/env node
import { existsSync, lstatSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  createDocumentMemorgExport,
  serializeDocumentMemorgExport,
  verifyDocumentMemorgExport,
} from '../document/memorg.js';

interface Options {
  output?: string;
  publicOutput?: string;
  check: boolean;
  help: boolean;
}

function parseArgs(argv: readonly string[]): Options {
  const options: Options = { check: false, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      options.help = true;
      continue;
    }
    if (arg === '--check') {
      options.check = true;
      continue;
    }
    if (arg === '--output') {
      const value = argv[++index];
      if (value === undefined || value.trim() === '') throw new Error('--output requires a path');
      options.output = value;
      continue;
    }
    if (arg === '--public-output') {
      const value = argv[++index];
      if (value === undefined || value.trim() === '') {
        throw new Error('--public-output requires a path');
      }
      options.publicOutput = value;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  return options;
}

function help(): string {
  return [
    'Usage: node dist/evals/run-document-memorg.js [options]',
    '',
    'Options:',
    '  --output <path>  Write the deterministic Memorg import artifact',
    '  --public-output <path>  Write a second hash-identical browser artifact',
    '  --check          Verify the generated artifact and print its summary',
    '  --help, -h       Show this help text',
  ].join('\n');
}

function writeOutput(path: string, text: string): void {
  const absolute = resolve(path);
  if (existsSync(absolute)) {
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error('refusing non-regular Memorg output file');
    }
  }
  writeFileSync(absolute, `${text}\n`, 'utf8');
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  process.stdout.write(`${help()}\n`);
  process.exit(0);
}

const exported = createDocumentMemorgExport();
const serialized = serializeDocumentMemorgExport(exported);
if (options.output === undefined) process.stdout.write(`${serialized}\n`);
else writeOutput(options.output, serialized);
if (options.publicOutput !== undefined) writeOutput(options.publicOutput, serialized);

if (options.check) {
  const result = verifyDocumentMemorgExport(serialized);
  process.stderr.write(
    `Memorg export verified: ${result.itemCount} items, ${result.documentCount} documents, ` +
      `${result.acceptedClaimCount} accepted claims, ${result.proposedClaimCount} proposed claims, ` +
      `${result.reviewedRuleCount} reviewed rules, ${result.questionCount} questions, ` +
      `sha256 ${result.sha256}\n`
  );
}
