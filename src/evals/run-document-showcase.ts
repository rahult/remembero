#!/usr/bin/env node
import { existsSync, lstatSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  evaluateDocumentShowcases,
  formatDocumentEvaluationReport,
} from './document-showcase.js';

interface Options {
  json: boolean;
  check: boolean;
  output?: string;
}

function parseArgs(argv: readonly string[]): Options {
  const options: Options = { json: false, check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--json') options.json = true;
    else if (arg === '--check') options.check = true;
    else if (arg === '--output') {
      const value = argv[++index];
      if (value === undefined || value.trim() === '') throw new Error('--output requires a path');
      options.output = value;
      options.json = true;
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node dist/evals/run-document-showcase.js [--json] [--check] [--output <path>]\n'
      );
      process.exit(0);
    } else throw new Error(`unknown argument: ${arg}`);
  }
  return options;
}

function writeOutput(path: string, text: string): void {
  const absolute = resolve(path);
  if (existsSync(absolute)) {
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error('refusing non-regular document evaluation output file');
    }
  }
  writeFileSync(absolute, `${text}\n`, 'utf8');
}

const options = parseArgs(process.argv.slice(2));
const report = evaluateDocumentShowcases();
const output = options.json
  ? JSON.stringify(report, null, 2)
  : formatDocumentEvaluationReport(report);

if (options.output === undefined) process.stdout.write(`${output}\n`);
else writeOutput(options.output, output);

if (options.check && report.aggregate.status !== 'pass') process.exitCode = 1;
