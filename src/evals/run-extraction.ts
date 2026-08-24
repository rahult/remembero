#!/usr/bin/env node
import { existsSync, lstatSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { serializeClause } from '../engine/index.js';
import { loadEnv } from '../env.js';
import {
  DEFAULT_MODEL,
  OpenRouterClient,
  addLlmUsage,
  emptyLlmUsageTotals,
  type LlmClient,
} from '../llm/client.js';
import { rememberText } from '../llm/pipeline.js';
import { MemoryStore } from '../store/store.js';
import {
  EXTRACTION_EVAL_CASES,
  extractionObservationIsCorrect,
  scoreExtractionEval,
  type ExtractionEvalObservation,
} from './extraction.js';

interface EvalArgs {
  models: string[];
  json: boolean;
  caseIds: Set<string> | null;
  output: string | undefined;
}

const USAGE = `Usage: npm run eval:extract -- [options]

Options:
  --models <a,b>       OpenRouter model IDs (default: LLM_MODEL or ${DEFAULT_MODEL})
  --cases <a,b>        Run only selected case IDs
  --json               Print machine-readable JSON
  --output <path>      Write machine-readable JSON to a regular file
`;

function listValue(argv: string[], index: number, flag: string): string[] {
  const value = argv[index + 1];
  if (!value) throw new Error(`${flag} needs a comma-separated value`);
  const items = value.split(',').map((item) => item.trim()).filter(Boolean);
  if (items.length === 0) throw new Error(`${flag} needs at least one value`);
  return items;
}

function parseArgs(argv: string[]): EvalArgs {
  const args: EvalArgs = {
    models: [process.env.LLM_MODEL ?? DEFAULT_MODEL],
    json: false,
    caseIds: null,
    output: undefined,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--models') {
      args.models = listValue(argv, index, arg);
      index++;
    } else if (arg === '--cases') {
      args.caseIds = new Set(listValue(argv, index, arg));
      index++;
    } else if (arg === '--json') {
      args.json = true;
    } else if (arg === '--output') {
      const value = argv[index + 1];
      if (!value || value.trim() === '') throw new Error('--output needs a path');
      args.output = value;
      args.json = true;
      index++;
    } else if (arg === '--help' || arg === '-h') {
      console.log(USAGE);
      process.exit(0);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return args;
}

function writeJsonOutput(path: string, text: string): void {
  const absolute = resolve(path);
  if (existsSync(absolute)) {
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error('refusing non-regular extraction evaluation output file');
    }
  }
  writeFileSync(absolute, `${text}\n`, 'utf8');
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function runConfiguration(
  model: string,
  caseIds: Set<string> | null,
  apiKey: string,
  baseUrl: string
): Promise<ExtractionEvalObservation[]> {
  const client = new OpenRouterClient({ apiKey, baseUrl, model });
  const cases = EXTRACTION_EVAL_CASES.filter((testCase) =>
    caseIds === null ? true : caseIds.has(testCase.id)
  );
  const observations: ExtractionEvalObservation[] = [];

  for (const testCase of cases) {
    const root = mkdtempSync(join(tmpdir(), 'rembero-extraction-eval-'));
    const store = new MemoryStore(root);
    if (testCase.initialProgram.trim() !== '') {
      store.importClauses('default', testCase.initialProgram);
    }
    let llmCalls = 0;
    let usage = emptyLlmUsageTotals();
    const llm: LlmClient = {
      async complete(messages) {
        llmCalls++;
        const completion = await client.completeWithUsage(messages);
        usage = addLlmUsage(usage, completion.usage);
        return completion.content;
      },
    };
    const started = performance.now();
    try {
      try {
        const result = await rememberText(
          { store, llm },
          testCase.input,
          'default',
          testCase.trust === undefined ? {} : { trust: testCase.trust }
        );
        observations.push({
          case: testCase,
          model,
          outcome: 'completed',
          actualClauses: store.load('default').map(serializeClause),
          added: result.added,
          duplicates: result.duplicates,
          retracted: result.retracted,
          llmCalls,
          usage,
          durationMs: performance.now() - started,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const expectedPattern = testCase.expectedErrorPattern;
        const rejected =
          testCase.expectedOutcome === 'rejected' &&
          expectedPattern !== undefined &&
          new RegExp(expectedPattern, 'i').test(message);
        observations.push({
          case: testCase,
          model,
          outcome: rejected ? 'rejected' : 'error',
          actualClauses: store.load('default').map(serializeClause),
          added: [],
          duplicates: 0,
          retracted: 0,
          llmCalls,
          usage,
          durationMs: performance.now() - started,
          error: message,
        });
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }
  return observations;
}

async function main(): Promise<void> {
  loadEnv();
  const args = parseArgs(process.argv.slice(2));
  if (args.caseIds !== null) {
    const known = new Set(EXTRACTION_EVAL_CASES.map((testCase) => testCase.id));
    const unknown = [...args.caseIds].filter((id) => !known.has(id));
    if (unknown.length > 0) throw new Error(`unknown case ID: ${unknown.join(', ')}`);
  }
  const apiKey = process.env.LLM_API_KEY;
  if (!apiKey) throw new Error('LLM_API_KEY is not set — add it to .env or the environment');
  const baseUrl = (process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');
  const runs: {
    model: string;
    score: ReturnType<typeof scoreExtractionEval>;
    observations: ExtractionEvalObservation[];
  }[] = [];

  for (const model of args.models) {
    if (!args.json) console.error(`Evaluating ${model}...`);
    const observations = await runConfiguration(
      model,
      args.caseIds,
      apiKey,
      baseUrl
    );
    runs.push({ model, score: scoreExtractionEval(observations), observations });
  }

  if (args.json) {
    const text = JSON.stringify({ generatedAt: new Date().toISOString(), runs }, null, 2);
    if (args.output === undefined) console.log(text);
    else {
      writeJsonOutput(args.output, text);
      console.error(`Wrote extraction evaluation to ${resolve(args.output)}`);
    }
    if (runs.some((run) => run.score.unexpectedErrors > 0)) process.exitCode = 1;
    return;
  }

  console.log('\nmodel | cases | accuracy | mutation precision | mutation recall | mutation F1 | safety | errors | seconds | input tokens | output tokens | cost USD');
  console.log('--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:');
  for (const run of runs) {
    const cost =
      run.score.costResponses === run.score.llmCalls
        ? run.score.costUsd.toFixed(6)
        : `${run.score.costUsd.toFixed(6)} partial`;
    console.log(
      `${run.model} | ${run.score.cases} | ${percent(run.score.accuracy)} | ${percent(run.score.mutationPrecision)} | ${percent(run.score.mutationRecall)} | ${percent(run.score.mutationF1)} | ${percent(run.score.safetyAccuracy)} | ${run.score.unexpectedErrors} | ${(run.score.durationMs / 1000).toFixed(1)} | ${run.score.promptTokens} | ${run.score.completionTokens} | ${cost}`
    );
  }

  for (const run of runs) {
    const failures = run.observations.filter(
      (observation) => !extractionObservationIsCorrect(observation)
    );
    if (failures.length === 0) continue;
    console.log(`\nFailures for ${run.model}:`);
    for (const failure of failures) {
      console.log(
        `- ${failure.case.id}: outcome=${failure.outcome}; expected=${failure.case.expectedOutcome}; added=${failure.added.join(' ') || '(none)'}; final=${failure.actualClauses.join(' ') || '(empty)'}${failure.error ? `; error=${failure.error}` : ''}`
      );
    }
  }
  if (runs.some((run) => run.score.unexpectedErrors > 0)) process.exitCode = 1;
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
