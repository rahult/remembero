#!/usr/bin/env node
import { existsSync, lstatSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { loadEnv } from '../env.js';
import {
  DEFAULT_MODEL,
  OpenRouterClient,
  addLlmUsage,
  emptyLlmUsageTotals,
  type LlmClient,
} from '../llm/client.js';
import { retrieveQuestion } from '../llm/pipeline.js';
import type { QueryPromptVariant } from '../llm/prompts.js';
import { MemoryStore } from '../store/store.js';
import {
  RECALL_EVAL_CASES,
  RECALL_EVAL_PROGRAM,
  bindingRows,
  observationIsCorrect,
  scoreRecallEval,
  type RecallEvalObservation,
} from './recall.js';

interface EvalArgs {
  models: string[];
  variants: QueryPromptVariant[];
  json: boolean;
  caseIds: Set<string> | null;
  schemaPredicateLimit: number | undefined;
  output: string | undefined;
}

const USAGE = `Usage: npm run eval:recall -- [options]

Options:
  --models <a,b>       OpenRouter model IDs (default: LLM_MODEL or ${DEFAULT_MODEL})
  --variants <a,b>     baseline,grounded (default: baseline,grounded)
  --cases <a,b>        Run only selected case IDs
  --schema-predicate-limit <n>  Detailed predicate budget for each recall pass
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
    variants: ['baseline', 'grounded'],
    json: false,
    caseIds: null,
    schemaPredicateLimit: undefined,
    output: undefined,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--models') {
      args.models = listValue(argv, index, arg);
      index++;
    } else if (arg === '--variants') {
      const variants = listValue(argv, index, arg);
      if (variants.some((variant) => variant !== 'baseline' && variant !== 'grounded')) {
        throw new Error(`unknown variant: ${variants.join(', ')}`);
      }
      args.variants = variants as QueryPromptVariant[];
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
    } else if (arg === '--schema-predicate-limit') {
      const value = Number(argv[index + 1]);
      if (!Number.isInteger(value) || value < 1 || value > 256) {
        throw new Error('--schema-predicate-limit needs an integer from 1 to 256');
      }
      args.schemaPredicateLimit = value;
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
      throw new Error('refusing non-regular recall evaluation output file');
    }
  }
  writeFileSync(absolute, `${text}\n`, 'utf8');
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function runConfiguration(
  model: string,
  variant: QueryPromptVariant,
  caseIds: Set<string> | null,
  apiKey: string,
  baseUrl: string,
  schemaPredicateLimit: number | undefined
): Promise<RecallEvalObservation[]> {
  const root = mkdtempSync(join(tmpdir(), 'rembero-recall-eval-'));
  try {
    const store = new MemoryStore(root);
    store.importClauses('default', RECALL_EVAL_PROGRAM);
    const client = new OpenRouterClient({ apiKey, baseUrl, model });
    const cases = RECALL_EVAL_CASES.filter((testCase) =>
      caseIds === null ? true : caseIds.has(testCase.id)
    );
    const observations: RecallEvalObservation[] = [];
    for (const testCase of cases) {
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
        const result = await retrieveQuestion(
          { store, llm },
          testCase.question,
          ['default'],
          {
            queryPromptVariant: variant,
            ...(schemaPredicateLimit === undefined
              ? {}
              : { schemaPredicateLimit }),
            ...(testCase.trustMode === undefined
              ? {}
              : { trustMode: testCase.trustMode }),
          }
        );
        observations.push({
          case: testCase,
          model,
          variant,
          status: result.status,
          query: result.query,
          actualRows:
            result.query === null ? [] : bindingRows(result.bindings, result.query),
          llmCalls,
          usage,
          durationMs: performance.now() - started,
        });
      } catch (error) {
        observations.push({
          case: testCase,
          model,
          variant,
          status: 'unanswerable',
          query: null,
          actualRows: [],
          llmCalls,
          usage,
          durationMs: performance.now() - started,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    return observations;
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  loadEnv();
  const args = parseArgs(process.argv.slice(2));
  if (args.caseIds !== null) {
    const known = new Set(RECALL_EVAL_CASES.map((testCase) => testCase.id));
    const unknown = [...args.caseIds].filter((id) => !known.has(id));
    if (unknown.length > 0) throw new Error(`unknown case ID: ${unknown.join(', ')}`);
  }
  const apiKey = process.env.LLM_API_KEY;
  if (!apiKey) throw new Error('LLM_API_KEY is not set — add it to .env or the environment');
  const baseUrl = (process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');
  const runs: { model: string; variant: QueryPromptVariant; schemaPredicateLimit: number | null; score: ReturnType<typeof scoreRecallEval>; observations: RecallEvalObservation[] }[] = [];

  for (const model of args.models) {
    for (const variant of args.variants) {
      if (!args.json) console.error(`Evaluating ${model} / ${variant}...`);
      const observations = await runConfiguration(
        model,
        variant,
        args.caseIds,
        apiKey,
        baseUrl,
        args.schemaPredicateLimit
      );
      runs.push({
        model,
        variant,
        schemaPredicateLimit: args.schemaPredicateLimit ?? null,
        score: scoreRecallEval(observations),
        observations,
      });
    }
  }

  if (args.json) {
    const text = JSON.stringify({ generatedAt: new Date().toISOString(), runs }, null, 2);
    if (args.output === undefined) console.log(text);
    else {
      writeJsonOutput(args.output, text);
      console.error(`Wrote recall evaluation to ${resolve(args.output)}`);
    }
    if (runs.some((run) => run.score.errors > 0)) process.exitCode = 1;
    return;
  }

  console.log('\nmodel | variant | schema limit | cases | accuracy | precision | recall | F1 | answerability | errors | seconds | input tokens | output tokens | cost USD');
  console.log('--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:');
  for (const run of runs) {
    const cost =
      run.score.costResponses === run.score.llmCalls
        ? run.score.costUsd.toFixed(6)
        : `${run.score.costUsd.toFixed(6)} partial`;
    console.log(
      `${run.model} | ${run.variant} | ${run.schemaPredicateLimit ?? 'default'} | ${run.score.cases} | ${percent(run.score.accuracy)} | ${percent(run.score.precision)} | ${percent(run.score.recall)} | ${percent(run.score.f1)} | ${percent(run.score.answerabilityAccuracy)} | ${run.score.errors} | ${(run.score.durationMs / 1000).toFixed(1)} | ${run.score.promptTokens} | ${run.score.completionTokens} | ${cost}`
    );
  }

  for (const run of runs) {
    const failures = run.observations.filter((observation) => !observationIsCorrect(observation));
    if (failures.length === 0) continue;
    console.log(`\nFailures for ${run.model} / ${run.variant}:`);
    for (const failure of failures) {
      const actual = failure.actualRows.map((row) => `[${row.join(', ')}]`).join(', ') || '(none)';
      const expected = failure.case.expectedRows.map((row) => `[${row.join(', ')}]`).join(', ') || '(none)';
      console.log(`- ${failure.case.id}: query=${failure.query ?? '(unanswerable)'}; expected=${expected}; actual=${actual}${failure.error ? `; error=${failure.error}` : ''}`);
    }
  }
  if (runs.some((run) => run.score.errors > 0)) process.exitCode = 1;
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
