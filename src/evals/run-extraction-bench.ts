#!/usr/bin/env node
/**
 * Run the extraction benchmark against one or more chat models.
 *
 *   npm run eval:extract-bench -- --models openai/gpt-5.6-luna
 *   npm run eval:extract-bench -- --base-url http://127.0.0.1:11434/v1 --api-key ollama --models llama3.2:3b
 *   npm run eval:extract-bench -- --phenomena negation,supersession --output results.json
 *
 * Each case runs against a fresh store seeded with its initial program, through
 * the same rememberText / rememberTranscriptText pipeline the product uses, so
 * the prompt, validator and integrity gate are all in the loop.
 */
import {
  existsSync,
  lstatSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
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
import { rememberText, rememberTranscriptText } from '../llm/pipeline.js';
import { MemoryStore } from '../store/store.js';
import {
  EXTRACTION_BENCH_CASES,
  EXTRACTION_BENCH_PHENOMENA,
  predicateDrift,
  scoreExtractionBench,
  type ExtractionBenchCase,
  type ExtractionPhenomenon,
} from './extraction-bench.js';
import {
  extractionObservationIsCorrect,
  type ExtractionEvalObservation,
} from './extraction.js';

interface Args {
  models: string[];
  phenomena: Set<ExtractionPhenomenon> | null;
  caseIds: Set<string> | null;
  json: boolean;
  output: string | undefined;
  baseUrl: string | undefined;
  apiKey: string | undefined;
}

const USAGE = `Usage: npm run eval:extract-bench -- [options]
Options:
  --models <a,b>        Model IDs (default: LLM_MODEL or ${DEFAULT_MODEL})
  --phenomena <a,b>     Only these phenomena (${EXTRACTION_BENCH_PHENOMENA.join(', ')})
  --cases <a,b>         Only these case IDs
  --base-url <url>      OpenAI-compatible base URL (default: LLM_BASE_URL or OpenRouter)
  --api-key <key>       Bearer token (default: LLM_API_KEY); any value works for local Ollama
  --json                Print machine-readable JSON
  --output <path>       Write machine-readable JSON to a regular file
`;

function list(argv: string[], index: number, flag: string): string[] {
  const value = argv[index + 1];
  if (!value) throw new Error(`${flag} needs a comma-separated value`);
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    models: [process.env.LLM_MODEL ?? DEFAULT_MODEL],
    phenomena: null,
    caseIds: null,
    json: false,
    output: undefined,
    baseUrl: undefined,
    apiKey: undefined,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--models') {
      args.models = list(argv, i, arg);
      i += 1;
    } else if (arg === '--phenomena') {
      const values = list(argv, i, arg);
      for (const value of values) {
        if (
          !(EXTRACTION_BENCH_PHENOMENA as readonly string[]).includes(value)
        ) {
          throw new Error(`unknown phenomenon '${value}'`);
        }
      }
      args.phenomena = new Set(values as ExtractionPhenomenon[]);
      i += 1;
    } else if (arg === '--cases') {
      args.caseIds = new Set(list(argv, i, arg));
      i += 1;
    } else if (arg === '--base-url') {
      args.baseUrl = argv[i + 1];
      i += 1;
    } else if (arg === '--api-key') {
      args.apiKey = argv[i + 1];
      i += 1;
    } else if (arg === '--json') {
      args.json = true;
    } else if (arg === '--output') {
      args.output = argv[i + 1];
      args.json = true;
      i += 1;
    } else if (arg === '--help' || arg === '-h') {
      console.log(USAGE);
      process.exit(0);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return args;
}

function selectedCases(args: Args): ExtractionBenchCase[] {
  return EXTRACTION_BENCH_CASES.filter(
    (c) =>
      (args.phenomena === null || args.phenomena.has(c.phenomenon)) &&
      (args.caseIds === null || args.caseIds.has(c.id)),
  );
}

async function runModel(
  model: string,
  cases: ExtractionBenchCase[],
  apiKey: string,
  baseUrl: string,
): Promise<ExtractionEvalObservation[]> {
  const client = new OpenRouterClient({ apiKey, baseUrl, model });
  const observations: ExtractionEvalObservation[] = [];
  for (const testCase of cases) {
    const root = mkdtempSync(join(tmpdir(), 'rembero-extraction-bench-'));
    const store = new MemoryStore(root);
    if (testCase.initialProgram.trim() !== '') {
      store.importClauses('default', testCase.initialProgram);
    }
    let llmCalls = 0;
    let usage = emptyLlmUsageTotals();
    const llm: LlmClient = {
      async complete(messages) {
        llmCalls += 1;
        const completion = await client.completeWithUsage(messages);
        usage = addLlmUsage(usage, completion.usage);
        return completion.content;
      },
    };
    const started = performance.now();
    try {
      const result =
        testCase.mode === 'transcript'
          ? await rememberTranscriptText(
              { store, llm },
              testCase.input,
              'default',
              {
                captureId: `bench-${testCase.id}`,
              },
            )
          : await rememberText({ store, llm }, testCase.input, 'default', {});
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
      observations.push({
        case: testCase,
        model,
        outcome: 'error',
        actualClauses: store.load('default').map(serializeClause),
        added: [],
        duplicates: 0,
        retracted: 0,
        llmCalls,
        usage,
        durationMs: performance.now() - started,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
    const last = observations.at(-1)!;
    const ok = extractionObservationIsCorrect(last);
    const drift = predicateDrift(testCase, last.actualClauses);
    console.error(
      `${ok ? 'PASS' : 'FAIL'} ${model} ${testCase.id}` +
        (drift.length > 0 ? ` (drift: ${drift.join(', ')})` : '') +
        (last.error ? ` (error: ${last.error.slice(0, 80)})` : ''),
    );
  }
  return observations;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function main(): Promise<void> {
  loadEnv();
  const args = parseArgs(process.argv.slice(2));
  const apiKey = args.apiKey ?? process.env.LLM_API_KEY;
  if (!apiKey) throw new Error('LLM_API_KEY (or --api-key) is required');
  const baseUrl = (
    args.baseUrl ??
    process.env.LLM_BASE_URL ??
    'https://openrouter.ai/api/v1'
  ).replace(/\/$/, '');
  const cases = selectedCases(args);
  const runs = [];
  for (const model of args.models) {
    const observations = await runModel(model, cases, apiKey, baseUrl);
    const score = scoreExtractionBench(observations);
    runs.push({
      model,
      score,
      failures: observations
        .filter((o) => !extractionObservationIsCorrect(o))
        .map((o) => ({
          id: o.case.id,
          phenomenon: (o.case as ExtractionBenchCase).phenomenon,
          actual: o.actualClauses,
          expected: o.case.expectedFinalProgram,
          drift: predicateDrift(o.case as ExtractionBenchCase, o.actualClauses),
          ...(o.error ? { error: o.error } : {}),
        })),
    });
    if (!args.json) {
      console.log(
        `\n${model}: ${percent(score.accuracy)} accurate (${score.cases} cases), F1 ${score.mutationF1.toFixed(3)}, drift ${percent(score.driftRate)}, $${score.costUsd.toFixed(4)}`,
      );
      for (const [phenomenon, bucket] of Object.entries(score.byPhenomenon)) {
        console.log(
          `  ${phenomenon.padEnd(22)} ${bucket.correct}/${bucket.cases}`,
        );
      }
    }
  }
  const payload = {
    benchmark: 'extraction-bench-v1',
    generatedAt: new Date().toISOString(),
    cases: cases.length,
    runs,
  };
  if (args.json) {
    const text = JSON.stringify(payload, null, 2);
    if (args.output) {
      const absolute = resolve(args.output);
      if (existsSync(absolute)) {
        const stat = lstatSync(absolute);
        if (stat.isSymbolicLink() || !stat.isFile())
          throw new Error('refusing non-regular output file');
      }
      writeFileSync(absolute, `${text}\n`, 'utf8');
      console.error(`written ${absolute}`);
    } else {
      console.log(text);
    }
  }
}

await main();
