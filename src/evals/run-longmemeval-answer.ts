#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { loadEnv } from '../env.js';
import { DEFAULT_MODEL, OpenRouterClient } from '../llm/client.js';
import { embeddingClientFromEnv } from '../llm/embeddings.js';
import { stringifyBoundedResult } from '../safety.js';
import {
  DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
  DEFAULT_LONGMEMEVAL_ANSWER_TOP_K,
  DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K,
  DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K,
  LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE,
  DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES,
  MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
  evaluateLongMemEvalAnswerInstance,
  longMemEvalAnswerRun,
  type LongMemEvalAnswerObservation,
  type LongMemEvalFormation,
} from './longmemeval-answer.js';
import { loadLongMemEvalS } from './longmemeval.js';
import {
  longMemEvalSplit,
  type LongMemEvalSplit,
} from './longmemeval-semantic.js';

interface Args {
  data: string;
  split: LongMemEvalSplit | 'all';
  limit: number | undefined;
  offset: number;
  topK: number;
  multiSessionTopK: number;
  temporalTopK: number;
  contextBytes: number;
  concurrency: number;
  readerModel: string;
  judgeModel: string;
  output: string | undefined;
  hypotheses: string | undefined;
  questionTypes: Set<string> | undefined;
  semanticQuestionTypes: Set<string>;
  multiSessionSemanticMaximumLexicalScore: number;
  prepareSemantic: boolean;
  caseIds: Set<string> | undefined;
  json: boolean;
  formation: LongMemEvalFormation;
  extractionModel: string | undefined;
  extractionBaseUrl: string | undefined;
  extractionApiKey: string | undefined;
  extractionCharacters: number | undefined;
  extractionAssistantCharacters: number | undefined;
  extractionMaxTokens: number | undefined;
  factsInContext: boolean;
  hybridRetrieval: 'shared' | 'reserved' | 'keyed';
  retrievalUnit: 'session' | 'turn';
  hybridQuestionTypes: Set<string> | undefined;
  reservedMinimumScore: number | undefined;
  entityRetrieval: boolean;
  engineRecall: boolean;
  engineRecallQuestionTypes: Set<string> | undefined;
  extractionCacheDir: string | undefined;
  temporalRangeModel: string | undefined;
  temporalRangeBaseUrl: string | undefined;
  temporalRangeApiKey: string | undefined;
  readingStrategy: 'direct' | 'notes' | 'two-call';
  aggregationReaderModel: string | undefined;
  aggregationReaderBaseUrl: string | undefined;
  aggregationReaderApiKey: string | undefined;
  readerMaxTokens: number | undefined;
  readerBaseUrl: string | undefined;
  readerApiKey: string | undefined;
}

const USAGE = `Usage: npm run bench:longmemeval:answer -- [options]

Options:
  --data <path>          Dataset path (default: .cache/longmemeval/...)
  --split <dev|test|all> Deterministic selection (default: dev)
  --limit <count>        Run the first 1-500 selected questions
  --offset <count>       Skip 0-499 selected questions for resumable slices
  --formation <mode>     raw (default): one placeholder fact per session over the raw text;
                         extracted: only what the product's transcript extraction writes;
                         hybrid: both. extracted/hybrid make one extraction call per session
  --extraction-model <id>      Model for the extraction (default: the reader model)
  --extraction-base-url <url>  OpenAI-compatible endpoint for it (default: LLM_BASE_URL)
  --extraction-api-key <key>   Key for it (default: EXTRACTION_API_KEY, else MODAL_SERVE_API_KEY,
                         else LLM_API_KEY; prefer the environment, a flag shows in process lists)
  --extraction-characters <n>  Cut each session to n characters before extraction (default 16000)
  --hybrid-retrieval <shared|reserved|keyed>  hybrid only. shared (default): raw text and
                         extracted facts compete for the top-k session slots; reserved: raw text
                         fills top-k and matched facts are appended as a dated block; keyed: the
                         session's facts are prepended to its own key (fact-augmented keys)
  --retrieval-unit <session|turn>  session (default) or one document per user turn, scored
                         separately and aggregated to sessions; whole sessions still come back
  --hybrid-question-types <csv>  Use the extractor only for these question types; others run raw
                         (and make no extraction calls)
  --reserved-min-score <n>  reserved only: minimum lexical score for an appended fact (default 1)
  --reader-base-url <url>  OpenAI-compatible endpoint for the reader (default: LLM_BASE_URL)
  --reader-api-key <key>   Key for it (default: READER_API_KEY, else LLM_API_KEY)
  --reader-max-tokens <n>  Completion budget per reader call (default 4096; reasoning models
                         such as DeepSeek v4.1 Flash exhaust it thinking and return nothing)
  --aggregation-reader-model <id>  A separate reader for multi-session, temporal and
                         knowledge-update questions; other types keep --reader-model
  --aggregation-reader-base-url <url>  OpenAI-compatible endpoint for that reader (default:
                         LLM_BASE_URL), e.g. http://127.0.0.1:11434/v1 for Ollama Cloud models
  --aggregation-reader-api-key <key>  Key for it (default: AGGREGATION_READER_API_KEY, else LLM_API_KEY)
  --reading <direct|notes|two-call>  for multi-session, temporal and knowledge-update questions:
                         notes = dated items first, then an "Answer:" line that alone is judged;
                         two-call = one call enumerates dated items from the history, a second
                         answers from that list alone
  --temporal-range-model <id>  Time-aware retrieval: this model reads the date range a
                         temporal question refers to (or refuses); in-range sessions rank first
  --temporal-range-base-url <url>  Endpoint for that model (default: the main endpoint)
  --temporal-range-api-key <key>  Key for it (default: TEMPORAL_RANGE_API_KEY, else LLM_API_KEY)
  --extraction-cache <dir>  Replay per-session extractions from this directory when present
                         (keyed by extractor model and transcript), else call and record
  --entity-retrieval     hybrid/extracted: one hop over the extracted facts from the question
                         (shared relation-and-subject or entity); found sessions take alternate
                         top-k slots with the lexical ranking
  --engine-recall        hybrid/extracted: the memory system's own recall authors a Datalog
                         query over the remembered facts (the extraction model writes it),
                         executes it, and the reader sees the query and rows ahead of the chats
  --engine-recall-question-types <csv>  Restrict engine recall to these question types
  --no-facts-in-context  Do not list a retrieved session's matched extracted facts to the reader
  --extraction-max-tokens <n>  Completion budget per extraction call (default 512; reasoning
                         models such as Luna spend it on thinking and need 4096 or more)
  --extraction-assistant-characters <n>  Keep only the first n characters of each assistant
                         turn in the extraction input (user turns stay whole; default: no cut)
  --top-k <count>        Retrieved sessions per question (default: 4)
  --multi-session-top-k <count>  Retrieved sessions for multi-session questions (default: 5)
  --temporal-top-k <count>  Retrieved sessions for temporal questions (default: 5)
  --context-bytes <n>    Answer-facing context budget (default: 57344)
  --concurrency <n>      Concurrent questions from 1-8 (default: 4)
  --reader-model <id>    Answer model (default: LLM_MODEL or ${DEFAULT_MODEL})
  --judge-model <id>     Judge model (default: openai/gpt-4o-2024-08-06)
  --output <path>        Write the complete JSON run artifact
  --hypotheses <path>    Write official two-field hypothesis JSONL
  --question-types <csv> Run only the named question types
  --cases <csv>          Run only the named question IDs
  --semantic-question-types <csv>  Question types eligible for semantic reranking
  --multi-semantic-max-score <n>  Multi-session local-score ceiling for semantic routing
  --prepare-semantic     Prepare document embeddings before measuring the user turn
  --local-only           Keep every question on local lexical retrieval
  --no-semantic-preferences  Compatibility alias for --local-only
  --json                 Print the complete run instead of its summary
`;

function requiredValue(argv: string[], index: number, flag: string): string {
  const value = argv[index + 1];
  if (value === undefined || value.trim() === '')
    throw new Error(`${flag} needs a value`);
  return value;
}

function boundedInteger(
  value: string,
  flag: string,
  minimum: number,
  maximum: number,
): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${flag} needs an integer from ${minimum} to ${maximum}`);
  }
  return parsed;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    data: resolve('.cache/longmemeval/longmemeval_s_cleaned.json'),
    split: 'dev',
    limit: undefined,
    offset: 0,
    topK: DEFAULT_LONGMEMEVAL_ANSWER_TOP_K,
    multiSessionTopK: DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K,
    temporalTopK: DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K,
    contextBytes: DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
    concurrency: 4,
    readerModel: process.env.LLM_MODEL ?? DEFAULT_MODEL,
    judgeModel:
      process.env.LONGMEMEVAL_JUDGE_MODEL ?? 'openai/gpt-4o-2024-08-06',
    output: undefined,
    hypotheses: undefined,
    questionTypes: undefined,
    semanticQuestionTypes: new Set(DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES),
    multiSessionSemanticMaximumLexicalScore:
      LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE,
    prepareSemantic: false,
    caseIds: undefined,
    json: false,
    formation: 'raw',
    extractionModel: undefined,
    extractionBaseUrl: undefined,
    extractionApiKey: undefined,
    extractionCharacters: undefined,
    extractionAssistantCharacters: undefined,
    extractionMaxTokens: undefined,
    factsInContext: true,
    hybridRetrieval: 'shared',
    retrievalUnit: 'session',
    hybridQuestionTypes: undefined,
    reservedMinimumScore: undefined,
    entityRetrieval: false,
    engineRecall: false,
    engineRecallQuestionTypes: undefined,
    extractionCacheDir: undefined,
    temporalRangeModel: undefined,
    temporalRangeBaseUrl: undefined,
    temporalRangeApiKey: undefined,
    readingStrategy: 'direct',
    aggregationReaderModel: undefined,
    aggregationReaderBaseUrl: undefined,
    aggregationReaderApiKey: undefined,
    readerMaxTokens: undefined,
    readerBaseUrl: undefined,
    readerApiKey: undefined,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--data')
      args.data = resolve(requiredValue(argv, index++, arg));
    else if (arg === '--split') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'dev' && value !== 'test' && value !== 'all') {
        throw new Error('--split needs dev, test, or all');
      }
      args.split = value;
    } else if (arg === '--limit') {
      args.limit = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        500,
      );
    } else if (arg === '--offset') {
      args.offset = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        0,
        499,
      );
    } else if (arg === '--top-k') {
      args.topK = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        100,
      );
    } else if (arg === '--multi-session-top-k') {
      args.multiSessionTopK = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        100,
      );
    } else if (arg === '--temporal-top-k') {
      args.temporalTopK = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        100,
      );
    } else if (arg === '--context-bytes') {
      args.contextBytes = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        4_096,
        MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
      );
    } else if (arg === '--concurrency') {
      args.concurrency = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        8,
      );
    } else if (arg === '--reader-model') {
      args.readerModel = requiredValue(argv, index++, arg);
    } else if (arg === '--judge-model') {
      args.judgeModel = requiredValue(argv, index++, arg);
    } else if (arg === '--output') {
      args.output = resolve(requiredValue(argv, index++, arg));
    } else if (arg === '--hypotheses') {
      args.hypotheses = resolve(requiredValue(argv, index++, arg));
    } else if (arg === '--question-types') {
      const values = requiredValue(argv, index++, arg)
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
      if (values.length === 0)
        throw new Error('--question-types needs at least one value');
      args.questionTypes = new Set(values);
    } else if (arg === '--cases') {
      const values = requiredValue(argv, index++, arg)
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
      if (values.length === 0)
        throw new Error('--cases needs at least one value');
      args.caseIds = new Set(values);
    } else if (arg === '--semantic-question-types') {
      const values = requiredValue(argv, index++, arg)
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
      if (values.length === 0) {
        throw new Error('--semantic-question-types needs at least one value');
      }
      args.semanticQuestionTypes = new Set(values);
    } else if (arg === '--multi-semantic-max-score') {
      args.multiSessionSemanticMaximumLexicalScore = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        0,
        10_000,
      );
    } else if (arg === '--prepare-semantic') {
      args.prepareSemantic = true;
    } else if (arg === '--formation') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'raw' && value !== 'extracted' && value !== 'hybrid') {
        throw new Error('--formation must be raw, extracted or hybrid');
      }
      args.formation = value;
    } else if (arg === '--extraction-model') {
      args.extractionModel = requiredValue(argv, index++, arg);
    } else if (arg === '--extraction-base-url') {
      args.extractionBaseUrl = requiredValue(argv, index++, arg).replace(
        /\/$/,
        '',
      );
    } else if (arg === '--extraction-api-key') {
      args.extractionApiKey = requiredValue(argv, index++, arg);
    } else if (arg === '--hybrid-retrieval') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'shared' && value !== 'reserved' && value !== 'keyed') {
        throw new Error('--hybrid-retrieval must be shared, reserved or keyed');
      }
      args.hybridRetrieval = value;
    } else if (arg === '--hybrid-question-types') {
      args.hybridQuestionTypes = new Set(
        requiredValue(argv, index++, arg)
          .split(',')
          .map((v) => v.trim())
          .filter(Boolean),
      );
    } else if (arg === '--reserved-min-score') {
      args.reservedMinimumScore = Number(requiredValue(argv, index++, arg));
    } else if (arg === '--retrieval-unit') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'session' && value !== 'turn') {
        throw new Error('--retrieval-unit must be session or turn');
      }
      args.retrievalUnit = value;
    } else if (arg === '--reader-base-url') {
      args.readerBaseUrl = requiredValue(argv, index++, arg).replace(/\/$/, '');
    } else if (arg === '--reader-api-key') {
      args.readerApiKey = requiredValue(argv, index++, arg);
    } else if (arg === '--reader-max-tokens') {
      args.readerMaxTokens = Number(requiredValue(argv, index++, arg));
    } else if (arg === '--aggregation-reader-model') {
      args.aggregationReaderModel = requiredValue(argv, index++, arg);
    } else if (arg === '--aggregation-reader-base-url') {
      args.aggregationReaderBaseUrl = requiredValue(argv, index++, arg).replace(
        /\/$/,
        '',
      );
    } else if (arg === '--aggregation-reader-api-key') {
      args.aggregationReaderApiKey = requiredValue(argv, index++, arg);
    } else if (arg === '--reading') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'direct' && value !== 'notes' && value !== 'two-call') {
        throw new Error('--reading must be direct, notes or two-call');
      }
      args.readingStrategy = value;
    } else if (arg === '--temporal-range-model') {
      args.temporalRangeModel = requiredValue(argv, index++, arg);
    } else if (arg === '--temporal-range-base-url') {
      args.temporalRangeBaseUrl = requiredValue(argv, index++, arg).replace(
        /\/$/,
        '',
      );
    } else if (arg === '--temporal-range-api-key') {
      args.temporalRangeApiKey = requiredValue(argv, index++, arg);
    } else if (arg === '--extraction-cache') {
      args.extractionCacheDir = resolve(requiredValue(argv, index++, arg));
    } else if (arg === '--entity-retrieval') {
      args.entityRetrieval = true;
    } else if (arg === '--engine-recall') {
      args.engineRecall = true;
    } else if (arg === '--engine-recall-question-types') {
      args.engineRecallQuestionTypes = new Set(
        requiredValue(argv, index++, arg)
          .split(',')
          .map((v) => v.trim())
          .filter(Boolean),
      );
    } else if (arg === '--no-facts-in-context') {
      args.factsInContext = false;
    } else if (arg === '--extraction-max-tokens') {
      args.extractionMaxTokens = Number(requiredValue(argv, index++, arg));
    } else if (arg === '--extraction-assistant-characters') {
      args.extractionAssistantCharacters = Number(
        requiredValue(argv, index++, arg),
      );
    } else if (arg === '--extraction-characters') {
      args.extractionCharacters = Number(requiredValue(argv, index++, arg));
    } else if (arg === '--local-only' || arg === '--no-semantic-preferences') {
      args.semanticQuestionTypes.clear();
    } else if (arg === '--json') args.json = true;
    else if (arg === '--help' || arg === '-h') {
      console.log(USAGE);
      process.exit(0);
    } else throw new Error(`unknown option: ${arg}`);
  }
  return args;
}

async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  operation: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(concurrency, values.length) }, async () => {
      while (true) {
        const index = next++;
        if (index >= values.length) return;
        results[index] = await operation(values[index]!, index);
      }
    }),
  );
  return results;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function warmUp(
  client: OpenRouterClient,
  label: string,
  attempts = 6,
): Promise<void> {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      await client.completeWithUsage([{ role: 'user', content: 'Say ok.' }], {
        maxTokens: 4,
      });
      return;
    } catch (error) {
      if (attempt === attempts) throw error;
      console.error(
        `${label} not ready (attempt ${attempt}/${attempts}): ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
}

async function main(): Promise<void> {
  loadEnv();
  const args = parseArgs(process.argv.slice(2));
  const apiKey = process.env.LLM_API_KEY;
  if (!apiKey)
    throw new Error(
      'LLM_API_KEY is not set — add it to .env or the environment',
    );
  const baseUrl = (
    process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1'
  ).replace(/\/$/, '');
  const loaded = await loadLongMemEvalS(args.data);
  const selected = loaded.instances.filter(
    (instance) =>
      (args.split === 'all'
        ? true
        : longMemEvalSplit(instance.question_id) === args.split) &&
      (args.questionTypes === undefined ||
        args.questionTypes.has(instance.question_type)) &&
      (args.caseIds === undefined || args.caseIds.has(instance.question_id)),
  );
  if (args.caseIds !== undefined && selected.length !== args.caseIds.size) {
    const found = new Set(selected.map(({ question_id: id }) => id));
    const missing = [...args.caseIds].filter((id) => !found.has(id));
    throw new Error(`unknown or out-of-split case ID: ${missing.join(', ')}`);
  }
  const available = selected.slice(args.offset);
  const instances =
    args.limit === undefined ? available : available.slice(0, args.limit);
  const reader = new OpenRouterClient({
    // the reader may live on another endpoint (e.g. Ollama Cloud) while the judge stays put
    apiKey: args.readerApiKey ?? process.env.READER_API_KEY ?? apiKey,
    baseUrl: args.readerBaseUrl ?? baseUrl,
    model: args.readerModel,
  });
  const aggregationReader =
    args.aggregationReaderModel === undefined
      ? undefined
      : new OpenRouterClient({
          // e.g. a local Ollama daemon signed in to Ollama Cloud (http://127.0.0.1:11434/v1)
          apiKey:
            args.aggregationReaderApiKey ??
            process.env.AGGREGATION_READER_API_KEY ??
            apiKey,
          baseUrl: args.aggregationReaderBaseUrl ?? baseUrl,
          model: args.aggregationReaderModel,
        });
  const temporalRangeExtractor =
    args.temporalRangeModel === undefined
      ? undefined
      : new OpenRouterClient({
          apiKey:
            args.temporalRangeApiKey ??
            process.env.TEMPORAL_RANGE_API_KEY ??
            apiKey,
          baseUrl: args.temporalRangeBaseUrl ?? baseUrl,
          model: args.temporalRangeModel,
        });
  const judge = new OpenRouterClient({
    apiKey,
    baseUrl,
    model: args.judgeModel,
  });
  // extracted/hybrid formations run the product's transcript extraction with its own
  // model, typically the fine-tuned dialect model on a self-hosted OpenAI-compatible endpoint
  const extractor =
    args.formation === 'raw'
      ? undefined
      : new OpenRouterClient({
          apiKey:
            // prefer environment variables: a key on the command line shows up in process lists
            args.extractionApiKey ??
            process.env.EXTRACTION_API_KEY ??
            process.env.MODAL_SERVE_API_KEY ??
            apiKey,
          baseUrl: args.extractionBaseUrl ?? baseUrl,
          model: args.extractionModel ?? args.readerModel,
        });
  if (args.engineRecall && extractor === undefined) {
    throw new Error('--engine-recall needs the extracted or hybrid formation');
  }
  // the writer that authors the engine's query is the extraction model
  const engineWriter = extractor as OpenRouterClient;
  const embeddings =
    args.semanticQuestionTypes.size > 0 ? embeddingClientFromEnv() : undefined;
  // A self-hosted extraction endpoint that scaled to zero takes longer to come up
  // than one request's timeout; wake it before the questions start.
  if (extractor !== undefined) await warmUp(extractor, 'extraction endpoint');
  let completed = 0;
  const observations = await mapConcurrent(
    instances,
    args.concurrency,
    async (instance) => {
      const observation = await evaluateLongMemEvalAnswerInstance(
        instance,
        reader,
        judge,
        {
          topK: args.topK,
          multiSessionTopK: args.multiSessionTopK,
          temporalTopK: args.temporalTopK,
          contextBytes: args.contextBytes,
          ...(embeddings === undefined ? {} : { embeddings }),
          semanticQuestionTypes: args.semanticQuestionTypes,
          multiSessionSemanticMaximumLexicalScore:
            args.multiSessionSemanticMaximumLexicalScore,
          prepareSemantic: args.prepareSemantic,
          formation: args.formation,
          ...(extractor === undefined ? {} : { extractor }),
          ...(args.extractionCharacters === undefined
            ? {}
            : { extractionCharacters: args.extractionCharacters }),
          ...(args.extractionAssistantCharacters === undefined
            ? {}
            : {
                extractionAssistantCharacters:
                  args.extractionAssistantCharacters,
              }),
          ...(args.extractionMaxTokens === undefined
            ? {}
            : { extractionMaxTokens: args.extractionMaxTokens }),
          factsInContext: args.factsInContext,
          hybridRetrieval: args.hybridRetrieval,
          retrievalUnit: args.retrievalUnit,
          readingStrategy: args.readingStrategy,
          ...(args.readerMaxTokens === undefined
            ? {}
            : { readerMaxTokens: args.readerMaxTokens }),
          ...(aggregationReader === undefined ? {} : { aggregationReader }),
          ...(temporalRangeExtractor === undefined
            ? {}
            : { temporalRangeExtractor }),
          entityRetrieval: args.entityRetrieval,
          ...(args.engineRecall
            ? {
                engineRecall: {
                  llm: engineWriter,
                  ...(args.engineRecallQuestionTypes === undefined
                    ? {}
                    : { questionTypes: args.engineRecallQuestionTypes }),
                },
              }
            : {}),
          ...(args.extractionCacheDir === undefined
            ? {}
            : { extractionCacheDir: args.extractionCacheDir }),
          ...(args.hybridQuestionTypes === undefined
            ? {}
            : { hybridQuestionTypes: args.hybridQuestionTypes }),
          ...(args.reservedMinimumScore === undefined
            ? {}
            : { reservedMinimumScore: args.reservedMinimumScore }),
        },
      );
      completed++;
      if (!args.json) {
        console.error(
          `[${completed}/${instances.length}] ${instance.question_id}: ${observation.status === 'judged' ? (observation.correct ? 'correct' : 'incorrect') : `error (${observation.error})`}`,
        );
      }
      return observation;
    },
  );
  const run = longMemEvalAnswerRun(observations, reader.model, judge.model, {
    topK: args.topK,
    multiSessionTopK: args.multiSessionTopK,
    temporalTopK: args.temporalTopK,
    contextBytes: args.contextBytes,
    embeddingModel: embeddings?.model ?? null,
    selection: args.split,
    sha256: loaded.sha256,
    semanticQuestionTypes: args.semanticQuestionTypes,
    multiSessionSemanticMaximumLexicalScore:
      args.multiSessionSemanticMaximumLexicalScore,
    prepareSemantic: args.prepareSemantic,
    formation: args.formation,
    extractionModel: extractor?.model ?? null,
    settings: {
      aggregationReaderModel: aggregationReader?.model ?? null,
      temporalRangeModel: temporalRangeExtractor?.model ?? null,
      readingStrategy: args.readingStrategy,
      hybridRetrieval: args.hybridRetrieval,
      retrievalUnit: args.retrievalUnit,
      entityRetrieval: args.entityRetrieval,
      engineRecall: args.engineRecall,
      engineRecallQuestionTypes:
        args.engineRecallQuestionTypes === undefined
          ? null
          : [...args.engineRecallQuestionTypes],
      hybridQuestionTypes:
        args.hybridQuestionTypes === undefined
          ? null
          : [...args.hybridQuestionTypes],
      factsInContext: args.factsInContext,
      readerMaxTokens: args.readerMaxTokens ?? 4096,
    },
  });
  const serialized = stringifyBoundedResult(run, 'LongMemEval answer run');
  if (args.output !== undefined) {
    mkdirSync(dirname(args.output), { recursive: true });
    writeFileSync(args.output, `${serialized}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    });
  }
  if (args.hypotheses !== undefined) {
    mkdirSync(dirname(args.hypotheses), { recursive: true });
    const lines = observations.flatMap((observation) =>
      observation.hypothesis === null
        ? []
        : [
            JSON.stringify({
              question_id: observation.questionId,
              hypothesis: observation.hypothesis,
            }),
          ],
    );
    writeFileSync(args.hypotheses, `${lines.join('\n')}\n`, {
      encoding: 'utf8',
      mode: 0o600,
    });
  }
  if (args.json) console.log(serialized);
  else {
    const { summary } = run;
    console.log(
      'LongMemEval-S Remembero durable formation + retrieval + answer',
    );
    console.log(`selection: ${args.split} (${summary.questions} questions)`);
    console.log(`reader / judge: ${reader.model} / ${judge.model}`);
    console.log(
      `formation: ${run.formation}${extractor === undefined ? '' : ` (extractor ${extractor.model})`}`,
    );
    if (args.formation !== 'raw') {
      const { extractionUsage: e } = summary;
      console.log(
        `extraction: ${e.calls} calls, ${e.facts} facts from ${e.sessionsWithFacts} sessions, ${e.errors} errors, ${e.totalTokens} tokens`,
      );
      for (const [kind, count] of Object.entries(e.errorKinds)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)) {
        console.log(`  error x${count}: ${kind}`);
      }
    }
    console.log(
      `accuracy: ${percent(summary.accuracy)} (${summary.correct}/${summary.questions})`,
    );
    console.log(`errors: ${summary.errors}`);
    console.log(
      `retrieval/context recall: ${percent(summary.retrievalRecallAtK)} / ${percent(summary.contextRecallAtK)} (top-k ${args.topK}; multi-session ${args.multiSessionTopK}; temporal ${args.temporalTopK})`,
    );
    console.log(
      `full/incomplete evidence accuracy: ${percent(summary.fullContextEvidenceAccuracy)} / ${percent(summary.incompleteContextEvidenceAccuracy)}`,
    );
    console.log(
      `formation p50/p95: ${summary.medianFormationMs.toFixed(1)} / ${summary.p95FormationMs.toFixed(1)} ms`,
    );
    console.log(
      `semantic preparation p50/p95: ${summary.medianSemanticPreparationMs.toFixed(1)} / ${summary.p95SemanticPreparationMs.toFixed(1)} ms`,
    );
    console.log(
      `user turn p50/p95: ${summary.medianUserTurnMs.toFixed(1)} / ${summary.p95UserTurnMs.toFixed(1)} ms`,
    );
    console.log(
      `full lifecycle p50/p95: ${summary.medianTotalMs.toFixed(1)} / ${summary.p95TotalMs.toFixed(1)} ms`,
    );
    console.log(
      `reader calls/tokens/cost: ${summary.readerUsage.calls} / ${summary.readerUsage.totalTokens} / $${summary.readerUsage.costUsd.toFixed(6)}`,
    );
    console.log(
      `judge calls/tokens/cost: ${summary.judgeUsage.calls} / ${summary.judgeUsage.totalTokens} / $${summary.judgeUsage.costUsd.toFixed(6)}`,
    );
    console.log(
      `embedding calls/tokens/cost: ${summary.embeddingUsage.calls} / ${summary.embeddingUsage.totalTokens} / $${summary.embeddingUsage.costUsd.toFixed(6)}`,
    );
    console.log(
      `preparation calls/tokens/cost: ${summary.semanticPreparationUsage.calls} / ${summary.semanticPreparationUsage.totalTokens} / $${summary.semanticPreparationUsage.costUsd.toFixed(6)}`,
    );
  }
  if (run.summary.errors > 0) process.exitCode = 1;
}

await main();
