#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadEnv } from '../env.js';
import {
  DEFAULT_LLM_TEMPERATURE,
  DEFAULT_LLM_TIMEOUT_MS,
  DEFAULT_MODEL,
  OpenRouterClient,
  checkedTemperature,
  checkedTimeoutMs,
} from '../llm/client.js';
import { embeddingClientFromEnv } from '../llm/embeddings.js';
import { stringifyBoundedResult } from '../safety.js';
import {
  DEFAULT_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
  DEFAULT_LONGMEMEVAL_ANSWER_TOP_K,
  DEFAULT_LONGMEMEVAL_MULTI_SESSION_TOP_K,
  DEFAULT_LONGMEMEVAL_TEMPORAL_TOP_K,
  LONGMEMEVAL_MULTI_SEMANTIC_MAX_LEXICAL_SCORE,
  DEFAULT_LONGMEMEVAL_SEMANTIC_QUESTION_TYPES,
  LONGMEMEVAL_QUESTION_TYPES,
  MAX_LONGMEMEVAL_ANSWER_CONTEXT_BYTES,
  evaluateLongMemEvalAnswerInstance,
  longMemEvalAnswerRun,
  type LongMemEvalAnswerObservation,
  type LongMemEvalCompletionClient,
  type LongMemEvalFormation,
} from './longmemeval-answer.js';
import { loadLongMemEvalS } from './longmemeval.js';
import {
  longMemEvalSplit,
  type LongMemEvalSplit,
} from './longmemeval-semantic.js';
import { mapConcurrent } from './map-concurrent.js';
import {
  DEFAULT_RERANK_CONCURRENCY,
  DEFAULT_RERANK_KEY_ENV,
  DEFAULT_RERANK_POOL,
  DEFAULT_RERANK_SESSION_CHARS,
  DEFAULT_TYPESAFE_MODEL,
  createLimiter,
  typesafeCostUsd,
  typesafeNouls,
  type TypesafeNouls,
} from './typesafe-rerank.js';
import {
  DEFAULT_COUNT_MAX,
  DEFAULT_COUNT_THRESHOLD,
} from './typesafe-count.js';
import { DEFAULT_ABSTRACT_BYTES, tiersFromFlags } from './reader-contract.js';
import {
  assertBuiltinMemorySystemScope,
  openMemorySystem,
} from './memory-systems-builtin.js';
import {
  summarizeMemorySystemUsage,
  type MemorySystemClient,
  type MemorySystemLane,
} from './memory-systems-protocol.js';

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
  turnUnitQuestionTypes?: Set<string>;
  turnUnitUnlessTemporal: boolean;
  hybridQuestionTypes: Set<string> | undefined;
  reservedMinimumScore: number | undefined;
  entityRetrieval: boolean;
  engineRecall: boolean;
  dateDistances: boolean;
  computedNotes: boolean;
  focusedBudget: boolean;
  structuredEvidence: boolean;
  fullSessions: number | null;
  abstractBytes: number;
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
  readerTemperature: number | undefined;
  readerTimeoutMs: number | undefined;
  judgeTemperature: number | undefined;
  readerBaseUrl: string | undefined;
  readerApiKey: string | undefined;
  memorySystem: string | undefined;
  memoryLane: MemorySystemLane;
  retrievalOnly: boolean;
  rerank: 'none' | 'typesafe';
  rerankPool: number;
  rerankSessionChars: number;
  rerankModel: string;
  rerankKeyEnv: string;
  rerankConcurrency: number;
  typesafeCount: boolean;
  typesafeCountMax: number;
  typesafeCountThreshold: number;
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
  --turn-unit-question-types <csv>  With --retrieval-unit turn: only these question types use
                         the turn unit, the rest retrieve by session (default: every type)
  --turn-unit-unless-temporal  With --retrieval-unit turn: questions whose text reads as temporal
                         (knowledge/temporal-question.ts, no label, no model) retrieve by session,
                         the rest by turn. Excludes --turn-unit-question-types
  --hybrid-question-types <csv>  Use the extractor only for these question types; others run raw
                         (and make no extraction calls)
  --reserved-min-score <n>  reserved only: minimum lexical score for an appended fact (default 1)
  --reader-base-url <url>  OpenAI-compatible endpoint for the reader (default: LLM_BASE_URL)
  --reader-api-key <key>   Key for it (default: READER_API_KEY, else LLM_API_KEY)
  --reader-max-tokens <n>  Completion budget per reader call (default 4096; reasoning models
                         such as DeepSeek v4.1 Flash exhaust it thinking and return nothing)
  --reader-temperature <n>  Sampling temperature for the reader, 0 to 2 (default 0; Moonshot's
                         Kimi K3 refuses anything but 1)
  --reader-timeout-ms <n>  Abort a reader request after n ms, 1000 to 600000 (default 60000;
                         a reasoning model on a full reader prompt can outlast the default)
  --judge-temperature <n>  Sampling temperature for the judge, 0 to 2 (default 0)
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
  --date-distances       Each session header states its distance to the question date (days,
                         weeks, months), so the reader copies the interval
  --computed-notes       After the chats, a deterministic block: temporal expressions in the user's
                         turns resolved against their session date, gaps between dated events, and
                         quantities with units totalled, each with its source sentence
  --focused-budget       Weight each retrieved session's share of the context by the question's
                         content words it contains, instead of an even split
  --full-sessions <n>    Context tiers: the n top-ranked sessions get their full text, every other
                         retrieved session a code-built abstract of its user sentences naming the
                         question (not combinable with --focused-budget)
  --abstract-bytes <n>   Byte cap of one abstract section, 120-2048 (default: 320; needs
                         --full-sessions)
  --structured-evidence  Before the chats, the extracted facts about the question, dated by their
                         session or a date inside them, grounded, deduplicated, later values current
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
  --memory-system <spec>  Replace Remembero's formation and search with another memory
                         layer: builtin:full-context, builtin:bm25, builtin:embed, or a path
                         to an adapter manifest whose protocol is rembero.memory-systems.v1.
                         One adapter process per --concurrency worker, alive for the run
  --memory-lane <retrieval|memories>  retrieval (default): the adapter ranks sessions and the
                         reader sees the same raw sessions it sees for Remembero. memories:
                         the adapter returns its own memory text and the reader answers from
                         that alone, inside the same byte budget
  --no-semantic-preferences  Compatibility alias for --local-only
  --retrieval-only       Stop after building the reader's context: retrieval (the temporal-range
                         model included, when set) and the context builder run with every
                         contract flag, the reader and the judge are never called, and no reader
                         or judge model or key is needed. Observations get status
                         "retrieval-only" and correct null
  --rerank <none|typesafe>  EXPERIMENTAL. typesafe: after lexical retrieval (and turn
                         aggregation), the top --rerank-pool sessions are each judged by the
                         TypeSafe System One API (https://docs.typesafe.ai) and re-ordered by
                         0.7 x evidence + 0.3 x relevant; the rest keep their order after them,
                         then the range ordering and top-k cut run as usual. A candidate that
                         fails the external-LLM safety check is not sent (counted as blocked).
                         Answers are cached under .cache/typesafe/. Default none
  --rerank-pool <n>      Candidate sessions to judge, 1-100 (default ${DEFAULT_RERANK_POOL})
  --rerank-session-chars <n>  Characters of turns sent per session, 500-100000 (default
                         ${DEFAULT_RERANK_SESSION_CHARS}); whole turns, best question overlap first
  --rerank-model <id>    TypeSafe model (default ${DEFAULT_TYPESAFE_MODEL})
  --rerank-key-env <name>  Environment variable holding the TypeSafe key (default
                         ${DEFAULT_RERANK_KEY_ENV}); the key is never taken from a flag
  --rerank-concurrency <n>  TypeSafe requests in flight across the run, 1-64 (default
                         ${DEFAULT_RERANK_CONCURRENCY}); the API allows 1200 requests/min
  --typesafe-count       EXPERIMENTAL. For a question whose text asks for a number of things
                         the user did or has, every user sentence of the sessions in the
                         reader's context that shares a content word with the question is sent
                         to TypeSafe on its own ("this sentence states one instance of the thing
                         the question counts"), and the ones above the threshold are counted in
                         code: Jev cannot count, so the harness does
                         (docs.typesafe.ai/model-jaggedness/jev-1.13). The tally leads the
                         computed-notes block. Uses --rerank-model, --rerank-key-env and
                         --rerank-concurrency; answers are cached under .cache/typesafe/
  --typesafe-count-max <n>  Candidate sentences per question, 1-500 (default
                         ${DEFAULT_COUNT_MAX}); the highest question overlap is kept
  --typesafe-count-threshold <p>  Noul a candidate must reach to be counted, 0-1 (default
                         ${DEFAULT_COUNT_THRESHOLD})
  --json                 Print the complete run instead of its summary

Every observation records evidenceCoverage {sessionsInContext, sessionsTotal, turnsInContext,
turnsTotal}: evidence sessions the reader's context renders, and evidence turns (has_answer)
whose whole text is in their session's rendered section. The summary, overall and per question
type, over answerable questions (abstention "_abs" left out):
  evidenceSessionsCompleteRate  share with every evidence session in context
  evidenceTurnsCompleteRate     share with every evidence turn in context
  meanEvidenceTurnCoverage      mean share of evidence turns in context
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

export function parseArgs(argv: string[]): Args {
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
    turnUnitUnlessTemporal: false,
    hybridQuestionTypes: undefined,
    reservedMinimumScore: undefined,
    entityRetrieval: false,
    engineRecall: false,
    dateDistances: false,
    computedNotes: false,
    focusedBudget: false,
    structuredEvidence: false,
    fullSessions: null,
    abstractBytes: DEFAULT_ABSTRACT_BYTES,
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
    readerTemperature: undefined,
    readerTimeoutMs: undefined,
    judgeTemperature: undefined,
    readerBaseUrl: undefined,
    readerApiKey: undefined,
    memorySystem: undefined,
    memoryLane: 'retrieval',
    retrievalOnly: false,
    rerank: 'none',
    rerankPool: DEFAULT_RERANK_POOL,
    rerankSessionChars: DEFAULT_RERANK_SESSION_CHARS,
    rerankModel: DEFAULT_TYPESAFE_MODEL,
    rerankKeyEnv: DEFAULT_RERANK_KEY_ENV,
    rerankConcurrency: DEFAULT_RERANK_CONCURRENCY,
    typesafeCount: false,
    typesafeCountMax: DEFAULT_COUNT_MAX,
    typesafeCountThreshold: DEFAULT_COUNT_THRESHOLD,
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
    } else if (arg === '--turn-unit-question-types') {
      const values = requiredValue(argv, index++, arg)
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);
      if (values.length === 0) {
        throw new Error('--turn-unit-question-types needs at least one value');
      }
      for (const value of values) {
        if (!LONGMEMEVAL_QUESTION_TYPES.has(value)) {
          throw new Error(
            `--turn-unit-question-types: unknown question type "${value}" (known: ${[...LONGMEMEVAL_QUESTION_TYPES].join(', ')})`,
          );
        }
      }
      args.turnUnitQuestionTypes = new Set(values);
    } else if (arg === '--turn-unit-unless-temporal') {
      args.turnUnitUnlessTemporal = true;
    } else if (arg === '--reader-base-url') {
      args.readerBaseUrl = requiredValue(argv, index++, arg).replace(/\/$/, '');
    } else if (arg === '--reader-api-key') {
      args.readerApiKey = requiredValue(argv, index++, arg);
    } else if (arg === '--reader-max-tokens') {
      args.readerMaxTokens = Number(requiredValue(argv, index++, arg));
    } else if (arg === '--reader-temperature') {
      args.readerTemperature = checkedTemperature(
        requiredValue(argv, index++, arg),
        arg,
      );
    } else if (arg === '--reader-timeout-ms') {
      args.readerTimeoutMs = checkedTimeoutMs(
        requiredValue(argv, index++, arg),
        arg,
      );
    } else if (arg === '--judge-temperature') {
      args.judgeTemperature = checkedTemperature(
        requiredValue(argv, index++, arg),
        arg,
      );
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
    } else if (arg === '--date-distances') {
      args.dateDistances = true;
    } else if (arg === '--computed-notes') {
      args.computedNotes = true;
    } else if (arg === '--focused-budget') {
      args.focusedBudget = true;
    } else if (arg === '--structured-evidence') {
      args.structuredEvidence = true;
    } else if (arg === '--full-sessions' || arg === '--abstract-bytes') {
      // validated with the reader contract's own rules once every flag is read
      requiredValue(argv, index++, arg);
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
    } else if (arg === '--memory-system') {
      args.memorySystem = requiredValue(argv, index++, arg);
    } else if (arg === '--memory-lane') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'retrieval' && value !== 'memories') {
        throw new Error('--memory-lane must be retrieval or memories');
      }
      args.memoryLane = value;
    } else if (arg === '--retrieval-only') {
      args.retrievalOnly = true;
    } else if (arg === '--rerank') {
      const value = requiredValue(argv, index++, arg);
      if (value !== 'none' && value !== 'typesafe') {
        throw new Error('--rerank must be none or typesafe');
      }
      args.rerank = value;
    } else if (arg === '--rerank-pool') {
      args.rerankPool = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        100,
      );
    } else if (arg === '--rerank-session-chars') {
      args.rerankSessionChars = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        500,
        100_000,
      );
    } else if (arg === '--rerank-model') {
      args.rerankModel = requiredValue(argv, index++, arg);
    } else if (arg === '--rerank-key-env') {
      const value = requiredValue(argv, index++, arg);
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
        throw new Error('--rerank-key-env needs an environment variable name');
      }
      args.rerankKeyEnv = value;
    } else if (arg === '--rerank-concurrency') {
      args.rerankConcurrency = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        64,
      );
    } else if (arg === '--typesafe-count') {
      args.typesafeCount = true;
    } else if (arg === '--typesafe-count-max') {
      args.typesafeCountMax = boundedInteger(
        requiredValue(argv, index++, arg),
        arg,
        1,
        500,
      );
    } else if (arg === '--typesafe-count-threshold') {
      const parsed = Number(requiredValue(argv, index++, arg));
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
        throw new Error(`${arg} needs a probability from 0 to 1`);
      }
      args.typesafeCountThreshold = parsed;
    } else if (arg === '--json') args.json = true;
    else if (arg === '--help' || arg === '-h') {
      console.log(USAGE);
      process.exit(0);
    } else throw new Error(`unknown option: ${arg}`);
  }
  if (args.turnUnitUnlessTemporal && args.turnUnitQuestionTypes !== undefined) {
    throw new Error(
      '--turn-unit-unless-temporal and --turn-unit-question-types are mutually exclusive: both pick which questions keep the turn unit',
    );
  }
  if (args.turnUnitUnlessTemporal && args.retrievalUnit !== 'turn') {
    throw new Error(
      '--turn-unit-unless-temporal needs --retrieval-unit turn: it picks which questions keep the turn unit',
    );
  }
  if (
    args.turnUnitQuestionTypes !== undefined &&
    args.retrievalUnit !== 'turn'
  ) {
    throw new Error(
      '--turn-unit-question-types needs --retrieval-unit turn: it picks which question types keep the turn unit',
    );
  }
  if (args.typesafeCount && args.memorySystem !== undefined) {
    throw new Error(
      '--typesafe-count counts the sessions Remembero retrieved; it cannot be combined with --memory-system',
    );
  }
  if (args.rerank !== 'none' && args.memorySystem !== undefined) {
    throw new Error(
      '--rerank re-orders Remembero\'s own retrieval; it cannot be combined with --memory-system',
    );
  }
  // the reader contract's rules: positive --full-sessions, --abstract-bytes 120-2048 and only
  // with tiers, and never tiers with --focused-budget (two budget policies are not a pair)
  const tiers = tiersFromFlags(argv);
  args.fullSessions = tiers.fullSessions;
  args.abstractBytes = tiers.abstractBytes;
  return args;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * The main endpoint's key (LLM_API_KEY). A retrieval-only run calls no reader or judge, so it
 * needs the key only when a model it does call (temporal range, extraction) has no key of its
 * own; every other run needs it always.
 */
export function runnerApiKey(
  args: Args,
  env: NodeJS.ProcessEnv,
): string | undefined {
  const apiKey = env.LLM_API_KEY || undefined;
  if (apiKey !== undefined) return apiKey;
  if (!args.retrievalOnly) {
    throw new Error(
      'LLM_API_KEY is not set — add it to .env or the environment',
    );
  }
  if (
    args.temporalRangeModel !== undefined &&
    args.temporalRangeApiKey === undefined &&
    !env.TEMPORAL_RANGE_API_KEY
  ) {
    throw new Error(
      '--temporal-range-model needs a key: pass --temporal-range-api-key, or set TEMPORAL_RANGE_API_KEY or LLM_API_KEY',
    );
  }
  const extracting =
    args.formation !== 'raw' || args.extractionModel !== undefined;
  if (
    extracting &&
    args.extractionApiKey === undefined &&
    !env.EXTRACTION_API_KEY &&
    !env.MODAL_SERVE_API_KEY
  ) {
    throw new Error(
      'extraction needs a key: pass --extraction-api-key, or set EXTRACTION_API_KEY, MODAL_SERVE_API_KEY or LLM_API_KEY',
    );
  }
  return undefined;
}

/** Stands in for the reader and the judge in a retrieval-only run: any call is a bug. */
export function unusedModelClient(role: string): LongMemEvalCompletionClient {
  return {
    model: 'none (retrieval-only)',
    completeWithUsage: async () => {
      throw new Error(`the ${role} must not be called in a retrieval-only run`);
    },
  };
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
      // not fatal: a fully cached extraction never calls the endpoint, and a real
      // outage shows up per question
      if (attempt === attempts) {
        console.error(
          `${label} not ready after ${attempts} attempts; continuing`,
        );
        return;
      }
      console.error(
        `${label} not ready (attempt ${attempt}/${attempts}): ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
}

async function main(): Promise<void> {
  loadEnv();
  const args = parseArgs(process.argv.slice(2));
  if (args.memorySystem !== undefined) {
    if (args.formation !== 'raw') {
      throw new Error(
        '--memory-system does its own formation; drop --formation',
      );
    }
    if (args.engineRecall || args.entityRetrieval) {
      throw new Error(
        '--memory-system cannot be combined with --engine-recall or --entity-retrieval',
      );
    }
    if (args.temporalRangeModel !== undefined) {
      throw new Error(
        '--memory-system cannot be combined with --temporal-range-model: time-aware retrieval is off for every system',
      );
    }
    if (args.semanticQuestionTypes.size > 0) {
      throw new Error(
        '--memory-system needs --local-only: the memory layer under test does the retrieval',
      );
    }
    // the Remembero rows are measured against the stock path: refuse the flags that would
    // move the native side without reaching the built-in
    assertBuiltinMemorySystemScope(args.memorySystem, {
      temporalRangeModel: args.temporalRangeModel,
      retrievalUnit: args.retrievalUnit,
    });
  }
  // a retrieval-only run may have no main key: the clients that would need it are either
  // never built or carry their own key (runnerApiKey refuses the rest)
  const apiKey = runnerApiKey(args, process.env) ?? '';
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
  const reader: LongMemEvalCompletionClient = args.retrievalOnly
    ? unusedModelClient('reader')
    : new OpenRouterClient({
        // the reader may live on another endpoint (e.g. Ollama Cloud) while the judge stays put
        apiKey: args.readerApiKey ?? process.env.READER_API_KEY ?? apiKey,
        baseUrl: args.readerBaseUrl ?? baseUrl,
        model: args.readerModel,
        ...(args.readerTemperature === undefined
          ? {}
          : { temperature: args.readerTemperature }),
        ...(args.readerTimeoutMs === undefined
          ? {}
          : { timeoutMs: args.readerTimeoutMs }),
      });
  const aggregationReader =
    args.aggregationReaderModel === undefined || args.retrievalOnly
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
  const judge: LongMemEvalCompletionClient = args.retrievalOnly
    ? unusedModelClient('judge')
    : new OpenRouterClient({
        apiKey,
        baseUrl,
        model: args.judgeModel,
        ...(args.judgeTemperature === undefined
          ? {}
          : { temperature: args.judgeTemperature }),
      });
  // extracted/hybrid formations run the product's transcript extraction with its own
  // model, typically the fine-tuned dialect model on a self-hosted OpenAI-compatible endpoint
  // raw formation makes no extraction calls of its own, but builtin:remembero-hybrid does
  // its own formation behind --memory-system: --extraction-model is what asks for a writer
  const extractor =
    args.formation === 'raw' && args.extractionModel === undefined
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
  // the engine's query is written from extracted facts, so it needs a formation that has them:
  // --extraction-model alone (which builtin:remembero-hybrid uses) is not one
  if (args.engineRecall && args.formation === 'raw') {
    throw new Error('--engine-recall needs the extracted or hybrid formation');
  }
  // the writer that authors the engine's query is the extraction model
  const engineWriter = extractor as OpenRouterClient;
  const embeddings =
    args.semanticQuestionTypes.size > 0 ? embeddingClientFromEnv() : undefined;
  // A self-hosted extraction endpoint that scaled to zero takes longer to come up
  // than one request's timeout; wake it before the questions start.
  if (extractor !== undefined) await warmUp(extractor, 'extraction endpoint');
  // a self-hosted reader scales to zero as well
  if (args.readerBaseUrl !== undefined && reader instanceof OpenRouterClient)
    await warmUp(reader, 'reader endpoint');
  // builtin:embed needs a vector client even though --local-only turns Remembero's own
  // semantic route off: there the embedding model is the memory system under test
  const memorySystemEmbeddings =
    args.memorySystem === 'builtin:embed'
      ? (embeddings ?? embeddingClientFromEnv())
      : embeddings;
  // one adapter process per worker: each client queues its own requests, so a question is
  // never interleaved with another inside one store
  const memorySystems: MemorySystemClient[] =
    args.memorySystem === undefined
      ? []
      : await Promise.all(
          Array.from(
            { length: args.concurrency },
            async () =>
              await openMemorySystem(args.memorySystem!, {
                ...(memorySystemEmbeddings === undefined
                  ? {}
                  : { embeddings: memorySystemEmbeddings }),
                ...(extractor === undefined ? {} : { extractor }),
                ...(args.extractionCacheDir === undefined
                  ? {}
                  : { extractionCacheDir: args.extractionCacheDir }),
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
              }),
          ),
        );
  let rerankNouls: TypesafeNouls | undefined;
  if (args.rerank === 'typesafe') {
    const rerankKey = process.env[args.rerankKeyEnv];
    if (rerankKey === undefined || rerankKey.trim() === '') {
      throw new Error(
        `--rerank typesafe needs a key in the environment variable ${args.rerankKeyEnv}`,
      );
    }
    // one limiter for the whole run: every question's candidates share it
    const limiter = createLimiter(args.rerankConcurrency);
    rerankNouls = (state, questions) =>
      typesafeNouls(state, questions, {
        apiKey: rerankKey,
        model: args.rerankModel,
        limiter,
      });
  }
  let countNouls: TypesafeNouls | undefined;
  if (args.typesafeCount) {
    const countKey = process.env[args.rerankKeyEnv];
    if (countKey === undefined || countKey.trim() === '') {
      throw new Error(
        `--typesafe-count needs a key in the environment variable ${args.rerankKeyEnv}`,
      );
    }
    // its own limiter, so counting never starves the re-rank of slots (or the other way round)
    const limiter = createLimiter(args.rerankConcurrency);
    countNouls = (state, questions) =>
      typesafeNouls(state, questions, {
        apiKey: countKey,
        model: args.rerankModel,
        limiter,
      });
  }
  let completed = 0;
  let observations: LongMemEvalAnswerObservation[];
  try {
    observations = await mapConcurrent(
      instances,
      args.concurrency,
      async (instance, _index, workerId) => {
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
            ...(args.turnUnitQuestionTypes === undefined
              ? {}
              : { turnUnitQuestionTypes: args.turnUnitQuestionTypes }),
            ...(args.turnUnitUnlessTemporal
              ? { turnUnitRule: 'unless-temporal' as const }
              : {}),
            readingStrategy: args.readingStrategy,
            ...(args.readerMaxTokens === undefined
              ? {}
              : { readerMaxTokens: args.readerMaxTokens }),
            ...(aggregationReader === undefined ? {} : { aggregationReader }),
            ...(temporalRangeExtractor === undefined
              ? {}
              : { temporalRangeExtractor }),
            entityRetrieval: args.entityRetrieval,
            dateDistances: args.dateDistances,
            computedNotes: args.computedNotes,
            focusedBudget: args.focusedBudget,
            structuredEvidence: args.structuredEvidence,
            ...(args.fullSessions === null
              ? {}
              : {
                  tiers: {
                    fullSessions: args.fullSessions,
                    abstractBytes: args.abstractBytes,
                  },
                }),
            ...(memorySystems.length === 0
              ? {}
              : {
                  memorySystem: {
                    // by worker, not by position: one live adapter per worker, never two
                    // in-flight questions queued behind the same process
                    client: memorySystems[workerId % memorySystems.length]!,
                    lane: args.memoryLane,
                  },
                }),
            retrievalOnly: args.retrievalOnly,
            ...(rerankNouls === undefined
              ? {}
              : {
                  rerank: {
                    nouls: rerankNouls,
                    pool: args.rerankPool,
                    sessionChars: args.rerankSessionChars,
                  },
                }),
            ...(countNouls === undefined
              ? {}
              : {
                  typesafeCount: {
                    nouls: countNouls,
                    max: args.typesafeCountMax,
                    threshold: args.typesafeCountThreshold,
                  },
                }),
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
          const coverage = observation.evidenceCoverage;
          const outcome =
            observation.status === 'judged'
              ? observation.correct
                ? 'correct'
                : 'incorrect'
              : observation.status === 'retrieval-only'
                ? 'context built'
                : `error (${observation.error})`;
          console.error(
            `[${completed}/${instances.length}] ${instance.question_id}: ${outcome}${coverage === null ? '' : `; evidence sessions ${coverage.sessionsInContext}/${coverage.sessionsTotal}, turns ${coverage.turnsInContext}/${coverage.turnsTotal}`}`,
          );
        }
        return observation;
      },
    );
  } finally {
    // the adapter processes outlive every question, but not the run
    await Promise.all(memorySystems.map(async (client) => await client.close()));
  }
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
    ...(args.memorySystem === undefined
      ? {}
      : { memorySystemId: args.memorySystem }),
    settings: {
      aggregationReaderModel: aggregationReader?.model ?? null,
      temporalRangeModel: temporalRangeExtractor?.model ?? null,
      readingStrategy: args.readingStrategy,
      hybridRetrieval: args.hybridRetrieval,
      retrievalUnit: args.retrievalUnit,
      turnUnitQuestionTypes:
        args.turnUnitQuestionTypes === undefined
          ? null
          : [...args.turnUnitQuestionTypes],
      turnUnitRule: args.turnUnitUnlessTemporal ? 'unless-temporal' : null,
      entityRetrieval: args.entityRetrieval,
      engineRecall: args.engineRecall,
      dateDistances: args.dateDistances,
      computedNotes: args.computedNotes,
      focusedBudget: args.focusedBudget,
      structuredEvidence: args.structuredEvidence,
      fullSessions: args.fullSessions,
      abstractBytes: args.fullSessions === null ? null : args.abstractBytes,
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
      readerTemperature: args.readerTemperature ?? DEFAULT_LLM_TEMPERATURE,
      readerTimeoutMs: args.readerTimeoutMs ?? DEFAULT_LLM_TIMEOUT_MS,
      judgeTemperature: args.judgeTemperature ?? DEFAULT_LLM_TEMPERATURE,
      memorySystem: args.memorySystem ?? null,
      retrievalOnly: args.retrievalOnly,
      rerank: args.rerank,
      rerankPool: args.rerank === 'none' ? null : args.rerankPool,
      rerankSessionChars:
        args.rerank === 'none' ? null : args.rerankSessionChars,
      rerankModel: args.rerank === 'none' ? null : args.rerankModel,
      rerankKeyEnv: args.rerank === 'none' ? null : args.rerankKeyEnv,
      rerankConcurrency: args.rerank === 'none' ? null : args.rerankConcurrency,
      typesafeCount: args.typesafeCount,
      typesafeCountMax: args.typesafeCount ? args.typesafeCountMax : null,
      typesafeCountThreshold: args.typesafeCount
        ? args.typesafeCountThreshold
        : null,
      memoryLane: args.memorySystem === undefined ? null : args.memoryLane,
      // the memory system's own embedder, recorded here rather than in the top-level
      // embeddingModel: that field means Remembero's semantic route, which --local-only turns off
      memorySystemEmbeddingModel:
        args.memorySystem === undefined
          ? null
          : (memorySystemEmbeddings?.model ?? null),
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
    if (args.retrievalOnly) {
      console.log(
        'accuracy: not measured (retrieval-only: no reader, no judge)',
      );
    } else {
      console.log(
        `accuracy: ${percent(summary.accuracy)} (${summary.correct}/${summary.questions})`,
      );
    }
    console.log(`errors: ${summary.errors}`);
    console.log(
      `retrieval/context recall: ${percent(summary.retrievalRecallAtK)} / ${percent(summary.contextRecallAtK)} (top-k ${args.topK}; multi-session ${args.multiSessionTopK}; temporal ${args.temporalTopK})`,
    );
    console.log(
      `evidence complete, sessions/turns: ${percent(summary.evidenceSessionsCompleteRate)} / ${percent(summary.evidenceTurnsCompleteRate)}; mean turn coverage ${percent(summary.meanEvidenceTurnCoverage)} (${summary.evidenceCoverageQuestions} answerable questions)`,
    );
    for (const [type, typed] of Object.entries(run.byQuestionType)) {
      console.log(
        `  ${type}: sessions ${percent(typed.evidenceSessionsCompleteRate)}, turns ${percent(typed.evidenceTurnsCompleteRate)}, mean turn coverage ${percent(typed.meanEvidenceTurnCoverage)} (${typed.evidenceCoverageQuestions})`,
      );
    }
    if (!args.retrievalOnly) {
      console.log(
        `full/incomplete evidence accuracy: ${percent(summary.fullContextEvidenceAccuracy)} / ${percent(summary.incompleteContextEvidenceAccuracy)}`,
      );
    }
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
    if (args.memorySystem !== undefined) {
      const adapter = summarizeMemorySystemUsage(observations);
      console.log(
        `memory system: ${args.memorySystem} (lane ${args.memoryLane})`,
      );
      console.log(
        `memory system calls/tokens/cost: ${adapter.modelCalls} / ${adapter.inputTokens + adapter.outputTokens} / $${adapter.costUsd.toFixed(6)}`,
      );
      console.log(
        `memory system ingest/search p50: ${adapter.medianIngestMs.toFixed(1)} / ${adapter.medianSearchMs.toFixed(1)} ms; dropped memory bytes ${adapter.droppedMemoryBytes}`,
      );
      const overreach = observations.filter(
        (observation) => observation.memorySystem?.overRequestedDepth === true,
      ).length;
      if (overreach > 0) {
        console.log(
          `note: ${overreach} questions returned more sessions than that question's top-k; the harness scored them at it anyway (full context does this by design)`,
        );
      }
    }
    console.log(
      `embedding calls/tokens/cost: ${summary.embeddingUsage.calls} / ${summary.embeddingUsage.totalTokens} / $${summary.embeddingUsage.costUsd.toFixed(6)}`,
    );
    if (summary.rerankUsage !== undefined) {
      const r = summary.rerankUsage;
      console.log(
        `typesafe rerank (experimental): ${r.candidates} candidates, ${r.calls} requests, ${r.cached} cached, ${r.blocked} blocked; ${r.inputTokens} input tokens, est. $${r.costUsd.toFixed(6)} at $0.042/M`,
      );
    }
    if (summary.countUsage !== undefined) {
      const c = summary.countUsage;
      console.log(
        `typesafe counted items (experimental): ${c.candidates} candidates, ${c.counted} counted, ${c.calls} requests, ${c.cached} cached; ${c.inputTokens} input tokens, est. $${c.costUsd.toFixed(6)} at $0.042/M`,
      );
    }
    if (summary.rerankUsage !== undefined || summary.countUsage !== undefined) {
      const tokens =
        (summary.rerankUsage?.inputTokens ?? 0) +
        (summary.countUsage?.inputTokens ?? 0);
      console.log(
        `typesafe total: ${tokens} input tokens, est. $${typesafeCostUsd(tokens).toFixed(6)}`,
      );
    }
    console.log(
      `preparation calls/tokens/cost: ${summary.semanticPreparationUsage.calls} / ${summary.semanticPreparationUsage.totalTokens} / $${summary.semanticPreparationUsage.costUsd.toFixed(6)}`,
    );
  }
  if (run.summary.errors > 0) process.exitCode = 1;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];

if (invokedDirectly) await main();
