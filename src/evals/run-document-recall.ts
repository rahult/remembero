/**
 * `npm run eval:doc-recall` — the page-scale sweep over XL-DocBench documents.
 *
 * A tier is a set of real documents whose length lands in its band (about 100, 500 or 1000
 * pages). Each document is retrieved over on its own — its windows, its questions — and the tier
 * row is the aggregate, so "1000 pages" means one document of that length rather than a pile of
 * short ones. A tier may instead ask for `assemblePages`, which concatenates its documents to an
 * exact page count: that is how the thin 1000-page band gets a second, larger-sample reading.
 *
 * Accuracy is XL-DocBench's own rule (ported in xl-docbench-score.ts), so the numbers sit beside
 * its published baselines. `--judge-model` adds a second opinion; it is not needed for a result.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { OpenRouterClient } from '../llm/client.js';
import {
  pagesFromPdf,
  pagesFromTextFile,
  DEFAULT_PAGES_PER_WINDOW,
  DEFAULT_WINDOW_BYTES,
  type DocumentPage,
} from './document-corpus.js';
import { evaluateDocumentTier } from './document-recall-run.js';
import { ollamaEmbed } from './document-index.js';
import { RANKERS, buildDocumentRanker, llmDecomposer, type RankerName } from './document-rankers.js';
import { DEFAULT_RERANK_KEY_ENV, typesafeNouls } from './typesafe-rerank.js';
import { assembleTier, type TierSource } from './document-tier.js';
import {
  summariseTier,
  type DocumentQuestion,
  type DocumentTier,
  type QuestionOutcome,
  type TierSummary,
} from './document-recall.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';

interface SpecSource {
  id: string;
  title: string;
  sourceUrl: string;
  sha256?: string;
  pdf?: string;
  text?: string;
  labelPages?: number;
  questions?: Array<{
    id: string;
    question: string;
    answer: string | null;
    evidencePages: number[];
    datasetKind?: string;
    answerFormat?: string;
    verificationRule?: string;
  }>;
}

interface CorpusSpec {
  name: string;
  labels?: string;
  tiers: Array<{ name: string; documents?: string[]; assemblePages?: number; band?: number[] }>;
  sources: SpecSource[];
}

const CACHE_DIR = '.cache/document-recall';

interface Args {
  spec: string;
  tiers?: Set<string>;
  readerModel: string;
  readerBaseUrl?: string;
  readerApiKey?: string;
  readerTimeoutMs?: number;
  judgeModel?: string;
  judgeBaseUrl?: string;
  judgeApiKey?: string;
  topK: number;
  contextBytes: number;
  pagesPerWindow: number;
  windowBytes: number;
  maxTokens?: number;
  concurrency: number;
  limitPerDocument?: number;
  oraclePages: boolean;
  ranker?: RankerName;
  price?: { inputPerMillion: number; outputPerMillion: number };
  output?: string;
  predictions?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    spec: 'benchmarks/document-recall/spec.json',
    readerModel: process.env.READER_MODEL ?? 'rembero-reader-v7',
    topK: 4,
    contextBytes: 24 * 1024,
    pagesPerWindow: DEFAULT_PAGES_PER_WINDOW,
    windowBytes: DEFAULT_WINDOW_BYTES,
    concurrency: 1,
    oraclePages: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = () => {
      const next = argv[index + 1];
      if (next === undefined) throw new Error(`${flag} needs a value`);
      index += 1;
      return next;
    };
    switch (flag) {
      case '--spec': args.spec = value(); break;
      case '--tiers': args.tiers = new Set(value().split(',').map((name) => name.trim())); break;
      case '--reader-model': args.readerModel = value(); break;
      case '--reader-base-url': args.readerBaseUrl = value(); break;
      case '--reader-api-key': args.readerApiKey = value(); break;
      case '--reader-timeout-ms': args.readerTimeoutMs = Number(value()); break;
      case '--judge-model': args.judgeModel = value(); break;
      case '--judge-base-url': args.judgeBaseUrl = value(); break;
      case '--judge-api-key': args.judgeApiKey = value(); break;
      case '--top-k': args.topK = Number(value()); break;
      case '--context-bytes': args.contextBytes = Number(value()); break;
      case '--pages-per-window': args.pagesPerWindow = Number(value()); break;
      case '--window-bytes': args.windowBytes = Number(value()); break;
      case '--max-tokens': args.maxTokens = Number(value()); break;
      case '--concurrency': args.concurrency = Number(value()); break;
      case '--limit-per-document': args.limitPerDocument = Number(value()); break;
      case '--oracle-pages': args.oraclePages = true; break;
      case '--ranker': {
        const name = value();
        if (!(RANKERS as readonly string[]).includes(name)) throw new Error(`unknown ranker ${name}`);
        args.ranker = name as RankerName;
        break;
      }
      case '--reader-price': {
        // "IN,OUT" dollars per million tokens, for an endpoint that reports no cost; 0,0 for local
        const [input, output] = value().split(',').map(Number);
        if (!Number.isFinite(input) || !Number.isFinite(output)) {
          throw new Error('--reader-price takes IN,OUT dollars per million tokens, e.g. 0.28,0.42');
        }
        args.price = { inputPerMillion: input!, outputPerMillion: output! };
        break;
      }
      case '--output': args.output = value(); break;
      case '--predictions': args.predictions = value(); break;
      default: throw new Error(`unknown flag: ${flag}`);
    }
  }
  return args;
}

/** Pages for one source, extracting from the PDF once and caching the page text. */
function loadPages(source: SpecSource): DocumentPage[] {
  if (source.text !== undefined) return pagesFromTextFile(source.text);
  if (source.pdf === undefined) throw new Error(`source ${source.id} has neither pdf nor text`);
  const pdf = resolve(source.pdf);
  if (!existsSync(pdf)) throw new Error(`source ${source.id}: ${pdf} is not downloaded`);
  if (source.sha256 !== undefined) {
    const digest = createHash('sha256').update(readFileSync(pdf)).digest('hex');
    if (digest !== source.sha256) {
      throw new Error(`source ${source.id}: sha256 is ${digest}, spec says ${source.sha256}`);
    }
  }
  mkdirSync(CACHE_DIR, { recursive: true });
  const cached = `${CACHE_DIR}/${source.id}.pages.txt`;
  if (existsSync(cached) && statSync(cached).mtimeMs > statSync(pdf).mtimeMs) {
    return pagesFromTextFile(cached);
  }
  const pages = pagesFromPdf(pdf);
  writeFileSync(cached, pages.map((page) => page.text).join('\f'));
  return pages;
}

function questionsOf(source: SpecSource): DocumentQuestion[] {
  return (source.questions ?? []).map((question) => ({
    ...question,
    sourceDocument: source.id,
    documentId: source.id,
  }));
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function row(summary: TierSummary): string {
  return [
    summary.tier.padEnd(6),
    String(summary.pages).padStart(5),
    String(summary.windows).padStart(7),
    `${summary.answerable}+${summary.unanswerable}`.padStart(6),
    percent(summary.accuracy).padStart(8),
    percent(summary.localeAccuracy).padStart(7),
    (summary.judgedAccuracy === undefined ? '-' : percent(summary.judgedAccuracy)).padStart(6),
    percent(summary.evidenceHitRate).padStart(8),
    percent(summary.meanPageRecall).padStart(11),
    percent(summary.meanWindowPrecision).padStart(11),
    percent(summary.falseAnswerRate).padStart(12),
    `${(summary.meanLatencyMs / 1000).toFixed(1)}s`.padStart(7),
    `${Math.round(summary.meanContextBytes / 1024)}KB`.padStart(7),
  ].join(' | ');
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const spec = JSON.parse(readFileSync(args.spec, 'utf8')) as CorpusSpec;
  const apiKey = process.env.LLM_API_KEY ?? '';
  const baseUrl = (process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');

  const reader = new OpenRouterClient({
    model: args.readerModel,
    apiKey: args.readerApiKey ?? process.env.READER_API_KEY ?? apiKey,
    baseUrl: args.readerBaseUrl ?? baseUrl,
    ...(args.readerTimeoutMs === undefined ? {} : { timeoutMs: args.readerTimeoutMs }),
  });
  const judge: LongMemEvalCompletionClient | undefined =
    args.judgeModel === undefined
      ? undefined
      : new OpenRouterClient({
          model: args.judgeModel,
          apiKey: args.judgeApiKey ?? process.env.JUDGE_API_KEY ?? apiKey,
          baseUrl: args.judgeBaseUrl ?? baseUrl,
        });

  const byId = new Map(spec.sources.map((source) => [source.id, source]));
  console.log(`corpus ${spec.name} · labels: ${spec.labels ?? 'unstated'}`);
  console.log(
    `reader ${reader.model}${judge === undefined ? '' : ` · judge ${judge.model}`} · ` +
      `depth ${args.topK} · ${args.pagesPerWindow} pages/window · ${Math.round(args.contextBytes / 1024)}KB context` +
      ` · ranker ${args.ranker ?? 'product'}` +
      `${args.oraclePages ? ' · ORACLE PAGES (retrieval bypassed: this is the reader ceiling)' : ''}\n`,
  );

  const summaries: TierSummary[] = [];
  const perTier: unknown[] = [];
  const predictions: Array<{ question_id: string; prediction: string }> = [];

  for (const tierSpec of spec.tiers) {
    if (args.tiers !== undefined && !args.tiers.has(tierSpec.name)) continue;
    const sources = (tierSpec.documents ?? []).map((id) => {
      const source = byId.get(id);
      if (source === undefined) throw new Error(`tier ${tierSpec.name} names unknown document ${id}`);
      return source;
    });
    if (sources.length === 0) continue;

    const outcomes: QuestionOutcome[] = [];
    const errors: Array<{ questionId: string; message: string }> = [];
    const perDocument: unknown[] = [];
    let tierPages = 0;
    let tierWindows = 0;

    if (tierSpec.assemblePages !== undefined) {
      // the concatenated variant: one document of an exact page count, built from several
      const tierSources: TierSource[] = sources.map((source) => ({
        id: source.id,
        title: source.title,
        sourceUrl: source.sourceUrl,
        sha256: source.sha256 ?? '',
        pages: loadPages(source),
        questions: source.questions ?? [],
      }));
      const assembled = assembleTier(tierSpec.name, tierSpec.assemblePages, tierSources);
      const questions =
        args.limitPerDocument === undefined
          ? assembled.tier.questions
          : assembled.tier.questions.slice(0, args.limitPerDocument);
      console.log(
        `${tierSpec.name}: assembled ${assembled.tier.pages} pages from ` +
          `${assembled.tier.members.length} documents, ${questions.length} questions` +
          (assembled.droppedQuestions.length > 0
            ? ` (${assembled.droppedQuestions.length} dropped by truncation)`
            : ''),
      );
      const result = await runOne({ ...assembled.tier, questions }, assembled.pages, args, reader, judge);
      outcomes.push(...result.outcomes);
      errors.push(...result.errors);
      tierPages = assembled.tier.pages;
      tierWindows = result.summary.windows;
      perDocument.push({ document: 'assembled', members: assembled.tier.members, summary: result.summary });
    } else {
      for (const source of sources) {
        const pages = loadPages(source);
        const all = questionsOf(source);
        const questions = args.limitPerDocument === undefined ? all : all.slice(0, args.limitPerDocument);
        console.log(
          `${tierSpec.name} · ${source.id} (${pages.length} pages, ${questions.length} questions) — ${source.title.slice(0, 70)}`,
        );
        const result = await runOne(
          { tier: `${tierSpec.name}/${source.id}`, pages: pages.length, members: [], questions },
          pages,
          args,
          reader,
          judge,
        );
        outcomes.push(...result.outcomes);
        errors.push(...result.errors);
        tierPages += pages.length;
        tierWindows += result.summary.windows;
        perDocument.push({
          document: source.id,
          title: source.title,
          sourceUrl: source.sourceUrl,
          sha256: source.sha256,
          pages: pages.length,
          summary: result.summary,
        });
      }
    }

    for (const outcome of outcomes) {
      predictions.push({ question_id: outcome.question.id, prediction: outcome.answer });
    }
    const summary = summariseTier(
      { tier: tierSpec.name, pages: tierPages, members: [], questions: [] },
      tierWindows,
      outcomes,
    );
    summaries.push(summary);
    perTier.push({
      tier: tierSpec.name,
      band: tierSpec.band,
      assembled: tierSpec.assemblePages !== undefined,
      summary,
      documents: perDocument,
      errors,
      outcomes: outcomes.map((outcome) => ({
        id: outcome.question.id,
        document: outcome.question.documentId,
        question: outcome.question.question,
        gold: outcome.question.answer,
        answerFormat: outcome.question.answerFormat,
        evidencePages: outcome.question.evidencePages,
        answer: outcome.answer,
        correct: outcome.correct,
        correctLocale: outcome.correctLocale,
        tokenF1: Number(outcome.tokenF1.toFixed(3)),
        anls: Number(outcome.anls.toFixed(3)),
        judged: outcome.judged,
        abstained: outcome.abstained,
        retrieval: outcome.retrieval,
        latencyMs: Math.round(outcome.latencyMs),
        contextBytes: outcome.contextBytes,
        readerTokens: outcome.readerTokens,
        readerCostUsd: outcome.readerCostUsd,
        retrievalMs: Math.round(outcome.retrievalMs ?? 0),
        readerMs: Math.round(outcome.readerMs ?? 0),
      })),
    });
    console.log(
      `  ${tierSpec.name}: accuracy ${percent(summary.accuracy)} (locale-tolerant ${percent(summary.localeAccuracy)}` +
        `${summary.judgedAccuracy === undefined ? '' : `, judge ${percent(summary.judgedAccuracy)}`}) · ` +
        `page hit ${percent(summary.evidenceHitRate)} · token F1 ${summary.meanTokenF1.toFixed(3)} · ${errors.length} errors\n`,
    );
  }

  console.log(
    'tier   | pages | windows |    Qs | accuracy | locale | judge | page hit | page recall | window prec | false answer | latency | context',
  );
  console.log(
    '------ | ----- | ------- | ----- | -------- | ------ | ----- | -------- | ----------- | ----------- | ------------ | ------- | -------',
  );
  for (const summary of summaries) console.log(row(summary));

  if (args.predictions !== undefined) {
    mkdirSync(dirname(resolve(args.predictions)), { recursive: true });
    writeFileSync(
      args.predictions,
      `${predictions.map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    console.log(`\nwrote ${args.predictions} (${predictions.length} predictions, for the benchmark's own evaluate.py)`);
  }
  if (args.output !== undefined) {
    mkdirSync(dirname(resolve(args.output)), { recursive: true });
    writeFileSync(
      args.output,
      `${JSON.stringify(
        {
          corpus: spec.name,
          labels: spec.labels,
          ranAt: new Date().toISOString(),
          reader: reader.model,
          judge: judge?.model ?? null,
          settings: {
            oraclePages: args.oraclePages,
            ranker: args.ranker ?? 'product',
            price: args.price ?? null,
            topK: args.topK,
            contextBytes: args.contextBytes,
            pagesPerWindow: args.pagesPerWindow,
            windowBytes: args.windowBytes,
          },
          tiers: perTier,
        },
        null,
        2,
      )}\n`,
    );
    console.log(`wrote ${args.output}`);
  }
}

/** The ranker factory for --ranker: embeddings, TypeSafe and decomposition wired only if needed. */
function rankerFactory(args: Args, documentId: string) {
  if (args.ranker === undefined || args.ranker === 'product') return undefined;
  const name = args.ranker;
  const typesafeKey = process.env[DEFAULT_RERANK_KEY_ENV];
  return (sessions: Parameters<typeof buildDocumentRanker>[1]) =>
    buildDocumentRanker(name, sessions, {
      embed: ollamaEmbed(),
      denseCachePath: `${CACHE_DIR}/${documentId}.dense.nomic.json`,
      ...(name.endsWith('+rerank') && typesafeKey
        ? { nouls: (state: unknown, questions: Parameters<typeof typesafeNouls>[1]) => typesafeNouls(state, questions, { apiKey: typesafeKey }) }
        : {}),
      ...(name.startsWith('decomposed')
        ? {
            decomposer: llmDecomposer(
              new OpenRouterClient({ model: 'deepseek-chat', apiKey: process.env.DEEPSEEK_API_KEY ?? '', baseUrl: 'https://api.deepseek.com/v1' }),
              `${CACHE_DIR}/decompose.json`,
            ),
          }
        : {}),
    });
}

async function runOne(
  tier: DocumentTier,
  pages: DocumentPage[],
  args: Args,
  reader: OpenRouterClient,
  judge: LongMemEvalCompletionClient | undefined,
) {
  return evaluateDocumentTier(tier, pages, {
    reader,
    ...(judge === undefined ? {} : { judge }),
    topK: args.topK,
    contextBytes: args.contextBytes,
    pagesPerWindow: args.pagesPerWindow,
    windowBytes: args.windowBytes,
    concurrency: args.concurrency,
    ...(args.oraclePages ? { oraclePages: true } : {}),
    ...(args.price === undefined ? {} : { price: args.price }),
    ...(() => {
      const documentId = tier.tier.includes('/') ? tier.tier.split('/')[1]! : tier.tier;
      const buildRanker = rankerFactory(args, documentId);
      return buildRanker === undefined ? {} : { buildRanker };
    })(),
    ...(args.maxTokens === undefined ? {} : { maxTokens: args.maxTokens }),
    onQuestion: (outcome, index, total) => {
      const mark = outcome.correct ? 'ok  ' : 'MISS';
      const hit = outcome.retrieval.evidenceHit ? 'page-hit ' : 'page-miss';
      console.log(
        `  [${index + 1}/${total}] ${mark} ${hit} ${(outcome.latencyMs / 1000).toFixed(0).padStart(3)}s ` +
          `gold=${String(outcome.question.answer).slice(0, 40)} | got=${outcome.answer.slice(0, 60)}`,
      );
    },
  });
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
