/**
 * `npm run eval:doc-extract` — the extraction half of the page-scale benchmark.
 *
 * The writer is the product's own `rememberTranscriptText`: one page of a real document at a
 * time, into a real store, with the schema of what it has already learned in the prompt. Two
 * numbers come out — how much of what it stored the page actually supports (precision), and how
 * many labelled questions the facts off their evidence pages alone can answer (recall).
 *
 * Extraction is sampled: a thousand pages through a local 4B writer is hours of GPU, so every
 * labelled evidence page is extracted (recall is exact) and precision carries its sample size.
 */

import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { OpenRouterClient } from '../llm/client.js';
import { rememberTranscriptText } from '../llm/pipeline.js';
import { MemoryStore } from '../store/store.js';
import { pagesFromPdf, pagesFromTextFile, type DocumentPage } from './document-corpus.js';
import { measureDocumentExtraction, type ExtractionSummary } from './document-extraction.js';
import type { DocumentQuestion } from './document-recall.js';

interface SpecSource {
  id: string;
  title: string;
  sourceUrl: string;
  pdf?: string;
  text?: string;
  questions?: Array<{
    id: string;
    question: string;
    answer: string | null;
    evidencePages: number[];
    answerFormat?: string;
    verificationRule?: string;
  }>;
}

interface CorpusSpec {
  name: string;
  tiers: Array<{ name: string; documents?: string[] }>;
  sources: SpecSource[];
}

const CACHE_DIR = '.cache/document-recall';

interface Args {
  spec: string;
  tiers?: Set<string>;
  documentsPerTier: number;
  samplePages: number;
  judgeFacts: number;
  writerModel: string;
  writerBaseUrl?: string;
  writerApiKey?: string;
  writerTimeoutMs?: number;
  judgeModel: string;
  judgeBaseUrl?: string;
  judgeApiKey?: string;
  concurrency: number;
  output?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    spec: 'benchmarks/document-recall/spec.json',
    documentsPerTier: 1,
    samplePages: 10,
    judgeFacts: 30,
    writerModel: process.env.WRITER_MODEL ?? 'rembero-reader-v7',
    judgeModel: process.env.JUDGE_MODEL ?? 'deepseek-chat',
    concurrency: 1,
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
      case '--documents-per-tier': args.documentsPerTier = Number(value()); break;
      case '--sample-pages': args.samplePages = Number(value()); break;
      case '--judge-facts': args.judgeFacts = Number(value()); break;
      case '--writer-model': args.writerModel = value(); break;
      case '--writer-base-url': args.writerBaseUrl = value(); break;
      case '--writer-api-key': args.writerApiKey = value(); break;
      case '--writer-timeout-ms': args.writerTimeoutMs = Number(value()); break;
      case '--judge-model': args.judgeModel = value(); break;
      case '--judge-base-url': args.judgeBaseUrl = value(); break;
      case '--judge-api-key': args.judgeApiKey = value(); break;
      case '--concurrency': args.concurrency = Number(value()); break;
      case '--output': args.output = value(); break;
      default: throw new Error(`unknown flag: ${flag}`);
    }
  }
  return args;
}

function loadPages(source: SpecSource): DocumentPage[] {
  if (source.text !== undefined) return pagesFromTextFile(source.text);
  if (source.pdf === undefined) throw new Error(`source ${source.id} has neither pdf nor text`);
  mkdirSync(CACHE_DIR, { recursive: true });
  const cached = `${CACHE_DIR}/${source.id}.pages.txt`;
  try {
    return pagesFromTextFile(cached);
  } catch {
    const pages = pagesFromPdf(resolve(source.pdf));
    writeFileSync(cached, pages.map((page) => page.text).join('\f'));
    return pages;
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const spec = JSON.parse(readFileSync(args.spec, 'utf8')) as CorpusSpec;
  const apiKey = process.env.LLM_API_KEY ?? '';
  const baseUrl = (process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');

  const writer = new OpenRouterClient({
    model: args.writerModel,
    apiKey: args.writerApiKey ?? process.env.WRITER_API_KEY ?? apiKey,
    baseUrl: args.writerBaseUrl ?? baseUrl,
    ...(args.writerTimeoutMs === undefined ? {} : { timeoutMs: args.writerTimeoutMs }),
  });
  const judge = new OpenRouterClient({
    model: args.judgeModel,
    apiKey: args.judgeApiKey ?? process.env.JUDGE_API_KEY ?? apiKey,
    baseUrl: args.judgeBaseUrl ?? baseUrl,
  });

  const byId = new Map(spec.sources.map((source) => [source.id, source]));
  console.log(`corpus ${spec.name} · writer ${writer.model} · fact judge ${judge.model}`);
  console.log(`sampling ${args.samplePages} pages beyond the evidence pages, judging ${args.judgeFacts} facts\n`);

  const results: Array<{ tier: string; document: string; pages: number; summary: ExtractionSummary }> = [];
  for (const tier of spec.tiers) {
    if (args.tiers !== undefined && !args.tiers.has(tier.name)) continue;
    for (const documentId of (tier.documents ?? []).slice(0, args.documentsPerTier)) {
      const source = byId.get(documentId);
      if (source === undefined) throw new Error(`tier ${tier.name} names unknown document ${documentId}`);
      const pages = loadPages(source);
      const questions: DocumentQuestion[] = (source.questions ?? []).map((question) => ({
        ...question,
        sourceDocument: source.id,
        documentId: source.id,
      }));
      // a store per document: the writer sees the schema it has built from this document only
      const root = mkdtempSync(join(tmpdir(), `doc-extract-${documentId}-`));
      const store = new MemoryStore(root);
      console.log(`${tier.name} · ${documentId}: ${pages.length} pages, ${questions.length} questions`);

      const result = await measureDocumentExtraction(pages, questions, {
        judge,
        samplePages: args.samplePages,
        judgeFacts: args.judgeFacts,
        concurrency: args.concurrency,
        extract: async (page) => {
          const outcome = await rememberTranscriptText(
            { store, llm: writer },
            `[page ${page.page} of ${source.title}]\n${page.text.slice(0, 12_000)}`,
            'document',
            { captureId: `${documentId}-p${page.page}`, origin: 'manual' },
          );
          return { facts: outcome.added };
        },
        onPage: (page, facts, index, total) => {
          console.log(`  [${index + 1}/${total}] page ${page}: ${facts} facts`);
        },
      });
      const summary = result.summary;
      console.log(
        `  ${documentId}: ${summary.facts} facts from ${summary.pagesSampled} pages ` +
          `(${summary.factsPerPage.toFixed(1)}/page, ${summary.extractionErrors} writer errors) · ` +
          `precision ${(summary.precision * 100).toFixed(1)}% of ${summary.factsJudged} judged · ` +
          `recall ${(summary.recall * 100).toFixed(1)}% of ${summary.questionsScored} questions\n`,
      );
      results.push({ tier: tier.name, document: documentId, pages: pages.length, summary });

      if (args.output !== undefined) {
        mkdirSync(dirname(resolve(args.output)), { recursive: true });
        writeFileSync(
          args.output,
          `${JSON.stringify(
            { corpus: spec.name, ranAt: new Date().toISOString(), writer: writer.model, judge: judge.model, results },
            null,
            2,
          )}\n`,
        );
      }
    }
  }

  console.log('tier   | document    | pages | sampled | facts | facts/pg | precision | recall | errors');
  console.log('------ | ----------- | ----- | ------- | ----- | -------- | --------- | ------ | ------');
  for (const row of results) {
    console.log(
      [
        row.tier.padEnd(6),
        row.document.padEnd(11),
        String(row.pages).padStart(5),
        String(row.summary.pagesSampled).padStart(7),
        String(row.summary.facts).padStart(5),
        row.summary.factsPerPage.toFixed(1).padStart(8),
        `${(row.summary.precision * 100).toFixed(1)}%`.padStart(9),
        `${(row.summary.recall * 100).toFixed(1)}%`.padStart(6),
        String(row.summary.extractionErrors).padStart(6),
      ].join(' | '),
    );
  }
  if (args.output !== undefined) console.log(`\nwrote ${args.output}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
