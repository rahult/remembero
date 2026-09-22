/**
 * `npm run eval:doc-retrieval` — compare page rankers without a reader.
 *
 * Page hit needs no model to score: a ranking either put a gold evidence page in the first k
 * windows or it did not. So every ranker can be swept over every question for the price of the
 * index (local embeddings) and, for the re-rank and decomposition arms, a few cents of TypeSafe
 * and DeepSeek. Build time and per-question query time are reported too: the speed lens.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { OpenRouterClient } from '../llm/client.js';
import { pagesFromTextFile, windowPages, windowsAsSessions } from './document-corpus.js';
import { ollamaEmbed } from './document-index.js';
import { RANKERS, buildDocumentRanker, llmDecomposer, type RankerName } from './document-rankers.js';
import { scoreRetrieval } from './document-recall.js';
import { loadFactsByPage } from './run-document-facts.js';
import { mapConcurrent } from './map-concurrent.js';
import { DEFAULT_RERANK_KEY_ENV, typesafeCostUsd, typesafeNouls } from './typesafe-rerank.js';

interface Spec {
  tiers: Array<{ name: string; documents?: string[] }>;
  sources: Array<{ id: string; questions?: Array<{ id: string; question: string; answer: string | null; evidencePages: number[] }> }>;
}

const DEPTHS = [4, 12] as const;

function parseArgs(argv: string[]) {
  const args = {
    spec: 'benchmarks/document-recall/spec.json',
    rankers: [...RANKERS] as RankerName[],
    tiers: undefined as Set<string> | undefined,
    output: undefined as string | undefined,
    concurrency: 4,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = () => {
      const next = argv[i + 1];
      if (next === undefined) throw new Error(`${flag} needs a value`);
      i += 1;
      return next;
    };
    if (flag === '--spec') args.spec = value();
    else if (flag === '--rankers') {
      const names = value().split(',').map((name) => name.trim());
      for (const name of names) if (!(RANKERS as readonly string[]).includes(name)) throw new Error(`unknown ranker ${name}`);
      args.rankers = names as RankerName[];
    } else if (flag === '--tiers') args.tiers = new Set(value().split(','));
    else if (flag === '--output') args.output = value();
    else if (flag === '--concurrency') args.concurrency = Number(value());
    else throw new Error(`unknown flag: ${flag}`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const spec = JSON.parse(readFileSync(args.spec, 'utf8')) as Spec;
  const byId = new Map(spec.sources.map((source) => [source.id, source]));
  const needsRerank = args.rankers.some((name) => name.endsWith('+rerank'));
  const needsDecompose = args.rankers.some((name) => name.startsWith('decomposed'));

  let typesafeTokens = 0;
  const typesafeKey = process.env[DEFAULT_RERANK_KEY_ENV];
  if (needsRerank && !typesafeKey) throw new Error(`${DEFAULT_RERANK_KEY_ENV} is not set`);
  const nouls = needsRerank
    ? async (state: unknown, questions: Parameters<typeof typesafeNouls>[1]) => {
        const result = await typesafeNouls(state, questions, { apiKey: typesafeKey! });
        if (!result.cached) typesafeTokens += result.inputTokens;
        return result;
      }
    : undefined;
  const decomposer = needsDecompose
    ? llmDecomposer(
        new OpenRouterClient({
          model: 'deepseek-chat',
          apiKey: process.env.DEEPSEEK_API_KEY ?? '',
          baseUrl: 'https://api.deepseek.com/v1',
        }),
        '.cache/document-recall/decompose.json',
      )
    : undefined;
  const embed = ollamaEmbed();

  type Row = { tier: string; ranker: RankerName; questions: number; hit: Record<number, number>; recall: Record<number, number>; precision: Record<number, number>; queryMs: number[]; buildMs: number };
  const rows = new Map<string, Row>();
  const perQuestion: unknown[] = [];

  for (const tier of spec.tiers) {
    if (args.tiers !== undefined && !args.tiers.has(tier.name)) continue;
    for (const documentId of tier.documents ?? []) {
      const source = byId.get(documentId)!;
      const pages = pagesFromTextFile(`.cache/document-recall/${documentId}.pages.txt`);
      const sessions = windowsAsSessions(windowPages(pages, { pagesPerWindow: 1, maxBytes: 12_288 }));
      const questions = (source.questions ?? []).filter((q) => q.answer !== null && q.evidencePages.length > 0);
      for (const name of args.rankers) {
        const started = performance.now();
        const ranker = await buildDocumentRanker(name, sessions, {
          embed,
          denseCachePath: `.cache/document-recall/${documentId}.dense.nomic.json`,
          ...(nouls === undefined ? {} : { nouls }),
          ...(decomposer === undefined ? {} : { decomposer }),
          ...(name.includes('facts') ? { factsByPage: loadFactsByPage(documentId) } : {}),
        });
        const buildMs = performance.now() - started;
        const key = `${tier.name}\u0000${name}`;
        const row = rows.get(key) ?? { tier: tier.name, ranker: name, questions: 0, hit: { 4: 0, 12: 0 }, recall: { 4: 0, 12: 0 }, precision: { 4: 0, 12: 0 }, queryMs: [], buildMs: 0 };
        row.buildMs += buildMs;
        await mapConcurrent(questions, args.concurrency, async (question) => {
          const t = performance.now();
          const ranked = await ranker.rank(question.question, Math.max(...DEPTHS));
          row.queryMs.push(performance.now() - t);
          row.questions += 1;
          const record: Record<string, unknown> = { tier: tier.name, document: documentId, ranker: name, id: question.id, evidencePages: question.evidencePages };
          for (const depth of DEPTHS) {
            const score = scoreRetrieval(ranked.slice(0, depth).map((id) => ({ id })), question.evidencePages);
            row.hit[depth]! += score.evidenceHit ? 1 : 0;
            row.recall[depth]! += score.pageRecall;
            row.precision[depth]! += score.windowPrecision;
            record[`hit${depth}`] = score.evidenceHit;
            record[`recall${depth}`] = score.pageRecall;
          }
          record.top = ranked.slice(0, 12);
          perQuestion.push(record);
        });
        rows.set(key, row);
        process.stderr.write(`${tier.name} ${documentId} ${name}: built in ${(buildMs / 1000).toFixed(1)}s\n`);
      }
    }
  }

  const pct = (a: number, n: number) => `${((100 * a) / (n || 1)).toFixed(1)}%`;
  const median = (xs: number[]) => {
    const sorted = [...xs].sort((a, b) => a - b);
    return sorted.length === 0 ? 0 : sorted[Math.floor(sorted.length / 2)]!;
  };
  console.log('\ntier  | ranker             | Qs | hit@4  | recall@4 | hit@12 | recall@12 | prec@12 | build   | query p50');
  console.log('----- | ------------------ | -- | ------ | -------- | ------ | --------- | ------- | ------- | ---------');
  for (const row of rows.values()) {
    console.log(
      [
        row.tier.padEnd(5),
        row.ranker.padEnd(18),
        String(row.questions).padStart(2),
        pct(row.hit[4]!, row.questions).padStart(6),
        pct(row.recall[4]!, row.questions).padStart(8),
        pct(row.hit[12]!, row.questions).padStart(6),
        pct(row.recall[12]!, row.questions).padStart(9),
        pct(row.precision[12]!, row.questions).padStart(7),
        `${(row.buildMs / 1000).toFixed(1)}s`.padStart(7),
        `${median(row.queryMs).toFixed(0)}ms`.padStart(9),
      ].join(' | '),
    );
  }
  if (needsRerank) console.log(`\nTypeSafe: ${typesafeTokens} new input tokens, $${typesafeCostUsd(typesafeTokens).toFixed(4)}`);
  if (args.output !== undefined) {
    mkdirSync(dirname(resolve(args.output)), { recursive: true });
    writeFileSync(
      args.output,
      `${JSON.stringify({ ranAt: new Date().toISOString(), depths: DEPTHS, rows: [...rows.values()], perQuestion }, null, 1)}\n`,
    );
    console.log(`wrote ${args.output}`);
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
