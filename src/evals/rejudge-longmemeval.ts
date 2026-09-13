/**
 * Re-judge stored LongMemEval runs with one judge, so every table shares it.
 *
 *   LLM_BASE_URL=https://api.deepseek.com/v1 LLM_API_KEY=… node dist/evals/rejudge-longmemeval.js \
 *     --judge-model deepseek-chat [--files glob] [--concurrency 8]
 *
 * Reads each run's stored hypotheses, judges them again, and writes a sidecar
 * `<run>.rejudged-<judge>.json` with per-question labels and a per-type summary. Originals are
 * never modified; a cache keyed by (question, hypothesis) makes reruns free.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { OpenRouterClient } from '../llm/client.js';
import {
  buildLongMemEvalJudgePrompt,
  parseLongMemEvalJudgeLabel,
} from './longmemeval-answer.js';
import type { LongMemEvalInstance } from './longmemeval.js';

function flag(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

async function main(): Promise<void> {
  const judgeModel = flag('--judge-model', 'deepseek-chat')!;
  const concurrency = Number(flag('--concurrency', '8'));
  const resultsDir = resolve(flag('--results', 'docs/research/results')!);
  const pattern = new RegExp(flag('--files', '^longmemeval-.*\\.json$')!);
  const dataset = JSON.parse(
    readFileSync(resolve(flag('--data', '.cache/longmemeval/longmemeval_s_cleaned.json')!), 'utf8'),
  ) as LongMemEvalInstance[];
  const byId = new Map(dataset.map((d) => [d.question_id, d]));
  const judge = new OpenRouterClient({
    apiKey: process.env.LLM_API_KEY ?? '',
    baseUrl: process.env.LLM_BASE_URL ?? 'https://api.deepseek.com/v1',
    model: judgeModel,
  });
  const cacheDir = resolve('.cache/longmemeval/rejudge-' + judgeModel.replace(/[^a-z0-9]+/gi, '-'));
  mkdirSync(cacheDir, { recursive: true });
  const { readdirSync } = await import('node:fs');
  const files = readdirSync(resultsDir)
    .filter((f) => pattern.test(f) && !f.includes('.rejudged-'))
    .sort();
  const table: Array<Record<string, unknown>> = [];
  let calls = 0;
  for (const file of files) {
    let run: { observations?: Array<{ questionId: string; questionType: string; hypothesis?: string | null; correct?: boolean; status?: string }>; judgeModel?: string };
    try {
      run = JSON.parse(readFileSync(join(resultsDir, file), 'utf8'));
    } catch {
      continue;
    }
    const observations = run.observations;
    if (!Array.isArray(observations) || observations.length === 0 || !('hypothesis' in observations[0]!)) continue;
    const labels: Array<{ questionId: string; questionType: string; original: boolean; rejudged: boolean | null; error?: string }> = [];
    let index = 0;
    const worker = async () => {
      while (index < observations.length) {
        const o = observations[index++]!;
        const instance = byId.get(o.questionId);
        const hypothesis = o.hypothesis;
        if (!instance || hypothesis === undefined || hypothesis === null || o.status === 'error') {
          labels.push({ questionId: o.questionId, questionType: o.questionType, original: Boolean(o.correct), rejudged: null, error: 'no hypothesis' });
          continue;
        }
        const key = createHash('sha256').update(`${o.questionId}\n${hypothesis}`).digest('hex');
        const cachePath = join(cacheDir, `${key}.json`);
        let verdict: boolean | null = null;
        let error: string | undefined;
        if (existsSync(cachePath)) {
          verdict = JSON.parse(readFileSync(cachePath, 'utf8')).verdict;
        } else {
          try {
            const completion = await judge.completeWithUsage(
              [{ role: 'user', content: buildLongMemEvalJudgePrompt(instance, String(hypothesis)) }],
              { maxTokens: 8 },
            );
            calls += 1;
            verdict = parseLongMemEvalJudgeLabel(completion.content);
            writeFileSync(cachePath, JSON.stringify({ verdict }));
          } catch (e) {
            error = e instanceof Error ? e.message : String(e);
          }
        }
        labels.push({ questionId: o.questionId, questionType: o.questionType, original: Boolean(o.correct), rejudged: verdict, ...(error ? { error } : {}) });
      }
    };
    await Promise.all(Array.from({ length: concurrency }, worker));
    const byType: Record<string, { questions: number; original: number; rejudged: number; errors: number }> = {};
    for (const l of labels) {
      const t = (byType[l.questionType] ??= { questions: 0, original: 0, rejudged: 0, errors: 0 });
      t.questions += 1;
      t.original += l.original ? 1 : 0;
      t.rejudged += l.rejudged ? 1 : 0;
      t.errors += l.rejudged === null ? 1 : 0;
    }
    const total = { questions: labels.length, original: labels.filter((l) => l.original).length, rejudged: labels.filter((l) => l.rejudged).length, errors: labels.filter((l) => l.rejudged === null).length };
    const sidecar = join(resultsDir, file.replace(/\.json$/, `.rejudged-${judgeModel.replace(/[^a-z0-9]+/gi, '-')}.json`));
    writeFileSync(sidecar, JSON.stringify({ run: file, originalJudge: run.judgeModel ?? null, judgeModel, total, byType, labels }, null, 2));
    table.push({ run: file, originalJudge: run.judgeModel ?? null, ...total, byType });
    console.log(`${basename(file)}: ${run.judgeModel ?? '?'} ${total.original}/${total.questions} -> ${judgeModel} ${total.rejudged}/${total.questions}${total.errors ? ` (${total.errors} errors)` : ''}`);
  }
  writeFileSync(join(resultsDir, `rejudge-${judgeModel.replace(/[^a-z0-9]+/gi, '-')}-summary.json`), JSON.stringify({ judgeModel, generatedAt: new Date().toISOString(), runs: table }, null, 2));
  console.log(`judge calls: ${calls}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
