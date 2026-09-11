#!/usr/bin/env node
/**
 * Regenerate docs/research/RUN-MATRIX.md: every training run and every LongMemEval
 * evaluation run, with model, data, cost and accuracy, in one place.
 *
 *   node scripts/run-matrix.mjs
 *
 * Sources: docs/research/run-matrix/training-runs.json (hand-kept: platform, minutes, cost,
 * held-out loss, notes) joined to the benchmark result files in docs/research/results by
 * alias; and every longmemeval-*.json result file, read directly (accuracy, recall, models,
 * settings, provider costs). Facts the result files do not record for older runs are in
 * docs/research/run-matrix/longmemeval-notes.json, keyed by file name.
 */
import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const resultsDir = join(root, 'docs', 'research', 'results');
const matrixDir = join(root, 'docs', 'research', 'run-matrix');
const out = join(root, 'docs', 'research', 'RUN-MATRIX.md');

const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));
const training = readJson(join(matrixDir, 'training-runs.json')).runs;
const notesPath = join(matrixDir, 'longmemeval-notes.json');
const lmeNotes = existsSync(notesPath) ? readJson(notesPath) : {};
const files = readdirSync(resultsDir);

const money = (v) => (v === null || v === undefined || Number.isNaN(v) ? '–' : `$${Number(v).toFixed(2)}`);
const pct = (v) => (v === null || v === undefined ? '–' : `${(v * 100).toFixed(1)}%`);

// ---- training runs -------------------------------------------------------------------

function queryScore(alias) {
  if (!alias) return '–';
  const file = files.find((f) => f === `agent-boundary-v2-${alias}-remembero-closure-summary.json`);
  if (!file) return '–';
  const r = readJson(join(resultsDir, file));
  const total = r.summary?.totals?.['remembero-closure']?.queryMean;
  return total === undefined ? '–' : `${Math.round(total)}/31`;
}

function extractionScore(alias) {
  if (!alias) return '–';
  const file = files.find((f) => f === `extraction-bench-v1-${alias}-closed.json`);
  if (!file) return '–';
  const r = readJson(join(resultsDir, file));
  const s = r.runs?.[0]?.score;
  return s ? `${Math.round(s.accuracy * s.cases)}/${s.cases} (${(s.accuracy * 100).toFixed(1)}%)` : '–';
}

const trainingRows = training.map((t) => [
  t.run,
  t.date,
  t.base,
  t.platform,
  t.tasks,
  t.data,
  t.minutes === null ? '–' : `${t.minutes}`,
  `${money(t.costUsd)}${t.costNote ? ` ${t.costNote}` : ''}`,
  t.heldoutLoss === null ? '–' : t.heldoutLoss.toFixed(4),
  queryScore(t.queryAlias),
  extractionScore(t.extractionAlias),
  t.notes,
]);

const trainingCost = training.reduce((a, t) => a + (t.costUsd ?? 0), 0);

// ---- LongMemEval runs -----------------------------------------------------------------

const lmeFiles = files.filter((f) => /^longmemeval-(extraction|ms|tr)-/.test(f)).sort();
const lmeRows = lmeFiles.map((file) => {
  const r = readJson(join(resultsDir, file));
  const s = r.summary ?? {};
  const note = lmeNotes[file] ?? {};
  const settings = r.settings ?? {};
  const extraction = s.extractionUsage ?? {};
  const readerCost = s.readerUsage?.costUsd ?? 0;
  const judgeCost = s.judgeUsage?.costUsd ?? 0;
  const embedCost = s.embeddingUsage?.costUsd ?? 0;
  const extractionCost = extraction.costUsd ?? 0;
  const extractionCalls = extraction.calls ?? 0;
  const gpuNote = extractionCalls > 0 ? ` + L4 ≈${money((extractionCalls / 12500) * 1.6)}` : '';
  const reader = settings.aggregationReaderModel
    ? `${r.readerModel} / ${settings.aggregationReaderModel} (agg)`
    : (note.reader ?? r.readerModel);
  const knobs = [
    r.formation,
    r.extractionModel ? `extractor ${r.extractionModel.replace('finetune/', '')}` : null,
    settings.hybridRetrieval && settings.hybridRetrieval !== 'shared' ? settings.hybridRetrieval : null,
    settings.retrievalUnit === 'turn' ? 'turn units' : null,
    settings.entityRetrieval ? 'entity retrieval' : null,
    settings.temporalRangeModel ? 'time range' : null,
    settings.readingStrategy && settings.readingStrategy !== 'direct' ? `reading ${settings.readingStrategy}` : null,
    settings.hybridQuestionTypes ? `routed: ${settings.hybridQuestionTypes.join(',')}` : null,
    r.retrieval === 'remembero-adaptive-source-search' ? 'semantic route' : 'lexical only',
    `k ${r.topK}/${r.multiSessionTopK}/${r.temporalTopK}`,
    note.knobs ?? null,
  ].filter(Boolean).join('; ');
  const sel = r.dataset?.selection ?? '–';
  const types = r.byQuestionType ?? {};
  const t = (k) => (types[k] ? `${types[k].correct}/${types[k].questions}` : '–');
  return [
    (r.generatedAt ?? '').slice(0, 10),
    file.replace(/^longmemeval-/, '').replace(/\.json$/, ''),
    `${sel} (${s.questions ?? '–'})`,
    reader,
    knobs,
    `${s.correct ?? '–'}/${s.questions ?? '–'} (${pct(s.accuracy)})`,
    pct(s.retrievalRecallAtK),
    t('multi-session'),
    t('temporal-reasoning'),
    t('knowledge-update'),
    `${money(readerCost + judgeCost + embedCost + extractionCost)}${gpuNote}`,
    `${s.errors ?? 0}`,
    note.note ?? '',
  ];
});

// ---- render ---------------------------------------------------------------------------

const table = (headers, rows) =>
  [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map((row) => `| ${row.map((c) => String(c).replace(/\|/g, '\\|')).join(' | ')} |`),
  ].join('\n');

const md = `# Run matrix

Generated by \`node scripts/run-matrix.mjs\` from the result files in \`docs/research/results\`
and the hand-kept facts in \`docs/research/run-matrix/\`. Regenerate after every run; do not
edit by hand.

## Training runs

Rank-32 LoRA, one epoch, learning rate 2e-4 throughout. Query is the agent-boundary benchmark
(query-correct out of 31, closure condition, seed 7); extraction is the schema-conditioned
extraction benchmark (closed vocabulary, 103 cases; v1.1 fixed four gold inputs on 2026-09-09,
so rounds before r12 were scored on v1). Costs: Tinker at its per-token rate, Modal at
H100 $0.001097/s over the run's wall time. Total training spend so far: **${money(trainingCost)}**
(approximate for the Tinker rounds).

${table(
  ['run', 'date', 'base', 'platform', 'tasks', 'data', 'min', 'cost', 'held-out loss', 'query /31', 'extraction', 'notes'],
  trainingRows,
)}

## LongMemEval-S answer runs

Reader and judge on OpenRouter unless the reader is an Ollama Cloud model (\`:cloud\` tag, via
the local daemon, covered by the subscription and shown as $0). Extraction runs on a Modal L4
at about $0.80/hour; the "L4 ≈" figure is that time, and a run whose sessions replayed from
the extraction cache made no extraction calls. Errors are provider timeouts or rate limits,
counted as wrong.

${table(
  ['date', 'result file', 'split (n)', 'reader', 'formation and knobs', 'accuracy', 'recall', 'multi', 'temporal', 'k-update', 'provider cost', 'errors', 'note'],
  lmeRows,
)}
`;

writeFileSync(out, md);
console.log(`wrote ${out}: ${trainingRows.length} training runs, ${lmeRows.length} LongMemEval runs`);
