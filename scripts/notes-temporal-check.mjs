#!/usr/bin/env node
/**
 * Rebuilds the computed-notes block for the 15 reader-v7 temporal misses that had every
 * evidence session in context (the list and the needed facts come from the v7 temporal-miss
 * analysis), and prints, per question, whether the fact the analysis says is needed appears.
 *
 * A check, not a tuning target. Needs `npm run build:core` (it loads dist/) and the dataset at
 * .cache/longmemeval/longmemeval_s_cleaned.json. No model is called.
 *
 *   node scripts/notes-temporal-check.mjs            # one line per question and the count
 *   node scripts/notes-temporal-check.mjs --show     # also print each block
 */
import { existsSync, readFileSync } from 'node:fs';
import { buildComputedNotes } from '../dist/knowledge/computed-notes.js';

const DATASET = '.cache/longmemeval/longmemeval_s_cleaned.json';
const RUN = 'docs/research/results/longmemeval-raw-v7pod-retr-session-15-10-24k-pair2-all500.json';
const show = process.argv.includes('--show');

/** Line-level test: some line of the block matches every pattern. */
const line = (...patterns) => (block) => block.split('\n').some((l) => patterns.every((p) => p.test(l)));
/** The first line after the header matches every pattern. */
const firstLine = (...patterns) => (block) => {
  const first = block.split('\n')[1] ?? '';
  return patterns.every((p) => p.test(first));
};
const verdict = (winner) => line(/^Which came first:/, new RegExp(`→ [^\\n]*${winner}`, 'i'));

const CHECKS = [
  { id: 'gpt4_a1b77f9c', need: "The Power's start (2022-03-06) is shown with its title", test: line(/2022-03-06/, /The Power/) },
  { id: 'gpt4_1916e0ea', need: 'FarmFresh cancellation 2023-01-05 → Instacart 2023-02-28 = 54 days, first', test: firstLine(/2023-01-05/, /2023-02-28/, /54 days/) },
  { id: '370a8ff4', need: 'flu recovery 2023-01-19 → 10th jog 2023-04-10 = 81 days (gold disagrees)', test: line(/2023-01-19/, /2023-04-10/, /81 days/) },
  { id: '8077ef71', need: 'networking event 2022-03-09 = 26 days before the question, and no 2022-05-15 future date', test: (b) => line(/2022-03-09/, /26 days/)(b) && !/2022-05-15/.test(b) },
  { id: '71017277', need: 'the event closest to "last Saturday" (2023-03-04) quotes the chandelier sentence', test: line(/closest dated event/, /2023-03-04 \("[^"]*chandelier/i) },
  { id: 'gpt4_e414231f', need: 'the past weekend covers 2023-03-19 (road bike pedals)', test: line(/weekend/i, /2023-03-19/, /2023-03-18/, /0 days away|inside/) },
  { id: 'gpt4_2312f94c', need: 'verdict: Galaxy S22 (2023-02-20) before the Dell XPS 13 (arrived 2023-02-25)', test: verdict('galaxy') },
  { id: '2c63a862', need: 'started with Rachel 2022-02-15 → saw the house 2022-03-01 = 14 days, first', test: firstLine(/2022-02-15/, /2022-03-01/, /14 days/) },
  { id: '982b5123', need: 'Airbnb booked ≈2022-12-21 (wedding 2023-03-21 minus three months), about 5 months ago', test: line(/2022-12-2\d/, /about 5 months/) },
  { id: 'cc6d1ec1', need: 'bird watching started ≈2023-02-21, workshop 2023-04-21: about 2 months', test: (b) => line(/2023-02-2\d/, /2023-04-2\d/, /(2(\.0)? months|59 days|60 days|61 days)/)(b) },
  { id: 'd01c6aa8', need: 'age at the move to the US ≈ 27', test: line(/\bage\b/i, /\b27\b/) },
  { id: 'gpt4_88806d6e', need: 'verdict: Tom met before Mark and Sarah', test: verdict('tom') },
  { id: 'gpt4_78cf46a3', need: 'verdict: phone case before the charger', test: verdict('case') },
  { id: 'gpt4_c27434e8', need: 'verdict: Japanese Zero before the Ferrari', test: verdict('zero') },
  { id: 'gpt4_fe651585', need: 'verdict: Alex (January) before Rachel (twins born February 12th)', test: verdict('alex') },
];

if (!existsSync(DATASET)) {
  console.error(`dataset missing: ${DATASET}`);
  process.exit(2);
}
const dataset = new Map(JSON.parse(readFileSync(DATASET, 'utf8')).map((x) => [x.question_id, x]));
const run = new Map(JSON.parse(readFileSync(RUN, 'utf8')).observations.map((o) => [o.questionId, o]));

const transcript = (session) =>
  session.map(({ role, content }) => `${role === 'user' ? 'USER' : 'ASSISTANT'}: ${content}`).join('\n\n');

let hits = 0;
for (const check of CHECKS) {
  const instance = dataset.get(check.id);
  const observation = run.get(check.id);
  if (!instance || !observation) {
    console.log(`?? ${check.id}: not in the dataset or the run`);
    continue;
  }
  const byId = new Map(instance.haystack_session_ids.map((id, i) => [id, i]));
  const sources = observation.contextSessionIds
    .filter((id) => byId.has(id))
    .map((id) => {
      const i = byId.get(id);
      return { ts: instance.haystack_dates[i], text: transcript(instance.haystack_sessions[i]) };
    });
  const block = buildComputedNotes(instance.question, instance.question_date, sources);
  const ok = check.test(block);
  if (ok) hits += 1;
  console.log(`${ok ? 'yes' : 'no '} ${check.id}: ${check.need}`);
  if (show) console.log(`    Q (${instance.question_date}): ${instance.question}\n${block.replace(/^/gm, '    | ')}`);
}
console.log(`\n${hits}/${CHECKS.length} blocks now carry the needed fact`);
