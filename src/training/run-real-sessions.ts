/**
 * Real-session extraction: label with the frontier model, measure a small model, export
 * training data. See real-sessions.ts for what is selected and why it cannot leak.
 *
 *   node dist/training/run-real-sessions.js label   --count 3400 --out data/real/labels.jsonl
 *        [--model glm-5.3-flash:cloud --base-url http://127.0.0.1:11434/v1 --max-facts 8]
 *   node dist/training/run-real-sessions.js measure --labels data/real/labels.jsonl \
 *        --model finetune/... --base-url https://.../v1 --offset 3100 --count 300 \
 *        --out docs/research/results/extraction-recall-v1-<model>.json
 *   node dist/training/run-real-sessions.js export  --labels data/real/labels.jsonl \
 *        --train-count 3000 --heldout-count 100 --base data/training-r14 --out data/training-r17
 *
 * `label` and `measure` resume: sessions already in the output are skipped.
 */
import { createHash } from 'node:crypto';
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadEnv } from '../env.js';
import { loadLongMemEvalS } from '../evals/longmemeval.js';
import { longMemEvalSplit } from '../evals/longmemeval-semantic.js';
import { OpenRouterClient, type ChatMessage } from '../llm/client.js';
import { rememberTranscriptText } from '../llm/pipeline.js';
import { MemoryStore } from '../store/store.js';
import { createRng } from './rng.js';
import {
  acceptDistilled,
  assembleHaystack,
  parseQuestionReply,
  parseTypeWeights,
  pickType,
  predicateGroups,
  questionWriterPrompt,
  readerMessages,
  toDistilledConversation,
  type DistilledExample,
} from './reader-distill.js';
import type { Rng } from './rng.js';
import {
  generateReaderExamples,
  rewritePrompt,
  rewriteQuestions,
  toReaderConversation,
  type LabelledSession,
  type ReaderExample,
} from './reader-data.js';
import {
  compareFactSets,
  realTranscript,
  selectTrainingSessions,
  toRealConversation,
  type SelectedSession,
} from './real-sessions.js';

interface LabelRow {
  id: string;
  date: string;
  model: string;
  facts: string[];
  error?: string;
  promptTokens?: number | null;
  completionTokens?: number | null;
}

function flag(name: string, fallback?: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] !== undefined
    ? process.argv[index + 1]
    : fallback;
}

function readRows(path: string): Map<string, LabelRow> {
  const rows = new Map<string, LabelRow>();
  if (!existsSync(path)) return rows;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as LabelRow;
    rows.set(row.id, row);
  }
  return rows;
}

async function orderedSessions(seed: number): Promise<SelectedSession[]> {
  const data = flag('--data', '.cache/longmemeval/longmemeval_s_cleaned.json')!;
  const { instances } = await loadLongMemEvalS(resolve(data));
  const chosen = selectTrainingSessions(instances, longMemEvalSplit);
  chosen.sort((a, b) => a.id.localeCompare(b.id));
  return createRng(seed).shuffle(chosen);
}

/** Run the product's transcript extraction on one session with a fresh, empty store. */
/**
 * Capped, atomic labels: the labeller is asked for at most `maxFacts` durable
 * facts with short canonical atoms, and the result is cut to that many. Luna's
 * uncapped labels (6.8 per session, verbose atoms) taught r17 to over-write.
 */
function cappedMessages(
  messages: ChatMessage[],
  maxFacts: number,
): ChatMessage[] {
  const last = messages.at(-1);
  if (last === undefined || last.role !== 'user') return messages;
  return [
    ...messages.slice(0, -1),
    {
      role: 'user',
      content: `${last.content}\n\nReturn at most ${maxFacts} facts: the most durable ones about the user, one relation each, with short lowercase atoms (a name or a one-or-two-word noun), never a sentence as an argument. Skip transient details of the task at hand.`,
    },
  ];
}

async function extractOne(
  client: OpenRouterClient,
  session: SelectedSession,
  maxTokens: number,
  root: string,
  maxFacts?: number,
): Promise<LabelRow> {
  let promptTokens: number | null = 0;
  let completionTokens: number | null = 0;
  const llm = {
    complete: async (messages: ChatMessage[]): Promise<string> => {
      const completion = await client.completeWithUsage(
        maxFacts === undefined ? messages : cappedMessages(messages, maxFacts),
        { maxTokens },
      );
      promptTokens =
        promptTokens === null || completion.usage.promptTokens === null
          ? null
          : promptTokens + completion.usage.promptTokens;
      completionTokens =
        completionTokens === null || completion.usage.completionTokens === null
          ? null
          : completionTokens + completion.usage.completionTokens;
      return completion.content;
    },
  };
  const store = new MemoryStore(mkdtempSync(join(root, 'real-')));
  try {
    const result = await rememberTranscriptText(
      { store, llm },
      realTranscript(session.session),
      'real',
      { captureId: session.id, opId: session.id, sourceText: 'real session' },
    );
    return {
      id: session.id,
      date: session.date,
      model: client.model,
      facts:
        maxFacts === undefined ? result.added : result.added.slice(0, maxFacts),
      promptTokens,
      completionTokens,
    };
  } catch (error) {
    return {
      id: session.id,
      date: session.date,
      model: client.model,
      facts: [],
      error:
        error instanceof Error ? error.message.slice(0, 200) : String(error),
      promptTokens,
      completionTokens,
    };
  }
}

async function runPool<T>(
  items: readonly T[],
  concurrency: number,
  work: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, async () => {
      while (next < items.length) {
        const index = next++;
        await work(items[index]!, index);
      }
    }),
  );
}

async function label(): Promise<void> {
  const out = flag('--out', 'data/real/labels.jsonl')!;
  const count = Number(flag('--count', '3400'));
  const offset = Number(flag('--offset', '0'));
  const seed = Number(flag('--seed', '7'));
  const concurrency = Number(flag('--concurrency', '8'));
  const maxTokens = Number(flag('--max-tokens', '4096'));
  const maxFactsFlag = flag('--max-facts');
  const maxFacts =
    maxFactsFlag === undefined ? undefined : Number(maxFactsFlag);
  const apiKey = flag('--api-key', process.env.LLM_API_KEY);
  if (!apiKey) throw new Error('LLM_API_KEY is not set');
  const client = new OpenRouterClient({
    apiKey,
    baseUrl: (
      flag('--base-url', process.env.LLM_BASE_URL) ??
      'https://openrouter.ai/api/v1'
    ).replace(/\/$/, ''),
    model: flag('--model', 'openai/gpt-5.6-luna')!,
  });
  mkdirSync(dirname(out), { recursive: true });
  const done = readRows(out);
  const sessions = (await orderedSessions(seed)).slice(offset, offset + count);
  const todo = sessions.filter((s) => !done.has(s.id));
  console.error(
    `labelling ${todo.length} of ${sessions.length} sessions with ${client.model} (${done.size} already done)`,
  );
  const root = mkdtempSync(join(tmpdir(), 'rembero-real-'));
  let finished = 0;
  await runPool(todo, concurrency, async (session) => {
    const row = await extractOne(client, session, maxTokens, root, maxFacts);
    appendFileSync(out, `${JSON.stringify(row)}\n`);
    finished += 1;
    if (finished % 50 === 0) console.error(`[${finished}/${todo.length}]`);
  });
  const rows = [...readRows(out).values()];
  const withFacts = rows.filter((r) => r.facts.length > 0).length;
  const errors = rows.filter((r) => r.error).length;
  console.log(
    JSON.stringify({
      sessions: rows.length,
      withFacts,
      errors,
      facts: rows.reduce((n, r) => n + r.facts.length, 0),
    }),
  );
}

async function measure(): Promise<void> {
  const labelsPath = flag('--labels', 'data/real/labels.jsonl')!;
  const out = flag('--out')!;
  const count = Number(flag('--count', '300'));
  const offset = Number(flag('--offset', '3100'));
  const seed = Number(flag('--seed', '7'));
  const concurrency = Number(flag('--concurrency', '6'));
  const maxTokens = Number(flag('--max-tokens', '512'));
  const selfAtom = flag('--self', 'user')!;
  const apiKey = flag(
    '--api-key',
    process.env.EXTRACTION_API_KEY ??
      process.env.MODAL_SERVE_API_KEY ??
      process.env.LLM_API_KEY,
  );
  if (!apiKey) throw new Error('no api key');
  const client = new OpenRouterClient({
    apiKey,
    baseUrl: (
      flag('--base-url', process.env.LLM_BASE_URL) ??
      'https://openrouter.ai/api/v1'
    ).replace(/\/$/, ''),
    model: flag('--model')!,
  });
  const labels = readRows(labelsPath);
  const sessions = (await orderedSessions(seed))
    .slice(offset, offset + count)
    .filter((s) => labels.has(s.id) && !labels.get(s.id)!.error);
  console.error(
    `measuring ${client.model} on ${sessions.length} labelled sessions`,
  );
  const root = mkdtempSync(join(tmpdir(), 'rembero-real-measure-'));
  const perSession: Array<
    {
      id: string;
      reference: string[];
      model: string[];
      error?: string;
    } & ReturnType<typeof compareFactSets>
  > = [];
  await runPool(sessions, concurrency, async (session) => {
    const row = await extractOne(client, session, maxTokens, root);
    const reference = labels.get(session.id)!.facts;
    perSession.push({
      id: session.id,
      reference,
      model: row.facts,
      ...(row.error ? { error: row.error } : {}),
      ...compareFactSets(reference, row.facts, selfAtom),
    });
  });
  const withReference = perSession.filter((s) => s.referenceFacts > 0);
  const detected = withReference.filter((s) => s.modelFacts > 0).length;
  const silentReference = perSession.filter((s) => s.referenceFacts === 0);
  const falseAlarms = silentReference.filter((s) => s.modelFacts > 0).length;
  const mean = (xs: number[]) =>
    xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
  const summary = {
    model: client.model,
    referenceModel: [...labels.values()][0]?.model ?? null,
    sessions: perSession.length,
    sessionsWithReferenceFacts: withReference.length,
    sessionDetectionRate:
      withReference.length === 0 ? 0 : detected / withReference.length,
    silentReferenceSessions: silentReference.length,
    falseAlarmRate:
      silentReference.length === 0 ? 0 : falseAlarms / silentReference.length,
    meanFactRecall: mean(withReference.map((s) => s.recall)),
    meanFactPrecision: mean(
      perSession.filter((s) => s.modelFacts > 0).map((s) => s.precision),
    ),
    referenceFacts: perSession.reduce((n, s) => n + s.referenceFacts, 0),
    modelFacts: perSession.reduce((n, s) => n + s.modelFacts, 0),
    errors: perSession.filter((s) => s.error).length,
  };
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify({ summary, perSession }, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
}

async function exportData(): Promise<void> {
  const labelsPath = flag('--labels', 'data/real/labels.jsonl')!;
  const trainCount = Number(flag('--train-count', '3000'));
  const heldoutCount = Number(flag('--heldout-count', '100'));
  const seed = Number(flag('--seed', '7'));
  const selfAtom = flag('--self', 'user')!;
  const base = flag('--base', 'data/training-r14')!;
  const out = flag('--out', 'data/training-r17')!;
  const labels = readRows(labelsPath);
  const ordered = (await orderedSessions(seed)).filter(
    (s) => labels.has(s.id) && !labels.get(s.id)!.error,
  );
  const train = ordered.slice(0, trainCount);
  const heldout = ordered.slice(trainCount, trainCount + heldoutCount);
  const lines = (sessions: SelectedSession[]) => {
    // a real store is mostly silence; cap "% nothing" rows at the number of rows with facts so
    // the model learns to write, not to stay quiet
    const withFacts = sessions.filter(
      (s) => labels.get(s.id)!.facts.length > 0,
    );
    const silent = sessions
      .filter((s) => labels.get(s.id)!.facts.length === 0)
      .slice(0, withFacts.length);
    return [...withFacts, ...silent].map((s) =>
      JSON.stringify(
        toRealConversation(s.session, labels.get(s.id)!.facts, selfAtom),
      ),
    );
  };
  const trainLines = lines(train);
  const heldoutLines = lines(heldout);
  mkdirSync(out, { recursive: true });
  const baseTrain = readFileSync(join(base, 'conversations.jsonl'), 'utf8')
    .trim()
    .split('\n');
  const baseHeld = readFileSync(join(base, 'heldout.jsonl'), 'utf8')
    .trim()
    .split('\n');
  const merged = createRng(seed).shuffle([...baseTrain, ...trainLines]);
  writeFileSync(join(out, 'conversations.jsonl'), `${merged.join('\n')}\n`);
  writeFileSync(
    join(out, 'heldout.jsonl'),
    `${[...baseHeld, ...heldoutLines].join('\n')}\n`,
  );
  const manifest = {
    base,
    labels: labelsPath,
    realTrain: trainLines.length,
    realTrainWithFacts: train.filter((s) => labels.get(s.id)!.facts.length > 0)
      .length,
    realHeldout: heldoutLines.length,
    train: merged.length,
    heldout: baseHeld.length + heldoutLines.length,
    generatedAt: new Date().toISOString(),
  };
  writeFileSync(
    join(out, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  console.log(JSON.stringify(manifest, null, 2));
}

/**
 * Reader training data: deterministic questions and gold answers over the labelled
 * sessions, rendered the way the evaluation renders a reader prompt. Whole
 * haystacks stay together on one side of the split: the first `--train-count`
 * sessions of the ordered list train, the rest are held out.
 */
async function exportReader(): Promise<void> {
  const labelsPath = flag('--labels', 'data/real/labels-glmflash8.jsonl')!;
  const out = flag('--out', 'data/training-reader')!;
  const trainCount = Number(flag('--train-count', '3000'));
  const perType = Number(flag('--per-type', '600'));
  const seed = Number(flag('--seed', '7'));
  const labels = readRows(labelsPath);
  const ordered = (await orderedSessions(seed)).filter(
    (s) => labels.has(s.id) && !labels.get(s.id)!.error,
  );
  const toLabelled = (s: SelectedSession): LabelledSession => ({
    id: s.id,
    date: s.date.slice(0, 10).replace(/\//g, '-'),
    facts: labels.get(s.id)!.facts,
    transcript: realTranscript(s.session),
  });
  const train = ordered.slice(0, trainCount).map(toLabelled);
  const held = ordered.slice(trainCount).map(toLabelled);
  const trainExamples = generateReaderExamples(train, createRng(seed * 7 + 1), {
    perType,
  });
  const heldExamples = generateReaderExamples(held, createRng(seed * 7 + 2), {
    perType: Math.max(10, Math.floor(perType / 20)),
  });
  mkdirSync(out, { recursive: true });
  // optional natural-language rewrite of the templated questions, cached by question text
  const rewriteModel = flag('--rewrite-model');
  let rewrites: { rewritten: number; kept: number } | undefined;
  if (rewriteModel !== undefined) {
    const apiKey = flag('--api-key', process.env.LLM_API_KEY);
    if (!apiKey) throw new Error('LLM_API_KEY is not set');
    const client = new OpenRouterClient({
      apiKey,
      baseUrl: (
        flag('--base-url', process.env.LLM_BASE_URL) ??
        'https://openrouter.ai/api/v1'
      ).replace(/\/$/, ''),
      model: rewriteModel,
    });
    const cacheDir = join(out, 'rewrite-cache');
    mkdirSync(cacheDir, { recursive: true });
    const rewriter = {
      rewrite: async (example: ReaderExample): Promise<string> => {
        const key = createHash('sha256')
          .update(`${rewriteModel}\n${example.question}`)
          .digest('hex');
        const path = join(cacheDir, `${key}.txt`);
        if (existsSync(path)) return readFileSync(path, 'utf8');
        const completion = await client.completeWithUsage(
          [{ role: 'user', content: rewritePrompt(example) }],
          { maxTokens: 120 },
        );
        writeFileSync(path, completion.content);
        return completion.content;
      },
    };
    const a = await rewriteQuestions(trainExamples, rewriter);
    const b = await rewriteQuestions(heldExamples, rewriter);
    rewrites = { rewritten: a.rewritten + b.rewritten, kept: a.kept + b.kept };
  }
  const lines = (examples: ReaderExample[]) =>
    examples.map((e) => JSON.stringify(toReaderConversation(e))).join('\n');
  writeFileSync(join(out, 'conversations.jsonl'), `${lines(trainExamples)}\n`);
  writeFileSync(join(out, 'heldout.jsonl'), `${lines(heldExamples)}\n`);
  const byType = (examples: ReaderExample[]) =>
    examples.reduce<Record<string, number>>((acc, e) => {
      acc[e.type] = (acc[e.type] ?? 0) + 1;
      return acc;
    }, {});
  const manifest = {
    labels: labelsPath,
    seed,
    trainSessions: train.length,
    heldoutSessions: held.length,
    train: trainExamples.length,
    heldout: heldExamples.length,
    byType: byType(trainExamples),
    ...(rewrites === undefined ? {} : { rewriteModel, rewrites }),
    generatedAt: new Date().toISOString(),
  };
  writeFileSync(
    join(out, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  console.log(JSON.stringify(manifest, null, 2));
}

/** "yes" / "no" at the start of a judge reply; anything else is undecided. */
export function parseSupportVerdict(reply: string): boolean | undefined {
  const head = reply.trim().toLowerCase().slice(0, 4);
  if (head.startsWith('yes')) return true;
  if (head.startsWith('no')) return false;
  return undefined;
}

/**
 * Judged precision. `measure` scores a model fact as precise only when it shares a
 * constant with a reference fact, so a true fact the capped labeller skipped, or a
 * paraphrase (sustainably_sourced_products vs sustainable_products), counts as
 * imprecise. This samples the unmatched model facts and asks a judge whether the
 * transcript states or clearly implies each one, giving the precision the loose
 * match understates.
 */
async function judgeUnmatched(): Promise<void> {
  const resultsPath = flag('--results')!;
  const sample = Number(flag('--sample', '150'));
  const seed = Number(flag('--seed', '7'));
  const model = flag('--model', 'z-ai/glm-5.3-flash')!;
  const apiKey = flag('--api-key', process.env.LLM_API_KEY);
  if (!apiKey) throw new Error('LLM_API_KEY is not set');
  const client = new OpenRouterClient({
    apiKey,
    baseUrl: (
      flag('--base-url', process.env.LLM_BASE_URL) ??
      'https://openrouter.ai/api/v1'
    ).replace(/\/$/, ''),
    model,
  });
  const results = JSON.parse(readFileSync(resultsPath, 'utf8')) as {
    summary: { model: string };
    perSession: Array<{ id: string; reference: string[]; model: string[] }>;
  };
  const sessions = new Map(
    (await orderedSessions(seed)).map((s) => [s.id, s] as const),
  );
  // unmatched = model facts sharing no non-self constant with any reference fact
  const constants = (fact: string) =>
    (fact.match(/\(([^)]*)\)/)?.[1] ?? '')
      .split(',')
      .map((a) => a.trim().replace(/^'|'$/g, '').toLowerCase())
      .filter((a) => a.length > 0 && a !== 'user');
  const candidates: Array<{ id: string; fact: string }> = [];
  for (const row of results.perSession) {
    const referenceConstants = new Set(row.reference.flatMap(constants));
    for (const fact of row.model) {
      if (!constants(fact).some((c) => referenceConstants.has(c))) {
        candidates.push({ id: row.id, fact });
      }
    }
  }
  const chosen = createRng(seed).shuffle(candidates).slice(0, sample);
  let supported = 0;
  let unsupported = 0;
  let undecided = 0;
  const rows: Array<{
    id: string;
    fact: string;
    verdict: boolean | undefined;
  }> = [];
  await runPool(chosen, 8, async ({ id, fact }) => {
    const session = sessions.get(id);
    if (session === undefined) return;
    const transcript = realTranscript(session.session).slice(0, 12_000);
    const completion = await client.completeWithUsage(
      [
        {
          role: 'user',
          content: `Below is a chat transcript and one fact a memory system extracted about the user ("user" is the person talking to the assistant). Does the transcript state or clearly imply this fact? Answer "yes" or "no" first, then one short reason.\n\nFact: ${fact}\n\nTranscript:\n${transcript}`,
        },
      ],
      { maxTokens: 400 },
    );
    const verdict = parseSupportVerdict(completion.content);
    if (verdict === true) supported += 1;
    else if (verdict === false) unsupported += 1;
    else undecided += 1;
    rows.push({ id, fact, verdict });
  });
  const out = {
    model: results.summary.model,
    judge: model,
    unmatchedFacts: candidates.length,
    judged: rows.length,
    supported,
    unsupported,
    undecided,
    supportedShare:
      rows.length === 0 ? null : supported / (supported + unsupported),
    rows,
  };
  const outPath = flag('--out');
  if (outPath !== undefined)
    writeFileSync(outPath, `${JSON.stringify(out, null, 2)}\n`);
  console.log(JSON.stringify({ ...out, rows: undefined }));
}

/**
 * Reader v3 data: teacher-distilled. For each example, assemble a haystack of real
 * sessions, have the teacher write a question of a drawn type, then answer it through
 * the evaluation's reader prompt; keep the pair when the answer is consistent with
 * the type (an abstention question got an abstention, any other got an answer).
 * Resumable: examples are appended to the output as they finish.
 */
async function distillReader(): Promise<void> {
  const labelsPath = flag('--labels', 'data/real/labels-glmflash8.jsonl')!;
  const out = flag('--out', 'data/training-reader-v3')!;
  if (process.argv.includes('--computed-notes')) process.env.REMEMBERO_READER_COMPUTED_NOTES = '1';
  const trainCount = Number(flag('--train-count', '3000'));
  const target = Number(flag('--examples', '4000'));
  const heldoutTarget = Number(flag('--heldout-examples', '200'));
  const seed = Number(flag('--seed', '7'));
  const concurrency = Number(flag('--concurrency', '8'));
  const model = flag('--model', 'z-ai/glm-5.3-flash')!;
  const apiKey = flag('--api-key', process.env.LLM_API_KEY);
  if (!apiKey) throw new Error('LLM_API_KEY is not set');
  const client = new OpenRouterClient({
    apiKey,
    baseUrl: (
      flag('--base-url', process.env.LLM_BASE_URL) ??
      'https://openrouter.ai/api/v1'
    ).replace(/\/$/, ''),
    model,
  });
  const labels = readRows(labelsPath);
  const ordered = (await orderedSessions(seed)).filter(
    (s) => labels.has(s.id) && !labels.get(s.id)!.error,
  );
  const toLabelled = (s: SelectedSession): LabelledSession => ({
    id: s.id,
    date: s.date.slice(0, 10).replace(/\//g, '-'),
    facts: labels.get(s.id)!.facts,
    transcript: realTranscript(s.session),
  });
  const trainPool = ordered.slice(0, trainCount).map(toLabelled);
  const heldPool = ordered.slice(trainCount).map(toLabelled);
  // --type-weights multi-session=40,temporal-reasoning=35,… reweights the draw; multi-session
  // and knowledge-update haystacks are seeded with sessions that carry the material
  const weightsFlag = flag('--type-weights');
  const weights =
    weightsFlag === undefined ? undefined : parseTypeWeights(weightsFlag);
  mkdirSync(out, { recursive: true });
  const stats = {
    attempted: 0,
    kept: 0,
    noQuestion: 0,
    rejected: 0,
    errors: 0,
  };
  const byType: Record<string, number> = {};
  const trainGroups = predicateGroups(trainPool);
  const heldGroups = predicateGroups(heldPool);

  const produce = async (
    pool: LabelledSession[],
    rng: Rng,
    want: number,
    file: string,
  ): Promise<void> => {
    const done = existsSync(file)
      ? readFileSync(file, 'utf8').split('\n').filter(Boolean).length
      : 0;
    let kept = done;
    const tasks = Array.from(
      { length: Math.max(0, want - done) * 2 },
      (_, i) => i,
    );
    await runPool(tasks, concurrency, async () => {
      if (kept >= want) return;
      stats.attempted += 1;
      const type =
        weights === undefined ? pickType(rng) : pickType(rng, weights);
      const haystack = assembleHaystack(pool, rng, {
        seed: type,
        groups: pool === trainPool ? trainGroups : heldGroups,
      });
      try {
        const written = await client.completeWithUsage(
          [{ role: 'user', content: questionWriterPrompt(haystack, type) }],
          // GLM Flash reasons before it answers; the budget must cover the thinking
          { maxTokens: 4_000 },
        );
        const parsed = parseQuestionReply(written.content);
        if (parsed === undefined) {
          stats.noQuestion += 1;
          return;
        }
        const messages = readerMessages(haystack, parsed.question, type);
        const answered = await client.completeWithUsage(messages, {
          maxTokens: 4_000,
        });
        if (!acceptDistilled(type, answered.content)) {
          stats.rejected += 1;
          return;
        }
        if (kept >= want) return;
        kept += 1;
        stats.kept += 1;
        byType[type] = (byType[type] ?? 0) + 1;
        const example: DistilledExample = {
          type,
          question: parsed.question,
          questionDate: haystack.questionDate,
          sessionIds: haystack.sessions.map((s) => s.id),
          evidence: parsed.evidence,
          answer: answered.content.trim(),
          messages,
        };
        appendFileSync(
          file,
          `${JSON.stringify(toDistilledConversation(example))}\n`,
        );
        appendFileSync(
          `${file}.meta.jsonl`,
          `${JSON.stringify({ ...example, messages: undefined })}\n`,
        );
        if (stats.kept % 100 === 0) console.error(`[kept ${stats.kept}]`);
      } catch (error) {
        stats.errors += 1;
        if (stats.errors % 20 === 1)
          console.error(
            `distill error: ${error instanceof Error ? error.message : String(error)}`,
          );
      }
    });
  };
  await produce(
    trainPool,
    createRng(seed * 11 + 1),
    target,
    join(out, 'conversations.jsonl'),
  );
  await produce(
    heldPool,
    createRng(seed * 11 + 2),
    heldoutTarget,
    join(out, 'heldout.jsonl'),
  );
  const manifest = {
    labels: labelsPath,
    teacher: model,
    typeWeights: weightsFlag ?? 'default',
    seed,
    trainSessions: trainPool.length,
    heldoutSessions: heldPool.length,
    train: readFileSync(join(out, 'conversations.jsonl'), 'utf8')
      .split('\n')
      .filter(Boolean).length,
    heldout: existsSync(join(out, 'heldout.jsonl'))
      ? readFileSync(join(out, 'heldout.jsonl'), 'utf8')
          .split('\n')
          .filter(Boolean).length
      : 0,
    byType,
    stats,
    generatedAt: new Date().toISOString(),
  };
  writeFileSync(
    join(out, 'manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  console.log(JSON.stringify(manifest, null, 2));
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];

if (invokedDirectly) {
  loadEnv();
  const command = process.argv[2];
  if (command === 'label') await label();
  else if (command === 'measure') await measure();
  else if (command === 'export') await exportData();
  else if (command === 'reader') await exportReader();
  else if (command === 'judge') await judgeUnmatched();
  else if (command === 'distill') await distillReader();
  else {
    console.error(
      'usage: run-real-sessions.js label|measure|export|reader|judge|distill [flags]',
    );
    process.exit(1);
  }
}
