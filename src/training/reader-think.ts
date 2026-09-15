/**
 * Thinking regeneration: reader v6's rows re-answered under the thinking prompt.
 *
 * v6's assistant turns are direct answers. Under a thinking contract the multi-session,
 * temporal-reasoning, knowledge-update and abstention prompts ask for notes and a final
 * "Answer:" line, so a stored direct answer under those prompts would teach the student to
 * skip the step it is asked for. This rebuilds each such row's haystack, renders it under
 * the contract, and has the teacher answer again. The reply is kept only when it is
 * well-formed (acceptDistilled) and its final line states the same result as the stored
 * answer, which a judge decides; an abstention row needs only a final line that abstains.
 * A row whose answer moved goes to `disagreements.jsonl` with both answers, not into the
 * training set: v6's answers were already filtered, so a disagreement is more often the new
 * reply's mistake than the old one's.
 *
 * Single-session rows read directly under thinking, so they are re-rendered with their
 * stored answer and no model calls: the prompt is byte-identical to rerenderRow's.
 *
 * Every row's outcome is appended to `progress.jsonl` keyed by file and row index; a rerun
 * skips indices already decided and retries the ones that errored.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseLongMemEvalJudgeLabel } from '../evals/longmemeval-answer.js';
import type { ReaderContract } from '../evals/reader-contract.js';
import type { ChatMessage } from '../llm/client.js';
import type { Conversation } from './export.js';
import {
  acceptDistilled,
  completionAnswer,
  readerMessages,
  thinksOn,
  toDistilledConversation,
} from './reader-distill.js';
import {
  haystackFromMeta,
  type DistilledMetaRow,
  type SessionPool,
} from './reader-rerender.js';
import { runPool } from './run-pool.js';

/** The part of OpenRouterClient the command uses, so tests can stub it. */
export interface CompletionClient {
  completeWithUsage(
    messages: ChatMessage[],
    options?: { maxTokens?: number },
  ): Promise<{ content: string }>;
}

export interface ThinkClients {
  teacher: CompletionClient;
  judge: CompletionClient;
}

export const TEACHER_MAX_TOKENS = 4_000;
export const JUDGE_MAX_TOKENS = 16;

/** The judge's question: do the stored answer and the thinking reply's final line agree? */
export function agreementPrompt(
  question: string,
  storedAnswer: string,
  finalLine: string,
): string {
  return `A user asked their assistant a question about their own chat history, and two answers were written to it. Do the two answers state the same result for the question: the same numbers, dates or named items, or both saying the information is unavailable? Ignore differences in wording, length and explanation.

Question: ${question}

Answer A: ${storedAnswer}

Answer B: ${finalLine}

Answer yes or no only.`;
}

/** A kept row's meta: the thinking reply as the answer, the direct one it replaced beside it. */
export type ThinkMetaRow = DistilledMetaRow & { storedAnswer?: string };

export interface Disagreement {
  type: DistilledMetaRow['type'];
  question: string;
  storedAnswer: string;
  reply: string;
  finalLine: string;
}

export type ThinkRowResult =
  | { outcome: 'kept'; conversation: Conversation; meta: ThinkMetaRow }
  | { outcome: 'rendered'; conversation: Conversation; meta: ThinkMetaRow }
  | { outcome: 'rejectedFormat'; reply: string }
  | { outcome: 'disagreed'; disagreement: Disagreement }
  | { outcome: 'missing'; missing: string[] }
  | { outcome: 'error'; error: string };

export type ThinkOutcome = ThinkRowResult['outcome'];

/** One row through the thinking regeneration. Never throws: a failed call is an error outcome. */
export async function thinkRow(
  row: DistilledMetaRow,
  pool: SessionPool,
  contract: ReaderContract,
  clients: ThinkClients,
  teacherMaxTokens = TEACHER_MAX_TOKENS,
): Promise<ThinkRowResult> {
  const built = haystackFromMeta(row, pool);
  if ('missing' in built) return { outcome: 'missing', missing: built.missing };
  try {
    // the prompt for a type the contract does not think on is the direct one, so the stored
    // answer stays the assistant turn
    const messages = readerMessages(
      built.haystack,
      row.question,
      row.type,
      contract,
    );
    if (!thinksOn(contract.thinking, row.type))
      return {
        outcome: 'rendered',
        conversation: toDistilledConversation({ ...row, messages }),
        meta: row,
      };
    const answered = await clients.teacher.completeWithUsage(messages, {
      maxTokens: teacherMaxTokens,
    });
    const reply = answered.content.trim();
    if (!acceptDistilled(row.type, reply, contract.thinking))
      return { outcome: 'rejectedFormat', reply };
    const finalLine = completionAnswer(reply, contract.thinking, row.type);
    // an accepted abstention row abstained on its final line; that is the whole agreement
    if (row.type !== 'abstention') {
      const verdict = await clients.judge.completeWithUsage(
        [
          {
            role: 'user',
            content: agreementPrompt(row.question, row.answer, finalLine),
          },
        ],
        { maxTokens: JUDGE_MAX_TOKENS },
      );
      if (!parseLongMemEvalJudgeLabel(verdict.content))
        return {
          outcome: 'disagreed',
          disagreement: {
            type: row.type,
            question: row.question,
            storedAnswer: row.answer,
            reply,
            finalLine,
          },
        };
    }
    return {
      outcome: 'kept',
      conversation: toDistilledConversation({
        ...row,
        answer: reply,
        messages,
      }),
      meta: { ...row, answer: reply, storedAnswer: row.answer },
    };
  } catch (error) {
    return {
      outcome: 'error',
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export interface ProgressEntry {
  file: string;
  index: number;
  type: DistilledMetaRow['type'];
  outcome: ThinkOutcome;
  error?: string;
}

const progressKey = (file: string, index: number) => `${file}#${index}`;

/** The latest entry per file and row index; a retried row's later outcome wins. */
export function readProgress(path: string): Map<string, ProgressEntry> {
  const entries = new Map<string, ProgressEntry>();
  if (!existsSync(path)) return entries;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const entry = JSON.parse(line) as ProgressEntry;
    entries.set(progressKey(entry.file, entry.index), entry);
  }
  return entries;
}

const COUNT_NAMES: Record<ThinkOutcome, string> = {
  kept: 'kept',
  rendered: 'rendered',
  rejectedFormat: 'rejectedFormat',
  disagreed: 'disagreed',
  missing: 'missing',
  error: 'errors',
};

/** Outcome counts per type over both files, from the latest entry of every row. */
export function thinkCounts(
  progressPath: string,
): Record<string, Record<string, number>> {
  const counts: Record<string, Record<string, number>> = {};
  for (const entry of readProgress(progressPath).values()) {
    const byOutcome = (counts[entry.type] ??= {});
    const name = COUNT_NAMES[entry.outcome];
    byOutcome[name] = (byOutcome[name] ?? 0) + 1;
  }
  return counts;
}

export interface ThinkFileOptions {
  /** The source file name, e.g. `conversations.jsonl`; output keeps the same name. */
  file: string;
  rows: readonly DistilledMetaRow[];
  pool: SessionPool;
  contract: ReaderContract;
  clients: ThinkClients;
  outDir: string;
  concurrency?: number;
  teacherMaxTokens?: number;
  onRow?: (entry: ProgressEntry) => void;
}

/**
 * Every undecided row of one file through thinkRow. Rows are appended as they finish, the
 * conversation and its meta row in the same synchronous step so the twins stay line-aligned;
 * the progress entry follows them.
 */
export async function thinkFile(options: ThinkFileOptions): Promise<void> {
  const { file, outDir } = options;
  mkdirSync(outDir, { recursive: true });
  const progressPath = join(outDir, 'progress.jsonl');
  const done = readProgress(progressPath);
  const todo = options.rows
    .map((row, index) => ({ row, index }))
    .filter(({ index }) => {
      const entry = done.get(progressKey(file, index));
      return entry === undefined || entry.outcome === 'error';
    });
  await runPool(todo, options.concurrency ?? 8, async ({ row, index }) => {
    const result = await thinkRow(
      row,
      options.pool,
      options.contract,
      options.clients,
      options.teacherMaxTokens,
    );
    if (result.outcome === 'kept' || result.outcome === 'rendered') {
      appendFileSync(
        join(outDir, file),
        `${JSON.stringify(result.conversation)}\n`,
      );
      appendFileSync(
        join(outDir, `${file}.meta.jsonl`),
        `${JSON.stringify(result.meta)}\n`,
      );
    } else if (result.outcome === 'disagreed') {
      appendFileSync(
        join(outDir, 'disagreements.jsonl'),
        `${JSON.stringify({ file, index, ...result.disagreement })}\n`,
      );
    }
    const entry: ProgressEntry = {
      file,
      index,
      type: row.type,
      outcome: result.outcome,
      ...(result.outcome === 'error' ? { error: result.error } : {}),
    };
    appendFileSync(progressPath, `${JSON.stringify(entry)}\n`);
    options.onRow?.(entry);
  });
}

/**
 * The labelled session file a distilled directory was built from: its manifest's `labels`,
 * or the one file every part names (a merged manifest keeps its parts' labels).
 */
export function sourceLabels(manifest: {
  labels?: unknown;
  parts?: unknown;
}): string | undefined {
  if (typeof manifest.labels === 'string') return manifest.labels;
  if (manifest.parts === null || typeof manifest.parts !== 'object')
    return undefined;
  const labels = new Set(
    Object.values(manifest.parts as Record<string, { labels?: unknown }>).map(
      (part) => part?.labels,
    ),
  );
  if (labels.size !== 1) return undefined;
  const [only] = [...labels];
  return typeof only === 'string' ? only : undefined;
}
