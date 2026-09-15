/**
 * Miss mining: the distilled rows the current best reader gets wrong, kept as training data.
 *
 * A distilled directory holds teacher-answered questions rendered under the teacher's
 * contract. Each row's haystack is rebuilt and rendered again under the student's contract
 * (the prompt the student was trained on and is served with), the student answers, and the
 * LongMemEval judge for the row's type compares the student's judged text with the
 * teacher's. A row the student misses is kept exactly as the teacher wrote it: the source
 * conversation line byte for byte, so the miss set is rendered under the teacher's contract,
 * with its meta twin carrying the source file and row index.
 *
 * `results.jsonl` records every judged row (both judged texts and the student's whole
 * reply) for the later review of failure classes. Every row's outcome is appended to
 * `progress.jsonl` keyed by file and row index; a rerun skips judged rows and retries the
 * errored ones, reusing the student's reply when only the judge failed.
 */
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildLongMemEvalJudgePrompt } from '../evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../evals/longmemeval.js';
import {
  assertDistillReading,
  distillManifestContract,
  type ReaderContract,
} from '../evals/reader-contract.js';
import { completionAnswer, readerMessages } from './reader-distill.js';
import {
  haystackFromMeta,
  parseMetaRows,
  type DistilledMetaRow,
  type SessionPool,
} from './reader-rerender.js';
import {
  JUDGE_MAX_TOKENS,
  parseAgreementVerdict,
  progressKey,
  readProgress,
  type CompletionClient,
} from './reader-think.js';
import { runPool } from './run-pool.js';

export interface MineClients {
  student: CompletionClient;
  judge: CompletionClient;
}

/** The student's contract flags, each read without its `--student-` prefix. */
const STUDENT_SWITCHES = [
  '--date-distances',
  '--computed-notes',
  '--focused-budget',
  '--structured-evidence',
];
const STUDENT_VALUES = [
  '--reading',
  '--context-bytes',
  '--full-sessions',
  '--abstract-bytes',
];

/**
 * The student's contract from its `--student-` flags alone: the unprefixed contract flags
 * on the same command line are not the student's and are ignored.
 */
export function studentContractFromFlags(
  argv: readonly string[],
): ReturnType<typeof distillManifestContract> {
  const stripped: string[] = [];
  argv.forEach((token, index) => {
    if (!token.startsWith('--student-')) return;
    const name = `--${token.slice('--student-'.length)}`;
    if (STUDENT_SWITCHES.includes(name)) stripped.push(name);
    else if (STUDENT_VALUES.includes(name) && argv[index + 1] !== undefined)
      stripped.push(name, argv[index + 1]!);
  });
  const reading = stripped.indexOf('--reading');
  if (reading >= 0) {
    try {
      assertDistillReading(stripped);
    } catch {
      throw new Error(
        `--student-reading must be direct or notes, got ${stripped[reading + 1]}`,
      );
    }
  }
  const contract = distillManifestContract(stripped);
  // readerMessages renders distilled haystacks in date order; tiers need rank order
  if (contract.fullSessions !== null)
    throw new Error(
      'mine cannot render tiered student contracts (--student-full-sessions)',
    );
  return contract;
}

/** The contract object of a distilled manifest: what the teacher's rows are rendered under. */
export type TeacherContract = Record<string, unknown> & {
  id: string;
  thinking: boolean;
};

/**
 * The teacher's contract from the distilled directory's manifest. Manifests written before
 * the thinking step have no `thinking` field; their id then carries no `+think` either.
 */
export function teacherContractFromManifest(manifest: {
  contract?: unknown;
}): TeacherContract {
  const contract = manifest.contract;
  if (
    contract === null ||
    typeof contract !== 'object' ||
    typeof (contract as { id?: unknown }).id !== 'string'
  )
    throw new Error(
      'the distilled manifest has no contract; mine needs the contract its rows were rendered under',
    );
  const { id, thinking } = contract as { id: string; thinking?: unknown };
  const thinks = thinking === true;
  if (thinks !== id.includes('+think'))
    throw new Error(
      `the distilled manifest's contract ${id} disagrees with its thinking field (${String(thinking)})`,
    );
  return { ...(contract as Record<string, unknown>), id, thinking: thinks };
}

/**
 * The LongMemEval judge prompt on a synthetic instance: an abstention row's id ends in
 * `_abs`, which is how the judge picks its abstention branch; every other row goes through
 * its type's branch. The teacher's judged text stands as the correct answer.
 */
export function mineJudgePrompt(
  row: DistilledMetaRow,
  expected: string,
  hypothesis: string,
): string {
  const instance = {
    question_id: row.type === 'abstention' ? 'mine_abs' : 'mine',
    question_type: row.type,
    question: row.question,
    answer: expected,
  } as unknown as LongMemEvalInstance;
  return buildLongMemEvalJudgePrompt(instance, hypothesis);
}

/**
 * A distilled file's meta rows with their conversation lines as written. The twins must be
 * line-aligned: the same count, and each line's assistant turn the meta row's answer.
 */
export function readDistilledFile(
  dir: string,
  file: string,
): { rows: DistilledMetaRow[]; lines: string[] } {
  const rows = parseMetaRows(join(dir, `${file}.meta.jsonl`));
  const lines = readFileSync(join(dir, file), 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0);
  if (lines.length !== rows.length)
    throw new Error(
      `${join(dir, file)} has ${lines.length} rows but its meta has ${rows.length}; the twins are not aligned`,
    );
  lines.forEach((line, index) => {
    const conversation = JSON.parse(line) as {
      messages: Array<{ role: string; content: string }>;
    };
    const last = conversation.messages.at(-1);
    if (last?.role !== 'assistant' || last.content !== rows[index]!.answer)
      throw new Error(
        `${join(dir, file)} row ${index} does not answer as its meta row does; the twins are not aligned`,
      );
  });
  return { rows, lines };
}

export interface MineResult {
  file: string;
  index: number;
  type: DistilledMetaRow['type'];
  question: string;
  /** The teacher's judged text. */
  expected: string;
  /** The student's judged text. */
  hypothesis: string;
  /** The student's whole reply. */
  reply: string;
  correct: boolean;
}

export type MineOutcome = 'correct' | 'miss' | 'missing' | 'error';

export interface MineProgressEntry {
  file: string;
  index: number;
  type: DistilledMetaRow['type'];
  outcome: MineOutcome;
  error?: string;
  /** On an error after the student answered: its reply, reused on resume. */
  reply?: string;
}

export interface MineFileOptions {
  /** The source file name, e.g. `conversations.jsonl`. */
  file: string;
  rows: readonly DistilledMetaRow[];
  /** The source conversation lines, aligned with `rows`. */
  lines: readonly string[];
  pool: SessionPool;
  /** Whether the teacher's rows were rendered under thinking: what its judged text is. */
  teacherThinking: boolean;
  studentContract: ReaderContract;
  clients: MineClients;
  outDir: string;
  studentMaxTokens: number;
  concurrency?: number;
  onRow?: (entry: MineProgressEntry) => void;
}

/**
 * Every unjudged row of one file: the student answers under its contract, the judge decides.
 * The result row, and on a miss the teacher's line and its meta twin, are appended in one
 * synchronous step so the twins stay line-aligned; the progress entry follows them.
 */
export async function mineFile(options: MineFileOptions): Promise<void> {
  const { file, outDir, clients, studentContract } = options;
  mkdirSync(outDir, { recursive: true });
  const progressPath = join(outDir, 'progress.jsonl');
  const done = readProgress<MineProgressEntry>(progressPath);
  const todo = options.rows
    .map((row, index) => ({ row, index }))
    .filter(({ index }) => {
      const entry = done.get(progressKey(file, index));
      return entry === undefined || entry.outcome === 'error';
    });
  await runPool(todo, options.concurrency ?? 8, async ({ row, index }) => {
    const record = (
      entry: Omit<MineProgressEntry, 'file' | 'index' | 'type'>,
    ) => {
      const full: MineProgressEntry = { file, index, type: row.type, ...entry };
      appendFileSync(progressPath, `${JSON.stringify(full)}\n`);
      options.onRow?.(full);
    };
    const built = haystackFromMeta(row, options.pool);
    if ('missing' in built) {
      record({
        outcome: 'missing',
        error: `sessions not in pool: ${built.missing.join(', ')}`,
      });
      return;
    }
    let reply = done.get(progressKey(file, index))?.reply;
    try {
      if (reply === undefined) {
        const messages = readerMessages(
          built.haystack,
          row.question,
          row.type,
          studentContract,
        );
        reply = (
          await clients.student.completeWithUsage(messages, {
            maxTokens: options.studentMaxTokens,
          })
        ).content;
      }
      const expected = completionAnswer(
        row.answer,
        options.teacherThinking,
        row.type,
      );
      const hypothesis = completionAnswer(
        reply,
        studentContract.thinking,
        row.type,
      );
      const verdict = await clients.judge.completeWithUsage(
        [
          {
            role: 'user',
            content: mineJudgePrompt(row, expected, hypothesis),
          },
        ],
        { maxTokens: JUDGE_MAX_TOKENS },
      );
      const correct = parseAgreementVerdict(verdict.content);
      const result: MineResult = {
        file,
        index,
        type: row.type,
        question: row.question,
        expected,
        hypothesis,
        reply,
        correct,
      };
      appendFileSync(
        join(outDir, 'results.jsonl'),
        `${JSON.stringify(result)}\n`,
      );
      if (!correct) {
        appendFileSync(
          join(outDir, 'misses.jsonl'),
          `${options.lines[index]}\n`,
        );
        appendFileSync(
          join(outDir, 'misses.jsonl.meta.jsonl'),
          `${JSON.stringify({ ...row, file, index })}\n`,
        );
      }
      record({ outcome: correct ? 'correct' : 'miss' });
    } catch (error) {
      record({
        outcome: 'error',
        error: error instanceof Error ? error.message : String(error),
        ...(reply === undefined ? {} : { reply }),
      });
    }
  });
}

export interface MineCounts {
  byType: Record<string, { asked: number; correct: number; misses: number }>;
  errors: number;
  missing: number;
}

/** Per type over every file mined, from the latest entry of every row. */
export function mineCounts(progressPath: string): MineCounts {
  const counts: MineCounts = { byType: {}, errors: 0, missing: 0 };
  for (const entry of readProgress<MineProgressEntry>(progressPath).values()) {
    if (entry.outcome === 'error') counts.errors += 1;
    else if (entry.outcome === 'missing') counts.missing += 1;
    else {
      const type = (counts.byType[entry.type] ??= {
        asked: 0,
        correct: 0,
        misses: 0,
      });
      type.asked += 1;
      if (entry.outcome === 'correct') type.correct += 1;
      else type.misses += 1;
    }
  }
  return counts;
}

/** The source files `--files` may name. */
export const MINE_FILES = ['conversations', 'heldout'] as const;

export function parseMineFiles(value: string | undefined): string[] {
  if (value === undefined) return MINE_FILES.map((name) => `${name}.jsonl`);
  const names = value
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);
  for (const name of names)
    if (!(MINE_FILES as readonly string[]).includes(name))
      throw new Error(
        `--files takes conversations and/or heldout, got ${name}`,
      );
  if (names.length === 0)
    throw new Error('--files takes conversations and/or heldout');
  return [...new Set(names)].map((name) => `${name}.jsonl`);
}
