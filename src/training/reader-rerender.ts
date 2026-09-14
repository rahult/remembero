/**
 * Re-render distilled reader data through a different reader contract.
 *
 * Reader v4's 3,100 prompts were rendered before the contract existed: no date
 * distances, no computed notes. The harness now builds both at evaluation time, so a
 * reader trained on v4 reads a prompt at evaluation it never saw in training. Nothing
 * about the examples needs to change to fix that — the questions, the haystacks and the
 * teacher's answers are still good — only the rendering. So this rebuilds each haystack
 * from the meta row's session ids (in haystack order) against the same labelled session
 * pool, renders it with the contract asked for, and keeps the recorded answer as the
 * assistant turn. No teacher calls: the answers are v4's, byte for byte.
 *
 * A row whose sessions are no longer in the pool cannot be rebuilt, and is skipped and
 * counted rather than dropped silently — a shrinking pool would otherwise quietly shrink
 * the training set.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ReaderContract } from '../evals/reader-contract.js';
import type { Conversation } from './export.js';
import {
  readerMessages,
  toDistilledConversation,
  type DistillType,
  type Haystack,
} from './reader-distill.js';
import type { LabelledSession } from './reader-data.js';

/** One row of a distilled `*.meta.jsonl`: the example without its rendered messages. */
export interface DistilledMetaRow {
  type: DistillType;
  question: string;
  questionDate: string;
  sessionIds: string[];
  evidence: number[];
  answer: string;
}

export type SessionPool = ReadonlyMap<string, LabelledSession>;

/**
 * The haystack the row was built from, in its recorded session order, or the ids that are
 * no longer in the pool. The order matters: it is the order the sessions were rendered in
 * and the numbering the teacher's `evidence` refers to.
 */
export function haystackFromMeta(
  row: DistilledMetaRow,
  pool: SessionPool,
): { haystack: Haystack } | { missing: string[] } {
  const missing = row.sessionIds.filter((id) => !pool.has(id));
  if (missing.length > 0) return { missing };
  return {
    haystack: {
      sessions: row.sessionIds.map((id) => pool.get(id)!),
      questionDate: row.questionDate,
    },
  };
}

/** The row re-rendered with the contract, the recorded answer as the assistant turn. */
export function rerenderRow(
  row: DistilledMetaRow,
  pool: SessionPool,
  contract: ReaderContract,
): Conversation | undefined {
  const built = haystackFromMeta(row, pool);
  if ('missing' in built) return undefined;
  return toDistilledConversation({
    ...row,
    messages: readerMessages(built.haystack, row.question, row.type, contract),
  });
}

export interface RerenderResult {
  rows: number;
  skipped: number;
  /** The first few rows that could not be rebuilt, for the console. */
  missingExamples: Array<{ index: number; missing: string[] }>;
}

export function parseMetaRows(path: string): DistilledMetaRow[] {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as DistilledMetaRow);
}

/** Re-render every row of a meta file into `outPath`, one conversation per line. */
export function rerenderMetaFile(
  metaPath: string,
  outPath: string,
  pool: SessionPool,
  contract: ReaderContract,
  onProgress?: (done: number) => void,
): RerenderResult {
  const rows = parseMetaRows(metaPath);
  const lines: string[] = [];
  const missingExamples: RerenderResult['missingExamples'] = [];
  let skipped = 0;
  rows.forEach((row, index) => {
    const built = haystackFromMeta(row, pool);
    if ('missing' in built) {
      skipped += 1;
      if (missingExamples.length < 5)
        missingExamples.push({ index, missing: built.missing });
      return;
    }
    lines.push(
      JSON.stringify(
        toDistilledConversation({
          ...row,
          messages: readerMessages(
            built.haystack,
            row.question,
            row.type,
            contract,
          ),
        }),
      ),
    );
    onProgress?.(index + 1);
  });
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, lines.length === 0 ? '' : `${lines.join('\n')}\n`);
  return { rows: lines.length, skipped, missingExamples };
}
