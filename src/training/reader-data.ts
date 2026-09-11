/**
 * Reader training data without a teacher.
 *
 * Questions and gold answers are built deterministically from labelled real
 * sessions (a session's date, transcript and the facts the labeller wrote for
 * it), so the answer is exact by construction and no frontier model is in the
 * loop. Each example's context is rendered by the same builder the LongMemEval
 * evaluation uses, with distractor sessions and date distances, so the reader
 * trains on exactly what it will see.
 *
 * Types, after LongMemEval's: single-session-user, multi-session (count and
 * list), knowledge-update, temporal-reasoning, and abstention.
 */
import {
  buildLongMemEvalAnswerContext,
  describeDistance,
} from '../evals/longmemeval-answer.js';
import type { LongMemEvalInstance } from '../evals/longmemeval.js';
import type { Conversation } from './export.js';
import type { Rng } from './rng.js';

export interface LabelledSession {
  id: string;
  /** ISO day, YYYY-MM-DD. */
  date: string;
  facts: string[];
  transcript: string;
}

export type ReaderExampleType =
  | 'single-session-user'
  | 'multi-session-count'
  | 'multi-session-list'
  | 'knowledge-update'
  | 'temporal-reasoning'
  | 'abstention';

export interface ReaderExample {
  type: ReaderExampleType;
  question: string;
  questionDate: string;
  predicate: string;
  evidenceSessionIds: string[];
  distractorSessionIds: string[];
  /** The exact answer, written the way the reader should answer. */
  gold: string;
  /** Sessions rendered to the reader, evidence and distractors, in date order. */
  sessions: LabelledSession[];
}

export interface ReaderDataOptions {
  /** Examples attempted per type (default 200). */
  perType?: number;
  /** Fixed question date, else a date after the latest evidence session. */
  questionDate?: string;
  /** Distractor sessions per example (default 3). */
  distractors?: number;
  /** Self atom in the facts (default 'user'). */
  selfAtom?: string;
}

interface ParsedFact {
  predicate: string;
  args: string[];
  text: string;
}

function parseFact(text: string): ParsedFact | undefined {
  const match = /^([a-z][a-z0-9_]*)\((.*)\)\.$/.exec(text.trim());
  if (match === null) return undefined;
  const args = match[2].split(',').map((a) => a.trim());
  return { predicate: match[1], args, text: text.trim() };
}

/** `lives_in` → "live in", `interested_in` → "interested in" (crude, for templates). */
function words(predicate: string): string {
  return predicate.replace(/_/g, ' ');
}

/** Constant rendered for a question or an answer: quotes off, underscores to spaces. */
function plain(value: string): string {
  return value.replace(/^'(.*)'$/, '$1').replace(/_/g, ' ');
}

export function templatedQuestion(
  type: ReaderExampleType,
  predicate: string,
  value: string | null,
): string {
  const p = words(predicate);
  switch (type) {
    case 'single-session-user':
      return `What did I tell you about what I ${p}?`;
    case 'multi-session-count':
      return `How many different things have I mentioned that I ${p}, across all our chats?`;
    case 'multi-session-list':
      return `List everything I have told you that I ${p}, across all our chats.`;
    case 'knowledge-update':
      return `What is the current answer to what I ${p}? It may have changed.`;
    case 'temporal-reasoning':
      return `How long ago did I tell you that I ${p} ${value === null ? '' : plain(value)}?`.replace(
        /\s+\?/,
        '?',
      );
    case 'abstention':
      return `What have I told you about what I ${p}?`;
  }
}

function daysBetween(fromIso: string, toIso: string): number {
  const from = Date.UTC(
    Number(fromIso.slice(0, 4)),
    Number(fromIso.slice(5, 7)) - 1,
    Number(fromIso.slice(8, 10)),
  );
  const to = Date.UTC(
    Number(toIso.slice(0, 4)),
    Number(toIso.slice(5, 7)) - 1,
    Number(toIso.slice(8, 10)),
  );
  return Math.round((to - from) / 86_400_000);
}

function addDaysIso(iso: string, days: number): string {
  const base = Date.UTC(
    Number(iso.slice(0, 4)),
    Number(iso.slice(5, 7)) - 1,
    Number(iso.slice(8, 10)),
  );
  return new Date(base + days * 86_400_000).toISOString().slice(0, 10);
}

const ABSTAIN = 'The history does not say.';

export function generateReaderExamples(
  sessions: readonly LabelledSession[],
  rng: Rng,
  options: ReaderDataOptions = {},
): ReaderExample[] {
  const perType = options.perType ?? 200;
  const distractorCount = options.distractors ?? 3;
  const self = options.selfAtom ?? 'user';

  // facts about the self atom, grouped by predicate; each entry knows its session
  const byPredicate = new Map<
    string,
    Array<{ session: LabelledSession; fact: ParsedFact }>
  >();
  for (const session of sessions) {
    for (const text of session.facts) {
      const fact = parseFact(text);
      if (fact === undefined || fact.args.length !== 2 || fact.args[0] !== self)
        continue;
      const list = byPredicate.get(fact.predicate) ?? [];
      list.push({ session, fact });
      byPredicate.set(fact.predicate, list);
    }
  }
  const predicates = [...byPredicate.keys()].sort();
  if (predicates.length === 0) return [];

  const out: ReaderExample[] = [];
  const pickDistractors = (
    exclude: Set<string>,
    need: number,
  ): LabelledSession[] => {
    const pool = sessions.filter((s) => !exclude.has(s.id));
    return rng.shuffle(pool).slice(0, need);
  };
  const questionDateAfter = (latest: string): string =>
    options.questionDate ?? addDaysIso(latest, 7 + rng.int(120));
  const finish = (
    type: ReaderExampleType,
    predicate: string,
    evidence: LabelledSession[],
    gold: string,
    question: string,
    questionDate?: string,
  ): ReaderExample => {
    const ids = new Set(evidence.map((s) => s.id));
    const distractors = pickDistractors(ids, distractorCount);
    const latest =
      evidence
        .map((s) => s.date)
        .sort()
        .at(-1) ??
      distractors
        .map((s) => s.date)
        .sort()
        .at(-1) ??
      '2023-01-01';
    const all = [...evidence, ...distractors].sort((a, b) =>
      a.date.localeCompare(b.date),
    );
    return {
      type,
      question,
      questionDate: questionDate ?? questionDateAfter(latest),
      predicate,
      evidenceSessionIds: evidence.map((s) => s.id),
      distractorSessionIds: distractors.map((s) => s.id),
      gold,
      sessions: all,
    };
  };

  // single-session-user
  for (let i = 0; i < perType; i += 1) {
    const predicate = rng.pick(predicates);
    const entry = rng.pick(byPredicate.get(predicate)!);
    const value = entry.fact.args[1];
    out.push(
      finish(
        'single-session-user',
        predicate,
        [entry.session],
        plain(value),
        templatedQuestion('single-session-user', predicate, value),
      ),
    );
  }

  // multi-session count and list: predicates that appear in 3+ distinct sessions
  const multi = predicates.filter((p) => {
    const ids = new Set(byPredicate.get(p)!.map((e) => e.session.id));
    return ids.size >= 3;
  });
  for (let i = 0; i < perType && multi.length > 0; i += 1) {
    const predicate = rng.pick(multi);
    const entries = byPredicate.get(predicate)!;
    const bySession = new Map<
      string,
      { session: LabelledSession; values: string[] }
    >();
    for (const e of entries) {
      const slot = bySession.get(e.session.id) ?? {
        session: e.session,
        values: [],
      };
      slot.values.push(e.fact.args[1]);
      bySession.set(e.session.id, slot);
    }
    const chosen = rng
      .shuffle([...bySession.values()])
      .slice(0, Math.min(8, bySession.size));
    if (chosen.length < 3) continue;
    const items = chosen
      .sort((a, b) => a.session.date.localeCompare(b.session.date))
      .map((c) => `${plain(c.values[0])} (${c.session.date})`);
    const type: ReaderExampleType =
      i % 2 === 0 ? 'multi-session-count' : 'multi-session-list';
    const gold =
      type === 'multi-session-count'
        ? `${chosen.length}: ${items.join(', ')}.`
        : `${items.join(', ')}.`;
    out.push(
      finish(
        type,
        predicate,
        chosen.map((c) => c.session),
        gold,
        templatedQuestion(type, predicate, null),
      ),
    );
  }

  // knowledge-update: a predicate stated in exactly two sessions with different values
  // (three or more sessions read as an accumulating relation, which is multi-session material)
  const updatable = predicates.filter((p) => {
    const entries = byPredicate.get(p)!;
    const values = new Set(entries.map((e) => e.fact.args[1]));
    const ids = new Set(entries.map((e) => e.session.id));
    return values.size >= 2 && ids.size === 2;
  });
  for (let i = 0; i < perType && updatable.length > 0; i += 1) {
    const predicate = rng.pick(updatable);
    const entries = rng.shuffle(byPredicate.get(predicate)!);
    const first = entries[0];
    const second = entries.find(
      (e) =>
        e.session.id !== first.session.id &&
        e.fact.args[1] !== first.fact.args[1],
    );
    if (second === undefined) continue;
    const [earlier, later] =
      first.session.date <= second.session.date
        ? [first, second]
        : [second, first];
    if (earlier.session.date === later.session.date) continue;
    const gold = `${plain(later.fact.args[1])} (updated on ${later.session.date}; earlier it was ${plain(earlier.fact.args[1])}, stated ${earlier.session.date}).`;
    out.push(
      finish(
        'knowledge-update',
        predicate,
        [earlier.session, later.session],
        gold,
        templatedQuestion('knowledge-update', predicate, null),
      ),
    );
  }

  // temporal-reasoning: how long ago was one dated fact stated
  for (let i = 0; i < perType; i += 1) {
    const predicate = rng.pick(predicates);
    const entry = rng.pick(byPredicate.get(predicate)!);
    const questionDate = questionDateAfter(entry.session.date);
    const days = daysBetween(entry.session.date, questionDate);
    const weeks = Math.round(days / 7);
    const gold = `${days} day${days === 1 ? '' : 's'} (about ${weeks} week${weeks === 1 ? '' : 's'}) ago, on ${entry.session.date}.`;
    out.push(
      finish(
        'temporal-reasoning',
        predicate,
        [entry.session],
        gold,
        templatedQuestion('temporal-reasoning', predicate, entry.fact.args[1]),
        questionDate,
      ),
    );
  }

  // abstention: a predicate none of the shown sessions carries
  for (let i = 0; i < perType; i += 1) {
    const predicate = rng.pick(predicates);
    const carrying = new Set(
      byPredicate.get(predicate)!.map((e) => e.session.id),
    );
    const distractors = pickDistractors(carrying, distractorCount + 1);
    if (distractors.length === 0) continue;
    const example = finish(
      'abstention',
      predicate,
      [],
      ABSTAIN,
      templatedQuestion('abstention', predicate, null),
    );
    example.sessions = distractors.sort((a, b) => a.date.localeCompare(b.date));
    example.distractorSessionIds = distractors.map((s) => s.id);
    // record what the question was about so tests and audits can check the sessions do not carry it
    example.evidenceSessionIds = [...carrying].slice(0, 3);
    out.push(example);
  }

  return out;
}

const READER_SYSTEM =
  'Answer only from the supplied history. If it does not support an answer, say that you do not know. Be concise and do not invent details.';

export function toReaderConversation(example: ReaderExample): Conversation {
  const instance = {
    question_id: `reader-${example.type}`,
    question_type:
      example.type === 'abstention'
        ? 'single-session-user'
        : example.type.replace(/-(count|list)$/, ''),
    question: example.question,
    question_date: `${example.questionDate.replace(/-/g, '/')} (Sat) 09:00`,
    answer: '',
    haystack_session_ids: [],
    haystack_dates: [],
    haystack_sessions: [],
    answer_session_ids: [],
  } as unknown as LongMemEvalInstance;
  const context = buildLongMemEvalAnswerContext(
    instance,
    example.sessions.map((s) => ({
      opId: s.id,
      ts: `${s.date}T09:00:00.000Z`,
      text: s.transcript,
      facts: s.facts,
    })),
    24 * 1024,
    [],
    'direct',
    undefined,
    true,
  );
  return {
    messages: [
      { role: 'system', content: READER_SYSTEM },
      { role: 'user', content: context.messages[1].content },
      { role: 'assistant', content: example.gold },
    ],
  };
}

export { describeDistance };

/**
 * Optional question rewrite: the templated question is grammatical only by accident
 * ("that I plans to play"). A cheap model rewrites it as a natural question with the
 * same meaning; the rewrite is rejected when it leaks the answer's first value or
 * drifts from the predicate's words entirely.
 */
export interface QuestionRewriter {
  rewrite(example: ReaderExample): Promise<string>;
}

export function rewriteAccepted(
  example: ReaderExample,
  rewritten: string,
): boolean {
  const q = rewritten.trim();
  if (q.length < 8 || q.length > 240 || !q.endsWith('?')) return false;
  const lower = q.toLowerCase();
  const firstValue = example.gold.split(/[:(,]/)[0]?.trim().toLowerCase();
  if (
    example.type !== 'abstention' &&
    example.type !== 'temporal-reasoning' &&
    firstValue !== undefined &&
    firstValue.length >= 3 &&
    lower.includes(firstValue)
  ) {
    return false;
  }
  const predicateWords = words(example.predicate)
    .split(' ')
    .filter((w) => w.length >= 4);
  return (
    predicateWords.length === 0 ||
    predicateWords.some((w) => lower.includes(w.slice(0, 4)))
  );
}

export function rewritePrompt(example: ReaderExample): string {
  return `Rewrite the question below as one natural question a person would type to their assistant, keeping its meaning exactly: same subject, same time reference, same request (a count stays a count, "how long ago" stays "how long ago"). Do not answer it and do not add information. Reply with the question only.

Question: ${example.question}`;
}

export async function rewriteQuestions(
  examples: ReaderExample[],
  rewriter: QuestionRewriter,
  concurrency = 8,
): Promise<{ rewritten: number; kept: number }> {
  let index = 0;
  let rewritten = 0;
  let kept = 0;
  const worker = async () => {
    while (index < examples.length) {
      const example = examples[index++];
      try {
        const candidate = await rewriter.rewrite(example);
        if (rewriteAccepted(example, candidate)) {
          example.question = candidate.trim();
          rewritten += 1;
        } else kept += 1;
      } catch {
        kept += 1;
      }
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  return { rewritten, kept };
}
