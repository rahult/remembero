/**
 * One retrieval and one reading prompt, shared by the benchmark harness and the product.
 *
 * Everything the harness measured — ranking user turns instead of whole sessions, summing a
 * session's matching turns, the TypeSafe re-rank, the "in range first" ordering, the depth cut,
 * the per-session byte budget, the date-distance headers, the computed-notes block and the two
 * reader prompts — used to live inside `src/evals/longmemeval-answer.ts`, where the product
 * could not call it. It lives here now: the harness's context builder is a thin adapter over
 * `renderReadingPrompt`, and the product's `recall` path calls `retrieveSessions` and
 * `buildReadingPrompt`. A parity test (tests/session-retrieval.test.ts) holds the two prompts
 * byte-identical, because the benchmark numbers only describe the product while they are.
 *
 * The five answering decisions come from a `QuestionKind` (question-kind.ts), never a dataset
 * label: depth, the retrieval unit, whether a time range applies, whether the reader works in
 * notes, whether the answer is personalised, and whether assistant turns are evidence.
 */

import { canonicalKey, parseProgram, type Clause } from '../engine/index.js';
import { recallWords } from '../llm/schema.js';
import { assertSafeForExternalLlm } from '../safety.js';
import type { MemorySource } from '../store/store.js';
import { type ContextTiers } from '../evals/reader-contract.js';
import {
  DEFAULT_RERANK_POOL,
  DEFAULT_RERANK_SESSION_CHARS,
  rerankSessionOrder,
  type RerankStats,
  type TypesafeNouls,
} from '../evals/typesafe-rerank.js';
import {
  buildComputedNotes,
  STOPWORDS as COMPUTED_NOTES_STOPWORDS,
} from './computed-notes.js';
import { searchKnowledge, type KnowledgeSearchResult } from './search.js';
import { SEMANTIC_CHUNK_CHARACTERS } from './semantic-search.js';
import { buildStructuredEvidence } from './structured-evidence.js';
import type { QuestionKind } from './question-kind.js';

/** How much of a session's text is ever ranked or rendered. */
export const READING_SOURCE_CHARACTERS = 16_384;
export const DEFAULT_READING_CONTEXT_BYTES = 56 * 1024;
export const MAX_READING_CONTEXT_BYTES = 160 * 1024;

/** Retrieval depth per kind: gathering needs breadth, time needs a little, a lookup needs four. */
export const DEFAULT_AGGREGATION_DEPTH = 12;
export const DEFAULT_TEMPORAL_DEPTH = 10;
export const DEFAULT_READING_DEPTH = 4;

/** A session the reader can be shown: its id, its date, and its turns in order. */
export interface RetrievableSession {
  id: string;
  /** An ISO instant ("2024-02-01T09:00:00.000Z"); its day is what the reader sees. */
  date: string;
  turns: Array<{ role: 'user' | 'assistant'; text: string }>;
}

/** What one TypeSafe re-rank cost: candidates judged, live calls, cache hits, tokens. */
export type RerankUsage = RerankStats;

export interface SessionRetrievalOptions {
  /** The five answering decisions, from the question text or a TypeSafe classification. */
  kind: QuestionKind;
  /** Depth for a question that is neither aggregation nor temporal. */
  topK: number;
  contextBytes: number;
  dateDistances: boolean;
  computedNotes: boolean;
  /** Depth for an aggregation question (default 12). */
  aggregationTopK?: number;
  /** Depth for a temporal question (default 10). */
  temporalTopK?: number;
  /** The TypeSafe re-rank over the first `pool` sessions, before the depth cut. */
  rerank?: { client: TypesafeNouls; pool?: number; sessionChars?: number };
  /** Sessions inside the range are read before those outside it. Only temporal questions get one. */
  timeRange?: { start: string; end: string };
}

/** The depth this question's kind asks for. */
export function readingDepth(
  kind: QuestionKind,
  depths: {
    topK?: number;
    aggregationTopK?: number;
    temporalTopK?: number;
  } = {},
): number {
  if (kind.aggregation)
    return depths.aggregationTopK ?? DEFAULT_AGGREGATION_DEPTH;
  if (kind.temporal) return depths.temporalTopK ?? DEFAULT_TEMPORAL_DEPTH;
  return depths.topK ?? DEFAULT_READING_DEPTH;
}

/**
 * The retrieval unit: one document per turn, because a long session's one relevant sentence is
 * lost among its others — except on a temporal question, where the session is the dated thing
 * and the turn unit measured worse.
 */
export function readingUnit(kind: QuestionKind): 'turn' | 'session' {
  return kind.temporal ? 'session' : 'turn';
}

/** Does the reader work in notes first? Gathering, ordering in time and a superseded value do. */
export function readsInNotes(kind: QuestionKind): boolean {
  return kind.aggregation || kind.temporal || kind.update;
}

/** Whose turns are evidence: only a question about the assistant's own words needs its turns. */
export function readingContextRoles(kind: QuestionKind): 'user' | 'all' {
  return kind.assistantRecall ? 'all' : 'user';
}

/**
 * The session text the reader is shown, as "role: content" lines: the user's turns, because
 * assistant turns are ~87% of the characters and state no facts — unless the question asks what
 * the assistant said. A session with no user turn at all falls back to every turn, so it is
 * never empty.
 */
export function readingSessionText(
  turns: ReadonlyArray<{ role: string; content: string }>,
  roles: 'user' | 'all',
): string {
  const wanted =
    roles === 'all' ? turns : turns.filter(({ role }) => role === 'user');
  const shown = wanted.length === 0 ? turns : wanted;
  return shown.map(({ role, content }) => `${role}: ${content}`).join('\n');
}

/** The same rule over a stored session. */
export function sessionReadingText(
  session: RetrievableSession,
  roles: 'user' | 'all',
): string {
  return readingSessionText(
    session.turns.map(({ role, text }) => ({ role, content: text })),
    roles,
  );
}

/** The question date the reader is given: the day the question was asked. */
export function readingQuestionDate(askedAt: Date): string {
  return askedAt.toISOString().slice(0, 10);
}

/**
 * `label` names the caller in the two messages, so the benchmark keeps the wording its runner
 * has always printed and the product does not tell a user about LongMemEval.
 */
export function validateReadingOptions(
  topK: number,
  contextBytes: number,
  label = 'reading',
): void {
  if (!Number.isSafeInteger(topK) || topK < 1 || topK > 100) {
    throw new Error(`${label} topK must be an integer from 1 to 100`);
  }
  if (
    !Number.isSafeInteger(contextBytes) ||
    contextBytes < 4_096 ||
    contextBytes > MAX_READING_CONTEXT_BYTES
  ) {
    throw new Error(
      `${label} context bytes must be an integer from 4096 to ${MAX_READING_CONTEXT_BYTES}`,
    );
  }
}

/**
 * Turn ranks to session order: a session scores the sum of 1/log2(rank+1) over its matching
 * turns, so several near-misses can outrank one lucky hit. Positions are the search's own, so a
 * result whose source is missing still consumes one.
 */
export function aggregateSessionsByTurnRank(
  sessionsByRank: ReadonlyArray<string | undefined>,
): string[] {
  const sessionScore = new Map<string, number>();
  sessionsByRank.forEach((session, position) => {
    if (session === undefined) return;
    sessionScore.set(
      session,
      (sessionScore.get(session) ?? 0) + 1 / Math.log2(position + 2),
    );
  });
  return [...sessionScore.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([session]) => session);
}

/**
 * A stable partition: the items dated inside the range first, the rest after, each group in the
 * order it arrived. Without a range the order is untouched.
 */
export function orderInRangeFirst<T extends { ts: string }>(
  items: readonly T[],
  range: { start: string; end: string } | null | undefined,
): T[] {
  if (range === null || range === undefined) return [...items];
  const inRange = (ts: string): boolean =>
    ts.slice(0, 10) >= range.start && ts.slice(0, 10) <= range.end;
  return [
    ...items.filter((item) => inRange(item.ts)),
    ...items.filter((item) => !inRange(item.ts)),
  ];
}

function boundedUtf8(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, 'utf8') <= maxBytes) return value;
  let low = 0;
  let high = value.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (Buffer.byteLength(value.slice(0, middle), 'utf8') <= maxBytes)
      low = middle;
    else high = middle - 1;
  }
  return value.slice(0, low);
}

function sourceWindow(
  text: string,
  question: string,
  maxBytes: number,
  focusCharacterOffset?: number,
): string {
  const boundedSource = text.slice(0, READING_SOURCE_CHARACTERS);
  if (Buffer.byteLength(boundedSource, 'utf8') <= maxBytes)
    return boundedSource;
  if (focusCharacterOffset !== undefined) {
    const approximateCharacters = Math.max(
      1,
      Math.min(boundedSource.length, maxBytes),
    );
    const start = Math.max(
      0,
      Math.min(
        boundedSource.length - approximateCharacters,
        focusCharacterOffset -
          Math.floor((approximateCharacters - SEMANTIC_CHUNK_CHARACTERS) / 2),
      ),
    );
    return boundedUtf8(boundedSource.slice(start), maxBytes);
  }
  const words = [
    ...new Set(recallWords(question).filter((word) => word.length >= 3)),
  ];
  const approximateCharacters = Math.max(
    1,
    Math.min(boundedSource.length, maxBytes),
  );
  let bestStart = 0;
  let bestScore = -1;
  for (let start = 0; start < boundedSource.length; start += 512) {
    const candidate = boundedSource
      .slice(start, start + approximateCharacters)
      .toLowerCase();
    const score = words.reduce(
      (total, word) => total + (candidate.includes(word) ? 1 : 0),
      0,
    );
    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
    if (start + approximateCharacters >= boundedSource.length) break;
  }
  return boundedUtf8(boundedSource.slice(bestStart), maxBytes);
}

/**
 * Words that say nothing about what a question is after: the computed-notes stopwords plus
 * the auxiliaries that otherwise dominate abstract matching on real questions. Kept local so
 * the computed-notes block, part of every reader contract, renders exactly as before.
 */
const ABSTRACT_STOPWORDS: ReadonlySet<string> = new Set([
  ...COMPUTED_NOTES_STOPWORDS,
  ...'need from since about would could should there their which'.split(' '),
]);

/** The question's content words for abstract matching, canonical as recallWords makes them. */
function abstractContentWords(question: string): string[] {
  const raw = question
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  const words = raw.flatMap((word) => {
    if (ABSTRACT_STOPWORDS.has(word)) return [];
    const canonical = recallWords(word)[0];
    return canonical === undefined ||
      canonical.length < 4 ||
      ABSTRACT_STOPWORDS.has(canonical)
      ? []
      : [canonical];
  });
  return [...new Set(words)];
}

/**
 * Tokens a period follows without ending a sentence, lowercased with their inner periods kept
 * and the final one dropped (so "e.g." is "e.g"). One-letter tokens are handled by rule.
 */
const ABBREVIATIONS: ReadonlySet<string> = new Set([
  'dr', 'mr', 'mrs', 'ms', 'st', 'jr', 'sr', 'vs', 'etc', 'e.g', 'i.e',
  'prof', 'sgt', 'capt', 'gen', 'rev', 'hon', 'approx', 'dept', 'univ',
  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
]);

/**
 * One turn's sentences: split after . ! ? and whitespace, and at newlines, but not after a
 * period that closes a one-letter token (an initial, or the pieces of U.S., D.C., p.m.) or a
 * known abbreviation (Dr., Mrs., e.g.). Ordinary short words (it, up, ok, so) still end one.
 */
function turnSentences(turn: string): string[] {
  const sentences: string[] = [];
  for (const line of turn.split(/\n+/)) {
    let start = 0;
    for (const match of line.matchAll(/[.!?]+(?=\s)/g)) {
      const end = match.index! + match[0].length;
      if (match[0] === '.') {
        const token = /([A-Za-z][A-Za-z.]*)$/.exec(
          line.slice(start, match.index!),
        )?.[1];
        const lastPiece = token?.split('.').at(-1);
        if (
          token !== undefined &&
          (lastPiece!.length === 1 || ABBREVIATIONS.has(token.toLowerCase()))
        )
          continue;
      }
      sentences.push(line.slice(start, end));
      start = end;
    }
    sentences.push(line.slice(start));
  }
  return sentences
    .map((sentence) => sentence.replace(/\s+/g, ' ').trim())
    .filter((sentence) => sentence !== '');
}

/** The sentences of a session's user turns, in order; assistant turns never contribute. */
function userSentences(text: string): string[] {
  const turns = [...text.matchAll(/^(user|assistant)[ \t]*:/gim)];
  // a source with no role markers (a memory layer's own text) is the user's words throughout
  const userText =
    turns.length === 0
      ? [text]
      : turns.flatMap((turn, index) =>
          turn[1]!.toLowerCase() === 'user'
            ? [
                text.slice(
                  turn.index! + turn[0].length,
                  turns[index + 1]?.index ?? text.length,
                ),
              ]
            : [],
        );
  return userText.flatMap(turnSentences);
}

/** Cut to maxBytes, backing off to the last word boundary when the cut lands inside a word. */
function boundedAtWord(value: string, maxBytes: number): string {
  const bounded = boundedUtf8(value, maxBytes);
  if (bounded.length === value.length) return bounded;
  const space = bounded.lastIndexOf(' ');
  return space > 0 ? bounded.slice(0, space) : bounded;
}

/**
 * A tiered session's abstract: its header, then the user sentences that name the question.
 * Sentences rank by how many distinct content words they contain (ties by position) and are
 * taken greedily in that order, skipping any that no longer fit, then shown in their original
 * order. With no matching sentence, the first user sentence cut at a word boundary.
 */
function abstractSection(
  header: string,
  text: string,
  words: readonly string[],
  maxBytes: number,
): string {
  const room = maxBytes - Buffer.byteLength(header, 'utf8') - 1;
  if (room <= 0) return `${boundedUtf8(header, maxBytes - 1)}\n`;
  const sentences = userSentences(text);
  const ranked = sentences
    .map((sentence, position) => {
      const tokens = new Set(recallWords(sentence));
      return {
        sentence,
        position,
        bytes: Buffer.byteLength(sentence, 'utf8'),
        score: words.reduce((n, word) => n + (tokens.has(word) ? 1 : 0), 0),
      };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.position - b.position);
  let body = '';
  if (ranked.length === 0) {
    if (sentences.length > 0) body = boundedAtWord(sentences[0]!, room);
  } else {
    const chosen: typeof ranked = [];
    let used = 0;
    for (const candidate of ranked) {
      const cost = candidate.bytes + (chosen.length === 0 ? 0 : 1);
      if (used + cost > room) continue;
      chosen.push(candidate);
      used += cost;
    }
    body =
      chosen.length === 0
        ? // every matching sentence is longer than the room: the best one, cut at a word
          boundedAtWord(ranked[0]!.sentence, room)
        : chosen
            .sort((a, b) => a.position - b.position)
            .map(({ sentence }) => sentence)
            .join(' ');
  }
  return `${header}${body}\n`;
}

const MAX_ENGINE_RENDER_CHARACTERS = 4_000;

/** Question dates come as "2024/03/01 (Fri) 09:00"; sessions as ISO instants. */
function questionDay(questionDate: string): string {
  const match = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(questionDate);
  return match === null
    ? questionDate.slice(0, 10)
    : `${match[1]}-${match[2]}-${match[3]}`;
}

/**
 * "63 days, about 9 weeks or 2 months, before the question date 2024-03-01": the
 * arithmetic a reader would otherwise do itself, done once and exactly.
 */
export function describeDistance(
  sessionTs: string,
  questionDate: string,
): string {
  const day = questionDay(questionDate);
  const from = Date.UTC(
    Number(sessionTs.slice(0, 4)),
    Number(sessionTs.slice(5, 7)) - 1,
    Number(sessionTs.slice(8, 10)),
  );
  const to = Date.UTC(
    Number(day.slice(0, 4)),
    Number(day.slice(5, 7)) - 1,
    Number(day.slice(8, 10)),
  );
  const days = Math.round((to - from) / 86_400_000);
  if (!Number.isFinite(days)) return `question date ${questionDate}`;
  const relation = days >= 0 ? 'before' : 'after';
  const abs = Math.abs(days);
  const parts = [`${abs} day${abs === 1 ? '' : 's'}`];
  if (abs >= 14) {
    const weeks = Math.round(abs / 7);
    const months = Math.round(abs / 30.44);
    parts.push(
      abs >= 60
        ? `about ${weeks} weeks or ${months} month${months === 1 ? '' : 's'}`
        : `about ${weeks} weeks`,
    );
  }
  return `${parts.join(', ')}${parts.length > 1 ? ',' : ''} ${relation} the question date ${day}`;
}

/** One retrieved session as the prompt builder takes it. */
export interface ReadingSource {
  opId: string;
  ts: string;
  text?: string;
  redacted?: true;
  focusCharacterOffset?: number;
  /** Extracted facts of this session that matched the question, shown dated to the reader. */
  facts?: string[];
}

/** Full text for the top-ranked few, code-built abstracts for the rest. */
export type { ContextTiers };

export interface ReadingPromptRequest {
  question: string;
  /** Printed to the reader verbatim, and the date every distance is measured against. */
  questionDate: string;
  sources: ReadonlyArray<ReadingSource>;
  contextBytes?: number;
  /** Facts found apart from the retrieved sessions, appended as a dated block. */
  extraFacts?: ReadonlyArray<{ clause: string; ts: string }>;
  reading?: 'direct' | 'notes' | 'enumerate';
  engine?: { query: string; rendered: string };
  dateDistances?: boolean;
  computedNotes?: boolean;
  focusedBudget?: boolean;
  structuredEvidence?: boolean;
  tiers?: ContextTiers;
  /** A code-counted tally (see evals/typesafe-count.ts): the computed-notes block's first line. */
  countedLine?: string;
  /** Personalise the answer rather than look a fact up. */
  personalize?: boolean;
  /** Names this caller in the validation and safety messages (default "reading"). */
  label?: string;
}

export interface ReadingPrompt {
  system: string;
  user: string;
  /** The sessions the prompt actually rendered, best rank first. */
  contextSessionIds: string[];
  redactedRetrievedSessions: number;
  /** Each rendered session's section, exactly as it appears in the user message. */
  sections: Array<{ opId: string; section: string }>;
}

/**
 * The reader's prompt: the retrieved sessions dated and cut to the byte budget, whatever
 * computed blocks are switched on, and the system prompt this question's reading asks for.
 */
export function renderReadingPrompt(
  request: ReadingPromptRequest,
): ReadingPrompt {
  const question = request.question;
  const questionDate = request.questionDate;
  const rankedSources = request.sources;
  const contextBytes = request.contextBytes ?? DEFAULT_READING_CONTEXT_BYTES;
  const extraFacts = request.extraFacts ?? [];
  const reading = request.reading ?? 'direct';
  const engine = request.engine;
  const dateDistances = request.dateDistances ?? false;
  const computedNotes = request.computedNotes ?? false;
  const focusedBudget = request.focusedBudget ?? false;
  const structuredEvidence = request.structuredEvidence ?? false;
  const tiers = request.tiers;
  const countedLine = request.countedLine;
  const personalize = request.personalize ?? false;
  const label = request.label ?? 'reading';
  validateReadingOptions(
    Math.max(1, rankedSources.length),
    contextBytes,
    label,
  );
  if (tiers !== undefined) {
    if (focusedBudget)
      throw new Error(
        'context tiers cannot be combined with the focused budget: two budget policies in one prompt',
      );
    if (!Number.isInteger(tiers.fullSessions) || tiers.fullSessions <= 0)
      throw new Error(
        `context tiers need a positive integer fullSessions, got ${tiers.fullSessions}`,
      );
    if (!Number.isInteger(tiers.abstractBytes) || tiers.abstractBytes <= 0)
      throw new Error(
        `context tiers need a positive integer abstractBytes, got ${tiers.abstractBytes}`,
      );
  }
  const usable = rankedSources.filter(
    (source) => source.redacted !== true && source.text !== undefined,
  );
  // tiers: the first fullSessions by rank keep a sourceWindow body; the rest become abstracts,
  // and the full sessions split what the abstracts leave, exactly as the even split divides
  const fullCount =
    tiers === undefined
      ? usable.length
      : Math.min(usable.length, tiers.fullSessions);
  const abstractFor = (rank: number) =>
    tiers !== undefined && rank >= fullCount;
  const dateLineFor = (ts: string) =>
    dateDistances
      ? `Session date: ${ts.slice(0, 10)} (${describeDistance(ts, questionDate)})`
      : `Session date: ${ts}`;
  const factsLineFor = (facts: string[] | undefined) =>
    facts !== undefined && facts.length > 0
      ? `Remembered facts (stated in this session): ${facts.join(' ')}\n`
      : '';
  const abstractWords =
    tiers === undefined ? [] : abstractContentWords(question);
  const abstracts = new Map<number, string>();
  if (tiers !== undefined)
    for (let rank = fullCount; rank < usable.length; rank += 1) {
      const source = usable[rank]!;
      abstracts.set(
        rank,
        abstractSection(
          // an abstract is the lines matching the question only: no facts line
          `### Retrieved session ${rank + 1} (abstract)\n${dateLineFor(source.ts)}\n`,
          source.text!,
          abstractWords,
          tiers.abstractBytes,
        ),
      );
    }
  const abstractBytesUsed = [...abstracts.values()].reduce(
    (sum, section) => sum + Buffer.byteLength(section, 'utf8'),
    0,
  );
  if (abstracts.size > 0 && abstractBytesUsed > contextBytes - 256 * fullCount)
    throw new Error(
      `abstract sections take ${abstractBytesUsed} bytes, more than the ${contextBytes - 256 * fullCount} bytes the context leaves after 256 per full session; lower abstractBytes or the retrieval depth`,
    );
  const evenBytes = Math.max(
    256,
    Math.floor((contextBytes - abstractBytesUsed) / Math.max(1, fullCount)),
  );
  // focused budget: a session's share of the context grows with the number of the question's
  // content words it contains, so fifteen retrieved sessions do not each get a 1.6 KB sliver
  // that cuts the one sentence the question needs
  const focusWords = focusedBudget
    ? [...new Set(recallWords(question).filter((word) => word.length >= 4))]
    : [];
  const weights = usable.map((source) => {
    if (!focusedBudget) return 1;
    const low = source.text!.toLowerCase();
    return 1 + focusWords.reduce((n, w) => n + (low.includes(w) ? 1 : 0), 0) * 2;
  });
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const budgetFor = (index: number) =>
    focusedBudget
      ? Math.max(256, Math.floor((contextBytes * weights[index]!) / weightSum))
      : evenBytes;
  const selected = usable.map((source, rank) => {
    if (abstractFor(rank))
      return { ...source, rank, section: abstracts.get(rank)! };
    const facts = factsLineFor(source.facts);
    const dateLine = dateLineFor(source.ts);
    const header = `### Retrieved session ${rank + 1}\n${dateLine}\n${facts}`;
    const body = sourceWindow(
      source.text!,
      question,
      Math.max(1, budgetFor(rank) - Buffer.byteLength(header, 'utf8') - 2),
      source.focusCharacterOffset,
    );
    return { ...source, rank, section: `${header}${body}\n` };
  });
  const history = selected
    .sort(
      (left, right) =>
        left.ts.localeCompare(right.ts) || left.rank - right.rank,
    )
    .map(({ section }) => section)
    .join('\n');
  const remembered =
    extraFacts.length === 0
      ? ''
      : `\n### Supplementary remembered facts (extracted earlier, with the session date; they may be unrelated to the question, so ignore any that do not concern it and never treat them as evidence on their own)\n${extraFacts
          .map(({ ts, clause }) => `- ${ts}: ${clause}`)
          .join('\n')}\n`;
  const engineBlock =
    engine === undefined
      ? ''
      : `\n### Memory engine result\nThe memory system wrote this Datalog program over the facts it remembered from the whole history (every session, not only the chats above) and executed it. The rows are exact for the remembered facts, but the program may be broader than the question and facts can be missing or misread, so keep only the rows that fit the question, cross-check with the chats, and prefer the chats where they disagree.\nProgram: ${engine.query.replace(/\s*\n\s*/g, ' ')}\n${engine.rendered.slice(0, MAX_ENGINE_RENDER_CHARACTERS)}\n`;
  const pinnedLines = countedLine === undefined || countedLine === '' ? [] : [countedLine];
  const computedBlock =
    computedNotes || pinnedLines.length > 0
      ? (() => {
          const block = buildComputedNotes(
            question,
            questionDate,
            // without --computed-notes the block carries the counted line alone
            computedNotes ? selected.map(({ ts, text }) => ({ ts, text: text! })) : [],
            {},
            pinnedLines,
          );
          return block === '' ? '' : `\n${block}`;
        })()
      : '';
  // structured evidence goes first: dated claims to compose from, then the chats to check
  const evidenceBlock = structuredEvidence
    ? (() => {
        const block = buildStructuredEvidence(
          question,
          questionDate,
          [
            ...selected.map(({ ts, text, facts }) => ({ ts, text: text!, facts })),
            // supplementary facts have no session text here; they ground against themselves
            ...extraFacts.map(({ ts, clause }) => ({ ts, text: `USER: ${clause}`, facts: [clause] })),
          ],
        );
        return block === '' ? '' : `${block}\n`;
      })()
    : '';
  const user = `${evidenceBlock}History chats:\n\n${history || '[no safe relevant history retrieved]'}\n${remembered}${engineBlock}${computedBlock}Current date: ${questionDate}\nQuestion: ${question}\nAnswer:`;
  assertSafeForExternalLlm(user, `${label} prompt`);
  const system =
    personalize
      ? 'Use the supplied history to personalize the answer. You may use general knowledge for recommendations, but do not invent facts about the user. Briefly make the remembered preference or context driving the answer explicit.'
      : reading === 'enumerate'
        ? 'Do not answer the question yet. From the supplied history, list every item relevant to the question, one per line, each with its session date and the exact detail the history states (a count, a name, a date, an amount, a quote). Include every occurrence across sessions, keep duplicates apart, and add nothing the history does not say. If nothing is relevant, write "No relevant items." Output the list only.'
        : reading === 'notes'
          ? 'Answer only from the supplied history. Work in two steps. First, under "Notes:", list every relevant item the history states, one per line, each with its session date and the exact detail (a count, a name, a date, an amount). Then, on a final line starting with "Answer:", give the answer derived from those notes, concise and with the arithmetic or ordering made explicit when the question needs it. If the notes do not support an answer, the Answer line must say that you do not know. Do not invent details.'
          : 'Answer only from the supplied history. If it does not support an answer, say that you do not know. Be concise and do not invent details.';
  return {
    system,
    user,
    contextSessionIds: selected
      .sort((left, right) => left.rank - right.rank)
      .map(({ opId }) => opId),
    redactedRetrievedSessions: rankedSources.length - usable.length,
    sections: selected.map(({ opId, section }) => ({ opId, section })),
  };
}

/**
 * The product's reading prompt for the sessions it chose. The same call the harness makes, so
 * the two prompts are identical for the same sessions, question and options.
 */
export function buildReadingPrompt(
  question: string,
  askedAt: Date,
  chosen: readonly RetrievableSession[],
  options: SessionRetrievalOptions,
): { system: string; user: string } {
  const prompt = renderReadingPrompt({
    question,
    questionDate: readingQuestionDate(askedAt),
    sources: chosen.map((session) => ({
      opId: session.id,
      ts: session.date,
      text: sessionReadingText(session, readingContextRoles(options.kind)),
    })),
    contextBytes: options.contextBytes,
    reading: readsInNotes(options.kind) ? 'notes' : 'direct',
    dateDistances: options.dateDistances,
    computedNotes: options.computedNotes,
    personalize: options.kind.preference,
  });
  return { system: prompt.system, user: prompt.user };
}

/** The two errors `searchKnowledge` raises for a question it cannot search for at all. */
function unsearchableQuestion(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message === 'knowledge search text has no searchable words' ||
    /^knowledge search text exceeds \d+ (?:words|bytes)$/.test(message)
  );
}

/** One indexed document: a turn, or a whole session when the unit is the session. */
interface SessionDocument {
  clause: Clause;
  source: MemorySource;
  sessionId: string;
}

/**
 * The turn index the ranking reads. The clause is plumbing — the text under it is what scores.
 * Every turn of a run carries the predicate `session_turn`, and a session with no non-empty turn
 * falls back to one `session_text` document, so the predicate words are the same for every
 * candidate of a kind and shift their scores equally rather than reordering them.
 */
function sessionDocuments(
  sessions: readonly RetrievableSession[],
  unit: 'turn' | 'session',
): SessionDocument[] {
  const documents: SessionDocument[] = [];
  sessions.forEach((session, index) => {
    const indexed: Array<{ opId: string; clause: string; text: string }> = [];
    if (unit === 'turn') {
      // assistant turns are indexed too: a question about what the assistant said needs them
      let turnIndex = 0;
      for (const turn of session.turns) {
        if (turn.text.trim() === '') continue;
        indexed.push({
          opId: `${session.id}:t${turnIndex}`,
          clause: `session_turn(s_${index}, t_${turnIndex}).`,
          text: turn.text,
        });
        turnIndex += 1;
      }
    }
    if (indexed.length === 0) {
      indexed.push({
        opId: session.id,
        clause: `session_text(s_${index}).`,
        text: sessionReadingText(session, 'all'),
      });
    }
    for (const document of indexed) {
      const clause = parseProgram(document.clause)[0];
      if (clause === undefined) continue;
      documents.push({
        clause,
        sessionId: session.id,
        source: {
          namespace: 'sessions',
          opId: document.opId,
          ts: session.date,
          text: document.text,
        },
      });
    }
  });
  return documents;
}

/**
 * Rank a namespace's sessions for one question: turn-level lexical search (session-level on a
 * temporal question), the matching turns summed to their sessions, the optional TypeSafe
 * re-rank, the sessions inside a time range first, then the depth cut.
 *
 * `ranked` is the whole ordered candidate list; `chosen` is what the reader should be given.
 */
export async function retrieveSessions(
  question: string,
  askedAt: Date,
  sessions: readonly RetrievableSession[],
  options: SessionRetrievalOptions,
): Promise<{ ranked: string[]; chosen: string[]; rerank?: RerankUsage }> {
  const kind = options.kind;
  const depth = readingDepth(kind, options);
  validateReadingOptions(depth, options.contextBytes);
  const unit = readingUnit(kind);
  const contextRoles = readingContextRoles(kind);
  if (sessions.length === 0) return { ranked: [], chosen: [] };
  const documents = sessionDocuments(sessions, unit);
  const sourceIndex = new Map<string, MemorySource[]>();
  for (const document of documents) {
    const key = canonicalKey(document.clause);
    sourceIndex.set(key, [
      ...(sourceIndex.get(key) ?? []),
      document.source,
    ]);
  }
  const sessionOf = new Map(
    documents.map(({ source, sessionId }) => [source.opId, sessionId]),
  );
  const pool = options.rerank?.pool ?? DEFAULT_RERANK_POOL;
  // a range needs candidates beyond the depth to promote from; the turn unit needs several
  // turns per session; the re-rank needs a pool of sessions
  const baseLimit = Math.min(
    100,
    Math.max(
      options.timeRange === undefined ? 0 : depth * 4,
      unit === 'turn' ? depth * 6 : depth,
    ),
  );
  const searchLimit =
    options.rerank === undefined
      ? baseLimit
      : Math.min(100, Math.max(baseLimit, unit === 'turn' ? pool * 4 : pool));
  let search: KnowledgeSearchResult;
  try {
    search = searchKnowledge(
      documents.map(({ clause }) => clause),
      question,
      sourceIndex,
      {
        limit: searchLimit,
        minimumScore: 1,
        kinds: ['fact'],
        sourceCharacterLimit: READING_SOURCE_CHARACTERS,
      },
    );
  } catch (error) {
    // A question the index cannot be searched for — "why?", or one longer than the search
    // takes — is not a fault in the product: there is nothing to read. The harness never
    // reaches this (a benchmark question always has content words). Anything else is a fault.
    if (!unsearchableQuestion(error)) throw error;
    return { ranked: [], chosen: [] };
  }
  const sessionsByRank = search.results.map((result) => {
    const source = result.sources[0];
    return source === undefined
      ? undefined
      : (sessionOf.get(source.opId) ?? source.opId);
  });
  let order: string[];
  if (unit === 'turn') {
    // exactly the harness's split: the sessions a plain run would have seen come first, in
    // their own summed order, and whatever the re-rank's extra depth found is appended after
    // them, so widening the pool never reorders the head
    const base = sessionsByRank.slice(0, baseLimit);
    order = aggregateSessionsByTurnRank(base);
    if (base.length < sessionsByRank.length) {
      const known = new Set(order);
      order = [
        ...order,
        ...aggregateSessionsByTurnRank(sessionsByRank).filter(
          (session) => !known.has(session),
        ),
      ];
    }
  } else {
    order = [
      ...new Set(sessionsByRank.flatMap((id) => (id === undefined ? [] : [id]))),
    ];
  }
  let rerankUsage: RerankUsage | undefined;
  const rerank = options.rerank;
  if (rerank !== undefined && order.length > 0) {
    const byId = new Map(sessions.map((session) => [session.id, session]));
    const reranked = await rerankSessionOrder(order, {
      question,
      questionDate: readingQuestionDate(askedAt),
      pool,
      sessionChars: rerank.sessionChars ?? DEFAULT_RERANK_SESSION_CHARS,
      contextRoles,
      sessionFor: (sessionId) => {
        const session = byId.get(sessionId);
        return session === undefined
          ? undefined
          : {
              date: session.date,
              turns: session.turns.map(({ role, text }) => ({
                role,
                content: text,
              })),
            };
      },
      nouls: rerank.client,
    });
    order = reranked.order;
    rerankUsage = reranked.stats;
  }
  const dated = new Map(sessions.map((session) => [session.id, session.date]));
  const chosen = orderInRangeFirst(
    order.map((sessionId) => ({
      sessionId,
      ts: dated.get(sessionId) ?? '',
    })),
    options.timeRange,
  )
    .slice(0, depth)
    .map(({ sessionId }) => sessionId);
  return {
    ranked: order,
    chosen,
    ...(rerankUsage === undefined ? {} : { rerank: rerankUsage }),
  };
}
