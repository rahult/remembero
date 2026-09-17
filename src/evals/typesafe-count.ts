/**
 * Counting across sessions, done in code. TypeSafe's Jev cannot count
 * (https://docs.typesafe.ai/model-jaggedness/jev-1.13, "Counting"): the prescribed fix is to
 * iterate the candidates, ask one yes/no question per candidate, and add up the answers
 * yourself. So the user's sentences that share a content word with the question become
 * candidates, each gets one noul ("this sentence states one instance of the thing the question
 * counts"), and the harness counts the ones above the threshold itself. The client, its cache
 * and its limiter are the re-ranker's (typesafe-rerank.ts); nothing here talks to the network.
 */
import {
  canonicalWord,
  questionTerms,
  sameWord,
  userSentences,
} from '../knowledge/computed-notes.js';
import { countedThing } from '../knowledge/count-question.js';
import { assertSafeForExternalLlm } from '../safety.js';
import type { TypesafeNouls, TypesafeQuestion } from './typesafe-rerank.js';

export const DEFAULT_COUNT_MAX = 60;
export const DEFAULT_COUNT_THRESHOLD = 0.5;
/** Characters the counted line may spend on the list of instances (the N is never cut). */
export const DEFAULT_COUNT_LINE_CHARS = 1_200;

export const COUNT_QUESTIONS = {
  instance: {
    type: 'noul',
    instructions: 'The sentence states one instance of the thing the question counts.',
  },
} as const satisfies Record<string, TypesafeQuestion>;

export interface CountCandidate {
  /** The day of the session the sentence was said in, as the block prints it. */
  sessionDate: string;
  sentence: string;
  /** Position in the reading order (session date, then sentence order): the tie-break. */
  order: number;
  /** Question content words the sentence shares. */
  overlap: number;
}

export interface CountStats {
  candidates: number;
  /** Candidates whose noul reached the threshold. */
  counted: number;
  /** Candidates answered by a live request. */
  calls: number;
  /** Candidates answered from the on-disk cache. */
  cached: number;
  /** Input tokens of the live requests (cached answers cost nothing). */
  inputTokens: number;
}

/**
 * The harness keeps a retrieved session as "user: …" / "assistant: …" lines, while the sentence
 * splitter keys on the capture format's "USER:" / "ASSISTANT:". Upper-casing the markers is
 * what makes the split keep the user's own turns and drop the role prefix.
 */
function withRoleMarkers(text: string): string {
  return text.replace(/^(user|assistant):/gim, (marker) => marker.toUpperCase());
}

/** The sentence's content words, canonical, as the question's terms are. */
function sentenceTerms(sentence: string): Set<string> {
  return new Set(
    sentence
      .replace(/[’‘`]/g, "'")
      .split(/[^A-Za-z0-9'-]+/)
      .filter(Boolean)
      .map((word) => canonicalWord(word.replace(/^['-]+|['-]+$/g, '').replace(/'.*$/, ''))),
  );
}

/**
 * The user's sentences that share a content word with the question, read in session-date then
 * sentence order, capped at `max` by descending overlap (ties: the earlier sentence), and
 * returned back in reading order so the line is deterministic.
 */
export function selectCountCandidates(
  question: string,
  sources: ReadonlyArray<{ ts: string; text: string }>,
  max: number = DEFAULT_COUNT_MAX,
): CountCandidate[] {
  const terms = questionTerms(question);
  const candidates: CountCandidate[] = [];
  const seen = new Set<string>();
  const ordered = [...sources].sort((left, right) => left.ts.localeCompare(right.ts));
  for (const source of ordered) {
    const day = source.ts.slice(0, 10).replace(/\//g, '-');
    for (const sentence of userSentences(withRoleMarkers(source.text))) {
      const key = `${day}\n${sentence}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const words = sentenceTerms(sentence);
      const overlap = terms.filter(
        (term) => words.has(term) || [...words].some((word) => sameWord(word, term)),
      ).length;
      if (overlap === 0) continue;
      candidates.push({ sessionDate: day, sentence, order: candidates.length, overlap });
    }
  }
  if (candidates.length <= max) return candidates;
  return [...candidates]
    .sort((left, right) => right.overlap - left.overlap || left.order - right.order)
    .slice(0, max)
    .sort((left, right) => left.order - right.order);
}

function short(sentence: string, maximum = 110): string {
  const text = sentence.replace(/\s+/g, ' ').trim();
  return text.length <= maximum ? text : `${text.slice(0, maximum - 1)}…`;
}

/**
 * The block line: `Counted N <thing> the history states: <date> — <sentence>; …`. The list is
 * cut to `maxChars`, the count itself never is. No counted candidate means no line.
 */
export function countedNoteLine(
  question: string,
  counted: ReadonlyArray<CountCandidate>,
  maxChars: number = DEFAULT_COUNT_LINE_CHARS,
): string {
  if (counted.length === 0) return '';
  const head = `Counted ${counted.length} ${countedThing(question)} the history states: `;
  const items = counted.map(({ sessionDate, sentence }) => `${sessionDate} — ${short(sentence)}`);
  const shown: string[] = [];
  let used = 0;
  for (const item of items) {
    if (shown.length > 0 && used + item.length + 2 > maxChars) break;
    shown.push(item);
    used += item.length + 2;
  }
  const rest = items.length - shown.length;
  return `${head}${shown.join('; ')}${rest > 0 ? `; … and ${rest} more not shown` : ''}`;
}

/**
 * One TypeSafe request per candidate, counted in code. A candidate whose state fails the
 * external-LLM safety check is not sent and not counted.
 */
export async function countHistoryInstances(params: {
  question: string;
  sources: ReadonlyArray<{ ts: string; text: string }>;
  nouls: TypesafeNouls;
  max?: number;
  threshold?: number;
  maxChars?: number;
}): Promise<{ line: string; counted: CountCandidate[]; stats: CountStats }> {
  const threshold = params.threshold ?? DEFAULT_COUNT_THRESHOLD;
  const candidates = selectCountCandidates(
    params.question,
    params.sources,
    params.max ?? DEFAULT_COUNT_MAX,
  );
  const stats: CountStats = {
    candidates: candidates.length,
    counted: 0,
    calls: 0,
    cached: 0,
    inputTokens: 0,
  };
  const judged = await Promise.all(
    candidates.map(async (candidate) => {
      const state = {
        question: params.question,
        sentence: candidate.sentence,
        session_date: candidate.sessionDate,
      };
      try {
        assertSafeForExternalLlm(JSON.stringify(state), 'TypeSafe count candidate');
      } catch {
        return false;
      }
      const answer = await params.nouls(state, COUNT_QUESTIONS);
      if (answer.cached) stats.cached += 1;
      else {
        stats.calls += 1;
        stats.inputTokens += answer.inputTokens;
      }
      return answer.nouls.instance! >= threshold;
    }),
  );
  const counted = candidates.filter((_candidate, index) => judged[index] === true);
  stats.counted = counted.length;
  return {
    line: countedNoteLine(params.question, counted, params.maxChars),
    counted,
    stats,
  };
}
