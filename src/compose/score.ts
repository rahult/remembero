/**
 * Scoring a Compose answer with no model in the loop: every answer is correct, confidently
 * wrong, or declined. Declining is never wrong — it is the honest answer when the evidence did
 * not reach the reader — and it is the correct answer to an unanswerable question.
 *
 *   entity / set — every gold item appears (any accepted spelling) and no distractor does
 *   number       — the gold value appears (spelled out, with separators, or as "1.41 million")
 *                  and no distractor value does
 *   decision     — a clear yes or no that matches
 *   unknown      — the answer declines
 */

import type { ComposeGold } from './questions.js';

/**
 * correct — asserts the gold; wrong — asserts a distractor or another value (confidently wrong);
 * declined — says it does not know; partial — answers without naming any candidate, or names the
 * gold only in part (a surname when the full name was asked, some members of a set and no wrong
 * ones). Only `wrong` counts against certainty.
 */
export type ComposeOutcome = 'correct' | 'wrong' | 'declined' | 'partial';

const DECLINE = [
  'do not know', "don't know", 'dont know', 'unknown', 'cannot be determined', 'cannot determine',
  'not stated', 'not mentioned', 'no information', 'not provided', 'not specified', 'unable to',
  'insufficient', 'no record', 'not recorded', 'does not say', 'not available', 'no such contract',
  'not found', 'cannot answer', 'not in the', 'do not say', 'do not state', 'does not state',
  'do not record', 'does not record', 'do not show', 'does not show', 'no mention',
];

export function normalise(text: string): string {
  return text
    .toLowerCase()
    .replace(/[*_`]/g, '')
    .replace(/[’‘]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Whole-word, case-insensitive presence. */
export function mentions(answer: string, phrase: string): boolean {
  const body = normalise(phrase);
  if (body === '') return false;
  return new RegExp(`(^|[^\\p{L}\\p{N}])${escapeRegex(body)}($|[^\\p{L}\\p{N}])`, 'u').test(normalise(answer));
}

export function declines(answer: string): boolean {
  const text = normalise(answer);
  return DECLINE.some((phrase) => text.includes(phrase));
}

/** Every number the answer states, with "million", "m" and "k" applied and separators removed. */
export function numbersIn(answer: string): number[] {
  const out: number[] = [];
  const text = normalise(answer);
  const pattern = /(\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.(\d+))?\s*(million|mil|m\b|thousand|k\b)?/g;
  for (const match of text.matchAll(pattern)) {
    const whole = match[1]!.replace(/[ ,]/g, '');
    const value = Number(`${whole}${match[2] === undefined ? '' : `.${match[2]}`}`);
    const unit = match[3];
    const scale = unit === undefined ? 1 : unit.startsWith('m') ? 1_000_000 : 1_000;
    if (Number.isFinite(value)) out.push(value * scale);
  }
  return out;
}

function sameNumber(a: number, b: number): boolean {
  // money is written in several forms; "1.41 million" for 1,410,000 must match, 1.4 must not
  return Math.abs(a - b) <= Math.max(0.5, Math.abs(b) * 0.0005);
}

function decision(answer: string): boolean | undefined {
  const text = normalise(answer);
  const no = /\b(no|not|exceed(?:ed|s)?|outside|beyond|lacked|unauthori[sz]ed|without authority)\b/.test(text);
  const yes = /\b(yes|within|authori[sz]ed|permitted|allowed)\b/.test(text);
  if (/^\s*no\b/.test(text)) return false;
  if (/^\s*yes\b/.test(text)) return true;
  if (no && !yes) return false;
  if (yes && !no) return true;
  return undefined;
}

/** Index of the earliest whole-word mention of any phrase, or -1. */
function firstIndex(answer: string, phrases: readonly string[]): number {
  const text = normalise(answer);
  let best = -1;
  for (const phrase of phrases) {
    const body = normalise(phrase);
    if (body === '') continue;
    const match = new RegExp(`(^|[^\\p{L}\\p{N}])(${escapeRegex(body)})($|[^\\p{L}\\p{N}])`, 'u').exec(text);
    if (match === null) continue;
    const at = match.index + match[1]!.length;
    if (best < 0 || at < best) best = at;
  }
  return best;
}

/** Everything before the first contrast: what the answer claims, not what it explains away. */
const CONTRAST = /\b(while|whereas|by contrast|however|in contrast|all other|the other suppliers|the remaining|other suppliers|did hold|held valid|had valid|held a valid)\b/;

export function scoreCompose(answer: string, gold: ComposeGold): ComposeOutcome {
  if (gold.kind === 'unknown') {
    if (declines(answer)) return 'correct';
    // repeating an ambiguous surname without claiming who it is: not a confident misidentification
    if ((gold.partial ?? []).some((p) => mentions(answer, p)) && !gold.distractors.some((d) => mentions(answer, d))) return 'partial';
    return 'wrong';
  }

  switch (gold.kind) {
    case 'entity': {
      // the answer's claim is whichever candidate it names first
      const goldAt = firstIndex(answer, gold.items.flat());
      const wrongAt = firstIndex(answer, gold.distractors);
      if (goldAt >= 0 && (wrongAt < 0 || goldAt < wrongAt)) return 'correct';
      if (wrongAt >= 0) return 'wrong';
      if ((gold.partial ?? []).some((p) => mentions(answer, p))) return 'partial';
      return declines(answer) ? 'declined' : 'partial';
    }
    case 'set': {
      const claim = normalise(answer).split(CONTRAST)[0] ?? '';
      const found = gold.items.filter((spellings) => spellings.some((s) => mentions(claim, s))).length;
      const wrong = gold.distractors.some((d) => mentions(claim, d));
      if (wrong) return 'wrong';
      if (found === gold.items.length) return 'correct';
      if (found > 0) return 'partial';
      return declines(answer) ? 'declined' : 'wrong';
    }
    case 'number': {
      // the first number that is either the gold or a known wrong value is the one asserted
      for (const n of numbersIn(answer)) {
        if (sameNumber(n, gold.number!)) return 'correct';
        if (gold.distractorNumbers.some((d) => sameNumber(n, d))) return 'wrong';
      }
      return declines(answer) ? 'declined' : 'wrong';
    }
    case 'decision': {
      const said = decision(answer);
      if (said === undefined) return declines(answer) ? 'declined' : 'partial';
      return said === gold.allowed ? 'correct' : 'wrong';
    }
  }
}
