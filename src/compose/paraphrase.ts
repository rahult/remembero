/**
 * Rewrite a question in other words without changing what it asks.
 *
 * Templated questions can be parsed by a regular expression, which flatters any planner. A
 * paraphrase is kept only if everything the answer depends on survives it: contract references
 * verbatim, dates in any of their spellings, and role, supplier and standard names. A paraphrase
 * that drops or alters one is discarded and the original question is used.
 */

import { dateSpellings } from './dates.js';
import type { ComposeQuestion } from './questions.js';

export const PARAPHRASE_PROMPT = `Rewrite the question below the way a compliance officer or auditor might actually ask it — different wording and sentence structure, same meaning. Keep every contract reference, date, role title, supplier name and standard exactly as it appears (a date may be written in another format). Do not add or remove conditions. Reply with the rewritten question only.`;

function squash(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ');
}

/** Why a paraphrase is not faithful to the question's parameters, or undefined when it is. */
export function paraphraseProblem(question: ComposeQuestion, paraphrase: string): string | undefined {
  const text = squash(paraphrase);
  if (text.trim().length < 10) return 'empty';
  for (const [name, value] of Object.entries(question.params)) {
    if (name === 'date') {
      if (!dateSpellings(Number(value)).some((s) => text.includes(squash(s)))) return `date ${value} lost`;
    } else if (!text.includes(squash(String(value)))) {
      return `${name} "${value}" lost`;
    }
  }
  if (question.family === 'authority' && !/\b(yes|no|whether|did|was|were)\b/.test(text)) return 'no longer a yes/no question';
  return undefined;
}
