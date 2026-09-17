/**
 * What kind of question is this? The five answering decisions a memory system has to make
 * before it answers — how deep to retrieve, whether to reason about time, whether to read
 * the history in notes first, whether to personalise, and whether the assistant's own turns
 * are evidence — read from the question text alone.
 *
 * A benchmark hands them over as a `question_type` label; a deployed system has no label, so
 * a score that reads one describes the benchmark rather than the product. These are the
 * deterministic rules (no model): `isTemporalQuestion` for time, plus simple rules for
 * counting across events, a superseded value, a request for advice, and recall of what the
 * assistant said. `evals/typesafe-question-kind.ts` answers the same five with one model call.
 */

import { isRecommendationIntent } from './semantic-search.js';
import { isTemporalQuestion } from './temporal-question.js';

/** The five flags. They are independent: a question can be several kinds at once. */
export interface QuestionKind {
  /** The answer has to be gathered or counted over several separate things the user said. */
  aggregation: boolean;
  /** The question asks about time: a date, a distance, an order, or a past period. */
  temporal: boolean;
  /** The question asks for a value that changed, so an earlier statement is out of date. */
  update: boolean;
  /** The question asks for advice or a recommendation that should be personalised. */
  preference: boolean;
  /** The question asks what the assistant said, not what the user said. */
  assistantRecall: boolean;
}

/** Where a question's kind came from. */
export type QuestionKindSource =
  | 'label'
  | 'text'
  | 'typesafe'
  | 'typesafe-fallback';

/** How the harness decides a question's kind. */
export type QuestionClassification = 'label' | 'text' | 'typesafe';

/** A noul at or above this probability sets its flag. */
export const DEFAULT_QUESTION_KIND_THRESHOLD = 0.5;

/** The LongMemEval label each flag stands for, for the label route and for agreement counts. */
export const QUESTION_KIND_LABELS = {
  aggregation: 'multi-session',
  temporal: 'temporal-reasoning',
  update: 'knowledge-update',
  preference: 'single-session-preference',
  assistantRecall: 'single-session-assistant',
} as const satisfies Record<keyof QuestionKind, string>;

export const QUESTION_KIND_FLAGS = Object.keys(
  QUESTION_KIND_LABELS,
) as ReadonlyArray<keyof QuestionKind>;

/** Today's routing: the benchmark's own label, one flag per type. */
export function questionKindFromLabel(questionType: string): QuestionKind {
  return {
    aggregation: questionType === QUESTION_KIND_LABELS.aggregation,
    temporal: questionType === QUESTION_KIND_LABELS.temporal,
    update: questionType === QUESTION_KIND_LABELS.update,
    preference: questionType === QUESTION_KIND_LABELS.preference,
    assistantRecall: questionType === QUESTION_KIND_LABELS.assistantRecall,
  };
}

const TIME_UNIT = '(?:days?|weeks?|months?|years?|hours?|minutes?)';

/**
 * Counting, summing, comparing or listing over several things: a count or a total ("how many
 * museums did I visit", "how much did I spend in all"), a superlative over a set ("which
 * store did I spend the most at"), a difference between two of them, or a request for every
 * one of them. A count the user simply stated is not aggregation, but the question text
 * cannot tell the two apart, so a count always counts.
 */
const AGGREGATION: ReadonlyArray<RegExp> = [
  // "How many museums did I visit", "How much money have I spent"
  /\bhow (?:many|much)\b/,
  // "the total amount", "in total", "altogether", "combined", "on average"
  /\b(?:total|totals|altogether|in all|combined|overall|on average|average)\b/,
  // "Which store did I spend the most at", "the largest number of", "my longest trip";
  // "most recently" is an order, not a maximum over a set
  /\b(?:most(?! recent)|fewest|least|largest|smallest|highest|lowest|longest|shortest|biggest)\b/,
  // "How much more did I spend on Hawaii than Tokyo", "compared to", "the difference in price"
  /\b(?:more|less|fewer) .*\b(?:than|compared to)\b|\bcompared to\b|\bdifference (?:in|between)\b/,
  // "List all the books", "what are all the cities", "each of the trips"
  /\b(?:all (?:the|of|my)|every|each of)\b/,
  // "What percentage of packed shoes did I wear", "what share of"
  /\bpercentage\b|\bshare of\b/,
  // "the two hobbies that led me to", "the three road trip destinations": a named several
  /\b(?:the|my) (?:both|two|three|four|five|six|seven|eight|nine|ten)\b/,
  // "the number of times", "how many times"
  /\bnumber of\b|\bhow many times\b/,
];

/**
 * "How many days passed between A and B", "how many weeks did it take": counting time units
 * is arithmetic on two dates, not a count over a set — unless the question adds them up
 * ("how many hours in total did I spend driving to my three destinations combined").
 */
const COUNTED_TIME_UNITS = new RegExp(`\\bhow (?:many|much) ${TIME_UNIT}\\b`);
const SUMMED = /\b(?:total|altogether|in all|combined)\b/;

/**
 * A value that changed, so the remembered one may be stale: the question asks for what holds
 * now ("how many bikes do I currently own", "where is the painting hanging now", "the most
 * recent lens I bought", "how many pages have I read so far"), or names the change itself
 * ("did I switch to more water", "where did Rachel move to", "do I go more often than
 * before"). Counting inside a period is a window, not an update.
 */
const UPDATE: ReadonlyArray<RegExp> = [
  /\b(?:currently|current|now|nowadays|these days|at the moment|right now|to date|so far|as of now|up to now)\b/,
  /\bmost recent(?:ly)?\b|\blatest\b|\bnewest\b/,
  /\bstill\b|\bany more\b|\banymore\b/,
  /\b(?:switch(?:ed)?|change[ds]?|chang(?:ing|e)|updated?|upgrade[ds]?|replace[ds]?|move[ds]?|moved|relocat(?:ed|ion))\b/,
  /\bthan (?:i did )?(?:previously|before|used to)\b|\bused to\b/,
  /\b(?:same|different) \w+ (?:as|from|than)\b/,
  /\b(?:new|old) \w+\b.*\b(?:now|current)\b/,
  // a standing habit is a value that moves: "how often do I see my therapist"
  /\bhow often\b/,
  // "how long have I been using my Fitbit", "how long have my parents been staying with me"
  /\bhow long have (?:i|my|we)\b/,
  // a running tally the user keeps adding to: "how many issues have I finished reading",
  // "how many trips have I taken my camera on" (the present perfect, not a closed past)
  /\bhave i (?:been )?\w+(?:ed|en|ne|un|pt|de|wn)\b/,
];

/**
 * A question about the assistant's own earlier turn: what it said, recommended, listed or
 * provided. "I mentioned" is the user; "you mentioned" is the assistant.
 */
const ASSISTANT_RECALL: ReadonlyArray<RegExp> = [
  /\bremind me\b/,
  /\byou (?:told|mentioned|said|recommended|suggested|provided|gave|listed|explained|described|advised|wrote|shared|offered|made|proposed|noted)\b/,
  /\bdid you (?:say|mention|recommend|suggest|tell|list|provide|give|name|call)\b/,
  /\byou(?:'ve| have) (?:told|mentioned|recommended|suggested|provided|given|listed|explained)\b/,
  /\b(?:our|the) (?:previous|last|earlier|first) (?:chat|conversation|conversations|discussion|talk|session|exchange)\b/,
  /\bin our (?:chat|conversation)\b|\bwe (?:discussed|talked about|went over|covered)\b/,
  /\byour (?:recommendation|recommendations|suggestion|suggestions|advice|answer|reply|response|list|explanation)\b/,
  /\bthe (?:list|answer|recipe|plan) you\b/,
];

/**
 * Advice, a recommendation or an opinion to personalise, rather than a fact to look up.
 * `isRecommendationIntent` covers "recommend", "suggest", "advice", "tips", "ideas",
 * "should I", "what should", "decide", "choose", "looking for"; these add the openers
 * LongMemEval's preference questions use.
 */
const PREFERENCE: ReadonlyArray<RegExp> = [
  /\b(?:do|what do) you think\b/,
  /\bany (?:thoughts|pointers|guidance)\b/,
  /\bhelp me (?:pick|choose|decide|plan)\b/,
  /\bwhich .*\bwould you\b/,
  /\bcan you (?:help|propose)\b/,
];

/** A question that asks the assistant for something, not about the user's own history. */
export function isPreferenceQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'");
  return (
    isRecommendationIntent(lower) || PREFERENCE.some((rule) => rule.test(lower))
  );
}

/** Does the question ask what the assistant said in the past? */
export function isAssistantRecallQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'");
  return ASSISTANT_RECALL.some((rule) => rule.test(lower));
}

/** Does the answer have to be gathered over several separate statements? */
export function isAggregationQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'");
  if (COUNTED_TIME_UNITS.test(lower) && !SUMMED.test(lower)) return false;
  return AGGREGATION.some((rule) => rule.test(lower));
}

/** Does the question ask for a value that has since changed? */
export function isUpdateQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'");
  return UPDATE.some((rule) => rule.test(lower));
}

/**
 * The deterministic fallback: every flag from the question text, no model and no label.
 * A question the assistant is being asked to answer about itself is never counted as
 * aggregation, an update or a preference — its subject is the earlier reply.
 */
export function questionKindFromText(question: string): QuestionKind {
  const assistantRecall = isAssistantRecallQuestion(question);
  return {
    aggregation: !assistantRecall && isAggregationQuestion(question),
    temporal: isTemporalQuestion(question),
    update: !assistantRecall && isUpdateQuestion(question),
    preference: !assistantRecall && isPreferenceQuestion(question),
    assistantRecall,
  };
}
