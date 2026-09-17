/**
 * Does a question ask about time? A deterministic reading of the question text alone, for
 * choices a deployed memory system has to make without a benchmark's question-type label
 * (LongMemEval: temporal questions retrieve better by whole session than by turn).
 *
 * A question is temporal when it asks for a distance or duration between events, an order of
 * events, a date, or anchors what it asks about to a past period. Counting inside a period
 * ("how many plants did I buy last month") is aggregation, not time; so is a duration the user
 * simply told us ("how long is my commute"). No model is involved.
 */

import {
  MONTH_RE,
  holiday,
  resolveTemporalExpressions,
} from './computed-notes.js';

const TIME_UNIT = '(?:days?|weeks?|months?|years?)';
const NUMBER_WORD =
  '(?:\\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|a few|couple(?: of)?|a couple of|several)';
const WEEKDAY = '(?:sunday|monday|tuesday|wednesday|thursday|friday|saturday)';
/** An arbitrary past day the question's relative expressions are resolved against. */
const REFERENCE_DAY = '2000/06/15';

/** Rules that make a question temporal whatever else it asks. */
const DISTANCE_AND_ORDER: ReadonlyArray<RegExp> = [
  // "How many days passed between ...", "How many weeks had passed since ..."
  /\b(?:passed|elapsed)\b/,
  // "How many days ago did I ...", "How many months before my anniversary did ...";
  // not a frequency ("how many days a week"), an age ("how many years older", "how many
  // years will I be") or a sum of durations ("how many days in total", "all the films")
  new RegExp(
    `\\bhow many ${TIME_UNIT}\\b(?!\\s+(?:a|per|each|every|will|would|old)\\b)(?!.*\\b(?:older|younger|total|combined|all the)\\b)`,
  ),
  // "How long had I been bird watching when I attended ...": a duration up to another
  // event. A bare "how long have I been collecting X" is a remembered fact, not arithmetic.
  /\bhow long\b.*\b(?:when|before|after|until)\b/,
  // "Which event happened first, A or B?", "Who graduated first, second and third?",
  // "Which book did I finish reading first, 'X' or 'Y'?" (not "my first purchase")
  /\bfirst\s*(?:[,?]|$|\bor\b|\bamong\b|\bbetween\b)/,
  // "What was the first issue I had with my new car after its first service?"
  /\bfirst\b.*\b(?:after|since)\b/,
  // "What is the order of the three trips ...", "in the order from first to last"
  /\bthe order\b|\bin (?:what|which) order\b|\border in which\b/,
  // "... from earliest to latest"
  /\bearliest\b/,
  // "Which mode of transport did I use most recently, a bus or a train?" (the adverb; the
  // adjective in "my most recent trip" asks for the latest value, not an order)
  /\bmost recently\b/,
  // "Which happened earlier, ...?", "Did the trip come before or after the move?"
  /\b(?:happen(?:ed)?|came|come|occur(?:red)?) (?:first|last|earlier|later|before|after)\b/,
  /\bbefore or after\b/,
  // "When did I book the Airbnb?"
  /^when\b|\bwhen (?:did|was|were|do|does|had|have|is)\b/,
  // "What was the date on which ...", "On what day did I ...", "Which month did I start ..."
  /\bwhat (?:was |is )?the date\b|\b(?:what|which) date\b|\bon what day\b/,
  /\b(?:what|which) (?:month|year|day)\b(?! of the week)/,
];

/**
 * "How many charity events did I attend before the 'Run for the Cure' event?": what happened
 * before another event is an ordering question. Not when the question asks for a superseded
 * value ("my previous status before I got the current one", "my last name before I changed
 * it"): that is an update, not an order.
 */
const BOUNDED_BY_EVENT = /\b(?:before|after) (?:the|my|i)\b/;
const SUPERSEDED = /\b(?:previous|earlier|changed|updated|initial)\b|^before\b/;

/** A question about the assistant's own earlier answer is recall, whatever words it uses. */
const ASSISTANT_RECALL =
  /\bremind me\b|\byou (?:told|mentioned|said|recommended|suggested|provided|gave|wrote|created)\b|\bour previous (?:chat|conversation)\b/;

/**
 * Counting, summing or a clock time: a period or event bound in such a question is a window,
 * not the thing asked ("how many plants did I buy last month", "what time did I reach the
 * clinic on Monday", "the page count of the novels I finished in January").
 */
const WINDOWED = new RegExp(
  `\\bhow many\\b(?!\\s+${TIME_UNIT}\\b)|\\bhow much\\b|\\btotal\\b|\\baltogether\\b|\\bin all\\b|\\bcount\\b|\\bnumber of\\b|\\bwhat time\\b`,
);

/** Past periods and dates that anchor what is asked. */
const PAST_PERIOD: ReadonlyArray<RegExp> = [
  // "the past weekend", "last Tuesday", "in the past three months", "over the last few weeks"
  new RegExp(
    `\\b(?:last|past|previous)\\s+(?:${NUMBER_WORD}\\s+)?(?:${TIME_UNIT}|weekends?|night|summer|winter|spring|fall|autumn|${WEEKDAY})\\b`,
  ),
  // "a week ago", "on the Wednesday two months ago"
  new RegExp(`\\b${NUMBER_WORD}\\s+${TIME_UNIT}\\s+ago\\b`),
  // "the weekend before the party", "the day after the concert"
  /\bthe (?:day|night|week|weekend|month) (?:before|after)\b/,
  // "in March and April", "the first BBQ event in June", "during February"
  new RegExp(
    `\\b(?:in|during|since|from|until|by|early|late|mid)[- ](?:the month of )?${MONTH_RE}\\b`,
  ),
];

function mentionsPastPeriod(question: string, lower: string): boolean {
  if (PAST_PERIOD.some((rule) => rule.test(lower))) return true;
  // the computed-notes recognisers: "yesterday", "last Saturday", "March 15th", "on Monday",
  // "last summer"; only past or absolute dates count ("tonight", "next week" are plans)
  const reference = REFERENCE_DAY.replace(/\//g, '-');
  const dated = resolveTemporalExpressions(question, REFERENCE_DAY).some(
    (event) => event.kind === 'absolute' || event.iso < reference,
  );
  if (dated) return true;
  // "on Valentine's day", "the week before Thanksgiving"
  const words = lower.replace(/[?.,!]/g, ' ').split(/\s+/).filter(Boolean);
  for (let start = 0; start < words.length; start++) {
    for (let length = 1; length <= 4 && start + length <= words.length; length++) {
      if (holiday(words.slice(start, start + length).join(' '), 2000) !== undefined) {
        return true;
      }
    }
  }
  return false;
}

export function isTemporalQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'").trim();
  if (ASSISTANT_RECALL.test(lower)) return false;
  if (DISTANCE_AND_ORDER.some((rule) => rule.test(lower))) return true;
  if (BOUNDED_BY_EVENT.test(lower) && !SUPERSEDED.test(lower)) {
    // a count bounded by an event still asks for an order; a clock time does not
    if (!/\bwhat time\b/.test(lower)) return true;
  }
  if (WINDOWED.test(lower)) return false;
  return mentionsPastPeriod(question, lower);
}
