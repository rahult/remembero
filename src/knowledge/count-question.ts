/**
 * Does a question ask for a number of things the user did or has? A deterministic reading of
 * the question text alone, for the code-counted note the reader is handed (TypeSafe's Jev
 * cannot count: the candidates are judged one at a time and added up in code).
 *
 * A question counts when it asks "how many X", "how many times", "the total number of X" or
 * "the count of X" about the user's own history. It does not count when it asks for a span of
 * time ("how long", "how many days"), an amount of one thing ("how much did I spend"), a
 * stated frequency ("how many days a week"), a property of one thing ("how many bedrooms does
 * my apartment have", "how many people were at the wedding") or the assistant's own earlier
 * answer ("how many options did you suggest"). No model is involved.
 */

/** Units that make "how many <unit>" a distance or a duration, not a tally of things. */
const TIME_UNIT = '(?:days?|weeks?|months?|years?|hours?|minutes?|nights?|decades?)';

/** The ways a question asks for a number of things. */
const COUNT_ASK: ReadonlyArray<RegExp> = [
  // "How many books have I read?", "How many times did I go?" ("times" is a tally, not a unit
  // of time); "how many days ago" is a distance and is excluded here
  new RegExp(`\\bhow many\\b(?!\\s+(?:of\\s+)?${TIME_UNIT}\\b)`),
  // "What is the total number of marathons I have run?"
  /\b(?:total|overall) (?:number|count) of\b/,
  // "What is the count of gyms I joined?", "the number of times I flew"
  /\b(?:the\s+)?(?:count|number) of\b/,
];

/** A span of time, an amount of one thing, or a clock reading: never a tally. */
const NOT_A_TALLY = /\bhow long\b|\bhow much\b|\bhow old\b|\bwhat time\b|\bhow far\b/;

/** "how many days a week", "how many cups per day": a rate the user simply told us. */
const RATE = /\b(?:a|per|each|every)\s+(?:day|week|month|year|night|session|time)\b/;

/** A question about the assistant's own earlier answer is recall, not a tally of the user's life. */
const ASSISTANT_RECALL =
  /\byou (?:told|mentioned|said|recommended|suggested|provided|gave|listed|wrote|created|shared)\b|\bremind me\b|\bour previous (?:chat|conversation)\b/;

/**
 * One thing's stated property, not a tally across the history: "How many bedrooms does my new
 * apartment have?", "How many people were at the wedding?", "How many pages is the novel?".
 * The tell is a stative verb after the counted noun phrase instead of something the user did.
 */
const SINGLE_DETAIL: ReadonlyArray<RegExp> = [
  /\bhow many\b[^?]*?\b(?:does|do)\s+(?:my|our|the|this|that|it|he|she|they)\b/,
  /\bhow many\b[^?]*?\b(?:was|were|is|are)\s+(?:there|at|in|on|inside|included)\b/,
  /\bhow many\s+[a-z-]+\s+(?:is|are|was|were)\s+(?:my|our|the|this|that|it)\b/,
];

/** The question is about the user's own history, not the world's. */
const ABOUT_THE_USER = /\b(?:i|we|my|our|me|us|i've|we've)\b/;

export function isCountQuestion(question: string): boolean {
  const lower = question.toLowerCase().replace(/[’`]/g, "'").trim();
  if (!ABOUT_THE_USER.test(lower)) return false;
  if (NOT_A_TALLY.test(lower)) return false;
  if (ASSISTANT_RECALL.test(lower)) return false;
  if (RATE.test(lower)) return false;
  if (SINGLE_DETAIL.some((rule) => rule.test(lower))) return false;
  return COUNT_ASK.some((rule) => rule.test(lower));
}

/** Words that end the counted noun phrase: an auxiliary, a subject, or a new phrase. */
const PHRASE_END = new Set(
  'have has had did do does am is are was were be been will would could should can may might i we you he she they it that which who whom whose in on at during since before after over from between until by with about while when where and or so than there'.split(
    ' ',
  ),
);

/**
 * The thing the question counts, verbatim: the words after "how many" (or "the number of") up
 * to the first auxiliary, subject or preposition, at most six. "How many different types of
 * tea have I tried?" gives "different types of tea". Falls back to "items".
 */
export function countedThing(question: string): string {
  const text = question.replace(/[’`]/g, "'").replace(/[?!.]+\s*$/, '').trim();
  const start =
    /\bhow many\s+/i.exec(text) ??
    /\b(?:total |overall )?(?:number|count) of\s+/i.exec(text);
  if (start === null) return 'items';
  const after = text.slice(start.index + start[0].length);
  const words: string[] = [];
  for (const word of after.split(/\s+/)) {
    const bare = word.replace(/[^A-Za-z0-9'-]/g, '').toLowerCase();
    if (bare === '' || PHRASE_END.has(bare)) break;
    words.push(word.replace(/[,;:]+$/, ''));
    if (words.length === 6) break;
  }
  // a phrase left hanging on "of" or an article says nothing on its own
  while (words.length > 0 && /^(?:of|the|a|an|my|our)$/i.test(words[words.length - 1]!)) {
    words.pop();
  }
  return words.length === 0 ? 'items' : words.join(' ');
}
