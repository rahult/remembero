/**
 * Computed notes: the English structure a small reader gets wrong, done deterministically.
 *
 * A reader that finds the right sentences still miscounts, mis-sums and mis-subtracts dates
 * ("Feb 14 to Mar 15" answered as one day). This module resolves every temporal expression in
 * the user's turns against the session it was said in, computes distances to the question date
 * and gaps between dated events, reads quantities with their units, and totals them. The reader
 * is handed the arithmetic with the sentence each number came from, so it copies instead of
 * computing. No model is involved.
 */

const MONTHS: Record<string, number> = {
  jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4, may: 5,
  jun: 6, june: 6, jul: 7, july: 7, aug: 8, august: 8, sep: 9, sept: 9, september: 9,
  oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12,
};
export const MONTH_RE = '(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)';
const WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
const SMALL_NUMBERS: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
  ten: 10, eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30, forty: 40, fifty: 50,
  couple: 2, few: 3, several: 3, half: 0.5, dozen: 12,
};
export const STOPWORDS: ReadonlySet<string> = new Set(
  'the a an and or of to in on at for with by from as is are was were be been being i me my mine we our you your it its this that these those what which who whom whose how many much long when where did do does done have has had between total combined spent spend take took get got go went ago about into over than then there their they them his her hers he she him days day weeks week months month years year hours hour minutes minute time times number amount percentage percent'.split(' '),
);

export interface DatedEvent {
  iso: string;
  expression: string;
  sentence: string;
  sessionDay: string;
  /** 'month': a month with no day ("in January"); 'undated': a past event with no date, dated by its session. */
  kind: 'absolute' | 'relative' | 'weekday' | 'month' | 'undated';
  approximate?: boolean;
  assumedYear?: boolean;
  /** Where the expression starts in the sentence. */
  index?: number;
  /** The year was assumed and then moved back one: the sentence is past tense and the session year put it later than the session. */
  yearRolledBack?: boolean;
  /** "for three months now": the start of something still going on. */
  start?: boolean;
  /** "in January": iso is the first day of the month. */
  monthOnly?: boolean;
  /** "a week before I got my new phone": the dated event this one is counted from. */
  anchor?: { iso: string; expression: string; sentence: string };
  /** Days either side the date may be off by, for rough expressions. */
  slack?: number;
}

export interface Quantity {
  value: number;
  unit: string;
  raw: string;
  sentence: string;
}

const DAY_MS = 86_400_000;

function utc(y: number, m: number, d: number): number {
  return Date.UTC(y, m - 1, d);
}

function toIso(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function dayOf(ts: string): string {
  const match = /(\d{4})[/-](\d{2})[/-](\d{2})/.exec(ts);
  return match === null ? ts.slice(0, 10) : `${match[1]}-${match[2]}-${match[3]}`;
}

function msOf(day: string): number {
  return utc(Number(day.slice(0, 4)), Number(day.slice(5, 7)), Number(day.slice(8, 10)));
}

const IRREGULAR: Record<string, string> = {
  met: 'meet', got: 'get', gotten: 'get', bought: 'buy', went: 'go', gone: 'go', began: 'begin', begun: 'begin',
  ran: 'run', saw: 'see', seen: 'see', took: 'take', taken: 'take', made: 'make', came: 'come', found: 'find',
  left: 'leave', lost: 'lose', did: 'do', done: 'do', became: 'become', gave: 'give', given: 'give', flew: 'fly',
  flown: 'fly', felt: 'feel', heard: 'hear', kept: 'keep', sold: 'sell', spent: 'spend', told: 'tell', won: 'win',
  wrote: 'write', written: 'write', drove: 'drive', driven: 'drive', ate: 'eat', eaten: 'eat', paid: 'pay',
  sent: 'send', built: 'build', fell: 'fall', caught: 'catch', taught: 'teach', thought: 'think', brought: 'bring',
  had: 'have', was: 'be', were: 'be', sat: 'sit', stood: 'stand', wore: 'wear', worn: 'wear', swam: 'swim',
};
/** Past forms that end in -ed but usually describe a plan or a feeling, not a finished event. */
const NOT_PAST_ED = new Set('booked scheduled planned reserved interested excited supposed expected registered signed invited needed wanted hoped used tired prepared concerned worried pleased thrilled'.split(' '));
const FUTURE_CUE = /\b(?:will|won't|\w+['’]ll|going to|gonna|plan(?:ning)? to|planning|planned|upcoming|next|scheduled|booked|reserved|tickets?|coming up|looking forward|hope to|hoping to|want to|would like|appointment|deadline|due|(?:i'm|i’m|i am|we're|we’re|we are)\s+(?:flying|going|heading|traveling|travelling|visiting|moving|leaving|driving))\b/i;

/**
 * A word as the notes compare it: lower case, an irregular past mapped to its base, plurals
 * singular. A stem made by removing -ed or -ing ends in "~" ("loved" → "lov~",
 * "cancelled" → "cancel~"), so it can match "love" and "cancel" without "care" matching "car";
 * compare with `sameWord` / `hasWord`.
 */
export function canonicalWord(word: string): string {
  let w = word.toLowerCase();
  w = IRREGULAR[w] ?? w;
  if (w.length > 4 && w.endsWith('ies')) w = `${w.slice(0, -3)}y`;
  else if (w.length > 4 && /(?:ss|sh|ch|x|z)es$/.test(w)) w = w.slice(0, -2);
  else if (w.length > 3 && w.endsWith('s') && !/(?:ss|us|is)$/.test(w)) w = w.slice(0, -1);
  for (const suffix of ['ing', 'ed']) {
    if (w.endsWith(suffix) && w.length - suffix.length >= 3) {
      let stem = w.slice(0, -suffix.length);
      if (/([b-df-hj-np-tv-z])\1$/.test(stem) && !/(?:ll|ss)$/.test(stem)) stem = stem.slice(0, -1);
      else if (/ll$/.test(stem) && stem.length > 4) stem = stem.slice(0, -1);
      return `${stem}~`;
    }
  }
  return w;
}

/** Two canonical words name the same word: equal, or a stem and its base ("lov~" and "love"). */
export function sameWord(a: string, b: string): boolean {
  if (a === b) return true;
  const stemA = a.endsWith('~');
  const stemB = b.endsWith('~');
  if (stemA === stemB) return false;
  const [stem, base] = stemA ? [a.slice(0, -1), b] : [b.slice(0, -1), a];
  return base === stem || base === `${stem}e` || (base.endsWith('e') && base.slice(0, -1) === stem);
}

/** Does a set of canonical words contain this word, as itself or as its stem/base? */
function hasWord(words: ReadonlySet<string>, word: string): boolean {
  if (words.has(word)) return true;
  if (word.endsWith('~')) {
    const stem = word.slice(0, -1);
    return words.has(stem) || words.has(`${stem}e`);
  }
  return words.has(`${word}~`) || (word.endsWith('e') && words.has(`${word.slice(0, -1)}~`));
}

/** The word without its stem marker, for stopword checks. */
function plainWord(word: string): string {
  return word.endsWith('~') ? word.slice(0, -1) : word;
}

function tokenize(text: string): string[] {
  return text
    .replace(/[’‘`]/g, "'")
    .replace(/'s\b/gi, '')
    .split(/[^A-Za-z0-9'-]+/)
    .map((w) => w.replace(/^['-]+|['-]+$/g, '').replace(/'.*$/, ''))
    .filter(Boolean);
}

const wordCache = new Map<string, Set<string>>();
/** Every word of a text in canonical form, for whole-word matching. */
function wordSet(text: string): Set<string> {
  let set = wordCache.get(text);
  if (set === undefined) {
    set = new Set(tokenize(text).map(canonicalWord));
    if (wordCache.size > 5000) wordCache.clear();
    wordCache.set(text, set);
  }
  return set;
}

/** Verbs that name the same kind of event: "received" and "got" both acquire something. */
const VERB_CLASSES: ReadonlyArray<ReadonlySet<string>> = [
  ['get', 'receive', 'buy', 'purchase', 'arrive', 'deliver', 'acquire', 'give', 'gift', 'adopt'],
  ['start', 'begin', 'launch'],
  ['finish', 'complete', 'end'],
  ['meet', 'introduce'],
  ['lose', 'misplace'],
  ['move', 'relocate'],
  ['fix', 'repair', 'service', 'replace', 'upgrade'],
  ['attend', 'visit', 'join'],
].map((words) => new Set(words.map(canonicalWord)));

function verbClasses(text: string): Set<number> {
  const words = wordSet(text);
  const out = new Set<number>();
  VERB_CLASSES.forEach((cls, i) => {
    for (const w of cls) if (hasWord(words, w)) out.add(i);
  });
  return out;
}

/** Past-tense words of a text, in canonical form. */
function pastVerbs(text: string): Set<string> {
  const out = new Set<string>();
  for (const raw of tokenize(text)) {
    const w = raw.toLowerCase();
    if ((IRREGULAR[w] !== undefined && w !== 'had' && w !== 'was' && w !== 'were') || (w.length >= 5 && w.endsWith('ed') && !NOT_PAST_ED.has(w))) out.add(canonicalWord(w));
  }
  return out;
}

/** Clause boundaries: ", and", ", but", ", so", ";", " - ". A relative clause ("which I started") stays with its antecedent. */
function clauseSpans(sentence: string): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  const boundary = /,\s+(?:and|but|so)\s+|;\s*|\s+[-–—]\s+/g;
  let start = 0;
  let m: RegExpExecArray | null;
  while ((m = boundary.exec(sentence)) !== null) {
    spans.push([start, m.index]);
    start = m.index + m[0].length;
  }
  spans.push([start, sentence.length]);
  return spans;
}

function clauseAt(sentence: string, index: number): [number, number] {
  return clauseSpans(sentence).find(([from, to]) => index >= from && index < to) ?? [0, sentence.length];
}

/** Is the event at `index` told in the past tense? Only the clause up to the expression counts. */
function toldAsPast(sentence: string, index: number): boolean {
  if (FUTURE_CUE.test(sentence)) return false;
  const [from] = clauseAt(sentence, index);
  const before = sentence.slice(from, index);
  return pastVerbs(before).size > 0 || /\balready\b/i.test(before) || /\b(?:was|were)\b/i.test(before);
}

/** User turns only: the assistant's hypotheticals must not become the user's dates. */
export function userSentences(text: string): string[] {
  return userTurns(text).flat();
}

/** The user's turns, each split into sentences. */
export function userTurns(text: string): string[][] {
  const out: string[][] = [];
  // split by turn marker so an assistant turn's later paragraphs keep their role
  const marker = /^(USER|ASSISTANT):\s*/gm;
  const turns: Array<{ role: string; body: string }> = [];
  let m: RegExpExecArray | null;
  let last: { role: string; start: number } | undefined;
  while ((m = marker.exec(text)) !== null) {
    if (last) turns.push({ role: last.role, body: text.slice(last.start, m.index) });
    last = { role: m[1]!, start: m.index + m[0].length };
  }
  if (last) turns.push({ role: last.role, body: text.slice(last.start) });
  else turns.push({ role: 'USER', body: text });
  for (const { role, body } of turns) {
    if (role !== 'USER') continue;
    const sentences: string[] = [];
    for (const sentence of body.split(/(?<=[.!?])\s+|\n+/)) {
      const s = sentence.trim();
      if (s.length > 2) sentences.push(s);
    }
    if (sentences.length > 0) out.push(sentences);
  }
  return out;
}

function nthWeekday(y: number, mo: number, weekday: number, n: number): number {
  const first = new Date(utc(y, mo, 1)).getUTCDay();
  const day = 1 + ((weekday - first + 7) % 7) + (n - 1) * 7;
  return utc(y, mo, day);
}

function lastWeekday(y: number, mo: number, weekday: number): number {
  const lastDay = new Date(utc(y, mo + 1, 0)).getUTCDate();
  const last = new Date(utc(y, mo, lastDay)).getUTCDay();
  return utc(y, mo, lastDay - ((last - weekday + 7) % 7));
}

/** US holidays with fixed or rule-based dates, as a UTC ms value in the given year. */
export function holiday(name: string, y: number): number | undefined {
  const n = name.toLowerCase().replace(/[’']/g, "'").trim();
  if (n === 'thanksgiving') return nthWeekday(y, 11, 4, 4);
  if (n === 'black friday') return nthWeekday(y, 11, 4, 4) + DAY_MS;
  if (n === 'christmas' || n === 'christmas day') return utc(y, 12, 25);
  if (n === 'christmas eve') return utc(y, 12, 24);
  if (/^new year'?s? day$/.test(n)) return utc(y, 1, 1);
  if (/^new year'?s? eve$/.test(n)) return utc(y, 12, 31);
  if (n === 'halloween') return utc(y, 10, 31);
  if (n === 'independence day' || n === 'the fourth of july' || n === 'july 4th') return utc(y, 7, 4);
  if (/^valentine'?s? day$/.test(n)) return utc(y, 2, 14);
  if (n === 'labor day') return nthWeekday(y, 9, 1, 1);
  if (n === 'memorial day') return lastWeekday(y, 5, 1);
  return undefined;
}

function number(word: string): number | undefined {
  const w = word.toLowerCase();
  if (/^\d+(?:[.,]\d+)?$/.test(w)) return Number(w.replace(/,/g, ''));
  return SMALL_NUMBERS[w];
}

const COUNT_RE = '\\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|couple of|a couple of|few|a few|several|half a';

function parseCount(text: string): number | undefined {
  return number(text.replace(/^a /, '').replace(/ of$/, '').replace(/ a$/, '')) ?? SMALL_NUMBERS[text.split(' ').pop()!];
}

function unitDays(n: number, unit: string): number {
  const u = unit.toLowerCase();
  return Math.round(u === 'day' ? n : u === 'week' ? n * 7 : u === 'month' ? n * 30.44 : n * 365.25);
}

/** `ms` moved by n units (negative n goes back); whole months and years move on the calendar. */
function shiftMs(ms: number, n: number, unit: string): number {
  const u = unit.toLowerCase();
  if ((u === 'month' || u === 'year') && Number.isInteger(n)) {
    const d = new Date(ms);
    const months = d.getUTCMonth() + (u === 'month' ? n : 12 * n);
    const y = d.getUTCFullYear() + Math.floor(months / 12);
    const mo = ((months % 12) + 12) % 12;
    const last = new Date(Date.UTC(y, mo + 1, 0)).getUTCDate();
    return Date.UTC(y, mo, Math.min(d.getUTCDate(), last));
  }
  return ms + Math.sign(n) * unitDays(Math.abs(n), u) * DAY_MS;
}

const VAGUE_COUNT = /\b(?:few|several|couple)\b/i;

/**
 * Every temporal expression in the user's turns, resolved against the session day. With
 * `extended`, also "for three months now" (a start date) and "in January" (a month); those
 * are for the notes block, not for callers that read a question's own time reference.
 */
export function resolveTemporalExpressions(
  text: string,
  sessionTs: string,
  options: {
    extended?: boolean;
    /** Dated events from other sessions a chained expression may count from when this text has none. */
    anchors?: DatedEvent[];
  } = {},
): DatedEvent[] {
  const sessionDay = dayOf(sessionTs);
  const base = msOf(sessionDay);
  const year = Number(sessionDay.slice(0, 4));
  const events: DatedEvent[] = [];
  const pending: Array<{
    sentence: string;
    index: number;
    expression: string;
    count: number;
    unit: string;
    sign: number;
    approximate: boolean;
    slack: number;
    reference: [number, number] | undefined;
    terms: string[];
  }> = [];
  const push = (e: DatedEvent) => {
    if (!events.some((x) => x.iso === e.iso && x.sentence === e.sentence)) events.push(e);
  };
  // "the Walk I did on May 15th", said in February: the session year would put a finished
  // event in the future, so the year before is meant
  const assumed = (y: number, mo: number, d: number, sentence: string, index: number) => {
    const ms = utc(y, mo, d);
    if (ms > base && toldAsPast(sentence, index)) return { ms: utc(y - 1, mo, d), yearRolledBack: true as const };
    return { ms };
  };
  for (const sentence of userSentences(text)) {
    const s = sentence;
    let m: RegExpExecArray | null;
    // ISO and numeric dates
    const iso = /\b(\d{4})-(\d{2})-(\d{2})\b/g;
    while ((m = iso.exec(s)) !== null) push({ iso: `${m[1]}-${m[2]}-${m[3]}`, expression: m[0], sentence, sessionDay, kind: 'absolute', index: m.index });
    const us = /\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b/g;
    while ((m = us.exec(s)) !== null) {
      const y = m[3].length === 2 ? 2000 + Number(m[3]) : Number(m[3]);
      push({ iso: toIso(utc(y, Number(m[1]), Number(m[2]))), expression: m[0], sentence, sessionDay, kind: 'absolute', index: m.index });
    }
    // "on 2/15" without a year: month/day, the year assumed from the session
    const usShort = /(?<![\d/])(\d{1,2})\/(\d{1,2})(?![\d/])/g;
    while ((m = usShort.exec(s)) !== null) {
      const mo = Number(m[1]);
      const d = Number(m[2]);
      if (mo < 1 || mo > 12 || d < 1 || d > 31) continue;
      const { ms, yearRolledBack } = assumed(year, mo, d, s, m.index);
      push({ iso: toIso(ms), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: true, index: m.index, ...(yearRolledBack ? { yearRolledBack } : {}) });
    }
    // "March 7th, 2023" / "March 7" / "7 March 2023" / "7th of March"
    const monthDay = new RegExp(`\\b(${MONTH_RE})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b`, 'gi');
    while ((m = monthDay.exec(s)) !== null) {
      const mo = MONTHS[m[1].toLowerCase()];
      if (!mo) continue;
      // "the March 15th issue", "the June 3 edition": a date naming a thing, not when it happened
      if (/^\s+(issue|edition|newsletter|magazine|episode|release|deadline|version)\b/i.test(s.slice(m.index + m[0].length))) continue;
      if (m[3]) push({ iso: toIso(utc(Number(m[3]), mo, Number(m[2]))), expression: m[0], sentence, sessionDay, kind: 'absolute', index: m.index });
      else {
        const { ms, yearRolledBack } = assumed(year, mo, Number(m[2]), s, m.index);
        push({ iso: toIso(ms), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: true, index: m.index, ...(yearRolledBack ? { yearRolledBack } : {}) });
      }
    }
    const dayMonth = new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(${MONTH_RE})\\.?(?:,?\\s+(\\d{4}))?\\b`, 'gi');
    while ((m = dayMonth.exec(s)) !== null) {
      const mo = MONTHS[m[2].toLowerCase()];
      if (!mo) continue;
      if (m[3]) push({ iso: toIso(utc(Number(m[3]), mo, Number(m[1]))), expression: m[0], sentence, sessionDay, kind: 'absolute', index: m.index });
      else {
        const { ms, yearRolledBack } = assumed(year, mo, Number(m[1]), s, m.index);
        push({ iso: toIso(ms), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: true, index: m.index, ...(yearRolledBack ? { yearRolledBack } : {}) });
      }
    }
    // relative to the session day
    const fixed: Array<[RegExp, number]> = [
      [/\bthe day before yesterday\b/gi, -2],
      [/\byesterday\b/gi, -1],
      [/\blast night\b/gi, -1],
      [/\b(?:today|tonight|this (?:morning|afternoon|evening))\b/gi, 0],
      [/\btomorrow\b/gi, 1],
    ];
    for (const [re, delta] of fixed) {
      while ((m = re.exec(s)) !== null) push({ iso: toIso(base + delta * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', index: m.index });
    }
    // "two days before" counts from the session only when nothing follows it; "a week before I
    // got my new phone" counts from that event (the chains below)
    const ago = new RegExp(`\\b(${COUNT_RE})\\s+(day|week|month|year)s?\\s+(ago|earlier|before(?=\\s*(?:$|[,.;:!?)]|[-–—]|(?:and|but|so|that|then|when)\\b)))`, 'gi');
    while ((m = ago.exec(s)) !== null) {
      const n = parseCount(m[1]!);
      if (n === undefined) continue;
      const unit = m[2]!.toLowerCase();
      const days = unitDays(n, unit);
      const vague = VAGUE_COUNT.test(m[1]!);
      const approximate = vague || (unit !== 'day' && unit !== 'week');
      push({ iso: toIso(shiftMs(base, -n, unit)), expression: m[0], sentence, sessionDay, kind: 'relative', approximate, index: m.index, ...(approximate ? { slack: Math.round(days * (vague ? 0.33 : 0.15)) } : {}) });
    }
    const ahead = /\b(?:in\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?|(\d+|a|one|two|three|four|five|six)\s+(day|week|month|year)s?\s+from now)\b/gi;
    while ((m = ahead.exec(s)) !== null) {
      const n = number(m[1] ?? m[3] ?? '');
      const unit = (m[2] ?? m[4] ?? '').toLowerCase();
      if (n === undefined || !unit) continue;
      const days = unit === 'day' ? n : unit === 'week' ? n * 7 : unit === 'month' ? n * 30.44 : n * 365.25;
      push({ iso: toIso(base + Math.round(days) * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: unit !== 'day' && unit !== 'week', index: m.index, ...(unit !== 'day' && unit !== 'week' ? { slack: Math.round(days * 0.15) } : {}) });
    }
    const lastUnit = /\blast (week|month|year)\b/gi;
    while ((m = lastUnit.exec(s)) !== null) {
      const days = m[1].toLowerCase() === 'week' ? 7 : m[1].toLowerCase() === 'month' ? 30 : 365;
      push({ iso: toIso(base - days * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: true, index: m.index, slack: days === 7 ? 3 : days === 30 ? 15 : 60 });
    }
    // seasons: "last summer" → the middle of that season in the previous year, approximate
    const season = /\b(last|this|next)\s+(summer|winter|spring|fall|autumn)\b/gi;
    while ((m = season.exec(s)) !== null) {
      const mid: Record<string, [number, number]> = { spring: [4, 15], summer: [7, 15], fall: [10, 15], autumn: [10, 15], winter: [1, 15] };
      const [mo, d] = mid[m[2].toLowerCase()]!;
      let y = year;
      if (m[1].toLowerCase() === 'last') y = utc(year, mo, d) < base ? year : year - 1;
      if (m[1].toLowerCase() === 'last' && utc(year, mo, d) < base) y = year;  // this year's season already passed
      if (m[1].toLowerCase() === 'last' && utc(year, mo, d) >= base) y = year - 1;
      if (m[1].toLowerCase() === 'next') y = utc(year, mo, d) > base ? year : year + 1;
      push({ iso: toIso(utc(y, mo, d)), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: true, index: m.index, slack: 45 });
    }
    // anchored offsets: "a week before Black Friday", "two days after Christmas", "3 days before March 7"
    const anchored = new RegExp(`\\b(\\d+|a|an|one|two|three|four|five|six|seven|ten)\\s+(day|week|month)s?\\s+(before|after|prior to|following)\\s+(black friday|thanksgiving|christmas(?: day)?|christmas eve|new year'?s? day|new year'?s? eve|halloween|independence day|the fourth of july|july 4th|valentine'?s? day|labor day|memorial day|(?:${MONTH_RE})\\.?\\s+\\d{1,2}(?:st|nd|rd|th)?(?:,?\\s+\\d{4})?)`, 'gi');
    const anchoredAt = new Set<number>();
    while ((m = anchored.exec(s)) !== null) {
      anchoredAt.add(m.index);
      const n = number(m[1]);
      if (n === undefined) continue;
      const unit = m[2].toLowerCase();
      const days = unit === 'day' ? n : unit === 'week' ? n * 7 : Math.round(n * 30.44);
      const sign = /before|prior/i.test(m[3]) ? -1 : 1;
      const anchor = holiday(m[4], year) ?? (() => {
        const mm = new RegExp(`(${MONTH_RE})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?`, 'i').exec(m[4]);
        if (!mm) return undefined;
        const mo = MONTHS[mm[1].toLowerCase()];
        return mo ? utc(mm[3] ? Number(mm[3]) : year, mo, Number(mm[2])) : undefined;
      })();
      if (anchor === undefined) continue;
      push({ iso: toIso(anchor + sign * days * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: unit === 'month', index: m.index, ...(unit === 'month' ? { slack: Math.round(days * 0.15) } : {}) });
    }
    // "a week before I got my new phone", "had to book three months in advance": counted from
    // another dated event, found once every sentence is read
    const chain = new RegExp(`\\b(${COUNT_RE})\\s+(day|week|month|year)s?\\s+(?:(in advance)(?:\\s+of\\s+([^,.;!?]+))?|(before|after|prior to|following)\\s+(?!(?:that|then|and|but|so|when)\\b)([^,.;!?]+))`, 'gi');
    while ((m = chain.exec(s)) !== null) {
      if (anchoredAt.has(m.index)) continue;
      const n = parseCount(m[1]!);
      if (n === undefined) continue;
      const unit = m[2]!.toLowerCase();
      const reference = m[4] ?? m[6];
      const referenceStart = reference === undefined ? -1 : m.index + m[0].length - reference.length;
      const vague = VAGUE_COUNT.test(m[1]!);
      const days = unitDays(n, unit);
      pending.push({
        sentence,
        index: m.index,
        expression: m[0],
        count: n,
        unit,
        sign: m[5] !== undefined && /after|following/i.test(m[5]) ? 1 : -1,
        approximate: vague || unit === 'month' || unit === 'year',
        slack: vague ? Math.round(days * 0.33) : unit === 'month' || unit === 'year' ? Math.round(days * 0.15) : 0,
        reference: reference === undefined ? undefined : [referenceStart, referenceStart + reference.length],
        // the referenced event's words: the phrase after "before", or for a bare "in advance"
        // the rest of the sentence ("for my best friend's wedding")
        terms: questionTerms(reference ?? `${s.slice(0, m.index)} ${s.slice(m.index + m[0].length)}`),
      });
    }
    if (options.extended) {
      // "I've been getting into bird watching for about three months now": started about then
      const since = new RegExp(`\\bfor\\s+(?:about\\s+|around\\s+|almost\\s+|nearly\\s+|over\\s+|roughly\\s+)?(the\\s+(?:past|last)\\s+)?(?:about\\s+|around\\s+)?(${COUNT_RE})\\s+(day|week|month|year)s?(\\s+now)?\\b`, 'gi');
      // present perfect only: "I had been training for six months before the marathon" is not now
      const ongoing = /(?:['’]ve|\bhave|\bhas)\s+(?:been|lived|worked|known|had|owned|played|practiced|practised|studied|used|collected)\b/i;
      while ((m = since.exec(s)) !== null) {
        if (s.trim().endsWith('?') || /\bhad\s+been\b/i.test(s) || !(m[4] || ongoing.test(s)) || FUTURE_CUE.test(s)) continue;
        const n = parseCount(m[2]!);
        if (n === undefined) continue;
        const days = unitDays(n, m[3]!);
        const slack = Math.max(1, Math.round(days * (VAGUE_COUNT.test(m[2]!) ? 0.33 : 0.15)));
        push({ iso: toIso(shiftMs(base, -n, m[3]!)), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: true, start: true, index: m.index, slack });
      }
      // "in January": the month, in the year the tense points to
      const monthOnly = new RegExp(`\\b(in|last|this|since|during|early|late|mid)[- ](?:the month of\\s+)?(${MONTH_RE})\\b(?!\\.?\\s*\\d{1,2}(?:st|nd|rd|th)?\\b)(?:,?\\s+(\\d{4}))?`, 'gi');
      while ((m = monthOnly.exec(s)) !== null) {
        if (m[2] === 'may' || m[2] === 'mar') continue;  // the verb, not the month
        const mo = MONTHS[m[2]!.toLowerCase()];
        if (!mo) continue;
        const sessionMonth = Number(sessionDay.slice(5, 7));
        let y = year;
        if (m[3]) y = Number(m[3]);
        else if (m[1]!.toLowerCase() === 'this') y = year;
        // only the words before the month tell its tense ("since late July, I'm looking forward")
        else if (FUTURE_CUE.test(s.slice(0, m.index + m[0].length))) y = mo >= sessionMonth ? year : year + 1;
        else y = mo <= sessionMonth ? year : year - 1;
        push({ iso: `${y}-${String(mo).padStart(2, '0')}-01`, expression: m[0], sentence, sessionDay, kind: 'month', approximate: true, monthOnly: true, index: m.index, ...(m[3] ? {} : { assumedYear: true }) });
      }
    }
    // weekdays: "last Saturday", "on Monday", "this Tuesday" → the most recent occurrence before the session day
    const weekday = /\b(last|this|on|next)\s+(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/gi;
    while ((m = weekday.exec(s)) !== null) {
      const target = WEEKDAYS.indexOf(m[2].toLowerCase());
      const current = new Date(base).getUTCDay();
      let delta = (current - target + 7) % 7;
      if (delta === 0) delta = 7;
      let iso: string;
      if (m[1].toLowerCase() === 'next') iso = toIso(base + ((target - current + 7) % 7 || 7) * DAY_MS);
      else iso = toIso(base - delta * DAY_MS);
      push({ iso, expression: m[0], sentence, sessionDay, kind: 'weekday', index: m.index });
    }
  }
  const plain = events.filter((e) => !e.start);
  for (const c of pending) {
    // a date inside the referenced phrase is the anchor itself
    let anchor = c.reference === undefined ? undefined : plain.find((e) => e.sentence === c.sentence && e.index !== undefined && e.index >= c.reference![0] && e.index < c.reference![1]);
    // otherwise the one dated event elsewhere that shares the most words with the reference,
    // in this session first, then in the others; a tie between different dates means the
    // reference is ambiguous, so nothing is dated
    const need = c.reference === undefined ? 2 : 1;
    const pick = (pool: DatedEvent[]) => {
      const scored = pool
        .filter((e) => e.sentence !== c.sentence && !e.start && e.anchor === undefined && e.kind !== 'undated')
        .map((e) => ({ e, score: c.terms.filter((t) => hasWord(wordSet(e.sentence), t)).length }))
        .filter((x) => x.score >= need);
      const top = Math.max(0, ...scored.map((x) => x.score));
      const best = scored.filter((x) => x.score === top);
      return best.length === 0 || new Set(best.map((x) => x.e.iso)).size > 1 ? undefined : best[0]!.e;
    };
    anchor ??= pick(plain) ?? pick(options.anchors ?? []);
    if (anchor === undefined) continue;
    const approximate = c.approximate || anchor.approximate === true;
    const slack = c.slack + (anchor.slack ?? 0);
    push({
      iso: toIso(shiftMs(msOf(anchor.iso), c.sign * c.count, c.unit)),
      expression: c.expression,
      sentence: c.sentence,
      sessionDay,
      kind: 'relative',
      index: c.index,
      anchor: { iso: anchor.iso, expression: anchor.expression, sentence: anchor.sentence },
      ...(approximate ? { approximate } : {}),
      ...(slack > 0 ? { slack } : {}),
    });
  }
  return events;
}

const UNIT_RE = '(hours?|hrs?|minutes?|mins?|seconds?|days?|weeks?|months?|years?|miles?|mi|km|kilometers?|kilometres?|pounds?|lbs?|kg|kilograms?|grams?|dollars?|bucks|percent|%|times?|cups?|gallons?|litres?|liters?|pairs?|sessions?|classes?|courses?|books?|people|guests?|fish|plants?|trips?|festivals?|events?|bottles?|glasses?|servings?|reps?|sets?|laps?|steps?|calories|kcal|pages?|chapters?|episodes?|movies?|films?|songs?|albums?|tickets?|items?|boxes?|bags?|pieces?|slices?|eggs|chickens?|hens|cats?|dogs?|kids?|children|meetings?|calls?|emails?|posts?|hikes?|runs?|races?|workouts?|lessons?|shifts?|nights?|stops?|countries|cities|states?|rooms?|bedrooms?|acres?|square feet|sq ft|inches|inch|feet|foot|cm|meters?|metres?)';

function canonicalUnit(u: string): string {
  const w = u.toLowerCase();
  const map: Record<string, string> = { hrs: 'hour', hr: 'hour', mins: 'minute', min: 'minute', lbs: 'pound', lb: 'pound', kg: 'kg', kilograms: 'kg', kilogram: 'kg', mi: 'mile', km: 'km', kilometers: 'km', kilometres: 'km', kilometer: 'km', kilometre: 'km', '%': 'percent', bucks: 'dollar', people: 'person', children: 'child', kids: 'kid', countries: 'country', cities: 'city', feet: 'foot', inches: 'inch', 'sq ft': 'square foot', 'square feet': 'square foot' };
  if (map[w]) return map[w];
  return w.endsWith('ies') ? `${w.slice(0, -3)}y` : w.endsWith('ses') || w.endsWith('shes') || w.endsWith('ches') ? w.slice(0, -2) : w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w;
}

export function extractQuantities(text: string): Quantity[] {
  const out: Quantity[] = [];
  for (const sentence of userSentences(text)) {
    let s = sentence;
    let m: RegExpExecArray | null;
    // durations "4 hours 22 minutes" → minutes (and remove so the parts are not double counted)
    const duration = /\b(\d+)\s*(?:hours?|hrs?|h)\s*(?:and\s+)?(\d+)\s*(?:minutes?|mins?|m)\b/gi;
    while ((m = duration.exec(s)) !== null) {
      out.push({ value: Number(m[1]) * 60 + Number(m[2]), unit: 'minute', raw: m[0], sentence });
    }
    s = s.replace(duration, ' ');
    const money = /\$\s?(\d[\d,]*(?:\.\d+)?)/g;
    while ((m = money.exec(s)) !== null) out.push({ value: Number(m[1].replace(/,/g, '')), unit: 'dollar', raw: m[0], sentence });
    const qty = new RegExp(`\\b(\\d+(?:[.,]\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|dozen|half)\\s+(?:more\\s+|new\\s+|extra\\s+|additional\\s+)?${UNIT_RE}\\b`, 'gi');
    while ((m = qty.exec(s)) !== null) {
      const value = number(m[1]);
      if (value === undefined) continue;
      out.push({ value, unit: canonicalUnit(m[2]), raw: m[0], sentence });
    }
  }
  return out;
}

export function questionKeywords(question: string): string[] {
  const words = question.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length >= 4 && !STOPWORDS.has(w));
  return [...new Set(words.map((w) => (w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w)))];
}

/** Words that say nothing about which event a question means, on top of STOPWORDS. */
const EXTRA_STOP = new Set(
  'new old one two three four five six seven eight nine ten got get let put say see use way why yet own per far few too out off now not may can will just also really please tell remind recall remember mention mentioned told kind sort thing things lot bit like first second third last next previous earlier later don didn doesn isn wasn aren weren haven hasn couldn wouldn shouldn won'.split(' '),
);

/**
 * The question's content words for whole-word matching, in canonical form: every word of three
 * letters or more that is not a stopword, and every capitalised name at any length ("Tom", "SF").
 */
export function questionTerms(text: string): string[] {
  const out = new Set<string>();
  for (const raw of tokenize(text)) {
    const lower = raw.toLowerCase();
    if (STOPWORDS.has(lower) || EXTRA_STOP.has(lower) || SMALL_NUMBERS[lower] !== undefined) continue;
    if (lower.length < 3 && !(/^[A-Z]/.test(raw) && raw.length >= 2)) continue;
    const word = canonicalWord(lower);
    if (!STOPWORDS.has(plainWord(word)) && !EXTRA_STOP.has(plainWord(word))) out.add(word);
  }
  return [...out];
}

/** Capitalised names in the question, as written. */
function questionNames(text: string): string[] {
  return [...new Set(tokenize(text).filter((raw) => /^[A-Z]/.test(raw) && raw.length >= 2 && !STOPWORDS.has(raw.toLowerCase()) && !EXTRA_STOP.has(raw.toLowerCase())))];
}

function escapeRe(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * "my new laptop, Dell XPS 13": within one turn, the common noun set beside a name the
 * question uses. A later "the laptop" in the same turn then refers to that name. A noun set
 * beside two different names is dropped.
 */
function turnAliases(turn: string, names: string[]): Map<string, string> {
  const found = new Map<string, Set<string>>();
  for (const name of names) {
    const re = new RegExp(`\\b(?:[Mm]y|[Oo]ur|[Tt]he|[Aa]n?|[Hh]is|[Hh]er|[Tt]heir)\\s+(?:[A-Za-z-]+\\s+)?([a-z][a-z-]+),?\\s+(?:the\\s+)?${escapeRe(name)}\\b`, 'g');
    let m: RegExpExecArray | null;
    while ((m = re.exec(turn)) !== null) {
      const noun = m[1]!.toLowerCase();
      if (STOPWORDS.has(noun) || EXTRA_STOP.has(noun) || noun.length < 3) continue;
      found.set(noun, new Set([...(found.get(noun) ?? []), canonicalWord(name)]));
    }
  }
  const out = new Map<string, string>();
  for (const [noun, terms] of found) if (terms.size === 1) out.set(noun, [...terms][0]!);
  return out;
}

/** Names a sentence refers to through "the <noun>" / "my <noun>" set beside them in its turn. */
function aliasReferences(sentence: string, aliases: Map<string, string>): string[] {
  const out: string[] = [];
  for (const [noun, term] of aliases) {
    if (new RegExp(`\\b(?:the|my|our|this|that)\\s+(?:new\\s+|old\\s+)?${escapeRe(noun)}(?:'s)?\\b`, 'i').test(sentence)) out.push(term);
  }
  return out;
}

/** The verbs a question asks about, in canonical form: "did I cancel", "when I moved". */
function questionVerbs(question: string): Set<string> {
  const out = new Set<string>();
  // the word right after the subject: "the day I cancelled", "did I cancel", "since I started";
  // not "baked" in "baked goods"
  const re = /\b(?:i|we)\s+(?:(?:did|had|have|has|first|finally|just|last|ever|recently|also|actually)\s+)*([a-z]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(question)) !== null) {
    const raw = m[1]!.toLowerCase();
    if (['when', 'was', 'were', 'am', 'the', 'a', 'my', 'to', 'and', 'or', 'in', 'on', 'at', 'with', 'for', 'need', 'want', 'like', 'think', 'should', 'would', 'could', 'can', 'will', 'might', 'must'].includes(raw)) continue;
    out.add(canonicalWord(raw));
  }
  return new Set([...out].filter((v) => !['do', 'be', 'have', 'pass'].includes(plainWord(v))));
}

const AGE_RE = /\b(?:i'm|i’m|i am|as)\s+(?:a\s+|an\s+)?(\d{1,3})[- ]years?[- ]old\b/i;

function distance(iso: string, questionDay: string): string {
  const days = Math.round((msOf(questionDay) - msOf(iso)) / DAY_MS);
  const abs = Math.abs(days);
  const relation = days >= 0 ? 'before the question date' : 'after the question date';
  if (abs >= 60) return `about ${Math.round(abs / 30.44)} months (${Math.round(abs / 7)} weeks, ${abs} days) ${relation}`;
  if (abs >= 14) return `about ${Math.round(abs / 7)} weeks (${abs} days) ${relation}`;
  return `${abs} days ${relation}`;
}

function gap(a: string, b: string): string {
  const days = Math.round((msOf(b) - msOf(a)) / DAY_MS);
  const weeks = days >= 7 ? ` (${(days / 7).toFixed(days % 7 === 0 ? 0 : 1)} weeks)` : '';
  return `${days} days${weeks}`;
}

function snippet(sentence: string, max = 90): string {
  const s = sentence.replace(/\s+/g, ' ').trim();
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/** A start offset moved forward to the next word; an end offset moved back to the previous one. */
function wordStart(text: string, at: number): number {
  if (at <= 0) return 0;
  if (text[at - 1] === ' ') return at;
  const space = text.indexOf(' ', at);
  return space === -1 ? at : space + 1;
}

function wordEnd(text: string, at: number, atLeast: number): number {
  if (at >= text.length) return text.length;
  const space = text.lastIndexOf(' ', at);
  return space > atLeast ? space : at;
}

function renderRange(text: string, from: number, to: number): string {
  const body = text.slice(from, to).replace(/\s+/g, ' ').trim();
  return `${from > 0 ? '…' : ''}${body}${to < text.length ? '…' : ''}`;
}

/** The clause holding the date, cut to `max` characters around the expression. */
function clauseRange(e: DatedEvent, max: number): [number, number] {
  const sentence = e.sentence;
  if (e.index === undefined) return [0, wordEnd(sentence, Math.min(sentence.length, max - 1), 0)];
  const [from, to] = clauseAt(sentence, e.index);
  const expressionEnd = e.index + e.expression.length;
  let start = Math.max(from, Math.min(e.index, expressionEnd + 12 - max));
  if (start > from) start = Math.min(wordStart(sentence, start), e.index);
  const end = Math.min(to, start + max);
  return [start, end < to ? wordEnd(sentence, end, expressionEnd) : end];
}

/**
 * The part of a long sentence that holds the event's date: its clause, cut around the
 * expression. When that part shows none of `mustShow` (canonical words) and the sentence has
 * one, the window widens to it, or the words around it are quoted in front.
 */
function focusSnippet(e: DatedEvent, max = 90, mustShow: string[] = []): string {
  const sentence = e.sentence;
  if (sentence.length <= max) return snippet(sentence, max);
  let [from, to] = clauseRange(e, max);
  if (mustShow.length > 0 && !mustShow.some((t) => hasWord(wordSet(sentence.slice(from, to)), t))) {
    const word = /[A-Za-z0-9'-]+/g;
    let m: RegExpExecArray | null;
    while ((m = word.exec(sentence)) !== null) {
      const token = canonicalWord(m[0].replace(/^['-]+|['-]+$/g, '').replace(/'.*$/, ''));
      if (mustShow.some((t) => sameWord(t, token))) break;
    }
    if (m !== null) {
      const at = m.index;
      if (e.index === undefined) {
        from = wordStart(sentence, Math.max(0, at - 30));
        const end = Math.min(sentence.length, from + max);
        to = end < sentence.length ? wordEnd(sentence, end, at + m[0].length) : end;
      } else if (at < from && to - at <= max + 30) {
        from = wordStart(sentence, Math.max(0, at - 12));
      } else {
        const pieceFrom = wordStart(sentence, Math.max(0, at - 20));
        const pieceTo = wordEnd(sentence, Math.min(sentence.length, at + m[0].length + 20), at + m[0].length);
        const piece = renderRange(sentence, pieceFrom, pieceTo);
        const window = renderRange(sentence, from, to);
        return at < from ? `${piece} ${window}` : `${window} ${piece}`;
      }
    }
  }
  return renderRange(sentence, from, to);
}

/** An event's date as the notes print it: rough dates marked, a month shown as a month. */
function whenOf(e: DatedEvent): string {
  if (e.monthOnly) return `${e.iso.slice(0, 7)} (month only)`;
  if (e.kind === 'undated') return `on or before ${e.sessionDay} (no date stated)`;
  return `${e.approximate ? '≈' : ''}${e.iso}`;
}

/** The days an event may fall on, as UTC ms. */
function windowOf(e: DatedEvent): [number, number] {
  const ms = msOf(e.iso);
  if (e.monthOnly) {
    const d = new Date(ms);
    return [ms, Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)];
  }
  if (e.kind === 'undated') return [Number.NEGATIVE_INFINITY, msOf(e.sessionDay)];
  const slack = (e.slack ?? 0) * DAY_MS;
  return [ms - slack, ms + slack];
}

interface Alternative {
  /** As the question words it, without a leading article. */
  text: string;
  /** Display words (the older keyword reading). */
  ks: string[];
  /** Canonical words that identify this alternative and not the other ones. */
  terms: string[];
}

const ORDINALS = new Set(['first', 'last', 'second', 'third', 'next', 'previous', 'earlier', 'later']);

/** "X or Y", "A, B and C", "did I do X before or after Y": the question's alternatives. */
function questionAlternatives(question: string): Alternative[] {
  // "Which three events happened in order: A, B, and C?" lists the alternatives after the colon
  const q = question.replace(/[?]+\s*$/, '').replace(/^[^:]*:\s*/, '').trim();
  const beforeOrAfter = /^(.*?)\s+before or after\s+(.*)$/i.exec(q);
  const parts = beforeOrAfter
    ? [beforeOrAfter[1]!.replace(/^(?:did|do|does|was|were|have|had|has)\s+(?:i|we)\s+/i, ''), beforeOrAfter[2]!]
    : q.split(/\s+or\s+|,\s*(?:and\s+)?|\s+and then\s+/i).filter((part) => !/\b(which|what|how|when|did|do|does|who|where)\b/i.test(part));
  const alternatives = parts
    .map((part) => ({
      text: part.trim().replace(/^(?:the|a|an)\s+/i, '').replace(/[.,;:!]+$/, ''),
      // number words and ordinals match everywhere ("three weeks ago"); they do not identify an alternative
      ks: questionKeywords(part).filter((k) => SMALL_NUMBERS[k] === undefined && !ORDINALS.has(k)),
      terms: questionTerms(part),
    }))
    .filter((a) => a.terms.length > 0);
  // "the narrator losing their phone charger or the narrator receiving their new phone case":
  // words every alternative shares say nothing about which one a sentence means
  const shared = alternatives.length >= 2 ? alternatives[0]!.terms.filter((t) => alternatives.every((a) => a.terms.includes(t))) : [];
  if (shared.length > 0 && alternatives.every((a) => a.terms.some((t) => !shared.includes(t)))) {
    for (const a of alternatives) a.terms = a.terms.filter((t) => !shared.includes(t));
  }
  return alternatives.map((a) => ({ ...a, ks: a.ks.length > 0 ? a.ks : a.terms }));
}

/**
 * The order of the question's two alternatives. Each alternative is tied to a dated event whose
 * clause names it (or whose sentence names it and not the other one); a clause naming both
 * decides nothing. The order is too close to call only when the two date ranges overlap.
 */
function orderVerdict(
  alternatives: Alternative[],
  pool: DatedEvent[],
  sentenceWords: (sentence: string) => Set<string>,
  clauseWords: (e: DatedEvent) => Set<string>,
  questionClasses: Set<number>,
): string | undefined {
  if (alternatives.length !== 2) return undefined;
  const names = (words: Set<string>, a: Alternative) => a.terms.some((t) => hasWord(words, t));
  // the distance from the event's date to the nearest verb of the kind the question asks about
  const verbDistance = (e: DatedEvent) => {
    if (e.index === undefined || questionClasses.size === 0) return Number.POSITIVE_INFINITY;
    const [from, to] = clauseAt(e.sentence, e.index);
    let best = Number.POSITIVE_INFINITY;
    const word = /[A-Za-z'-]+/g;
    const clause = e.sentence.slice(from, to);
    let m: RegExpExecArray | null;
    while ((m = word.exec(clause)) !== null) {
      const c = canonicalWord(m[0]);
      if (VERB_CLASSES.some((cls, i) => questionClasses.has(i) && [...cls].some((w) => sameWord(w, c)))) best = Math.min(best, Math.abs(from + m.index - e.index));
    }
    return best;
  };
  const picks = alternatives.map((alt, i) => {
    const other = alternatives[1 - i]!;
    const found = pool.flatMap((e) => {
      const clause = clauseWords(e);
      if (names(clause, other)) return [];
      if (names(clause, alt)) return [{ e, clause: 1 }];
      const sentence = sentenceWords(e.sentence);
      return names(sentence, alt) && !names(sentence, other) ? [{ e, clause: 0 }] : [];
    });
    const ranked = found
      .map((x) => ({ ...x, distance: verbDistance(x.e), hits: alt.terms.filter((t) => hasWord(sentenceWords(x.e.sentence), t)).length }))
      .sort((l, r) => Number(Number.isFinite(r.distance)) - Number(Number.isFinite(l.distance)) || l.distance - r.distance || r.clause - l.clause || r.hits - l.hits);
    return { alt, ranked };
  });
  const [a, b] = picks as [(typeof picks)[number], (typeof picks)[number]];
  const first = a.ranked[0];
  const second = b.ranked[0];
  if (first === undefined || second === undefined || first.e === second.e) return undefined;
  const describe = (p: typeof a) => {
    const chosen = p.ranked[0]!.e;
    const others = [...new Map(p.ranked.slice(1).filter((x) => x.e.iso !== chosen.iso).map((x) => [x.e.iso, x.e])).values()].slice(0, 2);
    const also = others.length === 0 ? '' : ` [other dates the history gives it: ${others.map((o) => `${whenOf(o)} ("${focusSnippet(o, 50)}")`).join('; ')}]`;
    return `"${p.alt.text}" ${whenOf(chosen)} ("${focusSnippet(chosen, 70, p.alt.terms)}", said ${chosen.sessionDay})${also}`;
  };
  const [aFrom, aTo] = windowOf(first.e);
  const [bFrom, bTo] = windowOf(second.e);
  if (aTo < bFrom) return `Which came first: ${describe(a)} is earlier than ${describe(b)} → ${a.alt.text} came first.`;
  if (bTo < aFrom) return `Which came first: ${describe(b)} is earlier than ${describe(a)} → ${b.alt.text} came first.`;
  if (aFrom === aTo && bFrom === bTo && aFrom === bFrom) return `Which came first: ${describe(a)} and ${describe(b)} are dated the same day.`;
  return `Which came first: ${describe(a)} and ${describe(b)} are too close to order from these dates (the ranges their rough expressions allow overlap).`;
}

type GapUnit = 'day' | 'week' | 'month' | 'year';

interface GapQuestion {
  first: string;
  /** undefined: counted to the question date ("how many days ago") */
  second: string | undefined;
  unit: GapUnit | undefined;
  /** "how long had I been X when Y": X is something that began */
  firstIsStart?: boolean;
  /** The question implies the first side happened no later than the second. */
  ordered?: boolean;
}

/** A question asking for the time between two events (or from one event to now), split into its sides. */
function gapQuestion(question: string): GapQuestion | undefined {
  const q = question.replace(/[?]+\s*$/, '').trim();
  const unitOf = (u: string | undefined) => (u === undefined ? undefined : (u.toLowerCase().replace(/s$/, '') as GapUnit));
  const U = '(days?|weeks?|months?|years?)';
  let m: RegExpExecArray | null;
  if ((m = new RegExp(`\\bhow many ${U}\\b.*?\\bbetween\\s+(.+)$`, 'i').exec(q))) {
    const rest = m[2]!;
    const split = /\s+and\s+(?=(?:the|my|when|i|we)\b)/i.exec(rest) ?? /\s+and\s+/i.exec(rest);
    if (split === null) return undefined;
    return { first: rest.slice(0, split.index), second: rest.slice(split.index + split[0].length), unit: unitOf(m[1]) };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+(?:have\\s+|had\\s+|has\\s+)?(?:passed|elapsed|gone by|been)\\s+since\\s+(.+?)\\s+(?:when|until|by the time|before)\\s+(.+)$`, 'i').exec(q))) {
    return { first: m[2]!, second: m[3]!, unit: unitOf(m[1]), ordered: true };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+ago\\s+(?:did|was|were|had|have)\\s+(?:i|we)\\s+(.+)$`, 'i').exec(q))) {
    // "how many days ago did I launch my website when I signed my first client": X counted to Y
    const when = /\s+when\s+(?:i|we)\s+(.+)$/i.exec(m[2]!);
    if (when !== null) return { first: m[2]!.slice(0, when.index), second: when[1]!, unit: unitOf(m[1]), ordered: true };
    return { first: m[2]!, second: undefined, unit: unitOf(m[1]) };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+(?:have\\s+|had\\s+|has\\s+)?(?:passed|elapsed|gone by|been)\\s+since\\s+(.+)$`, 'i').exec(q))) {
    return { first: m[2]!, second: undefined, unit: unitOf(m[1]) };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+after\\s+(.+?)\\s+did\\s+(?:i|we)\\s+(.+)$`, 'i').exec(q))) {
    return { first: m[2]!, second: m[3]!, unit: unitOf(m[1]), ordered: true };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+before\\s+(.+?)\\s+did\\s+(?:i|we)\\s+(.+)$`, 'i').exec(q))) {
    return { first: m[3]!, second: m[2]!, unit: unitOf(m[1]), ordered: true };
  }
  if ((m = new RegExp(`\\bhow many ${U}\\s+(?:did it take|had passed|passed|was it)(?:\\s+for\\s+(?:me|us))?(?:\\s+(?:me|us))?(?:\\s+to)?\\s+(.+?)\\s+after\\s+(.+)$`, 'i').exec(q))) {
    return { first: m[3]!, second: m[2]!, unit: unitOf(m[1]), ordered: true };
  }
  if ((m = /\bhow long\s+(?:had|have)\s+(?:i|we)\s+been\s+(.+?)\s+(?:when|before|by the time)\s+(?:i\s+|we\s+)?(.+)$/i.exec(q))) {
    return { first: m[1]!, second: m[2]!, unit: undefined, firstIsStart: true, ordered: true };
  }
  return undefined;
}

/** A side of a gap question as the notes print it: "the day I cancelled X" → "cancelled X". */
function sideText(side: string): string {
  return side
    .trim()
    .replace(/^(?:the\s+(?:day|time|date|moment)\s+)?(?:when\s+|that\s+)?(?:i|we)\s+(?:did\s+)?/i, '')
    .replace(/[.,;:!]+$/, '');
}

/** The verb a side of a gap question starts with, in canonical form. */
function sideVerb(side: string): string | undefined {
  const first = /^([a-z]+)/i.exec(sideText(side))?.[1];
  if (first === undefined) return undefined;
  const verb = canonicalWord(first);
  return ['do', 'be', 'have', 'the', 'my', 'a', 'an'].includes(verb) ? undefined : verb;
}

function inUnit(days: number, unit: GapUnit | undefined): string {
  // "how long": no unit asked, so a long gap is given in months and weeks both
  if (unit === undefined && days >= 45) {
    return `about ${Math.round(days / 30.44)} months (${(days / 30.44).toFixed(1)} months, ${(days / 7).toFixed(1)} weeks, ${days} days)`;
  }
  const u = unit ?? (days >= 14 ? 'week' : 'day');
  if (u === 'week') return `${Number.isInteger(days / 7) ? days / 7 : (days / 7).toFixed(1)} weeks (${days} days)`;
  if (u === 'month') return `about ${Math.round(days / 30.44)} months (${(days / 30.44).toFixed(1)} months, ${days} days)`;
  if (u === 'year') return `about ${Math.round(days / 365.25)} years (${(days / 365.25).toFixed(1)} years, ${days} days)`;
  return `${days} ${days === 1 ? 'day' : 'days'} (${days + 1} counting both the first and the last day)`;
}

/**
 * The block for the reader, or '' when the history has nothing datable or countable near the
 * question. Every line names the sentence it came from so the reader can check it.
 */
export function buildComputedNotes(
  question: string,
  questionDate: string,
  sources: Array<{ ts: string; text: string }>,
  limits: { maxEvents?: number; maxQuantities?: number; maxChars?: number } = {},
  /** Lines a caller computed in code (a counted tally); they lead the block and are never cut. */
  pinnedLines: readonly string[] = [],
): string {
  const questionDay = dayOf(questionDate);
  const terms = questionTerms(question);
  const names = questionNames(question);
  // every user sentence's words, plus the names it refers to through a noun set beside them in its turn
  const sentenceWords = new Map<string, Set<string>>();
  const sentenceAliases = new Map<string, Map<string, string>>();
  const allUserWords = new Set<string>();
  for (const source of sources) {
    for (const turn of userTurns(source.text)) {
      const aliases = names.length > 0 ? turnAliases(turn.join(' '), names) : new Map<string, string>();
      for (const sentence of turn) {
        const words = new Set([...wordSet(sentence), ...aliasReferences(sentence, aliases)]);
        if (aliases.size > 0) sentenceAliases.set(sentence, new Map([...(sentenceAliases.get(sentence) ?? []), ...aliases]));
        const known = sentenceWords.get(sentence);
        sentenceWords.set(sentence, known === undefined ? words : new Set([...known, ...words]));
        for (const w of words) allUserWords.add(w);
      }
    }
  }
  const wordsOf = (sentence: string) => sentenceWords.get(sentence) ?? wordSet(sentence);
  const clauseWordsOf = (e: DatedEvent) => {
    if (e.index === undefined) return wordsOf(e.sentence);
    const [from, to] = clauseAt(e.sentence, e.index);
    const clause = e.sentence.slice(from, to);
    return new Set([...wordSet(clause), ...aliasReferences(clause, sentenceAliases.get(e.sentence) ?? new Map())]);
  };
  const maxEvents = limits.maxEvents ?? 12;
  const maxQuantities = limits.maxQuantities ?? 16;
  const maxChars = limits.maxChars ?? 2600;

  const q = question.toLowerCase();
  const asksTotal = /\b(total|combined|altogether|in all|sum|how much|add(?:ed)? up)\b/.test(q);
  const asksOrder = /\b(first|last|earlier|later|before|after|order|sequence|which .* (came|happened))\b/.test(q);
  const unitsInQuestion = new Set(
    (q.match(new RegExp(`\\b${UNIT_RE}\\b`, 'gi')) ?? []).map((u) => canonicalUnit(u)),
  );
  if (/\b(money|cost|spend|spent|paid|pay|price|raise[d]?|earn)\b/.test(q)) unitsInQuestion.add('dollar');
  if (/\bpercent|%|percentage\b/.test(q)) unitsInQuestion.add('percent');
  const keywordHits = (sentence: string) => {
    const words = wordsOf(sentence);
    return terms.filter((t) => hasWord(words, t)).length;
  };

  const asksTime = /\b(when|how long|how old|how many (?:days|weeks|months|years)|ago|first|earlier|later|before|after|since|between|date|passed|order)\b/.test(q);
  const verbs = questionVerbs(question);
  const events: DatedEvent[] = [];
  const allEvents: DatedEvent[] = [];
  const undatedEvents: DatedEvent[] = [];
  const quantities: Quantity[] = [];
  const ordered = [...sources].sort((l, r) => l.ts.localeCompare(r.ts));
  // a first reading of every session, so a chained expression can count from another session's event
  const firstReading = ordered.map((source) => resolveTemporalExpressions(source.text, source.ts, { extended: true }));
  for (const [i, source] of ordered.entries()) {
    const anchors = firstReading.flatMap((events, j) => (j === i ? [] : events));
    const resolved = resolveTemporalExpressions(source.text, source.ts, { extended: true, anchors });
    for (const e of resolved) {
      allEvents.push(e);
      if (keywordHits(e.sentence) > 0) events.push(e);
    }
    // "I'm glad I cancelled my FarmFresh subscription": a past event the question asks about,
    // told with no date, happened on or before the day it was said
    if (asksTime && verbs.size > 0) {
      const withDate = new Set(resolved.map((e) => e.sentence));
      const sessionDay = dayOf(source.ts);
      for (const sentence of new Set(userSentences(source.text))) {
        if (withDate.has(sentence) || sentence.endsWith('?') || FUTURE_CUE.test(sentence)) continue;
        if (![...pastVerbs(sentence)].some((v) => [...verbs].some((q) => sameWord(q, v)))) continue;
        const words = wordsOf(sentence);
        if (terms.filter((t) => ![...verbs].some((q) => sameWord(q, t)) && hasWord(words, t)).length === 0) continue;
        const e: DatedEvent = { iso: sessionDay, expression: '', sentence, sessionDay, kind: 'undated', approximate: true };
        undatedEvents.push(e);
        events.push(e);
      }
    }
    for (const x of extractQuantities(source.text)) {
      // a quantity belongs to the question when its unit is named in the question, or its
      // sentence shares two content words with it; one shared word lets in every stray number
      if (unitsInQuestion.has(x.unit) ? keywordHits(x.sentence) >= 1 : keywordHits(x.sentence) >= 2) quantities.push(x);
    }
  }
  const lines: string[] = [];
  const questionClasses = verbClasses(question);
  const alternatives = asksOrder ? questionAlternatives(question) : [];
  // a two-way verdict only: not for "which three events ... in order" or "first, second and third"
  if (/\b(first|earlier|sooner)\b|\bbefore or after\b/.test(q) && !/\b(second|third|fourth|three|four|five|order|sequence|last)\b/.test(q)) {
    const verdict = orderVerdict(alternatives, events, wordsOf, clauseWordsOf, questionClasses);
    if (verdict !== undefined) lines.push(verdict);
  }
  // the question's own time reference ("last Saturday", "two months ago", "the past weekend"):
  // resolve it against the question date and point at the dated events closest to it
  const questionRefs = resolveTemporalExpressions(`USER: ${question.replace(/\bthe past weekend\b/i, 'last Saturday')}`, `${questionDay}T00:00:00Z`).filter((e) => e.kind !== 'absolute');
  const dated = events.slice(0, maxEvents);
  if (questionRefs.length > 0 && allEvents.length > 0) {
    const ref = questionRefs[0]!;
    // every dated event counts here, whatever its wording: the question asks what happened then
    // one event per date: the one whose sentence best matches the question, not the last one said
    const score = (e: DatedEvent) => keywordHits(e.sentence) + ([...verbClasses(e.sentence)].some((c) => questionClasses.has(c)) ? 1 : 0);
    const bestByDate = new Map<string, DatedEvent>();
    for (const e of allEvents) {
      const kept = bestByDate.get(e.iso);
      if (kept === undefined || score(e) > score(kept)) bestByDate.set(e.iso, e);
    }
    const byDistance = [...bestByDate.values()].sort((l, r) => Math.abs(msOf(l.iso) - msOf(ref.iso)) - Math.abs(msOf(r.iso) - msOf(ref.iso)));
    const closest = byDistance.slice(0, 2).map((e) => `${e.iso} ("${snippet(e.sentence, 60)}", ${Math.round(Math.abs(msOf(e.iso) - msOf(ref.iso)) / DAY_MS)} days away)`);
    lines.push(`The question's "${ref.expression}" counted from the question date ${questionDay} is ${ref.iso}${ref.approximate ? ' (approximate)' : ''}; the closest dated event${closest.length > 1 ? 's' : ''}: ${closest.join('; ')}.`);
  }
  // "How old was I when I moved here?": a stated age, and the question's event dated before it
  if (/\bhow old\b|\bage\b/.test(q)) {
    const stated = [...sources]
      .sort((l, r) => r.ts.localeCompare(l.ts))
      .flatMap((source) => userSentences(source.text).map((sentence) => ({ sentence, day: dayOf(source.ts), m: AGE_RE.exec(sentence) })))
      .find((x) => x.m !== null);
    if (stated !== undefined) {
      const age = Number(stated.m![1]);
      const candidates = events.filter((e) => e.kind !== 'undated' && !AGE_RE.test(e.sentence) && e.iso < stated.day);
      const rank = (e: DatedEvent) => keywordHits(e.sentence) * 2 + (e.start ? 1 : 0);
      const top = Math.max(0, ...candidates.map(rank));
      const best = candidates.filter((e) => rank(e) === top);
      if (top > 0 && new Set(best.map((e) => e.iso)).size === 1) {
        const e = best[0]!;
        const years = Math.round((msOf(stated.day) - msOf(e.iso)) / (365.25 * DAY_MS));
        lines.push(`Age: "${snippet(stated.sentence, 60)}" makes you ${age} on ${stated.day}; "${snippet(e.sentence, 70)}" dates ${e.approximate ? '≈' : ''}${e.iso}${e.start ? ' (when that began)' : ''}, about ${years} years earlier, so the age then was about ${age - years} (${age} − ${years}; one less if the birthday falls in between).`);
      }
    }
  }
  if (dated.length > 0) {
    lines.push('Dated events (each temporal expression resolved against the date of the session it was said in):');
    for (const e of dated) {
      const flags = [
        e.approximate && e.kind !== 'undated' ? 'approximate' : '',
        e.monthOnly ? 'month only, no day stated' : '',
        e.start ? 'when something still going on began' : '',
        e.anchor ? `counted from ${e.anchor.iso} ("${snippet(e.anchor.sentence, 50)}", "${e.anchor.expression}")` : '',
        e.yearRolledBack ? 'no year stated; the year before the session, since the sentence tells it as past' : e.assumedYear ? 'year assumed from the session date' : '',
      ].filter(Boolean).join('; ');
      const said = e.kind === 'undated' ? `said ${e.sessionDay}; no date stated, so dated by the session it was said in (it happened on or before that day)` : `said ${e.sessionDay}, "${e.expression}"`;
      const when = e.monthOnly ? `${e.iso.slice(0, 7)} (month only)` : e.iso;
      const far = e.monthOnly ? `the month began ${distance(e.iso, questionDay)}` : distance(e.iso, questionDay);
      lines.push(`- ${when}: "${focusSnippet(e)}" [${said}${flags ? `; ${flags}` : ''}] — ${far}`);
    }
    const distinct = [...new Map(dated.map((e) => [e.iso, e])).values()].sort((l, r) => l.iso.localeCompare(r.iso));
    if (asksOrder) {
      // "X or Y": which alternatives the dated events actually cover
      if (alternatives.length >= 2) {
        const coverage = alternatives.map((a) => ({ ...a, dated: dated.some((e) => a.terms.some((t) => hasWord(wordsOf(e.sentence), t))) }));
        const undated = coverage.filter((c) => !c.dated);
        if (undated.length > 0 && coverage.some((c) => c.dated)) {
          const label = (c: { ks: string[] }) => `"${c.ks.slice(0, 4).join(' ')}"`;
          // a side the user never names is unknown; a side the user names without a date is only undated
          const unnamed = undated.filter((c) => !c.terms.some((t) => hasWord(allUserWords, t)));
          const named = undated.filter((c) => c.terms.some((t) => hasWord(allUserWords, t)));
          if (unnamed.length > 0) {
            lines.push(`Coverage: the dated events above match ${coverage.filter((c) => c.dated).map(label).join(', ')} but none match ${unnamed.map(label).join(', ')}; if the history never dates one side of the question, the honest answer is that it does not say.`);
          }
          for (const c of named) lines.push(`Coverage: ${label(c)} is named in the history, but no sentence naming it carries a date.`);
        }
      }
    }
    if (asksOrder && distinct.length >= 2) {
      lines.push(`Order of the dated events the history dates, earliest first (if the question names an event that is not here, the history may not date it, and the honest answer may be that it does not say): ${distinct.map((e) => `${e.iso} ("${snippet(e.sentence, 36)}")${e.approximate ? ' [approximate]' : ''}`).join(' → ')}`);
    }
    if (distinct.length >= 2) {
      const pairs: Array<[DatedEvent, DatedEvent]> = [];
      if (distinct.length <= 6) {
        for (let i = 0; i < distinct.length; i++) for (let j = i + 1; j < distinct.length; j++) pairs.push([distinct[i]!, distinct[j]!]);
      } else {
        for (let i = 0; i + 1 < distinct.length; i++) pairs.push([distinct[i]!, distinct[i + 1]!]);
      }
      // the pair whose two sentences together cover the most question words comes first: a
      // small reader copies the first gap it sees
      const covered = (a: DatedEvent, b: DatedEvent) => new Set(terms.filter((t) => hasWord(wordsOf(a.sentence), t) || hasWord(wordsOf(b.sentence), t))).size;
      const distinctCover = (a: DatedEvent, b: DatedEvent) => Math.min(keywordHits(a.sentence), keywordHits(b.sentence));
      pairs.sort((l, r) => covered(r[0], r[1]) - covered(l[0], l[1]) || distinctCover(r[0], r[1]) - distinctCover(l[0], l[1]));
      const shown = pairs.slice(0, 4);
      lines.push(`Gaps between dated events (the pair whose sentences best match the question is listed first${pairs.length > shown.length ? `; ${pairs.length - shown.length} other pairs omitted` : ''}):`);
      for (const [a, b] of shown) lines.push(`- ${a.iso} ("${snippet(a.sentence, 40)}") to ${b.iso} ("${snippet(b.sentence, 40)}"): ${gap(a.iso, b.iso)}${a.approximate || b.approximate ? ' [approximate: one end is a rough date, like "a month ago" or a date only implied by the session]' : ''}`);
    }
  }
  const counted = quantities.slice(0, maxQuantities);
  if (counted.length > 0) {
    lines.push('Quantities stated in the history (with the sentence each comes from):');
    const byUnit = new Map<string, Quantity[]>();
    for (const q of counted) byUnit.set(q.unit, [...(byUnit.get(q.unit) ?? []), q]);
    for (const [unit, items] of byUnit) {
      for (const q of items) lines.push(`- ${q.value} ${unit}${q.value === 1 ? '' : 's'}: "${snippet(q.sentence)}"`);
      // a figure is tightly the question's when its sentence shares two content words with it,
      // a unit the question names counting as one
      const tight = items.filter((q) => keywordHits(q.sentence) + (unitsInQuestion.has(unit) ? 1 : 0) >= 2);
      if (asksTotal && tight.length >= 2 && tight.length === items.length) {
        const total = tight.reduce((sum, q) => sum + q.value, 0);
        lines.push(`  sum of the ${tight.length} ${unit} figures above: ${Number(total.toFixed(2))} ${unit}s (check that every figure belongs to the question before using it)`);
      }
      if (items.length >= 2) {
        if (items.length === 2) {
          const [x, y] = items;
          const diff = Math.abs(x!.value - y!.value);
          lines.push(`  difference between them: ${Number(diff.toFixed(2))} ${unit}s`);
          if (unit !== 'percent' && x!.value > 0 && y!.value > 0) {
            const [small, large] = x!.value <= y!.value ? [x!.value, y!.value] : [y!.value, x!.value];
            lines.push(`  ratio: ${small} of ${large} = ${Math.round((small / large) * 100)}%`);
          }
        }
      }
    }
  }
  // the gap the question itself asks for, first: a small reader copies the first figure it sees,
  // and the character limit below must not cut it
  const asked = gapQuestion(question);
  let pinned = 0;
  if (asked !== undefined) {
    const pool = [...allEvents, ...undatedEvents];
    const firstTerms = questionTerms(asked.first);
    const secondTerms = asked.second === undefined ? [] : questionTerms(asked.second);
    const pick = (side: string, terms: string[], other: string[], exclude: DatedEvent | undefined, preferStart: boolean) => {
      const specific = terms.filter((t) => !other.includes(t));
      const otherSpecific = other.filter((t) => !terms.includes(t));
      const use = specific.length > 0 ? specific : terms;
      if (use.length === 0) return undefined;
      const need = Math.min(2, use.length);
      const verb = sideVerb(side);
      // a word the question capitalises ("Museum of Modern Art") is a name: it counts only
      // where the sentence capitalises it too (or refers to it through "the <noun>")
      const sideNames = new Set(questionNames(side).map(canonicalWord));
      const hits = (sentence: string, list: string[]) => {
        const words = wordsOf(sentence);
        const capitals = new Set([...tokenize(sentence).filter((w) => /^[A-Z]/.test(w)).map(canonicalWord), ...aliasReferences(sentence, sentenceAliases.get(sentence) ?? new Map())]);
        return list.filter((t) => hasWord(words, t) && (!sideNames.has(t) || hasWord(capitals, t))).length;
      };
      const scored = pool
        .filter((e) => e !== exclude && !(exclude !== undefined && e.sentence === exclude.sentence && e.iso === exclude.iso))
        .map((e) => {
          const own = hits(e.sentence, use);
          return { e, own, other: hits(e.sentence, otherSpecific), key: [own, verb !== undefined && hasWord(wordsOf(e.sentence), verb) ? 1 : 0, preferStart && e.start ? 1 : 0, e.kind === 'undated' ? 0 : 1] };
        })
        // a sentence that could serve as the other side's event just as well is ambiguous
        .filter((x) => x.own >= need && !(x.other >= x.own && x.other >= Math.min(2, otherSpecific.length)))
        .sort((l, r) => r.key[0]! - l.key[0]! || r.key[1]! - l.key[1]! || r.key[2]! - l.key[2]! || r.key[3]! - l.key[3]!);
      const top = scored[0];
      if (top === undefined) return undefined;
      const tied = scored.filter((x) => x.key.every((k, i) => k === top.key[i]));
      // two dates that fit a side equally well: the side is ambiguous, so no gap is stated
      return new Set(tied.map((x) => x.e.iso)).size > 1 ? undefined : top.e;
    };
    const firstSpecific = firstTerms.filter((t) => !secondTerms.includes(t));
    let firstEvent: DatedEvent | undefined;
    let secondEvent: DatedEvent | undefined;
    if (asked.second !== undefined && firstSpecific.length === 0) {
      secondEvent = pick(asked.second, secondTerms, firstTerms, undefined, false);
      firstEvent = pick(asked.first, firstTerms, secondTerms, secondEvent, asked.firstIsStart === true);
    } else {
      firstEvent = pick(asked.first, firstTerms, secondTerms, undefined, asked.firstIsStart === true);
      if (asked.second !== undefined) secondEvent = pick(asked.second, secondTerms, firstTerms, firstEvent, false);
    }
    if (firstEvent !== undefined && (asked.second === undefined || secondEvent !== undefined)) {
      const label = (side: string, e: DatedEvent) => {
        const own = questionTerms(side);
        const other = side === asked.first ? secondTerms : firstTerms;
        const show = own.filter((t) => !other.includes(t) && t !== sideVerb(side));
        return `"${sideText(side)}" ${whenOf(e)} ("${focusSnippet(e, 60, show.length > 0 ? show : own)}", said ${e.sessionDay}${e.start ? '; when it began' : ''})`;
      };
      const ends = asked.second === undefined || secondEvent === undefined
        ? [{ text: label(asked.first, firstEvent), ms: msOf(firstEvent.iso) }, { text: `the question date ${questionDay}`, ms: msOf(questionDay) }]
        : [{ text: label(asked.first, firstEvent), ms: msOf(firstEvent.iso) }, { text: label(asked.second, secondEvent), ms: msOf(secondEvent.iso) }].sort((l, r) => l.ms - r.ms);
      const days = Math.round(Math.abs(ends[1]!.ms - ends[0]!.ms) / DAY_MS);
      const involved = [firstEvent, secondEvent].filter((e): e is DatedEvent => e !== undefined);
      // "N days before the party did I order the gift": an order the dates contradict means a
      // side was matched to the wrong sentence
      const [firstFrom] = windowOf(firstEvent);
      const [, secondTo] = secondEvent === undefined ? [0, msOf(questionDay)] : windowOf(secondEvent);
      const contradicted = (asked.ordered === true || secondEvent === undefined) && firstFrom > secondTo;
      const earlier = secondEvent === undefined || msOf(firstEvent.iso) <= msOf(secondEvent.iso) ? firstEvent : secondEvent;
      const undatedEnd = involved.find((e) => e.kind === 'undated');
      const caveats = [
        undatedEnd === undefined ? '' : `the ${undatedEnd === earlier ? 'earlier' : 'later'} end was told without a date and is dated by the day it was said, so the real gap may be ${undatedEnd === earlier ? 'longer' : 'shorter'}`,
        involved.some((e) => e.monthOnly) ? 'a month-only end is counted from the first of that month' : '',
        involved.some((e) => e.approximate && e.kind !== 'undated' && !e.monthOnly) ? 'an end is a rough expression' : '',
      ].filter(Boolean);
      if (!contradicted) {
        lines.unshift(`Gap the question asks for: ${ends[0]!.text} to ${ends[1]!.text} = ${inUnit(days, asked.unit)}${caveats.length > 0 ? ` [approximate: ${caveats.join('; ')}]` : ''}.`);
        pinned = 1;
      }
    }
  }
  if (pinnedLines.length > 0) lines.unshift(...pinnedLines);
  if (lines.length === 0) return '';
  const header = '### Computed from the history (deterministic: dates resolved against each session\'s date, arithmetic exact; use these figures rather than recomputing, and check each against the sentence it quotes)\n';
  let block = header + lines.join('\n') + '\n';
  // the lines the question asks for directly are never cut
  const askedLine = lines[pinnedLines.length];
  const protectedLines =
    pinnedLines.length +
    (pinned > 0 || askedLine?.startsWith('Which came first:') === true ? 1 : 0);
  const floor =
    header.length +
    lines.slice(0, protectedLines).reduce((total, line) => total + line.length + 2, 0);
  const limit = Math.max(maxChars, floor);
  if (block.length > limit) block = `${block.slice(0, limit - 2)}…\n`;
  return block;
}

/** The block's lines without the header, for callers that render their own section. */
export function computedNoteLines(
  question: string,
  questionDate: string,
  sources: Array<{ ts: string; text: string }>,
  limits: { maxEvents?: number; maxQuantities?: number; maxChars?: number } = {},
): string[] {
  const block = buildComputedNotes(question, questionDate, sources, limits);
  if (block === '') return [];
  return block.split('\n').slice(1).filter((line) => line.trim() !== '');
}
