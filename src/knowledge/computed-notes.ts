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
const MONTH_RE = '(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)';
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
  kind: 'absolute' | 'relative' | 'weekday';
  approximate?: boolean;
  assumedYear?: boolean;
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

/** User turns only: the assistant's hypotheticals must not become the user's dates. */
export function userSentences(text: string): string[] {
  const out: string[] = [];
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
    for (const sentence of body.split(/(?<=[.!?])\s+|\n+/)) {
      const s = sentence.trim();
      if (s.length > 2) out.push(s);
    }
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

export function resolveTemporalExpressions(text: string, sessionTs: string): DatedEvent[] {
  const sessionDay = dayOf(sessionTs);
  const base = msOf(sessionDay);
  const year = Number(sessionDay.slice(0, 4));
  const events: DatedEvent[] = [];
  const push = (e: DatedEvent) => {
    if (!events.some((x) => x.iso === e.iso && x.sentence === e.sentence)) events.push(e);
  };
  for (const sentence of userSentences(text)) {
    const s = sentence;
    let m: RegExpExecArray | null;
    // ISO and numeric dates
    const iso = /\b(\d{4})-(\d{2})-(\d{2})\b/g;
    while ((m = iso.exec(s)) !== null) push({ iso: `${m[1]}-${m[2]}-${m[3]}`, expression: m[0], sentence, sessionDay, kind: 'absolute' });
    const us = /\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b/g;
    while ((m = us.exec(s)) !== null) {
      const y = m[3].length === 2 ? 2000 + Number(m[3]) : Number(m[3]);
      push({ iso: toIso(utc(y, Number(m[1]), Number(m[2]))), expression: m[0], sentence, sessionDay, kind: 'absolute' });
    }
    // "on 2/15" without a year: month/day, the year assumed from the session
    const usShort = /(?<![\d/])(\d{1,2})\/(\d{1,2})(?![\d/])/g;
    while ((m = usShort.exec(s)) !== null) {
      const mo = Number(m[1]);
      const d = Number(m[2]);
      if (mo < 1 || mo > 12 || d < 1 || d > 31) continue;
      push({ iso: toIso(utc(year, mo, d)), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: true });
    }
    // "March 7th, 2023" / "March 7" / "7 March 2023" / "7th of March"
    const monthDay = new RegExp(`\\b(${MONTH_RE})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b`, 'gi');
    while ((m = monthDay.exec(s)) !== null) {
      const mo = MONTHS[m[1].toLowerCase()];
      if (!mo) continue;
      // "the March 15th issue", "the June 3 edition": a date naming a thing, not when it happened
      if (/^\s+(issue|edition|newsletter|magazine|episode|release|deadline|version)\b/i.test(s.slice(m.index + m[0].length))) continue;
      const y = m[3] ? Number(m[3]) : year;
      push({ iso: toIso(utc(y, mo, Number(m[2]))), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: !m[3] });
    }
    const dayMonth = new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(${MONTH_RE})\\.?(?:,?\\s+(\\d{4}))?\\b`, 'gi');
    while ((m = dayMonth.exec(s)) !== null) {
      const mo = MONTHS[m[2].toLowerCase()];
      if (!mo) continue;
      const y = m[3] ? Number(m[3]) : year;
      push({ iso: toIso(utc(y, mo, Number(m[1]))), expression: m[0], sentence, sessionDay, kind: 'absolute', assumedYear: !m[3] });
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
      while ((m = re.exec(s)) !== null) push({ iso: toIso(base + delta * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative' });
    }
    const ago = /\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|couple of|a couple of|few|a few|several|half a)\s+(day|week|month|year)s?\s+(ago|earlier|before)\b/gi;
    while ((m = ago.exec(s)) !== null) {
      const n = number(m[1].replace(/^a /, '').replace(/ of$/, '').replace(/ a$/, '')) ?? SMALL_NUMBERS[m[1].split(' ').pop()!];
      if (n === undefined) continue;
      const unit = m[2].toLowerCase();
      const days = unit === 'day' ? n : unit === 'week' ? n * 7 : unit === 'month' ? n * 30.44 : n * 365.25;
      push({ iso: toIso(base - Math.round(days) * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: unit !== 'day' && unit !== 'week' });
    }
    const ahead = /\b(?:in\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?|(\d+|a|one|two|three|four|five|six)\s+(day|week|month|year)s?\s+from now)\b/gi;
    while ((m = ahead.exec(s)) !== null) {
      const n = number(m[1] ?? m[3] ?? '');
      const unit = (m[2] ?? m[4] ?? '').toLowerCase();
      if (n === undefined || !unit) continue;
      const days = unit === 'day' ? n : unit === 'week' ? n * 7 : unit === 'month' ? n * 30.44 : n * 365.25;
      push({ iso: toIso(base + Math.round(days) * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: unit !== 'day' && unit !== 'week' });
    }
    const lastUnit = /\blast (week|month|year)\b/gi;
    while ((m = lastUnit.exec(s)) !== null) {
      const days = m[1].toLowerCase() === 'week' ? 7 : m[1].toLowerCase() === 'month' ? 30 : 365;
      push({ iso: toIso(base - days * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: true });
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
      push({ iso: toIso(utc(y, mo, d)), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: true });
    }
    // anchored offsets: "a week before Black Friday", "two days after Christmas", "3 days before March 7"
    const anchored = new RegExp(`\\b(\\d+|a|an|one|two|three|four|five|six|seven|ten)\\s+(day|week|month)s?\\s+(before|after|prior to|following)\\s+(black friday|thanksgiving|christmas(?: day)?|christmas eve|new year'?s? day|new year'?s? eve|halloween|independence day|the fourth of july|july 4th|valentine'?s? day|labor day|memorial day|(?:${MONTH_RE})\\.?\\s+\\d{1,2}(?:st|nd|rd|th)?(?:,?\\s+\\d{4})?)`, 'gi');
    while ((m = anchored.exec(s)) !== null) {
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
      push({ iso: toIso(anchor + sign * days * DAY_MS), expression: m[0], sentence, sessionDay, kind: 'relative', approximate: unit === 'month' });
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
      push({ iso, expression: m[0], sentence, sessionDay, kind: 'weekday' });
    }
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

function relevant(sentence: string, keywords: string[]): boolean {
  const s = sentence.toLowerCase();
  return keywords.some((k) => s.includes(k));
}

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

/**
 * The block for the reader, or '' when the history has nothing datable or countable near the
 * question. Every line names the sentence it came from so the reader can check it.
 */
export function buildComputedNotes(
  question: string,
  questionDate: string,
  sources: Array<{ ts: string; text: string }>,
  limits: { maxEvents?: number; maxQuantities?: number; maxChars?: number } = {},
): string {
  const questionDay = dayOf(questionDate);
  const keywords = questionKeywords(question);
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
  const keywordHits = (sentence: string) => keywords.filter((k) => sentence.toLowerCase().includes(k)).length;

  const events: DatedEvent[] = [];
  const allEvents: DatedEvent[] = [];
  const quantities: Quantity[] = [];
  for (const source of [...sources].sort((l, r) => l.ts.localeCompare(r.ts))) {
    for (const e of resolveTemporalExpressions(source.text, source.ts)) {
      allEvents.push(e);
      if (relevant(e.sentence, keywords)) events.push(e);
    }
    for (const x of extractQuantities(source.text)) {
      // a quantity belongs to the question when its unit is named in the question, or its
      // sentence shares two content words with it; one shared word lets in every stray number
      if (unitsInQuestion.has(x.unit) ? keywordHits(x.sentence) >= 1 : keywordHits(x.sentence) >= 2) quantities.push(x);
    }
  }
  const lines: string[] = [];
  // the question's own time reference ("last Saturday", "two months ago", "the past weekend"):
  // resolve it against the question date and point at the dated events closest to it
  const questionRefs = resolveTemporalExpressions(`USER: ${question.replace(/\bthe past weekend\b/i, 'last Saturday')}`, `${questionDay}T00:00:00Z`).filter((e) => e.kind !== 'absolute');
  const dated = events.slice(0, maxEvents);
  if (questionRefs.length > 0 && allEvents.length > 0) {
    const ref = questionRefs[0]!;
    // every dated event counts here, whatever its wording: the question asks what happened then
    const byDistance = [...new Map(allEvents.map((e) => [e.iso, e])).values()].sort((l, r) => Math.abs(msOf(l.iso) - msOf(ref.iso)) - Math.abs(msOf(r.iso) - msOf(ref.iso)));
    const closest = byDistance.slice(0, 2).map((e) => `${e.iso} ("${snippet(e.sentence, 60)}", ${Math.round(Math.abs(msOf(e.iso) - msOf(ref.iso)) / DAY_MS)} days away)`);
    lines.push(`The question's "${ref.expression}" counted from the question date ${questionDay} is ${ref.iso}${ref.approximate ? ' (approximate)' : ''}; the closest dated event${closest.length > 1 ? 's' : ''}: ${closest.join('; ')}.`);
  }
  if (dated.length > 0) {
    lines.push('Dated events (each temporal expression resolved against the date of the session it was said in):');
    for (const e of dated) {
      const flags = [e.approximate ? 'approximate' : '', e.assumedYear ? 'year assumed from the session date' : ''].filter(Boolean).join('; ');
      lines.push(`- ${e.iso}: "${snippet(e.sentence)}" [said ${e.sessionDay}, "${e.expression}"${flags ? `; ${flags}` : ''}] — ${distance(e.iso, questionDay)}`);
    }
    const distinct = [...new Map(dated.map((e) => [e.iso, e])).values()].sort((l, r) => l.iso.localeCompare(r.iso));
    if (asksOrder) {
      // "X or Y": which alternatives the dated events actually cover
      const alternatives = question
        .split(/\s+or\s+|,\s*(?:and\s+)?|\s+and then\s+/i)
        .filter((part) => !/\b(which|what|how|when|did|do|does|who|where)\b/i.test(part))
        // number words and ordinals match everywhere ("three weeks ago"); they do not identify an alternative
        .map((part) => questionKeywords(part).filter((k) => SMALL_NUMBERS[k] === undefined && !/^(first|last|second|third|next|previous|earlier|later)$/.test(k)))
        .filter((ks) => ks.length > 0);
      if (alternatives.length >= 2) {
        const coverage = alternatives.map((ks) => ({ ks, dated: distinct.some((e) => ks.some((k) => e.sentence.toLowerCase().includes(k))) }));
        const undated = coverage.filter((c) => !c.dated);
        if (undated.length > 0 && coverage.some((c) => c.dated)) {
          lines.push(`Coverage: the dated events above match ${coverage.filter((c) => c.dated).map((c) => `"${c.ks.slice(0, 4).join(' ')}"`).join(', ')} but none match ${undated.map((c) => `"${c.ks.slice(0, 4).join(' ')}"`).join(', ')}; if the history never dates one side of the question, the honest answer is that it does not say.`);
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
      const covered = (a: DatedEvent, b: DatedEvent) => new Set(keywords.filter((k) => a.sentence.toLowerCase().includes(k) || b.sentence.toLowerCase().includes(k))).size;
      const distinctCover = (a: DatedEvent, b: DatedEvent) => Math.min(keywordHits(a.sentence), keywordHits(b.sentence));
      pairs.sort((l, r) => covered(r[0], r[1]) - covered(l[0], l[1]) || distinctCover(r[0], r[1]) - distinctCover(l[0], l[1]));
      const shown = pairs.slice(0, 4);
      lines.push(`Gaps between dated events (the pair whose sentences best match the question is listed first${pairs.length > shown.length ? `; ${pairs.length - shown.length} other pairs omitted` : ''}):`);
      for (const [a, b] of shown) lines.push(`- ${a.iso} ("${snippet(a.sentence, 40)}") to ${b.iso} ("${snippet(b.sentence, 40)}"): ${gap(a.iso, b.iso)}${a.approximate || b.approximate ? ' [approximate: one end is a rough expression like "a month ago"]' : ''}`);
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
  if (lines.length === 0) return '';
  const header = '### Computed from the history (deterministic: dates resolved against each session\'s date, arithmetic exact; use these figures rather than recomputing, and check each against the sentence it quotes)\n';
  let block = header + lines.join('\n') + '\n';
  if (block.length > maxChars) block = `${block.slice(0, maxChars - 2)}…\n`;
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
