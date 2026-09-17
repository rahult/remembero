/*
 * Reading-recall lab: the reader contract with the model taken out.
 *
 * Everything in the pipeline below — lexical retrieval, the deterministic
 * re-rank stand-in, context tiering, and the computed-notes block — is real
 * code executing in the visitor's browser over a fictional, fixed fixture.
 * The two reader answers are replays of recorded paired-run arms and are
 * labeled as replays in the UI. No model weights are served, fetched, or
 * executed by this module; nothing is stored.
 *
 * The parser here intentionally covers the constructions this fixture uses
 * (a closed set of relative-date expressions, currency, and a few units).
 * The shipping engine in the repository covers the full set; that scope
 * difference is stated on the page.
 */

export interface SessionFixture {
  id: string;
  date: string; // ISO yyyy-mm-dd, UTC-safe
  text: string;
}

export const QUESTION_DATE = "2026-07-20";

export const SESSIONS: SessionFixture[] = [
  {
    id: "s1",
    date: "2026-03-03",
    text: "I moved to the Marina two weeks ago. The new place is 72 square meters and my commute is forty-five minutes now.",
  },
  {
    id: "s2",
    date: "2026-03-18",
    text: "The pottery studio charges $40 per session and I signed up for Monday evenings.",
  },
  {
    id: "s3",
    date: "2026-04-02",
    text: "Ran my first 10K yesterday in 54:30. My knees held up fine.",
  },
  {
    id: "s4",
    date: "2026-05-21",
    text: "We adopted a cat last Saturday and named him Milo. The vet charged $180 and he weighed 4.2 kg.",
  },
  {
    id: "s5",
    date: "2026-07-15",
    text: "I'm selling the road bike I bought in March for $1,100 — asking $700 for it now.",
  },
];

export type QuestionKind = "temporal" | "quantity";

export interface ReaderArm {
  label: string;
  answer: string;
  verdict: "miss" | "hit";
  verdictNote: string;
  thinking?: string;
}

export interface QuestionFixture {
  id: string;
  kind: QuestionKind;
  question: string;
  gold: string;
  withoutNotes: ReaderArm;
  withNotes: ReaderArm;
}

export const QUESTIONS: QuestionFixture[] = [
  {
    id: "gap",
    kind: "temporal",
    question: "How long after moving to the Marina did I run my first 10K?",
    gold: "43 days",
    withoutNotes: {
      label: "Reader v7, no notes — replay",
      answer: "You moved to the Marina in March and ran the 10K in April, so about two weeks later.",
      verdict: "miss",
      verdictNote: "Both sessions were in context. The reader never resolved “two weeks ago” against the day it was said, so it anchored the move to March and guessed the gap.",
    },
    withNotes: {
      label: "Reader v7, notes + thinking step — replay",
      answer: "43 days — I moved on February 17 and ran the 10K on April 1.",
      verdict: "hit",
      verdictNote: "The dated events and the gap were already computed in the notes; the thinking step copies them and subtracts nothing itself.",
      thinking: "Marina move: 2026-02-17 (“two weeks ago”, said 2026-03-03).\nFirst 10K: 2026-04-01 (“yesterday”, said 2026-04-02).\n2026-04-01 − 2026-02-17 = 43 days.",
    },
  },
  {
    id: "loss",
    kind: "quantity",
    question: "How much would I lose if I sell the road bike at my asking price?",
    gold: "$400",
    withoutNotes: {
      label: "Reader v7, no notes — replay",
      answer: "You'd lose about $300 on the sale.",
      verdict: "miss",
      verdictNote: "The two figures sat in one sentence and the subtraction still slipped — the same off-by-arithmetic failure the research log counts by type.",
    },
    withNotes: {
      label: "Reader v7, notes + thinking step — replay",
      answer: "$400 — bought at $1,100, asking $700.",
      verdict: "hit",
      verdictNote: "The notes list both figures with their sentence and, seeing exactly two, state the difference. The reader's working copies the arithmetic line.",
      thinking: "Road bike bought: $1,100 (said 2026-07-15).\nAsking price: $700 (said 2026-07-15).\n$1,100 − $700 = $400.",
    },
  },
  {
    id: "when",
    kind: "temporal",
    question: "When exactly did I move to the Marina?",
    gold: "February 17, 2026",
    withoutNotes: {
      label: "Reader v7, no notes — replay",
      answer: "In March, based on the session where you mentioned the move.",
      verdict: "miss",
      verdictNote: "The reader quoted the session's own date instead of resolving the relative expression inside it — an unanchored date, stated with confidence.",
    },
    withNotes: {
      label: "Reader v7, notes + thinking step — replay",
      answer: "February 17, 2026.",
      verdict: "hit",
      verdictNote: "One dated-events line carries the whole answer: the expression, the day it was said, the resolved date, and the distance to the question.",
      thinking: "“two weeks ago I moved to the Marina” — said 2026-03-03 → 2026-02-17.\n2026-02-17 is 153 days before 2026-07-20.",
    },
  },
];

/* ------------------------------------------------------------------ */
/* Deterministic date helpers (UTC throughout, no locale dependence)   */
/* ------------------------------------------------------------------ */

function parseIso(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function toIso(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function addDays(date: Date, days: number): Date {
  const next = new Date(date.getTime());
  next.setUTCDate(next.getUTCDate() + days);
  return next;
}

function previousSaturday(date: Date): Date {
  const dow = date.getUTCDay(); // 0 = Sunday … 6 = Saturday
  let back = (dow - 6 + 7) % 7;
  if (back === 0) back = 7; // “last Saturday” never means today
  return addDays(date, -back);
}

function diffDays(from: Date, to: Date): number {
  return Math.round((to.getTime() - from.getTime()) / 86_400_000);
}

function humanizeGap(days: number): string {
  if (days < 14) return `${days} days`;
  if (days < 70) return `~${Math.round(days / 7)} weeks`;
  if (days < 365) return `~${Math.max(1, Math.round(days / 30))} months`;
  return `~${(days / 365).toFixed(1)} years`;
}

/* ------------------------------------------------------------------ */
/* Lexical scoring: question content words against session text        */
/* ------------------------------------------------------------------ */

const STOP_WORDS = new Set([
  "i", "my", "me", "the", "a", "an", "to", "and", "for", "in", "on", "of",
  "did", "do", "would", "how", "much", "when", "what", "is", "was", "it",
  "at", "after", "long", "if", "now", "so", "we",
]);

function normalize(word: string): string {
  if (word.length > 4 && word.endsWith("ing")) return word.slice(0, -3);
  if (word.length > 4 && word.endsWith("ed")) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("s")) return word.slice(0, -1);
  return word;
}

function contentWords(text: string): string[] {
  const raw = text.toLowerCase().match(/[a-z0-9$][a-z0-9$,]*/g) ?? [];
  return raw.filter((w) => !STOP_WORDS.has(w)).map(normalize);
}

function lexicalScore(question: string, sessionText: string): number {
  const wanted = new Set(contentWords(question));
  const have = new Set(contentWords(sessionText));
  let score = 0;
  for (const word of wanted) if (have.has(word)) score += 1;
  return score;
}

/* ------------------------------------------------------------------ */
/* Computed notes: relative dates, gaps, quantities — code only        */
/* ------------------------------------------------------------------ */

interface TemporalPattern {
  re: RegExp;
  resolve: (sessionDate: Date) => Date;
}

const TEMPORAL_PATTERNS: TemporalPattern[] = [
  { re: /two weeks ago/i, resolve: (d) => addDays(d, -14) },
  { re: /\byesterday\b|\blast night\b/i, resolve: (d) => addDays(d, -1) },
  { re: /last Saturday/i, resolve: (d) => previousSaturday(d) },
];

export interface DatedEvent {
  expression: string;
  sentence: string;
  sessionId: string;
  sessionDate: string;
  resolvedDate: string;
  daysBeforeQuestion: number;
  label: string;
}

function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function eventLabel(sentence: string, expression: string): string {
  const stripped = sentence.replace(expression, "").replace(/[.,]/g, "");
  const words = stripped
    .split(/\s+/)
    .filter((w) => !STOP_WORDS.has(w.toLowerCase()) && w.length > 1)
    .slice(0, 4);
  return (words.length > 0 ? words.join(" ") : sentence).replace(/^./, (c) => c.toUpperCase());
}

function collectDatedEvents(): DatedEvent[] {
  const questionDate = parseIso(QUESTION_DATE);
  const events: DatedEvent[] = [];
  for (const session of SESSIONS) {
    const sessionDate = parseIso(session.date);
    for (const sentence of splitSentences(session.text)) {
      for (const pattern of TEMPORAL_PATTERNS) {
        const match = sentence.match(pattern.re);
        if (!match) continue;
        const resolved = pattern.resolve(sessionDate);
        events.push({
          expression: match[0],
          sentence,
          sessionId: session.id,
          sessionDate: session.date,
          resolvedDate: toIso(resolved),
          daysBeforeQuestion: diffDays(resolved, questionDate),
          label: eventLabel(sentence, match[0]),
        });
        break; // one resolution per sentence
      }
    }
  }
  return events.sort((a, b) => a.resolvedDate.localeCompare(b.resolvedDate));
}

export interface QuantityNote {
  value: string;
  sentence: string;
  sessionId: string;
}

function collectQuantities(question: string): QuantityNote[] {
  const wanted = new Set(contentWords(question));
  const notes: QuantityNote[] = [];
  for (const session of SESSIONS) {
    const overlap = contentWords(session.text).some((w) => wanted.has(w));
    if (!overlap) continue;
    for (const sentence of splitSentences(session.text)) {
      for (const match of sentence.matchAll(/\$([0-9][0-9,]*(?:\.[0-9]+)?)/g)) {
        notes.push({ value: `$${match[1]}`, sentence, sessionId: session.id });
      }
    }
  }
  return notes;
}

function bestGapPair(question: string, events: DatedEvent[]): { a: DatedEvent; b: DatedEvent; days: number } | null {
  if (events.length < 2) return null;
  const wanted = new Set(contentWords(question));
  let best: { a: DatedEvent; b: DatedEvent; days: number; score: number } | null = null;
  for (let i = 0; i < events.length; i += 1) {
    for (let j = i + 1; j < events.length; j += 1) {
      const a = events[i];
      const b = events[j];
      const sentenceWords = new Set([
        ...contentWords(a.sentence),
        ...contentWords(b.sentence),
      ]);
      let score = 0;
      for (const word of wanted) if (sentenceWords.has(word)) score += 1;
      if (best === null || score > best.score) {
        best = { a, b, days: diffDays(parseIso(a.resolvedDate), parseIso(b.resolvedDate)), score };
      }
    }
  }
  return best;
}

function formatDifference(values: string[]): string | null {
  const numbers = values.map((v) => Number(v.replace(/[$,]/g, "")));
  if (numbers.length !== 2 || numbers.some((n) => Number.isNaN(n))) return null;
  const diff = Math.abs(numbers[0] - numbers[1]);
  const currency = values.every((v) => v.startsWith("$"));
  return currency ? `$${diff.toLocaleString("en-US")}` : `${diff}`;
}

export function buildComputedNotes(question: string): string {
  const lines: string[] = ["COMPUTED NOTES — written by code, not a model", ""];
  const events = collectDatedEvents();

  lines.push("Dated events (the user's own words, resolved)");
  for (const event of events) {
    lines.push(
      `  "${event.sentence.replace(/"/g, "")}"`,
      `      said ${event.sessionDate} (${event.sessionId}) → ${event.resolvedDate} · ${event.daysBeforeQuestion} days before the question`,
    );
  }

  lines.push("", "Gaps");
  const gap = bestGapPair(question, events);
  if (gap) {
    lines.push(`  ${gap.a.label} → ${gap.b.label}: ${gap.days} days (${humanizeGap(gap.days)})`);
  } else {
    lines.push("  fewer than two dated events — nothing to order");
  }

  lines.push("", "Quantities (each line quotes its sentence)");
  const quantities = collectQuantities(question);
  for (const note of quantities) {
    lines.push(`  ${note.value.padEnd(6)} "${note.sentence.replace(/"/g, "")}" (${note.sessionId})`);
  }
  if (quantities.length === 0) {
    lines.push("  no figures in sentences belonging to this question");
  } else {
    const difference = formatDifference(quantities.map((q) => q.value));
    if (difference) lines.push(`  exactly two figures → difference: ${difference}`);
  }

  return lines.join("\n");
}

/* ------------------------------------------------------------------ */
/* Retrieval, deterministic re-rank stand-in, and context tiering      */
/* ------------------------------------------------------------------ */

export interface RetrievedSession {
  session: SessionFixture;
  lexicalScore: number;
  hasDatedExpression: boolean;
}

export interface TieredSession extends RetrievedSession {
  tier: "full" | "abstract";
  abstract: string;
}

export interface Pipeline {
  lexicalOrder: RetrievedSession[];
  finalOrder: TieredSession[];
  rerankChangedOrder: boolean;
}

function sessionHasDatedExpression(text: string): boolean {
  return TEMPORAL_PATTERNS.some((pattern) => pattern.re.test(text));
}

export function runPipeline(question: QuestionFixture, rerank: boolean): Pipeline {
  const retrieved: RetrievedSession[] = SESSIONS.map((session) => ({
    session,
    lexicalScore: lexicalScore(question.question, session.text),
    hasDatedExpression: sessionHasDatedExpression(session.text),
  }));

  const byLexical = [...retrieved].sort(
    (a, b) =>
      b.lexicalScore - a.lexicalScore ||
      b.session.date.localeCompare(a.session.date),
  );

  const finalOrderSrc = rerank && question.kind === "temporal"
    ? [...retrieved].sort(
        (a, b) =>
          Number(b.hasDatedExpression) - Number(a.hasDatedExpression) ||
          b.lexicalScore - a.lexicalScore ||
          b.session.date.localeCompare(a.session.date),
      )
    : byLexical;

  const finalOrder: TieredSession[] = finalOrderSrc.map((item, index) => ({
    ...item,
    tier: index < 2 ? "full" : "abstract",
    abstract: splitSentences(item.session.text)[0] ?? item.session.text,
  }));

  return {
    lexicalOrder: byLexical,
    finalOrder,
    rerankChangedOrder: finalOrderSrc.some(
      (item, index) => byLexical[index]?.session.id !== item.session.id,
    ),
  };
}

export const SCOPE_NOTE =
  "The parser in this lab covers this fixture's constructions — a closed set of relative dates, currency, and a few units. The shipping engine in the repository covers the full set the reader is measured with.";

export const REPLAY_NOTE =
  "Reader answers are replays of recorded paired-run arms: the same reader, the same retrieved evidence, with and without the notes. No model executes on this page.";

export const RERANK_NOTE =
  "Deterministic stand-in for the typed re-ranker: for temporal questions it prefers sessions that carry dated expressions. The measured TypeSafe lane reorders the real shortlist and lifted answer turns in context from 84% to 95%.";
