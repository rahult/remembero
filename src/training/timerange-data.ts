/**
 * Time-range training data: teach the writer adapter the time-range extractor
 * role the LongMemEval harness gives a frontier model today. Given the question
 * date and a question, reply with the absolute range the question refers to, or
 * refuse. The paper's finding, confirmed here with Luna and GLM, is that a
 * model that guesses ranges hurts and one that refuses three questions in four
 * helps, so refusals are the majority class and every cue is a real cue.
 *
 * The prompt is `temporalRangePrompt` from the harness, byte for byte, so the
 * trained adapter plugs in as `--temporal-range-model`.
 */
import { temporalRangePrompt } from '../evals/longmemeval-answer.js';
import type { Conversation } from './export.js';
import type { Rng } from './rng.js';

export interface TimeRangeExample {
  questionDate: string;
  question: string;
  /** The cue phrase used, for tests and manifests; '' for a refusal. */
  cue: string;
  range: { start: string; end: string } | null;
}

const ACTIVITIES = [
  'what did I cook',
  'which books did I finish',
  'how many times did I go running',
  'what did I buy for the house',
  'which movies did I watch',
  'what did I work on',
  'where did I travel',
  'who did I meet up with',
  'what did I plant in the garden',
  'which recipes did I try',
  'what did I fix around the apartment',
  'which concerts did I go to',
  'what did I learn in my classes',
  'which restaurants did I try',
  'what did I do with the kids',
];

const EVENTS = [
  'my dentist appointment',
  'the camping trip',
  'my sister’s wedding',
  'the marathon',
  'my job interview',
  'the move to the new apartment',
  'my birthday',
  'the conference',
  'the road trip',
  'my last haircut',
];

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const WIDEN_DAYS = 3;

function iso(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function utc(year: number, monthIndex: number, day: number): Date {
  return new Date(Date.UTC(year, monthIndex, day));
}

function addDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * 86_400_000);
}

function widen(start: Date, end: Date): { start: string; end: string } {
  return {
    start: iso(addDays(start, -WIDEN_DAYS)),
    end: iso(addDays(end, WIDEN_DAYS)),
  };
}

function endOfMonth(year: number, monthIndex: number): Date {
  return utc(year, monthIndex + 1, 0);
}

interface Cue {
  phrase: (
    rng: Rng,
    today: Date,
  ) => { text: string; range: { start: string; end: string } };
}

/** Cues that name a period; each computes its range from the question date. */
const RANGED_CUES: Cue[] = [
  {
    phrase: (_rng, today) => {
      const start = utc(today.getUTCFullYear(), today.getUTCMonth() - 1, 1);
      const end = endOfMonth(today.getUTCFullYear(), today.getUTCMonth() - 1);
      return { text: 'last month', range: widen(start, end) };
    },
  },
  {
    phrase: (_rng, today) => {
      const dow = today.getUTCDay();
      const thisMonday = addDays(today, -((dow + 6) % 7));
      const start = addDays(thisMonday, -7);
      return { text: 'last week', range: widen(start, addDays(start, 6)) };
    },
  },
  {
    phrase: (_rng, today) => {
      const start = utc(today.getUTCFullYear(), 0, 1);
      return { text: 'this year', range: widen(start, today) };
    },
  },
  {
    phrase: (_rng, today) => {
      const year = today.getUTCFullYear() - 1;
      return {
        text: `in ${year}`,
        range: widen(utc(year, 0, 1), utc(year, 11, 31)),
      };
    },
  },
  {
    phrase: (rng, today) => {
      // a month earlier this year or late last year
      const back = 1 + rng.int(9);
      const monthIndex = today.getUTCMonth() - back;
      const year = today.getUTCFullYear() + Math.floor(monthIndex / 12);
      const m = ((monthIndex % 12) + 12) % 12;
      const text = rng.pick([
        `in ${MONTHS[m]}`,
        `back in ${MONTHS[m]}`,
        `during ${MONTHS[m]}`,
      ]);
      return { text, range: widen(utc(year, m, 1), endOfMonth(year, m)) };
    },
  },
  {
    phrase: (rng, today) => {
      const weeks = 2 + rng.int(5);
      const anchor = addDays(today, -7 * weeks);
      return {
        text: rng.pick([
          `${weeks} weeks ago`,
          `about ${weeks} weeks ago`,
          `around ${weeks} weeks back`,
        ]),
        range: widen(addDays(anchor, -4), addDays(anchor, 4)),
      };
    },
  },
  {
    phrase: (_rng, today) => {
      const start = addDays(today, -1);
      return { text: 'yesterday', range: widen(start, start) };
    },
  },
  {
    phrase: (_rng, today) => {
      const dow = today.getUTCDay();
      const monday = addDays(today, -((dow + 6) % 7));
      return { text: 'earlier this week', range: widen(monday, today) };
    },
  },
  {
    phrase: (_rng, today) => {
      const start = addDays(today, -14);
      return { text: 'over the past two weeks', range: widen(start, today) };
    },
  },
  {
    phrase: (rng, today) => {
      const year =
        today.getUTCMonth() < 8
          ? today.getUTCFullYear() - 1
          : today.getUTCFullYear();
      const season = rng.pick(['summer', 'spring', 'winter', 'autumn']);
      const spans: Record<string, [number, number]> = {
        spring: [2, 4],
        summer: [5, 7],
        autumn: [8, 10],
        winter: [11, 1],
      };
      const [a, b] = spans[season];
      const start = utc(year, a, 1);
      const end =
        season === 'winter' ? endOfMonth(year + 1, b) : endOfMonth(year, b);
      return { text: `last ${season}`, range: widen(start, end) };
    },
  },
];

/** Questions with no period to retrieve by: arithmetic about a past event, first/last, or none. */
const REFUSAL_TEMPLATES: Array<(rng: Rng) => string> = [
  (rng) => `How many weeks ago was ${rng.pick(EVENTS)}?`,
  (rng) => `How long has it been since ${rng.pick(EVENTS)}?`,
  (rng) =>
    `How many days passed between ${rng.pick(EVENTS)} and ${rng.pick(EVENTS)}?`,
  (rng) => `When did I mention ${rng.pick(EVENTS)}?`,
  (rng) => `What was the first thing I told you about ${rng.pick(EVENTS)}?`,
  (rng) => `How many months have passed since ${rng.pick(EVENTS)}?`,
  (rng) => `${capital(rng.pick(ACTIVITIES))} the most often?`,
  (rng) => `What do I usually do on weekends?`,
  (rng) => `Which of my hobbies did I take up most recently?`,
  (rng) => `What is my dentist’s name?`,
  (rng) => `How many kits have I built in total?`,
  (rng) => `Which book did I finish right after ${rng.pick(EVENTS)}?`,
];

function capital(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function randomDate(rng: Rng): Date {
  const year = 2022 + rng.int(3);
  const month = rng.int(12);
  const day = 1 + rng.int(28);
  return utc(year, month, day);
}

export function generateTimeRangeExamples(
  rng: Rng,
  count: number,
): TimeRangeExample[] {
  const out: TimeRangeExample[] = [];
  for (let i = 0; i < count; i += 1) {
    const today = randomDate(rng);
    const questionDate = iso(today);
    // refusals in the majority: the evaluation found 20 to 36 cues per 133 questions
    if (rng.next() < 0.65) {
      out.push({
        questionDate,
        question: rng.pick(REFUSAL_TEMPLATES)(rng),
        cue: '',
        range: null,
      });
      continue;
    }
    const cue = rng.pick(RANGED_CUES).phrase(rng, today);
    const activity = rng.pick(ACTIVITIES);
    const question = rng.pick([
      `${capital(activity)} ${cue.text}?`,
      `${capital(cue.text)}, ${activity}?`,
      `Can you remind me ${activity} ${cue.text}?`,
    ]);
    out.push({ questionDate, question, cue: cue.text, range: cue.range });
  }
  return out;
}

export function toTimeRangeConversation(
  example: TimeRangeExample,
): Conversation {
  return {
    messages: [
      {
        role: 'user',
        content: temporalRangePrompt(example.question, example.questionDate),
      },
      {
        role: 'assistant',
        content:
          example.range === null
            ? '{"none":true}'
            : JSON.stringify({
                start: example.range.start,
                end: example.range.end,
              }),
      },
    ],
  };
}
