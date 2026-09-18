/*
 * Paragraph ladder: one realistic paragraph of raw text, taken apart in four
 * layers of increasing complexity — facts, dates, logic, change over time.
 *
 * Execution boundary: the fact layer is a replay of a recorded writer
 * reading (the writer is a model; none runs here). Everything below it —
 * date resolution, the arithmetic, the rule firing, and the validity
 * timelines — is deterministic code computing in the visitor's browser from
 * the paragraph's stated date. No model weights are served; nothing is
 * stored; nothing leaves the tab.
 */

export const PARAGRAPH_SAID_ON = "2026-03-14";

export const PARAGRAPH =
  "Team update, 14 March — quick heads-up. I'm stepping back from the Orion rollout at the end of this month; Nadia takes it over from April. We moved the launch two weeks out, so it's now the 28th, not the 14th. Budget-wise we're at $48k of the $60k cap; if the vendor renews at the old rate we'll exceed it by roughly $5k.";

export interface FactCard {
  headline: string;
  machine: string;
  quote: string;
}

export const FACTS: readonly FactCard[] = [
  {
    headline: "You are leading the Orion rollout — until the end of March",
    machine: "rollout_lead(orion, you)",
    quote: "I'm stepping back from the Orion rollout at the end of this month",
  },
  {
    headline: "Nadia takes over the rollout",
    machine: "rollout_lead(orion, nadia)",
    quote: "Nadia takes it over from April",
  },
  {
    headline: "The launch date moved — to the 28th, from the 14th",
    machine: "launch_date(orion, 2026-03-28)",
    quote: "We moved the launch two weeks out, so it's now the 28th, not the 14th",
  },
  {
    headline: "Spend stands at $48k against a $60k cap",
    machine: "budget(orion, $48k of $60k)",
    quote: "we're at $48k of the $60k cap",
  },
];

export const CONDITIONAL_RULE: FactCard = {
  headline: "If the vendor renews at the old rate, spend exceeds the cap by about $5k",
  machine: "exceeds_cap(orion) :- renews_at(orion, old_rate).",
  quote: "if the vendor renews at the old rate we'll exceed it by roughly $5k",
};

/* ------------------------------------------------------------------ */
/* Layer 2 — dates, resolved by code against the day it was said       */
/* ------------------------------------------------------------------ */

function parseIso(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function toIso(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function endOfMonth(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + 1, 0));
}

function firstOfNextMonth(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + 1, 1));
}

function diffDays(from: Date, to: Date): number {
  return Math.round((to.getTime() - from.getTime()) / 86_400_000);
}

export interface ResolvedDate {
  expression: string;
  sentence: string;
  resolved: string;
  daysAfterSaid: number;
}

export function resolveDates(): ResolvedDate[] {
  const said = parseIso(PARAGRAPH_SAID_ON);
  const rows: Array<{ expression: string; sentence: string; date: Date }> = [
    { expression: "at the end of this month", sentence: "I'm stepping back from the Orion rollout at the end of this month", date: endOfMonth(said) },
    { expression: "from April", sentence: "Nadia takes it over from April", date: firstOfNextMonth(said) },
    { expression: "two weeks out (from the 14th)", sentence: "We moved the launch two weeks out, so it's now the 28th", date: new Date(Date.UTC(said.getUTCFullYear(), said.getUTCMonth(), 28)) },
  ];
  return rows.map((row) => ({
    expression: row.expression,
    sentence: row.sentence,
    resolved: toIso(row.date),
    daysAfterSaid: diffDays(said, row.date),
  }));
}

/* ------------------------------------------------------------------ */
/* Layer 3 — logic: arithmetic and the rule, computed by code          */
/* ------------------------------------------------------------------ */

export interface LogicLine {
  label: string;
  detail: string;
  machine: string;
}

export function computeLogic(): LogicLine[] {
  const said = parseIso(PARAGRAPH_SAID_ON);
  const oldLaunch = new Date(Date.UTC(said.getUTCFullYear(), said.getUTCMonth(), 14));
  const newLaunch = new Date(Date.UTC(said.getUTCFullYear(), said.getUTCMonth(), 28));
  const shift = diffDays(oldLaunch, newLaunch);
  const spend = 48_000;
  const cap = 60_000;
  const share = Math.round((spend / cap) * 100);
  const projected = cap + 5_000;
  return [
    {
      label: `The launch moved ${shift} days`,
      detail: `"two weeks out" from the 14th — code checks the arithmetic: 14 March + ${shift} days = 28 March. The paragraph says so; the code confirms it.`,
      machine: `launch_shift(orion, ${shift}d)`,
    },
    {
      label: `Spend is ${share}% of the cap`,
      detail: `$${spend.toLocaleString("en-US")} of $${cap.toLocaleString("en-US")} — computed, not estimated, so questions about headroom start from the right number.`,
      machine: `budget_share(orion, ${share}%)`,
    },
    {
      label: "The conditional fires only on its condition",
      detail: `IF the vendor renews at the old rate THEN projected spend ≈ $${projected.toLocaleString("en-US")} (cap + ~$5k). No renewal, no overage — an unmet condition never becomes a fact.`,
      machine: "exceeds_cap(orion) :- renews_at(orion, old_rate).",
    },
  ];
}

/* ------------------------------------------------------------------ */
/* Layer 4 — change over time: validity timelines, built by code       */
/* ------------------------------------------------------------------ */

export interface TimelineBand {
  fact: string;
  machine: string;
  spans: Array<{ value: string; from: string; to: string | null; status: "current" | "superseded" }>;
}

export function buildChange(): TimelineBand[] {
  const said = parseIso(PARAGRAPH_SAID_ON);
  const iso = (d: Date) => toIso(d);
  return [
    {
      fact: "Who leads the Orion rollout",
      machine: "rollout_lead(orion, _)",
      spans: [
        { value: "You", from: iso(said), to: iso(endOfMonth(said)), status: "superseded" },
        { value: "Nadia", from: iso(firstOfNextMonth(said)), to: null, status: "current" },
      ],
    },
    {
      fact: "When the Orion launch is",
      machine: "launch_date(orion, _)",
      spans: [
        { value: "14 March", from: iso(said), to: iso(said), status: "superseded" },
        { value: "28 March", from: iso(said), to: null, status: "current" },
      ],
    },
  ];
}

export const LADDER_REPLAY_NOTE =
  "The fact layer is a replay of a recorded writer reading. The date resolution, arithmetic, rule, and timelines below it are computed by code in your browser from the paragraph's stated date — run it twice, get the same answer.";

export const LADDER_SCOPE_NOTE =
  "Each layer is the same move — text becomes knowledge you can query — with more machinery behind it. The labs take the layers further: the writer–reader lab runs this on six months of messy history.";
