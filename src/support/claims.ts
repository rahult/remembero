/**
 * The stable contract: claims in, verdict out. A claim is a positive ground atom citing one
 * span; the admission gate is the part that carries the certainty — the LLM is a sensor, and
 * every claim it produces is admitted only if code can verify it against the span it cites.
 *
 * Three rules, each one earned in the compose engine's adversarial rounds:
 *  - same-span: every argument must appear on the ONE span the claim cites. No claim may be
 *    assembled from words scattered across the transcript; joins are engine work.
 *  - typed: every predicate is registered with argument types, and arguments are canonicalized
 *    (dates, money, durations) before the engine ever sees them.
 *  - positive only: claims state facts, never negations or verdicts. Absence, deadlines and
 *    comparisons are computed symbolically — an extracted negative is a hallucination with
 *    extra steps.
 *
 * Trust is recorded, not scored: `system` claims come from structured fields (no model), `model`
 * claims passed the gate, `hand` claims are the human-gated pack. There is deliberately no
 * confidence number — a model that invents facts cannot tell which ones it invented.
 */

import { createHash } from 'node:crypto';
import type { Source, Span } from './sources.js';
import { findSpan } from './sources.js';

export type ArgType = 'id' | 'word' | 'date' | 'money' | 'hours' | 'minutes' | 'percent';
export type Scope = 'case' | 'pack';
export type Trust = 'system' | 'model' | 'hand';

export interface PredicateSpec {
  args: ArgType[];
  scope: Scope;
  note: string;
}

/**
 * The SLA family's registry. Adding a policy family = extending this registry with typed
 * predicates and writing the rule templates that consume them; the gate and engine mechanics
 * do not change. (Pack predicates are extracted from the customer's SOP into a template —
 * hand-gated before use, per the plan.)
 */
export const PREDICATES: Record<string, PredicateSpec> = {
  // case facts
  opened: { args: ['id', 'date'], scope: 'case', note: 'the ticket was opened at a time' },
  first_response: { args: ['id', 'date'], scope: 'case', note: 'the first agent response on a ticket, at a time' },
  priority: { args: ['id', 'word'], scope: 'case', note: "the ticket's priority (p1..p4)" },
  customer_of: { args: ['id', 'id'], scope: 'case', note: 'which customer a ticket belongs to' },
  tier: { args: ['id', 'word'], scope: 'case', note: "the customer's support tier" },
  monthly_fee: { args: ['id', 'money'], scope: 'case', note: "the customer's monthly fee in dollars" },
  // pack parameters (policy)
  sla_clock: { args: ['word', 'word', 'hours'], scope: 'pack', note: 'max first-response hours for a tier and priority' },
  business_hours: { args: ['id', 'word', 'hours', 'hours'], scope: 'pack', note: 'a calendar is open on a weekday between two hours' },
  holiday: { args: ['id', 'date'], scope: 'pack', note: 'a date a calendar observes as a holiday' },
  zone: { args: ['id', 'id'], scope: 'pack', note: "a calendar's timezone (v0 supports UTC)" },
  default_calendar: { args: ['id'], scope: 'pack', note: 'the calendar applied unless a customer binds another' },
  credit_band: { args: ['minutes', 'percent'], scope: 'pack', note: 'breach reaching this many business minutes earns this percent of the monthly fee' },
};

export interface ClaimInput {
  scope: Scope;
  predicate: string;
  args: (string | number)[];
  sourceId: string;
  spanId: string;
  trust: Trust;
  /** Optional event time; falls back to the cited span's timestamp. */
  at?: string;
}

export interface Claim {
  id: string;
  scope: Scope;
  predicate: string;
  args: (string | number)[];
  source: { sourceId: string; spanId: string; trust: Trust };
  at?: string;
}

export type Admitted = { ok: true; claim: Claim };
export type Rejected = { ok: false; reason: string; input: ClaimInput };
export type GateResult = Admitted | Rejected;

let claimCounter = 0;
function nextClaimId(): string {
  claimCounter += 1;
  return `c${String(claimCounter).padStart(5, '0')}`;
}

/** Lowercase, collapse whitespace, drop commas and unify dashes so surface forms match loosely. */
export function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[–—]/g, '-')
    .replace(/,/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];

/** Parse a date argument into an ISO instant or calendar date; undefined when unparseable. */
export function parseDateArg(value: string): string | undefined {
  const v = value.trim();
  if (/^\d{4}-\d{2}-\d{2}(T[\d:.]+(Z|[+-]\d{2}:?\d{2})?)?$/.test(v)) return v;
  const datetime = v.match(/^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(?:UTC|GMT))?$/i);
  if (datetime) {
    const [, d, h, mi, s] = datetime;
    return `${d}T${h!.padStart(2, '0')}:${mi}:${s ?? '00'}Z`;
  }
  const m = v.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (m) {
    const [, a, b, y] = m;
    return `${y}-${a!.padStart(2, '0')}-${b!.padStart(2, '0')}`;
  }
  const named = v.match(/^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?:\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm))?$/i);
  if (named) {
    const month = MONTHS.findIndex((mo) => mo.startsWith(named[1]!.toLowerCase()));
    if (month >= 0) {
      const date = `${named[3]}-${String(month + 1).padStart(2, '0')}-${named[2]!.padStart(2, '0')}`;
      if (named[4] === undefined) return date;
      const hour = (Number(named[4]) % 12) + (named[6]!.toLowerCase() === 'pm' ? 12 : 0);
      return `${date}T${String(hour).padStart(2, '0')}:${named[5]}:00Z`;
    }
  }
  return undefined;
}

/** Every surface form of an argument that counts as that argument appearing on a span. */
export function surfacesFor(arg: string | number, type: ArgType): string[] {
  const s = String(arg);
  if (type === 'date') {
    const iso = parseDateArg(s);
    if (iso === undefined) return [s.toLowerCase()];
    const [, y, mo, d] = iso.match(/^(\d{4})-(\d{2})-(\d{2})/) ?? [];
    const month = MONTHS[Number(mo) - 1];
    const short = month?.slice(0, 3);
    return [
      `${y}-${mo}-${d}`,
      `${mo}/${d}/${y}`,
      `${month} ${Number(d)} ${y}`,
      `${short} ${Number(d)} ${y}`,
      `${Number(d)} ${month} ${y}`,
    ].filter((x): x is string => Boolean(x)).map((x) => x.toLowerCase());
  }
  if (type === 'money' || type === 'minutes' || type === 'hours' || type === 'percent') {
    const n = Number(arg);
    if (!Number.isFinite(n)) return [s.toLowerCase()];
    const out = [String(n)];
    if (Number.isInteger(n)) out.push(String(n).padStart(2, '0'), `${n}.00`, `$${n}`, `$${n}.00`);
    return out;
  }
  return [s.toLowerCase()];
}

function checkTypes(predicate: string, args: (string | number)[]): string | undefined {
  const spec = PREDICATES[predicate]!;
  for (let i = 0; i < spec.args.length; i += 1) {
    const type = spec.args[i]!;
    const arg = args[i];
    if (type === 'date') {
      if (typeof arg !== 'string' || parseDateArg(arg) === undefined) return `arg ${i + 1} is not a parseable date: ${String(arg)}`;
    } else if (type === 'money' || type === 'minutes' || type === 'hours' || type === 'percent') {
      if (typeof arg !== 'number' || !Number.isFinite(arg)) return `arg ${i + 1} must be a number: ${String(arg)}`;
    } else {
      if (typeof arg !== 'string' || arg.trim() === '') return `arg ${i + 1} must be a non-empty string`;
    }
  }
  return undefined;
}

export function canonicalArgs(predicate: string, args: (string | number)[]): (string | number)[] {
  const spec = PREDICATES[predicate]!;
  return args.map((arg, i) => {
    const type = spec.args[i]!;
    if (type === 'date') return parseDateArg(String(arg))!;
    if (type === 'word' || type === 'id') return String(arg).trim().toLowerCase();
    return arg;
  });
}

/**
 * The gate. Admits a claim only when the predicate is registered, the arguments type-check,
 * and EVERY argument appears on the single cited span (system-trust claims are quoted from
 * the field value, so the same check applies — the quote must be on the span).
 */
export function admit(input: ClaimInput, span: Span): GateResult {
  const spec = PREDICATES[input.predicate];
  if (!spec) return { ok: false, reason: `unknown predicate ${input.predicate}`, input };
  if (spec.scope !== input.scope) return { ok: false, reason: `${input.predicate} is a ${spec.scope} predicate, claimed as ${input.scope}`, input };
  if (input.args.length !== spec.args.length) return { ok: false, reason: `${input.predicate} takes ${spec.args.length} arguments, got ${input.args.length}`, input };
  const typeError = checkTypes(input.predicate, input.args);
  if (typeError) return { ok: false, reason: typeError, input };
  if (span.sourceId !== input.sourceId || span.spanId !== input.spanId) return { ok: false, reason: `cited span ${input.sourceId}/${input.spanId} does not exist`, input };
  const haystack = normalizeText(span.text);
  const args = canonicalArgs(input.predicate, input.args);
  for (let i = 0; i < args.length; i += 1) {
    const surfaces = surfacesFor(args[i]!, spec.args[i]!);
    if (!surfaces.some((surface) => haystack.includes(surface))) {
      return { ok: false, reason: `arg ${i + 1} (${String(args[i])}) not stated on span ${span.spanId}`, input };
    }
  }
  return {
    ok: true,
    claim: {
      id: nextClaimId(),
      scope: input.scope,
      predicate: input.predicate,
      args,
      source: { sourceId: input.sourceId, spanId: input.spanId, trust: input.trust },
      at: input.at ?? span.at,
    },
  };
}

export interface GateReport {
  admitted: Claim[];
  rejected: Array<{ reason: string; predicate: string; spanId: string }>;
}

/** Gate a batch of claims against a set of sources; rejections are kept, with reasons. */
export function admitAll(inputs: ClaimInput[], sources: Source[]): GateReport {
  const admitted: Claim[] = [];
  const rejected: GateReport['rejected'] = [];
  for (const input of inputs) {
    const span = findSpan(sources, input.sourceId, input.spanId);
    const result = span
      ? admit(input, span)
      : { ok: false as const, reason: `cited span ${input.sourceId}/${input.spanId} does not exist`, input };
    if (result.ok) admitted.push(result.claim);
    else rejected.push({ reason: result.reason, predicate: input.predicate, spanId: input.spanId });
  }
  return { admitted, rejected };
}

/** Stable content hash of a claim, for the append-only verdict store. */
export function hashClaim(claim: Claim): string {
  return createHash('sha256').update(JSON.stringify([claim.predicate, claim.args, claim.source, claim.at ?? null])).digest('hex').slice(0, 16);
}

export function hashJson(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 16);
}
