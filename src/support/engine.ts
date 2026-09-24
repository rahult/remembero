/**
 * The symbolic layer: closed query shapes over admitted claims — never free-form rules —
 * returning a verdict with its proof. Same claims + same rules => same verdict + same proof;
 * the LLM is nowhere on this path.
 *
 * Shapes are an enum, like the compose engine's nine families: a question that does not map
 * to a shape is rejected by the planner, never guessed. Every miss is an Unknown with a
 * structural reason — missing fact, conflicting facts, provably incomplete calendar, pack
 * parameter absent — and the proof says what was searched. A decline with the evidence
 * present is the failure mode that killed readers; here it is impossible by construction.
 *
 * v0 policy family: first-response SLA credits. The wedge on purpose — breach is arithmetic
 * over timestamps plus a business calendar (weekends, holidays, hours), exactly the multi-hop
 * date math readers fumble and this engine proves.
 */

import type { Claim } from './claims.js';

export const ENGINE_RULE = 'sla-credit@v1';

export type Shape = 'sla_credits' | 'sla_met' | 'credit_percent';
export const SHAPES: readonly Shape[] = ['sla_credits', 'sla_met', 'credit_percent'];

export type Verdict = 'allow' | 'deny' | 'unknown';

export interface Proof {
  shape: Shape;
  params: Record<string, string>;
  verdict: Verdict;
  rule: string;
  pack: { id: string; version: string };
  decidedAt: string;
  /** One line a human can read: what was concluded and why. */
  summary: string;
  /** Every claim the verdict rests on, with span provenance. */
  facts: Array<{ id: string; predicate: string; args: (string | number)[]; source: Claim['source']; at?: string }>;
  computation?: Record<string, string | number>;
  unknown?: { reasons: string[]; missing: string[]; spansSearched?: number };
}

export interface DecideOptions {
  packId: string;
  packVersion: string;
  /** How many spans the window assembler scanned, for the Unknown's "what was searched". */
  spansSearched?: number;
  decidedAt?: string;
}

const WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] as const;
const MAX_WINDOW_DAYS = 400;

interface Ctx {
  claims: Claim[];
  byTicket: (predicate: string, ticket: string) => Claim[];
  pack: (predicate: string) => Claim[];
}

function ctx(claims: Claim[]): Ctx {
  return {
    claims,
    byTicket: (predicate, ticket) =>
      claims.filter((c) => c.predicate === predicate && c.args[0] === ticket),
    pack: (predicate) => claims.filter((c) => c.predicate === predicate),
  };
}

/** One value per key: undefined when absent, `conflict` marker when two distinct values. */
function sole<T extends (string | number)[]>(
  rows: Claim[],
  value: (c: Claim) => T[number],
): { value?: T[number]; conflict?: T[number][] } {
  if (rows.length === 0) return {};
  const distinct = [...new Set(rows.map(value))];
  return distinct.length === 1 ? { value: distinct[0] } : { conflict: distinct };
}

function factOf(claim: Claim): Proof['facts'][number] {
  return { id: claim.id, predicate: claim.predicate, args: claim.args, source: claim.source, at: claim.at };
}

/** Business minutes between two ISO instants on a calendar — the part readers get wrong. */
function businessMinutes(startIso: string, endIso: string, calendar: string, c: Ctx, facts: Proof['facts']):
  { minutes?: number; error?: string } {
  const hours = new Map<string, { open: number; close: number }>();
  for (const claim of c.pack('business_hours').filter((h) => h.args[0] === calendar)) {
    const weekday = String(claim.args[1]);
    const open = claim.args[2] as number;
    const close = claim.args[3] as number;
    const existing = hours.get(weekday);
    if (existing && (existing.open !== open || existing.close !== close)) return { error: `calendar-incomplete: conflicting hours stated for ${weekday}` };
    hours.set(weekday, { open, close });
  }
  const holidays = new Set(c.pack('holiday').filter((h) => h.args[0] === calendar).map((h) => String(h.args[1])));
  for (const h of c.pack('holiday').filter((h) => h.args[0] === calendar)) facts.push(factOf(h));

  const zone = sole(c.pack('zone').filter((z) => z.args[0] === calendar), (z) => String(z.args[1]));
  if (zone.conflict) return { error: 'conflicting-zone' };
  if (zone.value === undefined) return { error: `calendar-incomplete: no timezone stated for calendar ${calendar}` };
  if (zone.value !== 'utc') return { error: `unsupported-zone: ${zone.value} (v0 supports UTC)` };

  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return { error: 'unparseable-timestamp' };

  let minutes = 0;
  const cursor = new Date(start);
  cursor.setUTCHours(0, 0, 0, 0);
  for (let day = 0; day < MAX_WINDOW_DAYS; day += 1) {
    const dayStart = cursor.getTime();
    if (dayStart > end) break;
    const weekday = WEEKDAYS[cursor.getUTCDay() === 0 ? 6 : cursor.getUTCDay() - 1]!;
    const iso = cursor.toISOString().slice(0, 10);
    const window = hours.get(weekday);
    if (window && !holidays.has(iso)) {
      const openMs = dayStart + window.open * 3_600_000;
      const closeMs = dayStart + window.close * 3_600_000;
      const overlap = Math.min(closeMs, end) - Math.max(openMs, start);
      if (overlap > 0) minutes += Math.round(overlap / 60_000);
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return { minutes };
}

/** The whole verdict path for one ticket. Deterministic: no clock reads, no randomness. */
export function decide(shape: Shape, params: Record<string, string>, claims: Claim[], o: DecideOptions): Proof {
  const c = ctx(claims);
  // ids are canonical lowercase everywhere (the gate canonicalizes claim ids the same way)
  const ticket = (params.ticket ?? '').toLowerCase();
  const facts: Proof['facts'] = [];
  const base = { shape, params, rule: ENGINE_RULE, pack: { id: o.packId, version: o.packVersion }, facts };

  const fail = (reasons: string[], missing: string[]): Proof => ({
    ...base,
    verdict: 'unknown',
    decidedAt: o.decidedAt ?? new Date().toISOString(),
    summary: `Unknown: ${reasons.join('; ')}`,
    unknown: { reasons, missing, spansSearched: o.spansSearched ?? 0 },
  });

  if (!ticket) return fail(['no ticket bound — every parameter must bind to an id'], ['ticket']);

  // case facts, each single-valued per ticket (conflicts decline: two candidates is the namesake lesson)
  const need = (predicate: string): { claim?: Claim; missing?: string; conflict?: boolean } => {
    const rows = c.byTicket(predicate, ticket);
    if (rows.length === 0) return { missing: `${predicate} for ticket ${ticket} (no admitted claim)`.toLowerCase() };
    const distinct = [...new Set(rows.map((r) => String(r.args[1])))];
    if (distinct.length > 1) return { conflict: true };
    facts.push(factOf(rows[0]!));
    return { claim: rows[0] };
  };

  const opened = need('opened');
  if (opened.conflict) return fail(['conflicting opened times for ticket ' + ticket], []);
  if (opened.missing) return fail([`cannot place the start of the clock: ${opened.missing}`], [opened.missing]);
  const response = need('first_response');
  if (response.conflict) return fail(['conflicting first-response times for ticket ' + ticket], []);
  if (!response.claim) {
    return fail(['the ticket has no first response yet — the clock is still running, so the breach cannot be judged'], [response.missing!]);
  }

  const customerOf = need('customer_of');
  if (customerOf.conflict) return fail(['conflicting customer bindings for ticket ' + ticket], []);
  if (customerOf.missing) return fail([customerOf.missing], [customerOf.missing]);
  const customer = String(customerOf.claim!.args[1]);

  const tier = c.claims.filter((x) => x.predicate === 'tier' && x.args[0] === customer);
  const tierSole = sole(tier, (x) => String(x.args[1]));
  if (tierSole.conflict) return fail(['conflicting tiers for customer ' + customer], []);
  if (tierSole.value === undefined) return fail([`tier for customer ${customer} not stated`], [`tier for customer ${customer}`]);
  const tierRow = tier.find((x) => String(x.args[1]) === tierSole.value)!;
  facts.push(factOf(tierRow));

  const priority = need('priority');
  if (priority.conflict) return fail(['conflicting priorities for ticket ' + ticket], []);
  if (priority.missing) return fail([priority.missing], [priority.missing]);

  // pack: the clock for this tier+priority, declining when the pack does not cover it
  const clocks = c.pack('sla_clock').filter(
    (x) => x.args[0] === tierSole.value && x.args[1] === String(priority.claim!.args[1]),
  );
  const clockSole = sole(clocks, (x) => x.args[2] as number);
  if (clockSole.conflict) return fail([`conflicting sla clocks in the pack for ${tierSole.value}/${priority.claim!.args[1]}`], []);
  if (clockSole.value === undefined) {
    const available = c.pack('sla_clock').map((x) => `${x.args[0]}/${x.args[1]}=${x.args[2]}h`).sort();
    return fail(
      [`the pack states no first-response clock for tier ${tierSole.value} priority ${priority.claim!.args[1]} (pack has: ${available.join(', ') || 'nothing'})`],
      [`sla_clock(${tierSole.value}, ${priority.claim!.args[1]})`],
    );
  }
  const allowedMinutes = (clockSole.value as number) * 60;
  facts.push(factOf(clocks[0]!));

  const calendarBinding = c.claims.filter((x) => x.predicate === 'calendar_of' && x.args[0] === customer);
  const calendarSole = sole(calendarBinding, (x) => String(x.args[1]));
  if (calendarSole.conflict) return fail(['conflicting calendar bindings for customer ' + customer], []);
  let calendar: string;
  if (calendarSole.value !== undefined) {
    calendar = String(calendarSole.value);
    facts.push(factOf(calendarBinding[0]!));
  } else {
    const def = sole(c.pack('default_calendar'), (x) => String(x.args[0]));
    if (def.conflict || def.value === undefined) return fail(['no calendar bound to the customer and no pack default'], ['default_calendar']);
    calendar = String(def.value);
    facts.push(factOf(c.pack('default_calendar')[0]!));
  }

  const startedAt = String(opened.claim!.args[1]);
  const respondedAt = String(response.claim.args[1]);
  if (Date.parse(respondedAt) < Date.parse(startedAt)) {
    return fail(['first response precedes the opening of the ticket — the timestamps cannot both be right'], []);
  }

  const walked = businessMinutes(startedAt, respondedAt, calendar, c, facts);
  if (walked.error) return fail([walked.error], []);
  const computation: Record<string, string | number> = {
    calendar,
    allowedMinutes,
    businessMinutes: walked.minutes!,
    breachMinutes: Math.max(0, walked.minutes! - allowedMinutes),
  };

  if (shape === 'sla_met') {
    const met = walked.minutes! <= allowedMinutes;
    return {
      ...base,
      verdict: met ? 'allow' : 'deny',
      decidedAt: o.decidedAt ?? new Date().toISOString(),
      summary: met
        ? `Ticket ${ticket} was answered within the ${tierSole.value}/${priority.claim!.args[1]} clock (${walked.minutes} of ${allowedMinutes} business minutes).`
        : `Ticket ${ticket} breached the ${tierSole.value}/${priority.claim!.args[1]} clock by ${computation.breachMinutes} business minutes.`,
      computation,
    };
  }

  const breach = walked.minutes! - allowedMinutes;
  if (breach <= 0) {
    return {
      ...base,
      verdict: 'deny',
      decidedAt: o.decidedAt ?? new Date().toISOString(),
      summary: `No credit: ticket ${ticket} was answered within the clock (${walked.minutes} of ${allowedMinutes} business minutes).`,
      computation,
    };
  }

  // the credit schedule, straight from pack data
  const bands = c.pack('credit_band');
  const bandRows = [...bands].sort((a, b) => (a.args[0] as number) - (b.args[0] as number));
  const thresholds = new Map<number, number>();
  for (const b of bandRows) {
    const t = b.args[0] as number;
    if (thresholds.has(t) && thresholds.get(t) !== b.args[1]) return fail([`conflicting credit bands in the pack at ${t} minutes`], []);
    thresholds.set(t, b.args[1] as number);
  }
  for (const b of bandRows) facts.push(factOf(b));
  const matched = [...thresholds.entries()].filter(([t]) => t <= breach).sort((a, b) => a[0] - b[0]).pop();
  if (!matched) {
    return {
      ...base,
      verdict: 'deny',
      decidedAt: o.decidedAt ?? new Date().toISOString(),
      summary: `No credit: the ${breach}-minute breach is below the smallest credit band (${bandRows[0]!.args[0]} business minutes).`,
      computation,
    };
  }
  const percent = matched[1];
  computation.creditBandMinutes = matched[0];
  computation.creditPercent = percent;

  if (shape === 'credit_percent') {
    return {
      ...base,
      verdict: 'allow',
      decidedAt: o.decidedAt ?? new Date().toISOString(),
      summary: `Credit of ${percent}% of the monthly fee: ticket ${ticket} breached by ${breach} business minutes.`,
      computation,
    };
  }

  const fees = c.claims.filter((x) => x.predicate === 'monthly_fee' && x.args[0] === customer);
  const feeSole = sole(fees, (x) => x.args[1] as number);
  if (feeSole.conflict) return fail(['conflicting monthly fees for customer ' + customer], []);
  if (feeSole.value === undefined) {
    return fail([`monthly fee for customer ${customer} not stated — the credit percent is proved (${percent}%) but the amount needs the fee`], [`monthly_fee(${customer})`]);
  }
  const fee = feeSole.value as number;
  facts.push(factOf(fees[0]!));
  const amount = Math.round(fee * percent) / 100;
  computation.monthlyFee = fee;
  computation.creditAmount = amount;
  return {
    ...base,
    verdict: 'allow',
    decidedAt: o.decidedAt ?? new Date().toISOString(),
    summary: `Credit of ${percent}% of the ${fee}/mo fee (${amount}): ticket ${ticket} breached by ${breach} business minutes.`,
    computation,
  };
}
