/**
 * The gold corpus for the SLA family: cases whose verdicts are CERTAIN BY CONSTRUCTION.
 * A seeded generator builds case facts, the engine computes gold from those facts directly,
 * and renderers project the same facts into surfaces — a tidy ticket, a chat transcript, and
 * a noisy re-rendering. The stack (gate + engine) must then:
 *   - reproduce the gold verdict on every rendering (verdict accuracy);
 *   - produce the SAME verdict, summary and computation on the noisy rendering
 *     (surface-noise robustness, compared without volatile fields);
 *   - convert a deleted fact into an Unknown with the right reason (information loss),
 *     never a confident answer.
 *
 * The gold engine runs on the base facts, the way the compose benchmark computes labels from
 * the world the documents were rendered from — labels are never written by hand.
 *
 * `--extract` runs the LLM extractor behind the gate instead of the deterministic harvest,
 * measuring end-to-end verdict match and the gate's admit/reject ledger. Without it the run
 * is free and deterministic.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { Rng } from '../compose/rng.js';
import { OpenRouterClient } from '../llm/client.js';
import { admitAll, hashJson, type Claim, type ClaimInput } from '../support/claims.js';
import { decide, type Shape } from '../support/engine.js';
import { claimsFromTicket, extractClaims } from '../support/extract.js';
import { chatSource, ticketSource, type ChatInput, type Source, type TicketInput } from '../support/sources.js';

export interface CaseFact {
  ticket: string;
  customer: string;
  tier: 'enterprise' | 'team' | 'starter';
  priority: 'p1' | 'p2';
  fee: number;
  openedIso: string;
  /** First-response instant; undefined = genuinely still open. */
  responseIso?: string;
  label: string;
}

const TIERS = [
  { tier: 'enterprise', fee: 1200, clocks: { p1: 240, p2: 480 } },
  { tier: 'team', fee: 400, clocks: { p1: 120, p2: 360 } },
  { tier: 'starter', fee: 99, clocks: { p1: 1440, p2: 1440 } },
] as const;

const HOLIDAYS = new Set(['2026-12-25', '2026-01-01']);

/** Walk FORWARD over business windows — the mirror of the engine's walk — so a case can be
 *  aimed exactly at a breach band. */
function addBusinessMinutes(startIso: string, minutes: number): string {
  const start = Date.parse(startIso);
  const cursor = new Date(start);
  cursor.setUTCHours(0, 0, 0, 0);
  let remaining = minutes;
  let at = start;
  for (let day = 0; day < 400; day += 1) {
    const dayStart = cursor.getTime();
    const weekday = cursor.getUTCDay();
    const iso = cursor.toISOString().slice(0, 10);
    if (weekday >= 1 && weekday <= 5 && !HOLIDAYS.has(iso)) {
      const openMs = dayStart + 9 * 3_600_000;
      const closeMs = dayStart + 18 * 3_600_000;
      const pos = Math.max(at, openMs);
      if (pos < closeMs) {
        const available = Math.floor((closeMs - pos) / 60_000);
        if (available >= remaining) return new Date(pos + remaining * 60_000).toISOString();
        remaining -= available;
        at = closeMs;
      } else {
        at = Math.max(at, closeMs);
      }
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  throw new Error('addBusinessMinutes ran off the window');
}

export function generateCases(count: number, seed: number): CaseFact[] {
  const rng = new Rng(seed);
  const offsets: Array<{ label: string; minutes: number }> = [
    { label: 'met', minutes: 0 },
    { label: 'met', minutes: 15 },
    { label: 'band1', minutes: 90 },
    { label: 'band1', minutes: 200 },
    { label: 'band2', minutes: 300 },
    { label: 'band2', minutes: 400 },
    { label: 'band3', minutes: 510 },
    { label: 'band3', minutes: 700 },
  ];
  const cases: CaseFact[] = [];
  for (let i = 0; i < count; i += 1) {
    const t = rng.pick(TIERS);
    const priority = rng.pick(['p1', 'p2'] as const);
    const month = rng.pick([11, 12, 1]);
    const year = month === 1 ? 2027 : 2026;
    const day = rng.int(1, 25);
    const openedIso = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T10:00:00Z`;
    const allowed = t.clocks[priority];
    const offset = rng.pick(offsets);
    const answered = rng.chance(0.9);
    cases.push({
      ticket: `TCK-${10_000 + i}`,
      customer: `cust-${100 + i}`,
      tier: t.tier,
      priority,
      fee: t.fee,
      openedIso,
      responseIso: answered ? addBusinessMinutes(openedIso, allowed + offset.minutes) : undefined,
      label: answered ? offset.label : 'open',
    });
  }
  return cases;
}

/** The base fact base: claims straight from the case facts — what gold is computed from. */
function baseClaimInputs(f: CaseFact): ClaimInput[] {
  const inputs: ClaimInput[] = [
    { scope: 'case', predicate: 'opened', args: [f.ticket, f.openedIso], sourceId: f.ticket, spanId: 'field.opened', trust: 'system' },
    { scope: 'case', predicate: 'priority', args: [f.ticket, f.priority], sourceId: f.ticket, spanId: 'field.priority', trust: 'system' },
    { scope: 'case', predicate: 'customer_of', args: [f.ticket, f.customer], sourceId: f.ticket, spanId: 'field.customer', trust: 'system' },
    { scope: 'case', predicate: 'tier', args: [f.customer, f.tier], sourceId: f.ticket, spanId: 'field.tier', trust: 'system' },
    { scope: 'case', predicate: 'monthly_fee', args: [f.customer, f.fee], sourceId: f.ticket, spanId: 'field.monthly_fee', trust: 'system' },
  ];
  if (f.responseIso !== undefined) {
    inputs.push({ scope: 'case', predicate: 'first_response', args: [f.ticket, f.responseIso], sourceId: f.ticket, spanId: 'comment.r1', trust: 'system' });
  }
  return inputs;
}

function renderTidy(f: CaseFact): { ticket: TicketInput } {
  return {
    ticket: {
      id: f.ticket,
      customer: f.customer,
      fields: {
        opened: f.openedIso,
        priority: f.priority,
        customer: f.customer,
        tier: f.tier,
        monthly_fee: f.fee,
      },
      comments: f.responseIso === undefined ? [] : [{
        actor: 'support-agent',
        spanId: 'comment.r1',
        text: `First agent response logged ${f.responseIso} by the on-call engineer.`,
      }],
    },
  };
}

const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

function prettyDate(iso: string, noisy: boolean): string {
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return iso;
  const [, y, mo, d, h, mi] = m;
  if (noisy) {
    const hour = Number(h);
    const clock = `${hour % 12 === 0 ? 12 : hour % 12}:${mi} ${hour < 12 ? 'AM' : 'PM'}`;
    return `${MONTH_NAMES[Number(mo) - 1]} ${Number(d)}, ${y} at ${clock}`;
  }
  return `${y}-${mo}-${d} ${h}:${mi} UTC`;
}

function renderChat(f: CaseFact, noisy: boolean): ChatInput {
  const rng = new Rng(Array.from(f.ticket).reduce((a, ch) => a + ch.charCodeAt(0), noisy ? 7 : 0));
  const filler = noisy
    ? ['any update here?? still waiting...', 'this is the third time ive had to chase someone', 'fyi our finance team asked about this one too']
    : ['Checking in on this.', 'Could you confirm the status?'];
  const turns: ChatInput['turns'] = [
    { actor: 'requester', at: f.openedIso, text: `${noisy ? 'hi — we just hit a sev issue' : 'We opened a ticket'}. Ticket ${f.ticket}, opened ${prettyDate(f.openedIso, noisy)}.` },
  ];
  if (f.responseIso !== undefined) {
    turns.push({ actor: 'support-agent', text: `First response on ${f.ticket}: ${prettyDate(f.responseIso, noisy)} — ${noisy ? 'sorry for the wait, looking now' : 'we are investigating'}.` });
  } else {
    turns.push({ actor: 'requester', text: filler[0]! });
  }
  turns.push({ actor: 'support-agent', text: rng.pick(filler) });
  return { id: `${f.ticket}-chat`, turns };
}

function caseSources(f: CaseFact, chat: boolean, noisy = false): Source[] {
  const { ticket } = renderTidy(f);
  const sources: Source[] = [ticketSource(ticket)];
  if (chat) sources.push(chatSource(renderChat(f, noisy)));
  return sources;
}

/** Deterministic harvest: claims citing the actual rendered spans (the $0, seeded extractor). */
function harvest(f: CaseFact, sources: Source[], chat: boolean): ClaimInput[] {
  const { ticket } = renderTidy(f);
  const inputs = claimsFromTicket(ticket);
  if (!chat) {
    if (f.responseIso !== undefined) {
      inputs.push({ scope: 'case', predicate: 'first_response', args: [f.ticket, f.responseIso], sourceId: ticket.id, spanId: 'comment.r1', trust: 'system' });
    }
    return inputs;
  }
  const transcript = sources.find((s) => s.kind === 'chat')!;
  const openedSpan = transcript.spans.find((sp) => sp.actor === 'requester');
  if (openedSpan) {
    inputs.push({ scope: 'case', predicate: 'opened', args: [f.ticket, f.openedIso], sourceId: transcript.sourceId, spanId: openedSpan.spanId, trust: 'model' });
  }
  if (f.responseIso !== undefined) {
    const responseSpan = transcript.spans.find((sp) => sp.text.includes('First response on '));
    if (responseSpan) {
      inputs.push({ scope: 'case', predicate: 'first_response', args: [f.ticket, f.responseIso], sourceId: transcript.sourceId, spanId: responseSpan.spanId, trust: 'model' });
    }
  }
  return inputs;
}

export function admitted(f: CaseFact, chat: boolean, noisy = false, drop: 'none' | 'response' | 'tier' = 'none'): { sources: Source[]; claims: Claim[] } {
  const sources = caseSources(f, chat, noisy);
  let inputs = harvest(f, sources, chat);
  if (drop === 'response') inputs = inputs.filter((i) => i.predicate !== 'first_response');
  if (drop === 'tier') inputs = inputs.filter((i) => i.predicate !== 'tier');
  const report = admitAll(inputs, sources);
  return { sources, claims: report.admitted };
}

// ─── the refund family ───────────────────────────────────────────────────────────────────

export interface RefundFact {
  ticket: string;
  customer: string;
  tier: 'standard' | 'pro';
  purchasedIso: string;
  returnIso?: string;
  itemState: 'sealed' | 'opened';
  label: string;
}

const REFUND_TIERS = [
  { tier: 'standard', window: 30 },
  { tier: 'pro', window: 60 },
] as const;

export function generateRefundCases(count: number, seed: number): RefundFact[] {
  const rng = new Rng(seed + 7);
  const cases: RefundFact[] = [];
  for (let i = 0; i < count; i += 1) {
    const t = rng.pick(REFUND_TIERS);
    const month = rng.pick([10, 11, 12]);
    const day = rng.int(1, 25);
    const purchasedIso = `2026-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T12:00:00Z`;
    const offset = rng.pick([-3, 3, 10, 20, 45, 75]);
    const withReturn = rng.chance(0.92);
    const returnIso = withReturn
      ? new Date(Date.parse(purchasedIso) + offset * 86_400_000).toISOString()
      : undefined;
    const itemState = rng.pick(['sealed', 'opened'] as const);
    const days = withReturn ? offset : 0;
    const label = !withReturn ? 'open'
      : offset < 0 ? 'backdated'
      : days > t.window ? 'late'
      : itemState === 'opened' ? 'final-sale'
      : 'eligible';
    cases.push({
      ticket: `RTN-${20_000 + i}`,
      customer: `cust-${500 + i}`,
      tier: t.tier,
      purchasedIso,
      returnIso,
      itemState,
      label,
    });
  }
  return cases;
}

function renderTidyRefund(f: RefundFact): TicketInput {
  return {
    id: f.ticket,
    customer: f.customer,
    fields: {
      purchased: f.purchasedIso,
      return_requested: f.returnIso,
      item_state: f.itemState,
      customer: f.customer,
      tier: f.tier,
    },
  };
}

function renderChatRefund(f: RefundFact, noisy: boolean): ChatInput {
  const turns: ChatInput['turns'] = [
    { actor: 'requester', text: `Ticket ${f.ticket}: ${noisy ? 'bought this a while back, want my money back pls' : 'I need to return this product'}. Purchased ${prettyDate(f.purchasedIso, noisy)}.` },
  ];
  if (f.returnIso !== undefined) {
    turns.push({ actor: 'support-agent', text: `Return on ${f.ticket} received ${prettyDate(f.returnIso, noisy)} — item is ${f.itemState}.` });
  } else {
    turns.push({ actor: 'requester', text: noisy ? 'actually still deciding, keep it open' : 'Still deciding whether to return it.' });
  }
  return { id: `${f.ticket}-chat`, turns };
}

function refundSources(f: RefundFact, chat: boolean, noisy = false): Source[] {
  const ticket = renderTidyRefund(f);
  const sources: Source[] = [ticketSource(ticket)];
  if (chat) sources.push(chatSource(renderChatRefund(f, noisy)));
  return sources;
}

export function refundAdmitted(f: RefundFact, chat: boolean, noisy = false, drop: 'none' | 'state' = 'none'): { sources: Source[]; claims: Claim[] } {
  const sources = refundSources(f, chat, noisy);
  let inputs = claimsFromTicket(renderTidyRefund(f));
  if (chat) {
    const transcript = sources.find((s) => s.kind === 'chat')!;
    const first = transcript.spans[0]!;
    inputs.push({ scope: 'case', predicate: 'purchased', args: [f.ticket, f.purchasedIso], sourceId: transcript.sourceId, spanId: first.spanId, trust: 'model' });
    if (f.returnIso !== undefined) {
      const span = transcript.spans.find((sp) => sp.text.includes('Return on '))!;
      inputs.push({ scope: 'case', predicate: 'return_requested', args: [f.ticket, f.returnIso], sourceId: transcript.sourceId, spanId: span.spanId, trust: 'model' });
    }
  }
  if (drop === 'state') inputs = inputs.filter((i) => i.predicate !== 'item_state');
  const report = admitAll(inputs, sources);
  return { sources, claims: report.admitted };
}

interface ArmResult {
  arm: string;
  cases: number;
  match: number;
  failures: Array<{ ticket: string; expected: string; got: string }>;
  gate: { admitted: number; rejected: number };
}

export const DECIDED_AT = '2026-09-24T00:00:00Z';

function runArm<T extends { ticket: string }>(
  name: string,
  cases: T[],
  pack: { id: string; version: string; claims: Claim[] },
  shape: Shape,
  build: (f: T) => Claim[],
  expect: 'match-gold' | 'unknown',
  goldClaims: (f: T) => Claim[],
  goldSourceList: (f: T) => Source[],
): ArmResult {
  const result: ArmResult = { arm: name, cases: cases.length, match: 0, failures: [], gate: { admitted: 0, rejected: 0 } };
  for (const f of cases) {
    const gold = decide(shape, { ticket: f.ticket }, [...pack.claims, ...goldClaims(f)], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
    const claims = build(f);
    const proof = decide(shape, { ticket: f.ticket }, [...pack.claims, ...claims], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT, spansSearched: 8 });
    result.gate.admitted += claims.length;
    const ok = expect === 'unknown'
      ? proof.verdict === 'unknown'
      : proof.verdict === gold.verdict && proof.summary === gold.summary;
    if (ok) result.match += 1;
    else result.failures.push({ ticket: f.ticket, expected: expect === 'unknown' ? 'unknown' : `${gold.verdict}: ${gold.summary}`, got: `${proof.verdict}: ${proof.summary}` });
  }
  return result;
}

async function main() {
  const argv = process.argv.slice(2);
  const at = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback;
  };
  const count = Number(at('--cases', '40'));
  const seed = Number(at('--seed', '20260924'));
  const family = at('--family', 'sla') as 'sla' | 'refund';
  const useExtract = argv.includes('--extract');
  const out = at('--out', 'results/support-gold.json');

  const { loadPack } = await import('../support/pack.js');
  const pack = family === 'refund'
    ? loadPack('benchmarks/support/pack-refund.json')
    : loadPack('benchmarks/support/pack.json');
  const shape = family === 'refund' ? 'refund_eligible' : 'sla_credits';
  const cases: Array<CaseFact | RefundFact> = family === 'refund' ? generateRefundCases(count, seed) : generateCases(count, seed);
  const answered = cases.filter((f) => (family === 'refund' ? (f as RefundFact).returnIso !== undefined : (f as CaseFact).responseIso !== undefined));

  const slaGoldClaims = (f: CaseFact) => admitAll(baseClaimInputs(f), caseSources(f, false)).admitted;
  const slaGoldSources = (f: CaseFact) => caseSources(f, false);
  const refundGoldClaims = (f: RefundFact) => admitAll(claimsFromTicket(renderTidyRefund(f)), refundSources(f, false)).admitted;
  const refundGoldSources = (f: RefundFact) => refundSources(f, false);

  const arms: ArmResult[] = family === 'refund'
    ? [
        runArm('tidy-ticket', cases as RefundFact[], pack, shape, (f) => refundAdmitted(f, false).claims, 'match-gold', refundGoldClaims, refundGoldSources),
        runArm('chat-transcript', cases as RefundFact[], pack, shape, (f) => refundAdmitted(f, true).claims, 'match-gold', refundGoldClaims, refundGoldSources),
        runArm('chat-noisy', cases as RefundFact[], pack, shape, (f) => refundAdmitted(f, true, true).claims, 'match-gold', refundGoldClaims, refundGoldSources),
        runArm('info-loss-state', cases as RefundFact[], pack, shape, (f) => refundAdmitted(f, false, false, 'state').claims, 'unknown', refundGoldClaims, refundGoldSources),
      ]
    : [
        runArm('tidy-ticket', cases as CaseFact[], pack, shape, (f) => admitted(f, false).claims, 'match-gold', slaGoldClaims, slaGoldSources),
        runArm('chat-transcript', cases as CaseFact[], pack, shape, (f) => admitted(f, true).claims, 'match-gold', slaGoldClaims, slaGoldSources),
        runArm('chat-noisy', cases as CaseFact[], pack, shape, (f) => admitted(f, true, true).claims, 'match-gold', slaGoldClaims, slaGoldSources),
        // information loss: the deleted fact must turn a proved verdict into an explained Unknown
        runArm('info-loss-response', answered as CaseFact[], pack, shape, (f) => admitted(f, false, false, 'response').claims, 'unknown', slaGoldClaims, slaGoldSources),
        runArm('info-loss-tier', cases as CaseFact[], pack, shape, (f) => admitted(f, false, false, 'tier').claims, 'unknown', slaGoldClaims, slaGoldSources),
      ];

  // noise robustness as proof-equality (minus volatile fields) between the two chat renderings
  let noiseFlips = 0;
  for (const f of cases) {
    const clean = decide(shape, { ticket: f.ticket }, [...pack.claims, ...(family === 'refund' ? refundAdmitted(f as RefundFact, true).claims : admitted(f as CaseFact, true).claims)], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
    const noisy = decide(shape, { ticket: f.ticket }, [...pack.claims, ...(family === 'refund' ? refundAdmitted(f as RefundFact, true, true).claims : admitted(f as CaseFact, true, true).claims)], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
    const key = (p: typeof clean) => hashJson([p.verdict, p.summary, p.computation]);
    if (key(clean) !== key(noisy)) noiseFlips += 1;
  }

  let extractNote = 'harvest (deterministic, $0)';
  if (useExtract && family === 'sla') {
    const useOllama = argv.includes('--ollama');
    const modelName = at('--model', useOllama ? 'llama3.1:8b' : 'deepseek/deepseek-chat');
    const client = useOllama
      ? new OpenRouterClient({ model: modelName, apiKey: 'ollama', baseUrl: at('--base-url', 'http://127.0.0.1:11434/v1'), temperature: 0 })
      : new OpenRouterClient({ model: modelName, apiKey: process.env.OPENROUTER_API_KEY ?? process.env.LLM_API_KEY ?? '', baseUrl: process.env.LLM_BASE_URL ?? 'https://openrouter.ai/api/v1', temperature: 0 });
    let match = 0;
    let admittedTotal = 0;
    let rejectedTotal = 0;
    let failedSpans = 0;
    let lastError: string | undefined;
    const rejections: Array<{ predicate: string; reason: string }> = [];
    for (const f of cases as CaseFact[]) {
      const sources = caseSources(f, true);
      const extraction = await extractClaims(client, sources);
      admittedTotal += extraction.admittedCount;
      rejectedTotal += extraction.rejectedCount;
      failedSpans += extraction.failedSpans;
      if (extraction.lastError) lastError = extraction.lastError;
      rejections.push(...extraction.rejections);
      const goldSources = caseSources(f, false);
      const gold = decide('sla_credits', { ticket: f.ticket }, [...pack.claims, ...admitAll(baseClaimInputs(f), goldSources).admitted], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
      const fieldClaims = admitAll(claimsFromTicket(renderTidy(f).ticket), sources).admitted;
      const proof = decide('sla_credits', { ticket: f.ticket }, [...pack.claims, ...fieldClaims, ...extraction.admitted], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
      if (proof.verdict === gold.verdict) match += 1;
    }
    arms.push({ arm: 'llm-extract', cases: cases.length, match, failures: [], gate: { admitted: admittedTotal, rejected: rejectedTotal } });
    extractNote = failedSpans > 0
      ? `${modelName} behind the gate: EXTRACTION FAILED on ${failedSpans} spans (last error: ${lastError ?? 'unknown'}) — no live sensor verdict`
      : `${modelName} behind the gate: ${rejectedTotal} of ${admittedTotal + rejectedTotal} candidate claims rejected`;
  } else if (useExtract) {
    extractNote = '--extract is wired for the sla family only; the refund arms ran on the deterministic harvest';
  }

  const report = {
    ranAt: new Date().toISOString(),
    seed,
    shape,
    pack: { id: pack.id, version: pack.version, claims: pack.claims.length, unconsumedLines: pack.unconsumedLines },
    noiseFlips,
    extractNote,
    arms,
  };
  mkdirSync('results', { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 1)}\n`);
  for (const arm of arms) {
    console.log(`${arm.arm.padEnd(20)} ${String(arm.match)}/${arm.cases} ${'ok'.padEnd(12)} gate ${arm.gate.admitted} admitted / ${arm.gate.rejected} rejected`);
    for (const m of arm.failures.slice(0, 3)) console.log(`   FAIL ${m.ticket}: expected ${m.expected} | got ${m.got}`);
  }
  console.log(`noise flips between clean and noisy chat renderings: ${noiseFlips}`);
  console.log(`pack coverage review: ${pack.unconsumedLines.length} unconsumed SOP line(s) flagged`);
  console.log(`wrote ${out} — extraction: ${extractNote}`);
}

if (process.argv[1]?.endsWith('run-support-gold.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
