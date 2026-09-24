import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { admit, admitAll, hashJson, normalizeText, parseDateArg, type ClaimInput } from '../src/support/claims.js';
import { decide } from '../src/support/engine.js';
import { loadPack } from '../src/support/pack.js';
import { plan } from '../src/support/planner.js';
import { chatSource, ticketSource, type Source } from '../src/support/sources.js';
import { VerdictStore } from '../src/support/store.js';
import { admitted, DECIDED_AT, generateCases } from '../src/evals/run-support-gold.js';

const PACK = loadPack('benchmarks/support/pack.json');
const OPTS = { packId: PACK.id, packVersion: PACK.version, decidedAt: DECIDED_AT };

function caseSources(responseIso = '2026-12-29T10:00:00Z', extraTurn?: string): Source[] {
  const clock = responseIso.slice(0, 16).replace('T', ' ');
  const turns: Array<{ actor: string; text: string }> = [
    { actor: 'requester', text: 'Our API is down. Ticket TCK-1042 opened 2026-12-24 16:30 UTC.' },
    { actor: 'support-agent', text: `First response on TCK-1042: ${clock} UTC — we shipped a fix.` },
  ];
  if (extraTurn) turns.push({ actor: 'support-agent', text: extraTurn });
  return [
    ticketSource({
      id: 'TCK-1042',
      customer: 'cust-acme',
      fields: {
        opened: '2026-12-24T16:30:00Z',
        priority: 'p1',
        customer: 'cust-acme',
        tier: 'enterprise',
        monthly_fee: 1200,
      },
      comments: [{ actor: 'support-agent', spanId: 'comment.r1', text: `First agent response logged ${responseIso} by the on-call engineer.` }],
    }),
    chatSource({ id: 'TCK-1042-chat', turns }),
  ];
}

function input(predicate: string, args: (string | number)[], spanId: string, sourceId = 'TCK-1042'): ClaimInput {
  return { scope: 'case', predicate, args, sourceId, spanId, trust: 'model' };
}

function caseInputs(responseIso = '2026-12-29T10:00:00Z'): ClaimInput[] {
  return [
    input('opened', ['TCK-1042', '2026-12-24T16:30:00Z'], 'field.opened'),
    input('priority', ['TCK-1042', 'p1'], 'field.priority'),
    input('customer_of', ['TCK-1042', 'cust-acme'], 'field.customer'),
    input('tier', ['cust-acme', 'enterprise'], 'field.tier'),
    input('monthly_fee', ['cust-acme', 1200], 'field.monthly_fee'),
    input('first_response', ['TCK-1042', responseIso], 'comment.r1'),
  ];
}

function runCase(responseIso = '2026-12-29T10:00:00Z', extra: ClaimInput[] = [], extraTurn?: string): ReturnType<typeof decide> {
  const sources = caseSources(responseIso, extraTurn);
  const { admitted: claims } = admitAll([...caseInputs(responseIso), ...extra], sources);
  return decide('sla_credits', { ticket: 'TCK-1042' }, [...PACK.claims, ...claims], OPTS);
}

afterAll(() => {
  rmSync(join(tmpdir(), 'support-store-test'), { recursive: true, force: true });
});

describe('admission gate', () => {
  const sources = caseSources();
  const spans = sources.flatMap((s) => s.spans);
  const span = (spanId: string) => spans.find((sp) => sp.spanId === spanId)!;

  it('admits a claim whose every argument is on the cited span', () => {
    const result = admit(input('opened', ['TCK-1042', '2026-12-24T16:30:00Z'], 'field.opened'), span('field.opened'));
    expect(result.ok).toBe(true);
  });

  it('admits a date stated in another surface form (named month)', () => {
    const result = admit(input('first_response', ['TCK-1042', '2026-12-29T10:00:00Z'], 'turn-002', 'TCK-1042-chat'), span('turn-002'));
    expect(result.ok).toBe(true);
  });

  it('rejects an argument not stated on the span', () => {
    const result = admit(input('first_response', ['TCK-1042', '2026-12-29T10:00:00Z'], 'field.opened'), span('field.opened'));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain('not stated on span');
  });

  it('rejects a fact assembled from two spans', () => {
    // the date lives on the comment; the priority word lives on a field — one span cannot state both
    const result = admit(input('priority', ['TCK-1042', 'p1'], 'comment.r1'), span('comment.r1'));
    expect(result.ok).toBe(false);
  });

  it('rejects unknown predicates and wrong arities', () => {
    expect(admit(input('verdict', ['TCK-1042', 'allow'], 'field.opened'), span('field.opened')).ok).toBe(false);
    expect(admit(input('opened', ['TCK-1042'], 'field.opened'), span('field.opened')).ok).toBe(false);
  });

  it('rejects unparseable dates and non-numeric money', () => {
    expect(admit(input('opened', ['TCK-1042', 'sometime last week'], 'field.opened'), span('field.opened')).ok).toBe(false);
    expect(admit(input('monthly_fee', ['cust-acme', 'a lot'], 'field.monthly_fee'), span('field.monthly_fee')).ok).toBe(false);
  });

  it('keeps rejections with reasons in admitAll', () => {
    const report = admitAll([...caseInputs(), input('tier', ['cust-acme', 'starter'], 'field.tier')], sources);
    expect(report.admitted).toHaveLength(6);
    expect(report.rejected).toHaveLength(1);
  });

  it('canonicalizes dates so the engine sees ISO', () => {
    expect(parseDateArg('December 29, 2026')).toBe('2026-12-29');
    expect(parseDateArg('12/29/2026')).toBe('2026-12-29');
    expect(normalizeText('Ticket TCK-1042, opened 2026-12-24')).toContain('tck-1042');
  });

  it('canonicalizes the datetime forms local models actually emit', () => {
    expect(parseDateArg('2026-12-29 10:00 UTC')).toBe('2026-12-29T10:00:00Z');
    expect(parseDateArg('December 29, 2026 at 11:30 AM')).toBe('2026-12-29T11:30:00Z');
    expect(parseDateArg('December 29, 2026 at 2:05 PM')).toBe('2026-12-29T14:05:00Z');
  });

  it('accepts a sensor that answers with a JSON array (llama house style), gated as always', async () => {
    const { extractClaims } = await import('../src/support/extract.js');
    const fake = {
      completeWithUsage: async () => ({
        content: '[\n  {"predicate": "first_response", "args": ["TCK-1042", "2026-12-29 10:00 UTC"], "quote": "First response on TCK-1042: 2026-12-29 10:00 UTC"},\n  {"predicate": "priority", "args": ["TCK-1042", "p2"], "quote": "priority: p1"}\n]',
      }),
    };
    const report = await extractClaims(fake as never, caseSources());
    // the same reply hits every span; only the spans that actually state the fact admit it
    expect(report.admitted.filter((c) => c.predicate === 'first_response').length).toBeGreaterThanOrEqual(1);
    // every p2 claim cites a span that says p1 or nothing: the gate rejects each one
    expect(report.rejectedCount).toBeGreaterThanOrEqual(1);
    expect(report.rejections.every((r) => r.reason.includes('not stated on span') || r.reason.includes('unknown predicate'))).toBe(true);
  });
});

describe('engine — the worked example', () => {
  // opened Thu 2026-12-24 16:30 UTC; response Tue 2026-12-29 10:00 UTC; Christmas Friday is a holiday.
  // business minutes: Thu 90 + Mon 540 + Tue 60 = 690; allowed 240; breach 450 -> band 240 -> 25% of 1200 = 300.
  it('computes the credit over the business calendar, holidays and weekends included', () => {
    const proof = runCase();
    expect(proof.verdict).toBe('allow');
    expect(proof.computation).toMatchObject({ businessMinutes: 690, allowedMinutes: 240, breachMinutes: 450, creditPercent: 25, creditAmount: 300 });
  });

  it('is deterministic: same claims and pack, same decision content', () => {
    const a = runCase();
    const b = runCase();
    const key = (p: typeof a) => hashJson([p.verdict, p.summary, p.computation, p.facts.map((f) => [f.predicate, f.args])]);
    expect(key(a)).toBe(key(b));
  });

  it('denies when the clock was met and when the breach is below the smallest band', () => {
    // 30 business minutes after opening is well within the 240-minute clock
    const within = runCase('2026-12-24T17:00:00Z');
    expect(within.verdict).toBe('deny');
    expect(within.summary).toContain('within the clock');
    // exactly at the clock: Thu 90 + Mon 150 = 240 -> breach 0 -> deny
    const boundary = runCase('2026-12-28T11:30:00Z');
    expect(boundary.verdict).toBe('deny');
    // breach of 30 business minutes (Thu 90 + Mon 180 = 270) is below the smallest band (60) -> deny
    const tinyBreach = runCase('2026-12-28T12:00:00Z');
    expect(tinyBreach.verdict).toBe('deny');
    expect(tinyBreach.summary).toContain('below the smallest credit band');
  });

  it('says Unknown, with the reason, when the response is missing', () => {
    const sources = caseSources();
    const { admitted: claims } = admitAll(caseInputs().filter((i) => i.predicate !== 'first_response'), sources);
    const proof = decide('sla_credits', { ticket: 'TCK-1042' }, [...PACK.claims, ...claims], OPTS);
    expect(proof.verdict).toBe('unknown');
    expect(proof.unknown?.reasons[0]).toContain('no first response yet');
  });

  it('says Unknown on conflicting facts, never guessing', () => {
    const extra = input('first_response', ['TCK-1042', '2026-12-26T09:00:00Z'], 'turn-003', 'TCK-1042-chat');
    const proof = runCase('2026-12-29T10:00:00Z', [extra], 'An earlier status update said the first response on TCK-1042 happened 2026-12-26 09:00 UTC.');
    expect(proof.verdict).toBe('unknown');
    expect(proof.unknown?.reasons[0]).toContain('conflicting');
  });

  it('says Unknown when the response precedes the opening', () => {
    const proof = runCase('2026-12-23T09:00:00Z');
    expect(proof.verdict).toBe('unknown');
    expect(proof.unknown?.reasons[0]).toContain('precedes');
  });

  it('says Unknown when the pack has no clock for the tier — and says what the pack has', () => {
    // the SOURCE states a tier the pack does not cover; the gate admits it because the span says it
    const sources = [ticketSource({
      id: 'TCK-1042',
      customer: 'cust-acme',
      fields: { opened: '2026-12-24T16:30:00Z', priority: 'p1', customer: 'cust-acme', tier: 'platinum', monthly_fee: 1200 },
      comments: [{ actor: 'support-agent', spanId: 'comment.r1', text: 'First agent response logged 2026-12-29T10:00:00Z by the on-call engineer.' }],
    })];
    const inputs = caseInputs().map((i) => (i.predicate === 'tier' ? { ...i, args: ['cust-acme', 'platinum'] } : i));
    const { admitted: claims } = admitAll(inputs, sources);
    const proof = decide('sla_credits', { ticket: 'TCK-1042' }, [...PACK.claims, ...claims], OPTS);
    expect(proof.verdict).toBe('unknown');
    expect(proof.unknown?.reasons[0]).toContain('platinum');
    expect(proof.unknown?.reasons[0]).toContain('pack has');
  });

  it('binds by id: an unbound ticket is rejected by the caller, not guessed', () => {
    const p = plan('is a credit due for the customer who shouted at us');
    expect(p.ok).toBe(false);
  });
});

describe('planner', () => {
  it('maps a credit question with an id to sla_credits, canonicalizing the id', () => {
    const p = plan('is a credit due for TCK-1042?');
    expect(p).toEqual({ ok: true, shape: 'sla_credits', params: { ticket: 'tck-1042' } });
  });

  it('maps an sla-met question and rejects a shapeless one', () => {
    expect(plan('was the sla breached on tck-7007')).toMatchObject({ ok: true, shape: 'sla_met' });
    expect(plan('what is the meaning of life')).toMatchObject({ ok: false });
  });
});

describe('pack', () => {
  it('loads hand-gated claims only when every one passes the gate', () => {
    expect(PACK.claims.length).toBeGreaterThanOrEqual(17);
    expect(PACK.claims.every((c) => c.scope === 'pack')).toBe(true);
  });

  it('flags SOP lines the pack does not consume — the unmodeled-clause review', () => {
    const flagged = PACK.unconsumedLines.map((l) => l.text).join('\n');
    expect(flagged).toContain('follow-the-sun');
  });

  it('refuses to load a tampered pack whose claim is not grounded', () => {
    const dir = mkdtempSync(join(tmpdir(), 'support-store-test'));
    const sopPath = join(dir, 'sop.md');
    writeFileSync(sopPath, readFileSync('benchmarks/support/demo-sla.sop.md', 'utf8').replace('within 4 business hours', 'within 6 business hours'));
    const packPath = join(dir, 'pack.json');
    writeFileSync(packPath, JSON.stringify({ ...JSON.parse(readFileSync('benchmarks/support/pack.json', 'utf8')), sopPath }));
    // the SOP no longer states the parameter, so the claim refuses to bind — either refusal is a refusal
    expect(() => loadPack(packPath)).toThrow(/admission gate|not in the SOP/);
  });
});

describe('verdict store', () => {
  it('appends only, sequences, and hashes decisions stably', () => {
    const dir = mkdtempSync(join(tmpdir(), 'support-store-test'));
    const store = new VerdictStore(join(dir, 'verdicts.jsonl'));
    const proof = runCase();
    const r1 = store.append(proof, { id: PACK.id, version: PACK.version, contentHash: PACK.contentHash }, '2026-09-24T01:00:00Z');
    const r2 = store.append(runCase(), { id: PACK.id, version: PACK.version, contentHash: PACK.contentHash }, '2026-09-24T02:00:00Z');
    expect(r1.seq).toBe(1);
    expect(r2.seq).toBe(2);
    expect(r1.decisionHash).toBe(r2.decisionHash);
    const lines = readFileSync(join(dir, 'verdicts.jsonl'), 'utf8').trim().split('\n');
    expect(lines).toHaveLength(2);
    expect(lines[0]!.startsWith('{"seq":1')).toBe(true);
  });
});

describe('gold corpus — certain labels, three surfaces', () => {
  const cases = generateCases(24, 20260924);

  it('reproduces the gold verdict on the tidy ticket and both chat renderings', () => {
    for (const arm of ['tidy', 'chat', 'noisy'] as const) {
      for (const f of cases) {
        const { claims } = admitted(f, arm !== 'tidy', arm === 'noisy');
        const proof = decide('sla_credits', { ticket: f.ticket }, [...PACK.claims, ...claims], OPTS);
        const goldSources = [ticketSource({
          id: f.ticket,
          customer: f.customer,
          fields: { opened: f.openedIso, priority: f.priority, customer: f.customer, tier: f.tier, monthly_fee: f.fee },
          comments: f.responseIso === undefined ? [] : [{ actor: 'a', spanId: 'comment.r1', text: `First agent response logged ${f.responseIso}.` }],
        })];
        const gold = admitAll(f.responseIso === undefined ? [] : [{ scope: 'case', predicate: 'first_response', args: [f.ticket, f.responseIso], sourceId: f.ticket, spanId: 'comment.r1', trust: 'system' }], goldSources);
        const base = [
          { scope: 'case', predicate: 'opened', args: [f.ticket, f.openedIso], sourceId: f.ticket, spanId: 'field.opened', trust: 'system' },
          { scope: 'case', predicate: 'priority', args: [f.ticket, f.priority], sourceId: f.ticket, spanId: 'field.priority', trust: 'system' },
          { scope: 'case', predicate: 'customer_of', args: [f.ticket, f.customer], sourceId: f.ticket, spanId: 'field.customer', trust: 'system' },
          { scope: 'case', predicate: 'tier', args: [f.customer, f.tier], sourceId: f.ticket, spanId: 'field.tier', trust: 'system' },
          { scope: 'case', predicate: 'monthly_fee', args: [f.customer, f.fee], sourceId: f.ticket, spanId: 'field.monthly_fee', trust: 'system' },
        ] as ClaimInput[];
        const goldProof = decide('sla_credits', { ticket: f.ticket }, [...PACK.claims, ...gold.admitted, ...admitAll(base, goldSources).admitted], OPTS);
        expect(`${proof.verdict}: ${proof.summary}`, `${arm} ${f.ticket}`).toBe(`${goldProof.verdict}: ${goldProof.summary}`);
      }
    }
  });

  it('converts the deleted response fact into Unknown, never a confident answer', () => {
    for (const f of cases.filter((c) => c.responseIso !== undefined)) {
      const { claims } = admitted(f, false, false, 'response');
      const proof = decide('sla_credits', { ticket: f.ticket }, [...PACK.claims, ...claims], OPTS);
      expect(proof.verdict, f.ticket).toBe('unknown');
    }
  });

  it('converts the deleted tier fact into Unknown', () => {
    for (const f of cases) {
      const { claims } = admitted(f, false, false, 'tier');
      const proof = decide('sla_credits', { ticket: f.ticket }, [...PACK.claims, ...claims], OPTS);
      expect(proof.verdict, f.ticket).toBe('unknown');
    }
  });
});
