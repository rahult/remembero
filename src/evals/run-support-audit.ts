/**
 * The audit run: Phase 1's money report, on synthetic history. A seeded generator produces a
 * support team's past quarter — case facts AND what the agents actually did (granted, missed,
 * over-granted, denied) with realistic error rates — then the engine decides every case from
 * the claims alone. Comparing the two yields the artifact a lighthouse buyer cares about:
 *
 *  - RECOVERY: credits that were due (engine allow) but never issued — money owed out.
 *  - LEAKAGE: credits issued (action granted) that the engine denies — money wrongly out.
 *  - AGREEMENT: where engine and humans concur, the baseline for trust.
 *  - UNKNOWNS: cases the engine declines, each with its reason — the honest residue.
 *
 * On synthetic history the error rates are assumptions, not measurements; the point is the
 * report FORMAT and the decision path end to end. Point the same runner at a real export and
 * it becomes the Phase 1 audit for real.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { Rng } from '../compose/rng.js';
import { admitAll, hashJson, type Claim } from '../support/claims.js';
import { decide, type Proof } from '../support/engine.js';
import { loadPack } from '../support/pack.js';
import { admitted, DECIDED_AT, generateCases, type CaseFact } from './run-support-gold.js';

type Action =
  | { kind: 'granted'; amount: number }
  | { kind: 'none' }
  | { kind: 'escalated' };

/** What the (synthetic) support team actually did, with realistic human error rates, given the
 *  case's true verdict. The engine is the labeler — no band math is duplicated here. */
function recordAction(gold: Proof, f: CaseFact, rng: Rng): Action {
  if (gold.verdict === 'unknown') return { kind: 'escalated' };
  if (gold.verdict === 'deny') {
    // no credit due: humans mostly deny, but one in ten grants "goodwill"
    return rng.chance(0.1) ? { kind: 'granted', amount: Math.round(f.fee * 0.1) } : { kind: 'none' };
  }
  const due = Number(gold.computation?.creditAmount ?? 0);
  const roll = rng.next();
  if (roll < 0.12) return { kind: 'none' }; // missed the credit entirely
  if (roll < 0.18) return { kind: 'granted', amount: due * 2 }; // over-granted
  return { kind: 'granted', amount: due };
}

async function main() {
  const argv = process.argv.slice(2);
  const at = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback;
  };
  const count = Number(at('--cases', '200'));
  const seed = Number(at('--seed', '20260924'));
  const out = at('--out', 'results/support-audit.json');

  const pack = loadPack('benchmarks/support/pack.json');
  const rng = new Rng(seed + 1);
  const cases = generateCases(count, seed);

  const rows: Array<{ ticket: string; verdict: string; summary: string; action: string; category: string; delta: number }> = [];
  let recovery = 0;
  let leakage = 0;
  let agreedAllow = 0;
  let agreedDeny = 0;
  let escalated = 0;
  let amountMismatches = 0;

  for (const f of cases) {
    const { claims } = admitted(f, false);
    const proof: Proof = decide('sla_credits', { ticket: f.ticket }, [...pack.claims, ...claims], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
    const action = recordAction(proof, f, rng);
    const granted = action.kind === 'granted' ? action.amount : 0;
    const due = proof.verdict === 'allow' ? Number(proof.computation?.creditAmount ?? 0) : 0;

    let category: string;
    let delta = 0;
    if (proof.verdict === 'unknown') {
      category = action.kind === 'escalated' ? 'agree-escalate' : 'unknown-vs-action';
      escalated += 1;
    } else if (proof.verdict === 'allow' && action.kind === 'granted') {
      delta = Math.abs(due - granted);
      if (delta === 0) { category = 'agree-allow'; agreedAllow += 1; }
      else { category = 'amount-mismatch'; amountMismatches += 1; }
    } else if (proof.verdict === 'allow' && action.kind !== 'granted') {
      category = 'RECOVERY'; recovery += due;
    } else if (proof.verdict === 'deny' && action.kind === 'granted') {
      category = 'LEAKAGE'; leakage += granted;
    } else if (proof.verdict === 'deny' && action.kind === 'none') {
      category = 'agree-deny'; agreedDeny += 1;
    } else {
      category = 'other'; // engine allow vs escalated, etc.
    }
    rows.push({ ticket: f.ticket, verdict: proof.verdict, summary: proof.summary, action: action.kind === 'granted' ? `granted $${action.amount}` : action.kind, category, delta });
  }

  const categories = new Map<string, number>();
  for (const r of rows) categories.set(r.category, (categories.get(r.category) ?? 0) + 1);
  const report = {
    ranAt: new Date().toISOString(),
    seed,
    cases: count,
    pack: { id: pack.id, version: pack.version },
    money: {
      recoveryUsd: recovery,
      leakageUsd: leakage,
      note: 'synthetic history: error rates are assumptions (12% missed credits, 6% over-granted, 10% goodwill grants); the report format is the deliverable — real exports drop in unchanged',
    },
    agreement: { allow: agreedAllow, deny: agreedDeny, escalated, amountMismatches },
    categories: [...categories.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ category: k, cases: v })),
    rows,
  };
  mkdirSync('results', { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 1)}\n`);
  console.log(`audit of ${count} synthetic historical cases (pack ${pack.id}@${pack.version})`);
  for (const c of report.categories) console.log(`   ${String(c.cases).padStart(4)}  ${c.category}`);
  console.log(`RECOVERY (due but never issued): $${recovery}`);
  console.log(`LEAKAGE (issued but not due):    $${leakage}`);
  console.log(`wrote ${out}`);
}

if (process.argv[1]?.endsWith('run-support-audit.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
