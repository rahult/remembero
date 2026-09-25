/**
 * Phase 4: the end-to-end loop, minus the pull that needs a real help desk.
 *
 *   npm run support:pipeline -- --export <export.jsonl> [--family sla|refund]
 *
 * The export is one JSON object per line — the adapter seam a real CSV/Zendesk/Intercom
 * converter targets:
 *   {"ticket": {...}, "chat": {...}?, "recorded": {"kind": "granted"|"none"|"escalated", "amount"?}?}
 *
 * The pipeline decides every case (plan -> gate -> engine), appends every verdict to the
 * append-only store, and writes the weekly disputes report: RECOVERY (due but not granted),
 * LEAKAGE (granted but not due), agreement counts, and the Unknowns with their reasons.
 * `--synthesize N` deterministically generates the export instead, so the whole loop runs
 * today; a real export replaces it without touching this code.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { Rng } from '../compose/rng.js';
import { admitAll } from '../support/claims.js';
import { decide, type Proof } from '../support/engine.js';
import { claimsFromTicket } from '../support/extract.js';
import { loadPack } from '../support/pack.js';
import { chatSource, ticketSource, type ChatInput, type Source, type TicketInput } from '../support/sources.js';
import { VerdictStore } from '../support/store.js';
import { DECIDED_AT, generateCases, generateRefundCases, type CaseFact, type RefundFact } from './run-support-gold.js';

interface ExportRow {
  ticket: TicketInput;
  chat?: ChatInput;
  recorded?: { kind: 'granted' | 'none' | 'escalated'; amount?: number };
}

interface DisputeRow {
  ticket: string;
  category: string;
  verdict: string;
  summary: string;
  recorded: string;
  amount?: number;
}

export interface PipelineReport {
  ranAt: string;
  family: string;
  cases: number;
  decided: number;
  rejected: number;
  money: { recoveryUsd: number; leakageUsd: number };
  agreement: Record<string, number>;
  disputes: DisputeRow[];
  unknownReasons: Record<string, number>;
}

export function runPipeline(rows: ExportRow[], family: 'sla' | 'refund', packPath: string, storePath: string, decidedAt = new Date().toISOString()): PipelineReport {
  const pack = loadPack(packPath);
  const shape = family === 'refund' ? 'refund_eligible' : 'sla_credits';
  const store = new VerdictStore(storePath);

  const report: PipelineReport = {
    ranAt: decidedAt,
    family,
    cases: rows.length,
    decided: 0,
    rejected: 0,
    money: { recoveryUsd: 0, leakageUsd: 0 },
    agreement: {},
    disputes: [],
    unknownReasons: {},
  };

  for (const row of rows) {
    const sources: Source[] = [ticketSource(row.ticket)];
    if (row.chat) sources.push(chatSource(row.chat));
    const p = { shape: shape as 'refund_eligible' | 'sla_credits', params: { ticket: String(row.ticket.id).toLowerCase() } };
    const { admitted: claims } = admitAll(claimsFromTicket(row.ticket), sources);
    const proof = decide(p.shape, p.params, [...pack.claims, ...claims], {
      packId: pack.id,
      packVersion: pack.version,
      decidedAt,
      spansSearched: sources.reduce((n, s) => n + s.spans.length, 0),
    });
    store.append(proof, { id: pack.id, version: pack.version, contentHash: pack.contentHash }, decidedAt);

    const recorded = row.recorded?.kind ?? 'unrecorded';
    const granted = row.recorded?.kind === 'granted' ? row.recorded.amount ?? 0 : 0;
    const due = proof.verdict === 'allow' ? Number(proof.computation?.creditAmount ?? 0) : 0;

    let category: string;
    if (proof.verdict === 'unknown') {
      category = recorded === 'escalated' ? 'agree-escalate' : `unknown-vs-${recorded}`;
      const reason = proof.unknown?.reasons[0]?.slice(0, 80) ?? 'unknown';
      report.unknownReasons[reason] = (report.unknownReasons[reason] ?? 0) + 1;
    } else if (proof.verdict === 'allow' && recorded === 'granted') {
      category = due === granted ? 'agree-allow' : 'amount-mismatch';
    } else if (proof.verdict === 'allow') {
      category = 'RECOVERY';
      report.money.recoveryUsd += due;
    } else if (proof.verdict === 'deny' && recorded === 'granted') {
      category = 'LEAKAGE';
      report.money.leakageUsd += granted;
    } else {
      category = `agree-${proof.verdict === 'deny' ? 'deny' : 'other'}`;
    }
    report.agreement[category] = (report.agreement[category] ?? 0) + 1;
    if (category === 'RECOVERY' || category === 'LEAKAGE' || category === 'amount-mismatch' || proof.verdict === 'unknown') {
      report.disputes.push({
        ticket: String(row.ticket.id),
        category,
        verdict: proof.verdict,
        summary: proof.summary,
        recorded: recorded === 'granted' ? `granted $${granted}` : recorded,
        ...(category === 'RECOVERY' ? { amount: due } : category === 'LEAKAGE' ? { amount: granted } : {}),
      });
    }
    if (proof.verdict === 'unknown') report.rejected += 1;
    else report.decided += 1;
  }
  return report;
}

/** Realistic recorded actions, given the case's true verdict — the audit's error model. */
function recordedFrom(truth: Proof, fee: number, rng: Rng): NonNullable<ExportRow['recorded']> {
  if (truth.verdict === 'unknown') return { kind: 'escalated' };
  if (truth.verdict === 'deny') return rng.chance(0.1) ? { kind: 'granted', amount: Math.round(fee * 0.1) } : { kind: 'none' };
  const due = Number(truth.computation?.creditAmount ?? 0);
  const roll = rng.next();
  if (roll < 0.12) return { kind: 'none' };
  if (roll < 0.18) return { kind: 'granted', amount: due * 2 };
  return { kind: 'granted', amount: due };
}

/** The demo export: the seeded case generators plus the audit's error model, decided by the
 *  engine itself (truth first, then the recorded action derived from it). */
function synthesizeExport(count: number, family: 'sla' | 'refund', seed: number, packPath: string): ExportRow[] {
  const rng = new Rng(seed + 3);
  const pack = loadPack(packPath);
  const shape = family === 'refund' ? 'refund_eligible' : 'sla_credits';
  const rows: ExportRow[] = [];
  const facts: Array<{ ticket: TicketInput; fee: number; truth: ReturnType<typeof decide> }> = [];
  const raw: Array<CaseFact | RefundFact> = family === 'refund' ? generateRefundCases(count, seed) : generateCases(count, seed);
  for (const f of raw) {
    const isRefund = (f as RefundFact).purchasedIso !== undefined;
    const ticket: TicketInput = isRefund
      ? { id: (f as RefundFact).ticket, customer: (f as RefundFact).customer, fields: { purchased: (f as RefundFact).purchasedIso, return_requested: (f as RefundFact).returnIso, item_state: (f as RefundFact).itemState, customer: (f as RefundFact).customer, tier: (f as RefundFact).tier } }
      : { id: (f as CaseFact).ticket, customer: (f as CaseFact).customer, fields: { opened: (f as CaseFact).openedIso, priority: (f as CaseFact).priority, customer: (f as CaseFact).customer, tier: (f as CaseFact).tier, monthly_fee: (f as CaseFact).fee, first_response: (f as CaseFact).responseIso } };
    const sources: Source[] = [ticketSource(ticket)];
    const { admitted: claims } = admitAll(claimsFromTicket(ticket), sources);
    const truth = decide(shape, { ticket: ticket.id.toLowerCase() }, [...pack.claims, ...claims], { packId: pack.id, packVersion: pack.version, decidedAt: DECIDED_AT });
    facts.push({ ticket, fee: isRefund ? 0 : (f as CaseFact).fee, truth });
  }
  for (const { ticket, fee, truth } of facts) {
    rows.push({ ticket, recorded: recordedFrom(truth, fee, rng) });
  }
  return rows;
}

async function main() {
  const argv = process.argv.slice(2);
  const at = (name: string, fallback?: string) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback;
  };
  const family = (at('--family', 'sla') ?? 'sla') as 'sla' | 'refund';
  const exportPath = at('--export');
  const writeExportPath = at('--write-export');
  const count = Number(at('--synthesize', exportPath || writeExportPath ? '60' : '120'));
  const seed = Number(at('--seed', '20260924'));
  const storePath = at('--store', 'results/support-pipeline-verdicts.jsonl')!;
  const reportPath = at('--report', 'results/support-pipeline-report.json')!;
  const packPath = family === 'refund' ? 'benchmarks/support/pack-refund.json' : 'benchmarks/support/pack.json';

  if (writeExportPath) {
    // the committed demo export, generated by the same code the pipeline runs: truth decided
    // by the engine, recorded actions derived through the audit's error model
    const rows = synthesizeExport(count, family, seed, packPath);
    writeFileSync(writeExportPath, `${rows.map((r) => JSON.stringify(r)).join('\n')}\n`);
    console.log(`wrote ${rows.length}-case export to ${writeExportPath}`);
  }

  const rows: ExportRow[] = exportPath
    ? readFileSync(exportPath, 'utf8').split('\n').filter((l) => l.trim().length > 0).map((l) => JSON.parse(l) as ExportRow)
    : synthesizeExport(count, family, seed, packPath);

  const report = runPipeline(rows, family, packPath, storePath, DECIDED_AT);
  mkdirSync('results', { recursive: true });
  writeFileSync(reportPath, `${JSON.stringify(report, null, 1)}\n`);
  console.log(`pipeline: ${report.cases} cases (${family}), ${report.decided} proved, ${report.rejected} unknown`);
  for (const [k, v] of Object.entries(report.agreement)) console.log(`   ${String(v).padStart(4)}  ${k}`);
  console.log(`RECOVERY $${report.money.recoveryUsd}  LEAKAGE $${report.money.leakageUsd}  -> ${reportPath}`);
  console.log(`verdicts appended to ${storePath}`);
}

if (process.argv[1]?.endsWith('run-support-pipeline.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
