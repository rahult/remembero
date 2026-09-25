/**
 * The pack: policy parameters extracted from a customer's long-form SOP into a versioned,
 * human-gated file. The LLM extracts parameters; it never writes rules — rule templates are
 * code, the pack is data. A pack ships only when EVERY claim passes the same admission gate
 * as case claims (each parameter grounded on one line of the SOP), because a misread
 * threshold produces confidently-wrong verdicts with beautiful proofs — the gate catches
 * hallucination, and the human sign-off catches misreading.
 *
 * Coverage review is the pack's other duty: the loader reports SOP lines that state pack
 * predicates but are consumed by no claim, so an unmodeled exception clause is flagged to
 * the reviewer instead of silently ignored.
 */

import { readFileSync } from 'node:fs';
import { admitAll, hashJson, type Claim, type ClaimInput } from './claims.js';
import { policySource, type Source } from './sources.js';

export interface PackFile {
  id: string;
  version: string;
  sopPath: string;
  /** Every pack parameter, with the SOP phrase that grounds it. */
  claims: Array<{ predicate: string; args: (string | number)[]; quote: string }>;
}

export interface Pack {
  id: string;
  version: string;
  claims: Claim[];
  sources: Source[];
  /** SOP lines that look like they state a pack predicate but no claim consumes them. */
  unconsumedLines: Array<{ spanId: string; text: string }>;
  contentHash: string;
}

const PACK_PREDICATES = new Set(['sla_clock', 'business_hours', 'holiday', 'zone', 'default_calendar', 'credit_band']);

export function loadPack(packPath: string, sopRoot = ''): Pack {
  const file = JSON.parse(readFileSync(packPath, 'utf8')) as PackFile;
  const sopText = readFileSync(`${sopRoot}${file.sopPath}`, 'utf8');
  const sources = [policySource(file.id, sopText)];
  const policy = sources[0]!;

  const inputs: ClaimInput[] = file.claims.map((c) => {
    // bind the claim to the SOP line its quote grounds
    const span = policy.spans.find((sp) => sp.text.toLowerCase().includes(c.quote.toLowerCase()));
    if (!span) throw new Error(`pack claim ${c.predicate}(${c.args.join(', ')}) cites a phrase not in the SOP: "${c.quote}"`);
    return { scope: 'pack', predicate: c.predicate, args: c.args, sourceId: policy.sourceId, spanId: span.spanId, trust: 'hand' };
  });
  const report = admitAll(inputs, sources);
  if (report.rejected.length > 0) {
    const detail = report.rejected.map((r) => `${r.predicate}: ${r.reason}`).join('\n  ');
    throw new Error(`pack ${file.id}@${file.version} does not pass the admission gate — refusing to load:\n  ${detail}`);
  }

  const consumed = new Set(report.admitted.map((c) => c.source.spanId));
  const unconsumedLines = policy.spans
    .filter((sp) => {
      const lower = sp.text.toLowerCase();
      const looksRelevant = [...PACK_PREDICATES].some((p) => lower.includes(p.replace(/_/g, ' ')))
        || /\b(business hours|holiday|calendar|percent|first response|sla)\b/.test(lower)
        || /\b(refund|return|final sale|store credit|reseller|sealed|window)\b/.test(lower)
        || /\b(mon|tue|wed|thu|fri|sat|sun)/.test(lower);
      return looksRelevant && !consumed.has(sp.spanId) && !sp.text.startsWith('#');
    })
    .map((sp) => ({ spanId: sp.spanId, text: sp.text }));

  return {
    id: file.id,
    version: file.version,
    claims: report.admitted,
    sources,
    unconsumedLines,
    contentHash: hashJson(file),
  };
}
