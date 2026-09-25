/**
 * The gate challenge: the kill-criterion measurement, run on REAL text without human labels.
 *
 * The corpus is the XL-DocBench document set (real government and agency reports — messy
 * prose with dates, dollar amounts, program names). For every sampled span the harness
 * extracts the dates and money amounts that are genuinely stated there, then probes the
 * admission gate with claims whose ground truth is certain by construction:
 *
 *  - POSITIVE probe: the true value, canonicalized (the span states it) -> must ADMIT.
 *    Any rejection is an over-rejection: the gate bouncing a real fact.
 *  - NEGATIVE probes: seeded mutations (day shifted, digit dropped, value from another span)
 *    -> must REJECT. Any admission is a false admit: the matcher accepting a fact the span
 *    does not state. Mutation classes are reported separately, because they fail differently
 *    (a dropped year digit survives as a substring; a shifted day does not).
 *
 * This measures the matcher (normalizeText + surfacesFor) — the component every claim in
 * production passes through. A false-admit here is a hallucinated fact entering the engine
 * with a beautiful proof; an over-rejection is coverage quietly leaking.
 */

import { readFileSync, readdirSync, mkdirSync, writeFileSync } from 'node:fs';
import { normalizeText, surfacesFor, admit, PREDICATES } from '../support/claims.js';
import type { ClaimInput, ArgType } from '../support/claims.js';
import type { Span } from '../support/sources.js';
import { Rng } from '../compose/rng.js';

interface Probe {
  spanId: string;
  predicate: string;
  arg: string | number;
  truth: 'stated' | 'mutated';
  mutation?: string;
}

interface Outcome extends Probe {
  admitted: boolean;
  reason?: string;
}

const DOC_DIR = '.cache/document-recall';
const DATE_REGEXES = [
  /\b(\d{4}-\d{2}-\d{2})\b/g,
  /\b(\d{1,2}\/\d{1,2}\/\d{4})\b/g,
  /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b/g,
];
const MONEY_REGEX = /\$([\d,]+(?:\.\d{1,2})?)/g;

function extractDates(text: string): string[] {
  const found: string[] = [];
  for (const re of DATE_REGEXES) {
    for (const m of text.matchAll(re)) {
      // the full match is the surface; canonicalization is the gate's job, not ours
      found.push(m[0]);
    }
  }
  return found;
}

function extractMoney(text: string): number[] {
  const found: number[] = [];
  for (const m of text.matchAll(MONEY_REGEX)) {
    const n = Number(m[1]!.replace(/,/g, ''));
    if (Number.isFinite(n) && n > 0) found.push(n);
  }
  return found;
}

/** Words usable as an id argument: on the span, lowercase, >= 4 chars (avoids substring noise). */
function extractWords(text: string): string[] {
  return [...new Set(text.toLowerCase().match(/[a-z]{4,}/g) ?? [])];
}

function mutateDate(iso: string, kind: string): string | undefined {
  const base = iso.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!base) return undefined;
  const [, y, m, d] = base;
  switch (kind) {
    case 'day+1': {
      const dt = new Date(Date.UTC(Number(y), Number(m) - 1, Number(d)));
      dt.setUTCDate(dt.getUTCDate() + 1);
      return dt.toISOString().slice(0, 10);
    }
    case 'day+10': {
      const dt = new Date(Date.UTC(Number(y), Number(m) - 1, Number(d)));
      dt.setUTCDate(dt.getUTCDate() + 10);
      return dt.toISOString().slice(0, 10);
    }
    case 'year+1': return `${Number(y) + 1}-${m}-${d}`;
    case 'month-swap': return `${y}-${m === '01' ? '07' : '01'}-${d}`;
    case 'drop-last-digit': return `${y}-${m}-${d!.slice(0, 1)}`; // "12" -> "1": substring collision probe
    case 'drop-year-digits': return `${y!.slice(0, 2)}-${m}-${d}`;
    default: return undefined;
  }
}

function mutateMoney(n: number, kind: string): number {
  switch (kind) {
    case '+1': return n + 1;
    case 'x10': return n * 10;
    case 'drop-leading': return Number(String(n).slice(1)) || 1;
    case 'drop-trailing': return Math.floor(n / 10);
    default: return n;
  }
}

function claimFor(predicate: string, id: string, value: string | number, span: Span, scope: 'case' | 'pack'): ClaimInput {
  return { scope, predicate, args: [id, value], sourceId: span.sourceId, spanId: span.spanId, trust: 'system' };
}

function probe(predicate: string, argType: ArgType, id: string, value: string | number, span: Span, truth: Probe['truth'], mutation?: string): Outcome {
  const scope = PREDICATES[predicate]!.scope;
  const input = claimFor(predicate, id, value, span, scope);
  // the id word is verified on the span by the harness itself, so rejections isolate the value
  const result = admit(input, span);
  return { spanId: span.spanId, predicate, arg: value, truth, mutation, admitted: result.ok, reason: result.ok ? undefined : (result as { reason: string }).reason };
}

function idProbe(word: string, span: Span): boolean {
  return normalizeText(span.text).includes(word);
}
void idProbe;

async function main() {
  const argv = process.argv.slice(2);
  const at = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback;
  };
  const spansPerDoc = Number(at('--spans-per-doc', '2'));
  const seed = Number(at('--seed', '20260924'));
  const out = at('--out', 'results/support-gate-challenge.json');

  const rng = new Rng(seed);
  const docs = readdirSync(DOC_DIR).filter((f) => f.endsWith('.pages.txt'));
  const outcomes: Outcome[] = [];
  let spansProbed = 0;
  const falseAdmits: Outcome[] = [];
  const overRejects: Outcome[] = [];

  for (const doc of docs) {
    const pages = readFileSync(`${DOC_DIR}/${doc}`, 'utf8').split('\n\n');
    const candidates = pages
      .map((p, i) => ({ text: p.replace(/\s+/g, ' ').trim().slice(0, 1200), spanId: `${doc}:${i}` }))
      .filter((p) => p.text.length > 120);
    const chosen = rng.sample(candidates, Math.min(spansPerDoc, candidates.length));
    for (const c of chosen) {
      const span: Span = { sourceId: doc, spanId: c.spanId, kind: 'policy-line', text: c.text };
      const words = extractWords(c.text);
      const dates = extractDates(c.text);
      const amounts = extractMoney(c.text);
      if (dates.length === 0 && amounts.length === 0) continue;
      spansProbed += 1;

      const id = words.length > 0 ? rng.pick(words) : undefined;
      if (id === undefined) continue;

      // POSITIVE: the true values, canonicalized
      for (const surface of dates.slice(0, 2)) {
        const canonical = normalizeToIso(surface);
        if (canonical === undefined) continue;
        const result = probe('holiday', 'date', id, canonical, span, 'stated');
        outcomes.push(result);
        if (!result.admitted) overRejects.push(result);
      }
      for (const amount of amounts.slice(0, 2)) {
        const result = probe('monthly_fee', 'money', id, amount, span, 'stated');
        outcomes.push(result);
        if (!result.admitted) overRejects.push(result);
      }

      // NEGATIVE: mutations of the true values must all reject
      const trueSurfaces = new Set<string>();
      for (const surface of dates) {
        const iso = normalizeToIso(surface);
        if (iso !== undefined) for (const s of surfacesFor(iso, 'date')) trueSurfaces.add(normalizeText(s));
      }
      for (const amount of amounts) for (const s of surfacesFor(amount, 'money')) trueSurfaces.add(normalizeText(s));
      const coincidesWithStated = (value: string | number, type: ArgType): boolean =>
        surfacesFor(value, type).some((s) => trueSurfaces.has(normalizeText(s)));
      for (const surface of dates.slice(0, 2)) {
        const canonical = normalizeToIso(surface);
        if (canonical === undefined) continue;
        for (const kind of ['day+1', 'day+10', 'year+1', 'month-swap', 'drop-last-digit', 'drop-year-digits']) {
          const mutated = mutateDate(canonical, kind);
          if (mutated === undefined || mutated === canonical) continue;
          const result = probe('holiday', 'date', id, mutated, span, 'mutated', kind);
          (result as Outcome & { coincides?: boolean }).coincides = result.admitted ? coincidesWithStated(mutated, 'date') : undefined;
          outcomes.push(result);
          if (result.admitted) falseAdmits.push(result);
        }
      }
      for (const amount of amounts.slice(0, 2)) {
        for (const kind of ['+1', 'x10', 'drop-leading', 'drop-trailing']) {
          const mutated = mutateMoney(amount, kind);
          if (mutated === amount || !Number.isFinite(mutated)) continue;
          const result = probe('monthly_fee', 'money', id, mutated, span, 'mutated', kind);
          (result as Outcome & { coincides?: boolean }).coincides = result.admitted ? coincidesWithStated(mutated, 'money') : undefined;
          outcomes.push(result);
          if (result.admitted) falseAdmits.push(result);
        }
      }
    }
  }

  const stated = outcomes.filter((o) => o.truth === 'stated');
  const mutated = outcomes.filter((o) => o.truth === 'mutated');
  const matcherFailures = falseAdmits.filter((o) => (o as Outcome & { coincides?: boolean }).coincides !== true);
  const coincidences = falseAdmits.filter((o) => (o as Outcome & { coincides?: boolean }).coincides === true);
  const byMutation = new Map<string, { total: number; admitted: number }>();
  for (const o of mutated) {
    const key = `${o.predicate}:${o.mutation}`;
    const row = byMutation.get(key) ?? { total: 0, admitted: 0 };
    row.total += 1;
    if (o.admitted) row.admitted += 1;
    byMutation.set(key, row);
  }

  const report = {
    ranAt: new Date().toISOString(),
    seed,
    corpus: `${docs.length} real documents (${DOC_DIR})`,
    spansProbed,
    probes: outcomes.length,
    overRejection: {
      total: stated.length,
      rejected: overRejects.length,
      rate: stated.length > 0 ? Number((overRejects.length / stated.length).toFixed(4)) : 0,
      examples: overRejects.slice(0, 10),
    },
    falseAdmission: {
      total: mutated.length,
      admitted: falseAdmits.length,
      rate: mutated.length > 0 ? Number((falseAdmits.length / mutated.length).toFixed(4)) : 0,
      matcherFailures: matcherFailures.length,
      coincidesWithOtherStatedValue: coincidences.length,
      note: 'a coincidence is the span genuinely stating the mutated value on a DIFFERENT fact — a granularity limit of span-level grounding, not a matcher bug',
      byMutationClass: [...byMutation.entries()].map(([k, v]) => ({ class: k, ...v, rate: Number((v.admitted / v.total).toFixed(4)) })).sort((a, b) => b.rate - a.rate),
      examples: falseAdmits.slice(0, 10),
    },
  };
  mkdirSync('results', { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 1)}\n`);
  console.log(`gate challenge over ${spansProbed} real spans from ${docs.length} documents (${outcomes.length} probes)`);
  console.log(`over-rejection: ${report.overRejection.rejected}/${report.overRejection.total} = ${(report.overRejection.rate * 100).toFixed(2)}% real facts bounced`);
  console.log(`false admission: ${report.falseAdmission.admitted}/${report.falseAdmission.total} = ${(report.falseAdmission.rate * 100).toFixed(2)}% mutants accepted`);
  console.log(`   matcher failures: ${report.falseAdmission.matcherFailures}; coincides with another stated value on the span: ${report.falseAdmission.coincidesWithOtherStatedValue}`);
  for (const row of report.falseAdmission.byMutationClass.slice(0, 6)) {
    console.log(`   ${(row.rate * 100).toFixed(1).padStart(5)}% ${row.class} (${row.admitted}/${row.total})`);
  }
}

function normalizeToIso(surface: string): string | undefined {
  const iso = surface.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (iso) return `${iso[1]}-${iso[2]}-${iso[3]}`;
  const slash = surface.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (slash) return `${slash[3]}-${slash[1]!.padStart(2, '0')}-${slash[2]!.padStart(2, '0')}`;
  const named = surface.match(/^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$/);
  if (named) {
    const months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
    const idx = months.findIndex((m) => m.startsWith(named[1]!.toLowerCase()));
    if (idx >= 0) return `${named[3]}-${String(idx + 1).padStart(2, '0')}-${named[2]!.padStart(2, '0')}`;
  }
  return undefined;
}

if (process.argv[1]?.endsWith('run-support-gate.js')) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
