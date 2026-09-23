/**
 * Schema-guided extraction of a document into the facts the engine reasons over, with every fact
 * checked against its page before it is believed.
 *
 * The writer copies names as the page writes them (the engine resolves "K. Whitcombe", "Mx
 * Halloran" and "the Chief Risk Officer" to a person itself); dates come back as yyyymmdd and
 * money as whole dollars. Then the grounding check: every date must appear on the page in one of
 * the forms documents use, every amount in one of the forms money is written, every name and
 * reference verbatim. A fact that fails is dropped and counted — an unsupported fact never
 * reaches the engine, which is what lets the engine's answers be trusted.
 */

import { parseProgram } from '../engine/index.js';
import { normalizeExtractionOutput } from '../llm/extraction-guard.js';
import { dateSpellings, OPEN_END } from './dates.js';
import { formatMoney } from './render.js';

export const EXTRACTION_SCHEMA = `Every fact starts with 'Org', the organisation whose records the page belongs to (its name is in the page heading).
appointed('Org', 'Name as written', 'Role title', Date).   a person took a role from a date
ceased('Org', 'Name as written', 'Role title', Date).      a person stopped holding a role on a date
authority('Org', 'Role title', Limit, From, To).           the role may approve contracts up to Limit dollars from From until To; To is 99991231 when "until further notice"
approved('Org', 'CN-1234', 'Approver as written', Date).   a contract was approved; copy the approver exactly as named (full name, initial and surname, or role title)
contract('Org', 'CN-1234', 'Supplier name', Start, End, Value).   a contract register row; Value is the value at signing
amendment('Org', 'CN-1234', Number, Effective, Value).     a variation setting the contract value from a date
site_operator('Org', 'Site name', 'Supplier name').
certificate('Org', 'Supplier name', 'Standard', Issued, Expires).
incident('Org', 'INC-123A', Date, 'Site name', Severity).`;

export const EXTRACTION_PROMPT = `You extract facts from one page of an organisation's records into a fixed Datalog schema.

Schema (use only these predicates, with exactly these arguments):
${EXTRACTION_SCHEMA}

Rules:
- One fact per line, nothing else. If the page has no such facts, output exactly: % nothing
- Dates as yyyymmdd integers: 14/03/2025, 14 March 2025, 14 Mar 2025 and 2025-03-14 are all 20250314.
- Money as whole dollars: AUD 1,410,000, A$1,410,000 and $1.41 million are all 1410000.
- Quote every name, title and reference with single quotes. Copy people as the page writes them but without an honorific (Dr, Ms, Mr, Mx, Prof). Drop a leading "the" from role titles and site names.
- Only what the page states as done or in force. Never a proposal, a motion, or anything discussed but not resolved.`;

export const PREDICATES: Record<string, number> = {
  appointed: 4,
  ceased: 4,
  authority: 5,
  approved: 4,
  contract: 6,
  amendment: 5,
  site_operator: 3,
  certificate: 5,
  incident: 5,
};

export type Value = string | number;
export interface Fact {
  predicate: string;
  args: Value[];
}

/** Lower-case, collapse whitespace, drop an honorific or a leading "the"; ISO/IEC 27001 → iso 27001. */
export function normaliseName(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^(the|dr|ms|mr|mrs|mx|prof)\.?\s+/, '')
    .replace(/^iso\/iec\s+/, 'iso ');
}

/** Parse one reply into schema facts; anything off-schema or unparsable is dropped. */
export function parseSchemaFacts(reply: string): Fact[] {
  if (reply.trim() === '% nothing') return [];
  const facts: Fact[] = [];
  for (const line of normalizeExtractionOutput(reply)) {
    let clauses;
    try {
      clauses = parseProgram(line);
    } catch {
      continue;
    }
    for (const clause of clauses) {
      if (clause.body.length !== 0) continue;
      const predicate = clause.head.predicate;
      if (PREDICATES[predicate] !== clause.head.args.length) continue;
      const args: Value[] = [];
      let ok = true;
      for (const term of clause.head.args) {
        if (term.type === 'num') args.push(term.value);
        else if (term.type === 'atom') args.push(term.value);
        else ok = false;
      }
      if (ok) facts.push({ predicate, args });
    }
  }
  return facts;
}

/** Which argument positions hold a date, an amount, or a name/reference, per predicate. */
const SHAPE: Record<string, Array<'name' | 'date' | 'money' | 'plain'>> = {
  appointed: ['name', 'name', 'name', 'date'],
  ceased: ['name', 'name', 'name', 'date'],
  authority: ['name', 'name', 'money', 'date', 'date'],
  approved: ['name', 'name', 'name', 'date'],
  contract: ['name', 'name', 'name', 'date', 'date', 'money'],
  amendment: ['name', 'name', 'plain', 'date', 'money'],
  site_operator: ['name', 'name', 'name'],
  certificate: ['name', 'name', 'name', 'date', 'date'],
  incident: ['name', 'name', 'date', 'name', 'plain'],
};

function squash(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Does the page support this fact? Every date, amount and name must be on the page as written.
 * Returns the reason it is not, or undefined when it is.
 */
export function unsupportedReason(fact: Fact, page: string): string | undefined {
  const pageReason = supportedOn(fact, squash(page), 0);
  if (pageReason !== undefined) return pageReason;
  // every argument after the organisation must be stated by one line: a name, a role and a date
  // scattered over a page ("Present: the Chair and directors" above "V. Dubois approved…") are not
  // an appointment. The organisation may come from the heading.
  const lines = page.split(/\n/).map(squash).filter((line) => line.trim() !== '');
  if (!lines.some((line) => supportedOn(fact, line, 1) === undefined)) return 'not stated on any single line';
  return undefined;
}

function supportedOn(fact: Fact, text: string, fromArgument: number): string | undefined {
  const shape = SHAPE[fact.predicate]!;
  for (const [i, kind] of shape.entries()) {
    if (i < fromArgument) continue;
    const value = fact.args[i]!;
    if (kind === 'date') {
      if (typeof value !== 'number') return `argument ${i + 1} is not a date`;
      const month = Math.floor(value / 100) % 100;
      const day = value % 100;
      if (value !== OPEN_END && (value < 19_000_101 || value > 21_001_231 || month < 1 || month > 12 || day < 1 || day > 31)) {
        return `${value} is not a valid date`;
      }
      if (value === OPEN_END) {
        if (!/until further notice|continuing/.test(text)) return 'open-ended authority the page does not state';
        continue;
      }
      if (!dateSpellings(value).some((spelling) => text.includes(squash(spelling)))) return `date ${value} is not on the page`;
    } else if (kind === 'money') {
      if (typeof value !== 'number') return `argument ${i + 1} is not an amount`;
      const forms = [formatMoney(value, 'full'), formatMoney(value, 'symbol'), formatMoney(value, 'short')];
      if (!forms.some((form) => text.includes(squash(form)))) return `amount ${value} is not on the page`;
    } else if (kind === 'name') {
      if (typeof value !== 'string') return `argument ${i + 1} is not a name`;
      if (!text.includes(squash(value).replace(/^the /, ''))) return `"${value}" is not on the page`;
    }
  }
  return undefined;
}

const REF = /^(cn|inc)-[0-9a-z]+$/i;

/**
 * Type checks the schema implies, applied before anything reaches the engine. A contract
 * reference must look like one — when the writer swapped it with the approver ("V. Dubois
 * approved contract CN-6425" read in sentence order) the arguments are put back; a person slot
 * holding a role title is dropped rather than letting a role "hold" itself.
 */
export function repairFacts(facts: readonly Fact[]): { kept: Fact[]; repaired: number; rejected: Fact[] } {
  const roles = new Set<string>();
  for (const f of facts) {
    if (f.predicate === 'authority') roles.add(normaliseName(String(f.args[1])));
    if (f.predicate === 'appointed' || f.predicate === 'ceased') roles.add(normaliseName(String(f.args[2])));
  }
  const kept: Fact[] = [];
  const rejected: Fact[] = [];
  let repaired = 0;
  for (const fact of facts) {
    const args = [...fact.args];
    if (fact.predicate === 'approved' && !REF.test(String(args[1])) && REF.test(String(args[2]))) {
      [args[1], args[2]] = [args[2]!, args[1]!];
      repaired += 1;
    }
    const refSecond = ['approved', 'contract', 'amendment', 'incident'].includes(fact.predicate);
    if (refSecond && !REF.test(String(args[1]))) {
      rejected.push(fact);
      continue;
    }
    if ((fact.predicate === 'appointed' || fact.predicate === 'ceased') && roles.has(normaliseName(String(args[1])))) {
      rejected.push(fact);
      continue;
    }
    kept.push({ predicate: fact.predicate, args });
  }
  return { kept, repaired, rejected };
}
