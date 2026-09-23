/**
 * Answer Compose questions with the Datalog engine over extracted facts — the "Proved" path.
 *
 * The engine says Unknown whenever the facts do not settle the question: no row, more than one
 * row where one is expected, an approver it cannot resolve to a person, an incident at a site
 * with no known operator. That is the certainty contract: a Proved answer or Unknown, never a
 * guess. Which query to run is chosen from the question's parameters — an oracle planner, so this
 * measures extraction and reasoning, not question understanding (which is the next step).
 */

import { evaluate, parseProgram, parseQuery, type Clause } from '../engine/index.js';
import { normaliseName, type Fact } from './extract.js';

export const ENGINE_RULES = `
when(D) :- qday(D).
when(D) :- approved(_, _, _, D).
cday(D) :- qday(D).
cday(D) :- incident(_, _, D, _, _).
term(O, P, R, F) :- appointed(O, N, R, F), alias(O, N, P).
stop(O, P, R, E) :- ceased(O, M, R, E), alias(O, M, P).
ended(O, P, R, F, D) :- term(O, P, R, F), stop(O, P, R, E), when(D), F <= E, E <= D.
holds(O, P, R, D) :- term(O, P, R, F), when(D), F <= D, \\+ ended(O, P, R, F, D).
approver(O, C, P) :- approved(O, C, A, _), alias(O, A, P).
approver(O, C, P) :- approved(O, C, A, D), holds(O, P, A, D).
has_approval(O, C) :- approved(O, C, _, _).
value_entry(O, C, V, E) :- contract(O, C, _, _, _, V), approved(O, C, _, E).
value_entry(O, C, V, S) :- contract(O, C, _, S, _, V), \\+ has_approval(O, C).
value_entry(O, C, V, E) :- amendment(O, C, _, E, V).
later_value(O, C, E, D) :- value_entry(O, C, _, E), value_entry(O, C, _, E2), when(D), E < E2, E2 <= D.
value_on(O, C, V, D) :- value_entry(O, C, V, E), when(D), E <= D, \\+ later_value(O, C, E, D).
limit_on(O, P, L, D) :- holds(O, P, R, D), authority(O, R, L, F, T), F <= D, D < T.
within_authority(O, C) :- approved(O, C, _, D), approver(O, C, P), limit_on(O, P, L, D), value_on(O, C, V, D), V <= L.
decidable(O, C) :- approved(O, C, _, D), approver(O, C, P), holds(O, P, _, D), value_on(O, C, _, D).
has_previous(O, C, N) :- amendment(O, C, N, _, _), amendment(O, C, M, _, _), M = N - 1.
amendment_gap(O, C) :- amendment(O, C, N, _, _), N > 1, \\+ has_previous(O, C, N).
cert_valid_on(O, S, Std, D) :- certificate(O, S, Std, I, X), cday(D), I <= D, D < X.
has_operator(O, Site) :- site_operator(O, Site, _).
orphan_incident(O, I) :- incident(O, I, _, Site, _), \\+ has_operator(O, Site).
uncertified_incident(O, I, Std) :- incident(O, I, D, Site, _), site_operator(O, Site, S), standard(Std), \\+ cert_valid_on(O, S, Std, D).
supplier(O, S) :- contract(O, _, S, _, _, _).
supplier(O, S) :- site_operator(O, _, S).
`;

function quote(text: string): string {
  return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

/** Normalise every name-like argument, so the engine joins on one spelling. */
function normaliseFact(fact: Fact): Fact {
  return { predicate: fact.predicate, args: fact.args.map((a) => (typeof a === 'string' ? normaliseName(a) : a)) };
}

/**
 * Alias facts from the appointments: each full name answers to itself, its surname and its
 * initial-and-surname form. A surname two people share is dropped rather than guessed.
 */
export function aliasFacts(facts: readonly Fact[]): string[] {
  const out: string[] = [];
  const orgs = new Set(facts.filter((f) => f.predicate === 'appointed').map((f) => String(f.args[0])));
  for (const org of orgs) {
    const full = [...new Set(facts.filter((f) => f.predicate === 'appointed' && f.args[0] === org).map((f) => String(f.args[1])))];
    const byAlias = new Map<string, Set<string>>();
    const add = (alias: string, name: string) => byAlias.set(alias, new Set([...(byAlias.get(alias) ?? []), name]));
    for (const name of full) {
      const parts = name.split(' ');
      const last = parts[parts.length - 1]!;
      add(name, name);
      add(last, name);
      if (parts.length > 1) add(`${parts[0]![0]}. ${last}`, name);
    }
    for (const [alias, names] of byAlias) {
      if (names.size === 1) out.push(`alias(${quote(org)}, ${quote(alias)}, ${quote([...names][0]!)}).`);
    }
  }
  return out;
}

export class EngineAnswerer {
  private readonly clauses: Clause[];

  constructor(facts: readonly Fact[], questionDays: readonly number[]) {
    const normalised = facts.map(normaliseFact);
    // only the dates questions ask about; approval and incident dates come from their own facts
    const days = new Set<number>(questionDays);
    const text = [
      ...normalised.map((f) => `${f.predicate}(${f.args.map((a) => (typeof a === 'number' ? String(a) : quote(a))).join(', ')}).`),
      ...aliasFacts(normalised),
      ...[...days].map((d) => `qday(${d}).`),
      `standard('iso 9001').`,
      `standard('iso 27001').`,
      ENGINE_RULES,
    ];
    // one bad line must not sink the program; parse fact by fact
    const clauses: Clause[] = [];
    for (const line of text) {
      try {
        clauses.push(...parseProgram(line));
      } catch {
        // skipped: a fact the parser rejects is a fact the engine cannot use
      }
    }
    this.clauses = clauses;
  }

  column(goal: string, variable: string): Array<string | number> {
    const out = new Set<string | number>();
    for (const row of evaluate(this.clauses, parseQuery(goal), { maxRows: 10_000 })) {
      const term = row[variable];
      if (term !== undefined && (term.type === 'atom' || term.type === 'num')) out.add(term.value);
    }
    return [...out];
  }

  holds(goal: string): boolean {
    return evaluate(this.clauses, parseQuery(goal), { maxRows: 1 }).length > 0;
  }

  /** The engine's answer as text, or "unknown" when the facts do not settle it. */
  answer(family: string, params: Record<string, string | number>): string {
    const one = (values: Array<string | number>) => (values.length === 1 ? String(values[0]) : 'unknown');
    const c = (key: string) => quote(normaliseName(String(params[key])));
    // the organisation when the question names one; otherwise a variable the contract binds
    const org = params.organisation === undefined ? 'O' : c('organisation');
    const date = params.date;
    switch (family) {
      case 'needle':
        return one(this.column(`contract(${org}, ${c('contract')}, _, _, _, V)`, 'V'));
      case 'validity':
        return one(this.column(`holds(${org}, P, ${c('role')}, ${date})`, 'P'));
      case 'identity':
        return one(this.column(`approver(${org}, ${c('contract')}, P)`, 'P'));
      case 'supersession':
        // a numbered amendment with its predecessor missing: the record is provably incomplete
        if (this.holds(`amendment_gap(${org}, ${c('contract')})`)) return 'unknown';
        return one(this.column(`value_on(${org}, ${c('contract')}, V, ${date})`, 'V'));
      case 'comparison': {
        if (this.holds(`amendment_gap(${org}, ${c('contractA')})`) || this.holds(`amendment_gap(${org}, ${c('contractB')})`)) return 'unknown';
        const a = this.column(`value_on(${org}, ${c('contractA')}, V, ${date})`, 'V');
        const b = this.column(`value_on(${org}, ${c('contractB')}, V, ${date})`, 'V');
        if (a.length !== 1 || b.length !== 1 || a[0] === b[0]) return 'unknown';
        return Number(a[0]) > Number(b[0]) ? String(params.contractA) : String(params.contractB);
      }
      case 'aggregate': {
        if (params.standard !== undefined) {
          if (params.organisation === undefined) return 'unknown';
          if (this.column(`orphan_incident(${org}, I)`, 'I').length > 0) return 'unknown';
          return String(this.column(`uncertified_incident(${org}, I, ${c('standard')})`, 'I').length);
        }
        const orgs = this.column(`site_operator(O, _, ${c('supplier')})`, 'O');
        if (orgs.length !== 1) return 'unknown';
        const o = quote(String(orgs[0]));
        if (this.column(`orphan_incident(${o}, I)`, 'I').length > 0) return 'unknown';
        return String(this.column(`incident(${o}, I, _, Site, _), site_operator(${o}, Site, ${c('supplier')})`, 'I').length);
      }
      case 'absence': {
        if (params.organisation === undefined) return 'unknown';
        const lacking = this.column(`supplier(${org}, S), \\+ cert_valid_on(${org}, S, ${c('standard')}, ${date})`, 'S');
        return lacking.length === 0 ? 'unknown' : lacking.join('; ');
      }
      case 'authority': {
        const contract = c('contract');
        if (this.holds(`amendment_gap(${org}, ${contract})`)) return 'unknown';
        if (this.holds(`within_authority(${org}, ${contract})`)) return 'yes';
        return this.holds(`decidable(${org}, ${contract})`) ? 'no' : 'unknown';
      }
      case 'unanswerable':
        return params.role !== undefined
          ? one(this.column(`holds(${org}, P, ${c('role')}, ${date})`, 'P'))
          : one(this.column(`contract(${org}, ${c('contract')}, _, _, _, V)`, 'V'));
      default:
        return 'unknown';
    }
  }
}
