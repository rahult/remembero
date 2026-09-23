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
person(P) :- appointed(N, _, _), alias(N, P).
ended(P, R, F, D) :- appointed(N, R, F), alias(N, P), ceased(M, R, E), alias(M, P), day(D), F <= E, E <= D.
holds(P, R, D) :- appointed(N, R, F), alias(N, P), day(D), F <= D, \\+ ended(P, R, F, D).
approver(C, P) :- approved(C, A, _), alias(A, P).
approver(C, P) :- approved(C, A, D), holds(P, A, D).
has_approval(C) :- approved(C, _, _).
value_entry(C, V, E) :- contract(C, _, _, _, V), approved(C, _, E).
value_entry(C, V, S) :- contract(C, _, S, _, V), \\+ has_approval(C).
value_entry(C, V, E) :- amendment(C, _, E, V).
later_value(C, E, D) :- value_entry(C, _, E), value_entry(C, _, E2), day(D), E < E2, E2 <= D.
value_on(C, V, D) :- value_entry(C, V, E), day(D), E <= D, \\+ later_value(C, E, D).
limit_on(P, L, D) :- holds(P, R, D), authority(R, L, F, T), F <= D, D < T.
within_authority(C) :- approved(C, _, D), approver(C, P), limit_on(P, L, D), value_on(C, V, D), V <= L.
decidable(C) :- approved(C, _, D), approver(C, P), holds(P, _, D), value_on(C, _, D).
cert_valid_on(S, Std, D) :- certificate(S, Std, I, X), day(D), I <= D, D < X.
has_operator(Site) :- site_operator(Site, _).
orphan_incident(I) :- incident(I, _, Site, _), \\+ has_operator(Site).
uncertified_incident(I, Std) :- incident(I, D, Site, _), site_operator(Site, S), standard(Std), \\+ cert_valid_on(S, Std, D).
supplier(S) :- contract(_, S, _, _, _).
supplier(S) :- site_operator(_, S).
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
  const full = [...new Set(facts.filter((f) => f.predicate === 'appointed').map((f) => String(f.args[0])))];
  const byAlias = new Map<string, Set<string>>();
  const add = (alias: string, name: string) => byAlias.set(alias, new Set([...(byAlias.get(alias) ?? []), name]));
  for (const name of full) {
    const parts = name.split(' ');
    const last = parts[parts.length - 1]!;
    add(name, name);
    add(last, name);
    if (parts.length > 1) add(`${parts[0]![0]}. ${last}`, name);
  }
  return [...byAlias.entries()].filter(([, names]) => names.size === 1).map(([alias, names]) => `alias(${quote(alias)}, ${quote([...names][0]!)}).`);
}

export class EngineAnswerer {
  private readonly clauses: Clause[];

  constructor(facts: readonly Fact[], questionDays: readonly number[]) {
    const normalised = facts.map(normaliseFact);
    const days = new Set<number>(questionDays);
    for (const fact of normalised) for (const a of fact.args) if (typeof a === 'number' && a > 19_000_000) days.add(a);
    const text = [
      ...normalised.map((f) => `${f.predicate}(${f.args.map((a) => (typeof a === 'number' ? String(a) : quote(a))).join(', ')}).`),
      ...aliasFacts(normalised),
      ...[...days].map((d) => `day(${d}).`),
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
    const date = params.date;
    switch (family) {
      case 'needle':
        return one(this.column(`contract(${c('contract')}, _, _, _, V)`, 'V'));
      case 'validity':
        return one(this.column(`holds(P, ${c('role')}, ${date})`, 'P'));
      case 'identity':
        return one(this.column(`approver(${c('contract')}, P)`, 'P'));
      case 'supersession':
        return one(this.column(`value_on(${c('contract')}, V, ${date})`, 'V'));
      case 'comparison': {
        const a = this.column(`value_on(${c('contractA')}, V, ${date})`, 'V');
        const b = this.column(`value_on(${c('contractB')}, V, ${date})`, 'V');
        if (a.length !== 1 || b.length !== 1 || a[0] === b[0]) return 'unknown';
        return Number(a[0]) > Number(b[0]) ? String(params.contractA) : String(params.contractB);
      }
      case 'aggregate': {
        if (this.column('orphan_incident(I)', 'I').length > 0) return 'unknown';
        if (params.standard !== undefined) {
          return String(this.column(`uncertified_incident(I, ${c('standard')})`, 'I').length);
        }
        return String(this.column(`incident(I, _, Site, _), site_operator(Site, ${c('supplier')})`, 'I').length);
      }
      case 'absence': {
        const lacking = this.column(`supplier(S), \\+ cert_valid_on(S, ${c('standard')}, ${date})`, 'S');
        return lacking.length === 0 ? 'unknown' : lacking.join('; ');
      }
      case 'authority': {
        const contract = c('contract');
        if (this.holds(`within_authority(${contract})`)) return 'yes';
        return this.holds(`decidable(${contract})`) ? 'no' : 'unknown';
      }
      case 'unanswerable':
        return params.role !== undefined
          ? one(this.column(`holds(P, ${c('role')}, ${date})`, 'P'))
          : one(this.column(`contract(${c('contract')}, _, _, _, V)`, 'V'));
      default:
        return 'unknown';
    }
  }
}
