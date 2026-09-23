/**
 * A world as a Datalog program: its facts, and the rules that turn them into the answers the
 * benchmark asks for. The gold answer to every question is this program's result — which is why
 * a Compose label cannot be wrong unless the rule below is.
 *
 * Dates are yyyymmdd integers, so date order is integer order. `day(D)` lists every date the
 * rules may be asked about; the engine needs a positive relation to bind a date before comparing.
 */

import { evaluate, parseProgram, parseQuery, type Bindings, type Clause } from '../engine/index.js';
import { OPEN_END, type Ymd } from './dates.js';
import { personName, type World } from './world.js';

export const RULES = `
holds_role_on(R, P, D) :- role_held(R, P, F, T), day(D), F <= D, D < T.
authority_on(R, L, D) :- authority(R, L, F, T), day(D), F <= D, D < T.
limit_on(P, L, D) :- holds_role_on(R, P, D), authority_on(R, L, D).
has_authority_on(P, D) :- limit_on(P, _, D).
later_value(C, E, D) :- contract_value(C, _, E), contract_value(C, _, E2), day(D), E < E2, E2 <= D.
value_on(C, V, D) :- contract_value(C, V, E), day(D), E <= D, \\+ later_value(C, E, D).
may_approve(P, C, D) :- limit_on(P, L, D), value_on(C, V, D), V <= L.
cert_valid_on(S, Std, D) :- cert(S, Std, I, X), day(D), I <= D, D < X.
incident_uncertified(I, Std) :- incident(I, Site, D, _), site(Site, S), standard(Std), \\+ cert_valid_on(S, Std, D).
`;

function quote(text: string): string {
  return `'${text.replace(/'/g, "\\'")}'`;
}

/** Every fact the documents are written from. */
export function worldFacts(world: World): string[] {
  const facts: string[] = [];
  for (const p of world.people) facts.push(`person_name(${p.id}, ${quote(personName(p))}).`);
  for (const t of world.roleTerms) facts.push(`role_held(${t.role}, ${t.person}, ${t.from}, ${t.to}).`);
  for (const a of world.authorities) facts.push(`authority(${a.role}, ${a.limit}, ${a.from}, ${a.to}).`);
  for (const s of world.suppliers) facts.push(`supplier_name(${s.id}, ${quote(s.name)}).`);
  for (const s of world.sites) facts.push(`site(${s.id}, ${s.supplier}).`);
  for (const c of world.contracts) facts.push(`contract(${c.id}, ${c.supplier}, ${c.from}, ${c.to}).`);
  for (const v of world.contractValues) facts.push(`contract_value(${v.contract}, ${v.value}, ${v.effective}).`);
  for (const a of world.approvals) facts.push(`approved(${a.contract}, ${a.person}, ${a.on}).`);
  for (const c of world.certificates) facts.push(`cert(${c.supplier}, ${c.standard}, ${c.issued}, ${c.expires}).`);
  for (const i of world.incidents) facts.push(`incident(${i.id}, ${i.site}, ${i.on}, ${i.severity}).`);
  facts.push('standard(iso9001).', 'standard(iso27001).');
  return facts;
}

/** Every date the world or its questions mention, so the rules can bind them. */
export function worldDays(world: World, extra: readonly Ymd[] = []): Ymd[] {
  const days = new Set<Ymd>(extra);
  for (const t of world.roleTerms) days.add(t.from);
  for (const v of world.contractValues) days.add(v.effective);
  for (const a of world.approvals) days.add(a.on);
  for (const i of world.incidents) days.add(i.on);
  days.delete(OPEN_END);
  return [...days].sort((a, b) => a - b);
}

/** The world as a program the engine can answer from. */
export class WorldProgram {
  private readonly clauses: Clause[];

  constructor(readonly world: World, questionDays: readonly Ymd[] = []) {
    const text = [
      ...worldFacts(world),
      ...worldDays(world, questionDays).map((d) => `day(${d}).`),
      RULES,
    ].join('\n');
    this.clauses = parseProgram(text);
  }

  query(goal: string): Bindings[] {
    return evaluate(this.clauses, parseQuery(goal), { maxRows: 10_000 });
  }

  /** The values of one variable across the result rows, deduplicated. */
  column(goal: string, variable: string): Array<string | number> {
    const out = new Set<string | number>();
    for (const row of this.query(goal)) {
      const term = row[variable];
      if (term === undefined) continue;
      if (term.type === 'atom' || term.type === 'num') out.add(term.value);
    }
    return [...out];
  }

  holds(goal: string): boolean {
    return this.query(goal).length > 0;
  }
}
