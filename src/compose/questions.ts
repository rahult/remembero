/**
 * Question families, each generated from a world and answered by the engine.
 *
 * A question carries three things beside its text: the gold answer (from `WorldProgram`), the
 * ways a correct answer may spell it, and the distractors — plausible wrong answers the world
 * makes available (the previous office holder, the superseded value, the supplier that *was*
 * certified). An answer naming a distractor is confidently wrong, which is the number this
 * benchmark exists to drive to zero.
 */

import { Rng } from './rng.js';
import { addDays, dateSpellings, formatDate, OPEN_END, ymd, type Ymd } from './dates.js';
import { WorldProgram } from './program.js';
import { keys } from './render.js';
import { personName, roleName, STANDARD_NAMES, WORLD_END, WORLD_START, type Person, type World } from './world.js';

export const FAMILIES = [
  'needle',
  'validity',
  'identity',
  'supersession',
  'comparison',
  'aggregate',
  'absence',
  'authority',
  'unanswerable',
] as const;
export type Family = (typeof FAMILIES)[number];

export type GoldKind = 'entity' | 'set' | 'number' | 'decision' | 'unknown';

export interface ComposeGold {
  kind: GoldKind;
  /** For entity and set: one list of acceptable spellings per item that must appear. */
  items: string[][];
  /** For number: the value. */
  number?: number;
  /** For decision: whether the action was allowed. */
  allowed?: boolean;
  /** Spellings of wrong-but-plausible answers; naming one is confidently wrong. */
  distractors: string[];
  /** Spellings that name the gold only in part (a surname when the full name was asked). */
  partial?: string[];
  distractorNumbers: number[];
  /** Human-readable gold, for reports. */
  display: string;
}

export interface ComposeQuestion {
  id: string;
  family: Family;
  /** Facts the answer combines. */
  hops: number;
  question: string;
  gold: ComposeGold;
  /** Evidence keys (see render.ts); their pages are the question's evidence pages. */
  evidence: string[];
  /** The engine query the gold came from, kept for audit and for training the planner. */
  datalog: string;
  /**
   * What the question is about, as the documents write it (a contract reference, a role title, a
   * yyyymmdd date, a supplier name, a standard). An engine planner builds its query from these;
   * a model reader never sees them.
   */
  params: Record<string, string | number>;
}

function personSpellings(p: Person): string[] {
  return [personName(p), `${p.title} ${p.last}`, `${p.first[0]}. ${p.last}`, p.last];
}

function supplierSpellings(name: string): string[] {
  const stem = name.replace(/ Pty Ltd$/, '');
  return [name, stem, stem.split(' ')[0]!];
}

export function generateQuestions(world: World, perFamily: Partial<Record<Family, number>> = {}): ComposeQuestion[] {
  const rng = new Rng(world.seed ^ 0x9e37);
  const person = new Map(world.people.map((p) => [p.id, p]));
  const supplier = new Map(world.suppliers.map((s) => [s.id, s]));
  const contract = new Map(world.contracts.map((c) => [c.id, c]));
  const questions: ComposeQuestion[] = [];
  const everyoneBut = (who: Person) => world.people.filter((p) => p.id !== who.id).flatMap((p) => [personName(p), p.last]);
  const count = (family: Family, fallback: number) => perFamily[family] ?? fallback;
  const say = (d: Ymd) => formatDate(d, rng.pick(['long', 'long', 'iso'] as const));

  // dates the questions will ask about must be `day` facts; gather them first
  const askDays: Ymd[] = [];
  const pickDay = (from: Ymd, to: Ymd) => {
    const d = addDays(from, rng.int(1, Math.max(1, dayGap(from, to) - 1)));
    askDays.push(d);
    return d;
  };
  const plans: Array<(program: WorldProgram) => ComposeQuestion | undefined> = [];
  let n = 0;
  const id = (family: Family) => `w${world.seed}-${family}-${++n}`;

  // --- needle: one register row ---
  for (const c of rng.sample(world.contracts, Math.min(count('needle', 6), world.contracts.length))) {
    plans.push(() => {
      const values = world.contractValues.filter((v) => v.contract === c.id);
      const signed = values.find((v) => v.amendment === 0)!;
      return {
        id: id('needle'),
        family: 'needle',
        hops: 1,
        question: `What was the value of contract ${c.ref} at signing?`,
        gold: { kind: 'number', items: [], number: signed.value, distractors: [], distractorNumbers: values.filter((v) => v.amendment > 0).map((v) => v.value), display: String(signed.value) },
        evidence: [keys.value(c.id, 0)],
        datalog: `value_on(${c.id}, V, ${signed.effective})`,
        params: { contract: c.ref },
      };
    });
  }

  // --- validity: who held a role on a date ---
  const heldTerms = world.roleTerms.filter((t) => t.from < WORLD_END);
  for (const term of rng.sample(heldTerms, Math.min(count('validity', 6), heldTerms.length))) {
    const on = pickDay(term.from, term.to === OPEN_END ? WORLD_END : term.to);
    plans.push((program) => {
      const holders = program.column(`holds_role_on(${term.role}, P, ${on})`, 'P');
      if (holders.length !== 1) return undefined;
      const holder = person.get(String(holders[0]))!;
      const held = world.roleTerms.find((t) => t.role === term.role && t.person === holder.id && t.from <= on && on < t.to)!;
      return {
        id: id('validity'),
        family: 'validity',
        hops: 2,
        question: `Who was the ${roleName(world, term.role)} of ${world.organisation} on ${say(on)}?`,
        gold: { kind: 'entity', items: [personSpellings(holder)], distractors: everyoneBut(holder), distractorNumbers: [], display: personName(holder) },
        evidence: [keys.roleStart(held.role, holder.id), ...(held.to === OPEN_END ? [] : [keys.roleEnd(held.role, holder.id)])],
        datalog: `holds_role_on(${term.role}, P, ${on})`,
        params: { role: roleName(world, term.role), date: on, organisation: world.organisation },
      };
    });
  }

  // --- identity: the minutes name the approver by initial or only by role ---
  const indirect = world.approvals.filter((a) => a.namedAs !== 'name');
  for (const approval of rng.sample(indirect, Math.min(count('identity', 5), indirect.length))) {
    plans.push((program) => {
      const c = contract.get(approval.contract)!;
      const found = program.column(`approved(${c.id}, P, _)`, 'P');
      const approver = person.get(String(found[0]))!;
      const term = world.roleTerms.find((t) => t.person === approver.id && t.from <= approval.on && approval.on < t.to)!;
      return {
        id: id('identity'),
        family: 'identity',
        hops: approval.namedAs === 'role' ? 3 : 2,
        question: `What is the full name of the person who approved contract ${c.ref}?`,
        gold: { kind: 'entity', items: [[personName(approver), `${approver.title} ${approver.first} ${approver.last}`]], distractors: everyoneBut(approver), partial: [approver.last], distractorNumbers: [], display: personName(approver) },
        evidence: [keys.approval(c.id), keys.roleStart(term.role, approver.id), keys.person(approver.id)],
        datalog: `approved(${c.id}, P, D), person_name(P, N)`,
        params: { contract: c.ref },
      };
    });
  }

  // --- supersession: value on a date after amendments ---
  const amended = [...new Set(world.contractValues.filter((v) => v.amendment > 0).map((v) => v.contract))];
  for (const contractId of rng.sample(amended, Math.min(count('supersession', 5), amended.length))) {
    const values = world.contractValues.filter((v) => v.contract === contractId).sort((a, b) => a.effective - b.effective);
    const latest = values[values.length - 1]!;
    const on = pickDay(latest.effective, addDays(latest.effective, 200));
    plans.push((program) => {
      const c = contract.get(contractId)!;
      const found = program.column(`value_on(${contractId}, V, ${on})`, 'V');
      if (found.length !== 1) return undefined;
      return {
        id: id('supersession'),
        family: 'supersession',
        hops: values.length,
        question: `What was the value of contract ${c.ref} on ${say(on)}, taking any amendments into account?`,
        gold: { kind: 'number', items: [], number: Number(found[0]), distractors: [], distractorNumbers: values.filter((v) => v !== latest).map((v) => v.value), display: String(found[0]) },
        evidence: values.map((v) => keys.value(contractId, v.amendment)),
        datalog: `value_on(${contractId}, V, ${on})`,
        params: { contract: c.ref, date: on },
      };
    });
  }

  // --- comparison: which of two contracts was worth more on a date ---
  for (let k = 0; k < count('comparison', 5); k += 1) {
    const [a, b] = rng.sample(world.contracts, 2) as [typeof world.contracts[0], typeof world.contracts[0]];
    const start = Math.max(a.from, b.from);
    const on = pickDay(start, addDays(start, 600));
    plans.push((program) => {
      const va = Number(program.column(`value_on(${a.id}, V, ${on})`, 'V')[0]);
      const vb = Number(program.column(`value_on(${b.id}, V, ${on})`, 'V')[0]);
      if (!Number.isFinite(va) || !Number.isFinite(vb) || va === vb) return undefined;
      const [win, lose] = va > vb ? [a, b] : [b, a];
      const valueKeys = (cid: string, day: Ymd) =>
        world.contractValues.filter((v) => v.contract === cid && v.effective <= day).map((v) => keys.value(cid, v.amendment));
      return {
        id: id('comparison'),
        family: 'comparison',
        hops: valueKeys(a.id, on).length + valueKeys(b.id, on).length,
        question: `On ${say(on)}, which contract had the higher value: ${a.ref} or ${b.ref}?`,
        gold: { kind: 'entity', items: [[win.ref]], distractors: [lose.ref], distractorNumbers: [], display: win.ref },
        evidence: [...valueKeys(a.id, on), ...valueKeys(b.id, on)],
        datalog: `value_on(${a.id}, VA, ${on}), value_on(${b.id}, VB, ${on}), VA > VB`,
        params: { contractA: a.ref, contractB: b.ref, date: on },
      };
    });
  }

  // --- aggregate: incidents at sites whose operator lacked a valid certificate that day ---
  for (const standard of rng.shuffle(['iso9001', 'iso27001'] as const).slice(0, count('aggregate', 2))) {
    plans.push((program) => {
      const uncertified = program.column(`incident_uncertified(I, ${standard})`, 'I');
      const all = world.incidents.length;
      const touched = world.incidents.filter((i) => uncertified.includes(i.id));
      return {
        id: id('aggregate'),
        family: 'aggregate',
        hops: 3,
        question: `How many safety incidents in ${world.organisation}'s incident log occurred at a site whose operating supplier did not hold a valid ${STANDARD_NAMES[standard]} certificate on the day of the incident?`,
        gold: { kind: 'number', items: [], number: uncertified.length, distractors: [], distractorNumbers: [all, all - uncertified.length].filter((x) => x !== uncertified.length), display: String(uncertified.length) },
        evidence: [...touched.map((i) => keys.incident(i.id)), ...touched.map((i) => keys.site(i.site))],
        datalog: `incident_uncertified(I, ${standard})`,
        params: { standard: STANDARD_NAMES[standard], organisation: world.organisation },
      };
    });
  }
  // and a per-supplier count, which needs the site join but not the certificates
  for (const s of rng.sample(world.suppliers, Math.min(count('aggregate', 2), world.suppliers.length))) {
    plans.push((program) => {
      const found = program.column(`incident(I, Site, _, _), site(Site, ${s.id})`, 'I');
      const siteKeys = world.sites.filter((x) => x.supplier === s.id).map((x) => keys.site(x.id));
      return {
        id: id('aggregate'),
        family: 'aggregate',
        hops: 2,
        question: `How many safety incidents were recorded at sites operated by ${s.name}?`,
        gold: { kind: 'number', items: [], number: found.length, distractors: [], distractorNumbers: [world.incidents.length].filter((x) => x !== found.length), display: String(found.length) },
        evidence: [...siteKeys, ...world.incidents.filter((i) => found.includes(i.id)).map((i) => keys.incident(i.id))],
        datalog: `incident(I, Site, _, _), site(Site, ${s.id})`,
        params: { supplier: s.name },
      };
    });
  }

  // --- absence: which suppliers lacked a valid certificate on a date ---
  for (let k = 0; k < count('absence', 4); k += 1) {
    const standard = rng.pick(['iso9001', 'iso27001'] as const);
    const on = pickDay(ymd(2022, 1, 1), ymd(2026, 3, 1));
    plans.push((program) => {
      const valid = new Set(program.column(`cert_valid_on(S, ${standard}, ${on})`, 'S').map(String));
      const lacking = world.suppliers.filter((s) => !valid.has(s.id));
      if (lacking.length === 0 || lacking.length === world.suppliers.length) return undefined;
      return {
        id: id('absence'),
        family: 'absence',
        hops: world.suppliers.length,
        question: `Which of ${world.organisation}'s contracted suppliers did not hold a valid ${STANDARD_NAMES[standard]} certificate on ${say(on)}?`,
        gold: {
          kind: 'set',
          items: lacking.map((s) => supplierSpellings(s.name)),
          distractors: world.suppliers.filter((s) => valid.has(s.id)).map((s) => s.name.replace(/ Pty Ltd$/, '')),
          distractorNumbers: [],
          display: lacking.map((s) => s.name).join('; '),
        },
        evidence: world.certificates.filter((c) => c.standard === standard).map((c) => keys.cert(c.supplier, c.standard, c.issued)),
        datalog: `supplier_name(S, _), \\+ cert_valid_on(S, ${standard}, ${on})`,
        params: { standard: STANDARD_NAMES[standard], date: on, organisation: world.organisation },
      };
    });
  }

  // --- authority: was an approval within the approver's delegated limit? ---
  for (const approval of rng.sample(world.approvals, Math.min(count('authority', 8), world.approvals.length))) {
    plans.push((program) => {
      const c = contract.get(approval.contract)!;
      const allowed = program.holds(`may_approve(${approval.person}, ${c.id}, ${approval.on})`);
      const term = world.roleTerms.find((t) => t.person === approval.person && t.from <= approval.on && approval.on < t.to)!;
      const authority = world.authorities.find((a) => a.role === term.role && a.from <= approval.on && approval.on < a.to);
      return {
        id: id('authority'),
        family: 'authority',
        hops: 4,
        question: `Was the approval of contract ${c.ref} within the approver's delegated authority on the day it was approved? Answer yes or no.`,
        gold: { kind: 'decision', items: [], allowed, distractors: [], distractorNumbers: [], display: allowed ? 'yes' : 'no' },
        evidence: [
          keys.approval(c.id),
          keys.value(c.id, 0),
          keys.roleStart(term.role, approval.person),
          ...(authority === undefined ? [] : [keys.authority(authority.role, authority.from)]),
        ],
        datalog: `may_approve(${approval.person}, ${c.id}, ${approval.on})`,
        params: { contract: c.ref },
      };
    });
  }

  // --- unanswerable: before the records start, or a contract that does not exist ---
  for (let k = 0; k < count('unanswerable', 4); k += 1) {
    const early = k % 2 === 0;
    plans.push((): ComposeQuestion => {
      if (early) {
        const role = rng.pick(world.roles);
        const on = addDays(WORLD_START, -rng.int(200, 900));
        return {
          id: id('unanswerable'),
          family: 'unanswerable',
          hops: 1,
          question: `Who was the ${role.name} of ${world.organisation} on ${say(on)}?`,
          gold: { kind: 'unknown', items: [], distractors: world.roleTerms.filter((t) => t.role === role.id).map((t) => person.get(t.person)!.last), distractorNumbers: [], display: 'unknown' },
          evidence: [],
          datalog: `holds_role_on(${role.id}, P, ${on})  % no row: before the first record`,
          params: { role: role.name, date: on, organisation: world.organisation },
        };
      }
      const used = new Set(world.contracts.map((c) => c.ref));
      let ref: string;
      do ref = `CN-${rng.int(1000, 9999)}`;
      while (used.has(ref));
      return {
        id: id('unanswerable'),
        family: 'unanswerable',
        hops: 1,
        question: `What was the value of contract ${ref} at signing?`,
        gold: { kind: 'unknown', items: [], distractors: [], distractorNumbers: [], display: 'unknown' },
        evidence: [],
        datalog: `contract(c${ref.slice(3)}, _, _, _)  % no such contract`,
        params: { contract: ref },
      };
    });
  }

  const program = new WorldProgram(world, askDays);
  for (const plan of plans) {
    const question = plan(program);
    if (question !== undefined) questions.push(question);
  }
  return questions;
}

function dayGap(from: Ymd, to: Ymd): number {
  // coarse is fine: pickDay only needs a day strictly inside the range
  const a = Math.floor(from / 10_000) * 365 + (Math.floor(from / 100) % 100) * 30 + (from % 100);
  const b = Math.floor(to / 10_000) * 365 + (Math.floor(to / 100) % 100) * 30 + (to % 100);
  return Math.max(2, b - a);
}

export { dateSpellings };
