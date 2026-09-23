/**
 * A world's facts in the extraction schema, written the way the documents write them — what a
 * perfect extractor would return. Used to check the engine's rules and query templates in
 * isolation (they should score 100% on it) and as the reference for extraction precision/recall.
 */

import { OPEN_END } from './dates.js';
import { paginate, renderWorld } from './render.js';
import type { Fact } from './extract.js';
import { personName, roleName, STANDARD_NAMES, type World } from './world.js';

/**
 * The truth as the documents state it: the renderer's own per-line facts, so a resignation written
 * with the full name is recorded with the full name and one written "Mx Brennan" with the surname.
 */
export function renderedSchemaFacts(world: World): Fact[] {
  return paginate(renderWorld(world)).flatMap((page) => page.facts);
}

export function worldAsSchemaFacts(world: World): Fact[] {
  const org = world.organisation;
  const person = new Map(world.people.map((p) => [p.id, p]));
  const supplier = new Map(world.suppliers.map((s) => [s.id, s.name]));
  const site = new Map(world.sites.map((s) => [s.id, s.name]));
  const contract = new Map(world.contracts.map((c) => [c.id, c]));
  const facts: Fact[] = [];
  for (const t of world.roleTerms) {
    const p = person.get(t.person)!;
    facts.push({ predicate: 'appointed', args: [personName(p), roleName(world, t.role), t.from] });
    if (t.to !== OPEN_END) facts.push({ predicate: 'ceased', args: [p.last, roleName(world, t.role), t.to] });
  }
  for (const a of world.authorities) facts.push({ predicate: 'authority', args: [roleName(world, a.role), a.limit, a.from, a.to] });
  for (const a of world.approvals) {
    const p = person.get(a.person)!;
    const term = world.roleTerms.find((t) => t.person === p.id && t.from <= a.on && a.on < t.to)!;
    const written =
      a.namedAs === 'name' ? personName(p) : a.namedAs === 'initial' ? `${p.first[0]}. ${p.last}` : a.namedAs === 'surname' ? p.last : roleName(world, term.role);
    facts.push({ predicate: 'approved', args: [contract.get(a.contract)!.ref, written, a.on] });
  }
  for (const c of world.contracts) {
    const signed = world.contractValues.find((v) => v.contract === c.id && v.amendment === 0)!;
    facts.push({ predicate: 'contract', args: [c.ref, supplier.get(c.supplier)!, c.from, c.to, signed.value] });
  }
  for (const v of world.contractValues.filter((x) => x.amendment > 0)) {
    facts.push({ predicate: 'amendment', args: [contract.get(v.contract)!.ref, v.amendment, v.effective, v.value] });
  }
  for (const s of world.sites) facts.push({ predicate: 'site_operator', args: [s.name, supplier.get(s.supplier)!] });
  for (const c of world.certificates) facts.push({ predicate: 'certificate', args: [supplier.get(c.supplier)!, STANDARD_NAMES[c.standard], c.issued, c.expires] });
  for (const i of world.incidents) facts.push({ predicate: 'incident', args: [i.ref, i.on, site.get(i.site)!, i.severity] });
  return facts.map((f) => ({ predicate: f.predicate, args: [org, ...f.args] }));
}
