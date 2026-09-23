/**
 * The world written as documents. Each line carries the evidence keys of the facts it states, so
 * once lines are paginated every fact has a page, and every question's evidence pages are known.
 *
 * Variety is deliberate and seeded: dates in four formats, money in three, approvers named by
 * full name, by initial, or only by the role they held that day. Proposals are written in the
 * same minutes as resolutions but state nothing — a reader that counts them is wrong.
 */

import { Rng } from './rng.js';
import { formatDate, OPEN_END, type DateStyle, type Ymd } from './dates.js';
import { personName, roleName, STANDARD_NAMES, type Person, type World } from './world.js';

export interface Line {
  text: string;
  /** Evidence keys of the facts this line states; empty for headings and proposals. */
  keys: string[];
  /**
   * The line's facts in the extraction schema, as a perfect extractor would write them (names as
   * the line writes them, without honorifics; dates as yyyymmdd; money in whole dollars). The
   * training targets for a local extractor come from here.
   */
  facts?: SchemaFact[];
}

export interface SchemaFact {
  predicate: string;
  args: Array<string | number>;
}

export interface Section {
  document: string;
  heading: string;
  lines: Line[];
}

export interface RenderedPage {
  text: string;
  keys: string[];
  facts: SchemaFact[];
}

export const keys = {
  person: (id: string) => `person:${id}`,
  roleStart: (role: string, person: string) => `role:${role}:${person}:from`,
  roleEnd: (role: string, person: string) => `role:${role}:${person}:to`,
  authority: (role: string, from: Ymd) => `authority:${role}:${from}`,
  contract: (id: string) => `contract:${id}`,
  value: (id: string, amendment: number) => `value:${id}:${amendment}`,
  approval: (id: string) => `approval:${id}`,
  site: (id: string) => `site:${id}`,
  cert: (supplier: string, standard: string, issued: Ymd) => `cert:${supplier}:${standard}:${issued}`,
  incident: (id: string) => `incident:${id}`,
};

export function formatMoney(amount: number, style: 'full' | 'short' | 'symbol'): string {
  if (style === 'short' && amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(amount % 10_000 === 0 ? 2 : 3).replace(/\.?0+$/, '')} million`;
  if (style === 'symbol') return `A$${amount.toLocaleString('en-AU')}`;
  return `AUD ${amount.toLocaleString('en-AU')}`;
}

function honorific(person: Person): string {
  return `${person.title} ${person.first} ${person.last}`;
}

export function renderWorld(world: World, seed = world.seed): Section[] {
  const rng = new Rng(seed ^ 0x5eed);
  const person = new Map(world.people.map((p) => [p.id, p]));
  const supplier = new Map(world.suppliers.map((s) => [s.id, s]));
  const site = new Map(world.sites.map((s) => [s.id, s]));
  const contract = new Map(world.contracts.map((c) => [c.id, c]));
  const org = world.organisation;
  const sections: Section[] = [];
  const date = (d: Ymd, style?: DateStyle) => formatDate(d, style ?? rng.pick(['long', 'long', 'short', 'iso'] as const));
  const money = (n: number) => formatMoney(n, rng.pick(['full', 'full', 'short', 'symbol'] as const));

  // --- board minutes: one meeting per date on which something was decided ---
  type Item = { on: Ymd; line: Line };
  const items: Item[] = [];
  for (const term of world.roleTerms) {
    const p = person.get(term.person)!;
    const role = roleName(world, term.role);
    items.push({
      on: term.from,
      line: {
        text: rng.pick([
          `The Board confirmed the appointment of ${honorific(p)} as ${role} with effect from ${date(term.from)}.`,
          `${honorific(p)} was appointed ${role}, effective ${date(term.from)}.`,
          `The Board noted that ${personName(p)} assumes the role of ${role} from ${date(term.from)}.`,
        ]),
        keys: [keys.person(p.id), keys.roleStart(term.role, p.id)],
        facts: [{ predicate: 'appointed', args: [org, personName(p), role, term.from] }],
      },
    });
    if (term.to !== OPEN_END) {
      // both sentences are built before one is picked, as they always were: each draws a date
      // style, and changing the number of draws would re-render every existing benchmark
      const variants = [
        `${personName(p)} stepped down as ${role} on ${date(term.to)}.`,
        `The Board recorded that ${p.title} ${p.last} ceased to hold the office of ${role} on ${date(term.to)}.`,
      ];
      const text = rng.pick(variants);
      const fullName = text === variants[0];
      items.push({
        on: term.to,
        line: {
          text,
          keys: [keys.person(p.id), keys.roleEnd(term.role, p.id)],
          facts: [{ predicate: 'ceased', args: [org, fullName ? personName(p) : p.last, role, term.to] }],
        },
      });
    }
  }
  for (const a of world.authorities) {
    const role = roleName(world, a.role);
    const until = a.to === OPEN_END ? 'until further notice' : `until ${date(a.to)}`;
    items.push({
      on: a.from,
      line: {
        text: rng.pick([
          `RESOLVED that the ${role} may approve contracts with a value of up to ${money(a.limit)}, from ${date(a.from)} ${until}.`,
          `The Board delegated to the ${role} authority to approve contracts not exceeding ${money(a.limit)}, effective ${date(a.from)} and ${until === 'until further notice' ? 'continuing until further notice' : until}.`,
        ]),
        keys: [keys.authority(a.role, a.from)],
        facts: [{ predicate: 'authority', args: [org, role, a.limit, a.from, a.to] }],
      },
    });
  }
  for (const proposal of world.proposals) {
    items.push({
      on: proposal.on,
      line: {
        text: rng.pick([
          `It was proposed that the ${roleName(world, proposal.role)}'s approval limit be raised to ${money(proposal.limit)}. No resolution was passed and the matter was deferred.`,
          `A motion to increase the ${roleName(world, proposal.role)}'s contract approval limit to ${money(proposal.limit)} was discussed but not put to a vote.`,
        ]),
        keys: [],
      },
    });
  }
  for (const approval of world.approvals) {
    const c = contract.get(approval.contract)!;
    const p = person.get(approval.person)!;
    const heldRole = world.roleTerms.find((t) => t.person === p.id && t.from <= approval.on && approval.on < t.to)!;
    const who =
      approval.namedAs === 'name'
        ? personName(p)
        : approval.namedAs === 'initial'
          ? `${p.first[0]}. ${p.last}`
          : `the ${roleName(world, heldRole.role)}`;
    items.push({
      on: approval.on,
      line: {
        text: rng.pick([
          `Contract ${c.ref} with ${supplier.get(c.supplier)!.name} was approved by ${who} on ${date(approval.on)}.`,
          `${who[0]!.toUpperCase()}${who.slice(1)} approved contract ${c.ref} (${supplier.get(c.supplier)!.name}) on ${date(approval.on)}.`,
        ]),
        // a role-named approval also needs the appointment that says who held the role that day
        keys: [keys.approval(c.id), ...(approval.namedAs === 'role' ? [keys.roleStart(heldRole.role, p.id)] : [])],
        facts: [{ predicate: 'approved', args: [org, c.ref, who.replace(/^the /, ''), approval.on] }],
      },
    });
  }
  const byMeeting = new Map<Ymd, Line[]>();
  for (const item of items.sort((a, b) => a.on - b.on)) {
    // a meeting is held on the first of the month the item falls in
    const meeting = Math.floor(item.on / 100) * 100 + 1;
    byMeeting.set(meeting, [...(byMeeting.get(meeting) ?? []), item.line]);
  }
  for (const [meeting, lines] of byMeeting) {
    sections.push({
      document: 'Board minutes',
      heading: `${org} — Minutes of the Board meeting held ${date(meeting, 'long')}`,
      lines: [
        { text: `Present: the Chair and directors. Apologies were noted. The minutes of the previous meeting were confirmed.`, keys: [] },
        ...lines,
      ],
    });
  }

  // --- contract register: a table, split across the document ---
  const registerStyle = rng.pick(['slash', 'iso'] as const);
  const registerRows: Line[] = world.contracts.map((c) => {
    const signed = world.contractValues.find((v) => v.contract === c.id && v.amendment === 0)!;
    return {
      text: `${c.ref} | ${supplier.get(c.supplier)!.name} | ${formatDate(c.from, registerStyle)} | ${formatDate(c.to, registerStyle)} | ${formatMoney(signed.value, 'full')}`,
      keys: [keys.contract(c.id), keys.value(c.id, 0)],
      facts: [{ predicate: 'contract', args: [org, c.ref, supplier.get(c.supplier)!.name, c.from, c.to, signed.value] }],
    };
  });
  const half = Math.ceil(registerRows.length / 2);
  for (const [part, rows] of [registerRows.slice(0, half), registerRows.slice(half)].entries()) {
    sections.push({
      document: 'Contract register',
      heading: `${org} — Contract register, part ${part + 1} of 2 (reference | supplier | start | end | value at signing)`,
      lines: rows,
    });
  }

  // --- amendments ---
  const amendments = world.contractValues.filter((v) => v.amendment > 0);
  if (amendments.length > 0) {
    sections.push({
      document: 'Contract amendments',
      heading: `${org} — Schedule of contract variations`,
      lines: amendments.map((v) => {
        const c = contract.get(v.contract)!;
        return {
          text: rng.pick([
            `Amendment ${v.amendment} to ${c.ref}, effective ${date(v.effective, 'iso')}, revises the contract value to ${money(v.value)}.`,
            `Variation no. ${v.amendment} (${c.ref}, ${supplier.get(c.supplier)!.name}): from ${date(v.effective)} the contract value is ${money(v.value)}.`,
          ]),
          keys: [keys.value(c.id, v.amendment)],
          facts: [{ predicate: 'amendment', args: [org, c.ref, v.amendment, v.effective, v.value] }],
        };
      }),
    });
  }

  // --- sites ---
  sections.push({
    document: 'Supplier sites',
    heading: `${org} — Operating sites of contracted suppliers`,
    lines: world.sites.map((s) => ({
      text: rng.pick([
        `The ${s.name} is operated by ${supplier.get(s.supplier)!.name}.`,
        `${supplier.get(s.supplier)!.name} operates the ${s.name}.`,
      ]),
      keys: [keys.site(s.id)],
      facts: [{ predicate: 'site_operator', args: [org, s.name, supplier.get(s.supplier)!.name] }],
    })),
  });

  // --- certification register ---
  sections.push({
    document: 'Certification register',
    heading: `${org} — Supplier certification register (supplier | standard | issued | expires)`,
    lines: world.certificates.map((c) => ({
      text: `${supplier.get(c.supplier)!.name} | ${STANDARD_NAMES[c.standard]} | ${formatDate(c.issued, 'slash')} | ${formatDate(c.expires, 'slash')}`,
      keys: [keys.cert(c.supplier, c.standard, c.issued)],
      facts: [{ predicate: 'certificate', args: [org, supplier.get(c.supplier)!.name, STANDARD_NAMES[c.standard], c.issued, c.expires] }],
    })),
  });

  // --- incident log ---
  sections.push({
    document: 'Incident log',
    heading: `${org} — Safety incident log (reference | date | site | severity 1-5)`,
    lines: [...world.incidents]
      .sort((a, b) => a.on - b.on)
      .map((i) => ({
        text: `${i.ref} | ${formatDate(i.on, 'short')} | ${site.get(i.site)!.name} | ${i.severity}`,
        keys: [keys.incident(i.id)],
        facts: [{ predicate: 'incident', args: [org, i.ref, i.on, site.get(i.site)!.name, i.severity] }],
      })),
  });

  return sections;
}

/** Pages of about `maxChars`, a section never shares a page with another. */
export function paginate(sections: readonly Section[], maxChars = 2_400): RenderedPage[] {
  const pages: RenderedPage[] = [];
  for (const section of sections) {
    let text = `${section.heading}\n\n`;
    let pageKeys: string[] = [];
    let pageFacts: SchemaFact[] = [];
    let continued = false;
    for (const line of section.lines) {
      if (text.length + line.text.length + 1 > maxChars && pageKeys.length + (continued ? 1 : 0) > 0) {
        pages.push({ text, keys: [...new Set(pageKeys)], facts: pageFacts });
        text = `${section.heading} (continued)\n\n`;
        pageKeys = [];
        pageFacts = [];
        continued = true;
      }
      text += `${line.text}\n`;
      pageKeys.push(...line.keys);
      pageFacts.push(...(line.facts ?? []));
    }
    pages.push({ text, keys: [...new Set(pageKeys)], facts: pageFacts });
  }
  return pages;
}
