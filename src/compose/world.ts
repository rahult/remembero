/**
 * A generated organisation, as Datalog facts.
 *
 * Everything the Compose benchmark asks about is here, and nothing else is: the documents are
 * written *from* these facts and the gold answers are *computed from* them by the engine, so a
 * label is right by construction. Names come from one of three disjoint pools (train, dev, test)
 * so a model trained on train worlds has never seen a test name.
 */

import { Rng } from './rng.js';
import { addDays, OPEN_END, ymd, type Ymd } from './dates.js';

export type Split = 'train' | 'dev' | 'test';

export interface Person {
  id: string;
  first: string;
  last: string;
  title: string;
}

export interface Role {
  id: string;
  name: string;
}

export interface RoleTerm {
  role: string;
  person: string;
  from: Ymd;
  to: Ymd;
}

export interface Authority {
  role: string;
  limit: number;
  from: Ymd;
  to: Ymd;
}

export interface Supplier {
  id: string;
  name: string;
}

export interface Site {
  id: string;
  name: string;
  supplier: string;
}

export interface Contract {
  id: string;
  ref: string;
  supplier: string;
  from: Ymd;
  to: Ymd;
}

/** A contract's value from a date; the first entry is the value it was signed at. */
export interface ContractValue {
  contract: string;
  value: number;
  effective: Ymd;
  amendment: number;
}

export interface Approval {
  contract: string;
  person: string;
  on: Ymd;
  /** How the minutes name the approver: by name, by initials, or by the role held that day. */
  namedAs: 'name' | 'initial' | 'role';
}

export interface Certificate {
  supplier: string;
  standard: 'iso9001' | 'iso27001';
  issued: Ymd;
  expires: Ymd;
}

export interface Incident {
  id: string;
  ref: string;
  site: string;
  on: Ymd;
  severity: number;
}

/** Written into the documents but never a fact: a proposal nobody adopted. */
export interface Proposal {
  role: string;
  limit: number;
  on: Ymd;
}

export interface World {
  seed: number;
  split: Split;
  organisation: string;
  people: Person[];
  roles: Role[];
  roleTerms: RoleTerm[];
  authorities: Authority[];
  suppliers: Supplier[];
  sites: Site[];
  contracts: Contract[];
  contractValues: ContractValue[];
  approvals: Approval[];
  certificates: Certificate[];
  incidents: Incident[];
  proposals: Proposal[];
}

export const STANDARD_NAMES: Record<Certificate['standard'], string> = {
  iso9001: 'ISO 9001',
  iso27001: 'ISO/IEC 27001',
};

export const WORLD_START: Ymd = ymd(2021, 1, 1);
export const WORLD_END: Ymd = ymd(2026, 6, 30);

const ROLES: Role[] = [
  { id: 'cro', name: 'Chief Risk Officer' },
  { id: 'cfo', name: 'Chief Financial Officer' },
  { id: 'coo', name: 'Chief Operating Officer' },
  { id: 'hop', name: 'Head of Procurement' },
  { id: 'gc', name: 'General Counsel' },
  { id: 'hos', name: 'Head of Safety' },
];

/** Roles that can be given authority to approve contracts. */
const APPROVING_ROLES = ['cro', 'cfo', 'coo', 'hop'];

const FIRST = [
  'Amara', 'Tobias', 'Ingrid', 'Rafael', 'Mei', 'Declan', 'Priya', 'Soren', 'Lucia', 'Kwame',
  'Hana', 'Matteo', 'Elif', 'Callum', 'Noor', 'Anders', 'Yara', 'Felix', 'Ines', 'Rohan',
  'Saoirse', 'Tariq', 'Greta', 'Emeka', 'Linnea', 'Marco', 'Aiko', 'Bastian', 'Zara', 'Oskar',
  'Nadia', 'Henrik', 'Leilani', 'Dmitri', 'Anouk', 'Kofi', 'Freya', 'Joaquin', 'Signe', 'Arjun',
  'Maeve', 'Hugo', 'Esme', 'Tomas', 'Ayesha', 'Pieter', 'Keira', 'Viktor', 'Roisin', 'Idris',
  'Clara', 'Emil', 'Farah', 'Lorcan', 'Ottilie', 'Sanjay', 'Vera', 'Wiremu', 'Astrid', 'Bruno',
];
const LAST = [
  'Osei', 'Lindqvist', 'Moreau', 'Achterberg', 'Nakashima', 'Fairbairn', 'Raghavan', 'Castellanos',
  'Okonkwo', 'Brennan', 'Varga', 'Halloran', 'Sandoval', 'Petrakis', 'Whitcombe', 'Adeyemi',
  'Kowalczyk', 'Tremblay', 'Haddad', 'Oyelaran', 'Szabo', 'Ferreira', 'Mbeki', 'Novak',
  'Engstrom', 'Quigley', 'Dubois', 'Ivanova', 'Kerrigan', 'Solberg', 'Yilmaz', 'Carvalho',
  'Lachance', 'Ostrowski', 'Pemberton', 'Rautio', 'Balogun', 'Vasquez', 'Holmqvist', 'Delacroix',
  'Obi', 'Strand', 'Marchetti', 'Nwosu', 'Galloway', 'Ekberg', 'Zielinski', 'Andrade',
  'Thorsen', 'Mahlangu', 'Beaumont', 'Kaczmarek', 'Ferrante', 'Lundgren', 'Okafor', 'Rinaldi',
  'Sorensen', 'Tanaka', 'Uddin', 'Weatherby',
];
const TITLES = ['Dr', 'Ms', 'Mr', 'Mx', 'Prof'];
const SUPPLIER_STEMS = [
  'Halstead', 'Brightwater', 'Kestrel', 'Northgate', 'Aldercroft', 'Pennant', 'Silverline',
  'Corvid', 'Tamarind', 'Ashdown', 'Blackmoor', 'Greyfield', 'Larkspur', 'Meridian', 'Oakhurst',
  'Quarrystone', 'Redfern', 'Sablewood', 'Thornbury', 'Umberly', 'Vantage', 'Westmere',
  'Yarrowby', 'Zephyrine', 'Cobalt Ridge', 'Driftwood', 'Emberton', 'Foxglove', 'Glenrock',
  'Harrowgate', 'Ironbark', 'Juniper Bay', 'Kingsmead', 'Lowmoor', 'Marlowe', 'Nettlefield',
];
const SUPPLIER_TRADES = ['Logistics', 'Facilities', 'Data Services', 'Engineering', 'Freight', 'Security', 'Catering', 'Systems'];
const TOWNS = [
  'Tamworth', 'Ballarat', 'Wagga', 'Dubbo', 'Bendigo', 'Orange', 'Mildura', 'Toowoomba', 'Geelong',
  'Launceston', 'Cairns', 'Albury', 'Bathurst', 'Shepparton', 'Rockhampton', 'Bunbury', 'Gympie',
  'Warrnambool', 'Goulburn', 'Horsham', 'Kalgoorlie', 'Lismore', 'Mackay', 'Nowra', 'Parkes',
  'Queanbeyan', 'Sale', 'Taree', 'Wodonga', 'Yeppoon', 'Armidale', 'Broome', 'Cessnock',
];
const SITE_KINDS = ['depot', 'warehouse', 'data centre', 'workshop', 'yard'];
export const ORGANISATIONS = ['Carrow Group', 'Estuary Holdings', 'Ferrous Pacific', 'Lumen Utilities', 'Tidewater Transit', 'Marlin Health'];

/** Split a pool into three disjoint thirds, one per split. */
function third<T>(values: readonly T[], split: Split): T[] {
  const index = split === 'train' ? 0 : split === 'dev' ? 1 : 2;
  return values.filter((_value, i) => i % 3 === index);
}

export interface WorldOptions {
  /** Force the organisation's name (a sister organisation must not share the main one's). */
  organisation?: string;
  /** Contract references already used elsewhere in the same haystack. */
  avoidRefs?: ReadonlySet<string>;
  people?: number;
  suppliers?: number;
  contracts?: number;
  incidents?: number;
}

export function generateWorld(seed: number, split: Split, options: WorldOptions = {}): World {
  const rng = new Rng(seed);
  const firstPool = third(FIRST, split);
  const lastPool = third(LAST, split);
  const stemPool = third(SUPPLIER_STEMS, split);
  const townPool = third(TOWNS, split);

  const personCount = options.people ?? 16;
  const lasts = rng.sample(lastPool, Math.min(personCount, lastPool.length));
  const people: Person[] = lasts.map((last, i) => ({
    id: `p${String(i + 1).padStart(2, '0')}`,
    first: rng.pick(firstPool),
    last,
    title: rng.pick(TITLES),
  }));

  // two or three holders per role over the world's lifetime; nobody holds two roles at once
  const roleTerms: RoleTerm[] = [];
  const available = rng.shuffle(people.map((p) => p.id));
  for (const role of ROLES) {
    const holders = rng.int(2, 3);
    let from = WORLD_START;
    for (let h = 0; h < holders; h += 1) {
      const person = available.shift();
      if (person === undefined) break;
      const last = h === holders - 1;
      const to = last ? OPEN_END : addDays(from, rng.int(420, 820));
      roleTerms.push({ role: role.id, person, from, to });
      if (last || to > WORLD_END) break;
      // an interregnum now and then: a week or two with no holder
      from = rng.chance(0.3) ? addDays(to, rng.int(7, 20)) : to;
    }
  }

  // authority to approve contracts, by role, in one or two periods with different limits
  const authorities: Authority[] = [];
  for (const role of APPROVING_ROLES) {
    const base = rng.pick([100_000, 150_000, 250_000, 500_000, 750_000]);
    const change = addDays(WORLD_START, rng.int(500, 1_300));
    authorities.push({ role, limit: base, from: WORLD_START, to: change });
    if (rng.chance(0.75)) {
      authorities.push({ role, limit: base * rng.pick([2, 3, 0.5]), from: change, to: OPEN_END });
    }
  }
  const proposals: Proposal[] = APPROVING_ROLES.slice(0, 3).map((role) => ({
    role,
    limit: rng.pick([1_000_000, 1_500_000, 2_000_000]),
    on: addDays(WORLD_START, rng.int(200, 1_800)),
  }));

  const supplierCount = options.suppliers ?? 8;
  const stems = rng.sample(stemPool, Math.min(supplierCount, stemPool.length));
  const suppliers: Supplier[] = stems.map((stem, i) => ({
    id: `s${String(i + 1).padStart(2, '0')}`,
    name: `${stem} ${rng.pick(SUPPLIER_TRADES)} Pty Ltd`,
  }));
  const towns = rng.shuffle(townPool);
  const sites: Site[] = [];
  for (const supplier of suppliers) {
    for (let k = 0; k < rng.int(1, 2); k += 1) {
      const town = towns.shift();
      if (town === undefined) break;
      sites.push({ id: `site${sites.length + 1}`, name: `${town} ${rng.pick(SITE_KINDS)}`, supplier: supplier.id });
    }
  }

  const contracts: Contract[] = [];
  const contractValues: ContractValue[] = [];
  const approvals: Approval[] = [];
  const usedRefs = new Set<string>();
  for (let i = 0; i < (options.contracts ?? 14); i += 1) {
    let ref: string;
    do ref = `CN-${rng.int(1000, 9999)}`;
    while (usedRefs.has(ref) || options.avoidRefs?.has(ref));
    usedRefs.add(ref);
    const from = addDays(WORLD_START, rng.int(60, 1_700));
    const contract: Contract = {
      id: `c${ref.slice(3)}`,
      ref,
      supplier: rng.pick(suppliers).id,
      from,
      to: addDays(from, rng.pick([365, 730, 1_095])),
    };
    contracts.push(contract);
    // approved a few days before it starts, by whoever held an approving role that day; the
    // value agreed at approval is in force from the approval date
    const on = addDays(from, -rng.int(3, 20));
    const signed = rng.int(8, 160) * 5_000;
    contractValues.push({ contract: contract.id, value: signed, effective: on, amendment: 0 });
    let effective = from;
    for (let a = 1; a <= (rng.chance(0.45) ? rng.int(1, 2) : 0); a += 1) {
      effective = addDays(effective, rng.int(90, 300));
      const factor = rng.pick([1.1, 1.25, 0.9, 1.4]);
      contractValues.push({
        contract: contract.id,
        value: Math.round((signed * factor) / 5_000) * 5_000,
        effective,
        amendment: a,
      });
    }
    const candidates = roleTerms.filter((t) => APPROVING_ROLES.includes(t.role) && t.from <= on && on < t.to);
    if (candidates.length > 0) {
      approvals.push({
        contract: contract.id,
        person: rng.pick(candidates).person,
        on,
        namedAs: rng.pick(['name', 'initial', 'role'] as const),
      });
    }
  }

  const certificates: Certificate[] = [];
  for (const supplier of suppliers) {
    for (const standard of ['iso9001', 'iso27001'] as const) {
      if (standard === 'iso27001' && rng.chance(0.4)) continue;
      let issued = addDays(WORLD_START, rng.int(-400, 300));
      while (issued < WORLD_END) {
        const expires = addDays(issued, rng.pick([730, 1_095]));
        certificates.push({ supplier: supplier.id, standard, issued, expires });
        // most renew on time; some lapse for months before renewing, some never renew
        if (rng.chance(0.2)) break;
        issued = rng.chance(0.35) ? addDays(expires, rng.int(30, 240)) : expires;
      }
    }
  }

  const incidents: Incident[] = [];
  for (let i = 0; i < (options.incidents ?? 18); i += 1) {
    incidents.push({
      id: `inc${i + 1}`,
      ref: `INC-${String(rng.int(100, 999))}${String.fromCharCode(65 + (i % 26))}`,
      site: rng.pick(sites).id,
      on: addDays(WORLD_START, rng.int(30, 1_990)),
      severity: rng.int(1, 5),
    });
  }

  return {
    seed,
    split,
    organisation: options.organisation ?? rng.pick(ORGANISATIONS),
    people,
    roles: ROLES,
    roleTerms,
    authorities,
    suppliers,
    sites,
    contracts,
    contractValues,
    approvals,
    certificates,
    incidents,
    proposals,
  };
}

export function personName(person: Person): string {
  return `${person.first} ${person.last}`;
}

export function roleName(world: World, roleId: string): string {
  return world.roles.find((role) => role.id === roleId)!.name;
}
