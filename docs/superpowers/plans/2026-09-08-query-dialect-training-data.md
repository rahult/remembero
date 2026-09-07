# Query-Dialect Training Data Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a few thousand execution-verified (question → flat Datalog) examples as Tinker-ready JSONL, plus the Python recipe and proxy needed to train and evaluate a small model on them.

**Architecture:** A deterministic pipeline under `src/training/`: seeded synthetic worlds → question templates emitting flat-dialect programs in both argument directions → execution verification through the engine (closure synthesis included) → Luna paraphrasing with entity-preservation checks and a disk cache → JSONL export in the `{"messages":[...]}` shape `tinker_cookbook.supervised.data.FromConversationFileBuilder` reads. Python files under `benchmarks/tinker/` launch training and expose Tinker sampling as an Ollama-compatible endpoint so the existing agent-boundary harness evaluates the result unchanged.

**Tech Stack:** TypeScript (ESM, Node 22+), vitest, the in-repo Datalog engine (`src/engine`), `OpenRouterClient` (`src/llm/client.ts`), Python 3.11 + `tinker-cookbook` for training/serving.

**Spec:** `docs/superpowers/specs/2026-09-08-query-dialect-training-data-design.md`

## Global Constraints

- Training targets never contain a recursive rule; chains use `p_plus`.
- Training vocabulary never uses the benchmark predicates `works_on, reports_to, status, blocker, waits_on, prefers_meeting, review_slot, promised_update`.
- Same seed → byte-identical worlds and candidates.
- Every chain template is emitted with the anchor in both argument positions, recorded as `direction`.
- Paraphrase model `openai/gpt-5.6-luna`; cache under `data/training/cache/`; failures fall back to the templated question.
- JSONL lines are `{"messages":[{"role":"system"|"user"|"assistant","content":string}]}`.
- Training launch is manual and confirmed each time; never in npm scripts or CI.
- Test command: `npx vitest run tests/training.test.ts`. Format new files with `npx prettier --write`.

---

### Task 1: Seeded RNG and synthetic worlds

**Files:**
- Create: `src/training/rng.ts`
- Create: `src/training/worlds.ts`
- Test: `tests/training.test.ts`

**Interfaces:**
- Produces:
  ```ts
  // rng.ts
  export interface Rng { next(): number; int(maxExclusive: number): number; pick<T>(items: readonly T[]): T; shuffle<T>(items: readonly T[]): T[]; }
  export function createRng(seed: number): Rng;
  // worlds.ts
  export type RelationKind = 'hierarchy' | 'dependency' | 'attribute' | 'membership' | 'schedule';
  export type Orientation = 'child-first' | 'parent-first'; // edge(child, parent) vs edge(parent, child)
  export interface Relation { name: string; kind: RelationKind; args: string[]; orientation?: Orientation; upPhrase?: string; downPhrase?: string; }
  export interface World { id: string; seed: number; relations: Relation[]; entities: string[]; facts: string[]; }
  export const BENCHMARK_PREDICATES: readonly string[];
  export function generateWorld(seed: number): World;
  export function worldClauses(world: World): Clause[];
  export function schemaListing(world: World): string;
  export function relationsOfKind(world: World, kind: RelationKind): Relation[];
  ```

- [ ] **Step 1: Write the failing tests**

```ts
// tests/training.test.ts
import { describe, expect, it } from 'vitest';
import { createRng } from '../src/training/rng.js';
import {
  BENCHMARK_PREDICATES,
  generateWorld,
  relationsOfKind,
  schemaListing,
  worldClauses,
} from '../src/training/worlds.js';
import { evaluate, parseQuery } from '../src/engine/index.js';

describe('training: rng', () => {
  it('is deterministic under a seed', () => {
    const a = createRng(7); const b = createRng(7);
    expect([a.int(100), a.int(100), a.int(100)]).toEqual([b.int(100), b.int(100), b.int(100)]);
  });
});

describe('training: worlds', () => {
  it('produces byte-identical worlds for the same seed', () => {
    expect(JSON.stringify(generateWorld(3))).toBe(JSON.stringify(generateWorld(3)));
    expect(JSON.stringify(generateWorld(3))).not.toBe(JSON.stringify(generateWorld(4)));
  });

  it('satisfies the structural invariants on twenty seeds', () => {
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      const edges = [...relationsOfKind(world, 'hierarchy'), ...relationsOfKind(world, 'dependency')];
      expect(edges.length).toBeGreaterThan(0);
      expect(relationsOfKind(world, 'attribute').length).toBeGreaterThan(0);
      expect(relationsOfKind(world, 'membership').length).toBeGreaterThan(0);
      expect(world.entities.length).toBeGreaterThanOrEqual(12);
      expect(world.entities.length).toBeLessThanOrEqual(30);
      for (const relation of world.relations) {
        expect(BENCHMARK_PREDICATES).not.toContain(relation.name);
      }
      // every edge relation has a chain of at least three hops somewhere
      const clauses = worldClauses(world);
      for (const edge of edges) {
        const three = evaluate(clauses, parseQuery(`${edge.name}(A, B), ${edge.name}(B, C), ${edge.name}(C, D).`));
        expect(three.length, `${edge.name} in seed ${seed}`).toBeGreaterThan(0);
      }
      // at least one attribute relation leaves some entity without a value
      const gaps = relationsOfKind(world, 'attribute').some((attr) =>
        world.entities.some((entity) => evaluate(clauses, parseQuery(`${attr.name}(${entity}, V).`)).length === 0),
      );
      expect(gaps).toBe(true);
    }
  });

  it('lists the schema with argument names', () => {
    const world = generateWorld(1);
    const listing = schemaListing(world);
    for (const relation of world.relations) {
      expect(listing).toContain(`${relation.name}(${relation.args.join(', ')})`);
    }
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/rng.js`.

- [ ] **Step 3: Implement rng.ts**

```ts
// src/training/rng.ts
/** Deterministic mulberry32 generator so a seed reproduces a whole data set. */
export interface Rng {
  next(): number;
  int(maxExclusive: number): number;
  pick<T>(items: readonly T[]): T;
  shuffle<T>(items: readonly T[]): T[];
}

export function createRng(seed: number): Rng {
  let state = seed >>> 0 || 0x9e3779b9;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const int = (maxExclusive: number) => Math.floor(next() * maxExclusive);
  return {
    next,
    int,
    pick: (items) => {
      if (items.length === 0) throw new Error('pick from empty list');
      return items[int(items.length)];
    },
    shuffle: (items) => {
      const copy = [...items];
      for (let i = copy.length - 1; i > 0; i -= 1) {
        const j = int(i + 1);
        [copy[i], copy[j]] = [copy[j], copy[i]];
      }
      return copy;
    },
  };
}
```

- [ ] **Step 4: Implement worlds.ts**

```ts
// src/training/worlds.ts
import { type Clause, parseProgram } from '../engine/index.js';
import { createRng, type Rng } from './rng.js';

export type RelationKind = 'hierarchy' | 'dependency' | 'attribute' | 'membership' | 'schedule';
/** child-first: edge(child, parent) like reports_to; parent-first: edge(parent, child) like manages. */
export type Orientation = 'child-first' | 'parent-first';

export interface Relation {
  name: string;
  kind: RelationKind;
  /** Capitalized argument names shown in the schema listing. */
  args: string[];
  orientation?: Orientation;
  /** Natural phrase for "things above/upstream of X" and "things below/downstream of X". */
  upPhrase?: string;
  downPhrase?: string;
}

export interface World {
  id: string;
  seed: number;
  relations: Relation[];
  entities: string[];
  /** Serialized ground facts, one per entry, ending in a period. */
  facts: string[];
}

export const BENCHMARK_PREDICATES = [
  'works_on', 'reports_to', 'status', 'blocker', 'waits_on',
  'prefers_meeting', 'review_slot', 'promised_update',
] as const;

interface EdgeVocab { name: string; args: [string, string]; orientation: Orientation; up: string; down: string }
interface AttrVocab { name: string; args: [string, string]; values: string[] }
interface GroupVocab { name: string; args: [string, string]; groups: string[] }
interface ScheduleVocab { name: string; args: [string, string, string]; slots: string[]; windows: string[] }

interface Theme {
  entities: string[];
  hierarchy: EdgeVocab[];
  dependency: EdgeVocab[];
  attributes: AttrVocab[];
  memberships: GroupVocab[];
  schedules: ScheduleVocab[];
}

const THEMES: Theme[] = [
  {
    entities: ['ana', 'bo', 'cleo', 'dev', 'esi', 'femi', 'gus', 'hana', 'ivo', 'jun', 'kai', 'lena', 'milo', 'nia', 'omar', 'pia', 'quin', 'rosa', 'sam', 'tal', 'uma', 'vik', 'wren', 'xan', 'yara', 'zed', 'aiko', 'bram', 'cyd', 'dara'],
    hierarchy: [
      { name: 'manages', args: ['Manager', 'Person'], orientation: 'parent-first', up: 'everyone above {x} in the management chain', down: 'everyone who ultimately reports to {x}' },
      { name: 'mentors', args: ['Mentor', 'Mentee'], orientation: 'parent-first', up: "{x}'s mentors, their mentors, and so on", down: 'everyone {x} mentors directly or indirectly' },
      { name: 'answers_to', args: ['Person', 'Lead'], orientation: 'child-first', up: 'every lead above {x}', down: 'everyone under {x}, at any depth' },
    ],
    dependency: [
      { name: 'hands_off_to', args: ['Person', 'Next'], orientation: 'child-first', up: 'everyone downstream of {x} in the handoff chain', down: 'everyone whose work eventually reaches {x}' },
    ],
    attributes: [
      { name: 'prefers', args: ['Person', 'Window'], values: ['early', 'late', 'midday'] },
      { name: 'speaks', args: ['Person', 'Language'], values: ['spanish', 'hindi', 'french', 'mandarin'] },
      { name: 'level', args: ['Person', 'Level'], values: ['junior', 'senior', 'staff'] },
    ],
    memberships: [
      { name: 'member_of', args: ['Person', 'Guild'], groups: ['platform', 'growth', 'data', 'design'] },
      { name: 'assigned_to', args: ['Person', 'Squad'], groups: ['alpha', 'bravo', 'delta'] },
    ],
    schedules: [
      { name: 'standup', args: ['Squad', 'Day', 'Window'], slots: ['monday', 'tuesday', 'thursday'], windows: ['early', 'late'] },
    ],
  },
  {
    entities: ['auth', 'billing', 'cache', 'catalog', 'checkout', 'cdn', 'db_primary', 'db_replica', 'email', 'gateway', 'inventory', 'ledger', 'logging', 'metrics', 'notify', 'orders', 'payments', 'pricing', 'queue', 'ratings', 'search', 'sessions', 'shipping', 'storage', 'tax', 'users', 'vault', 'webhooks'],
    hierarchy: [
      { name: 'part_of', args: ['Component', 'System'], orientation: 'child-first', up: 'every system that contains {x}, all the way up', down: 'every component inside {x}, at any depth' },
      { name: 'owns', args: ['Owner', 'Service'], orientation: 'parent-first', up: 'every owner above {x} in the ownership tree', down: 'everything {x} owns directly or transitively' },
    ],
    dependency: [
      { name: 'depends_on', args: ['Service', 'Upstream'], orientation: 'child-first', up: 'everything {x} ultimately depends on', down: 'everything that ultimately depends on {x}' },
      { name: 'feeds', args: ['Producer', 'Consumer'], orientation: 'parent-first', up: 'everything upstream that eventually feeds {x}', down: 'everything downstream that {x} eventually feeds' },
      { name: 'calls', args: ['Caller', 'Callee'], orientation: 'child-first', up: 'every service {x} reaches through calls', down: 'every service that reaches {x} through calls' },
    ],
    attributes: [
      { name: 'tier', args: ['Service', 'Tier'], values: ['gold', 'silver', 'bronze'] },
      { name: 'runtime', args: ['Service', 'Runtime'], values: ['node', 'go', 'rust', 'python'] },
      { name: 'region', args: ['Service', 'Region'], values: ['sydney', 'oregon', 'dublin'] },
    ],
    memberships: [
      { name: 'in_team', args: ['Service', 'Team'], groups: ['core', 'edge', 'money', 'infra'] },
      { name: 'tagged', args: ['Service', 'Tag'], groups: ['pci', 'public', 'internal'] },
    ],
    schedules: [
      { name: 'deploy_slot', args: ['Team', 'Day', 'Window'], slots: ['monday', 'wednesday', 'friday'], windows: ['morning', 'evening'] },
    ],
  },
  {
    entities: ['arles', 'bergen', 'cusco', 'derry', 'essen', 'fes', 'ghent', 'hobart', 'izmir', 'jaipur', 'kobe', 'leon', 'malmo', 'nara', 'oslo', 'porto', 'quito', 'riga', 'split', 'tunis', 'udine', 'vigo', 'wuhan', 'york', 'zadar', 'basel'],
    hierarchy: [
      { name: 'located_in', args: ['Place', 'Region'], orientation: 'child-first', up: 'every region containing {x}, up to the top', down: 'every place inside {x}, at any depth' },
      { name: 'contains', args: ['Region', 'Place'], orientation: 'parent-first', up: 'every region that contains {x}', down: 'everything inside {x}, however deep' },
    ],
    dependency: [
      { name: 'ships_to', args: ['Origin', 'Destination'], orientation: 'child-first', up: 'every place goods from {x} eventually reach', down: 'every place whose goods eventually reach {x}' },
      { name: 'precedes', args: ['Stop', 'NextStop'], orientation: 'child-first', up: 'every stop after {x} on the route', down: 'every stop before {x} on the route' },
    ],
    attributes: [
      { name: 'climate', args: ['Place', 'Climate'], values: ['arid', 'temperate', 'tropical'] },
      { name: 'currency', args: ['Place', 'Currency'], values: ['euro', 'dollar', 'yen', 'dirham'] },
    ],
    memberships: [
      { name: 'on_route', args: ['Place', 'Route'], groups: ['north', 'coastal', 'inland'] },
      { name: 'hub_for', args: ['Place', 'Carrier'], groups: ['skyline', 'oceanic', 'railnet'] },
    ],
    schedules: [
      { name: 'market_day', args: ['Route', 'Day', 'Window'], slots: ['tuesday', 'saturday'], windows: ['dawn', 'dusk'] },
    ],
  },
];

/** Build a forest/DAG with at least one chain of three hops over `nodes`. */
function chainEdges(nodes: string[], rng: Rng): Array<[string, string]> {
  const edges: Array<[string, string]> = [];
  // guaranteed spine of length >= 4 nodes (3 hops)
  const spineLength = 4 + rng.int(2);
  for (let i = 0; i + 1 < spineLength && i + 1 < nodes.length; i += 1) edges.push([nodes[i], nodes[i + 1]]);
  // attach the remaining nodes as children of earlier nodes (keeps it acyclic)
  for (let i = spineLength; i < nodes.length; i += 1) {
    if (rng.next() < 0.15) continue; // leave a few isolated
    const parentIndex = rng.int(i);
    edges.push([nodes[i], nodes[parentIndex]]);
  }
  return edges;
}

function fact(name: string, ...args: string[]): string {
  return `${name}(${args.join(', ')}).`;
}

export function generateWorld(seed: number): World {
  const rng = createRng(seed);
  const theme = THEMES[seed % THEMES.length];
  const entityCount = 12 + rng.int(19); // 12..30
  const entities = rng.shuffle(theme.entities).slice(0, Math.min(entityCount, theme.entities.length));
  const relations: Relation[] = [];
  const facts: string[] = [];

  const edgeRelations: EdgeVocab[] = [];
  const hierarchyCount = 1 + rng.int(Math.min(2, theme.hierarchy.length));
  edgeRelations.push(...rng.shuffle(theme.hierarchy).slice(0, hierarchyCount));
  if (rng.next() < 0.7) edgeRelations.push(rng.pick(theme.dependency));
  for (const edge of edgeRelations) {
    relations.push({
      name: edge.name,
      kind: theme.hierarchy.includes(edge) ? 'hierarchy' : 'dependency',
      args: [...edge.args],
      orientation: edge.orientation,
      upPhrase: edge.up,
      downPhrase: edge.down,
    });
    const nodes = rng.shuffle(entities).slice(0, 6 + rng.int(entities.length - 5));
    for (const [child, parent] of chainEdges(nodes, rng)) {
      facts.push(edge.orientation === 'child-first' ? fact(edge.name, child, parent) : fact(edge.name, parent, child));
    }
  }

  const attrCount = 1 + rng.int(theme.attributes.length);
  for (const attr of rng.shuffle(theme.attributes).slice(0, attrCount)) {
    relations.push({ name: attr.name, kind: 'attribute', args: [...attr.args] });
    for (const entity of entities) {
      if (rng.next() < 0.7) facts.push(fact(attr.name, entity, rng.pick(attr.values)));
    }
  }
  // guarantee a gap: the first attribute relation skips the last entity
  const firstAttr = relations.find((r) => r.kind === 'attribute')!;
  const last = entities[entities.length - 1];
  for (let i = facts.length - 1; i >= 0; i -= 1) {
    if (facts[i].startsWith(`${firstAttr.name}(${last},`)) facts.splice(i, 1);
  }

  const group = rng.pick(theme.memberships);
  relations.push({ name: group.name, kind: 'membership', args: [...group.args] });
  for (const entity of entities) {
    const count = 1 + (rng.next() < 0.3 ? 1 : 0);
    for (const g of rng.shuffle(group.groups).slice(0, count)) facts.push(fact(group.name, entity, g));
  }

  if (rng.next() < 0.6) {
    const schedule = rng.pick(theme.schedules);
    relations.push({ name: schedule.name, kind: 'schedule', args: [...schedule.args] });
    for (const g of group.groups) {
      if (rng.next() < 0.6) facts.push(fact(schedule.name, g, rng.pick(schedule.slots), rng.pick(schedule.windows)));
    }
  }

  return { id: `world-${seed}`, seed, relations, entities, facts: [...new Set(facts)] };
}

export function worldClauses(world: World): Clause[] {
  return parseProgram(world.facts.join('\n'));
}

export function schemaListing(world: World): string {
  return world.relations.map((r) => `${r.name}(${r.args.join(', ')})`).join('\n');
}

export function relationsOfKind(world: World, kind: RelationKind): Relation[] {
  return world.relations.filter((r) => r.kind === kind);
}
```

- [ ] **Step 5: Run tests**

Run: `npx vitest run tests/training.test.ts`
Expected: PASS (5 tests). If the three-hop invariant fails for a seed, `chainEdges` spine is too short for small node sets: raise the minimum node slice to 6 (already) and confirm `spineLength <= nodes.length`.

- [ ] **Step 6: Commit**

```bash
npx prettier --write src/training/rng.ts src/training/worlds.ts tests/training.test.ts
git add src/training/rng.ts src/training/worlds.ts tests/training.test.ts
git commit -m "Add seeded synthetic worlds for query-dialect training data"
```

---

### Task 2: Question templates in the flat dialect

**Files:**
- Create: `src/training/templates.ts`
- Test: `tests/training.test.ts` (append)

**Interfaces:**
- Consumes: `World`, `Relation`, `relationsOfKind`, `worldClauses` from Task 1; `Rng`.
- Produces:
  ```ts
  export type Category = 'direct' | 'join' | 'multihop-up' | 'multihop-down' | 'root' | 'leaves' | 'chain-filter' | 'yes-no' | 'absence' | 'count';
  export type Direction = 'anchor-first' | 'anchor-second' | 'none';
  export interface Candidate { world: string; category: Category; direction: Direction; template: string; question: string; program: string; expectEmpty: boolean; requiresClosure: boolean; }
  export function generateCandidates(world: World, rng: Rng): Candidate[];
  ```

- [ ] **Step 1: Write the failing tests**

```ts
// append to tests/training.test.ts
import { generateCandidates } from '../src/training/templates.js';
import { parseProgram, isIntegrityConstraint } from '../src/engine/index.js';

describe('training: templates', () => {
  it('emits every category and balances chain directions', () => {
    const counts: Record<string, number> = {};
    const directions: Record<string, number> = { 'anchor-first': 0, 'anchor-second': 0 };
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed * 31))) {
        counts[candidate.category] = (counts[candidate.category] ?? 0) + 1;
        if (candidate.requiresClosure && candidate.direction !== 'none') directions[candidate.direction] += 1;
      }
    }
    for (const category of ['direct', 'join', 'multihop-up', 'multihop-down', 'root', 'leaves', 'chain-filter', 'yes-no', 'absence', 'count']) {
      expect(counts[category], category).toBeGreaterThan(0);
    }
    const ratio = directions['anchor-first'] / directions['anchor-second'];
    expect(ratio).toBeGreaterThan(0.6);
    expect(ratio).toBeLessThan(1.7);
  });

  it('never emits a recursive rule', () => {
    for (let seed = 1; seed <= 20; seed += 1) {
      const world = generateWorld(seed);
      for (const candidate of generateCandidates(world, createRng(seed))) {
        if (!candidate.program.includes(':-')) continue;
        const clauses = parseProgram(candidate.program).filter((c) => !isIntegrityConstraint(c));
        for (const clause of clauses) {
          for (const goal of clause.body) {
            const literal = 'predicate' in goal ? goal : 'not' in goal ? goal.not : undefined;
            expect(literal?.predicate).not.toBe(clause.head.predicate);
          }
        }
      }
    }
  });

  it('mentions every constant of the program in the question', () => {
    const world = generateWorld(2);
    for (const candidate of generateCandidates(world, createRng(2))) {
      const constants = candidate.program.match(/\b[a-z][a-z0-9_]*\b/g) ?? [];
      const entityConstants = constants.filter((c) => world.entities.includes(c));
      for (const constant of entityConstants) {
        expect(candidate.question.toLowerCase(), candidate.program).toContain(constant);
      }
    }
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/templates.js`.

- [ ] **Step 3: Implement templates.ts**

```ts
// src/training/templates.ts
import { evaluate, parseQuery } from '../engine/index.js';
import type { Rng } from './rng.js';
import { relationsOfKind, worldClauses, type Relation, type World } from './worlds.js';

export type Category =
  | 'direct' | 'join' | 'multihop-up' | 'multihop-down' | 'root' | 'leaves'
  | 'chain-filter' | 'yes-no' | 'absence' | 'count';
export type Direction = 'anchor-first' | 'anchor-second' | 'none';

export interface Candidate {
  world: string;
  category: Category;
  direction: Direction;
  template: string;
  question: string;
  program: string;
  /** True when an empty answer is the correct answer (yes/no false). */
  expectEmpty: boolean;
  /** True when the program uses a _plus predicate. */
  requiresClosure: boolean;
}

/** Edge literal with the anchor placed by orientation; returns [literal, direction]. */
function chainLiteral(edge: Relation, anchor: string, variable: string, upward: boolean): [string, Direction] {
  // child-first edge(child, parent): "up" from anchor = edge_plus(anchor, V); parent-first: edge_plus(V, anchor).
  const anchorFirst = (edge.orientation === 'child-first') === upward;
  return anchorFirst
    ? [`${edge.name}_plus(${anchor}, ${variable})`, 'anchor-first']
    : [`${edge.name}_plus(${variable}, ${anchor})`, 'anchor-second'];
}

function oneHop(edge: Relation, anchor: string, variable: string, upward: boolean): string {
  const anchorFirst = (edge.orientation === 'child-first') === upward;
  return anchorFirst ? `${edge.name}(${anchor}, ${variable})` : `${edge.name}(${variable}, ${anchor})`;
}

function phrase(template: string | undefined, x: string): string {
  return (template ?? 'everything connected to {x}').replaceAll('{x}', x);
}

function nodesWithEdges(world: World, edge: Relation): string[] {
  const clauses = worldClauses(world);
  return world.entities.filter(
    (e) => evaluate(clauses, parseQuery(`${edge.name}(${e}, _).`)).length > 0 ||
      evaluate(clauses, parseQuery(`${edge.name}(_, ${e}).`)).length > 0,
  );
}

export function generateCandidates(world: World, rng: Rng): Candidate[] {
  const out: Candidate[] = [];
  const add = (c: Omit<Candidate, 'world'>) => out.push({ world: world.id, ...c });
  const edges = [...relationsOfKind(world, 'hierarchy'), ...relationsOfKind(world, 'dependency')];
  const attributes = relationsOfKind(world, 'attribute');
  const groups = relationsOfKind(world, 'membership');
  const clauses = worldClauses(world);
  const valuesOf = (attr: Relation) =>
    [...new Set(evaluate(clauses, parseQuery(`${attr.name}(_, V).`)).map((b) => (b.V as { value: string }).value))];
  const groupsOf = (group: Relation) =>
    [...new Set(evaluate(clauses, parseQuery(`${group.name}(_, G).`)).map((b) => (b.G as { value: string }).value))];

  // direct
  for (const attr of attributes) {
    for (const x of rng.shuffle(world.entities).slice(0, 3)) {
      add({ category: 'direct', direction: 'none', template: 'direct-value', expectEmpty: false, requiresClosure: false,
        question: rng.pick([`What is the ${attr.args[1].toLowerCase()} of ${x}?`, `Which ${attr.args[1].toLowerCase()} does ${x} have?`, `${x}: ${attr.args[1].toLowerCase()}?`]),
        program: `q(V) :- ${attr.name}(${x}, V).` });
    }
    for (const v of rng.shuffle(valuesOf(attr)).slice(0, 2)) {
      add({ category: 'direct', direction: 'none', template: 'direct-holders', expectEmpty: false, requiresClosure: false,
        question: rng.pick([`Who has ${attr.args[1].toLowerCase()} ${v}?`, `List everything whose ${attr.args[1].toLowerCase()} is ${v}.`]),
        program: `q(X) :- ${attr.name}(X, ${v}).` });
    }
  }

  // join
  for (const group of groups) {
    for (const attr of attributes.slice(0, 2)) {
      const g = rng.pick(groupsOf(group));
      const v = rng.pick(valuesOf(attr));
      add({ category: 'join', direction: 'none', template: 'join-group-attr', expectEmpty: false, requiresClosure: false,
        question: rng.pick([`Which members of ${g} have ${attr.args[1].toLowerCase()} ${v}?`, `In ${g}, who has ${attr.args[1].toLowerCase()} ${v}?`]),
        program: `q(X) :- ${group.name}(X, ${g}), ${attr.name}(X, ${v}).` });
    }
    const g = rng.pick(groupsOf(group));
    add({ category: 'join', direction: 'none', template: 'join-pairs', expectEmpty: false, requiresClosure: false,
      question: `Which pairs of different members both belong to ${g}?`,
      program: `q(A, B) :- ${group.name}(A, ${g}), ${group.name}(B, ${g}), A != B.` });
  }

  // chains
  for (const edge of edges) {
    const nodes = rng.shuffle(nodesWithEdges(world, edge));
    for (const x of nodes.slice(0, 3)) {
      const [up, upDir] = chainLiteral(edge, x, 'Y', true);
      add({ category: 'multihop-up', direction: upDir, template: 'chain-up', expectEmpty: false, requiresClosure: true,
        question: `List ${phrase(edge.upPhrase, x)}.`, program: `q(Y) :- ${up}.` });
      const [down, downDir] = chainLiteral(edge, x, 'Y', false);
      add({ category: 'multihop-down', direction: downDir, template: 'chain-down', expectEmpty: false, requiresClosure: true,
        question: `List ${phrase(edge.downPhrase, x)}.`, program: `q(Y) :- ${down}.` });
    }
    for (const x of nodes.slice(3, 5)) {
      const [up, upDir] = chainLiteral(edge, x, 'R', true);
      add({ category: 'root', direction: upDir, template: 'chain-root', expectEmpty: false, requiresClosure: true,
        question: `Following ${edge.name} from ${x} to the end: what is the final one that has nothing further above it?`,
        program: `q(R) :- ${up}, \\+ ${oneHop(edge, 'R', '_', true)}.` });
      const [down, downDir] = chainLiteral(edge, x, 'L', false);
      add({ category: 'leaves', direction: downDir, template: 'chain-leaves', expectEmpty: false, requiresClosure: true,
        question: `Under ${x} via ${edge.name}, which ones have nothing further below them?`,
        program: `q(L) :- ${down}, \\+ ${oneHop(edge, 'L', '_', false)}.` });
    }
    if (attributes.length > 0) {
      const attr = rng.pick(attributes);
      const v = rng.pick(valuesOf(attr));
      const x = rng.pick(nodes);
      const [up, upDir] = chainLiteral(edge, x, 'Y', true);
      add({ category: 'chain-filter', direction: upDir, template: 'chain-filter-up', expectEmpty: false, requiresClosure: true,
        question: `Among ${phrase(edge.upPhrase, x)}, which have ${attr.args[1].toLowerCase()} ${v}?`,
        program: `q(Y) :- ${up}, ${attr.name}(Y, ${v}).` });
    }
    // yes/no: one true pair, one false pair
    const reach = (a: string, b: string) => evaluate(clauses, parseQuery(`${chainLiteral(edge, a, 'Y', true)[0]}, Y = ${b}.`)).length > 0;
    const pairs = rng.shuffle(nodes.flatMap((a) => nodes.filter((b) => b !== a).map((b) => [a, b] as const)));
    const truePair = pairs.find(([a, b]) => reach(a, b));
    const falsePair = pairs.find(([a, b]) => !reach(a, b));
    for (const [pair, truth] of [[truePair, true], [falsePair, false]] as const) {
      if (!pair) continue;
      const [a, b] = pair;
      const [up, upDir] = chainLiteral(edge, a, 'Y', true);
      add({ category: 'yes-no', direction: upDir, template: `yes-no-${truth}`, expectEmpty: !truth, requiresClosure: true,
        question: rng.pick([`Is ${b} anywhere above ${a} via ${edge.name}? Yes or no.`, `Does ${a} ultimately reach ${b} through ${edge.name}?`]),
        program: `q(Y) :- ${up}, Y = ${b}.` });
    }
    // count
    const x = rng.pick(nodes);
    const [up, upDir] = chainLiteral(edge, x, 'Y', true);
    add({ category: 'count', direction: upDir, template: 'count-chain', expectEmpty: false, requiresClosure: true,
      question: `How many are there in total among ${phrase(edge.upPhrase, x)}?`,
      program: `count(*) as N where ${up}` });
  }

  // absence
  for (const group of groups) {
    for (const attr of attributes) {
      add({ category: 'absence', direction: 'none', template: 'absence-attr', expectEmpty: false, requiresClosure: false,
        question: rng.pick([`Which members of any ${group.args[1].toLowerCase()} have no recorded ${attr.args[1].toLowerCase()}?`, `Who is in a ${group.args[1].toLowerCase()} but has no ${attr.args[1].toLowerCase()} on file?`]),
        program: `q(E) :- ${group.name}(E, _), \\+ ${attr.name}(E, _).` });
    }
  }
  for (const group of groups) {
    const g = rng.pick(groupsOf(group));
    add({ category: 'count', direction: 'none', template: 'count-group', expectEmpty: false, requiresClosure: false,
      question: `How many members does ${g} have?`, program: `count(*) as N where ${group.name}(X, ${g})` });
  }
  return out;
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/training.test.ts`
Expected: PASS. If the direction ratio test fails, check that each theme mixes `child-first` and `parent-first` edges (they do) and that `chainLiteral` flips on orientation.

- [ ] **Step 5: Commit**

```bash
npx prettier --write src/training/templates.ts tests/training.test.ts
git add src/training/templates.ts tests/training.test.ts
git commit -m "Add flat-dialect question templates for training data"
```

---

### Task 3: Execution verification

**Files:**
- Create: `src/training/verify.ts`
- Test: `tests/training.test.ts` (append)

**Interfaces:**
- Consumes: `Candidate` (Task 2), `World`, `worldClauses` (Task 1).
- Produces:
  ```ts
  export interface Example extends Candidate { answer: string[] }        // sorted "X=a Y=b" rows
  export interface Rejection { candidate: Candidate; reason: string }
  export function executeProgram(clauses: Clause[], program: string): Bindings[];
  export function assertRecursionFree(program: string): void;             // throws
  export function verifyCandidate(world: World, candidate: Candidate): Example | Rejection;
  export function verifyAll(world: World, candidates: Candidate[]): { examples: Example[]; rejections: Rejection[] };
  export function assertRejectionRates(candidates: Candidate[], rejections: Rejection[], maxRate?: number): void;
  export function isRejection(value: Example | Rejection): value is Rejection;
  ```

- [ ] **Step 1: Write the failing tests**

```ts
// append to tests/training.test.ts
import { assertRecursionFree, assertRejectionRates, executeProgram, isRejection, verifyAll, verifyCandidate } from '../src/training/verify.js';

describe('training: verify', () => {
  const world = generateWorld(5);
  it('executes rule programs and aggregate queries', () => {
    const clauses = worldClauses(world);
    const edge = [...relationsOfKind(world, 'hierarchy'), ...relationsOfKind(world, 'dependency')][0];
    const rows = executeProgram(clauses, `q(A, B) :- ${edge.name}(A, B).`);
    expect(rows.length).toBeGreaterThan(0);
    const count = executeProgram(clauses, `count(*) as N where ${edge.name}(A, B)`);
    expect(count).toHaveLength(1);
  });

  it('rejects recursive programs', () => {
    expect(() => assertRecursionFree('above(M) :- edge(x, M).\nabove(M) :- above(X), edge(X, M).')).toThrow(/recurs/i);
    expect(() => assertRecursionFree('a(X) :- b(X).\nb(X) :- a(X).')).toThrow(/recurs/i);
    expect(() => assertRecursionFree('q(Y) :- edge_plus(x, Y).')).not.toThrow();
  });

  it('rejects a closure example whose one-hop answer is identical', () => {
    const mini = { ...world, facts: ['edge(a, b).', 'edge(c, d).'], entities: ['a', 'b', 'c', 'd'],
      relations: [{ name: 'edge', kind: 'hierarchy' as const, args: ['A', 'B'], orientation: 'child-first' as const }] };
    const result = verifyCandidate(mini, { world: mini.id, category: 'multihop-up', direction: 'anchor-first', template: 't',
      question: 'above a', program: 'q(Y) :- edge_plus(a, Y).', expectEmpty: false, requiresClosure: true });
    expect(isRejection(result) && result.reason).toMatch(/closure/i);
  });

  it('keeps most generated candidates and records rejections by template', () => {
    const candidates = generateCandidates(world, createRng(5));
    const { examples, rejections } = verifyAll(world, candidates);
    expect(examples.length).toBeGreaterThan(candidates.length / 2);
    expect(() => assertRejectionRates(candidates, rejections)).not.toThrow();
    expect(() => assertRejectionRates(candidates, candidates.map((c) => ({ candidate: c, reason: 'x' })))).toThrow(/rejection rate/i);
    for (const example of examples) {
      if (example.expectEmpty) expect(example.answer).toEqual([]);
      else expect(example.answer.length).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/verify.js`.

- [ ] **Step 3: Implement verify.ts**

```ts
// src/training/verify.ts
import {
  type Bindings, type Clause, evaluateQuerySpec, isComparison, isIntegrityConstraint, isNegation,
  parseProgram, parseQuerySpec, serializeTerm,
} from '../engine/index.js';
import type { Candidate } from './templates.js';
import { worldClauses, type World } from './worlds.js';

export interface Example extends Candidate { answer: string[] }
export interface Rejection { candidate: Candidate; reason: string }
export const MAX_ANSWER_ROWS = 30;

export function isRejection(value: Example | Rejection): value is Rejection {
  return 'reason' in value;
}

/** Rule programs answer their first head; bare queries (including aggregates) run as query specs. */
export function executeProgram(clauses: Clause[], program: string): Bindings[] {
  if (program.includes(':-')) {
    const rules = parseProgram(program);
    const head = rules[0]?.head;
    if (!head) throw new Error('program has no rule');
    return evaluateQuerySpec([...clauses, ...rules], { kind: 'relational', goals: [head] }, { maxRows: MAX_ANSWER_ROWS + 1 });
  }
  return evaluateQuerySpec(clauses, parseQuerySpec(program), { maxRows: MAX_ANSWER_ROWS + 1 });
}

export function assertRecursionFree(program: string): void {
  if (!program.includes(':-')) return;
  const rules = parseProgram(program).filter((c) => !isIntegrityConstraint(c));
  const deps = new Map<string, Set<string>>();
  for (const rule of rules) {
    const set = deps.get(rule.head.predicate) ?? new Set<string>();
    for (const goal of rule.body) {
      if (isComparison(goal)) continue;
      set.add(isNegation(goal) ? goal.not.predicate : goal.predicate);
    }
    deps.set(rule.head.predicate, set);
  }
  const visiting = new Set<string>();
  const done = new Set<string>();
  const visit = (node: string, path: string[]) => {
    if (done.has(node)) return;
    if (visiting.has(node)) throw new Error(`recursive rule: ${[...path, node].join(' -> ')}`);
    visiting.add(node);
    for (const next of deps.get(node) ?? []) if (deps.has(next)) visit(next, [...path, node]);
    visiting.delete(node);
    done.add(node);
  };
  for (const head of deps.keys()) visit(head, []);
}

function rowStrings(bindings: Bindings[]): string[] {
  return bindings
    .map((b) => Object.entries(b).map(([k, t]) => `${k}=${serializeTerm(t)}`).sort().join(' '))
    .sort();
}

export function verifyCandidate(world: World, candidate: Candidate): Example | Rejection {
  const reject = (reason: string): Rejection => ({ candidate, reason });
  let rows: Bindings[];
  try {
    assertRecursionFree(candidate.program);
    rows = executeProgram(worldClauses(world), candidate.program);
  } catch (error) {
    return reject(error instanceof Error ? error.message : String(error));
  }
  if (rows.length > MAX_ANSWER_ROWS) return reject(`more than ${MAX_ANSWER_ROWS} rows`);
  if (candidate.expectEmpty && rows.length > 0) return reject('expected empty answer');
  if (!candidate.expectEmpty && rows.length === 0) return reject('empty answer');
  if (candidate.requiresClosure && !candidate.expectEmpty) {
    const oneHop = candidate.program.replaceAll('_plus(', '(');
    try {
      const hopRows = executeProgram(worldClauses(world), oneHop);
      if (rowStrings(hopRows).join('|') === rowStrings(rows).join('|')) return reject('closure not required: one-hop answer identical');
    } catch {
      // if the one-hop form does not run, the closure form is certainly needed
    }
  }
  return { ...candidate, answer: rowStrings(rows) };
}

export function verifyAll(world: World, candidates: Candidate[]): { examples: Example[]; rejections: Rejection[] } {
  const examples: Example[] = [];
  const rejections: Rejection[] = [];
  for (const candidate of candidates) {
    const result = verifyCandidate(world, candidate);
    if (isRejection(result)) rejections.push(result); else examples.push(result);
  }
  return { examples, rejections };
}

export function assertRejectionRates(candidates: Candidate[], rejections: Rejection[], maxRate = 0.5): void {
  const total = new Map<string, number>();
  const rejected = new Map<string, number>();
  for (const c of candidates) total.set(c.template, (total.get(c.template) ?? 0) + 1);
  for (const r of rejections) rejected.set(r.candidate.template, (rejected.get(r.candidate.template) ?? 0) + 1);
  const offenders = [...total].filter(([template, n]) => (rejected.get(template) ?? 0) / n > maxRate);
  if (offenders.length > 0) {
    throw new Error(`rejection rate above ${maxRate} for templates: ${offenders.map(([t]) => t).join(', ')} — fix the template, do not skip`);
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/training.test.ts`
Expected: PASS. If `keeps most generated candidates` fails, print `rejections` grouped by reason and fix the offending template in `templates.ts` (typical causes: an attribute value with no holders, a group with one member for `join-pairs`).

- [ ] **Step 5: Commit**

```bash
npx prettier --write src/training/verify.ts tests/training.test.ts
git add src/training/verify.ts tests/training.test.ts
git commit -m "Verify training candidates by executing them on their worlds"
```

---

### Task 4: Luna paraphrasing with entity preservation and cache

**Files:**
- Create: `src/training/paraphrase.ts`
- Test: `tests/training.test.ts` (append)

**Interfaces:**
- Consumes: `Example` (Task 3); `LlmClient` from `src/llm/client.ts` (`complete(messages): Promise<string>`).
- Produces:
  ```ts
  export const PARAPHRASE_MODEL = 'openai/gpt-5.6-luna';
  export interface Paraphraser { paraphrase(example: Example, count: number): Promise<string[]> }
  export function preservesConstants(example: Example, paraphrase: string): boolean;
  export function createLlmParaphraser(client: LlmClient, cacheDir: string): Paraphraser;
  export async function paraphraseExamples(examples: Example[], paraphraser: Paraphraser, count: number, concurrency?: number): Promise<Example[]>;
  ```

- [ ] **Step 1: Write the failing tests**

```ts
// append to tests/training.test.ts
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLlmParaphraser, paraphraseExamples, preservesConstants } from '../src/training/paraphrase.js';

describe('training: paraphrase', () => {
  const example = { world: 'w', category: 'multihop-up' as const, direction: 'anchor-first' as const, template: 't',
    question: 'List everyone above ana in the management chain.', program: 'q(Y) :- manages_plus(Y, ana).', expectEmpty: false, requiresClosure: true, answer: ['Y=bo'] };

  it('keeps paraphrases that preserve every entity constant and drops those that do not', () => {
    expect(preservesConstants(example, "Who is above Ana, all the way up?")).toBe(true);
    expect(preservesConstants(example, 'Who is above Bo, all the way up?')).toBe(false);
  });

  it('caches results so a second call makes no client request', async () => {
    let calls = 0;
    const client = { complete: async () => { calls += 1; return JSON.stringify(['Who sits above ana?', 'Name ana\'s whole chain of command.', 'Who is above bo?']); } };
    const dir = mkdtempSync(join(tmpdir(), 'para-'));
    const paraphraser = createLlmParaphraser(client, dir);
    const first = await paraphraser.paraphrase(example, 3);
    const second = await paraphraser.paraphrase(example, 3);
    expect(first).toEqual(['Who sits above ana?', "Name ana's whole chain of command."]);
    expect(second).toEqual(first);
    expect(calls).toBe(1);
  });

  it('falls back to the templated question when the client fails', async () => {
    const client = { complete: async () => { throw new Error('boom'); } };
    const dir = mkdtempSync(join(tmpdir(), 'para-'));
    const out = await paraphraseExamples([example], createLlmParaphraser(client, dir), 2);
    expect(out).toHaveLength(1);
    expect(out[0].question).toBe(example.question);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/paraphrase.js`.

- [ ] **Step 3: Implement paraphrase.ts**

```ts
// src/training/paraphrase.ts
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { LlmClient } from '../llm/client.js';
import type { Example } from './verify.js';

export const PARAPHRASE_MODEL = 'openai/gpt-5.6-luna';

export interface Paraphraser {
  paraphrase(example: Example, count: number): Promise<string[]>;
}

/** Lowercase constants of the program that also appear in the question — these must survive. */
function anchoredConstants(example: Example): string[] {
  const constants = example.program.match(/\b[a-z][a-z0-9_]*\b/g) ?? [];
  const q = example.question.toLowerCase();
  return [...new Set(constants)].filter((c) => !c.endsWith('_plus') && q.includes(c));
}

export function preservesConstants(example: Example, paraphrase: string): boolean {
  const p = paraphrase.toLowerCase();
  return anchoredConstants(example).every((c) => p.includes(c) || p.includes(c.replaceAll('_', ' ')));
}

const SYSTEM = `You rewrite questions about a small knowledge base. Return ONLY a JSON array of strings.
Rules: keep every proper name and value EXACTLY as written (same spelling, lowercase); keep the meaning
and the direction (above/below, depends on/depended on by) identical; vary sentence structure and
vocabulary; no numbering, no prose outside the JSON array.`;

export function createLlmParaphraser(client: LlmClient, cacheDir: string): Paraphraser {
  mkdirSync(cacheDir, { recursive: true });
  return {
    async paraphrase(example, count) {
      const key = createHash('sha256').update(`${PARAPHRASE_MODEL}\n${count}\n${example.question}\n${example.program}`).digest('hex');
      const path = join(cacheDir, `paraphrase-${key}.json`);
      if (existsSync(path)) return JSON.parse(readFileSync(path, 'utf8')) as string[];
      const raw = await client.complete([
        { role: 'system', content: SYSTEM },
        { role: 'user', content: `Write ${count} paraphrases of: ${example.question}` },
      ]);
      const start = raw.indexOf('['); const end = raw.lastIndexOf(']');
      let parsed: unknown = [];
      try { parsed = JSON.parse(raw.slice(start, end + 1)); } catch { parsed = []; }
      const kept = (Array.isArray(parsed) ? parsed : [])
        .filter((p): p is string => typeof p === 'string' && p.trim().length > 0)
        .map((p) => p.trim())
        .filter((p) => p.toLowerCase() !== example.question.toLowerCase())
        .filter((p) => preservesConstants(example, p))
        .slice(0, count);
      writeFileSync(path, JSON.stringify(kept));
      return kept;
    },
  };
}

export async function paraphraseExamples(examples: Example[], paraphraser: Paraphraser, count: number, concurrency = 8): Promise<Example[]> {
  const out: Example[] = [];
  let index = 0;
  const worker = async () => {
    while (index < examples.length) {
      const example = examples[index++];
      out.push(example);
      if (count <= 0) continue;
      try {
        for (const question of await paraphraser.paraphrase(example, count)) {
          out.push({ ...example, question, template: `${example.template}#paraphrase` });
        }
      } catch {
        // degrade to the templated question only
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, examples.length) }, worker));
  return out;
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/training.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
npx prettier --write src/training/paraphrase.ts tests/training.test.ts
git add src/training/paraphrase.ts tests/training.test.ts
git commit -m "Paraphrase training questions with Luna, entity-preserving and cached"
```

---

### Task 5: Tinker JSONL export, manifest, and holdout

**Files:**
- Create: `src/training/export.ts`
- Test: `tests/training.test.ts` (append)

**Interfaces:**
- Consumes: `World`, `schemaListing` (Task 1); `Example` (Task 3).
- Produces:
  ```ts
  export const DIALECT_CARD: string;
  export function systemPrompt(world: World): string;
  export function toConversation(world: World, example: Example): { messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> };
  export interface Manifest { seed: number; generatedAt: string; worlds: number; heldoutWorlds: string[]; train: number; heldout: number; byCategory: Record<string, number>; byDirection: Record<string, number>; rejectionsByTemplate: Record<string, number>; paraphraseModel: string | null; paraphrasesPerExample: number; }
  export function exportDataset(input: { worlds: World[]; examples: Example[]; heldoutWorldIds: Set<string>; rejections: Rejection[]; seed: number; paraphraseModel: string | null; paraphrasesPerExample: number }): { train: string; heldout: string; manifest: Manifest };
  export function chooseHeldoutWorlds(worlds: World[], fraction?: number): Set<string>;
  ```

- [ ] **Step 1: Write the failing tests**

```ts
// append to tests/training.test.ts
import { chooseHeldoutWorlds, exportDataset, toConversation } from '../src/training/export.js';

describe('training: export', () => {
  it('emits Tinker conversation lines with system schema, user question, assistant program', () => {
    const world = generateWorld(9);
    const { examples, rejections } = verifyAll(world, generateCandidates(world, createRng(9)));
    const worlds = [world, generateWorld(10)];
    const heldout = chooseHeldoutWorlds(worlds, 0.5);
    const { train, heldout: held, manifest } = exportDataset({ worlds, examples, heldoutWorldIds: heldout, rejections, seed: 9, paraphraseModel: null, paraphrasesPerExample: 0 });
    const lines = (heldout.has(world.id) ? held : train).trim().split('\n');
    expect(lines.length).toBe(examples.length);
    for (const line of lines) {
      const parsed = JSON.parse(line) as { messages: Array<{ role: string; content: string }> };
      expect(parsed.messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant']);
      expect(parsed.messages[0].content).toContain('p_plus');
      expect(parsed.messages[0].content).toContain(world.relations[0].name);
      expect(parsed.messages[2].content).toMatch(/^(q\(|count\()/);
    }
    expect(manifest.train + manifest.heldout).toBe(examples.length);
    expect(Object.values(manifest.byCategory).reduce((a, b) => a + b, 0)).toBe(examples.length);
  });

  it('serializes the assistant program canonically', () => {
    const world = generateWorld(9);
    const example = { world: world.id, category: 'direct' as const, direction: 'none' as const, template: 't', question: 'x', program: 'q(V)   :-   tier( auth ,V ).', expectEmpty: false, requiresClosure: false, answer: [] };
    expect(toConversation(world, example).messages[2].content).toBe('q(V) :- tier(auth, V).');
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/export.js`.

- [ ] **Step 3: Implement export.ts**

```ts
// src/training/export.ts
import { parseProgram, serializeClause } from '../engine/index.js';
import type { Rejection, Example } from './verify.js';
import { schemaListing, type World } from './worlds.js';

export const DIALECT_CARD = `Write ONE Datalog program that answers the question. Reply with ONLY the program.
- Rule shape: q(A, B) :- predicate(A), other(B, A).   The first rule's head is the answer.
- Variables start uppercase (X, Person). Constants are lowercase exactly as listed.
- _ is a wildcard. \\+ predicate(X) means "no such fact". Comparisons: A != B, X = value.
- Any binary predicate p also answers p_plus(X, Y): Y is reachable from X in ONE OR MORE hops.
  Use p_plus for chains ("above", "below", "ultimately", "directly or transitively"). NEVER write recursive rules.
- Yes/no: select the value being checked: q(Y) :- p_plus(x, Y), Y = target.   A row means yes.
- Counting: count(*) as N where predicate(X, value)`;

export function systemPrompt(world: World): string {
  return `You query a knowledge base with these predicates:\n${schemaListing(world)}\n\n${DIALECT_CARD}`;
}

function canonicalProgram(program: string): string {
  if (!program.includes(':-')) return program.replace(/\s+/g, ' ').trim();
  return parseProgram(program).map(serializeClause).join('\n');
}

export function toConversation(world: World, example: Example) {
  return {
    messages: [
      { role: 'system' as const, content: systemPrompt(world) },
      { role: 'user' as const, content: example.question },
      { role: 'assistant' as const, content: canonicalProgram(example.program) },
    ],
  };
}

export interface Manifest {
  seed: number; generatedAt: string; worlds: number; heldoutWorlds: string[]; train: number; heldout: number;
  byCategory: Record<string, number>; byDirection: Record<string, number>; rejectionsByTemplate: Record<string, number>;
  paraphraseModel: string | null; paraphrasesPerExample: number;
}

export function chooseHeldoutWorlds(worlds: World[], fraction = 0.1): Set<string> {
  const count = Math.max(1, Math.round(worlds.length * fraction));
  // deterministic: highest seeds are held out
  return new Set([...worlds].sort((a, b) => b.seed - a.seed).slice(0, count).map((w) => w.id));
}

function tally<T>(items: T[], key: (item: T) => string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of items) out[key(item)] = (out[key(item)] ?? 0) + 1;
  return Object.fromEntries(Object.entries(out).sort(([a], [b]) => a.localeCompare(b)));
}

export function exportDataset(input: {
  worlds: World[]; examples: Example[]; heldoutWorldIds: Set<string>; rejections: Rejection[];
  seed: number; paraphraseModel: string | null; paraphrasesPerExample: number;
}): { train: string; heldout: string; manifest: Manifest } {
  const byId = new Map(input.worlds.map((w) => [w.id, w]));
  const train: string[] = [];
  const heldout: string[] = [];
  for (const example of input.examples) {
    const world = byId.get(example.world);
    if (!world) throw new Error(`unknown world ${example.world}`);
    const line = JSON.stringify(toConversation(world, example));
    (input.heldoutWorldIds.has(world.id) ? heldout : train).push(line);
  }
  return {
    train: `${train.join('\n')}\n`,
    heldout: `${heldout.join('\n')}\n`,
    manifest: {
      seed: input.seed,
      generatedAt: new Date().toISOString(),
      worlds: input.worlds.length,
      heldoutWorlds: [...input.heldoutWorldIds].sort(),
      train: train.length,
      heldout: heldout.length,
      byCategory: tally(input.examples, (e) => e.category),
      byDirection: tally(input.examples, (e) => e.direction),
      rejectionsByTemplate: tally(input.rejections, (r) => r.candidate.template),
      paraphraseModel: input.paraphraseModel,
      paraphrasesPerExample: input.paraphrasesPerExample,
    },
  };
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/training.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
npx prettier --write src/training/export.ts tests/training.test.ts
git add src/training/export.ts tests/training.test.ts
git commit -m "Export training examples as Tinker conversation JSONL with manifest"
```

---

### Task 6: CLI runner, npm script, gitignore, and a dry run

**Files:**
- Create: `src/training/run.ts`
- Modify: `package.json` (scripts)
- Modify: `.gitignore`
- Test: `tests/training.test.ts` (append) — smoke test of the pipeline function

**Interfaces:**
- Consumes: everything above; `clientFromEnv` and `loadEnv` from `src/llm/client.ts` and `src/env.ts`.
- Produces:
  ```ts
  export interface RunOptions { examples: number; worlds: number; paraphrases: number; seed: number; out: string; paraphrase: boolean }
  export async function generateTrainingData(options: RunOptions, paraphraser?: Paraphraser): Promise<Manifest>;
  ```

- [ ] **Step 1: Write the failing test**

```ts
// append to tests/training.test.ts
import { existsSync, readFileSync } from 'node:fs';
import { generateTrainingData } from '../src/training/run.js';

describe('training: run', () => {
  it('generates at least the requested number of verified examples without paraphrasing', async () => {
    const out = mkdtempSync(join(tmpdir(), 'train-'));
    const manifest = await generateTrainingData({ examples: 200, worlds: 8, paraphrases: 0, seed: 11, out, paraphrase: false });
    expect(manifest.train + manifest.heldout).toBeGreaterThanOrEqual(200);
    expect(existsSync(join(out, 'conversations.jsonl'))).toBe(true);
    expect(existsSync(join(out, 'heldout.jsonl'))).toBe(true);
    expect(JSON.parse(readFileSync(join(out, 'manifest.json'), 'utf8')).seed).toBe(11);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/training.test.ts`
Expected: FAIL, cannot resolve `../src/training/run.js`.

- [ ] **Step 3: Implement run.ts**

```ts
// src/training/run.ts
/**
 * Generate execution-verified query-dialect training data.
 *   npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training
 *   npm run train:data -- --no-paraphrase          # zero-cost dry run
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadEnv } from '../env.js';
import { clientFromEnv } from '../llm/client.js';
import { chooseHeldoutWorlds, exportDataset, type Manifest } from './export.js';
import { createLlmParaphraser, PARAPHRASE_MODEL, paraphraseExamples, type Paraphraser } from './paraphrase.js';
import { createRng } from './rng.js';
import { generateCandidates } from './templates.js';
import { assertRejectionRates, verifyAll, type Example, type Rejection } from './verify.js';
import { generateWorld } from './worlds.js';

export interface RunOptions {
  examples: number; worlds: number; paraphrases: number; seed: number; out: string; paraphrase: boolean;
}

export async function generateTrainingData(options: RunOptions, paraphraser?: Paraphraser): Promise<Manifest> {
  const worlds = Array.from({ length: options.worlds }, (_, i) => generateWorld(options.seed * 1000 + i + 1));
  const examples: Example[] = [];
  const rejections: Rejection[] = [];
  let round = 0;
  // keep drawing candidates (fresh rng per round) until the target is met
  while (examples.length < options.examples && round < 50) {
    for (const world of worlds) {
      const candidates = generateCandidates(world, createRng(options.seed * 7919 + world.seed * 31 + round));
      const verified = verifyAll(world, candidates);
      if (round === 0) assertRejectionRates(candidates, verified.rejections);
      examples.push(...verified.examples);
      rejections.push(...verified.rejections);
    }
    round += 1;
  }
  // dedupe identical (world, question, program)
  const seen = new Set<string>();
  const unique = examples.filter((e) => {
    const key = `${e.world}\n${e.question}\n${e.program}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const heldout = chooseHeldoutWorlds(worlds);
  let final = unique;
  let model: string | null = null;
  if (options.paraphrase && options.paraphrases > 0) {
    loadEnv();
    const p = paraphraser ?? createLlmParaphraser(
      clientFromEnv({ ...process.env, LLM_MODEL: PARAPHRASE_MODEL }),
      join(options.out, 'cache'),
    );
    final = await paraphraseExamples(unique, p, options.paraphrases);
    model = PARAPHRASE_MODEL;
  }
  const { train, heldout: held, manifest } = exportDataset({
    worlds, examples: final, heldoutWorldIds: heldout, rejections, seed: options.seed,
    paraphraseModel: model, paraphrasesPerExample: model ? options.paraphrases : 0,
  });
  mkdirSync(options.out, { recursive: true });
  writeFileSync(join(options.out, 'conversations.jsonl'), train);
  writeFileSync(join(options.out, 'heldout.jsonl'), held);
  writeFileSync(join(options.out, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

function flag(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback;
}

if (process.argv[1]?.endsWith('run.js') || process.argv[1]?.endsWith('run.ts')) {
  const manifest = await generateTrainingData({
    examples: Number(flag('--examples', '3000')),
    worlds: Number(flag('--worlds', '60')),
    paraphrases: Number(flag('--paraphrases', '3')),
    seed: Number(flag('--seed', '7')),
    out: flag('--out', 'data/training'),
    paraphrase: !process.argv.includes('--no-paraphrase'),
  });
  console.log(JSON.stringify(manifest, null, 2));
}
```

Check `clientFromEnv` signature in `src/llm/client.ts:228` — it takes `env = process.env` and reads `LLM_MODEL`; if it reads a different variable name, pass that one instead.

- [ ] **Step 4: Add npm script and gitignore entries**

In `package.json` scripts, add:
```json
"train:data": "npm run build:core && node dist/training/run.js"
```
In `.gitignore`, append:
```
data/training/*
!data/training/manifest.json
```

- [ ] **Step 5: Run tests, then a real dry run**

Run: `npx vitest run tests/training.test.ts` → PASS.
Run: `npm run train:data -- --examples 3000 --worlds 60 --no-paraphrase --out data/training`
Expected: manifest printed with `train >= 2700`, `heldout > 0`, no template above the rejection threshold.

- [ ] **Step 6: Commit**

```bash
npx prettier --write src/training/run.ts tests/training.test.ts
git add src/training/run.ts tests/training.test.ts package.json .gitignore data/training/manifest.json
git commit -m "Add train:data pipeline runner with manifest"
```

---

### Task 7: Harness closure condition (evaluation prompt)

**Files:**
- Modify: `src/evals/agent-boundary.ts` (add `'remembero-closure'` condition, `datalogClosureSystemPrompt()`, `DATALOG_CLOSURE_FEW_SHOT`)
- Modify: `src/evals/run-agent-boundary.ts:150-152` (pick the prompt by condition), `--conditions` flag, results filename suffix
- Test: `tests/agent-boundary.test.ts` (append)

**Interfaces:**
- Produces: `AGENT_BOUNDARY_CONDITIONS` gains `'remembero-closure'`; `datalogClosureSystemPrompt(): string`; `DATALOG_CLOSURE_FEW_SHOT`.
- Runner: `--conditions a,b` restricts; when restricted, the results file is `agent-boundary-v2-<model>-<conditions>-summary.json` so published files are never overwritten.

- [ ] **Step 1: Write the failing test**

```ts
// append to tests/agent-boundary.test.ts (follow existing imports from '../src/evals/agent-boundary.js' and the DB setup used by the DATALOG_FEW_SHOT test)
it('closure few-shot programs run on the seeded database without any recursive rule', async () => {
  const db = await openRememberoDatabase(':memory:');
  try {
    db.exec(AGENT_BOUNDARY_SEED_SQL);
    for (const example of DATALOG_CLOSURE_FEW_SHOT) {
      expect(example.program).not.toMatch(/^(\w+)\(.*:-.*\b\1\(/m);
      expect(() => db.datalogQuery(example.program)).not.toThrow();
    }
    expect(datalogClosureSystemPrompt()).toContain('_plus');
    expect(AGENT_BOUNDARY_CONDITIONS).toContain('remembero-closure');
  } finally { db.close(); }
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/agent-boundary.test.ts` → FAIL: `DATALOG_CLOSURE_FEW_SHOT` is not exported.

- [ ] **Step 3: Implement**

In `src/evals/agent-boundary.ts`:
- `AGENT_BOUNDARY_CONDITIONS = ['sql', 'sql-gated', 'remembero', 'remembero-closure'] as const;`
- Add after `DATALOG_FEW_SHOT`:
```ts
export const DATALOG_CLOSURE_FEW_SHOT: ReadonlyArray<{ q: string; program: string }> = DATALOG_FEW_SHOT.map((ex) => {
  if (ex.q === 'Every manager above nora in the chain?') return { q: ex.q, program: `above(M) :- reports_to_plus(nora, M).` };
  if (ex.q === 'What is the final upstream dependency atlas waits on?') return { q: ex.q, program: `root(R) :- waits_on_plus(atlas, R), \\+ waits_on(R, _).` };
  if (ex.q === 'Which pairs of different people share a project?') return { q: 'Does priya ultimately report up to ava? Yes or no.', program: `q(M) :- reports_to_plus(priya, M), M = ava.` };
  return ex;
});

const DATALOG_CLOSURE_SCHEMA_PROMPT = `${DATALOG_SCHEMA_PROMPT}
reports_to_plus(Person, Manager)  -- Manager is anywhere above Person: one OR MORE steps up the chain
waits_on_plus(Item, Upstream)     -- Upstream is anywhere up the chain from Item: one OR MORE hops

For any question about a chain, "ultimately", "directly or transitively", "above", "below",
"up the chain", or "the end of the chain": use reports_to_plus or waits_on_plus.
NEVER write recursive rules yourself — the _plus predicates already contain the whole chain.
For a yes/no question, select the value being checked so the rows show it:
q(X) :- waits_on_plus(atlas, X), X = legal_signoff.   (a row means yes; no rows means no)`;

export function datalogClosureSystemPrompt(): string {
  const examples = DATALOG_CLOSURE_FEW_SHOT.map((e) => `Q: ${e.q}\n${e.program}`).join('\n');
  return `${DATALOG_CLOSURE_SCHEMA_PROMPT}

Write ONE Datalog program that answers the user's question.
Reply with ONLY the Datalog, no prose, no markdown fences.

${DATALOG_CHEATSHEET.replace('- Recursion is allowed: a rule body may reuse its own head predicate.', '- Do NOT write recursive rules; use the _plus predicates for chains.')}

Examples:
${examples}`;
}
```
In `src/evals/run-agent-boundary.ts`:
- `executeQuery`: treat `'remembero-closure'` like `'remembero'` (the engine synthesizes closures; nothing appended).
- `applyTrapWrite`: unchanged (`else` branch already covers the new condition).
- prompt selection: `condition === 'remembero' ? datalogSystemPrompt() : condition === 'remembero-closure' ? datalogClosureSystemPrompt() : sqlSystemPrompt()`.
- `--conditions` flag: `const conditions = flagValue('--conditions')?.split(',') ?? AGENT_BOUNDARY_CONDITIONS`; loop over `conditions`; `summarize` and the printing loops iterate `conditions`; result filename adds `-${conditions.join('+')}` when the flag is present.
- Check `tests/agent-boundary.test.ts` and `tests/evals.test.ts` for assertions on `AGENT_BOUNDARY_CONDITIONS.length === 3`; update to 4 where the test is about the list, and confirm the summarize helpers type-check with the new member.

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/agent-boundary.test.ts tests/evals.test.ts` → PASS. Then `npx tsc -p tsconfig.json --noEmit` → clean.

- [ ] **Step 5: Baseline the closure condition on the local models (no training yet)**

Run: `npm run build:core && node dist/evals/run-agent-boundary.js --model llama3.2:3b --conditions remembero-closure --seeds 7`
Expected: writes `docs/research/results/agent-boundary-v2-llama3.2-3b-remembero-closure-summary.json`; published files untouched (`git status` shows only new files). Repeat for `qwen2.5-coder:7b` and `llama3.1:8b`.

- [ ] **Step 6: Commit**

```bash
git add src/evals/agent-boundary.ts src/evals/run-agent-boundary.ts tests/agent-boundary.test.ts docs/research/results/agent-boundary-v2-*-remembero-closure-summary.json
git commit -m "Add remembero-closure harness condition and per-condition runs"
```

---

### Task 8: Tinker training recipe and Ollama-compatible sampling proxy

**Files:**
- Create: `benchmarks/tinker/README.md`
- Create: `benchmarks/tinker/requirements.txt`
- Create: `benchmarks/tinker/sl_query_dialect.py`
- Create: `benchmarks/tinker/ollama_proxy.py`

No vitest coverage (Python, paid). Manual verification steps below. Reference: tinker-cookbook `1f962ed` — `tinker_cookbook/recipes/sl_basic.py`, `tinker_cookbook/supervised/data.py::FromConversationFileBuilder`, `tinker_cookbook/completers.py` (`TinkerMessageCompleter` over `tinker.SamplingClient`).

- [ ] **Step 1: requirements.txt**

```
tinker
tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@1f962ed
chz
python-dotenv
```

- [ ] **Step 2: sl_query_dialect.py**

```python
"""Supervised fine-tune of a small model on Remembero's flat query dialect.

Usage (manual, costs money, confirm before running):
  python benchmarks/tinker/sl_query_dialect.py \
      --data data/training/conversations.jsonl \
      --model Llama-3.2-3B --log-path runs/tinker/llama-3.2-3b-dialect
Reads TINKER_API_KEY from .env at the repo root.
"""
import argparse
import asyncio
import os
from pathlib import Path

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

ORG = {"Llama-3.2-1B": "meta-llama", "Llama-3.2-3B": "meta-llama", "Qwen3-4B-Instruct-2507": "Qwen", "Qwen3.5-4B": "Qwen"}


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY missing (put it in .env)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Llama-3.2-3B", choices=sorted(ORG))
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=100)
    args = parser.parse_args()

    model_name = f"{ORG[args.model]}/{args.model}"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(common_config=common, file_path=args.data, test_size=args.test_size)
    config = chz.Blueprint(train.Config).apply({
        "log_path": args.log_path,
        "model_name": model_name,
        "recipe_name": "remembero_query_dialect",
        "renderer_name": renderer_name,
        "dataset_builder": dataset,
        "learning_rate": args.lr,
        "lr_schedule": "linear",
        "num_epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "eval_every": 20,
        "save_every": 50,
    }).make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
```

Verify field names against `tinker_cookbook/supervised/train.py::Config` before running (`lora_rank`, `save_every`, `eval_every` exist at `1f962ed`; adjust if the pinned commit differs).

- [ ] **Step 3: ollama_proxy.py**

```python
"""Expose a Tinker sampler as an Ollama-compatible /api/chat so the TypeScript
agent-boundary harness evaluates a fine-tune unchanged:

  python benchmarks/tinker/ollama_proxy.py --checkpoint tinker://<run>/sampler_weights/final \
      --base-model meta-llama/Llama-3.2-3B --port 11435
  OLLAMA_URL=http://127.0.0.1:11435 node dist/evals/run-agent-boundary.js \
      --model dialect-llama-3.2-3b --conditions remembero-closure --seeds 7
"""
import argparse
import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter


def build_completer(checkpoint: str, base_model: str, max_tokens: int) -> TinkerMessageCompleter:
    client = tinker.ServiceClient()
    sampling = client.create_sampling_client(model_path=checkpoint, base_model=base_model)
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(base_model), tokenizer=sampling.get_tokenizer())
    return TinkerMessageCompleter(sampling_client=sampling, renderer=renderer, max_tokens=max_tokens)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY missing (put it in .env)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()
    completer = build_completer(args.checkpoint, args.base_model, args.max_tokens)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/chat":
                self.send_response(404); self.end_headers(); return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = [{"role": m["role"], "content": m["content"]} for m in body["messages"]]
            # temperature/seed from body["options"] are accepted but sampling is greedy for eval parity
            reply = asyncio.run(completer(messages))
            payload = {"model": body.get("model"), "message": {"role": "assistant", "content": reply["content"]}, "done": True}
            data = json.dumps(payload).encode()
            self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(data)

    print(f"tinker sampler proxy on http://127.0.0.1:{args.port}/api/chat")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

Confirm the constructor and call signature of `TinkerMessageCompleter` and how greedy sampling is requested (`temperature=0` argument or `SamplingParams`) against `tinker_cookbook/completers.py` at `1f962ed` before first run; adjust the two lines that construct and call the completer.

- [ ] **Step 4: README.md**

Document: prerequisites (`python -m venv .venv && pip install -r benchmarks/tinker/requirements.txt`), the three commands (generate data → train → proxy + harness), that the run costs money and is never automated, where checkpoints land (`tinker://` paths printed by the trainer), and how to record the result JSON next to the published matrix.

- [ ] **Step 5: Manual verification (no paid run yet)**

```bash
python -c "import ast,sys; [ast.parse(open(p).read()) for p in ['benchmarks/tinker/sl_query_dialect.py','benchmarks/tinker/ollama_proxy.py']]; print('syntax ok')"
```
Expected: `syntax ok`. Do not launch training until the user confirms in chat.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/tinker
git commit -m "Add Tinker recipe and sampling proxy for query-dialect fine-tuning"
```

---

### Task 9: Docs and memory

**Files:**
- Modify: `docs/CLOSURE-PREDICATES.md` (add "Training data" pointer)
- Modify: `README.md` (one line under the evals list pointing at `npm run train:data` and `benchmarks/tinker/README.md`)
- Modify: `/Users/rahult/.claude/projects/-Volumes-Atlas-Code-projects-rembero/memory/rembero-small-model-direction.md` (record Tinker model list finding and chosen base model)

- [ ] **Step 1: Edit docs** — add to `docs/CLOSURE-PREDICATES.md` a final section:
```markdown
## Training data

`npm run train:data` generates execution-verified question → program pairs in this dialect
across synthetic worlds; see `docs/superpowers/specs/2026-09-08-query-dialect-training-data-design.md`
and `benchmarks/tinker/README.md` for fine-tuning and evaluation.
```
- [ ] **Step 2: Update memory** — append the Tinker model availability and chosen base model to the memory file's body; keep the index line.
- [ ] **Step 3: Commit**
```bash
git add docs/CLOSURE-PREDICATES.md README.md
git commit -m "Document the query-dialect training pipeline"
```
