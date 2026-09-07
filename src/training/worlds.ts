/**
 * Seeded synthetic worlds for query-dialect training data. A world is a
 * randomly drawn schema (edge, attribute, membership and schedule relations)
 * plus ground facts, themed so entity names and predicate names differ across
 * worlds and never overlap the agent-boundary benchmark's vocabulary.
 */
import { type Clause, parseProgram } from '../engine/index.js';
import { createRng, type Rng } from './rng.js';

export type RelationKind =
  'hierarchy' | 'dependency' | 'attribute' | 'membership' | 'schedule';

/** child-first: edge(child, parent) like reports_to; parent-first: edge(parent, child) like manages. */
export type Orientation = 'child-first' | 'parent-first';

export interface Relation {
  name: string;
  kind: RelationKind;
  /** Capitalized argument names shown in the schema listing. */
  args: string[];
  orientation?: Orientation;
  /** Natural phrase for "things above/upstream of {x}". */
  upPhrase?: string;
  /** Natural phrase for "things below/downstream of {x}". */
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
  'works_on',
  'reports_to',
  'status',
  'blocker',
  'waits_on',
  'prefers_meeting',
  'review_slot',
  'promised_update',
] as const;

interface EdgeVocab {
  name: string;
  args: [string, string];
  orientation: Orientation;
  up: string;
  down: string;
}
interface AttrVocab {
  name: string;
  args: [string, string];
  values: string[];
}
interface GroupVocab {
  name: string;
  args: [string, string];
  groups: string[];
}
interface ScheduleVocab {
  name: string;
  args: [string, string, string];
  slots: string[];
  windows: string[];
}

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
    entities: [
      'ana',
      'bo',
      'cleo',
      'dev',
      'esi',
      'femi',
      'gus',
      'hana',
      'ivo',
      'jun',
      'kai',
      'lena',
      'milo',
      'nia',
      'omar',
      'pia',
      'quin',
      'rosa',
      'sam',
      'tal',
      'uma',
      'vik',
      'wren',
      'xan',
      'yara',
      'zed',
      'aiko',
      'bram',
      'cyd',
      'dara',
    ],
    hierarchy: [
      {
        name: 'manages',
        args: ['Manager', 'Person'],
        orientation: 'parent-first',
        up: 'everyone above {x} in the management chain',
        down: 'everyone who ultimately reports to {x}',
      },
      {
        name: 'mentors',
        args: ['Mentor', 'Mentee'],
        orientation: 'parent-first',
        up: "{x}'s mentors, their mentors, and so on",
        down: 'everyone {x} mentors directly or indirectly',
      },
      {
        name: 'answers_to',
        args: ['Person', 'Lead'],
        orientation: 'child-first',
        up: 'every lead above {x}',
        down: 'everyone under {x}, at any depth',
      },
    ],
    dependency: [
      {
        name: 'hands_off_to',
        args: ['Person', 'Next'],
        orientation: 'child-first',
        up: 'everyone downstream of {x} in the handoff chain',
        down: 'everyone whose work eventually reaches {x}',
      },
    ],
    attributes: [
      {
        name: 'prefers',
        args: ['Person', 'Window'],
        values: ['early', 'late', 'midday'],
      },
      {
        name: 'speaks',
        args: ['Person', 'Language'],
        values: ['spanish', 'hindi', 'french', 'mandarin'],
      },
      {
        name: 'level',
        args: ['Person', 'Level'],
        values: ['junior', 'senior', 'staff'],
      },
    ],
    memberships: [
      {
        name: 'member_of',
        args: ['Person', 'Guild'],
        groups: ['platform', 'growth', 'data', 'design'],
      },
      {
        name: 'assigned_to',
        args: ['Person', 'Squad'],
        groups: ['alpha', 'bravo', 'delta'],
      },
    ],
    schedules: [
      {
        name: 'standup',
        args: ['Squad', 'Day', 'Window'],
        slots: ['monday', 'tuesday', 'thursday'],
        windows: ['early', 'late'],
      },
    ],
  },
  {
    entities: [
      'auth',
      'billing',
      'cache',
      'catalog',
      'checkout',
      'cdn',
      'db_primary',
      'db_replica',
      'email',
      'gateway',
      'inventory',
      'ledger',
      'logging',
      'metrics',
      'notify',
      'orders',
      'payments',
      'pricing',
      'queue',
      'ratings',
      'search',
      'sessions',
      'shipping',
      'storage',
      'tax',
      'users',
      'vault',
      'webhooks',
    ],
    hierarchy: [
      {
        name: 'part_of',
        args: ['Component', 'System'],
        orientation: 'child-first',
        up: 'every system that contains {x}, all the way up',
        down: 'every component inside {x}, at any depth',
      },
      {
        name: 'owns',
        args: ['Owner', 'Service'],
        orientation: 'parent-first',
        up: 'every owner above {x} in the ownership tree',
        down: 'everything {x} owns directly or transitively',
      },
    ],
    dependency: [
      {
        name: 'depends_on',
        args: ['Service', 'Upstream'],
        orientation: 'child-first',
        up: 'everything {x} ultimately depends on',
        down: 'everything that ultimately depends on {x}',
      },
      {
        name: 'feeds',
        args: ['Producer', 'Consumer'],
        orientation: 'parent-first',
        up: 'everything upstream that eventually feeds {x}',
        down: 'everything downstream that {x} eventually feeds',
      },
      {
        name: 'calls',
        args: ['Caller', 'Callee'],
        orientation: 'child-first',
        up: 'every service {x} reaches through calls',
        down: 'every service that reaches {x} through calls',
      },
    ],
    attributes: [
      {
        name: 'tier',
        args: ['Service', 'Tier'],
        values: ['gold', 'silver', 'bronze'],
      },
      {
        name: 'runtime',
        args: ['Service', 'Runtime'],
        values: ['node', 'go', 'rust', 'python'],
      },
      {
        name: 'region',
        args: ['Service', 'Region'],
        values: ['sydney', 'oregon', 'dublin'],
      },
    ],
    memberships: [
      {
        name: 'in_team',
        args: ['Service', 'Team'],
        groups: ['core', 'edge', 'money', 'infra'],
      },
      {
        name: 'tagged',
        args: ['Service', 'Tag'],
        groups: ['pci', 'public', 'internal'],
      },
    ],
    schedules: [
      {
        name: 'deploy_slot',
        args: ['Team', 'Day', 'Window'],
        slots: ['monday', 'wednesday', 'friday'],
        windows: ['morning', 'evening'],
      },
    ],
  },
  {
    entities: [
      'arles',
      'bergen',
      'cusco',
      'derry',
      'essen',
      'fes',
      'ghent',
      'hobart',
      'izmir',
      'jaipur',
      'kobe',
      'leon',
      'malmo',
      'nara',
      'oslo',
      'porto',
      'quito',
      'riga',
      'split',
      'tunis',
      'udine',
      'vigo',
      'wuhan',
      'york',
      'zadar',
      'basel',
    ],
    hierarchy: [
      {
        name: 'located_in',
        args: ['Place', 'Region'],
        orientation: 'child-first',
        up: 'every region containing {x}, up to the top',
        down: 'every place inside {x}, at any depth',
      },
      {
        name: 'contains',
        args: ['Region', 'Place'],
        orientation: 'parent-first',
        up: 'every region that contains {x}',
        down: 'everything inside {x}, however deep',
      },
    ],
    dependency: [
      {
        name: 'ships_to',
        args: ['Origin', 'Destination'],
        orientation: 'child-first',
        up: 'every place goods from {x} eventually reach',
        down: 'every place whose goods eventually reach {x}',
      },
      {
        name: 'precedes',
        args: ['Stop', 'NextStop'],
        orientation: 'child-first',
        up: 'every stop after {x} on the route',
        down: 'every stop before {x} on the route',
      },
    ],
    attributes: [
      {
        name: 'climate',
        args: ['Place', 'Climate'],
        values: ['arid', 'temperate', 'tropical'],
      },
      {
        name: 'currency',
        args: ['Place', 'Currency'],
        values: ['euro', 'dollar', 'yen', 'dirham'],
      },
    ],
    memberships: [
      {
        name: 'on_route',
        args: ['Place', 'Route'],
        groups: ['north', 'coastal', 'inland'],
      },
      {
        name: 'hub_for',
        args: ['Place', 'Carrier'],
        groups: ['skyline', 'oceanic', 'railnet'],
      },
    ],
    schedules: [
      {
        name: 'market_day',
        args: ['Route', 'Day', 'Window'],
        slots: ['tuesday', 'saturday'],
        windows: ['dawn', 'dusk'],
      },
    ],
  },
];

/** Build a forest with a guaranteed spine of at least three hops over `nodes`; returns [child, parent] pairs. */
function chainEdges(nodes: string[], rng: Rng): Array<[string, string]> {
  const edges: Array<[string, string]> = [];
  const spineLength = Math.min(nodes.length, 4 + rng.int(2));
  for (let i = 0; i + 1 < spineLength; i += 1)
    edges.push([nodes[i], nodes[i + 1]]);
  for (let i = spineLength; i < nodes.length; i += 1) {
    if (rng.next() < 0.15) continue; // leave a few isolated
    edges.push([nodes[i], nodes[rng.int(i)]]);
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
  const entities = rng
    .shuffle(theme.entities)
    .slice(0, Math.min(entityCount, theme.entities.length));
  const relations: Relation[] = [];
  const facts: string[] = [];

  const hierarchyCount = 1 + rng.int(Math.min(2, theme.hierarchy.length));
  const edgeVocab: Array<{ vocab: EdgeVocab; kind: RelationKind }> = rng
    .shuffle(theme.hierarchy)
    .slice(0, hierarchyCount)
    .map((vocab) => ({ vocab, kind: 'hierarchy' as const }));
  if (rng.next() < 0.7) {
    edgeVocab.push({ vocab: rng.pick(theme.dependency), kind: 'dependency' });
  }
  for (const { vocab, kind } of edgeVocab) {
    relations.push({
      name: vocab.name,
      kind,
      args: [...vocab.args],
      orientation: vocab.orientation,
      upPhrase: vocab.up,
      downPhrase: vocab.down,
    });
    const nodes = rng
      .shuffle(entities)
      .slice(0, 6 + rng.int(entities.length - 5));
    for (const [child, parent] of chainEdges(nodes, rng)) {
      facts.push(
        vocab.orientation === 'child-first'
          ? fact(vocab.name, child, parent)
          : fact(vocab.name, parent, child),
      );
    }
  }

  const attrCount = 1 + rng.int(theme.attributes.length);
  const attrs = rng.shuffle(theme.attributes).slice(0, attrCount);
  const gapEntity = entities[entities.length - 1];
  for (const attr of attrs) {
    relations.push({
      name: attr.name,
      kind: 'attribute',
      args: [...attr.args],
    });
    for (const entity of entities) {
      // the first attribute relation always leaves the last entity without a value
      if (attr === attrs[0] && entity === gapEntity) continue;
      if (rng.next() < 0.7)
        facts.push(fact(attr.name, entity, rng.pick(attr.values)));
    }
  }

  const group = rng.pick(theme.memberships);
  relations.push({
    name: group.name,
    kind: 'membership',
    args: [...group.args],
  });
  for (const entity of entities) {
    const count = 1 + (rng.next() < 0.3 ? 1 : 0);
    for (const g of rng.shuffle(group.groups).slice(0, count)) {
      facts.push(fact(group.name, entity, g));
    }
  }

  if (rng.next() < 0.6) {
    const schedule = rng.pick(theme.schedules);
    relations.push({
      name: schedule.name,
      kind: 'schedule',
      args: [...schedule.args],
    });
    for (const g of group.groups) {
      if (rng.next() < 0.6) {
        facts.push(
          fact(
            schedule.name,
            g,
            rng.pick(schedule.slots),
            rng.pick(schedule.windows),
          ),
        );
      }
    }
  }

  return {
    id: `world-${seed}`,
    seed,
    relations,
    entities,
    facts: [...new Set(facts)],
  };
}

export function worldClauses(world: World): Clause[] {
  return parseProgram(world.facts.join('\n'));
}

export function schemaListing(world: World): string {
  return world.relations
    .map((r) => `${r.name}(${r.args.join(', ')})`)
    .join('\n');
}

export function relationsOfKind(world: World, kind: RelationKind): Relation[] {
  return world.relations.filter((r) => r.kind === kind);
}
