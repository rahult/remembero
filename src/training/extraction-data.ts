/**
 * Extraction training data from synthetic worlds.
 *
 * The facts are known first; a rendering model turns them into natural text;
 * the text is kept only if it mentions every constant of every fact and no
 * other entity of the world. That makes the gold label exact by construction,
 * the same trick the query data uses with the executor. Each example is
 * exported as the product's own extraction prompt (system), the text (user)
 * and the exact clause lines (assistant), so one adapter can learn the write
 * side alongside the read side.
 *
 * Kinds, matching the extraction benchmark's phenomena:
 *   state          1–3 facts stated plainly
 *   first_person   the subject is the self atom and the text says "I"/"my"
 *   supersession   a stored value changes: retract + add
 *   negation       "no longer" / "does not": nothing is added
 *   hedge          "might" / "maybe": nothing is added
 *   distractor     facts inside irrelevant prose
 *   coreference    two facts, the second sentence pronominal
 *   normalization  "DB Primary" in the text, db_primary in the fact
 *   date_number    numeric and date-valued facts
 *   quoted_name    a multi-word name the schema does not know, stored quoted
 *   implicit_subject  no named entity: "the team has 7 people" / "we deploy on Fridays"
 *                  -> headcount(team, 7) / deploy_day(team, friday)
 *   transcript     USER/ASSISTANT turns; only what the user states or confirms is stored
 */
import { type Clause, parseProgram, serializeClause } from '../engine/index.js';
import {
  NOTHING_SENTINEL,
  buildSchemaSummary,
  extractionSystemPrompt,
  transcriptExtractionSystemPrompt,
} from '../llm/prompts.js';
import type { Rng } from './rng.js';
import {
  relationsOfKind,
  worldClauses,
  type Relation,
  type World,
} from './worlds.js';

export type ExtractionKind =
  | 'state'
  | 'first_person'
  | 'supersession'
  | 'negation'
  | 'hedge'
  | 'distractor'
  | 'coreference'
  | 'normalization'
  | 'date_number'
  | 'quoted_name'
  | 'implicit_subject'
  | 'transcript'
  | 'event';

export interface FactSpec {
  predicate: string;
  args: string[];
}

export interface RenderRequest {
  facts: FactSpec[];
  /** Argument names of the facts' relation, for the instruction. */
  argNames: readonly string[];
  firstPerson: boolean;
  selfAtom: string;
  negated: boolean;
  hedged: boolean;
  distractor?: string;
  /** Name the shared subject once, then refer to it only with a pronoun. */
  pronoun?: boolean;
  /** How to write particular constants in the text (atom -> display form). */
  display?: Record<string, string>;
  /** First person plural: this atom is the speaker's own group, spoken as we/our, never by name. */
  pluralAtom?: string;
}

export type Renderer = (request: RenderRequest) => Promise<string>;

export interface ExtractionExample {
  world: string;
  kind: ExtractionKind;
  input: string;
  /** Facts already in the store when the text arrives (schema conditioning). */
  initialProgram: string;
  expectedAdded: string[];
  /** Retraction patterns, without the `retract` keyword or trailing period. */
  expectedRetract: string[];
  /** 'transcript' inputs are USER:/ASSISTANT: turns and use the transcript prompt. */
  mode: 'text' | 'transcript';
}

/** Multi-word names that exist in no world: stored quoted, capitals kept. */
const FRESH_NAMES = {
  company: [
    'Blue Harbour Analytics',
    'North Star Logistics',
    'Kestrel Data Labs',
    'Ironwood Capital',
    'Silver Fern Studios',
    'Red Kite Robotics',
  ],
  place: [
    'New York',
    'San Jose',
    'Cape Town',
    'Rio de Janeiro',
    'Hong Kong',
    'Tel Aviv',
  ],
  person: ['Mira Chen', 'Tom Okafor', 'Priya Nair', 'Liam Brandt'],
};

/** Attribute values that are common nouns, written lowercase in English. */
const LOWERCASE_VALUES = new Set([
  'early',
  'late',
  'midday',
  'junior',
  'senior',
  'staff',
  'gold',
  'silver',
  'bronze',
  'arid',
  'temperate',
  'tropical',
  'morning',
  'evening',
  'dawn',
  'dusk',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
  'node',
  'go',
  'rust',
  'python',
  'euro',
  'dollar',
  'yen',
  'dirham',
  'public',
  'internal',
  'pci',
]);

/**
 * Facts about a generic, unnamed subject. English states these without naming an
 * entity ("the team has seven people", "our deadline is ..."), and the right fact
 * uses the generic noun as its subject, not a name borrowed from the schema.
 */
const IMPLICIT_SUBJECT_FACTS: Array<{
  predicate: string;
  subject: 'team' | 'project' | 'service';
  argNames: [string, string];
  value: (rng: Rng) => string;
  seedValue: string;
}> = [
  {
    predicate: 'headcount',
    subject: 'team',
    argNames: ['Team', 'NumberOfPeople'],
    value: (rng) => String(3 + rng.int(40)),
    seedValue: '12',
  },
  {
    predicate: 'deploy_day',
    subject: 'team',
    argNames: ['Team', 'Day'],
    value: (rng) =>
      rng.pick(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']),
    seedValue: 'tuesday',
  },
  {
    predicate: 'standup_time',
    subject: 'team',
    argNames: ['Team', 'Time'],
    value: (rng) => `'${rng.pick(['9:00', '9:30', '10:15', '14:00'])}'`,
    seedValue: "'8:45'",
  },
  {
    predicate: 'deadline',
    subject: 'project',
    argNames: ['Project', 'Date'],
    value: (rng) =>
      `'${2025 + rng.int(3)}-${String(1 + rng.int(12)).padStart(2, '0')}-${String(1 + rng.int(28)).padStart(2, '0')}'`,
    seedValue: "'2024-11-30'",
  },
  {
    predicate: 'budget_dollars',
    subject: 'project',
    argNames: ['Project', 'Dollars'],
    value: (rng) => String((5 + rng.int(60)) * 10000),
    seedValue: '80000',
  },
  {
    predicate: 'sla_uptime_percent',
    subject: 'service',
    argNames: ['Service', 'Percent'],
    value: (rng) => rng.pick(['99.5', '99.9', '99.95', '99.99']),
    seedValue: '99.0',
  },
  {
    predicate: 'runs_node_version',
    subject: 'service',
    argNames: ['Service', 'NodeMajorVersion'],
    value: (rng) => String(rng.pick([18, 20, 22, 24])),
    seedValue: '16',
  },
];

/** Long, entity-free user requests: real transcripts bury a fact inside a paragraph like these. */
const USER_REQUESTS = [
  "I'm trying to get a bit more organized this quarter and could use some help thinking through how to structure my week. There is a lot going on between work and home and I keep dropping small things.",
  'Can you help me plan a short trip for next month? I want something relaxed, not too far, and ideally cheap. I have been meaning to take a break for ages.',
  "I've been struggling to keep up with reading lately and would love some suggestions for building a habit again. Evenings are usually when I have time, but I get distracted easily.",
  'Could you recommend a few task management apps that help with prioritizing work and personal tasks? I have tried a couple before and never stuck with them.',
  "I'm putting together a budget and want to track spending better. Any advice on categories that actually work in practice rather than in theory?",
  'I need to prepare for a presentation next week and I am nervous about the question-and-answer part. How do people usually rehearse for that?',
  "We're redoing the onboarding docs at work and I want them to be genuinely useful rather than a wall of text. What structure tends to work?",
  'I want to start cooking more at home instead of ordering in. Could you suggest a way to plan meals for the week without spending my whole Sunday on it?',
];

const USER_FOLLOWUPS = [
  'Any thoughts on where to start?',
  'What would you suggest?',
  'Does that change your recommendation?',
  'Anyway, what do you think?',
  'Could you sketch a plan for me?',
];

/** Long, entity-free assistant advice, the bulk of a real transcript and never a source of facts. */
const ASSISTANT_ADVICE = [
  'Happy to help. A good starting point is to write down everything competing for your attention, then sort it into three buckets: must happen this week, should happen this month, and nice to have. Most people find the first bucket is much smaller than it feels. From there, block two or three fixed slots in your calendar for the must-do items and protect them. Review the list briefly each evening so nothing sits unnoticed for long, and expect to adjust the buckets as the week unfolds.',
  'There are a few approaches that tend to work. First, pick a single place where every task lives, whether that is an app or a notebook, so you never have to wonder where something was written down. Second, decide once a day what the three most important items are and do those before anything else. Third, keep a separate list for ideas and someday items so they do not clutter the daily view. The tools matter less than the habit of reviewing them.',
  'That is a very reasonable plan. For a relaxed trip, I would look at places two to three hours away, book somewhere with a kitchen so you can eat in a few times, and leave at least one day completely unplanned. Check whether midweek dates are cheaper, and pack light so the travel itself is not a chore. If you tell me roughly what you enjoy, I can suggest a couple of specific options.',
  'Building a reading habit usually works best when the bar is low: ten minutes a day, same time, same chair. Keep the book visible and the phone in another room for that window. Many people also find that starting with shorter books or essays helps momentum, and that tracking pages read, even roughly, is surprisingly motivating. If evenings are hard, an audiobook during a commute or a walk can count too.',
];

/**
 * Episodic events people mention in passing ("by the way, I just got back from a
 * three-day trip to Big Sur"). Real transcripts carry most of their memorable facts
 * this way, and the speaker is always the subject.
 */
const EVENT_PLACES = [
  'Big Sur',
  'Yosemite',
  'Lisbon',
  'Cape Town',
  'Kyoto',
  'Banff',
  'Hobart',
  'Tulum',
];
const EVENT_ITEMS = [
  'espresso_machine',
  'road_bike',
  'standing_desk',
  'air_fryer',
  'record_player',
  'sewing_machine',
  'kayak',
  'telescope',
];
const EVENT_OUTINGS = [
  'jazz_concert',
  'pottery_class',
  'book_launch',
  'marathon',
  'wine_tasting',
  'film_festival',
];
const EVENT_PEOPLE = [
  'cousin',
  'sister',
  'coworker',
  'neighbour',
  'aunt',
  'brother',
];
const EVENT_TASKS = [
  'baby_shower_shopping',
  'nursery_painting',
  'moving_house',
  'job_application',
  'wedding_planning',
];
const EVENT_ACTIVITIES = [
  'pottery',
  'rock_climbing',
  'spanish_lessons',
  'sourdough_baking',
  'swimming',
];
const EVENT_TEMPLATES: Array<{
  predicate: string;
  argNames: string[];
  args: (rng: Rng) => string[];
}> = [
  {
    predicate: 'visited',
    argNames: ['Person', 'Place'],
    args: (rng) => [rng.pick(EVENT_PLACES)],
  },
  {
    predicate: 'trip_days',
    argNames: ['Person', 'Place', 'NumberOfDays'],
    args: (rng) => [rng.pick(EVENT_PLACES), String(2 + rng.int(9))],
  },
  {
    predicate: 'bought',
    argNames: ['Person', 'Item'],
    args: (rng) => [rng.pick(EVENT_ITEMS)],
  },
  {
    predicate: 'finished',
    argNames: ['Person', 'Item'],
    args: (rng) => [rng.pick(EVENT_ITEMS)],
  },
  {
    predicate: 'attended',
    argNames: ['Person', 'Event'],
    args: (rng) => [rng.pick(EVENT_OUTINGS)],
  },
  {
    predicate: 'helped',
    argNames: ['Person', 'WhoWasHelped', 'Task'],
    args: (rng) => [rng.pick(EVENT_PEOPLE), rng.pick(EVENT_TASKS)],
  },
  {
    predicate: 'ordered',
    argNames: ['Person', 'Item', 'Recipient'],
    args: (rng) => [rng.pick(EVENT_ITEMS), rng.pick(EVENT_PEOPLE)],
  },
  {
    predicate: 'started',
    argNames: ['Person', 'Activity'],
    args: (rng) => [rng.pick(EVENT_ACTIVITIES)],
  },
  {
    predicate: 'met',
    argNames: ['Person', 'Who'],
    args: (rng) => [rng.pick(EVENT_PEOPLE)],
  },
];
const ASIDE_LEADINS = [
  'By the way,',
  'Also,',
  'Oh, and',
  'Incidentally,',
  'Speaking of which,',
];

const ASSISTANT_ACKS = [
  'Got it. Shall I look at anything else?',
  'Noted.',
  "Understood, I'll keep that in mind.",
  'Thanks, that helps.',
];

const ASSISTANT_TOOL_OUTPUT = [
  '$ npm test\n47 passed, 2 failed (auth.test.ts, cache.test.ts)',
  '$ git status\nOn branch main\nnothing to commit, working tree clean',
  'Error: ECONNREFUSED 127.0.0.1:5432\n    at TCPConnectWrap.afterConnect',
];

const CODE_FENCE = '```';

const USER_TASKS = [
  'Can you check why the build is red?',
  'Run the tests.',
  'Summarise what we did.',
  'Please rename the helper and rerun lint.',
];

const NOISE = [
  'Long week, mostly meetings and a flaky CI pipeline that kept timing out.',
  'Quick note before I forget the rest of the day.',
  'The vendor call slipped again and the docs are still stale.',
  'Reminder that the office is closed on Friday for cleaning.',
];

/** Instruction for the rendering model: state exactly these facts, using these names verbatim. */
export function renderRequestText(request: RenderRequest): string {
  const lines = request.facts.map((fact) => {
    const named = fact.args.map(
      (value, i) => `${request.argNames[i] ?? `argument ${i + 1}`} is ${value}`,
    );
    return `- relation "${fact.predicate.replaceAll('_', ' ')}": ${named.join(', ')}`;
  });
  const voice = request.firstPerson
    ? `Write in the first person: the entity "${request.selfAtom}" is the speaker, so refer to it as I / me / my and never by name.`
    : request.pluralAtom
      ? `Write in the first person plural: the entity "${request.pluralAtom}" is the speaker's own group, so refer to it as we / our / us and never by name.`
      : 'Write in the third person, naming the entities exactly as given.';
  const polarity = request.negated
    ? 'State that these facts are NO LONGER true or are NOT the case (use "no longer", "does not", "never"), so a careful reader would store nothing new.'
    : request.hedged
      ? 'State these as uncertain possibilities ("might", "maybe", "is considering"), so a careful reader would store nothing.'
      : 'State them as settled facts.';
  const pronoun = request.pronoun
    ? 'Name the shared subject once in the first sentence, then refer to it only with a pronoun (she, he or they) in the rest.'
    : '';
  const display =
    request.display && Object.keys(request.display).length > 0
      ? `Write these constants as shown: ${Object.entries(request.display)
          .map(([atom, shown]) => `${atom} -> "${shown}"`)
          .join('; ')}.`
      : '';
  return `Write one or two short natural sentences, as a person telling an assistant something worth remembering, that convey exactly these facts and nothing else:
${lines.join('\n')}

${voice}
${polarity}
${[pronoun, display].filter(Boolean).join('\n')}
Use every constant exactly as written (lowercase, underscores may become spaces) unless told how to write it above. Never copy the field labels or the relation name literally; say it the way a person would. Do not add any other names, numbers, or facts. Reply with the sentences only.`;
}

function looseTokens(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .split(' ')
      .filter(Boolean),
  );
}

function mentioned(
  value: string,
  text: string,
  tokens: ReadonlySet<string>,
): boolean {
  const parts = value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  const compact = parts.join('');
  return (
    (compact.length > 0 &&
      text
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '')
        .includes(compact)) ||
    (parts.length > 0 && parts.every((p) => tokens.has(p)))
  );
}

/**
 * A rendering is valid when every constant of every fact appears in the text
 * (the self atom is exempt: it is spoken as "I") and no other entity of the
 * world appears, so the gold facts are exactly what the text states.
 */
export function verifyRendering(
  text: string,
  facts: Clause[],
  world: World,
  selfAtom: string,
  pluralAtom?: string,
): boolean {
  const tokens = looseTokens(text);
  const constants = new Set<string>();
  for (const clause of facts) {
    for (const term of clause.head.args) {
      if (term.type === 'atom' || term.type === 'num')
        constants.add(String(term.value));
    }
  }
  for (const constant of constants) {
    if (constant === selfAtom || constant === pluralAtom) continue;
    if (!mentioned(constant, text, tokens)) return false;
  }
  for (const entity of world.entities) {
    if (constants.has(entity) || entity === selfAtom) continue;
    if (tokens.has(entity.toLowerCase())) return false;
  }
  return true;
}

function factSpec(clause: Clause): FactSpec {
  return {
    predicate: clause.head.predicate,
    args: clause.head.args.map((t) => String((t as { value: unknown }).value)),
  };
}

function factText(predicate: string, args: readonly string[]): string {
  return `${predicate}(${args.join(', ')}).`;
}

/** "Mira works at Acme." -> "mira works at Acme." for splicing after a lead-in; names stay as rendered otherwise. */
function lowerFirst(text: string): string {
  // "I" stays capitalized: "By the way, I attended ..." not "i attended"
  if (/^I\b/.test(text)) return text;
  return text.charAt(0).toLowerCase() + text.slice(1);
}

function capitalFirst(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function edgeRelations(world: World): Relation[] {
  return [
    ...relationsOfKind(world, 'hierarchy'),
    ...relationsOfKind(world, 'dependency'),
  ];
}

/** Store facts about other entities using the same predicates, so the model is schema-conditioned. */
function schemaSeed(
  world: World,
  predicates: Iterable<string>,
  exclude: Set<string>,
): string[] {
  const wanted = new Set(predicates);
  return world.facts
    .filter((fact) => {
      const name = fact.slice(0, fact.indexOf('('));
      if (!wanted.has(name)) return false;
      const inside = fact.slice(fact.indexOf('(') + 1, fact.lastIndexOf(')'));
      return !inside.split(',').some((arg) => exclude.has(arg.trim()));
    })
    .slice(0, 6);
}

/** A capitalized, spaced or hyphenated surface form for an atom: db_primary -> "DB Primary". */
export function displayForm(atom: string, rng: Rng): string {
  const words = atom.split('_');
  const styles: Array<(w: string[]) => string> = [
    (w) => w.map((x) => x.charAt(0).toUpperCase() + x.slice(1)).join(' '),
    (w) => w.map((x) => x.toUpperCase()).join(' '),
    (w) => w.join('-'),
    (w) => w.map((x) => x.charAt(0).toUpperCase() + x.slice(1)).join('-'),
  ];
  const style = words.length > 1 ? rng.pick(styles) : styles[0];
  return style(words);
}

export interface GenerateOptions {
  selfAtom: string;
  /** Examples attempted per kind per world (each is one rendering call). */
  perKind?: number;
}

export async function generateExtractionExamples(
  world: World,
  rng: Rng,
  render: Renderer,
  options: GenerateOptions,
): Promise<ExtractionExample[]> {
  const perKind = options.perKind ?? 2;
  const out: ExtractionExample[] = [];
  const attributes = relationsOfKind(world, 'attribute');
  const edges = edgeRelations(world);
  const clauses = worldClauses(world);
  const factsOf = (relation: Relation) =>
    clauses.filter(
      (c) => c.head.predicate === relation.name && c.body.length === 0,
    );

  /** Render facts and keep the text only if it states exactly them; returns the text. */
  const renderVerified = async (
    facts: Clause[],
    relation: Relation,
    request: Omit<RenderRequest, 'facts' | 'argNames' | 'selfAtom'>,
  ): Promise<string | undefined> => {
    const { distractor, ...renderRequest } = request;
    // English capitalizes names. Unless the request already fixes display forms,
    // show single-word entity atoms capitalized most of the time so the model
    // learns "Mira" -> mira alongside "'Blue Harbour Analytics'" -> quoted.
    const display = { ...(renderRequest.display ?? {}) };
    if (renderRequest.display === undefined && rng.next() < 0.75) {
      for (const clause of facts) {
        for (const term of clause.head.args) {
          // any single-word atom: subjects and values alike (companies, cities and
          // languages are proper nouns in English), but not multi-word atoms,
          // numbers, the self atom, or lowercase enumerations like "early"/"late"
          if (
            term.type === 'atom' &&
            !term.value.includes('_') &&
            term.value !== options.selfAtom &&
            !LOWERCASE_VALUES.has(term.value)
          ) {
            display[term.value] =
              term.value.charAt(0).toUpperCase() + term.value.slice(1);
          }
        }
      }
    }
    const rendered = await render({
      facts: facts.map(factSpec),
      argNames: relation.args,
      selfAtom: options.selfAtom,
      ...renderRequest,
      ...(Object.keys(display).length > 0 ? { display } : {}),
    });
    // Distractor prose is prepended here, deterministically, rather than asked of
    // the renderer: an earlier version passed it as a request field the real
    // renderer never used, and 120 "distractor" examples were plain facts.
    const text =
      distractor === undefined ? rendered : `${distractor}${rendered}`;
    if (
      !verifyRendering(
        text,
        facts,
        world,
        options.selfAtom,
        renderRequest.pluralAtom,
      )
    )
      return undefined;
    return text;
  };

  const attempt = async (
    kind: ExtractionKind,
    facts: Clause[],
    relation: Relation,
    request: Omit<RenderRequest, 'facts' | 'argNames' | 'selfAtom'>,
    expected: { added: string[]; retract: string[]; initial: string[] },
  ) => {
    const text = await renderVerified(facts, relation, request);
    if (text === undefined) return;
    out.push({
      world: world.id,
      kind,
      input: text,
      initialProgram: bareSchema(kind, expected.retract)
        ? ''
        : expected.initial.join('\n'),
      expectedAdded: expected.added,
      expectedRetract: expected.retract,
      mode: 'text',
    });
  };

  // A real store starts empty, and the model must then name predicates from the text
  // alone; a quarter of the examples show that situation. Kinds that retract need the
  // stored fact, so they always keep their schema.
  const bareSchema = (kind: ExtractionKind, retract: string[]): boolean =>
    retract.length === 0 &&
    kind !== 'supersession' &&
    kind !== 'negation' &&
    rng.next() < 0.25;

  // schedules are three-place, so the model sees argument order beyond subject/value
  const schedules = relationsOfKind(world, 'schedule');
  const pool = [...attributes, ...edges, ...schedules];
  if (pool.length === 0) return out;
  const groups = relationsOfKind(world, 'membership');
  const groupNames = groups[0]
    ? [
        ...new Set(
          factsOf(groups[0]).map((c) =>
            String((c.head.args[1] as { value: string }).value),
          ),
        ),
      ]
    : [];

  for (let i = 0; i < perKind; i += 1) {
    // state: 1–3 facts from one relation about one or two entities
    const relation = rng.pick(pool);
    const candidates = rng.shuffle(factsOf(relation)).slice(0, 1 + rng.int(3));
    if (candidates.length > 0) {
      const subjects = new Set(
        candidates.map((c) =>
          String((c.head.args[0] as { value: string }).value),
        ),
      );
      await attempt(
        'state',
        candidates,
        relation,
        { firstPerson: false, negated: false, hedged: false },
        {
          added: candidates.map(serializeClause),
          retract: [],
          initial: schemaSeed(world, [relation.name], subjects),
        },
      );
    }

    // first_person: the self atom takes the Person slot of an attribute ("I prefer
    // early") or, half the time, of a relation ("my manager is Dana"), so possessives
    // and relations are learned alongside plain attributes
    const attr = attributes.length > 0 ? rng.pick(attributes) : undefined;
    const personEdges = edges.filter((r) => r.args.includes('Person'));
    const useEdge = personEdges.length > 0 && rng.next() < 0.5;
    const fpRelation = useEdge ? rng.pick(personEdges) : attr;
    if (fpRelation && factsOf(fpRelation).length > 0) {
      const sample = rng.pick(factsOf(fpRelation));
      const slot = useEdge ? fpRelation.args.indexOf('Person') : 0;
      const args = sample.head.args.map((t, i) =>
        i === slot ? options.selfAtom : String((t as { value: string }).value),
      );
      const fact = parseProgram(factText(fpRelation.name, args))[0];
      await attempt(
        'first_person',
        [fact],
        fpRelation,
        { firstPerson: true, negated: false, hedged: false },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: schemaSeed(
            world,
            [fpRelation.name],
            new Set([options.selfAtom]),
          ),
        },
      );
    }

    // supersession: an attribute value changes for one entity
    if (attr) {
      const existing = rng.pick(factsOf(attr));
      const subject = String(
        (existing.head.args[0] as { value: string }).value,
      );
      const oldValue = String(
        (existing.head.args[1] as { value: string }).value,
      );
      const values = [
        ...new Set(
          factsOf(attr).map((c) =>
            String((c.head.args[1] as { value: string }).value),
          ),
        ),
      ].filter((v) => v !== oldValue);
      if (values.length > 0) {
        const newValue = rng.pick(values);
        const fact = parseProgram(factText(attr.name, [subject, newValue]))[0];
        await attempt(
          'supersession',
          [fact],
          attr,
          { firstPerson: false, negated: false, hedged: false },
          {
            added: [serializeClause(fact)],
            retract: [`${attr.name}(${subject}, _)`],
            initial: [
              serializeClause(existing),
              ...schemaSeed(world, [attr.name], new Set([subject])),
            ],
          },
        );
      }
    }

    // negation: stated as not the case. Half the time the store holds the fact,
    // so the right output is a retraction alone; otherwise nothing at all.
    if (attr) {
      const sample = rng.pick(factsOf(attr));
      const subject = String((sample.head.args[0] as { value: string }).value);
      const stored = rng.next() < 0.5;
      await attempt(
        'negation',
        [sample],
        attr,
        { firstPerson: false, negated: true, hedged: false },
        {
          added: [],
          retract: stored ? [`${attr.name}(${subject}, _)`] : [],
          initial: [
            ...(stored ? [serializeClause(sample)] : []),
            ...schemaSeed(world, [attr.name], new Set([subject])),
          ],
        },
      );
    }

    // hedge
    if (attr) {
      const sample = rng.pick(factsOf(attr));
      await attempt(
        'hedge',
        [sample],
        attr,
        { firstPerson: false, negated: false, hedged: true },
        {
          added: [],
          retract: [],
          initial: schemaSeed(
            world,
            [attr.name],
            new Set([String((sample.head.args[0] as { value: string }).value)]),
          ),
        },
      );
    }

    // coreference: two facts about one subject, the second sentence pronominal
    if (attr && pool.length > 1) {
      const first = rng.pick(factsOf(attr));
      const subject = String((first.head.args[0] as { value: string }).value);
      const others = rng.shuffle(pool.filter((r) => r !== attr));
      let second: Clause | undefined;
      let other: Relation | undefined;
      for (const candidate of others) {
        second = factsOf(candidate).find(
          (c) =>
            String((c.head.args[0] as { value: string }).value) === subject,
        );
        if (second) {
          other = candidate;
          break;
        }
      }
      if (second && other) {
        await attempt(
          'coreference',
          [first, second],
          attr,
          { firstPerson: false, negated: false, hedged: false, pronoun: true },
          {
            added: [serializeClause(first), serializeClause(second)],
            retract: [],
            initial: schemaSeed(
              world,
              [attr.name, other.name],
              new Set([subject]),
            ),
          },
        );
      }
    }

    // normalization: the text shows "DB Primary" / "check-out"; the fact keeps db_primary
    {
      const rel = rng.pick(pool);
      const all = factsOf(rel);
      // prefer facts with multi-word atoms ("db_primary" -> "DB Primary"); otherwise
      // capitalize a single-word atom ("dev" -> "Dev"), which still teaches lowercasing
      const multi = all.filter((c) =>
        c.head.args.some((t) => t.type === 'atom' && t.value.includes('_')),
      );
      const source = multi.length > 0 ? multi : all;
      if (source.length > 0) {
        const fact = rng.pick(source);
        const display: Record<string, string> = {};
        for (const t of fact.head.args) {
          if (t.type !== 'atom') continue;
          if (t.value.includes('_') || multi.length === 0)
            display[t.value] = displayForm(t.value, rng);
        }
        await attempt(
          'normalization',
          [fact],
          rel,
          { firstPerson: false, negated: false, hedged: false, display },
          {
            added: [serializeClause(fact)],
            retract: [],
            initial: schemaSeed(
              world,
              [rel.name],
              new Set([String((fact.head.args[0] as { value: string }).value)]),
            ),
          },
        );
      }
    }

    // date_number: numeric and date-valued relations seeded into the schema
    {
      const group = groups[0];
      if (groupNames.length > 1) {
        const [target, ...others] = rng.shuffle(groupNames);
        const count = 3 + rng.int(40);
        const fact = parseProgram(`headcount(${target}, ${count}).`)[0];
        const relation: Relation = {
          name: 'headcount',
          kind: 'attribute',
          args: [group.args[1], 'NumberOfPeople'],
        };
        await attempt(
          'date_number',
          [fact],
          relation,
          { firstPerson: false, negated: false, hedged: false },
          {
            added: [serializeClause(fact)],
            retract: [],
            initial: others
              .slice(0, 2)
              .map((g, k) => `headcount(${g}, ${5 + k * 7}).`),
          },
        );
      }
      const person = rng.pick(world.entities);
      const year = 2019 + rng.int(7);
      const month = String(1 + rng.int(12)).padStart(2, '0');
      const day = String(1 + rng.int(28)).padStart(2, '0');
      const date = `'${year}-${month}-${day}'`;
      const fact = parseProgram(`started_on(${person}, ${date}).`)[0];
      const relation: Relation = {
        name: 'started_on',
        kind: 'attribute',
        args: [world.relations[0].args[0], 'Date'],
      };
      const otherPerson = world.entities.find((e) => e !== person) ?? person;
      await attempt(
        'date_number',
        [fact],
        relation,
        { firstPerson: false, negated: false, hedged: false },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: [`started_on(${otherPerson}, '2018-05-14').`],
        },
      );
      // a deadline keyed on a group, so dates are not only about people
      if (groupNames.length > 1) {
        const [target, other] = rng.shuffle(groupNames);
        const y = 2025 + rng.int(3);
        const m = String(1 + rng.int(12)).padStart(2, '0');
        const d = String(1 + rng.int(28)).padStart(2, '0');
        const dl = parseProgram(`deadline(${target}, '${y}-${m}-${d}').`)[0];
        const dlRelation: Relation = {
          name: 'deadline',
          kind: 'attribute',
          args: [group!.args[1], 'Date'],
        };
        await attempt(
          'date_number',
          [dl],
          dlRelation,
          { firstPerson: false, negated: false, hedged: false },
          {
            added: [serializeClause(dl)],
            retract: [],
            initial: [`deadline(${other}, '2024-11-30').`],
          },
        );
      }
    }

    // quoted_name: a multi-word entity the schema does not know, stored quoted
    {
      const person = rng.pick(world.entities);
      const other = world.entities.find((e) => e !== person) ?? person;
      const variant = rng.pick(['company', 'place', 'person'] as const);
      const fresh = rng.pick(FRESH_NAMES[variant]);
      const relation: Relation =
        variant === 'company'
          ? { name: 'works_at', kind: 'attribute', args: ['Person', 'Company'] }
          : variant === 'place'
            ? { name: 'lives_in', kind: 'attribute', args: ['Person', 'City'] }
            : {
                name: 'dentist',
                kind: 'attribute',
                args: ['Person', 'Dentist'],
              };
      const fact = parseProgram(`${relation.name}(${person}, '${fresh}').`)[0];
      const seedValue =
        variant === 'company'
          ? 'acme'
          : variant === 'place'
            ? 'osaka'
            : 'dr_lee';
      await attempt(
        'quoted_name',
        [fact],
        relation,
        {
          firstPerson: false,
          negated: false,
          hedged: false,
          display: { [fresh]: fresh },
        },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: [`${relation.name}(${other}, ${seedValue}).`],
        },
      );
    }

    // implicit_subject: a generic noun is the subject. Rendered either as "the team"/
    // "our project" (the noun is in the text) or, for the team, as plain "we" (the noun
    // is not in the text and the model must still write team). The schema seed shows
    // the predicate on a named group so that borrowing that name is the tempting error.
    {
      const spec = rng.pick(IMPLICIT_SUBJECT_FACTS);
      const value = spec.value(rng);
      const fact = parseProgram(
        factText(spec.predicate, [spec.subject, value]),
      )[0];
      const relation: Relation = {
        name: spec.predicate,
        kind: 'attribute',
        args: [...spec.argNames],
      };
      const plural = spec.subject === 'team' && rng.next() < 0.5;
      const seedSubject =
        groupNames.length > 0 ? rng.pick(groupNames) : 'onboarding';
      await attempt(
        'implicit_subject',
        [fact],
        relation,
        plural
          ? {
              firstPerson: false,
              negated: false,
              hedged: false,
              pluralAtom: 'team',
            }
          : {
              firstPerson: false,
              negated: false,
              hedged: false,
              display: {
                [spec.subject]: rng.pick([
                  `the ${spec.subject}`,
                  `our ${spec.subject}`,
                  `my ${spec.subject}`,
                ]),
              },
            },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: [`${spec.predicate}(${seedSubject}, ${spec.seedValue}).`],
        },
      );
    }

    // transcript: the user's sentence is one turn among assistant turns that contain
    // acknowledgements, guesses, summaries or tool output. Only the user's facts (or a
    // fact the user explicitly confirms) are gold. Each piece is verified on its own.
    {
      const relation = rng.pick(pool);
      const facts = rng.shuffle(factsOf(relation)).slice(0, 1 + rng.int(2));
      const guessRelation = rng.pick(pool);
      const guessCandidates = factsOf(guessRelation);
      const guessFact =
        guessCandidates.length > 0 ? rng.pick(guessCandidates) : undefined;
      if (facts.length > 0 && guessFact) {
        const speaker = rng.next() < 0.4 && attr ? 'first' : 'third';
        let userFacts = facts;
        let userRelation = relation;
        if (speaker === 'first' && attr && factsOf(attr).length > 0) {
          const sample = rng.pick(factsOf(attr));
          userRelation = attr;
          userFacts = [
            parseProgram(
              factText(attr.name, [
                options.selfAtom,
                String((sample.head.args[1] as { value: string }).value),
              ]),
            )[0],
          ];
        }
        const userText = await renderVerified(userFacts, userRelation, {
          firstPerson: speaker === 'first',
          negated: false,
          hedged: false,
        });
        const guessText = await renderVerified([guessFact], guessRelation, {
          firstPerson: false,
          negated: false,
          hedged: false,
        });
        if (userText !== undefined && guessText !== undefined) {
          const subjects = new Set(
            userFacts.map((c) =>
              String((c.head.args[0] as { value: string }).value),
            ),
          );
          const initial = [
            ...schemaSeed(world, [userRelation.name], subjects),
            ...schemaSeed(
              world,
              [guessRelation.name],
              new Set([
                String((guessFact.head.args[0] as { value: string }).value),
              ]),
            ),
          ];
          // embedded twice: the fact inside a long request is the common real shape
          const variant = rng.pick([
            'ack',
            'guess',
            'summary',
            'tool',
            'confirm',
            'code',
            'embedded',
            'embedded',
          ] as const);
          const turns: string[] = [];
          let added = userFacts.map(serializeClause);
          switch (variant) {
            case 'ack':
              turns.push(
                `USER: ${userText}`,
                `ASSISTANT: ${rng.pick(ASSISTANT_ACKS)}`,
              );
              break;
            case 'guess':
              // the assistant states a fact nobody confirmed; the user only asks for work
              turns.push(
                `USER: ${rng.pick(USER_TASKS)}`,
                `ASSISTANT: ${rng.pick(['It looks like', 'I assume'])} ${lowerFirst(guessText)}`,
              );
              added = [];
              break;
            case 'summary':
              turns.push(
                `USER: ${userText}`,
                `ASSISTANT: ${rng.pick(ASSISTANT_ACKS)}`,
                'USER: Summarise what we did.',
                `ASSISTANT: Summary of the session: ${lowerFirst(guessText)}`,
              );
              break;
            case 'tool':
              turns.push(
                `USER: ${rng.pick(USER_TASKS)} ${userText}`,
                `ASSISTANT: ${rng.pick(ASSISTANT_TOOL_OUTPUT)}`,
              );
              break;
            case 'confirm':
              // the user confirms the assistant's statement, which makes it the user's
              turns.push(
                `ASSISTANT: Just to confirm: ${lowerFirst(guessText)}`,
                `USER: Yes, that's right.`,
              );
              added = [serializeClause(guessFact)];
              break;
            case 'code':
              turns.push(
                `USER: Here is the failing snippet:\n${CODE_FENCE}ts\nconst rows = await db.query(sql);\n${CODE_FENCE}\nBy the way, ${lowerFirst(userText)}`,
                `ASSISTANT: ${rng.pick(ASSISTANT_ACKS)}`,
              );
              break;
            case 'embedded': {
              // the fact sits inside a long request; the assistant answers at length,
              // sometimes mentioning an unrelated fact of its own that must not be stored
              const advice = rng.pick(ASSISTANT_ADVICE);
              const aside =
                rng.next() < 0.5 ? ` ${capitalFirst(guessText)}` : '';
              turns.push(
                `USER: ${rng.pick(USER_REQUESTS)} ${userText} ${rng.pick(USER_FOLLOWUPS)}`,
                `ASSISTANT: ${advice}${aside}`,
              );
              if (rng.next() < 0.5) {
                turns.push(
                  `USER: ${rng.pick(USER_FOLLOWUPS)}`,
                  `ASSISTANT: ${rng.pick(ASSISTANT_ADVICE)}`,
                );
              }
              break;
            }
          }
          out.push({
            world: world.id,
            kind: 'transcript',
            input: turns.join('\n\n'),
            initialProgram: bareSchema('transcript', [])
              ? ''
              : initial.join('\n'),
            expectedAdded: added,
            expectedRetract: [],
            mode: 'transcript',
          });
        }
      }
    }

    // event: an episodic aside inside a request, in the first person. Half the examples
    // are plain text (the remember path), half are USER/ASSISTANT transcripts. Two per
    // round because real transcripts carry most memorable facts this way.
    for (let k = 0; k < 2; k += 1) {
      const template = rng.pick(EVENT_TEMPLATES);
      const values = template.args(rng);
      const quoted = values.map((v) => (/\s|[A-Z]/.test(v) ? `'${v}'` : v));
      const fact = parseProgram(
        factText(template.predicate, [options.selfAtom, ...quoted]),
      )[0];
      const relation: Relation = {
        name: template.predicate,
        kind: 'attribute',
        args: [...template.argNames],
      };
      // proper names keep their capitals; common nouns are spoken as words
      const display: Record<string, string> = {};
      for (const v of values)
        display[v] = v.includes('_') ? v.replaceAll('_', ' ') : v;
      const sentence = await renderVerified([fact], relation, {
        firstPerson: true,
        negated: false,
        hedged: false,
        display,
      });
      if (sentence === undefined) continue;
      const userTurn = `${rng.pick(USER_REQUESTS)} ${rng.pick(ASIDE_LEADINS)} ${lowerFirst(sentence)} ${rng.pick(USER_FOLLOWUPS)}`;
      const otherSubject = rng.pick(world.entities);
      const seed = `${template.predicate}(${otherSubject}, ${template
        .args(rng)
        .map((v) => (/\s|[A-Z]/.test(v) ? `'${v}'` : v))
        .join(', ')}).`;
      const transcript = rng.next() < 0.5;
      out.push({
        world: world.id,
        kind: 'event',
        input: transcript
          ? [
              `USER: ${userTurn}`,
              `ASSISTANT: ${rng.pick(ASSISTANT_ADVICE)}`,
            ].join('\n\n')
          : userTurn,
        initialProgram: bareSchema('event', []) ? '' : seed,
        expectedAdded: [serializeClause(fact)],
        expectedRetract: [],
        mode: transcript ? 'transcript' : 'text',
      });
    }

    // distractor: one real fact preceded by noise prose
    const rel2 = rng.pick(pool);
    const real = factsOf(rel2);
    if (real.length > 0) {
      const fact = rng.pick(real);
      await attempt(
        'distractor',
        [fact],
        rel2,
        {
          firstPerson: false,
          negated: false,
          hedged: false,
          distractor: `${rng.pick(NOISE)} `,
        },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: schemaSeed(
            world,
            [rel2.name],
            new Set([String((fact.head.args[0] as { value: string }).value)]),
          ),
        },
      );
    }
  }
  return out;
}

/** The product's extraction prompt over the example's store, the text, and the exact clause lines. */
export function toExtractionConversation(
  world: World,
  example: ExtractionExample,
  selfAtom: string,
): {
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
} {
  const schema = buildSchemaSummary(parseProgram(example.initialProgram));
  // transcript capture is additive: the transcript prompt forbids retract lines
  const answer =
    example.mode === 'transcript'
      ? example.expectedAdded
      : [
          ...example.expectedRetract.map((pattern) => `retract ${pattern}.`),
          ...example.expectedAdded,
        ];
  void world;
  return {
    messages: [
      {
        role: 'system',
        content:
          example.mode === 'transcript'
            ? transcriptExtractionSystemPrompt(schema, selfAtom)
            : extractionSystemPrompt(schema, 'accepted', selfAtom),
      },
      { role: 'user', content: example.input },
      {
        role: 'assistant',
        content: answer.length === 0 ? NOTHING_SENTINEL : answer.join('\n'),
      },
    ],
  };
}
