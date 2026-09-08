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
 */
import { type Clause, parseProgram, serializeClause } from '../engine/index.js';
import {
  NOTHING_SENTINEL,
  buildSchemaSummary,
  extractionSystemPrompt,
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
  | 'distractor';

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
  mode: 'text';
}

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
      (value, i) => `${request.argNames[i] ?? `arg${i + 1}`}=${value}`,
    );
    return `- ${fact.predicate}(${named.join(', ')})`;
  });
  const voice = request.firstPerson
    ? `Write in the first person: the entity "${request.selfAtom}" is the speaker, so refer to it as I / me / my and never by name.`
    : 'Write in the third person, naming the entities exactly as given.';
  const polarity = request.negated
    ? 'State that these facts are NO LONGER true or are NOT the case (use "no longer", "does not", "never"), so a careful reader would store nothing new.'
    : request.hedged
      ? 'State these as uncertain possibilities ("might", "maybe", "is considering"), so a careful reader would store nothing.'
      : 'State them as settled facts.';
  return `Write one or two short natural sentences, as a person telling an assistant something worth remembering, that convey exactly these facts and nothing else:
${lines.join('\n')}

${voice}
${polarity}
Use every constant exactly as written (lowercase, underscores may become spaces). Do not add any other names, numbers, or facts. Reply with the sentences only.`;
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
    if (constant === selfAtom) continue;
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

  const attempt = async (
    kind: ExtractionKind,
    facts: Clause[],
    relation: Relation,
    request: Omit<RenderRequest, 'facts' | 'argNames' | 'selfAtom'>,
    expected: { added: string[]; retract: string[]; initial: string[] },
  ) => {
    const text = await render({
      facts: facts.map(factSpec),
      argNames: relation.args,
      selfAtom: options.selfAtom,
      ...request,
    });
    if (!verifyRendering(text, facts, world, options.selfAtom)) return;
    out.push({
      world: world.id,
      kind,
      input: text,
      initialProgram: expected.initial.join('\n'),
      expectedAdded: expected.added,
      expectedRetract: expected.retract,
      mode: 'text',
    });
  };

  const pool = [...attributes, ...edges];
  if (pool.length === 0) return out;

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

    // first_person: rewrite one attribute fact's subject to the self atom
    const attr = attributes.length > 0 ? rng.pick(attributes) : undefined;
    if (attr) {
      const sample = rng.pick(factsOf(attr));
      const value = String((sample.head.args[1] as { value: string }).value);
      const fact = parseProgram(
        factText(attr.name, [options.selfAtom, value]),
      )[0];
      await attempt(
        'first_person',
        [fact],
        attr,
        { firstPerson: true, negated: false, hedged: false },
        {
          added: [serializeClause(fact)],
          retract: [],
          initial: schemaSeed(world, [attr.name], new Set([options.selfAtom])),
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
  const answer = [
    ...example.expectedRetract.map((pattern) => `retract ${pattern}.`),
    ...example.expectedAdded,
  ];
  void world;
  return {
    messages: [
      {
        role: 'system',
        content: extractionSystemPrompt(schema, 'accepted', selfAtom),
      },
      { role: 'user', content: example.input },
      {
        role: 'assistant',
        content: answer.length === 0 ? NOTHING_SENTINEL : answer.join('\n'),
      },
    ],
  };
}
