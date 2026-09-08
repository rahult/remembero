/**
 * Extraction benchmark: 100+ cases tagged by the phenomenon they exercise.
 *
 * The 15-case extraction eval is saturated (frontier models score 100%) and
 * has no negation, coreference, first-person input, transcript mode, or
 * realistic competitor predicates. This suite covers those, reports accuracy
 * per phenomenon, and measures predicate drift: added facts whose predicate is
 * neither expected nor already in the store. Scoring reuses the eval's exact
 * canonical-set comparison; drift is reported alongside, not forgiven.
 *
 * Conventions follow the extraction prompt: lowercase snake_case atoms,
 * multi-word or case-sensitive constants in single quotes, numbers bare,
 * supersession as retract-and-add, removal via "no longer". First-person
 * input maps to the SELF atom below until a configured self atom exists.
 *
 * Every case is schema-conditioned: one neutral fact per expected predicate is
 * seeded into the store (entity `zed`, values `sample_k`), as the product
 * always sends the store's schema with the prompt. On an empty store any
 * predicate name is legitimate, so unconditioned scoring would measure taste.
 */
import { parseProgram, predKey, serializeClause } from '../engine/index.js';
import {
  extractionObservationIsCorrect,
  scoreExtractionEval,
  type ExtractionEvalCase,
  type ExtractionEvalObservation,
  type ExtractionEvalScore,
} from './extraction.js';

export const EXTRACTION_BENCH_PHENOMENA = [
  'direct',
  'multi_fact',
  'negation',
  'coreference',
  'first_person',
  'supersession',
  'distractor_prose',
  'competitor_predicate',
  'hedge',
  'transcript',
  'entity_normalization',
  'date_number',
] as const;

export type ExtractionPhenomenon = (typeof EXTRACTION_BENCH_PHENOMENA)[number];

export interface ExtractionBenchCase extends ExtractionEvalCase {
  phenomenon: ExtractionPhenomenon;
  /** 'text' runs rememberText; 'transcript' runs rememberTranscriptText on USER:/ASSISTANT: blocks. */
  mode: 'text' | 'transcript';
}

/** First-person subject until a configured self atom exists (see the adversarial review). */
export const SELF = 'user';

interface Spec {
  id: string;
  phenomenon: ExtractionPhenomenon;
  input: string;
  initial?: string;
  final: string;
  added?: string;
  retractions?: number;
  mode?: 'text' | 'transcript';
  /**
   * Predicates the store already knows (as `name/arity`). One neutral fact per
   * predicate is seeded into the initial and final programs so the model is
   * schema-conditioned, as in the product. Defaults to the predicates of
   * `final`; pass explicit keys for cases whose correct output is empty but
   * whose tempting predicate exists (negation, hedge).
   */
  vocab?: string[];
}

/** Neutral entity and values for vocabulary seeds: never referenced by any input. */
const SEED_ENTITY = 'zed';

function seedFact(key: string): string {
  const [name, arityText] = key.split('/');
  const arity = Number(arityText);
  const args = [
    SEED_ENTITY,
    ...Array.from(
      { length: Math.max(arity - 1, 0) },
      (_, k) => `sample_${k + 1}`,
    ),
  ].slice(0, arity);
  return arity === 0 ? `${name}.` : `${name}(${args.join(', ')}).`;
}

function make(spec: Spec): ExtractionBenchCase {
  const authoredInitial = parseProgram(spec.initial ?? '').map(serializeClause);
  const known = new Set(
    parseProgram(spec.initial ?? '').map((c) => predKey(c.head)),
  );
  const vocab = spec.vocab ?? [
    ...new Set(parseProgram(spec.final).map((c) => predKey(c.head))),
  ];
  const seeds = vocab.filter((key) => !known.has(key)).map(seedFact);
  const initialProgram = [...seeds, ...authoredInitial].join('\n');
  const finalProgram = [
    ...seeds,
    ...parseProgram(spec.final).map(serializeClause),
  ].join('\n');
  const initialSet = new Set(parseProgram(initialProgram).map(serializeClause));
  const added =
    spec.added ??
    parseProgram(finalProgram)
      .map(serializeClause)
      .filter((clause) => !initialSet.has(clause))
      .join('\n');
  return {
    id: spec.id,
    phenomenon: spec.phenomenon,
    mode: spec.mode ?? 'text',
    input: spec.input,
    initialProgram,
    expectedFinalProgram: finalProgram,
    expectedAddedProgram: added,
    expectedOutcome: 'completed',
    expectedDuplicates: 0,
    expectedRetractions: spec.retractions ?? 0,
    tags: [spec.phenomenon],
  };
}

const PEOPLE = [
  ['Mira Chen', 'mira'],
  ['Tom Okafor', 'tom'],
  ['Priya Nair', 'priya'],
  ['Liam Brandt', 'liam'],
  ['Ava Ruiz', 'ava'],
  ['Dana Kowalski', 'dana'],
] as const;
const COMPANIES = [
  'acme',
  'initech',
  'globex',
  'umbrella',
  'hooli',
  'vandelay',
] as const;
const CITIES = [
  'melbourne',
  'sydney',
  'lisbon',
  'osaka',
  'toronto',
  'nairobi',
] as const;
const LANGS = [
  'python',
  'rust',
  'go',
  'typescript',
  'kotlin',
  'elixir',
] as const;

function cap(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}

const cases: ExtractionBenchCase[] = [];

// ---- direct ----------------------------------------------------------------
PEOPLE.forEach(([full, atom], i) => {
  const first = full.split(' ')[0];
  const company = COMPANIES[i];
  cases.push(
    make({
      id: `direct_${i}`,
      phenomenon: 'direct',
      input: `${first} works at ${cap(company)}.`,
      final: `works_at(${atom}, ${company}).`,
    }),
  );
});
CITIES.forEach((city, i) => {
  const [, atom] = PEOPLE[i];
  cases.push(
    make({
      id: `direct_city_${i}`,
      phenomenon: 'direct',
      input: `${cap(atom)} lives in ${cap(city)}.`,
      final: `lives_in(${atom}, ${city}).`,
    }),
  );
});

// ---- multi_fact ---------------------------------------------------------------
PEOPLE.forEach(([, atom], i) => {
  const company = COMPANIES[i];
  const city = CITIES[(i + 1) % CITIES.length];
  const lang = LANGS[i];
  cases.push(
    make({
      id: `multi_${i}`,
      phenomenon: 'multi_fact',
      input: `${cap(atom)} works at ${cap(company)}, lives in ${cap(city)}, and prefers ${cap(lang)}.`,
      final: `works_at(${atom}, ${company}). lives_in(${atom}, ${city}). prefers_language(${atom}, ${lang}).`,
    }),
  );
});
cases.push(
  make({
    id: 'multi_two_people',
    phenomenon: 'multi_fact',
    input: 'Mira and Tom both work at Acme; Tom lives in Osaka.',
    final: 'works_at(mira, acme). works_at(tom, acme). lives_in(tom, osaka).',
  }),
  make({
    id: 'multi_relationship',
    phenomenon: 'multi_fact',
    input: 'Priya reports to Liam, and Liam reports to Dana.',
    final: 'reports_to(priya, liam). reports_to(liam, dana).',
  }),
);

// ---- negation -----------------------------------------------------------------
PEOPLE.slice(0, 3).forEach(([, atom], i) => {
  const company = COMPANIES[i];
  cases.push(
    make({
      id: `negation_absent_${i}`,
      phenomenon: 'negation',
      input: `${cap(atom)} does not work at ${cap(company)}.`,
      final: '',
      vocab: ['works_at/2'],
    }),
  );
});
PEOPLE.slice(0, 3).forEach(([, atom], i) => {
  const company = COMPANIES[i];
  cases.push(
    make({
      id: `negation_retract_${i}`,
      phenomenon: 'negation',
      input: `${cap(atom)} no longer works at ${cap(company)}.`,
      initial: `works_at(${atom}, ${company}).`,
      final: '',
      retractions: 1,
    }),
  );
});
cases.push(
  make({
    id: 'negation_never',
    phenomenon: 'negation',
    input: 'Ava has never lived in Lisbon.',
    final: '',
    vocab: ['lives_in/2'],
  }),
  make({
    id: 'negation_mixed',
    phenomenon: 'negation',
    input: 'Dana lives in Toronto but does not work at Hooli.',
    final: 'lives_in(dana, toronto).',
    vocab: ['lives_in/2', 'works_at/2'],
  }),
);

// ---- coreference --------------------------------------------------------------
PEOPLE.forEach(([full, atom], i) => {
  const first = full.split(' ')[0];
  const company = COMPANIES[i];
  const city = CITIES[i];
  const pronoun = i % 2 === 0 ? 'She' : 'He';
  cases.push(
    make({
      id: `coref_${i}`,
      phenomenon: 'coreference',
      input: `${first} joined ${cap(company)} last year. ${pronoun} lives in ${cap(city)}.`,
      final: `works_at(${atom}, ${company}). lives_in(${atom}, ${city}).`,
    }),
  );
});
cases.push(
  make({
    id: 'coref_their',
    phenomenon: 'coreference',
    input: 'Tom leads the platform team. Their standup is on Tuesdays.',
    final: 'leads(tom, platform_team). standup_day(platform_team, tuesday).',
  }),
  make({
    id: 'coref_two_sentences_back',
    phenomenon: 'coreference',
    input:
      'Priya moved teams. The new team ships weekly. She now reports to Ava.',
    final: 'reports_to(priya, ava).',
  }),
);

// ---- first_person ---------------------------------------------------------------
[
  ['I live in Melbourne.', `lives_in(${SELF}, melbourne).`],
  ['I work at Acme.', `works_at(${SELF}, acme).`],
  ['My dentist is Dr Chen.', `dentist(${SELF}, dr_chen).`],
  ['I prefer dark mode.', `prefers(${SELF}, dark_mode).`],
  ['My manager is Liam.', `reports_to(${SELF}, liam).`],
  ["I'm allergic to peanuts.", `allergic_to(${SELF}, peanuts).`],
  [
    'We use Postgres for the ledger service.',
    `uses_database(ledger_service, postgres).`,
  ],
].forEach(([input, final], i) => {
  cases.push(
    make({ id: `first_person_${i}`, phenomenon: 'first_person', input, final }),
  );
});

// ---- supersession ---------------------------------------------------------------
PEOPLE.forEach(([, atom], i) => {
  const from = COMPANIES[i];
  const to = COMPANIES[(i + 1) % COMPANIES.length];
  cases.push(
    make({
      id: `supersede_company_${i}`,
      phenomenon: 'supersession',
      input: `${cap(atom)} has moved to ${cap(to)}; that is where ${i % 2 === 0 ? 'she' : 'he'} works now.`,
      initial: `works_at(${atom}, ${from}).`,
      final: `works_at(${atom}, ${to}).`,
      retractions: 1,
    }),
  );
});
cases.push(
  make({
    id: 'supersede_city_moved',
    phenomenon: 'supersession',
    input: 'Mira moved from Melbourne to Sydney.',
    initial: 'lives_in(mira, melbourne).',
    final: 'lives_in(mira, sydney).',
    retractions: 1,
  }),
  make({
    id: 'supersede_manager',
    phenomenon: 'supersession',
    input: 'Priya reports to Dana from now on.',
    initial: 'reports_to(priya, liam). reports_to(liam, dana).',
    final: 'reports_to(priya, dana). reports_to(liam, dana).',
    retractions: 1,
  }),
);

// ---- distractor_prose -------------------------------------------------------------
const PROSE = [
  'Quick update after a long week of meetings and a broken build that took ages to fix.',
  'Not much to report on the migration; the vendor call slipped again and the docs are stale.',
  'Reminder that the office kitchen is closed on Friday for cleaning.',
];
PEOPLE.forEach(([, atom], i) => {
  const company = COMPANIES[i];
  const prose = PROSE[i % PROSE.length];
  cases.push(
    make({
      id: `distractor_${i}`,
      phenomenon: 'distractor_prose',
      input: `${prose} Anyway, ${cap(atom)} works at ${cap(company)} these days. Talk soon.`,
      final: `works_at(${atom}, ${company}).`,
    }),
  );
});
cases.push(
  make({
    id: 'distractor_code',
    phenomenon: 'distractor_prose',
    input:
      'Ran `npm test` and got 3 failures in cache.test.ts (timeout). Unrelated: Liam lives in Lisbon now.',
    final: 'lives_in(liam, lisbon).',
  }),
  make({
    id: 'distractor_only_noise',
    phenomenon: 'distractor_prose',
    input: 'Build is red, retrying CI. Will circle back after lunch.',
    final: '',
    vocab: ['works_at/2', 'lives_in/2'],
  }),
);

// ---- competitor_predicate ----------------------------------------------------------
// The store already uses a different predicate name for the same relation;
// the model must reuse it rather than invent a synonym.
const COMPETITORS: Array<[string, string, string]> = [
  ['employer', 'works at', 'employer'],
  ['works_for', 'works at', 'works_for'],
  ['employed_by', 'is employed by', 'employed_by'],
  ['resides_in', 'lives in', 'resides_in'],
  ['home_city', 'lives in', 'home_city'],
  ['manager_of', 'is managed by', 'manager_of'],
  ['based_in', 'lives in', 'based_in'],
  ['job_at', 'works at', 'job_at'],
];
COMPETITORS.forEach(([existing, verb, predicate], i) => {
  const [, atom] = PEOPLE[i % PEOPLE.length];
  const [, other] = PEOPLE[(i + 1) % PEOPLE.length];
  const value =
    verb === 'lives in'
      ? CITIES[i % CITIES.length]
      : COMPANIES[i % COMPANIES.length];
  const managerCase = predicate === 'manager_of';
  const seedValue =
    verb === 'lives in'
      ? CITIES[(i + 2) % CITIES.length]
      : COMPANIES[(i + 2) % COMPANIES.length];
  const initial = managerCase
    ? `${existing}(${PEOPLE[(i + 2) % PEOPLE.length][1]}, ${other}).`
    : `${existing}(${other}, ${seedValue}).`;
  const final = managerCase
    ? `${initial} ${predicate}(liam, ${atom}).`
    : `${initial} ${predicate}(${atom}, ${value}).`;
  const input = managerCase
    ? `${cap(atom)} ${verb} Liam.`
    : `${cap(atom)} ${verb} ${cap(value)}.`;
  cases.push(
    make({
      id: `competitor_${i}`,
      phenomenon: 'competitor_predicate',
      input,
      initial,
      final,
    }),
  );
});

// ---- hedge ------------------------------------------------------------------------
[
  'Mira might move to Sydney next year.',
  'Tom is considering switching to Rust.',
  'Priya may be joining Globex, nothing confirmed.',
  'I think Liam works at Hooli, but I am not sure.',
  'Ava could end up leading the data team.',
  'Dana probably lives near Toronto.',
].forEach((input, i) => {
  cases.push(
    make({
      id: `hedge_${i}`,
      phenomenon: 'hedge',
      input,
      final: '',
      vocab: ['lives_in/2', 'works_at/2', 'prefers_language/2', 'leads/2'],
    }),
  );
});

// ---- transcript -------------------------------------------------------------------
const transcript = (turns: Array<['USER' | 'ASSISTANT', string]>) =>
  turns.map(([role, text]) => `${role}: ${text}`).join('\n\n');
cases.push(
  make({
    id: 'transcript_user_fact',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      [
        'USER',
        'For context, I work at Acme and our main service is called ledger.',
      ],
      ['ASSISTANT', 'Got it. Shall I look at the ledger service tests?'],
    ]),
    final: `works_at(${SELF}, acme).`,
  }),
  make({
    id: 'transcript_assistant_guess_ignored',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['USER', 'Can you check why the build is red?'],
      [
        'ASSISTANT',
        'It looks like you use Postgres 15 and the driver is outdated.',
      ],
    ]),
    final: '',
    vocab: ['runs_on/2'],
  }),
  make({
    id: 'transcript_user_confirms_assistant',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['ASSISTANT', 'Should I assume the ledger service runs on Postgres?'],
      ['USER', 'Yes, ledger runs on Postgres.'],
    ]),
    final: 'runs_on(ledger, postgres).',
  }),
  make({
    id: 'transcript_preference',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['USER', 'Please always use tabs, not spaces, in this repo.'],
      ['ASSISTANT', 'Understood, tabs it is.'],
    ]),
    final: `prefers_indentation(${SELF}, tabs).`,
  }),
  make({
    id: 'transcript_tool_output_ignored',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['USER', 'Run the tests.'],
      [
        'ASSISTANT',
        '$ npm test\n47 passed, 2 failed (auth.test.ts, cache.test.ts)',
      ],
    ]),
    final: '',
    vocab: ['works_at/2'],
  }),
  make({
    id: 'transcript_two_user_facts',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['USER', 'My manager is Dana and I am based in Melbourne.'],
      ['ASSISTANT', 'Noted.'],
      ['USER', 'Also we deploy on Fridays.'],
    ]),
    final: `reports_to(${SELF}, dana). lives_in(${SELF}, melbourne). deploy_day(team, friday).`,
  }),
);

// ---- entity_normalization ----------------------------------------------------------
[
  ['Dr. Mira Chen is my dentist.', `dentist(${SELF}, 'Mira Chen').`],
  [
    "Tom's company is called Blue Harbour Analytics.",
    `works_at(tom, 'Blue Harbour Analytics').`,
  ],
  ['Priya lives in New York.', `lives_in(priya, 'New York').`],
  ['The project is code-named Orchard.', `codename(project, orchard).`],
  ['Liam works at ACME Corp.', `works_at(liam, 'ACME Corp').`],
  ['Ava speaks Mandarin Chinese.', `speaks(ava, 'Mandarin Chinese').`],
].forEach(([input, final], i) => {
  cases.push(
    make({
      id: `normalize_${i}`,
      phenomenon: 'entity_normalization',
      input,
      final,
    }),
  );
});

// ---- date_number ------------------------------------------------------------------
[
  [
    'The filing deadline is 2026-10-31.',
    `filing_deadline(filing, '2026-10-31').`,
  ],
  [
    'Mira started at Acme on 2024-03-01.',
    `started_at(mira, acme, '2024-03-01').`,
  ],
  ['The team has 7 engineers.', `team_size(team, 7).`],
  ['Our SLA is 99.9 percent uptime.', `sla_uptime_percent(service, 99.9).`],
  ['Tom turns 40 in 2027.', `turns_age_in(tom, 40, 2027).`],
  ['The budget is 250000 dollars.', `budget_dollars(project, 250000).`],
].forEach(([input, final], i) => {
  cases.push(
    make({ id: `date_number_${i}`, phenomenon: 'date_number', input, final }),
  );
});

// ---- extra coverage to keep every phenomenon at eight or more -----------------------
[
  'Ava is thinking about learning Kotlin.',
  'Liam may relocate to Nairobi if the offer comes through.',
  'We might switch the ledger service to MySQL, undecided.',
].forEach((input, i) => {
  cases.push(
    make({
      id: `hedge_more_${i}`,
      phenomenon: 'hedge',
      input,
      final: '',
      vocab: ['prefers_language/2', 'lives_in/2', 'uses_database/2'],
    }),
  );
});
cases.push(
  make({
    id: 'negation_not_anymore_manager',
    phenomenon: 'negation',
    input: 'Liam is no longer managed by Dana.',
    initial: 'reports_to(liam, dana).',
    final: '',
    retractions: 1,
  }),
  make({
    id: 'negation_keep_other_facts',
    phenomenon: 'negation',
    input: 'Mira has stopped using Rust; she still lives in Melbourne.',
    initial: 'prefers_language(mira, rust). lives_in(mira, melbourne).',
    final: 'lives_in(mira, melbourne).',
    retractions: 1,
  }),
  make({
    id: 'transcript_code_and_fact',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      [
        'USER',
        'Here is the failing snippet:\n```ts\nconst x = cache.get(key);\n```\nBy the way, I prefer pnpm over npm.',
      ],
      ['ASSISTANT', 'Thanks. The cache call is missing an await.'],
    ]),
    final: `prefers_package_manager(${SELF}, pnpm).`,
  }),
  make({
    id: 'transcript_assistant_summary_ignored',
    phenomenon: 'transcript',
    mode: 'transcript',
    input: transcript([
      ['USER', 'Summarise what we did.'],
      [
        'ASSISTANT',
        'We migrated the ledger service to Postgres and set the deploy day to Friday.',
      ],
    ]),
    final: '',
    vocab: ['runs_on/2', 'deploy_day/2'],
  }),
  make({
    id: 'date_number_version',
    phenomenon: 'date_number',
    input: 'The ledger service runs Node 22.',
    final: 'runs_node_version(ledger_service, 22).',
  }),
  make({
    id: 'date_number_time',
    phenomenon: 'date_number',
    input: 'Standup is at 9:30 every weekday.',
    final: "standup_time(team, '9:30').",
  }),
  make({
    id: 'normalize_hyphen',
    phenomenon: 'entity_normalization',
    input: 'Dana works on the check-out service.',
    final: 'works_on(dana, checkout_service).',
  }),
  make({
    id: 'normalize_acronym',
    phenomenon: 'entity_normalization',
    input: 'Tom is the on-call for the API gateway.',
    final: 'on_call_for(tom, api_gateway).',
  }),
  make({
    id: 'first_person_possessive_team',
    phenomenon: 'first_person',
    input: 'My team ships on Fridays.',
    final: 'deploy_day(team, friday).',
  }),
);

export const EXTRACTION_BENCH_CASES: readonly ExtractionBenchCase[] = cases;

/** Added clauses whose predicate/arity is neither expected nor already stored: vocabulary drift. */
export function predicateDrift(
  testCase: ExtractionBenchCase,
  actualClauses: readonly string[],
): string[] {
  const allowed = new Set([
    ...parseProgram(testCase.initialProgram).map((c) => predKey(c.head)),
    ...parseProgram(testCase.expectedFinalProgram).map((c) => predKey(c.head)),
  ]);
  const drift = new Set<string>();
  for (const clause of parseProgram(actualClauses.join('\n'))) {
    const key = predKey(clause.head);
    if (!allowed.has(key)) drift.add(key);
  }
  return [...drift].sort();
}

export interface ExtractionBenchScore extends ExtractionEvalScore {
  byPhenomenon: Record<
    string,
    { cases: number; correct: number; accuracy: number }
  >;
  /** Fraction of cases where the model introduced a predicate outside the expected/stored vocabulary. */
  driftRate: number;
  driftedPredicates: Record<string, number>;
}

export function scoreExtractionBench(
  observations: ExtractionEvalObservation[],
): ExtractionBenchScore {
  const base = scoreExtractionEval(observations);
  const byPhenomenon: ExtractionBenchScore['byPhenomenon'] = {};
  const driftedPredicates: Record<string, number> = {};
  let drifted = 0;
  for (const observation of observations) {
    const testCase = observation.case as ExtractionBenchCase;
    const bucket = (byPhenomenon[testCase.phenomenon] ??= {
      cases: 0,
      correct: 0,
      accuracy: 0,
    });
    bucket.cases += 1;
    if (extractionObservationIsCorrect(observation)) bucket.correct += 1;
    let drift: string[] = [];
    try {
      drift = predicateDrift(testCase, observation.actualClauses);
    } catch {
      drift = [];
    }
    if (drift.length > 0) {
      drifted += 1;
      for (const key of drift)
        driftedPredicates[key] = (driftedPredicates[key] ?? 0) + 1;
    }
  }
  for (const bucket of Object.values(byPhenomenon)) {
    bucket.accuracy = bucket.cases === 0 ? 0 : bucket.correct / bucket.cases;
  }
  return {
    ...base,
    byPhenomenon,
    driftRate: observations.length === 0 ? 0 : drifted / observations.length,
    driftedPredicates,
  };
}
