/**
 * Agent-boundary benchmark: same seeded SQLite database, same small model,
 * three tool boundaries. One condition lets the model author arbitrary
 * read-only SQL applied without an integrity gate; a second authors the same
 * SQL but routes writes through the Remembero integrity gate (sql-gated,
 * isolating the gate from the query language); the third lets it author
 * arbitrary Datalog served by the Remembero bridge. The model writes every
 * query itself — nothing is prepared per question. Few-shot budgets are
 * deliberately asymmetric (ADR 0002): the Datalog condition receives a
 * syntax cheatsheet and more examples than SQL, because identical
 * in-context budgets cannot equalize the models' training-data prior.
 * Categories deliberately include
 * ground the baseline should tie on; expected answers are verified by
 * gold queries in the test suite so the benchmark cannot drift into fiction.
 */

export const AGENT_BOUNDARY_CONDITIONS = [
  'sql',
  'sql-gated',
  'remembero',
] as const;

export type AgentBoundaryCondition = (typeof AGENT_BOUNDARY_CONDITIONS)[number];

export type AgentBoundaryCategory =
  'direct' | 'join' | 'multihop' | 'absence' | 'write-trap';

export interface AgentBoundaryQuestion {
  id: string;
  category: AgentBoundaryCategory;
  question: string;
  /** Every term must appear in the normalized final answer. */
  expect: string[];
  /** No term may appear in the normalized final answer. */
  forbid?: string[];
  /** Gold queries that must reproduce the expected terms (test-enforced). */
  goldSql: string;
  goldDatalog: string;
  /**
   * write-trap only: a corrupting write the harness pushes through each
   * condition's write path before asking. Raw SQL applies it silently; the
   * gated conditions (sql-gated, remembero) check the integrity rules and
   * refuse.
   */
  trapWriteSql?: string;
  /**
   * control trap (ADR 0003): the write violates NO gate rule, so the gate
   * must NOT refuse it and the expected terms describe the POST-write state.
   * Proves the gate enforces declared rules rather than rejecting writes.
   */
  control?: boolean;
}

export const AGENT_BOUNDARY_SEED_SQL = `
  DROP TABLE IF EXISTS works_on;
  DROP TABLE IF EXISTS reports_to;
  DROP TABLE IF EXISTS status;
  DROP TABLE IF EXISTS blocker;
  DROP TABLE IF EXISTS waits_on;
  DROP TABLE IF EXISTS prefers_meeting;
  DROP TABLE IF EXISTS review_slot;
  DROP TABLE IF EXISTS promised_update;

  CREATE TABLE works_on(person TEXT NOT NULL, project TEXT NOT NULL);
  CREATE TABLE reports_to(person TEXT NOT NULL, manager TEXT NOT NULL);
  CREATE TABLE status(project TEXT NOT NULL, state TEXT NOT NULL);
  CREATE TABLE blocker(project TEXT NOT NULL, blocker TEXT NOT NULL);
  CREATE TABLE waits_on(item TEXT NOT NULL, upstream TEXT NOT NULL);
  CREATE TABLE prefers_meeting(person TEXT NOT NULL, window TEXT NOT NULL);
  CREATE TABLE review_slot(project TEXT NOT NULL, day TEXT NOT NULL, window TEXT NOT NULL);
  CREATE TABLE promised_update(owner TEXT NOT NULL, person TEXT NOT NULL, project TEXT NOT NULL);

  INSERT INTO works_on VALUES
    ('maya', 'atlas'), ('liam', 'atlas'),
    ('nora', 'orchard'), ('ava', 'orchard'),
    ('priya', 'beacon'), ('tom', 'beacon'), ('maya', 'beacon');
  INSERT INTO reports_to VALUES
    ('maya', 'liam'), ('liam', 'ava'), ('ava', 'dana'),
    ('priya', 'ava'), ('tom', 'liam'), ('nora', 'dana');
  INSERT INTO status VALUES
    ('atlas', 'blocked'), ('orchard', 'active'),
    ('beacon', 'blocked'), ('quasar', 'active');
  INSERT INTO blocker VALUES
    ('atlas', 'vendor_security_review'), ('beacon', 'legal_signoff');
  INSERT INTO waits_on VALUES
    ('atlas', 'vendor_security_review'),
    ('vendor_security_review', 'legal_signoff'),
    ('legal_signoff', 'procurement_freeze'),
    ('beacon', 'legal_signoff');
  INSERT INTO prefers_meeting VALUES
    ('maya', 'morning'), ('ava', 'afternoon'), ('tom', 'morning');
  INSERT INTO review_slot VALUES
    ('atlas', 'tuesday', 'morning'), ('orchard', 'thursday', 'afternoon');
  INSERT INTO promised_update VALUES
    ('rahul', 'maya', 'atlas'), ('rahul', 'nora', 'orchard'), ('rahul', 'priya', 'beacon');
`;

/**
 * Integrity rules the Remembero write gate enforces before any write commits.
 * Each is a one-line Datalog rule over the same tables the queries use.
 */
export const WRITE_GATE_RULES: ReadonlyArray<{
  name: string;
  program: string;
}> = [
  {
    name: 'active project keeps a blocker',
    program: `violation(P, B) :- status(P, active), blocker(P, B).`,
  },
  {
    name: 'blocked project has no blocker recorded',
    program: `violation(P) :- status(P, blocked), \\+ blocker(P, _).`,
  },
  {
    name: 'project has two contradictory states',
    program: `violation(P, S1, S2) :- status(P, S1), status(P, S2), S1 != S2.`,
  },
];

export const AGENT_BOUNDARY_QUESTIONS: readonly AgentBoundaryQuestion[] = [
  // ---- direct lookups: plain SQL should tie here -------------------------
  {
    id: 'd1',
    category: 'direct',
    question: 'What is the status of the atlas project?',
    expect: ['blocked'],
    goldSql: `SELECT state FROM status WHERE project = 'atlas'`,
    goldDatalog: `atlas_state(S) :- status(atlas, S).`,
  },
  {
    id: 'd2',
    category: 'direct',
    question: 'Who works on the orchard project?',
    expect: ['nora', 'ava'],
    goldSql: `SELECT person FROM works_on WHERE project = 'orchard'`,
    goldDatalog: `orchard_person(P) :- works_on(P, orchard).`,
  },
  {
    id: 'd3',
    category: 'direct',
    question: 'What blocker is recorded for the beacon project?',
    expect: ['legal signoff'],
    goldSql: `SELECT blocker FROM blocker WHERE project = 'beacon'`,
    goldDatalog: `beacon_blocker(B) :- blocker(beacon, B).`,
  },
  {
    id: 'd4',
    category: 'direct',
    question: 'On which day and in which window is the atlas review slot?',
    expect: ['tuesday', 'morning'],
    goldSql: `SELECT day, window FROM review_slot WHERE project = 'atlas'`,
    goldDatalog: `atlas_slot(D, W) :- review_slot(atlas, D, W).`,
  },
  {
    id: 'd5',
    category: 'direct',
    question: "Who is maya's direct manager?",
    expect: ['liam'],
    goldSql: `SELECT manager FROM reports_to WHERE person = 'maya'`,
    goldDatalog: `maya_manager(M) :- reports_to(maya, M).`,
  },
  {
    id: 'd6',
    category: 'direct',
    question: 'Which projects are currently blocked?',
    expect: ['atlas', 'beacon'],
    goldSql: `SELECT project FROM status WHERE state = 'blocked'`,
    goldDatalog: `blocked_project(P) :- status(P, blocked).`,
  },
  // ---- single joins: plain SQL should tie here ---------------------------
  {
    id: 'j1',
    category: 'join',
    question: 'Which blocked projects does maya work on?',
    expect: ['atlas', 'beacon'],
    goldSql: `SELECT w.project FROM works_on w JOIN status s ON s.project = w.project WHERE w.person = 'maya' AND s.state = 'blocked'`,
    goldDatalog: `maya_blocked(P) :- works_on(maya, P), status(P, blocked).`,
  },
  {
    id: 'j2',
    category: 'join',
    question:
      'Who was promised an update about a project that is currently blocked?',
    expect: ['maya', 'priya'],
    forbid: ['nora'],
    goldSql: `SELECT u.person FROM promised_update u JOIN status s ON s.project = u.project WHERE s.state = 'blocked'`,
    goldDatalog: `owed(P) :- promised_update(_, P, Proj), status(Proj, blocked).`,
  },
  {
    id: 'j3',
    category: 'join',
    question: 'Which people working on atlas prefer morning meetings?',
    expect: ['maya'],
    goldSql: `SELECT w.person FROM works_on w JOIN prefers_meeting p ON p.person = w.person WHERE w.project = 'atlas' AND p.window = 'morning'`,
    goldDatalog: `atlas_morning(P) :- works_on(P, atlas), prefers_meeting(P, morning).`,
  },
  {
    id: 'j4',
    category: 'join',
    question: 'Which blocked project has a review slot, and on which day?',
    expect: ['atlas', 'tuesday'],
    forbid: ['thursday'],
    goldSql: `SELECT r.project, r.day FROM review_slot r JOIN status s ON s.project = r.project WHERE s.state = 'blocked'`,
    goldDatalog: `slot(P, D) :- review_slot(P, D, _), status(P, blocked).`,
  },
  {
    id: 'j5',
    category: 'join',
    question: 'Who works on a project blocked by legal_signoff?',
    expect: ['priya', 'tom', 'maya'],
    goldSql: `SELECT w.person FROM works_on w JOIN blocker b ON b.project = w.project WHERE b.blocker = 'legal_signoff'`,
    goldDatalog: `affected(P) :- works_on(P, Proj), blocker(Proj, legal_signoff).`,
  },
  {
    id: 'j6',
    category: 'join',
    question: 'Which project has a review slot in the afternoon?',
    expect: ['orchard'],
    goldSql: `SELECT project FROM review_slot WHERE window = 'afternoon'`,
    goldDatalog: `afternoon_slot(P) :- review_slot(P, _, afternoon).`,
  },
  // ---- multi-hop: requires recursion or chained reasoning ----------------
  {
    id: 'm1',
    category: 'multihop',
    question:
      'List every person above maya in the reporting chain (her manager, that manager’s manager, and so on).',
    expect: ['liam', 'ava', 'dana'],
    goldSql: `WITH RECURSIVE chain(m) AS (
      SELECT manager FROM reports_to WHERE person = 'maya'
      UNION SELECT r.manager FROM reports_to r JOIN chain c ON r.person = c.m
    ) SELECT m FROM chain`,
    goldDatalog: `above(M) :- reports_to(maya, M).
above(M) :- above(X), reports_to(X, M).`,
  },
  {
    id: 'm2',
    category: 'multihop',
    question:
      'Follow the waits_on chain from atlas to the end: what is the final upstream dependency that waits on nothing else?',
    expect: ['procurement freeze'],
    goldSql: `WITH RECURSIVE chain(u) AS (
      SELECT upstream FROM waits_on WHERE item = 'atlas'
      UNION SELECT w.upstream FROM waits_on w JOIN chain c ON w.item = c.u
    ) SELECT u FROM chain WHERE u NOT IN (SELECT item FROM waits_on)`,
    goldDatalog: `root(R) :- reach(R), \\+ waits_on(R, _).
reach(X) :- waits_on(atlas, X).
reach(X) :- reach(M), waits_on(M, X).`,
  },
  {
    id: 'm3',
    category: 'multihop',
    question:
      'Which projects directly or transitively wait on procurement_freeze? Consider the whole waits_on chain.',
    expect: ['atlas', 'beacon'],
    goldSql: `WITH RECURSIVE below(i) AS (
      SELECT item FROM waits_on WHERE upstream = 'procurement_freeze'
      UNION SELECT w.item FROM waits_on w JOIN below b ON w.upstream = b.i
    ) SELECT i FROM below`,
    goldDatalog: `waiting(I) :- waits_on(I, procurement_freeze).
waiting(I) :- waits_on(I, M), waiting(M).`,
  },
  {
    id: 'm4',
    category: 'multihop',
    question:
      'Does tom ultimately report up to dana through the management chain? Say yes or no and name the chain.',
    expect: ['yes', 'liam', 'ava'],
    goldSql: `WITH RECURSIVE chain(m) AS (
      SELECT manager FROM reports_to WHERE person = 'tom'
      UNION SELECT r.manager FROM reports_to r JOIN chain c ON r.person = c.m
    ) SELECT m FROM chain`,
    goldDatalog: `tom_above(M) :- reports_to(tom, M).
tom_above(M) :- tom_above(X), reports_to(X, M).`,
  },
  {
    id: 'm5',
    category: 'multihop',
    question:
      'Through the dependency chain, is atlas ultimately waiting on procurement_freeze? Answer yes or no.',
    expect: ['yes'],
    goldSql: `WITH RECURSIVE chain(u) AS (
      SELECT upstream FROM waits_on WHERE item = 'atlas'
      UNION SELECT w.upstream FROM waits_on w JOIN chain c ON w.item = c.u
    ) SELECT u FROM chain WHERE u = 'procurement_freeze'`,
    goldDatalog: `reach(X) :- waits_on(atlas, X).
reach(X) :- reach(M), waits_on(M, X).`,
  },
  {
    id: 'm6',
    category: 'multihop',
    question: 'List everyone who directly or transitively reports to dana.',
    expect: ['ava', 'maya', 'tom'],
    goldSql: `WITH RECURSIVE tree(p) AS (
      SELECT person FROM reports_to WHERE manager = 'dana'
      UNION SELECT r.person FROM reports_to r JOIN tree t ON r.manager = t.p
    ) SELECT p FROM tree`,
    goldDatalog: `under(P) :- reports_to(P, dana).
under(P) :- reports_to(P, M), under(M).`,
  },
  // ---- absence: NULL/empty vs derived negation ---------------------------
  {
    id: 'a1',
    category: 'absence',
    question:
      'Which people working on atlas have no stored meeting preference?',
    expect: ['liam'],
    forbid: ['maya'],
    goldSql: `SELECT w.person FROM works_on w LEFT JOIN prefers_meeting p ON p.person = w.person WHERE w.project = 'atlas' AND p.person IS NULL`,
    goldDatalog: `no_pref(P) :- works_on(P, atlas), \\+ prefers_meeting(P, _).`,
  },
  {
    id: 'a2',
    category: 'absence',
    question: 'Which blocked projects have no review slot?',
    expect: ['beacon'],
    forbid: ['atlas'],
    goldSql: `SELECT s.project FROM status s LEFT JOIN review_slot r ON r.project = s.project WHERE s.state = 'blocked' AND r.project IS NULL`,
    goldDatalog: `no_slot(P) :- status(P, blocked), \\+ review_slot(P, _, _).`,
  },
  {
    id: 'a3',
    category: 'absence',
    question: 'Does priya have a stored meeting preference? Answer yes or no.',
    expect: ['no'],
    goldSql: `SELECT COUNT(*) AS n FROM prefers_meeting WHERE person = 'priya'`,
    goldDatalog: `priya_missing(P) :- works_on(priya, P), \\+ prefers_meeting(priya, _).`,
  },
  {
    id: 'a4',
    category: 'absence',
    question: 'Which projects have a status but no recorded blocker?',
    expect: ['orchard', 'quasar'],
    goldSql: `SELECT s.project FROM status s LEFT JOIN blocker b ON b.project = s.project WHERE b.project IS NULL`,
    goldDatalog: `no_blocker(P) :- status(P, _), \\+ blocker(P, _).`,
  },
  {
    id: 'a5',
    category: 'absence',
    question:
      'Who was promised an update but has no stored meeting preference?',
    expect: ['nora', 'priya'],
    forbid: ['maya'],
    goldSql: `SELECT u.person FROM promised_update u LEFT JOIN prefers_meeting p ON p.person = u.person WHERE p.person IS NULL`,
    goldDatalog: `owed_no_pref(P) :- promised_update(_, P, _), \\+ prefers_meeting(P, _).`,
  },
  {
    id: 'a6',
    category: 'absence',
    question:
      'Who manages at least one person but works on no project themselves?',
    expect: ['dana'],
    goldSql: `SELECT DISTINCT r.manager FROM reports_to r LEFT JOIN works_on w ON w.person = r.manager WHERE w.person IS NULL`,
    goldDatalog: `pure_manager(M) :- reports_to(_, M), \\+ works_on(M, _).`,
  },
  // ---- write traps: a corrupting write precedes the question -------------
  {
    id: 't1',
    category: 'write-trap',
    question: 'What is the current status of atlas?',
    expect: ['blocked'],
    forbid: ['active'],
    trapWriteSql: `UPDATE status SET state = 'active' WHERE project = 'atlas'`,
    goldSql: `SELECT state FROM status WHERE project = 'atlas'`,
    goldDatalog: `atlas_state(S) :- status(atlas, S).`,
  },
  {
    id: 't2',
    category: 'write-trap',
    question: 'Which projects are currently blocked?',
    expect: ['atlas', 'beacon'],
    trapWriteSql: `UPDATE status SET state = 'active' WHERE project = 'beacon'`,
    goldSql: `SELECT project FROM status WHERE state = 'blocked'`,
    goldDatalog: `blocked_project(P) :- status(P, blocked).`,
  },
  {
    id: 't3',
    category: 'write-trap',
    question: 'What single state is the atlas project in?',
    expect: ['blocked'],
    forbid: ['active'],
    trapWriteSql: `INSERT INTO status VALUES ('atlas', 'active')`,
    goldSql: `SELECT state FROM status WHERE project = 'atlas'`,
    goldDatalog: `atlas_state(S) :- status(atlas, S).`,
  },
  {
    id: 't4',
    category: 'write-trap',
    question: 'What blocker currently applies to atlas?',
    expect: ['vendor security review'],
    trapWriteSql: `DELETE FROM blocker WHERE project = 'atlas'`,
    goldSql: `SELECT blocker FROM blocker WHERE project = 'atlas'`,
    goldDatalog: `atlas_blocker(B) :- blocker(atlas, B).`,
  },
  // ---- v2 traps: authored after WRITE_GATE_RULES were frozen (ADR 0003) ---
  {
    id: 't5',
    category: 'write-trap',
    question: 'Which projects have a status but no recorded blocker?',
    expect: ['orchard', 'quasar'],
    forbid: ['permits'],
    trapWriteSql: `INSERT INTO blocker VALUES ('orchard', 'permits')`,
    goldSql: `SELECT s.project FROM status s LEFT JOIN blocker b ON b.project = s.project WHERE b.project IS NULL`,
    goldDatalog: `no_blocker(P) :- status(P, _), \\+ blocker(P, _).`,
  },
  {
    id: 't6',
    category: 'write-trap',
    question: 'What single state is the beacon project in?',
    expect: ['blocked'],
    forbid: ['active'],
    trapWriteSql: `INSERT INTO status VALUES ('beacon', 'active')`,
    goldSql: `SELECT state FROM status WHERE project = 'beacon'`,
    goldDatalog: `beacon_state(S) :- status(beacon, S).`,
  },
  {
    id: 'c1',
    category: 'write-trap',
    control: true,
    question: 'What meeting window does maya prefer?',
    expect: ['afternoon'],
    forbid: ['morning'],
    trapWriteSql: `UPDATE prefers_meeting SET window = 'afternoon' WHERE person = 'maya'`,
    goldSql: `SELECT window FROM prefers_meeting WHERE person = 'maya'`,
    goldDatalog: `maya_window(W) :- prefers_meeting(maya, W).`,
  },
] as const;

export function normalizeAnswer(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export interface GradeResult {
  passed: boolean;
  missing: string[];
  forbidden: string[];
  /** v2 only: seed-DB entities mentioned that are outside the gold answer set. */
  extraEntities: string[];
}

export function gradeAnswer(
  question: AgentBoundaryQuestion,
  answer: string,
): GradeResult {
  const normalized = ` ${normalizeAnswer(answer)} `;
  const missing = question.expect.filter(
    (term) => !normalized.includes(` ${normalizeAnswer(term)} `),
  );
  const forbidden = (question.forbid ?? []).filter((term) =>
    normalized.includes(` ${normalizeAnswer(term)} `),
  );
  return {
    passed: missing.length === 0 && forbidden.length === 0,
    missing,
    forbidden,
    extraEntities: [],
  };
}

/**
 * Entity lexicon derived from the seed SQL's cell values (ADR 0003): every
 * quoted literal in AGENT_BOUNDARY_SEED_SQL, normalized. Deterministic and
 * hand-authoring free.
 */
export function seedEntityLexicon(): Set<string> {
  const lexicon = new Set<string>();
  for (const match of AGENT_BOUNDARY_SEED_SQL.matchAll(/'([^']+)'/g)) {
    lexicon.add(normalizeAnswer(match[1]));
  }
  return lexicon;
}

/** Normalized entity set of gold-query result rows. */
export function entitiesFromRows(
  rows: Array<Record<string, unknown>>,
): Set<string> {
  const entities = new Set<string>();
  for (const row of rows) {
    for (const value of Object.values(row)) {
      entities.add(normalizeAnswer(String(value)));
    }
  }
  return entities;
}

/**
 * v2 answer-set grading (ADR 0003): all expected terms present, no forbidden
 * terms, and no lexicon entity outside the gold set — this fails
 * wrong-superset answers such as v1 j5 naming four people where three were
 * asked. Entities that only echo the question text (e.g. 'blocked' in
 * "Which projects are currently blocked?") are excluded: restating the
 * question is not an extra entity.
 */
export function gradeAnswerV2(
  question: AgentBoundaryQuestion,
  answer: string,
  goldEntities: ReadonlySet<string>,
  lexicon: ReadonlySet<string> = seedEntityLexicon(),
): GradeResult {
  const base = gradeAnswer(question, answer);
  const normalized = ` ${normalizeAnswer(answer)} `;
  const questionText = ` ${normalizeAnswer(question.question)} `;
  const extraEntities = [...lexicon].filter(
    (entity) =>
      entity.length > 0 &&
      !goldEntities.has(entity) &&
      !questionText.includes(` ${entity} `) &&
      normalized.includes(` ${entity} `),
  );
  return {
    passed: base.passed && extraEntities.length === 0,
    missing: base.missing,
    forbidden: base.forbidden,
    extraEntities,
  };
}

const SQL_SCHEMA_PROMPT = `You query a SQLite database with these tables:
works_on(person, project)
reports_to(person, manager)
status(project, state)            -- state is 'blocked' or 'active'
blocker(project, blocker)
waits_on(item, upstream)          -- dependency edges, may chain several hops
prefers_meeting(person, window)   -- window is 'morning' or 'afternoon'; some people have no row
review_slot(project, day, window)
promised_update(owner, person, project)`;

const DATALOG_SCHEMA_PROMPT = `You query a Remembero knowledge base with these predicates:
works_on(Person, Project)
reports_to(Person, Manager)
status(Project, State)            -- State is blocked or active
blocker(Project, Blocker)
waits_on(Item, Upstream)          -- dependency edges, may chain several hops
prefers_meeting(Person, Window)   -- Window is morning or afternoon; some people have no fact
review_slot(Project, Day, Window)
promised_update(Owner, Person, Project)`;

export function sqlSystemPrompt(): string {
  return `${SQL_SCHEMA_PROMPT}

Write ONE read-only SQLite query that answers the user's question. WITH RECURSIVE is allowed.
Reply with ONLY the SQL, no prose, no markdown fences.

Examples:
Q: Which projects are active?
SELECT project FROM status WHERE state = 'active'
Q: Every manager above nora in the chain?
WITH RECURSIVE chain(m) AS (SELECT manager FROM reports_to WHERE person = 'nora' UNION SELECT r.manager FROM reports_to r JOIN chain c ON r.person = c.m) SELECT m FROM chain
Q: Which projects have no review slot?
SELECT s.project FROM status s LEFT JOIN review_slot r ON r.project = s.project WHERE r.project IS NULL`;
}

const DATALOG_CHEATSHEET = `Dialect cheatsheet (this bridge exactly; anything else errors):
- A program is one or more rules; every rule ends with a period.
- Rule shape: head(A, B) :- predicate(A), other_predicate(B, A).
- The FIRST rule's head predicate is the query; its rows are the answer.
- Variables start uppercase (Person, X). Constants are lowercase (atlas, blocked).
- _ is a wildcard for a value you do not care about.
- \\+ predicate(X) means "there is no such fact" (negation).
- Comparisons come after a predicate: A != B, X > 3.
- Recursion is allowed: a rule body may reuse its own head predicate.
- NOT supported: cuts (!), lists ([...]), comments, prose, or a Q: line.`;

/**
 * Few-shot examples for the Datalog condition (ADR 0002 prior leveling).
 * Exported so the test suite executes every program against the seeded
 * database — an example that does not run is a benchmark bug.
 */
export const DATALOG_FEW_SHOT: ReadonlyArray<{ q: string; program: string }> = [
  {
    q: 'Which projects are active?',
    program: `active_project(P) :- status(P, active).`,
  },
  {
    q: "Who is maya's direct manager?",
    program: `maya_manager(M) :- reports_to(maya, M).`,
  },
  {
    q: 'Which blocked projects does maya work on?',
    program: `maya_blocked(P) :- works_on(maya, P), status(P, blocked).`,
  },
  {
    q: 'Which projects have no review slot?',
    program: `no_slot(P) :- status(P, _), \\+ review_slot(P, _, _).`,
  },
  {
    q: 'Who was promised an update but has no stored meeting preference?',
    program: `owed_no_pref(P) :- promised_update(_, P, _), \\+ prefers_meeting(P, _).`,
  },
  {
    q: 'Every manager above nora in the chain?',
    program: `above(M) :- reports_to(nora, M).
above(M) :- above(X), reports_to(X, M).`,
  },
  {
    q: 'What is the final upstream dependency atlas waits on?',
    program: `reach(X) :- waits_on(atlas, X).
reach(X) :- reach(M), waits_on(M, X).
root(R) :- reach(R), \\+ waits_on(R, _).`,
  },
  {
    q: 'Which pairs of different people share a project?',
    program: `pair(A, B) :- works_on(A, P), works_on(B, P), A != B.`,
  },
];

export function datalogSystemPrompt(): string {
  const examples = DATALOG_FEW_SHOT.map(
    (example) => `Q: ${example.q}\n${example.program}`,
  ).join('\n');
  return `${DATALOG_SCHEMA_PROMPT}

Write ONE Datalog program that answers the user's question.
Reply with ONLY the Datalog, no prose, no markdown fences.

${DATALOG_CHEATSHEET}

Examples:
${examples}`;
}

export function answerSystemPrompt(): string {
  return 'Answer the question using ONLY the tool result rows. Name the concrete values. If the rows are empty, say what is absent. One short sentence.';
}

export function stripFences(text: string): string {
  return text
    .replace(/^```[a-z]*\n?/gim, '')
    .replace(/```\s*$/gm, '')
    .trim();
}

const WRITE_KEYWORDS =
  /\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b/i;

export function assertReadOnlySql(sql: string): void {
  if (WRITE_KEYWORDS.test(sql)) {
    throw new Error('only read-only SELECT/WITH queries are allowed');
  }
  const head = sql.trim().slice(0, 6).toLowerCase();
  if (!head.startsWith('select') && !head.startsWith('with')) {
    throw new Error('query must start with SELECT or WITH');
  }
}
