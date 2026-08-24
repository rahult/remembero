import type { SqliteScalar } from "./sqlite-wasm";

export interface DemoPreset {
  id:
    | "follow_up"
    | "collaborators"
    | "recursive_paths"
    | "support_escalation"
    | "release_readiness"
    | "access_control";
  category: string;
  useCase: string;
  focusTable: string;
  title: string;
  description: string;
  program: string;
  setupSql?: string;
}

export interface LineageEvent {
  id: string;
  kind: "INSERT" | "SQL" | "DATALOG" | "RESET";
  target: string;
  detail: string;
  timestamp: string;
}

export const SAMPLE_SETUP_SQL = `
  DROP TABLE IF EXISTS project_owner;
  DROP TABLE IF EXISTS project_contributor;
  DROP TABLE IF EXISTS promised_update;
  DROP TABLE IF EXISTS status;
  DROP TABLE IF EXISTS edge;
  DROP TABLE IF EXISTS support_ticket;
  DROP TABLE IF EXISTS support_plan;
  DROP TABLE IF EXISTS release_candidate;
  DROP TABLE IF EXISTS release_check;
  DROP TABLE IF EXISTS release_approval;
  DROP TABLE IF EXISTS team_member;
  DROP TABLE IF EXISTS team_grant;
  DROP TABLE IF EXISTS workspace_document;

  CREATE TABLE project_owner(project TEXT NOT NULL, person TEXT NOT NULL);
  CREATE TABLE project_contributor(project TEXT NOT NULL, person TEXT NOT NULL);
  CREATE TABLE promised_update(owner TEXT NOT NULL, person TEXT NOT NULL, project TEXT NOT NULL);
  CREATE TABLE status(project TEXT NOT NULL, state TEXT NOT NULL);

  INSERT INTO project_owner VALUES
    ('atlas', 'rahul'),
    ('orchard', 'ava');
  INSERT INTO project_contributor VALUES
    ('atlas', 'maya'),
    ('atlas', 'liam'),
    ('orchard', 'nora');
  INSERT INTO promised_update VALUES
    ('rahul', 'maya', 'atlas'),
    ('rahul', 'liam', 'atlas');
  INSERT INTO status VALUES
    ('atlas', 'blocked'),
    ('orchard', 'active');
`;

export const PRESETS: readonly DemoPreset[] = [
  {
    id: "follow_up",
    category: "Project operations",
    useCase: "Turn blocked promises into a concrete follow-up list.",
    focusTable: "promised_update",
    title: "Who needs a follow-up?",
    description:
      "Find people who were promised an update for a project that is currently blocked.",
    program: `needs_follow_up(Person, Project) :-
  promised_update(rahul, Person, Project),
  status(Project, blocked).`,
  },
  {
    id: "collaborators",
    category: "Team context",
    useCase: "Find contributors without confusing ownership with collaboration.",
    focusTable: "project_contributor",
    title: "Who collaborates on each project?",
    description:
      "Join ownership and contribution tables while excluding the owner from the result.",
    program: `collaborator(Person, Project) :-
  project_owner(Project, Owner),
  project_contributor(Project, Person),
  Owner != Person.`,
  },
  {
    id: "recursive_paths",
    category: "Knowledge graph",
    useCase: "Trace a dependency chain until the deterministic fixpoint is reached.",
    focusTable: "edge",
    title: "Which nodes are reachable?",
    description:
      "Use two rules to compute a transitive path until SQLite reaches a deterministic fixpoint.",
    setupSql: `
      CREATE TABLE IF NOT EXISTS edge(source TEXT NOT NULL, target TEXT NOT NULL);
      DELETE FROM edge;
      INSERT INTO edge VALUES
        ('atlas', 'memory'),
        ('memory', 'proof'),
        ('proof', 'source');
    `,
    program: `path(X, Y) :- edge(X, Y).
path(X, Y) :- edge(X, Z), path(Z, Y).`,
  },
  {
    id: "support_escalation",
    category: "Customer support",
    useCase: "Route urgent tickets only when the customer has premium coverage.",
    focusTable: "support_ticket",
    title: "Which tickets need escalation?",
    description:
      "Find open urgent tickets for premium customers so the support team can escalate the right work.",
    setupSql: `
      DROP TABLE IF EXISTS support_ticket;
      DROP TABLE IF EXISTS support_plan;
      CREATE TABLE support_ticket(ticket TEXT NOT NULL, customer TEXT NOT NULL, priority TEXT NOT NULL, state TEXT NOT NULL);
      CREATE TABLE support_plan(customer TEXT NOT NULL, tier TEXT NOT NULL);
      INSERT INTO support_ticket VALUES
        ('ticket_102', 'maya', 'urgent', 'open'),
        ('ticket_103', 'liam', 'urgent', 'open'),
        ('ticket_104', 'maya', 'normal', 'open'),
        ('ticket_105', 'nora', 'urgent', 'closed');
      INSERT INTO support_plan VALUES
        ('maya', 'premium'),
        ('liam', 'standard'),
        ('nora', 'premium');
    `,
    program: `escalation(Customer, Ticket) :-
  support_ticket(Ticket, Customer, urgent, open),
  support_plan(Customer, premium).`,
  },
  {
    id: "release_readiness",
    category: "Release operations",
    useCase: "Require every named check and approval before a service can ship.",
    focusTable: "release_candidate",
    title: "Which service is ready to ship?",
    description:
      "Join release status, passing checks, and an explicit approval before treating a service as shippable.",
    setupSql: `
      DROP TABLE IF EXISTS release_candidate;
      DROP TABLE IF EXISTS release_check;
      DROP TABLE IF EXISTS release_approval;
      CREATE TABLE release_candidate(service TEXT NOT NULL, stage TEXT NOT NULL);
      CREATE TABLE release_check(service TEXT NOT NULL, check_name TEXT NOT NULL, state TEXT NOT NULL);
      CREATE TABLE release_approval(service TEXT NOT NULL, approver TEXT NOT NULL, state TEXT NOT NULL);
      INSERT INTO release_candidate VALUES
        ('checkout', 'production'),
        ('search', 'production'),
        ('billing', 'staging');
      INSERT INTO release_check VALUES
        ('checkout', 'tests', 'passing'),
        ('checkout', 'security', 'passing'),
        ('search', 'tests', 'passing'),
        ('search', 'security', 'failing'),
        ('billing', 'tests', 'passing'),
        ('billing', 'security', 'passing');
      INSERT INTO release_approval VALUES
        ('checkout', 'release_manager', 'approved'),
        ('search', 'release_manager', 'approved'),
        ('billing', 'release_manager', 'approved');
    `,
    program: `ship_ready(Service) :-
  release_candidate(Service, production),
  release_check(Service, tests, passing),
  release_check(Service, security, passing),
  release_approval(Service, release_manager, approved).`,
  },
  {
    id: "access_control",
    category: "Access control",
    useCase: "Show only published documents granted through a team membership.",
    focusTable: "team_member",
    title: "Who can read the launch plan?",
    description:
      "Combine team membership, document grants, and publication state to answer an access question exactly.",
    setupSql: `
      DROP TABLE IF EXISTS team_member;
      DROP TABLE IF EXISTS team_grant;
      DROP TABLE IF EXISTS workspace_document;
      CREATE TABLE team_member(person TEXT NOT NULL, team TEXT NOT NULL);
      CREATE TABLE team_grant(team TEXT NOT NULL, document TEXT NOT NULL);
      CREATE TABLE workspace_document(document TEXT NOT NULL, state TEXT NOT NULL);
      INSERT INTO team_member VALUES
        ('maya', 'launch'),
        ('liam', 'launch'),
        ('nora', 'support');
      INSERT INTO team_grant VALUES
        ('launch', 'launch_plan'),
        ('launch', 'runbook'),
        ('support', 'runbook');
      INSERT INTO workspace_document VALUES
        ('launch_plan', 'published'),
        ('runbook', 'published'),
        ('roadmap', 'draft');
    `,
    program: `can_read(Person, Document) :-
  team_member(Person, Team),
  team_grant(Team, Document),
  workspace_document(Document, published).`,
  },
] as const;

export const DEFAULT_SQL = "SELECT project, state FROM status ORDER BY project;";

export const INITIAL_LINEAGE: readonly LineageEvent[] = [
  {
    id: "seed-owner",
    kind: "INSERT",
    target: "project_owner",
    detail: "2 sample rows",
    timestamp: "sample seed",
  },
  {
    id: "seed-contributors",
    kind: "INSERT",
    target: "project_contributor",
    detail: "3 sample rows",
    timestamp: "sample seed",
  },
  {
    id: "seed-promises",
    kind: "INSERT",
    target: "promised_update",
    detail: "2 sample rows",
    timestamp: "sample seed",
  },
  {
    id: "seed-status",
    kind: "INSERT",
    target: "status",
    detail: "2 sample rows",
    timestamp: "sample seed",
  },
] as const;

export const INSERT_DEFAULTS: Record<string, Record<string, SqliteScalar>> = {
  project_owner: { project: "kiln", person: "maya" },
  project_contributor: { project: "kiln", person: "nora" },
  promised_update: { owner: "rahul", person: "nora", project: "kiln" },
  status: { project: "kiln", state: "blocked" },
  edge: { source: "source", target: "answer" },
  support_ticket: {
    ticket: "ticket_106",
    customer: "maya",
    priority: "urgent",
    state: "open",
  },
  support_plan: { customer: "maya", tier: "premium" },
  release_candidate: { service: "checkout", stage: "production" },
  release_check: { service: "checkout", check_name: "tests", state: "passing" },
  release_approval: {
    service: "checkout",
    approver: "release_manager",
    state: "approved",
  },
  team_member: { person: "maya", team: "launch" },
  team_grant: { team: "launch", document: "launch_plan" },
  workspace_document: { document: "launch_plan", state: "published" },
};

export const CONSTRAINT_EXAMPLE =
  ":- status(Project, active), status(Project, blocked).";

export function quoteIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

export function formatCell(value: SqliteScalar): string {
  if (value === null) return "NULL";
  return String(value);
}

export function eventTime(): string {
  return new Intl.DateTimeFormat("en-AU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
}
