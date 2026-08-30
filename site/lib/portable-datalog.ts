import {
  evaluateQuerySpecWithProof,
  isAggregateRule,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  parseProgram,
  parseQuerySpec,
  type Clause,
  type Goal,
  type Literal,
  type QuerySpec,
  type Term,
} from "./engine";

/**
 * The SQLite C extension compiles ordinary single-predicate rule programs.
 * Everything it cannot parse — stratified negation, comparisons, aggregation,
 * multi-predicate programs, bare query specs — runs on the portable
 * TypeScript engine over facts read out of the same SQLite tables. This
 * mirrors the decision the Node bridge makes in src/sqlite/extension.ts, so
 * browser and server semantics stay identical.
 */
export type PortableExecutionMode = "native" | "portable";

export interface PortableExec {
  (sql: string): Promise<Array<Record<string, unknown>>>;
}

export interface PortableExplanation {
  row: Record<string, string | number>;
  proof: unknown;
  proofs?: unknown[];
}

const ENGINE_LIMITS = {
  maxFacts: 60_000,
  maxIterations: 1_000,
  maxRows: 4_097,
  maxProofDepth: 128,
  maxProofNodes: 100_000,
  maxProofsPerRow: 4,
  maxAggregateRows: 50_000,
  maxAggregateProofRows: 256,
} as const;

export function portableExecutionMode(input: string): PortableExecutionMode {
  if (!input.includes(":-")) return "portable";
  let program: Clause[];
  try {
    program = parseProgram(input);
  } catch {
    // Let the native parser retain its established error contract.
    return "native";
  }
  if (program.some(isIntegrityConstraint)) return "portable";
  if (program.some(isAggregateRule)) return "portable";
  const heads = new Set(
    program.map((clause) => `${clause.head.predicate}/${clause.head.args.length}`),
  );
  if (heads.size > 1) return "portable";
  const goals = program.flatMap((clause) => clause.body);
  if (goals.some((goal) => isNegation(goal) || isComparison(goal))) {
    return "portable";
  }
  return "native";
}

function literalFromGoal(goal: Goal): Literal | undefined {
  if (isComparison(goal)) return undefined;
  return isNegation(goal) ? goal.not : goal;
}

interface PortableRequest {
  program: Clause[];
  query: QuerySpec;
  basePredicates: Array<{ predicate: string; arity: number }>;
}

function preparePortableRequest(input: string): PortableRequest {
  let program: Clause[];
  let query: QuerySpec;
  if (!input.includes(":-")) {
    program = [];
    query = parseQuerySpec(input);
  } else {
    program = parseProgram(input);
    if (program.some(isIntegrityConstraint)) {
      throw new Error(
        "integrity constraints are policies for the personal knowledge store, not SQLite queries",
      );
    }
    const target = program[0]?.head;
    if (target === undefined) throw new Error("expected a Datalog rule or query");
    const names = new Set<string>();
    for (const term of target.args) {
      if (term.type !== "var" || term.name === "_" || names.has(term.name)) {
        throw new Error("SQLite query rule head terms must be distinct named variables");
      }
      names.add(term.name);
    }
    if (target.args.length === 0) {
      throw new Error("SQLite query rule head must contain at least one named variable");
    }
    query = { kind: "relational", goals: [target] };
  }

  const derived = new Set(program.map((clause) => clause.head.predicate));
  const baseByName = new Map<string, number>();
  const addBase = (literal: Literal) => {
    if (derived.has(literal.predicate)) return;
    const existing = baseByName.get(literal.predicate);
    if (existing !== undefined && existing !== literal.args.length) {
      throw new Error(`predicate '${literal.predicate}' has inconsistent arity`);
    }
    baseByName.set(literal.predicate, literal.args.length);
  };
  for (const clause of program) {
    for (const goal of clause.body) {
      const literal = literalFromGoal(goal);
      if (literal !== undefined) addBase(literal);
    }
  }
  for (const goal of query.goals) {
    const literal = literalFromGoal(goal);
    if (literal !== undefined) addBase(literal);
  }
  return {
    program,
    query,
    basePredicates: [...baseByName]
      .map(([predicate, arity]) => ({ predicate, arity }))
      .sort(
        (left, right) =>
          left.predicate.localeCompare(right.predicate) || left.arity - right.arity,
      ),
  };
}

function quoteIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

function portableTerm(value: unknown, predicate: string, column: number): Term {
  if (typeof value === "string") return { type: "atom", value };
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(`predicate '${predicate}' contains a non-finite number`);
    }
    return { type: "num", value };
  }
  throw new Error(
    `predicate '${predicate}' column ${column + 1} contains a value the portable SQLite bridge cannot represent`,
  );
}

async function portableClauses(
  exec: PortableExec,
  request: PortableRequest,
): Promise<Clause[]> {
  const facts: Clause[] = [];
  for (const { predicate, arity } of request.basePredicates) {
    const info = await exec(`PRAGMA table_info(${quoteIdentifier(predicate)})`);
    const columns = info
      .slice()
      .sort((left, right) => Number(left.cid) - Number(right.cid))
      .map((column) => String(column.name));
    if (columns.length === 0) {
      throw new Error(`predicate '${predicate}' is unavailable`);
    }
    if (columns.length !== arity) {
      throw new Error(
        `predicate '${predicate}' expects ${columns.length} columns but the query supplies ${arity}`,
      );
    }
    const selected = columns
      .map((column, index) => `${quoteIdentifier(column)} AS ${quoteIdentifier(`c${index}`)}`)
      .join(", ");
    const rows = await exec(`SELECT ${selected} FROM ${quoteIdentifier(predicate)}`);
    for (const row of rows) {
      facts.push({
        head: {
          predicate,
          args: columns.map((_, index) => portableTerm(row[`c${index}`], predicate, index)),
        },
        body: [],
      });
    }
  }
  return [...facts, ...request.program];
}

function rowFromBindings(
  bindings: Record<string, Term>,
): Record<string, string | number> {
  return Object.fromEntries(
    Object.entries(bindings).map(([name, term]) => {
      if (term.type !== "atom" && term.type !== "num") {
        throw new Error("Datalog result contains an unbound term");
      }
      return [name, term.value];
    }),
  );
}

export async function runPortableQuery(
  exec: PortableExec,
  input: string,
): Promise<Array<Record<string, string | number>>> {
  const request = preparePortableRequest(input);
  const clauses = await portableClauses(exec, request);
  const results = evaluateQuerySpecWithProof(clauses, request.query, ENGINE_LIMITS);
  return results.map(({ bindings }) => rowFromBindings(bindings));
}

export async function runPortableExplain(
  exec: PortableExec,
  input: string,
): Promise<PortableExplanation[]> {
  const request = preparePortableRequest(input);
  const clauses = await portableClauses(exec, request);
  const results = evaluateQuerySpecWithProof(clauses, request.query, ENGINE_LIMITS);
  return results.map(({ bindings, proofs }) => {
    const proof = proofs[0];
    if (proof === undefined) throw new Error("Datalog explanation has no proof");
    return {
      row: rowFromBindings(bindings),
      proof,
      ...(proofs.length > 1 ? { proofs } : {}),
    };
  });
}
