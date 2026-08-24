import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { DatabaseSync } from 'node:sqlite';
import {
  type Bindings,
  type Clause,
  type Goal,
  type QueryProof,
  type QuerySpec,
  type ScalarExpression,
  type Term,
  evaluateQuerySpec,
  evaluateQuerySpecWithProof,
  isArithmeticExpression,
  isComparison,
  isIntegrityConstraint,
  isNegation,
  parseProgram,
  parseQuerySpec,
  predKey,
  serializeClause,
} from '../engine/index.js';
import {
  createSqliteMemoryStore,
  type SqliteMemoryStore,
  type SqliteMemoryStoreOptions,
} from '../store/sqlite-store.js';

interface DatalogDatabaseMethods {
  datalogSql(rule: string): string;
  datalogQuery(rule: string): DatalogRow[];
  datalogExplain(program: string): DatalogExplanation[];
  datalogPlan(input: string): SqliteDatalogPlan;
}

export interface RememberoDatabaseMethods extends DatalogDatabaseMethods {
  /** Durable governed memory stored inside this SQLite database. */
  readonly memory: SqliteMemoryStore;
}

export type RememberoDatabase = DatabaseSync & RememberoDatabaseMethods;

export interface OpenDatalogDatabaseOptions {
  extensionPath?: string;
  enableForeignKeyConstraints?: boolean;
  enableDoubleQuotedStringLiterals?: boolean;
  readOnly?: boolean;
}

export interface OpenRememberoDatabaseOptions extends OpenDatalogDatabaseOptions {
  memory?: SqliteMemoryStoreOptions;
}

export type DatalogRow = Record<string, unknown>;

export type DatalogProof = QueryProof;

export interface DatalogExplanation {
  row: DatalogRow;
  proof: DatalogProof;
  /** Present for standalone conjunctions with more than one proof-bearing goal. */
  proofs?: DatalogProof[];
}

export type SqliteDatalogExecutionMode = 'native' | 'portable';

export interface SqliteDatalogPlanColumn {
  name: string;
  declaredType: string | null;
  hidden: number;
}

export interface SqliteDatalogPlanRelation {
  predicate: string;
  arity: number;
  objectType: 'table' | 'view';
  temporary: boolean;
  columns: SqliteDatalogPlanColumn[];
}

export interface SqliteDatalogPlanDerivedPredicate {
  predicate: string;
  arity: number;
  recursive: boolean;
}

export interface SqliteDatalogPlan {
  mode: SqliteDatalogExecutionMode;
  executionBoundary: 'sqlite_scalar' | 'portable_snapshot';
  inputKind: 'rule_program' | 'relational_query' | 'aggregate_query';
  result: {
    predicate?: string;
    variables: string[];
  };
  derivedPredicates: SqliteDatalogPlanDerivedPredicate[];
  baseRelations: SqliteDatalogPlanRelation[];
  nativeSql?: string;
  scansData: false;
  bounds: {
    ruleBytes: number;
    baseRows: number;
    derivedFacts: number;
    queryRows: number;
    resultBytes: number;
  };
}

const MAX_RULE_BYTES = 64 * 1024;
const MAX_BASE_ROWS = 100_000;
const MAX_DERIVED_FACTS = 10_000;
const MAX_QUERY_ROWS = 10_000;
const MAX_RESULT_BYTES = 16 * 1024 * 1024;
const PORTABLE_SAVEPOINT = 'rembero_portable_bridge';
const OPEN_DATABASE_OPTION_KEYS = new Set([
  'extensionPath',
  'enableForeignKeyConstraints',
  'enableDoubleQuotedStringLiterals',
  'readOnly',
]);

function containsArithmeticSyntax(program: string): boolean {
  const previousSignificant = (index: number): string => {
    for (let cursor = index - 1; cursor >= 0; cursor--) {
      if (!/\s/.test(program[cursor])) return program[cursor];
    }
    return '';
  };
  const nextSignificant = (index: number): string => {
    for (let cursor = index + 1; cursor < program.length; cursor++) {
      if (!/\s/.test(program[cursor])) return program[cursor];
    }
    return '';
  };

  for (let index = 0; index < program.length; index++) {
    const char = program[index];
    if (char === '+' || char === '*' || char === '/') return true;
    if (char !== '-') continue;
    const previous = previousSignificant(index);
    if (previous === ':' || previous === '?') continue;
    const next = nextSignificant(index);
    const previousEndsOperand = /[a-zA-Z0-9_)]/.test(previous);
    const isNegativeNumericLiteral = !previousEndsOperand && /[0-9]/.test(next);
    if (!isNegativeNumericLiteral) return true;
  }
  return false;
}

interface SyntaxInspection {
  visible: string;
  negation: boolean;
  aggregation: boolean;
  identity: boolean;
  trust: boolean;
  arithmetic: boolean;
  rule: boolean;
  headless: boolean;
}

function inspectSyntax(program: string): SyntaxInspection {
  let quoted = false;
  let visible = '';
  let negation = false;
  for (let index = 0; index < program.length; index++) {
    const char = program[index];
    if (quoted) {
      visible += ' ';
      if (char === "'" && program[index + 1] === "'") {
        visible += ' ';
        index++;
      }
      else if (char === "'") quoted = false;
      continue;
    }
    if (char === "'") {
      quoted = true;
      visible += ' ';
      continue;
    }
    if (char === '%') {
      while (index < program.length && program[index] !== '\n') {
        visible += ' ';
        index++;
      }
      visible += '\n';
      continue;
    }
    if (char === '\\' && program[index + 1] === '+') {
      negation = true;
    }
    visible += char;
  }
  return {
    visible,
    negation,
    aggregation:
      /\b(?:count|sum|min|max)\s*\([^)]*\)\s+as\s+[A-Z][a-zA-Z0-9_]*\s+where\b/.test(
        visible
      ),
    identity: /\b(?:rembero_alias|rembero_entity_position)\s*\(/.test(visible),
    trust: /\brembero_tentative\s*\(/.test(visible),
    arithmetic: containsArithmeticSyntax(visible),
    rule: visible.includes(':-'),
    headless: visible.trimStart().startsWith(':-'),
  };
}

function assertNoIdentitySyntax(inspection: SyntaxInspection): void {
  if (inspection.identity) {
    throw new Error(
      'entity identity declarations are currently supported by the portable Datalog engine, not the SQLite extension'
    );
  }
  if (inspection.trust) {
    throw new Error(
      'tentative trust declarations are personal knowledge-store metadata, not SQLite predicates'
    );
  }
}

function assertSqlCompilable(program: string): void {
  const inspection = inspectSyntax(program);
  assertNoIdentitySyntax(inspection);
  if (inspection.negation) {
    throw new Error(
      'stratified negation cannot be compiled to one SQLite SELECT; use datalogQuery or datalogExplain'
    );
  }
  if (inspection.aggregation) {
    throw new Error(
      'aggregation cannot be compiled to one SQLite SELECT; use datalogQuery or datalogExplain'
    );
  }
  if (inspection.arithmetic) {
    throw new Error(
      'arithmetic comparison expressions cannot be compiled to one SQLite SELECT; use datalogQuery or datalogExplain'
    );
  }
}

export function sqliteDatalogExecutionMode(program: string): SqliteDatalogExecutionMode {
  const inspection = inspectSyntax(program);
  assertNoIdentitySyntax(inspection);
  if (
    inspection.negation ||
    inspection.aggregation ||
    inspection.arithmetic ||
    inspection.headless ||
    !inspection.rule
  ) {
    return 'portable';
  }
  try {
    const clauses = parseProgram(program);
    if (clauses.some(isIntegrityConstraint)) return 'portable';
    if (new Set(clauses.map((clause) => predKey(clause.head))).size > 1) {
      return 'portable';
    }
  } catch {
    // Let the native parser retain its established error contract for ordinary rules.
  }
  return 'native';
}

interface PortableRequest {
  program: Clause[];
  query: QuerySpec;
  basePredicates: Array<{ predicate: string; arity: number }>;
}

function literalFromGoal(goal: Goal) {
  if (isComparison(goal)) return undefined;
  return isNegation(goal) ? goal.not : goal;
}

function preparePortableRequest(input: string): PortableRequest {
  if (Buffer.byteLength(input, 'utf8') > MAX_RULE_BYTES) {
    throw new Error('Datalog program exceeds 64 KiB');
  }
  if (input.includes('\0')) throw new Error('Datalog program contains a NUL byte');
  const inspection = inspectSyntax(input);
  assertNoIdentitySyntax(inspection);
  let program: Clause[];
  let query: QuerySpec;

  if (!inspection.rule) {
    program = [];
    query = parseQuerySpec(input);
  } else {
    program = parseProgram(input);
    if (program.some(isIntegrityConstraint)) {
      throw new Error(
        'integrity constraints are policies for the personal knowledge store, not SQLite queries'
      );
    }
    const target = program[0]?.head;
    if (target === undefined) throw new Error('expected a Datalog rule or query');
    const names = new Set<string>();
    for (const term of target.args) {
      if (term.type !== 'var' || term.name === '_' || names.has(term.name)) {
        throw new Error('SQLite query rule head terms must be distinct named variables');
      }
      names.add(term.name);
    }
    if (target.args.length === 0) {
      throw new Error('SQLite query rule head must contain at least one named variable');
    }
    query = { kind: 'relational', goals: [target] };
  }

  const derivedByName = new Map<string, number>();
  for (const clause of program) {
    const existing = derivedByName.get(clause.head.predicate);
    if (existing !== undefined && existing !== clause.head.args.length) {
      throw new Error(`predicate '${clause.head.predicate}' has inconsistent arity`);
    }
    derivedByName.set(clause.head.predicate, clause.head.args.length);
  }
  const baseByName = new Map<string, number>();
  const addBase = (predicate: string, arity: number) => {
    const derivedArity = derivedByName.get(predicate);
    if (derivedArity !== undefined) {
      if (derivedArity !== arity) {
        throw new Error(`predicate '${predicate}' has inconsistent arity`);
      }
      return;
    }
    const existing = baseByName.get(predicate);
    if (existing !== undefined && existing !== arity) {
      throw new Error(`predicate '${predicate}' has inconsistent arity`);
    }
    baseByName.set(predicate, arity);
  };
  for (const clause of program) {
    for (const goal of clause.body) {
      const literal = literalFromGoal(goal);
      if (literal !== undefined) addBase(literal.predicate, literal.args.length);
    }
  }
  for (const goal of query.goals) {
    const literal = literalFromGoal(goal);
    if (literal !== undefined) addBase(literal.predicate, literal.args.length);
  }
  return {
    program,
    query,
    basePredicates: [...baseByName]
      .map(([predicate, arity]) => ({ predicate, arity }))
      .sort((left, right) =>
        left.predicate.localeCompare(right.predicate) || left.arity - right.arity
      ),
  };
}

function quoteIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

function portableTerm(value: unknown, predicate: string, column: number): Term {
  if (value === null) {
    throw new Error(
      `predicate '${predicate}' column ${column + 1} contains NULL, which the portable SQLite bridge cannot represent`
    );
  }
  if (value instanceof Uint8Array) {
    throw new Error(
      `predicate '${predicate}' column ${column + 1} contains a BLOB, which the portable SQLite bridge cannot represent`
    );
  }
  if (typeof value === 'bigint') {
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new Error(
        `predicate '${predicate}' column ${column + 1} contains an INTEGER outside the portable safe range`
      );
    }
    return { type: 'num', value: Number(value) };
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`predicate '${predicate}' contains a non-finite number`);
    }
    return { type: 'num', value };
  }
  if (typeof value === 'string') return { type: 'atom', value };
  throw new Error(`predicate '${predicate}' contains an unsupported SQLite value`);
}

function rowFromBindings(bindings: Bindings): DatalogRow {
  return Object.fromEntries(
    Object.entries(bindings).map(([name, term]) => {
      if (term.type !== 'atom' && term.type !== 'num') {
        throw new Error(`Datalog result variable '${name}' is not ground`);
      }
      return [name, term.value];
    })
  );
}

function expressionVariables(expression: ScalarExpression, add: (name: string) => void): void {
  if (!isArithmeticExpression(expression)) {
    if (expression.type === 'var' && expression.name !== '_') add(expression.name);
    return;
  }
  if (expression.kind === 'unary') expressionVariables(expression.operand, add);
  else {
    expressionVariables(expression.left, add);
    expressionVariables(expression.right, add);
  }
}

function queryVariables(query: QuerySpec): string[] {
  if (query.kind === 'aggregate') return [query.as];
  if (query.project !== undefined) return [...query.project];
  const result: string[] = [];
  const seen = new Set<string>();
  const add = (name: string) => {
    if (seen.has(name)) return;
    seen.add(name);
    result.push(name);
  };
  for (const goal of query.goals) {
    if (isComparison(goal)) {
      expressionVariables(goal.left, add);
      expressionVariables(goal.right, add);
      continue;
    }
    const literal = isNegation(goal) ? goal.not : goal;
    for (const term of literal.args) {
      if (term.type === 'var' && term.name !== '_') add(term.name);
    }
  }
  return result;
}

function derivedPredicatePlans(program: Clause[]): SqliteDatalogPlanDerivedPredicate[] {
  const arities = new Map<string, number>();
  for (const clause of program) arities.set(clause.head.predicate, clause.head.args.length);
  const dependencies = new Map<string, Set<string>>();
  for (const clause of program) {
    const values = dependencies.get(clause.head.predicate) ?? new Set<string>();
    for (const goal of clause.body) {
      const literal = literalFromGoal(goal);
      if (literal !== undefined && arities.has(literal.predicate)) {
        values.add(literal.predicate);
      }
    }
    dependencies.set(clause.head.predicate, values);
  }
  const recursive = (predicate: string): boolean => {
    const pending = [...(dependencies.get(predicate) ?? [])];
    const visited = new Set<string>();
    while (pending.length > 0) {
      const current = pending.pop()!;
      if (current === predicate) return true;
      if (visited.has(current)) continue;
      visited.add(current);
      pending.push(...(dependencies.get(current) ?? []));
    }
    return false;
  };
  return [...arities]
    .map(([predicate, arity]) => ({ predicate, arity, recursive: recursive(predicate) }))
    .sort((left, right) =>
      left.predicate.localeCompare(right.predicate) || left.arity - right.arity
    );
}

function assertResultBounds(result: unknown): void {
  const serialized = JSON.stringify(result);
  if (serialized === undefined || Buffer.byteLength(serialized, 'utf8') > MAX_RESULT_BYTES) {
    throw new Error('Datalog result exceeded 16 MiB');
  }
}

function platformLibraryName(): string {
  switch (process.platform) {
    case 'darwin':
      return 'rembero.dylib';
    case 'linux':
      return 'rembero.so';
    default:
      throw new Error(`Remembero SQLite V0 does not support ${process.platform}.`);
  }
}

function packageRoot(): string {
  return fileURLToPath(new URL('../../', import.meta.url));
}

export function buildSqliteExtension(): string {
  if (process.platform !== 'darwin' && process.platform !== 'linux') {
    throw new Error(`Remembero SQLite V0 does not support ${process.platform}.`);
  }
  const root = packageRoot();
  const output = execFileSync('sh', [resolve(root, 'native', 'build.sh')], {
    cwd: root,
    encoding: 'utf8',
  });
  const builtPath = output.trim().split('\n').at(-1) ?? '';
  if (!existsSync(builtPath)) {
    throw new Error('SQLite extension build completed without producing a library.');
  }
  return builtPath;
}

export function resolveSqliteExtensionPath(explicitPath?: string): string {
  const configured = explicitPath ?? process.env.REMBERO_SQLITE_EXTENSION;
  const candidate = resolve(configured ?? resolve(packageRoot(), 'build', platformLibraryName()));
  if (!existsSync(candidate)) {
    throw new Error(
      `Remembero SQLite extension not found at ${candidate}. ` +
        'Run "remembero sqlite-build" (or "npm run build:sqlite" in a checkout), ' +
        'or set REMBERO_SQLITE_EXTENSION.'
    );
  }
  return candidate;
}

export class DatalogDatabase {
  constructor(private readonly database: DatabaseSync) {}

  exec(sql: string): void {
    this.database.exec(sql);
  }

  datalogSql(rule: string): string {
    assertSqlCompilable(rule);
    const row = this.database.prepare('SELECT datalog_sql(?) AS sql').get(rule) as
      | { sql: unknown }
      | undefined;
    if (typeof row?.sql !== 'string') {
      throw new Error('SQLite datalog_sql returned an invalid result');
    }
    return row.sql;
  }

  datalogQuery(rule: string): DatalogRow[] {
    if (sqliteDatalogExecutionMode(rule) === 'portable') {
      return this.portableQuery(rule);
    }
    const row = this.database.prepare('SELECT datalog_query(?) AS result').get(rule) as
      | { result: unknown }
      | undefined;
    if (typeof row?.result !== 'string') {
      throw new Error('SQLite datalog_query returned an invalid result');
    }
    const result: unknown = JSON.parse(row.result);
    if (!Array.isArray(result)) {
      throw new Error('SQLite datalog_query returned invalid JSON');
    }
    return result as DatalogRow[];
  }

  datalogExplain(program: string): DatalogExplanation[] {
    if (sqliteDatalogExecutionMode(program) === 'portable') {
      return this.portableExplain(program);
    }
    const row = this.database.prepare('SELECT datalog_explain(?) AS result').get(program) as
      | { result: unknown }
      | undefined;
    if (typeof row?.result !== 'string') {
      throw new Error('SQLite datalog_explain returned an invalid result');
    }
    const result: unknown = JSON.parse(row.result);
    if (!Array.isArray(result)) {
      throw new Error('SQLite datalog_explain returned invalid JSON');
    }
    return result as DatalogExplanation[];
  }

  datalogPlan(input: string): SqliteDatalogPlan {
    return this.withPortableSnapshot(() => {
      const mode = sqliteDatalogExecutionMode(input);
      const request = preparePortableRequest(input);
      const derivedPredicates = derivedPredicatePlans(request.program);
      const target = request.program[0]?.head;
      const inputKind =
        request.program.length > 0
          ? 'rule_program'
          : request.query.kind === 'aggregate'
            ? 'aggregate_query'
            : 'relational_query';
      const variables =
        target === undefined
          ? queryVariables(request.query)
          : target.args.map((term) => (term.type === 'var' ? term.name : '_'));
      const baseRelations = request.basePredicates.map(({ predicate, arity }) =>
        this.relationPlan(predicate, arity)
      );
      const nativeSqlEligible =
        mode === 'native' &&
        request.program.length === 1 &&
        derivedPredicates.every(({ recursive }) => !recursive);
      return {
        mode,
        executionBoundary:
          mode === 'native' ? 'sqlite_scalar' : 'portable_snapshot',
        inputKind,
        result: {
          ...(target === undefined ? {} : { predicate: target.predicate }),
          variables,
        },
        derivedPredicates,
        baseRelations,
        ...(nativeSqlEligible ? { nativeSql: this.datalogSql(input) } : {}),
        scansData: false,
        bounds: {
          ruleBytes: MAX_RULE_BYTES,
          baseRows: MAX_BASE_ROWS,
          derivedFacts: MAX_DERIVED_FACTS,
          queryRows: MAX_QUERY_ROWS,
          resultBytes: MAX_RESULT_BYTES,
        },
      };
    });
  }

  close(): void {
    this.database.close();
  }

  private withPortableSnapshot<T>(operation: () => T): T {
    this.database.exec(`SAVEPOINT ${PORTABLE_SAVEPOINT}`);
    try {
      const result = operation();
      this.database.exec(`RELEASE ${PORTABLE_SAVEPOINT}`);
      return result;
    } catch (error) {
      let cleanupError: unknown;
      try {
        this.database.exec(`ROLLBACK TO ${PORTABLE_SAVEPOINT}`);
      } catch (rollbackError) {
        cleanupError = rollbackError;
      }
      try {
        this.database.exec(`RELEASE ${PORTABLE_SAVEPOINT}`);
      } catch (releaseError) {
        cleanupError ??= releaseError;
      }
      if (cleanupError !== undefined) {
        const primary = error instanceof Error ? error.message : String(error);
        const cleanup =
          cleanupError instanceof Error ? cleanupError.message : String(cleanupError);
        throw new Error(`${primary}; portable SQLite snapshot cleanup failed: ${cleanup}`, {
          cause: error,
        });
      }
      throw error;
    }
  }

  private relationPlan(predicate: string, arity: number): SqliteDatalogPlanRelation {
    const schema = this.database
      .prepare(`PRAGMA table_xinfo('${predicate.replaceAll("'", "''")}')`)
      .all() as Array<Record<string, unknown>>;
    const columns = schema
      .filter((column) => Number(column.hidden ?? 0) !== 1)
      .map((column): SqliteDatalogPlanColumn | undefined => {
        if (typeof column.name !== 'string') return undefined;
        return {
          name: column.name,
          declaredType: typeof column.type === 'string' && column.type.length > 0
            ? column.type
            : null,
          hidden: Number(column.hidden ?? 0),
        };
      })
      .filter((column): column is SqliteDatalogPlanColumn => column !== undefined);
    if (columns.length === 0) throw new Error(`predicate '${predicate}' is unavailable`);
    if (columns.length !== arity) {
      throw new Error(
        `predicate '${predicate}' expects ${columns.length} columns but the query supplies ${arity}`
      );
    }
    const object = this.database
      .prepare(
        `SELECT type, temporary FROM (
           SELECT type, 1 AS temporary FROM sqlite_temp_schema WHERE name = ?
           UNION ALL
           SELECT type, 0 AS temporary FROM sqlite_schema WHERE name = ?
         ) ORDER BY temporary DESC LIMIT 1`
      )
      .get(predicate, predicate) as
      | { type: unknown; temporary: unknown }
      | undefined;
    return {
      predicate,
      arity,
      objectType: object?.type === 'view' ? 'view' : 'table',
      temporary: Number(object?.temporary ?? 0) === 1,
      columns,
    };
  }

  private portableClauses(request: PortableRequest): Clause[] {
    const facts: Clause[] = [];
    let totalBytes = 0;
    for (const { predicate, arity } of request.basePredicates) {
      const columns = this.relationPlan(predicate, arity).columns.map(({ name }) => name);
      const selected = columns
        .map((column, index) => `${quoteIdentifier(column)} AS ${quoteIdentifier(`c${index}`)}`)
        .join(', ');
      const statement = this.database.prepare(
        `SELECT ${selected} FROM ${quoteIdentifier(predicate)}`
      );
      statement.setReadBigInts(true);
      for (const row of statement.iterate()) {
        if (facts.length >= MAX_BASE_ROWS) {
          throw new Error(`base relations exceed ${MAX_BASE_ROWS} rows`);
        }
        const clause: Clause = {
          head: {
            predicate,
            args: columns.map((_, index) => portableTerm(row[`c${index}`], predicate, index)),
          },
          body: [],
        };
        totalBytes += Buffer.byteLength(serializeClause(clause), 'utf8');
        if (totalBytes > MAX_RESULT_BYTES) {
          throw new Error('portable SQLite bridge input exceeded 16 MiB');
        }
        facts.push(clause);
      }
    }
    facts.sort((left, right) => serializeClause(left).localeCompare(serializeClause(right)));
    return [...facts, ...request.program];
  }

  private portableQuery(input: string): DatalogRow[] {
    return this.withPortableSnapshot(() => {
      const request = preparePortableRequest(input);
      const clauses = this.portableClauses(request);
      const bindings = evaluateQuerySpec(clauses, request.query, {
        maxFacts: MAX_BASE_ROWS + MAX_DERIVED_FACTS,
        maxIterations: 1_000,
        maxRows: MAX_QUERY_ROWS + 1,
        maxAggregateRows: MAX_BASE_ROWS,
      });
      if (bindings.length > MAX_QUERY_ROWS) {
        throw new Error(`Datalog query exceeded ${MAX_QUERY_ROWS} rows`);
      }
      const rows = bindings.map(rowFromBindings);
      assertResultBounds(rows);
      return rows;
    });
  }

  private portableExplain(input: string): DatalogExplanation[] {
    return this.withPortableSnapshot(() => {
      const request = preparePortableRequest(input);
      const clauses = this.portableClauses(request);
      const explained = evaluateQuerySpecWithProof(clauses, request.query, {
        maxFacts: MAX_BASE_ROWS + MAX_DERIVED_FACTS,
        maxIterations: 1_000,
        maxRows: MAX_QUERY_ROWS + 1,
        maxAggregateRows: MAX_BASE_ROWS,
        maxAggregateProofRows: 256,
        maxProofDepth: 128,
        maxProofNodes: 100_000,
      });
      if (explained.length > MAX_QUERY_ROWS) {
        throw new Error(`Datalog query exceeded ${MAX_QUERY_ROWS} rows`);
      }
      const results = explained.map(({ bindings, proofs }) => {
        const proof = proofs[0];
        if (proof === undefined) throw new Error('Datalog explanation has no proof');
        return {
          row: rowFromBindings(bindings),
          proof,
          ...(proofs.length > 1 ? { proofs } : {}),
        };
      });
      assertResultBounds(results);
      return results;
    });
  }
}

type DatalogEnabledDatabase = DatabaseSync & DatalogDatabaseMethods;

function attachDatalogDatabaseMethods(database: DatabaseSync): DatalogEnabledDatabase {
  const existing = database as Partial<DatalogEnabledDatabase>;
  if (
    typeof existing.datalogSql === 'function' &&
    typeof existing.datalogQuery === 'function' &&
    typeof existing.datalogExplain === 'function' &&
    typeof existing.datalogPlan === 'function'
  ) {
    return database as DatalogEnabledDatabase;
  }

  const adapter = new DatalogDatabase(database);
  Object.defineProperties(database, {
    datalogSql: {
      value: adapter.datalogSql.bind(adapter),
      configurable: true,
      writable: false,
    },
    datalogQuery: {
      value: adapter.datalogQuery.bind(adapter),
      configurable: true,
      writable: false,
    },
    datalogExplain: {
      value: adapter.datalogExplain.bind(adapter),
      configurable: true,
      writable: false,
    },
    datalogPlan: {
      value: adapter.datalogPlan.bind(adapter),
      configurable: true,
      writable: false,
    },
  });
  return database as DatalogEnabledDatabase;
}

function validateDatalogDatabaseOptions(
  options: OpenDatalogDatabaseOptions
): void {
  const rawOptions = options as OpenDatalogDatabaseOptions & {
    allowExtension?: unknown;
    open?: unknown;
  };
  if (Object.prototype.hasOwnProperty.call(rawOptions, 'open')) {
    throw new Error(
      'Remembero SQLite integration opens the database during initialization and does not accept an open option.'
    );
  }
  if (Object.prototype.hasOwnProperty.call(rawOptions, 'allowExtension')) {
    throw new Error(
      'Remembero SQLite integration manages allowExtension internally and does not accept an override.'
    );
  }
  const unsupported = Object.keys(rawOptions).find(
    (key) => !OPEN_DATABASE_OPTION_KEYS.has(key)
  );
  if (unsupported !== undefined) {
    throw new Error(`unsupported Remembero SQLite database option '${unsupported}'`);
  }
}

function disableFurtherExtensionLoading(database: DatabaseSync): void {
  database.enableLoadExtension(false);
  const disabled = () => {
    throw new Error(
      'SQLite extension loading is disabled after Remembero database initialization.'
    );
  };
  Object.defineProperties(database, {
    loadExtension: {
      value: disabled,
      configurable: false,
      writable: false,
    },
    enableLoadExtension: {
      value: disabled,
      configurable: false,
      writable: false,
    },
  });
}

async function importSqliteModule(): Promise<typeof import('node:sqlite')> {
  try {
    return await import('node:sqlite');
  } catch (error) {
    throw new Error('SQLite integration requires Node.js 22.13 or newer.', { cause: error });
  }
}

function openDatalogEnabledDatabaseSync(
  sqlite: typeof import('node:sqlite'),
  path: string,
  options: OpenDatalogDatabaseOptions = {}
): DatalogEnabledDatabase {
  validateDatalogDatabaseOptions(options);
  const {
    extensionPath,
    enableForeignKeyConstraints,
    enableDoubleQuotedStringLiterals,
    readOnly,
  } = options;
  const database = new sqlite.DatabaseSync(path, {
    ...(enableForeignKeyConstraints === undefined ? {} : { enableForeignKeyConstraints }),
    ...(enableDoubleQuotedStringLiterals === undefined
      ? {}
      : { enableDoubleQuotedStringLiterals }),
    ...(readOnly === undefined ? {} : { readOnly }),
    allowExtension: true,
  });
  try {
    database.loadExtension(resolveSqliteExtensionPath(extensionPath), 'sqlite3_rembero_init');
    disableFurtherExtensionLoading(database);
    return attachDatalogDatabaseMethods(database);
  } catch (error) {
    database.close();
    throw error;
  }
}

function attachSqliteMemoryStore(
  database: DatalogEnabledDatabase,
  options: SqliteMemoryStoreOptions
): RememberoDatabase {
  if ('memory' in database) {
    throw new Error("SQLite database already has a 'memory' property");
  }
  const memory = createSqliteMemoryStore(database, options);
  const closeDatabase = database.close.bind(database);
  const close = () => {
    try {
      memory.dispose();
    } finally {
      closeDatabase();
    }
  };
  Object.defineProperties(database, {
    memory: {
      value: memory,
      configurable: false,
      writable: false,
    },
    close: {
      value: close,
      configurable: false,
      writable: false,
    },
  });
  if (typeof Symbol.dispose === 'symbol' && Symbol.dispose in database) {
    Object.defineProperty(database, Symbol.dispose, {
      value: close,
      configurable: false,
      writable: false,
    });
  }
  return database as RememberoDatabase;
}

export async function openRememberoDatabase(
  path: string,
  options: OpenRememberoDatabaseOptions = {}
): Promise<RememberoDatabase> {
  const { memory = {}, ...databaseOptions } = options;
  const database = openDatalogEnabledDatabaseSync(
    await importSqliteModule(),
    path,
    databaseOptions
  );
  try {
    return attachSqliteMemoryStore(database, memory);
  } catch (error) {
    database.close();
    throw error;
  }
}

export async function openDatalogDatabase(
  path: string,
  options: OpenDatalogDatabaseOptions = {}
): Promise<DatalogDatabase> {
  return new DatalogDatabase(
    openDatalogEnabledDatabaseSync(await importSqliteModule(), path, options)
  );
}
