/**
 * Shared write gate for the agent-boundary benchmark, built on the product's
 * shipped enforcement primitives (enforceIntegrityCandidate / checkIntegrity)
 * instead of a benchmark-local reimplementation. The gated conditions
 * (sql-gated, remembero) and the test suite both route writes through this
 * module, so published gate evidence exercises the same mechanism the
 * knowledge store enforces on real memory writes.
 *
 * The three WRITE_GATE_RULES stay byte-frozen (ADR 0003). Their headless
 * constraint form is derived mechanically by stripping the `violation(...)`
 * head, and test-enforced equivalence proves the derived constraints make
 * identical gate decisions to the frozen rules.
 */
import { type Clause, parseProgram } from '../engine/index.js';
import {
  enforceIntegrityCandidate,
  IntegrityViolationError,
} from '../knowledge/enforcement.js';
import type { RememberoDatabase } from '../sqlite/extension.js';
import { AGENT_BOUNDARY_SEED_SQL, WRITE_GATE_RULES } from './agent-boundary.js';

const BARE_ATOM = /^[a-z][a-z0-9_]*$/;

function seedTableNames(): string[] {
  const names: string[] = [];
  for (const match of AGENT_BOUNDARY_SEED_SQL.matchAll(
    /CREATE TABLE (\w+)\(/g,
  )) {
    names.push(match[1]);
  }
  return names;
}

/**
 * Headless integrity constraints derived from the frozen WRITE_GATE_RULES:
 * `violation(P, B) :- body.` becomes `:- body.`. Mechanical transformation
 * only — semantics live in the shared body text.
 */
export function gateConstraintPrograms(): string[] {
  return WRITE_GATE_RULES.map((rule) => {
    const separator = rule.program.indexOf(':-');
    if (separator < 0) {
      throw new Error(
        `frozen gate rule '${rule.name}' lost its rule separator`,
      );
    }
    return `:-${rule.program.slice(separator + 2)}`;
  });
}

function gateConstraintClauses(): Clause[] {
  return gateConstraintPrograms().flatMap((program) => parseProgram(program));
}

/**
 * Materialize every table row as a Datalog fact clause so the portable
 * enforcement engine sees the same world the SQL/datalog query paths see.
 * Seed values are controlled lowercase atoms; anything else fails loudly
 * rather than being silently mangled.
 */
export function collectFactClauses(db: RememberoDatabase): Clause[] {
  const chunks: string[] = [];
  for (const table of seedTableNames()) {
    const columns = (
      db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
    ).map((column) => column.name);
    const rows = db.prepare(`SELECT * FROM ${table}`).all() as Array<
      Record<string, unknown>
    >;
    for (const row of rows) {
      const args = columns.map((column) => {
        const value = String(row[column]);
        if (!BARE_ATOM.test(value)) {
          throw new Error(
            `benchmark seed value '${value}' in ${table}.${column} is not a bare atom`,
          );
        }
        return value;
      });
      chunks.push(`${table}(${args.join(', ')}).`);
    }
  }
  return chunks.length > 0 ? parseProgram(chunks.join('\n')) : [];
}

export interface GatedWriteResult {
  refused: boolean;
  /** Violations blocking the write when refused (strict mode counts all). */
  violationCount: number;
}

/**
 * Apply a write through the product enforcement path: inside a savepoint,
 * evaluate the derived constraints over baseline and candidate fact sets in
 * strict mode, and roll back on any violation. Baseline cleanliness is
 * test-enforced, so strict is equivalent to the benchmark's original
 * "any violation refuses" semantics.
 */
export function applyGatedWrite(
  db: RememberoDatabase,
  trapWriteSql: string,
): GatedWriteResult {
  const constraints = gateConstraintClauses();
  const baseline = [...collectFactClauses(db), ...constraints];
  db.exec('SAVEPOINT gate');
  db.exec(trapWriteSql);
  try {
    const candidate = [...collectFactClauses(db), ...constraints];
    enforceIntegrityCandidate(baseline, candidate, new Map(), new Map(), {
      mode: 'strict',
    });
  } catch (error) {
    if (error instanceof IntegrityViolationError) {
      db.exec('ROLLBACK TO gate');
      db.exec('RELEASE gate');
      return { refused: true, violationCount: error.blockingViolations.length };
    }
    throw error;
  }
  db.exec('RELEASE gate');
  return { refused: false, violationCount: 0 };
}
