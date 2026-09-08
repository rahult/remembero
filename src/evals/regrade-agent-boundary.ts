/**
 * Re-grade the query leg of existing agent-boundary result files.
 *
 *   node dist/evals/regrade-agent-boundary.js docs/research/results/agent-boundary-v2-*.json
 *
 * For every stored outcome, re-executes the model-authored query on a freshly
 * seeded database (replaying the trap write the way the run did), grades the
 * rows with gradeQueryRows, and rewrites `queryCorrect` per outcome plus
 * `queryCorrect`/`queryMean` in the summary. End-to-end fields are untouched.
 * This is how query-correct numbers are obtained for result files produced
 * before the metric existed, and how they are refreshed when the grader
 * changes.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import {
  AGENT_BOUNDARY_QUESTIONS,
  AGENT_BOUNDARY_SEED_SQL,
  assertReadOnlySql,
  entitiesFromRows,
  gradeQueryRows,
  type AgentBoundaryCondition,
  type AgentBoundaryQuestion,
} from './agent-boundary.js';
import { applyGatedWrite } from './agent-boundary-gate.js';
import { openRememberoDatabase } from '../sqlite/extension.js';

interface StoredOutcome {
  id: string;
  condition: AgentBoundaryCondition;
  seed: number;
  passed: boolean;
  queryCorrect?: boolean;
  query: string;
}

interface StoredSeedStats {
  seed: number;
  passed: number;
  queryCorrect?: number;
  total: number;
}

interface StoredConditionStats {
  perSeed: StoredSeedStats[];
  mean: number;
  queryMean?: number;
}

interface StoredSummary {
  byCategory: Array<{
    category: string;
    conditions: Record<string, StoredConditionStats>;
  }>;
  totals: Record<string, StoredConditionStats>;
}

interface StoredResults {
  model: string;
  summary: StoredSummary;
  outcomes: StoredOutcome[];
}

export async function regradeQuery(
  question: AgentBoundaryQuestion,
  condition: AgentBoundaryCondition,
  query: string,
): Promise<boolean> {
  const db = await openRememberoDatabase(':memory:');
  try {
    db.exec(AGENT_BOUNDARY_SEED_SQL);
    const readGold = () =>
      entitiesFromRows(
        db.prepare(question.goldSql).all() as Array<Record<string, unknown>>,
      );
    let gold = question.control === true ? new Set<string>() : readGold();
    let refused = false;
    if (question.trapWriteSql !== undefined) {
      if (condition === 'sql') db.exec(question.trapWriteSql);
      else refused = applyGatedWrite(db, question.trapWriteSql).refused;
      if (question.control === true) gold = readGold();
    }
    if (question.control === true && refused) return false;
    let rows: Array<Record<string, unknown>>;
    try {
      if (condition === 'sql' || condition === 'sql-gated') {
        assertReadOnlySql(query);
        rows = db.prepare(query).all() as Array<Record<string, unknown>>;
      } else {
        rows = db.datalogQuery(query) as Array<Record<string, unknown>>;
      }
    } catch {
      return false;
    }
    return gradeQueryRows(question, rows, gold).passed;
  } finally {
    db.close();
  }
}

function refreshStats(
  stats: StoredConditionStats,
  outcomes: StoredOutcome[],
  condition: string,
  categoryFilter?: (id: string) => boolean,
): void {
  for (const seedRow of stats.perSeed) {
    seedRow.queryCorrect = outcomes.filter(
      (o) =>
        o.condition === condition &&
        o.seed === seedRow.seed &&
        o.queryCorrect === true &&
        (categoryFilter === undefined || categoryFilter(o.id)),
    ).length;
  }
  const values = stats.perSeed.map((row) => row.queryCorrect ?? 0);
  stats.queryMean =
    values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

export async function regradeFile(
  path: string,
): Promise<{ model: string; totals: string }> {
  const results = JSON.parse(readFileSync(path, 'utf8')) as StoredResults;
  const byId = new Map(AGENT_BOUNDARY_QUESTIONS.map((q) => [q.id, q]));
  for (const outcome of results.outcomes) {
    const question = byId.get(outcome.id);
    if (!question) throw new Error(`${path}: unknown question ${outcome.id}`);
    outcome.queryCorrect = await regradeQuery(
      question,
      outcome.condition,
      outcome.query,
    );
  }
  for (const [condition, stats] of Object.entries(results.summary.totals)) {
    refreshStats(stats, results.outcomes, condition);
  }
  for (const row of results.summary.byCategory) {
    for (const [condition, stats] of Object.entries(row.conditions)) {
      refreshStats(
        stats,
        results.outcomes,
        condition,
        (id) => byId.get(id)?.category === row.category,
      );
    }
  }
  writeFileSync(path, `${JSON.stringify(results, null, 2)}\n`);
  const totals = Object.entries(results.summary.totals)
    .map(
      ([condition, stats]) =>
        `${condition}=${stats.queryMean?.toFixed(1)}/${stats.perSeed[0]?.total ?? 0}`,
    )
    .join(' ');
  return { model: results.model, totals };
}

if (process.argv[1]?.endsWith('regrade-agent-boundary.js')) {
  for (const path of process.argv.slice(2)) {
    const { model, totals } = await regradeFile(path);
    console.log(`${path}\n  ${model}: ${totals}`);
  }
}
