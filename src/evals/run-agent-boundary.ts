/**
 * Runs the agent-boundary benchmark against a local Ollama model.
 *
 *   node dist/evals/run-agent-boundary.js --model llama3.2:3b [--seeds 7,42,123]
 *
 * Same seeded SQLite database, same model; the model authors every query
 * itself. The sql condition executes model-written read-only SQL with no
 * integrity gate; sql-gated executes the same SQL but routes writes through
 * the Remembero integrity gate; the remembero condition executes
 * model-written Datalog via the same bridge the MCP server and browser labs
 * use. Write-trap questions push a corrupting write through each condition's
 * write path first: raw SQL applies it, the gated conditions check their
 * integrity rules and refuse. v2: answer-set grading (ADR 0003), three seeds
 * with reported spread, and gate-protected correctness as the headline
 * write-trap metric.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AGENT_BOUNDARY_CONDITIONS,
  AGENT_BOUNDARY_QUESTIONS,
  AGENT_BOUNDARY_SEED_SQL,
  answerSystemPrompt,
  assertReadOnlySql,
  datalogSystemPrompt,
  entitiesFromRows,
  gradeAnswerV2,
  sqlSystemPrompt,
  stripFences,
  type AgentBoundaryCondition,
  type AgentBoundaryQuestion,
} from './agent-boundary.js';
import { applyGatedWrite } from './agent-boundary-gate.js';
import {
  openRememberoDatabase,
  type RememberoDatabase,
} from '../sqlite/extension.js';

const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const MAX_RESULT_ROWS = 30;
const MAX_ATTEMPTS = 2;

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

async function chat(
  model: string,
  messages: ChatMessage[],
  seed: number,
): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { temperature: 0, seed, num_ctx: 4096, num_predict: 400 },
    }),
  });
  if (!response.ok) {
    throw new Error(
      `ollama returned ${response.status}: ${await response.text()}`,
    );
  }
  const payload = (await response.json()) as { message?: { content?: string } };
  const content = payload.message?.content;
  if (typeof content !== 'string')
    throw new Error('ollama returned no message content');
  return content;
}

interface QuestionOutcome {
  id: string;
  category: string;
  condition: AgentBoundaryCondition;
  seed: number;
  passed: boolean;
  toolErrors: number;
  gateRefusedTrap?: boolean;
  query: string;
  answer: string;
  missing: string[];
  forbidden: string[];
  extraEntities: string[];
}

function executeQuery(
  db: RememberoDatabase,
  condition: AgentBoundaryCondition,
  raw: string,
): Array<Record<string, unknown>> {
  const query = stripFences(raw);
  if (condition === 'sql' || condition === 'sql-gated') {
    assertReadOnlySql(query);
    return db.prepare(query).all() as Array<Record<string, unknown>>;
  }
  return db.datalogQuery(query) as Array<Record<string, unknown>>;
}

function applyTrapWrite(
  db: RememberoDatabase,
  condition: AgentBoundaryCondition,
  trapWriteSql: string,
): { refused: boolean } {
  if (condition === 'sql') {
    db.exec(trapWriteSql);
    return { refused: false };
  }
  // sql-gated and remembero share the product enforcement path (ADR 0002):
  // the same enforceIntegrityCandidate the knowledge store uses on writes.
  return applyGatedWrite(db, trapWriteSql);
}

async function runQuestion(
  model: string,
  condition: AgentBoundaryCondition,
  question: AgentBoundaryQuestion,
  seed: number,
): Promise<QuestionOutcome> {
  const db = await openRememberoDatabase(':memory:');
  try {
    db.exec(AGENT_BOUNDARY_SEED_SQL);
    // Gold answer-set entities for v2 grading (ADR 0003). Non-control
    // questions are graded against the CLEAN truth, so their gold rows are
    // read before any trap write; control questions are graded against the
    // POST-write truth their benign write establishes.
    const readGoldEntities = () =>
      entitiesFromRows(
        db.prepare(question.goldSql).all() as Array<Record<string, unknown>>,
      );
    let goldEntities =
      question.control === true ? new Set<string>() : readGoldEntities();
    let gateRefusedTrap: boolean | undefined;
    if (question.trapWriteSql !== undefined) {
      gateRefusedTrap = applyTrapWrite(
        db,
        condition,
        question.trapWriteSql,
      ).refused;
      if (question.control === true) {
        goldEntities = readGoldEntities();
      }
    }

    const querySystem =
      condition === 'remembero' ? datalogSystemPrompt() : sqlSystemPrompt();
    const messages: ChatMessage[] = [
      { role: 'system', content: querySystem },
      { role: 'user', content: question.question },
    ];

    let rows: Array<Record<string, unknown>> | undefined;
    let query = '';
    let toolErrors = 0;
    for (
      let attempt = 0;
      attempt < MAX_ATTEMPTS && rows === undefined;
      attempt += 1
    ) {
      const rawQuery = await chat(model, messages, seed);
      query = stripFences(rawQuery);
      try {
        rows = executeQuery(db, condition, rawQuery);
      } catch (error) {
        toolErrors += 1;
        messages.push({ role: 'assistant', content: rawQuery });
        messages.push({
          role: 'user',
          content: `That query failed with: ${error instanceof Error ? error.message : String(error)}. Reply with ONLY a corrected query.`,
        });
      }
    }

    if (rows === undefined) {
      return {
        id: question.id,
        category: question.category,
        condition,
        seed,
        passed: false,
        toolErrors,
        ...(gateRefusedTrap === undefined ? {} : { gateRefusedTrap }),
        query,
        answer: '(no runnable query)',
        missing: question.expect,
        forbidden: [],
        extraEntities: [],
      };
    }

    const shown = rows.slice(0, MAX_RESULT_ROWS);
    const answer = await chat(
      model,
      [
        { role: 'system', content: answerSystemPrompt() },
        {
          role: 'user',
          content: `QUESTION: ${question.question}\nTOOL_RESULT (${rows.length} rows):\n${JSON.stringify(shown, null, 1)}`,
        },
      ],
      seed,
    );
    const grade = gradeAnswerV2(question, answer, goldEntities);
    // A control write violates no rule: a gate refusal is itself the failure.
    const passed =
      question.control === true && gateRefusedTrap === true
        ? false
        : grade.passed;
    return {
      id: question.id,
      category: question.category,
      condition,
      seed,
      passed,
      toolErrors,
      ...(gateRefusedTrap === undefined ? {} : { gateRefusedTrap }),
      query,
      answer: answer.trim(),
      missing: grade.missing,
      forbidden: grade.forbidden,
      extraEntities: grade.extraEntities,
    };
  } finally {
    db.close();
  }
}

interface ConditionSeedStats {
  seed: number;
  passed: number;
  total: number;
  toolErrors: number;
}

interface ConditionStats {
  perSeed: ConditionSeedStats[];
  /** Mean questions passed per seed. */
  mean: number;
  /** max minus min of per-seed passes — run-variance visibility (ADR 0003). */
  spread: number;
  toolErrors: number;
}

interface GateStats {
  trapOutcomes: number;
  trapRefusals: number;
  /** Gate refused the trap AND the answer still passed — the v2 headline. */
  gateProtectedPasses: number;
  /** Control writes refused — must stay 0; the gate may not reject benign writes. */
  controlRefusals: number;
}

function conditionStats(
  outcomes: QuestionOutcome[],
  condition: AgentBoundaryCondition,
  seeds: readonly number[],
): ConditionStats {
  const rows = outcomes.filter((row) => row.condition === condition);
  const perSeed = seeds.map((seed) => {
    const seedRows = rows.filter((row) => row.seed === seed);
    return {
      seed,
      passed: seedRows.filter((row) => row.passed).length,
      total: seedRows.length,
      toolErrors: seedRows.reduce((sum, row) => sum + row.toolErrors, 0),
    };
  });
  const passes = perSeed.map((row) => row.passed);
  return {
    perSeed,
    mean:
      passes.reduce((sum, value) => sum + value, 0) /
      Math.max(passes.length, 1),
    spread: passes.length > 0 ? Math.max(...passes) - Math.min(...passes) : 0,
    toolErrors: rows.reduce((sum, row) => sum + row.toolErrors, 0),
  };
}

function gateStats(
  outcomes: QuestionOutcome[],
  condition: AgentBoundaryCondition,
): GateStats {
  const rows = outcomes.filter(
    (row) => row.condition === condition && row.gateRefusedTrap !== undefined,
  );
  const controls = rows.filter((row) => row.id.startsWith('c'));
  const traps = rows.filter((row) => !row.id.startsWith('c'));
  return {
    trapOutcomes: traps.length,
    trapRefusals: traps.filter((row) => row.gateRefusedTrap === true).length,
    gateProtectedPasses: traps.filter(
      (row) => row.gateRefusedTrap === true && row.passed,
    ).length,
    controlRefusals: controls.filter((row) => row.gateRefusedTrap === true)
      .length,
  };
}

function summarize(outcomes: QuestionOutcome[], seeds: readonly number[]) {
  const categories = [...new Set(outcomes.map((outcome) => outcome.category))];
  const forConditions = (
    subset: QuestionOutcome[],
  ): Record<AgentBoundaryCondition, ConditionStats> =>
    Object.fromEntries(
      AGENT_BOUNDARY_CONDITIONS.map((condition) => [
        condition,
        conditionStats(subset, condition, seeds),
      ]),
    ) as Record<AgentBoundaryCondition, ConditionStats>;
  const byCategory = categories.map((category) => ({
    category,
    conditions: forConditions(
      outcomes.filter((row) => row.category === category),
    ),
  }));
  const totals = forConditions(outcomes);
  const gate = Object.fromEntries(
    AGENT_BOUNDARY_CONDITIONS.map((condition) => [
      condition,
      gateStats(outcomes, condition),
    ]),
  ) as Record<AgentBoundaryCondition, GateStats>;
  return { byCategory, totals, gate };
}

async function main(): Promise<void> {
  const modelFlag = process.argv.indexOf('--model');
  const model = modelFlag >= 0 ? process.argv[modelFlag + 1] : 'llama3.2:3b';
  const seedsFlag = process.argv.indexOf('--seeds');
  const seeds =
    seedsFlag >= 0
      ? process.argv[seedsFlag + 1]
          .split(',')
          .map((value) => Number(value.trim()))
      : [7, 42, 123];
  console.log(
    `agent-boundary benchmark v2 · model ${model} · ${AGENT_BOUNDARY_QUESTIONS.length} questions × ${AGENT_BOUNDARY_CONDITIONS.length} conditions × ${seeds.length} seeds`,
  );

  const outcomes: QuestionOutcome[] = [];
  for (const seed of seeds) {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      for (const condition of AGENT_BOUNDARY_CONDITIONS) {
        const outcome = await runQuestion(model, condition, question, seed);
        outcomes.push(outcome);
        console.log(
          `${outcome.passed ? 'PASS' : 'FAIL'} seed ${seed} ${question.id} ${condition}` +
            (outcome.toolErrors > 0
              ? ` (tool errors: ${outcome.toolErrors})`
              : '') +
            (outcome.gateRefusedTrap === true ? ' (gate refused trap)' : '') +
            (outcome.extraEntities.length > 0
              ? ` (extra entities: ${outcome.extraEntities.join(', ')})`
              : ''),
        );
      }
    }
  }

  const summary = summarize(outcomes, seeds);

  console.log('\nwrite-trap integrity (headline: gate-protected passes)');
  console.log(
    'condition     trap refusals   gate-protected   control refusals',
  );
  for (const condition of AGENT_BOUNDARY_CONDITIONS) {
    const stats = summary.gate[condition];
    console.log(
      `${condition.padEnd(14)}${`${stats.trapRefusals}/${stats.trapOutcomes}`.padStart(9)}${String(stats.gateProtectedPasses).padStart(16)}${String(stats.controlRefusals).padStart(19)}`,
    );
  }

  console.log(
    '\ncategory                mean passed per seed (per-seed counts)',
  );
  for (const row of summary.byCategory) {
    const cells = AGENT_BOUNDARY_CONDITIONS.map((condition) => {
      const stats = row.conditions[condition];
      const detail = stats.perSeed.map((seedRow) => seedRow.passed).join(',');
      const total = stats.perSeed[0]?.total ?? 0;
      return `${condition} ${stats.mean.toFixed(1)}/${total} (${detail})`;
    }).join('  ');
    console.log(`${row.category.padEnd(24)}${cells}`);
  }
  const totalCells = AGENT_BOUNDARY_CONDITIONS.map((condition) => {
    const stats = summary.totals[condition];
    const detail = stats.perSeed.map((seedRow) => seedRow.passed).join(',');
    const total = stats.perSeed[0]?.total ?? 0;
    return `${condition} ${stats.mean.toFixed(1)}/${total} (${detail})`;
  }).join('  ');
  console.log(`${'TOTAL'.padEnd(24)}${totalCells}`);
  const errorCells = AGENT_BOUNDARY_CONDITIONS.map(
    (condition) => `${condition} ${summary.totals[condition].toolErrors}`,
  ).join('  ');
  console.log(`${'tool errors'.padEnd(24)}${errorCells}`);

  const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
  const resultPath = join(
    projectRoot,
    'docs',
    'research',
    'results',
    `agent-boundary-v2-${model.replaceAll(/[^a-z0-9.]+/gi, '-')}-summary.json`,
  );
  mkdirSync(dirname(resultPath), { recursive: true });
  writeFileSync(
    resultPath,
    `${JSON.stringify(
      {
        benchmark: 'agent-boundary-v2',
        model,
        generatedAt: new Date().toISOString(),
        settings: { temperature: 0, seeds, attempts: MAX_ATTEMPTS },
        summary,
        outcomes,
      },
      null,
      2,
    )}\n`,
  );
  console.log(`\nresults written to ${resultPath}`);
}

await main();
