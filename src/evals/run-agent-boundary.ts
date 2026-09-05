/**
 * Runs the agent-boundary benchmark against a local Ollama model.
 *
 *   node dist/evals/run-agent-boundary.js --model llama3.2:3b
 *
 * Same seeded SQLite database, same model; the model authors every query
 * itself. The sql condition executes model-written read-only SQL with no
 * integrity gate; sql-gated executes the same SQL but routes writes through
 * the Remembero integrity gate; the remembero condition executes
 * model-written Datalog via the same bridge the MCP server and browser labs
 * use. Write-trap questions push a corrupting write through each condition's
 * write path first: raw SQL applies it, the gated conditions check their
 * integrity rules and refuse.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AGENT_BOUNDARY_CONDITIONS,
  AGENT_BOUNDARY_QUESTIONS,
  AGENT_BOUNDARY_SEED_SQL,
  WRITE_GATE_RULES,
  answerSystemPrompt,
  assertReadOnlySql,
  datalogSystemPrompt,
  gradeAnswer,
  sqlSystemPrompt,
  stripFences,
  type AgentBoundaryCondition,
  type AgentBoundaryQuestion,
} from './agent-boundary.js';
import { openRememberoDatabase, type RememberoDatabase } from '../sqlite/extension.js';

const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const MAX_RESULT_ROWS = 30;
const MAX_ATTEMPTS = 2;

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

async function chat(model: string, messages: ChatMessage[]): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: { temperature: 0, seed: 7, num_ctx: 4096, num_predict: 400 },
    }),
  });
  if (!response.ok) {
    throw new Error(`ollama returned ${response.status}: ${await response.text()}`);
  }
  const payload = (await response.json()) as { message?: { content?: string } };
  const content = payload.message?.content;
  if (typeof content !== 'string') throw new Error('ollama returned no message content');
  return content;
}

interface QuestionOutcome {
  id: string;
  category: string;
  condition: AgentBoundaryCondition;
  passed: boolean;
  toolErrors: number;
  gateRefusedTrap?: boolean;
  query: string;
  answer: string;
  missing: string[];
  forbidden: string[];
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
  // sql-gated and remembero share the gated write path (ADR 0002).
  db.exec('SAVEPOINT gate');
  db.exec(trapWriteSql);
  const violations = WRITE_GATE_RULES.flatMap((rule) => db.datalogQuery(rule.program));
  if (violations.length > 0) {
    db.exec('ROLLBACK TO gate');
    db.exec('RELEASE gate');
    return { refused: true };
  }
  db.exec('RELEASE gate');
  return { refused: false };
}

async function runQuestion(
  model: string,
  condition: AgentBoundaryCondition,
  question: AgentBoundaryQuestion,
): Promise<QuestionOutcome> {
  const db = await openRememberoDatabase(':memory:');
  try {
    db.exec(AGENT_BOUNDARY_SEED_SQL);
    let gateRefusedTrap: boolean | undefined;
    if (question.trapWriteSql !== undefined) {
      gateRefusedTrap = applyTrapWrite(db, condition, question.trapWriteSql).refused;
    }

    const querySystem = condition === 'remembero' ? datalogSystemPrompt() : sqlSystemPrompt();
    const messages: ChatMessage[] = [
      { role: 'system', content: querySystem },
      { role: 'user', content: question.question },
    ];

    let rows: Array<Record<string, unknown>> | undefined;
    let query = '';
    let toolErrors = 0;
    for (let attempt = 0; attempt < MAX_ATTEMPTS && rows === undefined; attempt += 1) {
      const rawQuery = await chat(model, messages);
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
        passed: false,
        toolErrors,
        ...(gateRefusedTrap === undefined ? {} : { gateRefusedTrap }),
        query,
        answer: '(no runnable query)',
        missing: question.expect,
        forbidden: [],
      };
    }

    const shown = rows.slice(0, MAX_RESULT_ROWS);
    const answer = await chat(model, [
      { role: 'system', content: answerSystemPrompt() },
      {
        role: 'user',
        content: `QUESTION: ${question.question}\nTOOL_RESULT (${rows.length} rows):\n${JSON.stringify(shown, null, 1)}`,
      },
    ]);
    const grade = gradeAnswer(question, answer);
    // A control write violates no rule: a gate refusal is itself the failure.
    const passed = question.control === true && gateRefusedTrap === true ? false : grade.passed;
    return {
      id: question.id,
      category: question.category,
      condition,
      passed,
      toolErrors,
      ...(gateRefusedTrap === undefined ? {} : { gateRefusedTrap }),
      query,
      answer: answer.trim(),
      missing: grade.missing,
      forbidden: grade.forbidden,
    };
  } finally {
    db.close();
  }
}

function summarize(outcomes: QuestionOutcome[]) {
  const categories = [...new Set(outcomes.map((outcome) => outcome.category))];
  const totalsFor = (
    subset: QuestionOutcome[],
    condition: AgentBoundaryCondition,
  ) => {
    const rows = subset.filter((row) => row.condition === condition);
    return {
      passed: rows.filter((row) => row.passed).length,
      total: rows.length,
      toolErrors: rows.reduce((sum, row) => sum + row.toolErrors, 0),
    };
  };
  const byCategory = categories.map((category) => {
    const rows = outcomes.filter((outcome) => outcome.category === category);
    return {
      category,
      conditions: Object.fromEntries(
        AGENT_BOUNDARY_CONDITIONS.map((condition) => [
          condition,
          totalsFor(rows, condition),
        ]),
      ) as Record<
        AgentBoundaryCondition,
        { passed: number; total: number; toolErrors: number }
      >,
    };
  });
  const totals = Object.fromEntries(
    AGENT_BOUNDARY_CONDITIONS.map((condition) => [
      condition,
      totalsFor(outcomes, condition),
    ]),
  ) as Record<
    AgentBoundaryCondition,
    { passed: number; total: number; toolErrors: number }
  >;
  return { byCategory, totals };
}

async function main(): Promise<void> {
  const modelFlag = process.argv.indexOf('--model');
  const model = modelFlag >= 0 ? process.argv[modelFlag + 1] : 'llama3.2:3b';
  console.log(`agent-boundary benchmark · model ${model} · ${AGENT_BOUNDARY_QUESTIONS.length} questions × 2 conditions`);

  const outcomes: QuestionOutcome[] = [];
  for (const question of AGENT_BOUNDARY_QUESTIONS) {
    for (const condition of AGENT_BOUNDARY_CONDITIONS) {
      const outcome = await runQuestion(model, condition, question);
      outcomes.push(outcome);
      console.log(
        `${outcome.passed ? 'PASS' : 'FAIL'} ${question.id} ${condition}` +
          (outcome.toolErrors > 0 ? ` (tool errors: ${outcome.toolErrors})` : '') +
          (outcome.gateRefusedTrap === true ? ' (gate refused trap)' : ''),
      );
    }
  }

  const summary = summarize(outcomes);
  const columns = AGENT_BOUNDARY_CONDITIONS.map((condition) => condition.padStart(11)).join('');
  console.log(`\ncategory${' '.repeat(17)}${columns}`);
  for (const row of summary.byCategory) {
    const cells = AGENT_BOUNDARY_CONDITIONS.map((condition) => {
      const stats = row.conditions[condition];
      return `${String(stats.passed).padStart(2)}/${stats.total}`.padStart(11);
    }).join('');
    console.log(`${row.category.padEnd(24)}${cells}`);
  }
  const totalCells = AGENT_BOUNDARY_CONDITIONS.map((condition) => {
    const stats = summary.totals[condition];
    return `${String(stats.passed).padStart(2)}/${stats.total}`.padStart(11);
  }).join('');
  console.log(`${'TOTAL'.padEnd(24)}${totalCells}`);
  const errorCells = AGENT_BOUNDARY_CONDITIONS.map((condition) =>
    String(summary.totals[condition].toolErrors).padStart(11),
  ).join('');
  console.log(`${'tool errors'.padEnd(24)}${errorCells}`);

  const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
  const resultPath = join(
    projectRoot,
    'docs',
    'research',
    'results',
    `agent-boundary-v1-${model.replaceAll(/[^a-z0-9.]+/gi, '-')}-summary.json`,
  );
  mkdirSync(dirname(resultPath), { recursive: true });
  writeFileSync(
    resultPath,
    `${JSON.stringify(
      {
        benchmark: 'agent-boundary-v1',
        model,
        generatedAt: new Date().toISOString(),
        settings: { temperature: 0, seed: 7, attempts: MAX_ATTEMPTS },
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
