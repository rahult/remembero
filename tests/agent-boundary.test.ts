import { describe, expect, it, beforeEach, afterEach } from 'vitest';

import {
  AGENT_BOUNDARY_CONDITIONS,
  AGENT_BOUNDARY_QUESTIONS,
  AGENT_BOUNDARY_SEED_SQL,
  DATALOG_CLOSURE_FEW_SHOT,
  DATALOG_FEW_SHOT,
  WRITE_GATE_RULES,
  assertReadOnlySql,
  datalogClosureSystemPrompt,
  datalogSystemPrompt,
  entitiesFromRows,
  gradeAnswer,
  gradeAnswerV2,
  gradeQueryRows,
  normalizeAnswer,
  type AgentBoundaryQuestion,
  seedEntityLexicon,
} from '../src/evals/agent-boundary.js';
import {
  applyGatedWrite,
  collectFactClauses,
  gateConstraintPrograms,
} from '../src/evals/agent-boundary-gate.js';
import {
  chatRequest,
  parseChatResponse,
  resolveAnswerLeg,
} from '../src/evals/agent-boundary-chat.js';
import { checkIntegrity } from '../src/knowledge/integrity.js';
import { parseProgram, parseQueryProgram } from '../src/engine/index.js';
import {
  openRememberoDatabase,
  type RememberoDatabase,
} from '../src/sqlite/extension.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

function valueSet(rows: Array<Record<string, unknown>>): Set<string> {
  const values = new Set<string>();
  for (const row of rows) {
    for (const value of Object.values(row)) {
      values.add(normalizeAnswer(String(value)));
    }
  }
  return values;
}

describe.skipIf(nodeMajor < 22)('agent-boundary benchmark ground truth', () => {
  let db: RememberoDatabase;

  beforeEach(async () => {
    db = await openRememberoDatabase(':memory:');
    db.exec(AGENT_BOUNDARY_SEED_SQL);
  });

  afterEach(() => {
    db.close();
  });

  it('every question’s gold SQL result covers its expected terms on the clean database', () => {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.control === true) continue; // control gold holds POST-write (ADR 0003)
      const rows = db.prepare(question.goldSql).all() as Array<
        Record<string, unknown>
      >;
      const values = valueSet(rows);
      for (const term of question.expect) {
        if (term === 'yes' || term === 'no') continue; // phrasing, not a cell value
        expect(
          [...values].some((value) => value.includes(normalizeAnswer(term))),
          `${question.id}: gold SQL should produce '${term}', got ${JSON.stringify([...values])}`,
        ).toBe(true);
      }
      for (const term of question.forbid ?? []) {
        expect(
          values.has(normalizeAnswer(term)),
          `${question.id}: gold SQL must not produce forbidden '${term}'`,
        ).toBe(false);
      }
    }
  });

  it('every question’s gold Datalog agrees with the expected terms on the clean database', () => {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.control === true) continue; // control gold holds POST-write (ADR 0003)
      const rows = db.datalogQuery(question.goldDatalog);
      const values = valueSet(rows as Array<Record<string, unknown>>);
      for (const term of question.expect) {
        if (term === 'yes' || term === 'no') continue;
        expect(
          [...values].some((value) => value.includes(normalizeAnswer(term))),
          `${question.id}: gold Datalog should produce '${term}', got ${JSON.stringify([...values])}`,
        ).toBe(true);
      }
    }
  });

  it('the write gate refuses every trap write while raw SQL applies it silently', () => {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.trapWriteSql === undefined) continue;
      if (question.control === true) continue; // controls are covered below
      // Raw SQL path: the write goes through without complaint.
      db.exec('SAVEPOINT raw_path');
      db.exec(question.trapWriteSql);
      const corrupted = db.prepare(question.goldSql).all() as Array<
        Record<string, unknown>
      >;
      const corruptedValues = valueSet(corrupted);
      const lostExpected = question.expect
        .filter((term) => term !== 'yes' && term !== 'no')
        .some(
          (term) =>
            ![...corruptedValues].some((v) =>
              v.includes(normalizeAnswer(term)),
            ),
        );
      const gainedForbidden = (question.forbid ?? []).some((term) =>
        corruptedValues.has(normalizeAnswer(term)),
      );
      expect(
        lostExpected || gainedForbidden,
        `${question.id}: the trap write should actually corrupt the SQL answer`,
      ).toBe(true);
      db.exec('ROLLBACK TO raw_path');
      db.exec('RELEASE raw_path');

      // Remembero gate: the product enforcement path refuses the write.
      const gated = applyGatedWrite(db, question.trapWriteSql);
      expect(
        gated.refused,
        `${question.id}: the write gate should refuse`,
      ).toBe(true);

      // After the refusal the truth is intact.
      const restored = db.prepare(question.goldSql).all() as Array<
        Record<string, unknown>
      >;
      const restoredValues = valueSet(restored);
      for (const term of question.expect) {
        if (term === 'yes' || term === 'no') continue;
        expect(
          [...restoredValues].some((v) => v.includes(normalizeAnswer(term))),
        ).toBe(true);
      }
    }
  });

  it('the benign-write control is refused by nobody and its gold holds post-write', () => {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.control !== true || question.trapWriteSql === undefined)
        continue;
      // Gated path: the product enforcement path must not refuse a benign
      // write; it commits, and gold matches POST-write state.
      const gated = applyGatedWrite(db, question.trapWriteSql);
      expect(
        gated.refused,
        `${question.id}: control write violates no rule; the gate must not refuse`,
      ).toBe(false);
      const rows = db.prepare(question.goldSql).all() as Array<
        Record<string, unknown>
      >;
      const values = valueSet(rows);
      for (const term of question.expect) {
        expect(
          [...values].some((v) => v.includes(normalizeAnswer(term))),
          `${question.id}: post-write gold SQL should produce '${term}'`,
        ).toBe(true);
      }
      const datalogValues = valueSet(
        db.datalogQuery(question.goldDatalog) as Array<Record<string, unknown>>,
      );
      for (const term of question.expect) {
        expect(
          [...datalogValues].some((v) => v.includes(normalizeAnswer(term))),
          `${question.id}: post-write gold Datalog should produce '${term}'`,
        ).toBe(true);
      }
    }
  });

  it('the clean database passes every write-gate rule', () => {
    for (const rule of WRITE_GATE_RULES) {
      expect(db.datalogQuery(rule.program)).toEqual([]);
    }
  });

  it('the clean database passes every derived constraint via the product checker', () => {
    const clauses = [
      ...collectFactClauses(db),
      ...gateConstraintPrograms().flatMap((program) => parseProgram(program)),
    ];
    expect(checkIntegrity(clauses).status).not.toBe('violations');
  });

  it('the product enforcement path decides identically to the frozen rules', () => {
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.trapWriteSql === undefined) continue;
      db.exec('SAVEPOINT equivalence');
      // Legacy decision: frozen violation-headed rules through the query bridge.
      db.exec('SAVEPOINT legacy');
      db.exec(question.trapWriteSql);
      const legacyViolations = WRITE_GATE_RULES.flatMap((rule) =>
        db.datalogQuery(rule.program),
      );
      db.exec('ROLLBACK TO legacy');
      db.exec('RELEASE legacy');
      // Product decision: strict enforcement over fact clauses.
      const gated = applyGatedWrite(db, question.trapWriteSql);
      expect(
        gated.refused,
        `${question.id}: product gate and frozen rules must agree`,
      ).toBe(legacyViolations.length > 0);
      db.exec('ROLLBACK TO equivalence');
      db.exec('RELEASE equivalence');
    }
  });

  it('the sql-gated arm shares the remembero gated write path', () => {
    expect(AGENT_BOUNDARY_CONDITIONS).toEqual([
      'sql',
      'sql-gated',
      'remembero',
      'remembero-closure',
    ]);
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.trapWriteSql === undefined || question.control === true)
        continue;
      // Same product gate the runner applies for sql-gated.
      const gated = applyGatedWrite(db, question.trapWriteSql);
      expect(
        gated.refused,
        `${question.id}: sql-gated should refuse via the shared gate`,
      ).toBe(true);
      const restored = db.prepare(question.goldSql).all() as Array<
        Record<string, unknown>
      >;
      const restoredValues = valueSet(restored);
      for (const term of question.expect) {
        if (term === 'yes' || term === 'no') continue;
        expect(
          [...restoredValues].some((v) => v.includes(normalizeAnswer(term))),
          `${question.id}: truth intact after sql-gated refusal`,
        ).toBe(true);
      }
    }
  });
});

describe('agent-boundary grading and safety helpers', () => {
  it('grades required and forbidden terms after normalization', () => {
    const question = AGENT_BOUNDARY_QUESTIONS.find(
      (entry) => entry.id === 'a2',
    )!;
    expect(
      gradeAnswer(question, 'Beacon is blocked with no review slot.').passed,
    ).toBe(true);
    expect(gradeAnswer(question, 'Atlas has no slot.').passed).toBe(false);
    const trap = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 't1')!;
    expect(gradeAnswer(trap, 'Atlas is currently active.').passed).toBe(false);
    expect(gradeAnswer(trap, 'Atlas remains blocked.').passed).toBe(true);
  });

  it('rejects write statements in the read-only SQL tool', () => {
    expect(() => assertReadOnlySql('SELECT * FROM status')).not.toThrow();
    expect(() =>
      assertReadOnlySql('WITH x AS (SELECT 1) SELECT * FROM x'),
    ).not.toThrow();
    expect(() =>
      assertReadOnlySql("UPDATE status SET state = 'active'"),
    ).toThrow();
    expect(() => assertReadOnlySql('DROP TABLE status')).toThrow();
  });

  it('keeps a balanced category mix including ground SQL should tie on', () => {
    const byCategory = new Map<string, number>();
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      byCategory.set(
        question.category,
        (byCategory.get(question.category) ?? 0) + 1,
      );
    }
    expect(byCategory.get('direct')).toBe(6);
    expect(byCategory.get('join')).toBe(6);
    expect(byCategory.get('multihop')).toBe(6);
    expect(byCategory.get('absence')).toBe(6);
    // 6 genuine traps + 1 benign-write control (ADR 0003)
    expect(byCategory.get('write-trap')).toBe(7);
  });
});

describe('agent-boundary v2 answer-set grading', () => {
  const lexicon = seedEntityLexicon();

  it('builds the entity lexicon from seed SQL cell values', () => {
    for (const entity of [
      'maya',
      'atlas',
      'legal signoff',
      'procurement freeze',
      'morning',
    ]) {
      expect(lexicon.has(entity), `lexicon should contain '${entity}'`).toBe(
        true,
      );
    }
  });

  it('collects gold entities from gold-query rows', () => {
    const entities = entitiesFromRows([
      { person: 'priya' },
      { person: 'tom' },
      { person: 'maya' },
    ]);
    expect(entities).toEqual(new Set(['priya', 'tom', 'maya']));
  });

  it('fails wrong-superset answers that v1 substring grading passed', () => {
    const j5 = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 'j5')!;
    const gold = new Set(['priya', 'tom', 'maya']);
    const v1RecordedAnswer =
      'The concrete values are: Maya, Liam, Priya, Tom. Since there is only one person named Maya, the answer is Maya.';
    expect(gradeAnswer(j5, v1RecordedAnswer).passed).toBe(true); // v1 behavior frozen
    const v2 = gradeAnswerV2(j5, v1RecordedAnswer, gold, lexicon);
    expect(v2.passed).toBe(false);
    expect(v2.extraEntities).toEqual(['liam']);
  });

  it('passes exact-set answers and ignores echoes of the question text', () => {
    const j5 = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 'j5')!;
    const gold = new Set(['priya', 'tom', 'maya']);
    const answer =
      'Priya, Tom, and Maya work on a project blocked by legal signoff.';
    expect(gradeAnswerV2(j5, answer, gold, lexicon).passed).toBe(true);
  });

  it('does not treat yes/no phrasing as entities', () => {
    const m5 = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 'm5')!;
    const gold = new Set(['procurement freeze']);
    expect(
      gradeAnswerV2(
        m5,
        'Yes, atlas ultimately waits on procurement freeze.',
        gold,
        lexicon,
      ).passed,
    ).toBe(true);
  });
});

describe('agent-boundary v2 Datalog prompt', () => {
  let db: RememberoDatabase;

  beforeEach(async () => {
    db = await openRememberoDatabase(':memory:');
    db.exec(AGENT_BOUNDARY_SEED_SQL);
  });

  afterEach(() => {
    db.close();
  });

  it('every few-shot example runs against the seeded database and answers its question', () => {
    expect(DATALOG_FEW_SHOT.length).toBe(8);
    for (const example of DATALOG_FEW_SHOT) {
      expect(
        () => db.datalogQuery(example.program),
        `few-shot '${example.q}' must parse and run on the bridge`,
      ).not.toThrow();
    }
    // The chain examples must answer the question asked, not a helper rule:
    // the sink rule is the query target regardless of rule order.
    const root = DATALOG_FEW_SHOT.find((e) =>
      e.q.startsWith('What is the final upstream'),
    );
    expect(db.datalogQuery(root!.program)).toEqual([
      { R: 'procurement_freeze' },
    ]);
    for (const example of DATALOG_CLOSURE_FEW_SHOT) {
      expect(() => db.datalogQuery(example.program), example.q).not.toThrow();
    }
    const yesNo = DATALOG_CLOSURE_FEW_SHOT.find((e) =>
      e.q.startsWith('Does priya'),
    );
    expect(db.datalogQuery(yesNo!.program)).toEqual([{ yes: 'true' }]);
  });

  it('the system prompt embeds the cheatsheet and all examples', () => {
    const prompt = datalogSystemPrompt();
    expect(prompt).toContain('Dialect cheatsheet');
    for (const example of DATALOG_FEW_SHOT) {
      expect(prompt).toContain(example.program);
    }
  });

  it('closure few-shot programs run on the seeded database without any recursive rule', () => {
    expect(DATALOG_CLOSURE_FEW_SHOT.length).toBe(DATALOG_FEW_SHOT.length);
    for (const example of DATALOG_CLOSURE_FEW_SHOT) {
      for (const clause of parseQueryProgram(example.program).clauses) {
        const body = JSON.stringify(clause.body);
        expect(body, example.program).not.toContain(
          `"predicate":"${clause.head.predicate}"`,
        );
      }
      expect(
        () => db.datalogQuery(example.program),
        `closure few-shot '${example.q}' must run on the bridge`,
      ).not.toThrow();
    }
    const prompt = datalogClosureSystemPrompt();
    expect(prompt).toContain('reports_to_plus');
    expect(prompt).not.toContain('Recursion is allowed');
    for (const example of DATALOG_CLOSURE_FEW_SHOT) {
      expect(prompt).toContain(example.program);
    }
  });
});

describe('agent-boundary runner: chat backends', () => {
  const messages = [
    { role: 'system' as const, content: 's' },
    { role: 'user' as const, content: 'u' },
  ];

  it('builds an Ollama /api/chat request by default and reads message.content', () => {
    const request = chatRequest(
      'ollama',
      'http://127.0.0.1:11434',
      'llama3.2:3b',
      messages,
      7,
    );
    expect(request.url).toBe('http://127.0.0.1:11434/api/chat');
    expect(request.body.options).toMatchObject({ temperature: 0, seed: 7 });
    expect(
      parseChatResponse('ollama', { message: { content: 'q(X) :- a(X).' } }),
    ).toBe('q(X) :- a(X).');
  });

  it('builds an OpenAI-compatible /v1/chat/completions request for Tinker proxies', () => {
    const request = chatRequest(
      'openai',
      'http://127.0.0.1:7462/',
      'tinker://run/sampler_weights/final',
      messages,
      7,
    );
    expect(request.url).toBe('http://127.0.0.1:7462/v1/chat/completions');
    expect(request.body).toMatchObject({
      temperature: 0,
      seed: 7,
      max_tokens: 400,
      messages,
    });
    expect(
      parseChatResponse('openai', {
        choices: [{ message: { content: 'q(X) :- a(X).' } }],
      }),
    ).toBe('q(X) :- a(X).');
    expect(() => parseChatResponse('openai', { choices: [] })).toThrow(
      /no message content/,
    );
    // a reasoning model that spent its budget thinking returns a choice with
    // null content: a failed attempt for the harness to retry, not a crash
    expect(
      parseChatResponse('openai', {
        choices: [{ message: { content: null, reasoning: '...' } }],
      }),
    ).toBe('');
  });
});

describe('agent-boundary runner: separate answer model', () => {
  it('resolves an answer-leg override from --answer-model, defaulting to the query model', () => {
    const argv = [
      'node',
      'run.js',
      '--model',
      'tinker://x',
      '--answer-model',
      'llama3.2:3b',
      '--answer-chat-api',
      'ollama',
    ];
    expect(resolveAnswerLeg(argv, {}, 'tinker://x', 'openai')).toEqual({
      model: 'llama3.2:3b',
      backend: 'ollama',
      url: 'http://127.0.0.1:11434',
    });
    expect(
      resolveAnswerLeg(['node', 'run.js'], {}, 'tinker://x', 'openai'),
    ).toEqual({
      model: 'tinker://x',
      backend: 'openai',
      url: undefined,
    });
  });
});

describe('agent-boundary: query-leg grading', () => {
  const q = (over: Partial<AgentBoundaryQuestion>): AgentBoundaryQuestion => ({
    id: 'x',
    category: 'multihop',
    question: 'q',
    expect: [],
    goldSql: 'SELECT 1',
    goldDatalog: 'q(X) :- a(X).',
    ...over,
  });

  it('fails a superset that returns seed entities outside the gold set', () => {
    // "Who is maya's direct manager?" answered with the whole chain
    const gold = new Set(['liam']);
    expect(
      gradeQueryRows(
        q({ expect: ['liam'] }),
        [{ M: 'liam' }, { M: 'ava' }, { M: 'dana' }],
        gold,
      ).passed,
    ).toBe(false);
    expect(
      gradeQueryRows(q({ expect: ['liam'] }), [{ M: 'liam' }], gold).passed,
    ).toBe(true);
  });

  it('does not count a single negative value as yes', () => {
    const gold = new Set(['procurement freeze']);
    expect(
      gradeQueryRows(q({ expect: ['yes'] }), [{ answer: 'No' }], gold).passed,
    ).toBe(false);
    expect(
      gradeQueryRows(q({ expect: ['yes'] }), [{ n: 0 }], gold).passed,
    ).toBe(false);
    expect(
      gradeQueryRows(q({ expect: ['yes'] }), [{ answer: 'Yes' }], gold).passed,
    ).toBe(true);
  });

  it('passes when the rows cover every expected entity and no forbidden one', () => {
    const gold = new Set(['liam', 'ava', 'dana']);
    expect(
      gradeQueryRows(
        q({ expect: ['liam', 'ava', 'dana'] }),
        [{ M: 'liam' }, { M: 'ava' }, { M: 'dana' }],
        gold,
      ).passed,
    ).toBe(true);
    expect(
      gradeQueryRows(q({ expect: ['liam', 'ava'] }), [{ M: 'liam' }]).passed,
    ).toBe(false);
    expect(
      gradeQueryRows(q({ expect: ['liam'], forbid: ['dana'] }), [
        { M: 'liam' },
        { M: 'dana' },
      ]).passed,
    ).toBe(false);
  });

  it('treats yes/no expectations as row presence and matches underscored entities', () => {
    const gold = new Set(['procurement freeze']);
    expect(
      gradeQueryRows(
        q({ expect: ['yes'] }),
        [{ X: 'procurement_freeze' }],
        gold,
      ).passed,
    ).toBe(true);
    expect(gradeQueryRows(q({ expect: ['yes'] }), [], gold).passed).toBe(false);
    expect(gradeQueryRows(q({ expect: ['no'] }), [], gold).passed).toBe(true);
    expect(
      gradeQueryRows(
        q({ expect: ['procurement freeze'] }),
        [{ R: 'procurement_freeze' }],
        gold,
      ).passed,
    ).toBe(true);
  });
});

describe('agent-boundary runner: hosted OpenAI-compatible providers', () => {
  it('sends a bearer token and an overridable completion budget', () => {
    const request = chatRequest(
      'openai',
      'https://openrouter.ai/api/v1',
      'z-ai/glm-5.3',
      [{ role: 'user', content: 'u' }],
      7,
      { apiKey: 'sk-test', maxTokens: 4000 },
    );
    expect(request.url).toBe('https://openrouter.ai/api/v1/chat/completions');
    expect(request.headers).toMatchObject({ authorization: 'Bearer sk-test' });
    expect(request.body).toMatchObject({ max_tokens: 4000 });
    const bare = chatRequest('ollama', 'http://127.0.0.1:11434', 'm', [], 7);
    expect(bare.headers.authorization).toBeUndefined();
  });
});
