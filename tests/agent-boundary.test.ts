import { describe, expect, it, beforeEach, afterEach } from 'vitest';

import {
  AGENT_BOUNDARY_CONDITIONS,
  AGENT_BOUNDARY_QUESTIONS,
  AGENT_BOUNDARY_SEED_SQL,
  WRITE_GATE_RULES,
  assertReadOnlySql,
  entitiesFromRows,
  gradeAnswer,
  gradeAnswerV2,
  normalizeAnswer,
  seedEntityLexicon,
} from '../src/evals/agent-boundary.js';
import { openRememberoDatabase, type RememberoDatabase } from '../src/sqlite/extension.js';

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
      const rows = db.prepare(question.goldSql).all() as Array<Record<string, unknown>>;
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
      // Raw SQL path: the write goes through without complaint.
      db.exec('SAVEPOINT raw_path');
      db.exec(question.trapWriteSql);
      const corrupted = db.prepare(question.goldSql).all() as Array<Record<string, unknown>>;
      const corruptedValues = valueSet(corrupted);
      const lostExpected = question.expect
        .filter((term) => term !== 'yes' && term !== 'no')
        .some((term) => ![...corruptedValues].some((v) => v.includes(normalizeAnswer(term))));
      const gainedForbidden = (question.forbid ?? []).some((term) =>
        corruptedValues.has(normalizeAnswer(term)),
      );
      expect(
        lostExpected || gainedForbidden,
        `${question.id}: the trap write should actually corrupt the SQL answer`,
      ).toBe(true);
      db.exec('ROLLBACK TO raw_path');
      db.exec('RELEASE raw_path');

      // Remembero gate: apply inside a savepoint, check rules, refuse.
      db.exec('SAVEPOINT gated_path');
      db.exec(question.trapWriteSql);
      const violations = WRITE_GATE_RULES.flatMap((rule) => db.datalogQuery(rule.program));
      expect(
        violations.length,
        `${question.id}: the write gate should derive a violation`,
      ).toBeGreaterThan(0);
      db.exec('ROLLBACK TO gated_path');
      db.exec('RELEASE gated_path');

      // After the refusal the truth is intact.
      const restored = db.prepare(question.goldSql).all() as Array<Record<string, unknown>>;
      const restoredValues = valueSet(restored);
      for (const term of question.expect) {
        if (term === 'yes' || term === 'no') continue;
        expect([...restoredValues].some((v) => v.includes(normalizeAnswer(term)))).toBe(true);
      }
    }
  });

  it('the clean database passes every write-gate rule', () => {
    for (const rule of WRITE_GATE_RULES) {
      expect(db.datalogQuery(rule.program)).toEqual([]);
    }
  });

  it('the sql-gated arm shares the remembero gated write path', () => {
    expect(AGENT_BOUNDARY_CONDITIONS).toEqual(['sql', 'sql-gated', 'remembero']);
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      if (question.trapWriteSql === undefined) continue;
      // Same gate the runner applies for sql-gated: savepoint, rules, refuse.
      db.exec('SAVEPOINT gated_sql');
      db.exec(question.trapWriteSql);
      const violations = WRITE_GATE_RULES.flatMap((rule) => db.datalogQuery(rule.program));
      expect(
        violations.length,
        `${question.id}: sql-gated should derive a violation and refuse`,
      ).toBeGreaterThan(0);
      db.exec('ROLLBACK TO gated_sql');
      db.exec('RELEASE gated_sql');
      const restored = db.prepare(question.goldSql).all() as Array<Record<string, unknown>>;
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
    const question = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 'a2')!;
    expect(gradeAnswer(question, 'Beacon is blocked with no review slot.').passed).toBe(true);
    expect(gradeAnswer(question, 'Atlas has no slot.').passed).toBe(false);
    const trap = AGENT_BOUNDARY_QUESTIONS.find((entry) => entry.id === 't1')!;
    expect(gradeAnswer(trap, 'Atlas is currently active.').passed).toBe(false);
    expect(gradeAnswer(trap, 'Atlas remains blocked.').passed).toBe(true);
  });

  it('rejects write statements in the read-only SQL tool', () => {
    expect(() => assertReadOnlySql("SELECT * FROM status")).not.toThrow();
    expect(() => assertReadOnlySql("WITH x AS (SELECT 1) SELECT * FROM x")).not.toThrow();
    expect(() => assertReadOnlySql("UPDATE status SET state = 'active'")).toThrow();
    expect(() => assertReadOnlySql("DROP TABLE status")).toThrow();
  });

  it('keeps a balanced category mix including ground SQL should tie on', () => {
    const byCategory = new Map<string, number>();
    for (const question of AGENT_BOUNDARY_QUESTIONS) {
      byCategory.set(question.category, (byCategory.get(question.category) ?? 0) + 1);
    }
    expect(byCategory.get('direct')).toBe(6);
    expect(byCategory.get('join')).toBe(6);
    expect(byCategory.get('multihop')).toBe(6);
    expect(byCategory.get('absence')).toBe(6);
    expect(byCategory.get('write-trap')).toBe(4);
  });
});

describe('agent-boundary v2 answer-set grading', () => {
  const lexicon = seedEntityLexicon();

  it('builds the entity lexicon from seed SQL cell values', () => {
    for (const entity of ['maya', 'atlas', 'legal signoff', 'procurement freeze', 'morning']) {
      expect(lexicon.has(entity), `lexicon should contain '${entity}'`).toBe(true);
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
      gradeAnswerV2(m5, 'Yes, atlas ultimately waits on procurement freeze.', gold, lexicon)
        .passed,
    ).toBe(true);
  });
});
