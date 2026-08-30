import { describe, expect, it, beforeAll, afterAll } from 'vitest';

import {
  CHAT_MEMORY_SCENARIOS,
  diagnoseWhyNot,
  runChatMemoryScenario,
} from '../site/lib/chat-memory-lab.js';
import {
  CHAT_MEMORY_SQLITE_SETUP,
  CHAT_MEMORY_SQLITE_VERIFY,
  WHY_NOT_PREMISES,
} from '../site/lib/chat-memory-tools.js';
import { openRememberoDatabase, type RememberoDatabase } from '../src/sqlite/extension.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

const SQL_ROOT_BLOCKER = `WITH RECURSIVE chain(item, upstream) AS (
  SELECT item, upstream FROM waits_on WHERE item = 'atlas'
  UNION
  SELECT w.item, w.upstream FROM waits_on AS w
  JOIN chain AS c ON w.item = c.upstream
)
SELECT upstream AS dependency FROM chain`;

describe.skipIf(nodeMajor < 22)('chat-memory lab structural scenarios over real SQLite', () => {
  let db: RememberoDatabase;

  beforeAll(async () => {
    db = await openRememberoDatabase(':memory:');
    db.exec(CHAT_MEMORY_SQLITE_SETUP);
  });

  afterAll(() => {
    db.close();
  });

  it('seed verification counts match the schema the lab installs', () => {
    const row = db.prepare(CHAT_MEMORY_SQLITE_VERIFY).get() as {
      tableCount: number;
      rowCount: number;
    };
    expect(row.tableCount).toBe(6);
    expect(row.rowCount).toBe(9);
  });

  it('root-blocker: recursive SQL and the Datalog rule agree, but only Datalog carries a proof', () => {
    const sqlRows = db.prepare(SQL_ROOT_BLOCKER).all() as Array<{ dependency: string }>;
    expect(sqlRows.map((row) => row.dependency)).toEqual([
      'vendor_security_review',
      'legal_signoff',
      'procurement_freeze',
    ]);

    const datalog = `root_blocker(Root) :- reaches(atlas, Root), \\+ waits_on(Root, _).
reaches(P, X) :- waits_on(P, X).
reaches(P, X) :- reaches(P, M), waits_on(M, X).`;
    const rows = db.datalogQuery(datalog);
    expect(rows).toEqual([{ Root: 'procurement_freeze' }]);

    const explanations = db.datalogExplain(datalog);
    expect(explanations.length).toBeGreaterThan(0);
    const serialized = JSON.stringify(explanations);
    expect(serialized).toContain('procurement_freeze');
    expect(serialized).toContain('legal_signoff');
  });

  it('write-gate: SQLite accepts the contradictory update silently; the Datalog constraint derives the violation', () => {
    db.exec('SAVEPOINT gate_test');
    try {
      db.exec("UPDATE status SET state = 'active' WHERE project = 'atlas'");
      const scheduled = db
        .prepare(
          `SELECT s.project, s.state, r.day, r.window
             FROM status AS s
             JOIN review_slot AS r ON r.project = s.project
            WHERE s.project = 'atlas'`,
        )
        .all() as Array<{ state: string; day: string }>;
      expect(scheduled).toEqual([
        { project: 'atlas', state: 'active', day: 'tuesday', window: 'morning' },
      ]);

      db.exec('CREATE TABLE proposed_status(project TEXT NOT NULL, state TEXT NOT NULL)');
      db.exec("INSERT INTO proposed_status VALUES ('atlas', 'active')");
      const violations = db.datalogQuery(
        'violation(P, B) :- proposed_status(P, active), blocker(P, B).',
      );
      expect(violations).toEqual([{ P: 'atlas', B: 'vendor_security_review' }]);
    } finally {
      db.exec('ROLLBACK TO gate_test');
      db.exec('RELEASE gate_test');
    }
    const restored = db
      .prepare("SELECT state FROM status WHERE project = 'atlas'")
      .get() as { state: string };
    expect(restored.state).toBe('blocked');
  });

  it('unknown-preference: SQL returns a NULL cell; Datalog derives the absence through negation', () => {
    const sqlRows = db
      .prepare(
        `SELECT m.person, m.meeting, p.window AS stored_preference
           FROM pending_meeting AS m
           LEFT JOIN prefers_meeting AS p ON p.person = m.person
          WHERE m.person = 'jordan'`,
      )
      .all() as Array<{ stored_preference: string | null }>;
    expect(sqlRows).toEqual([
      { person: 'jordan', meeting: 'roadmap_sync', stored_preference: null },
    ]);

    const rows = db.datalogQuery(
      `missing_preference(Person) :- pending_meeting(Person, _), \\+ prefers_meeting(Person, _).`,
    );
    expect(rows).toEqual([{ Person: 'jordan' }]);
  });

  it('why-not: the prepared SQL is empty while premise checks name the failing conditions', () => {
    const sqlRows = db
      .prepare(
        `SELECT s.project, r.day, r.window, b.blocker
           FROM status AS s
           JOIN review_slot AS r ON r.project = s.project
           JOIN blocker AS b ON b.project = s.project
           JOIN prefers_meeting AS p ON p.person = 'maya' AND p.window = r.window
          WHERE s.project = 'orchard' AND s.state = 'blocked'`,
      )
      .all();
    expect(sqlRows).toEqual([]);

    const verdicts = WHY_NOT_PREMISES.map((premise) => ({
      literal: premise.literal,
      holds: db.datalogQuery(premise.program).length > 0,
    }));
    expect(verdicts).toEqual([
      { literal: 'review_slot(orchard, Day, Window)', holds: false },
      { literal: 'status(orchard, blocked)', holds: false },
      { literal: 'blocker(orchard, Blocker)', holds: false },
      { literal: 'prefers_meeting(maya, Window)', holds: true },
    ]);
  });
});

describe('chat-memory lab TypeScript-engine comparisons', () => {
  it('every scenario answer satisfies its own required terms', () => {
    for (const scenario of CHAT_MEMORY_SCENARIOS) {
      const run = runChatMemoryScenario(scenario.id);
      const normalized = run.answer.toLowerCase();
      for (const term of scenario.requiredTerms) {
        if (scenario.id === 'write-gate' && term === 'refus') {
          expect(normalized).toContain('refus');
          continue;
        }
        expect(normalized).toContain(term.toLowerCase());
      }
    }
  });

  it('root-blocker proof trail walks the full dependency chain', () => {
    const run = runChatMemoryScenario('root-blocker');
    const trail = run.proofTrail.join('\n');
    expect(trail).toContain('vendor_security_review');
    expect(trail).toContain('legal_signoff');
    expect(trail).toContain('procurement_freeze');
    expect(run.absences.length).toBeGreaterThan(0);
  });

  it('write-gate comparison derives the violated constraint with culprit facts', () => {
    const run = runChatMemoryScenario('write-gate');
    expect(run.bindings.P).toBe('atlas');
    expect(run.bindings.B).toBe('vendor_security_review');
    expect(run.answer.toLowerCase()).toContain('refused');
  });

  it('why-not diagnosis names each failing premise and the holding one', () => {
    const premises = diagnoseWhyNot();
    expect(premises.map((premise) => premise.holds)).toEqual([false, false, false, true]);
    const run = runChatMemoryScenario('why-not');
    expect(run.absenceCount).toBe(3);
    expect(run.proofTrail.some((line) => line.startsWith('✗'))).toBe(true);
  });

  it('unknown-preference proof contains the verified absence', () => {
    const run = runChatMemoryScenario('unknown-preference');
    expect(run.absences.length).toBeGreaterThan(0);
    expect(run.absences[0]).toContain('prefers_meeting');
  });
});
