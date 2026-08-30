import { describe, expect, it, beforeAll, afterAll } from 'vitest';

import {
  portableExecutionMode,
  runPortableExplain,
  runPortableQuery,
  type PortableExec,
} from '../site/lib/portable-datalog.js';
import { CHAT_MEMORY_SQLITE_SETUP, WHY_NOT_PREMISES } from '../site/lib/chat-memory-tools.js';
import { proofStepTitle, proofStepValues } from '../site/lib/sqlite-wasm.js';
import { PRESETS } from '../site/lib/ide-demo.js';
import { openRememberoDatabase, type RememberoDatabase } from '../src/sqlite/extension.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

const ROOT_BLOCKER = `root_blocker(Root) :- reaches(atlas, Root), \\+ waits_on(Root, _).
reaches(P, X) :- waits_on(P, X).
reaches(P, X) :- reaches(P, M), waits_on(M, X).`;

describe.skipIf(nodeMajor < 22)(
  'browser portable Datalog fallback matches the lab programs',
  () => {
    let db: RememberoDatabase;
    let exec: PortableExec;

    beforeAll(async () => {
      db = await openRememberoDatabase(':memory:');
      db.exec(CHAT_MEMORY_SQLITE_SETUP);
      exec = async (sql: string) => db.prepare(sql).all() as Array<Record<string, unknown>>;
    });

    afterAll(() => {
      db.close();
    });

    it('routes every lab and preset program the C parser cannot handle to the portable engine', () => {
      expect(portableExecutionMode(ROOT_BLOCKER)).toBe('portable');
      expect(
        portableExecutionMode(
          'violation(P, B) :- proposed_status(P, active), blocker(P, B).',
        ),
      ).toBe('native');
      expect(
        portableExecutionMode(
          `missing_preference(Person) :- pending_meeting(Person, _), \\+ prefers_meeting(Person, _).`,
        ),
      ).toBe('portable');
      for (const premise of WHY_NOT_PREMISES) {
        // Premise programs must run somewhere; comparison forms go portable.
        expect(['native', 'portable']).toContain(portableExecutionMode(premise.program));
      }
    });

    it('answers root-blocker with the recursive chain and an explanation', async () => {
      const rows = await runPortableQuery(exec, ROOT_BLOCKER);
      expect(rows).toEqual([{ Root: 'procurement_freeze' }]);
      const explanations = await runPortableExplain(exec, ROOT_BLOCKER);
      const serialized = JSON.stringify(explanations);
      expect(serialized).toContain('procurement_freeze');
      expect(serialized).toContain('legal_signoff');
    });

    it('derives the proven absence for jordan through the portable engine', async () => {
      const rows = await runPortableQuery(
        exec,
        `missing_preference(Person) :- pending_meeting(Person, _), \\+ prefers_meeting(Person, _).`,
      );
      expect(rows).toEqual([{ Person: 'jordan' }]);
    });

    it('evaluates every why-not premise without error', async () => {
      const verdicts = [];
      for (const premise of WHY_NOT_PREMISES) {
        const rows =
          portableExecutionMode(premise.program) === 'portable'
            ? await runPortableQuery(exec, premise.program)
            : db.datalogQuery(premise.program);
        verdicts.push({ literal: premise.literal, holds: rows.length > 0 });
      }
      expect(verdicts).toEqual([
        { literal: 'review_slot(orchard, Day, Window)', holds: false },
        { literal: 'status(orchard, blocked)', holds: false },
        { literal: 'blocker(orchard, Blocker)', holds: false },
        { literal: 'prefers_meeting(maya, Window)', holds: true },
      ]);
    });

    it('renders verified-absence proof steps without assuming a values array', async () => {
      const entry = PRESETS.find((preset) => preset.id === 'proven_absence')!;
      const scratch = await openRememberoDatabase(':memory:');
      try {
        scratch.exec(entry.setupSql!);
        const scratchExec: PortableExec = async (sql) =>
          scratch.prepare(sql).all() as Array<Record<string, unknown>>;
        const explanations = await runPortableExplain(scratchExec, entry.program);
        expect(explanations.length).toBe(1);
        const collect = (proof: unknown, out: unknown[] = []): unknown[] => {
          out.push(proof);
          const because = (proof as { because?: unknown[] }).because ?? [];
          for (const child of because) collect(child, out);
          return out;
        };
        const steps = collect(explanations[0].proof);
        const absence = steps.find((step) => (step as { negated?: boolean }).negated);
        expect(absence).toBeDefined();
        // The display helpers must not throw on any step shape.
        for (const step of steps) {
          const title = proofStepTitle(step as never);
          const values = proofStepValues(step as never);
          expect(typeof title).toBe('string');
          expect(Array.isArray(values)).toBe(true);
        }
        const absenceTitle = proofStepTitle(absence as never);
        expect(absenceTitle).toBe('not team_grant');
      } finally {
        scratch.close();
      }
    });

    it('runs the beyond-SQL playground presets through the same routing the browser uses', async () => {
      const scratch = await openRememberoDatabase(':memory:');
      const scratchExec: PortableExec = async (sql) =>
        scratch.prepare(sql).all() as Array<Record<string, unknown>>;
      try {
        for (const preset of PRESETS) {
          if (!preset.setupSql) continue;
          scratch.exec(preset.setupSql);
          const rows =
            portableExecutionMode(preset.program) === 'portable'
              ? await runPortableQuery(scratchExec, preset.program)
              : scratch.datalogQuery(preset.program);
          if (preset.id === 'proven_absence') {
            expect(rows).toEqual([{ Document: 'postmortem' }]);
          }
          if (preset.id === 'write_gate') {
            expect(rows).toEqual([
              { Project: 'atlas', Blocker: 'vendor_security_review' },
            ]);
          }
          if (preset.id === 'why_not') {
            expect(rows).toEqual([{ Service: 'search', Check: 'security' }]);
          }
        }
      } finally {
        scratch.close();
      }
    });
  },
);
