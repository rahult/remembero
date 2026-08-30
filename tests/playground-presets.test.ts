import { describe, expect, it, beforeAll, afterAll } from 'vitest';

import { PRESETS } from '../site/lib/ide-demo.js';
import { openRememberoDatabase, type RememberoDatabase } from '../src/sqlite/extension.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);

function preset(id: string) {
  const found = PRESETS.find((entry) => entry.id === id);
  if (!found?.setupSql) throw new Error(`preset ${id} with setupSql not found`);
  return found;
}

describe.skipIf(nodeMajor < 22)('beyond-SQL playground presets over real SQLite', () => {
  let db: RememberoDatabase;

  beforeAll(async () => {
    db = await openRememberoDatabase(':memory:');
  });

  afterAll(() => {
    db.close();
  });

  it('proven_absence derives the ungoverned document through negation', () => {
    const entry = preset('proven_absence');
    db.exec(entry.setupSql!);
    const rows = db.datalogQuery(entry.program);
    expect(rows).toEqual([{ Document: 'postmortem' }]);
    const explanations = db.datalogExplain(entry.program);
    expect(JSON.stringify(explanations)).toContain('postmortem');
  });

  it('write_gate derives the exact contradiction a bare UPDATE would ignore', () => {
    const entry = preset('write_gate');
    db.exec(entry.setupSql!);
    const rows = db.datalogQuery(entry.program);
    expect(rows).toEqual([{ Project: 'atlas', Blocker: 'vendor_security_review' }]);
  });

  it('why_not names the failing requirement instead of returning an empty set', () => {
    const entry = preset('why_not');
    db.exec(entry.setupSql!);
    const rows = db.datalogQuery(entry.program);
    expect(rows).toEqual([{ Service: 'search', Check: 'security' }]);
  });
});
