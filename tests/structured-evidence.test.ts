import { describe, expect, it } from 'vitest';
import { buildStructuredEvidence } from '../src/knowledge/structured-evidence.js';

const q = 'How many online courses have I completed in total?';
const qd = '2023/05/30 (Tue) 16:30';

describe('buildStructuredEvidence', () => {
  it('dates facts by their session, orders them, and quotes nothing it cannot ground', () => {
    const block = buildStructuredEvidence(q, qd, [
      { ts: '2023-05-30T09:00:00Z', text: 'USER: I finished two courses on edX this week, both on statistics.', facts: ['completed 2 courses on edX', 'owns a red bicycle'] },
      { ts: '2023-05-23T09:00:00Z', text: 'USER: Just wrapped up my third Coursera course on deep learning.', facts: ['completed 3 courses on Coursera'] },
    ]);
    expect(block).toContain('Dated facts');
    expect(block.indexOf('2023-05-23')).toBeLessThan(block.indexOf('2023-05-30'));
    expect(block).toContain('completed 3 courses on Coursera');
    expect(block).toContain('completed 2 courses on edX');
    expect(block).not.toContain('red bicycle');
  });

  it('uses a temporal expression inside the fact when it has one', () => {
    const block = buildStructuredEvidence('When did I start the Coursera course?', '2023/05/30 (Tue) 16:30', [
      { ts: '2023-05-23T09:00:00Z', text: 'USER: I started the Coursera course two weeks ago.', facts: ['started the Coursera course two weeks ago'] },
    ]);
    expect(block).toMatch(/2023-05-09.*started the Coursera course two weeks ago/);
  });

  it('marks the later of two conflicting values current and keeps the earlier with its date', () => {
    const block = buildStructuredEvidence('Where do I live now?', '2023/05/30 (Tue) 16:30', [
      { ts: '2023-01-10T09:00:00Z', text: 'USER: I live in Boston these days.', facts: ['lives in Boston'] },
      { ts: '2023-04-02T09:00:00Z', text: 'USER: I moved, I live in Denver now.', facts: ['lives in Denver'] },
    ]);
    expect(block).toMatch(/2023-01-10.*lives in Boston.*superseded/);
    expect(block).toMatch(/2023-04-02.*lives in Denver.*current/);
  });

  it('is empty without facts', () => {
    expect(buildStructuredEvidence(q, qd, [{ ts: '2023-05-30T09:00:00Z', text: 'USER: hello' }])).toBe('');
  });
});
