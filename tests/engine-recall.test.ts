import { describe, expect, it } from 'vitest';
import { parseProgram, parseQueryProgram } from '../src/engine/index.js';
import {
  crossProductEstimate,
  dialectSchemaListing,
  runProgram,
} from '../src/evals/engine-recall.js';

const facts = parseProgram(
  Array.from(
    { length: 40 },
    (_, i) =>
      `learned(user, a${i}, b${i}). attended(user, c${i}, d${i}). wants(user, e${i}).`,
  ).join('\n') + '\nlongmem_session(session_1).',
);

describe('engine recall guards', () => {
  it('estimates the size of a wildcard cross product before the engine runs it', () => {
    const joined = parseQueryProgram(
      'count(*) as N where learned(user, X, _), attended(user, X, _)',
    );
    expect(crossProductEstimate(facts, joined.query, joined.clauses)).toBe(40);
    const product = parseQueryProgram(
      'count(*) as N where learned(user, _, _), attended(user, _, _), wants(user, _), learned(user, _, _)',
    );
    expect(crossProductEstimate(facts, product.query, product.clauses)).toBe(
      40 ** 4,
    );
  });

  it('lists the store predicates with placeholders and samples, plumbing excluded', () => {
    const listing = dialectSchemaListing(facts);
    expect(listing).toContain('learned(A1, A2, A3)');
    expect(listing).toContain('e.g. learned(user, a0, b0).');
    expect(listing).not.toContain('longmem_session');
  });

  it('renders a count with the items behind it and treats a zero count as empty', () => {
    const counted = runProgram(
      facts,
      new Map(),
      parseQueryProgram('count(*) as N where wants(user, X)'),
    );
    expect(counted.rows).toBe(40);
    expect(counted.rendered).toMatch(
      /^The program matched 40 remembered items/,
    );
    expect(counted.rendered).toContain('X = e0');
    const none = runProgram(
      facts,
      new Map(),
      parseQueryProgram('count(*) as N where wants(user, nothing_like_this)'),
    );
    expect(none.rows).toBe(0);
  });
});
