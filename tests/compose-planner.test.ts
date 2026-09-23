import { describe, expect, it } from 'vitest';
import { engineCall, parsePlan, planProblem } from '../src/compose/planner.js';

describe('planner', () => {
  it('parses a JSON plan, even wrapped in prose', () => {
    expect(parsePlan('Here: {"shape":"value_on_date","params":{"contract":"CN-1234","date":20250314}}')).toEqual({
      shape: 'value_on_date',
      params: { contract: 'CN-1234', date: 20250314 },
    });
  });

  it('treats an unknown shape or broken JSON as no plan', () => {
    expect(parsePlan('{"shape":"guess","params":{}}').shape).toBe('none');
    expect(parsePlan('not json').shape).toBe('none');
  });

  it('accepts parameters the question contains', () => {
    const q = 'What was the value of contract CN-1234 on 14 March 2025, taking any amendments into account?';
    expect(planProblem(q, { shape: 'value_on_date', params: { contract: 'CN-1234', date: 20250314 } })).toBeUndefined();
  });

  it('rejects an invented date or reference', () => {
    const q = 'What was the value of contract CN-1234 on 14 March 2025?';
    expect(planProblem(q, { shape: 'value_on_date', params: { contract: 'CN-1234', date: 20250315 } })).toMatch(/not in the question/);
    expect(planProblem(q, { shape: 'value_on_date', params: { contract: 'CN-9999', date: 20250314 } })).toMatch(/not in the question/);
  });

  it('rejects a plan missing a parameter', () => {
    expect(planProblem('Who was the Chief Risk Officer?', { shape: 'role_holder_on_date', params: { role: 'Chief Risk Officer' } })).toMatch(/missing date/);
  });

  it('maps a shape to the engine family', () => {
    expect(engineCall({ shape: 'approval_within_authority', params: { contract: 'CN-1' } }).family).toBe('authority');
  });
});
