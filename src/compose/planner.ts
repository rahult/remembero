/**
 * Turn a question into an engine call: which query shape, with which parameters.
 *
 * The oracle planner read these from the question generator. A real planner reads only the
 * question text. It picks one of the query shapes below and fills its parameters; code then
 * checks every parameter against the question (a contract reference must appear in it, a date
 * must be spelled in it) before the engine runs anything — a plan that invents a parameter is
 * rejected, and the answer is Unknown.
 */

import { dateSpellings } from './dates.js';

export const SHAPES = {
  value_at_signing: { family: 'needle', params: ['contract'] },
  role_holder_on_date: { family: 'validity', params: ['role', 'date', 'organisation'] },
  approver_of_contract: { family: 'identity', params: ['contract'] },
  value_on_date: { family: 'supersession', params: ['contract', 'date'] },
  higher_value_on_date: { family: 'comparison', params: ['contractA', 'contractB', 'date'] },
  count_incidents_without_certificate: { family: 'aggregate', params: ['standard', 'organisation'] },
  count_incidents_for_supplier: { family: 'aggregate', params: ['supplier'] },
  suppliers_without_certificate_on_date: { family: 'absence', params: ['standard', 'date', 'organisation'] },
  approval_within_authority: { family: 'authority', params: ['contract'] },
} as const;
export type Shape = keyof typeof SHAPES;

export const PLANNER_PROMPT = `You translate a question about an organisation's records into a query plan for a Datalog engine. You do not answer the question.

Choose exactly one shape and fill its parameters:
- value_at_signing {contract}: the value of a contract when it was signed
- role_holder_on_date {role, date, organisation}: who held an office (e.g. Chief Risk Officer) at an organisation on a date
- approver_of_contract {contract}: who approved a contract
- value_on_date {contract, date}: a contract's value on a date, after amendments
- higher_value_on_date {contractA, contractB, date}: which of two contracts was worth more on a date
- count_incidents_without_certificate {standard, organisation}: how many incidents happened at sites whose operating supplier lacked a valid certificate of a standard that day
- count_incidents_for_supplier {supplier}: how many incidents happened at sites operated by a supplier
- suppliers_without_certificate_on_date {standard, date, organisation}: which suppliers lacked a valid certificate of a standard on a date
- approval_within_authority {contract}: whether a contract's approval was within the approver's delegated authority

Parameters: contract references exactly as written (CN-1234); dates as yyyymmdd integers; role, supplier, organisation and standard as written in the question ("ISO 9001", "ISO/IEC 27001").
Reply with JSON only: {"shape": "...", "params": {...}}. If no shape fits, reply {"shape": "none", "params": {}}.`;

export interface Plan {
  shape: Shape | 'none';
  params: Record<string, string | number>;
}

export function parsePlan(reply: string): Plan {
  const json = /\{[\s\S]*\}/.exec(reply)?.[0];
  if (json === undefined) return { shape: 'none', params: {} };
  try {
    const value = JSON.parse(json) as { shape?: unknown; params?: unknown };
    const shape = typeof value.shape === 'string' && value.shape in SHAPES ? (value.shape as Shape) : 'none';
    const params: Record<string, string | number> = {};
    if (value.params !== null && typeof value.params === 'object') {
      for (const [k, v] of Object.entries(value.params as Record<string, unknown>)) {
        if (typeof v === 'string' || typeof v === 'number') params[k] = v;
      }
    }
    return { shape, params };
  } catch {
    return { shape: 'none', params: {} };
  }
}

function squash(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Is every parameter the plan needs present, and grounded in the question? Returns the reason it
 * is not, or undefined. Dates are grounded when one of their spellings appears in the question.
 */
export function planProblem(question: string, plan: Plan): string | undefined {
  if (plan.shape === 'none') return 'no shape fits';
  const text = squash(question);
  for (const name of SHAPES[plan.shape].params) {
    const value = plan.params[name];
    if (value === undefined) return `missing ${name}`;
    if (name === 'date') {
      const date = Number(value);
      if (!Number.isInteger(date) || date < 19_000_101 || date > 21_001_231) return `date ${value} is not yyyymmdd`;
      if (!dateSpellings(date).some((s) => text.includes(squash(s)))) return `date ${value} is not in the question`;
    } else if (!text.includes(squash(String(value)))) {
      return `${name} "${value}" is not in the question`;
    }
  }
  return undefined;
}

/** The engine family and parameters for a grounded plan. */
export function engineCall(plan: Plan): { family: string; params: Record<string, string | number> } {
  if (plan.shape === 'none') return { family: 'none', params: {} };
  const params = { ...plan.params };
  if (params.date !== undefined) params.date = Number(params.date);
  return { family: SHAPES[plan.shape].family, params };
}
