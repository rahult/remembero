/**
 * The planner: question -> one of the closed shapes, with parameters bound to ids. It is
 * deliberately dumb and deterministic at v0 (a local model takes this job later, emitting
 * into the same enum with the same rejection rule). The rule that matters: a question that
 * does not map to a shape, or that names an entity only by description, is REJECTED — never
 * guessed. Ambiguity becomes workflow (the agent supplies the ticket id), not resolution
 * heuristics.
 */

import { SHAPES, type Shape } from './engine.js';

export type Plan = { ok: true; shape: Shape; params: Record<string, string> } | { ok: false; reason: string };

const TICKET_ID = /\b([a-z]{2,5}-\d{2,})\b/i;

export function plan(question: string): Plan {
  const idMatch = question.match(TICKET_ID);
  // any id-style token binds (TCK-1042, RTN-30001, ...); ids are canonical lowercase
  const ticket = idMatch ? idMatch[1]!.toLowerCase() : undefined;
  const q = question.toLowerCase();

  let shape: Shape | undefined;
  if (/\b(deadline|last day|by when|how long do i have)\b/.test(q) && /\b(refund|return)\b/.test(q)) {
    shape = 'refund_deadline';
  } else if (/\b(refund|return)\b/.test(q)) {
    shape = 'refund_eligible';
  } else if (/\b(credit|compensation|payout)\b/.test(q)) {
    shape = /\bpercent|%\b/.test(q) ? 'credit_percent' : 'sla_credits';
  } else if (/\b(within (the )?sla|on time|met|breach(ed)?|late)\b/.test(q)) {
    shape = 'sla_met';
  }
  if (!shape) {
    return { ok: false, reason: `no query shape matches this question (shapes: ${SHAPES.join(', ')})` };
  }
  if (!ticket) {
    return { ok: false, reason: 'the question names no ticket id — bind by id (e.g. TCK-1042); the engine does not resolve descriptions' };
  }
  return { ok: true, shape, params: { ticket } };
}
