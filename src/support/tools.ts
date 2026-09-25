/**
 * The stack's tool surface, shared by both servers: the MCP server (stdio) and the HTTP
 * service expose the SAME three handlers, so the two surfaces cannot drift. Each handler
 * returns a plain JSON value; the servers only differ in transport framing.
 *
 * The LLM stays off this path by design: `decide` consumes the case's structured facts (the
 * system-of-record claims). Extraction is a separate concern that feeds the same gate.
 */

import { admit, admitAll, type ClaimInput } from './claims.js';
import { decide, SHAPES } from './engine.js';
import { claimsFromTicket } from './extract.js';
import { loadPack } from './pack.js';
import { plan } from './planner.js';
import { chatSource, ticketSource, type ChatInput, type Source, type Span, type TicketInput } from './sources.js';
import { VerdictStore } from './store.js';

export interface DecideArgs {
  case: { ticket: TicketInput; chat?: ChatInput };
  question: string;
  packPath?: string;
  storePath?: string;
}

export function handleDecide(args: DecideArgs): unknown {
  const pack = loadPack(args.packPath ?? 'benchmarks/support/pack.json');
  const sources: Source[] = [ticketSource(args.case.ticket)];
  if (args.case.chat) sources.push(chatSource(args.case.chat));
  const p = plan(args.question);
  if (!p.ok) return { rejected: p.reason };
  const report = admitAll(claimsFromTicket(args.case.ticket), sources);
  const proof = decide(p.shape, p.params, [...pack.claims, ...report.admitted], {
    packId: pack.id,
    packVersion: pack.version,
    spansSearched: sources.reduce((n, s) => n + s.spans.length, 0),
  });
  const store = new VerdictStore(args.storePath ?? 'results/support-verdicts.jsonl');
  const record = store.append(proof, { id: pack.id, version: pack.version, contentHash: pack.contentHash });
  return { seq: record.seq, decisionHash: record.decisionHash, proof };
}

export interface CheckClaimArgs {
  claim: Partial<ClaimInput>;
  spanText: string;
}

export function handleCheckClaim(args: CheckClaimArgs): unknown {
  const span: Span = {
    sourceId: String(args.claim.sourceId ?? 'probe'),
    spanId: String(args.claim.spanId ?? 'probe'),
    kind: 'policy-line',
    text: args.spanText,
  };
  const input: ClaimInput = {
    scope: args.claim.scope ?? 'case',
    predicate: String(args.claim.predicate ?? ''),
    args: (args.claim.args ?? []) as (string | number)[],
    sourceId: span.sourceId,
    spanId: span.spanId,
    trust: args.claim.trust ?? 'model',
  };
  return admit(input, span);
}

export interface PackCoverageArgs {
  packPath: string;
}

export function handlePackCoverage(args: PackCoverageArgs): unknown {
  const pack = loadPack(args.packPath);
  return {
    id: pack.id,
    version: pack.version,
    parameters: pack.claims.map((c) => `${c.predicate}(${c.args.join(', ')})`),
    unconsumedLines: pack.unconsumedLines,
  };
}

export const TOOL_DEFS = [
  {
    name: 'decide',
    description: `Decide a support question symbolically: plans the question into a closed query shape (${SHAPES.join(', ')}), gates the case's structured claims, and returns a verdict (allow | deny | unknown) with a proof citing spans and pack lines. Never guesses; Unknown carries the reason.`,
    inputSchema: {
      type: 'object',
      properties: {
        case: { type: 'object', description: 'The case: { ticket: { id, customer?, fields, comments? }, chat?: { id, turns } }' },
        question: { type: 'string', description: 'e.g. "is a credit due for TCK-1042" — must name the ticket id' },
        packPath: { type: 'string', description: 'pack JSON path (default benchmarks/support/pack.json)' },
        storePath: { type: 'string', description: 'append-only verdict store (default results/support-verdicts.jsonl)' },
      },
      required: ['case', 'question'],
    },
  },
  {
    name: 'check_claim',
    description: 'The admission gate, exposed: would this claim be admitted against this span text? Returns the verdict and the rejection reason.',
    inputSchema: {
      type: 'object',
      properties: {
        claim: { type: 'object', description: '{ predicate, args, sourceId?, spanId?, trust? }' },
        spanText: { type: 'string' },
      },
      required: ['claim', 'spanText'],
    },
  },
  {
    name: 'pack_coverage',
    description: 'Load a policy pack and return its parameters plus the SOP lines no claim consumes — the unmodeled-clause review.',
    inputSchema: {
      type: 'object',
      properties: { packPath: { type: 'string' } },
      required: ['packPath'],
    },
  },
] as const;
