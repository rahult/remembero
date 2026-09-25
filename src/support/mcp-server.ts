/**
 * The support stack as an MCP server — the surface that makes it usable from agent tooling.
 * Stdio, newline-delimited JSON-RPC 2.0, no dependencies beyond the stack itself.
 *
 * Tools:
 *   decide        { case, question, packPath?, storePath? }  plan -> gate -> engine -> proof
 *   check_claim   { claim, spanText }                        the admission gate, exposed
 *   pack_coverage { packPath }                               the pack's unconsumed SOP lines
 *
 * The LLM stays off this path by design: `decide` consumes the case's structured facts (the
 * system-of-record claims). Extraction is a separate concern that feeds the same gate.
 *
 * Run: node dist/support/mcp-server.js   (npm run support:mcp)
 */

import { createInterface } from 'node:readline';
import { admit, admitAll, type ClaimInput } from './claims.js';
import { decide, SHAPES } from './engine.js';
import { claimsFromTicket } from './extract.js';
import { loadPack } from './pack.js';
import { plan } from './planner.js';
import { chatSource, ticketSource, type ChatInput, type Source, type Span, type TicketInput } from './sources.js';
import { VerdictStore } from './store.js';

interface RpcRequest {
  jsonrpc: '2.0';
  id?: number | string | null;
  method: string;
  params?: Record<string, unknown>;
}

const SERVER_INFO = { name: 'rembero-support', version: '0.1.0' };

const TOOLS = [
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
    description: "Load a policy pack and return its parameters plus the SOP lines no claim consumes — the unmodeled-clause review.",
    inputSchema: {
      type: 'object',
      properties: { packPath: { type: 'string' } },
      required: ['packPath'],
    },
  },
];

function handleToolCall(params: Record<string, unknown>): { content: Array<{ type: string; text: string }> } {
  const name = params.name as string;
  const args = (params.arguments ?? {}) as Record<string, unknown>;
  let result: unknown;
  if (name === 'decide') {
    const caseFile = args.case as { ticket: TicketInput; chat?: ChatInput };
    const packPath = (args.packPath as string) ?? 'benchmarks/support/pack.json';
    const pack = loadPack(packPath);
    const sources: Source[] = [ticketSource(caseFile.ticket)];
    if (caseFile.chat) sources.push(chatSource(caseFile.chat));
    const p = plan(String(args.question ?? ''));
    if (!p.ok) result = { rejected: p.reason };
    else {
      const report = admitAll(claimsFromTicket(caseFile.ticket), sources);
      const proof = decide(p.shape, p.params, [...pack.claims, ...report.admitted], {
        packId: pack.id,
        packVersion: pack.version,
        spansSearched: sources.reduce((n, s) => n + s.spans.length, 0),
      });
      const store = new VerdictStore((args.storePath as string) ?? 'results/support-verdicts.jsonl');
      const record = store.append(proof, { id: pack.id, version: pack.version, contentHash: pack.contentHash });
      result = { seq: record.seq, decisionHash: record.decisionHash, proof };
    }
  } else if (name === 'check_claim') {
    const claim = args.claim as Partial<ClaimInput>;
    const span: Span = {
      sourceId: String(claim.sourceId ?? 'probe'),
      spanId: String(claim.spanId ?? 'probe'),
      kind: 'policy-line',
      text: String(args.spanText ?? ''),
    };
    const input: ClaimInput = {
      scope: claim.scope ?? 'case',
      predicate: String(claim.predicate ?? ''),
      args: (claim.args ?? []) as (string | number)[],
      sourceId: span.sourceId,
      spanId: span.spanId,
      trust: claim.trust ?? 'model',
    };
    result = admit(input, span);
  } else if (name === 'pack_coverage') {
    const pack = loadPack(String(args.packPath));
    result = { id: pack.id, version: pack.version, parameters: pack.claims.map((c) => `${c.predicate}(${c.args.join(', ')})`), unconsumedLines: pack.unconsumedLines };
  } else {
    result = { error: `unknown tool ${name}` };
  }
  return { content: [{ type: 'text', text: JSON.stringify(result, null, 1) }] };
}

export function handleRpc(msg: RpcRequest): string | null {
  if (msg.method === 'initialize') {
    return JSON.stringify({
      jsonrpc: '2.0', id: msg.id ?? null,
      result: { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: SERVER_INFO },
    });
  }
  if (msg.method.startsWith('notifications/')) return null;
  if (msg.method === 'tools/list') {
    return JSON.stringify({ jsonrpc: '2.0', id: msg.id ?? null, result: { tools: TOOLS } });
  }
  if (msg.method === 'tools/call') {
    return JSON.stringify({ jsonrpc: '2.0', id: msg.id ?? null, result: handleToolCall(msg.params ?? {}) });
  }
  if (msg.method === 'ping') {
    return JSON.stringify({ jsonrpc: '2.0', id: msg.id ?? null, result: {} });
  }
  return JSON.stringify({ jsonrpc: '2.0', id: msg.id ?? null, error: { code: -32601, message: `method not found: ${msg.method}` } });
}

export function main(): void {
  const rl = createInterface({ input: process.stdin });
  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (trimmed.length === 0) return;
    let msg: RpcRequest;
    try {
      msg = JSON.parse(trimmed) as RpcRequest;
    } catch {
      return;
    }
    const out = handleRpc(msg);
    if (out !== null) process.stdout.write(`${out}\n`);
  });
}

if (process.argv[1]?.endsWith('mcp-server.js')) {
  main();
}
