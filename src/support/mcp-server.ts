/**
 * The MCP server: stdio, newline-delimited JSON-RPC 2.0, framing only. Every tool's logic
 * lives in tools.ts so the MCP and HTTP surfaces cannot drift.
 *
 * Run: node dist/support/mcp-server.js   (npm run support:mcp)
 */

import { createInterface } from 'node:readline';
import { handleCheckClaim, handleDecide, handlePackCoverage, TOOL_DEFS } from './tools.js';

interface RpcRequest {
  jsonrpc: '2.0';
  id?: number | string | null;
  method: string;
  params?: Record<string, unknown>;
}

const SERVER_INFO = { name: 'rembero-support', version: '0.1.0' };

function handleToolCall(params: Record<string, unknown>): { content: Array<{ type: string; text: string }> } {
  const name = params.name as string;
  const args = (params.arguments ?? {}) as Record<string, never>;
  let result: unknown;
  if (name === 'decide') result = handleDecide(args as never);
  else if (name === 'check_claim') result = handleCheckClaim(args as never);
  else if (name === 'pack_coverage') result = handlePackCoverage(args as never);
  else result = { error: `unknown tool ${name}` };
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
    return JSON.stringify({ jsonrpc: '2.0', id: msg.id ?? null, result: { tools: TOOL_DEFS } });
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
