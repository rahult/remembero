/**
 * The HTTP service: the webhook-embeddable surface for help-desk integrations. Same three
 * handlers as the MCP server (tools.ts), plain JSON over node:http, no dependencies.
 *
 *   GET  /health                       -> { ok: true }
 *   POST /decide                       { case, question, packPath?, storePath? } -> proof
 *   POST /check_claim                  { claim, spanText }                       -> gate verdict
 *   GET  /pack_coverage?pack=<path>    -> parameters + unconsumed SOP lines
 *
 * Run: node dist/support/http-server.js [--port 8088]   (npm run support:http)
 * A help-desk webhook posts the ticket plus "is a credit due for TCK-1042" and gets the
 * proof back — attach it to the ticket, or draft the credit from it.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { handleCheckClaim, handleDecide, handlePackCoverage } from './tools.js';

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk: Buffer) => {
      body += chunk;
      if (body.length > 4_000_000) reject(new Error('body too large'));
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function send(res: ServerResponse, status: number, value: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(value, null, 1));
}

export function handleRequest(url: string, method: string, body: string): { status: number; value: unknown } {
  const path = url.split('?')[0]!;
  try {
    if (method === 'GET' && path === '/health') return { status: 200, value: { ok: true } };
    if (method === 'POST' && path === '/decide') return { status: 200, value: handleDecide(JSON.parse(body)) };
    if (method === 'POST' && path === '/check_claim') return { status: 200, value: handleCheckClaim(JSON.parse(body)) };
    if (method === 'GET' && path === '/pack_coverage') {
      const packPath = new URL(`http://x${url}`).searchParams.get('pack');
      if (!packPath) return { status: 400, value: { error: 'pack query parameter is required' } };
      return { status: 200, value: handlePackCoverage({ packPath }) };
    }
    return { status: 404, value: { error: `no route: ${method} ${path}` } };
  } catch (error) {
    return { status: 400, value: { error: error instanceof Error ? error.message : String(error) } };
  }
}

export function startServer(port: number): Promise<Server> {
  const server = createServer((req, res) => {
    void readBody(req)
      .then((body) => handleRequest(req.url ?? '/', req.method ?? 'GET', body))
      .then(({ status, value }) => send(res, status, value))
      .catch((error: unknown) => send(res, 500, { error: error instanceof Error ? error.message : String(error) }));
  });
  return new Promise((resolve) => {
    server.listen(port, () => resolve(server));
  });
}

if (process.argv[1]?.endsWith('http-server.js')) {
  const at = process.argv.indexOf('--port');
  const port = at >= 0 ? Number(process.argv[at + 1]) : Number(process.env.SUPPORT_HTTP_PORT ?? 8088);
  void startServer(port).then(() => {
    console.log(`support stack on http://127.0.0.1:${port} — POST /decide, POST /check_claim, GET /pack_coverage?pack=, GET /health`);
  });
}
