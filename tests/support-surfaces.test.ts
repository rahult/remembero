import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { afterAll, describe, expect, it } from 'vitest';
import { startServer } from '../src/support/http-server.js';
import { runPipeline, type PipelineReport } from '../src/evals/run-support-pipeline.js';

describe('mcp server', () => {
  it('answers initialize, tools/list, and decides the demo case through tools/call', async () => {
    const child = spawn('node', ['dist/support/mcp-server.js']);
    const lines: string[] = [];
    let buffer = '';
    child.stdout.on('data', (d: Buffer) => {
      buffer += d;
      let i: number;
      while ((i = buffer.indexOf('\n')) >= 0) {
        lines.push(buffer.slice(0, i));
        buffer = buffer.slice(i + 1);
      }
    });
    const send = (msg: object) => child.stdin.write(`${JSON.stringify(msg)}\n`);
    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} });
    send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
    send({
      jsonrpc: '2.0', id: 3, method: 'tools/call',
      params: { name: 'decide', arguments: { case: JSON.parse(readFileSync('benchmarks/support/demo-case.json', 'utf8')), question: 'is a credit due for TCK-1042', storePath: '/tmp/mcp-test-verdicts.jsonl' } },
    });
    await new Promise((r) => setTimeout(r, 1500));
    child.kill();
    expect(lines.length).toBeGreaterThanOrEqual(3);
    const init = JSON.parse(lines[0]!);
    expect(init.result.serverInfo.name).toBe('rembero-support');
    const tools = JSON.parse(lines[1]!);
    expect(tools.result.tools.map((t: { name: string }) => t.name)).toEqual(['decide', 'check_claim', 'pack_coverage']);
    const decided = JSON.parse(lines[2]!);
    const payload = JSON.parse(decided.result.content[0].text);
    expect(payload.proof.verdict).toBe('allow');
    expect(payload.proof.summary).toContain('25%');
  }, 15000);
});

describe('http server', () => {
  let server: import('node:http').Server;
  let base = '';
  afterAll(() => {
    server?.close();
  });

  it('serves health, decides, and exposes the gate over the same handlers as MCP', async () => {
    server = await startServer(0);
    const address = server.address();
    base = `http://127.0.0.1:${typeof address === 'object' && address ? address.port : 0}`;

    const health = await (await fetch(`${base}/health`)).json();
    expect(health).toEqual({ ok: true });

    const demoCase = JSON.parse(readFileSync('benchmarks/support/demo-case.json', 'utf8'));
    const decided = await (await fetch(`${base}/decide`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ case: demoCase, question: 'is a credit due for TCK-1042', storePath: '/tmp/http-test-verdicts.jsonl' }),
    })).json();
    expect(decided.proof.verdict).toBe('allow');
    expect(decided.proof.computation.creditAmount).toBe(300);

    const gate = await (await fetch(`${base}/check_claim`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ claim: { predicate: 'tier', args: ['cust-acme', 'platinum'] }, spanText: 'TCK-1042 customer cust-acme tier: enterprise' }),
    })).json();
    expect(gate.ok).toBe(false);
    expect(gate.reason).toContain('not stated on span');

    const coverage = await (await fetch(`${base}/pack_coverage?pack=benchmarks/support/pack-refund.json`)).json();
    expect(coverage.id).toBe('northwind-refund');
    expect(coverage.unconsumedLines.length).toBeGreaterThanOrEqual(3);
  });
});

describe('pipeline (Phase 4: export -> decide -> store -> disputes report)', () => {
  it('decides a real export file, appends verdicts, and reports the money', () => {
    const rows = readFileSync('benchmarks/support/sample-export.jsonl', 'utf8')
      .split('\n').filter((l) => l.trim().length > 0).map((l) => JSON.parse(l));
    const report: PipelineReport = runPipeline(rows, 'sla', 'benchmarks/support/pack.json', '/tmp/pipeline-test-verdicts.jsonl', '2026-09-25T00:00:00Z');
    expect(report.cases).toBe(rows.length);
    expect(report.decided + report.rejected).toBe(rows.length);
    expect(report.money.recoveryUsd).toBeGreaterThanOrEqual(0);
    const categories = Object.entries(report.agreement);
    expect(categories.length).toBeGreaterThanOrEqual(2);
    // every recovery dispute names a ticket and cites a proof summary
    for (const d of report.disputes.filter((x) => x.category === 'RECOVERY')) {
      expect(d.ticket).toMatch(/^TCK-/);
      expect(d.summary).toContain('Credit of');
    }
  });

  it('is deterministic: the same export decides identically', () => {
    const rows = readFileSync('benchmarks/support/sample-export.jsonl', 'utf8')
      .split('\n').filter((l) => l.trim().length > 0).map((l) => JSON.parse(l));
    const a = runPipeline(rows, 'sla', 'benchmarks/support/pack.json', '/tmp/pipeline-test-verdicts.jsonl', '2026-09-25T00:00:00Z');
    const b = runPipeline(rows, 'sla', 'benchmarks/support/pack.json', '/tmp/pipeline-test-verdicts.jsonl', '2026-09-25T00:00:00Z');
    expect(a.money).toEqual(b.money);
    expect(a.agreement).toEqual(b.agreement);
  });
});
