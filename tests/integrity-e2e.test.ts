import { spawnSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { describe, expect, it } from 'vitest';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { checkIntegrity } from '../src/knowledge/integrity.js';
import { createServer } from '../src/mcp/server.js';
import { MemoryStore } from '../src/store/store.js';

class ForbiddenLlm implements LlmClient {
  async complete(_messages: ChatMessage[]): Promise<string> {
    throw new Error('integrity checks must not call an LLM');
  }
}

describe('v0.9 integrity end to end', () => {
  it('is byte-stable across CLI, reload, MCP, and a resolving mutation', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rembero-integrity-e2e-'));
    const home = join(root, 'home');
    // The gate is on by default; this test deliberately seeds a violating store
    // to exercise `check`, so it opts out explicitly for the seed write.
    const env = {
      ...process.env,
      REMBERO_HOME: home,
      REMBERO_INTEGRITY_MODE: 'off',
    };
    const cli = resolve('dist/cli.js');
    const program = [
      'status(mira, active).',
      'status(mira, terminated).',
      'status(zoe, active).',
      'status(zoe, terminated).',
      ':- status(Person, active), status(Person, terminated).',
    ].join(' ');

    const asserted = spawnSync(process.execPath, [cli, 'assert', program], {
      encoding: 'utf8',
      env,
    });
    expect(asserted.status).toBe(0);

    const runCliCheck = () =>
      spawnSync(process.execPath, [cli, 'check'], { encoding: 'utf8', env });
    const first = runCliCheck();
    const second = runCliCheck();
    expect(first.status).toBe(2);
    expect(second.status).toBe(2);
    expect(second.stdout).toBe(first.stdout);
    const cliPayload = JSON.parse(first.stdout);
    expect(
      cliPayload.checks[0].rows.map(
        (row: { bindings: unknown }) => row.bindings,
      ),
    ).toEqual([{ Person: 'mira' }, { Person: 'zoe' }]);

    const store = new MemoryStore(join(home, 'memory'));
    const direct = checkIntegrity(
      store.clausesFor(['default']),
      store.sourcesFor(['default']),
    );
    expect(direct).toEqual(cliPayload);

    const server = createServer({ store, llm: new ForbiddenLlm() });
    const client = new Client({ name: 'integrity-e2e', version: '1.0.0' });
    const [clientTransport, serverTransport] =
      InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const checked = await client.callTool({
        name: 'check_integrity',
        arguments: {},
      });
      const text = checked.content.find((item) => item.type === 'text');
      const mcpPayload = JSON.parse(text?.type === 'text' ? text.text : '');
      expect(mcpPayload).toEqual(cliPayload);

      const forgotten = spawnSync(
        process.execPath,
        [cli, 'forget', 'status(mira, terminated)'],
        { encoding: 'utf8', env },
      );
      expect(forgotten.status).toBe(0);

      const resolvedCli = runCliCheck();
      expect(resolvedCli.status).toBe(2);
      const resolvedPayload = JSON.parse(resolvedCli.stdout);
      expect(resolvedPayload).toMatchObject({
        status: 'violations',
        violationCount: 1,
        checks: [{ rows: [{ bindings: { Person: 'zoe' } }] }],
      });
      expect(resolvedPayload.checks[0].id).toBe(cliPayload.checks[0].id);

      const checkedAgain = await client.callTool({
        name: 'check_integrity',
        arguments: {},
      });
      const againText = checkedAgain.content.find(
        (item) => item.type === 'text',
      );
      expect(
        JSON.parse(againText?.type === 'text' ? againText.text : ''),
      ).toEqual(resolvedPayload);
    } finally {
      await client.close();
      await server.close();
    }
  });
});
