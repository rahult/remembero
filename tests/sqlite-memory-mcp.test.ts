import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { beforeAll, describe, expect, it } from 'vitest';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';
import { createServer } from '../src/mcp/server.js';
import { openRememberoDatabase } from '../src/sqlite/extension.js';
import { serializeClause } from '../src/engine/index.js';

const nodeMajor = Number(process.versions.node.split('.')[0]);
const projectRoot = resolve(import.meta.dirname, '..');
let extensionPath: string;

class UnexpectedLlm implements LlmClient {
  async complete(_messages: ChatMessage[]): Promise<string> {
    throw new Error('This SQLite-backed MCP flow must not call an LLM.');
  }
}

describe.skipIf(nodeMajor < 22)('SQLite-backed MCP integration', () => {
  beforeAll(() => {
    const output = execFileSync('sh', ['native/build.sh'], {
      cwd: projectRoot,
      encoding: 'utf8',
    });
    extensionPath = output.trim().split('\n').at(-1) ?? '';
    expect(existsSync(extensionPath)).toBe(true);
  });

  it('serves governed memory over the real MCP protocol from the same SQLite database', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'rembero-sqlite-mcp-'));
    const path = join(directory, 'application.db');
    const database = await openRememberoDatabase(path, { extensionPath });
    database.exec(`
      CREATE TABLE works_at(person TEXT NOT NULL, company TEXT NOT NULL);
      INSERT INTO works_at VALUES ('mira', 'acme'), ('rahul', 'acme');
    `);

    const server = createServer({
      store: database.memory,
      llm: new UnexpectedLlm(),
    });
    const client = new Client({ name: 'rembero-sqlite-mcp-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);

    try {
      expect(database.prepare('SELECT count(*) AS count FROM works_at').get()).toEqual({
        count: 2,
      });
      expect(
        database.datalogQuery(
          'colleague(X, Y) :- works_at(X, C), works_at(Y, C), X != Y.'
        )
      ).toEqual([
        { X: 'mira', Y: 'rahul' },
        { X: 'rahul', Y: 'mira' },
      ]);

      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name)).toEqual(
        expect.arrayContaining(['assert_facts', 'query', 'explain_query', 'history'])
      );

      const asserted = await client.callTool({
        name: 'assert_facts',
        arguments: {
          clauses: 'prefers(mira, deterministic_systems).',
          opId: 'sqlite-mcp-assert',
        },
      });
      expect(asserted.isError).not.toBe(true);

      const queried = await client.callTool({
        name: 'query',
        arguments: { query: 'prefers(mira, Preference)' },
      });
      const queriedText = queried.content.find((item) => item.type === 'text');
      expect(JSON.parse(queriedText?.type === 'text' ? queriedText.text : '')).toMatchObject({
        bindings: [{ Preference: 'deterministic_systems' }],
      });

      const explained = await client.callTool({
        name: 'explain_query',
        arguments: { query: 'prefers(mira, Preference)' },
      });
      const explainedText = explained.content.find((item) => item.type === 'text');
      expect(JSON.parse(explainedText?.type === 'text' ? explainedText.text : '')).toMatchObject({
        rows: [{ proofs: [{ sources: [{ opId: 'sqlite-mcp-assert' }] }] }],
      });
    } finally {
      await client.close();
      await server.close();
      database.close();
    }

    const reopened = await openRememberoDatabase(path, { extensionPath });
    try {
      expect(reopened.prepare('SELECT person, company FROM works_at ORDER BY person').all()).toEqual([
        { person: 'mira', company: 'acme' },
        { person: 'rahul', company: 'acme' },
      ]);
      expect(reopened.memory.load('default').map(serializeClause)).toEqual([
        'prefers(mira, deterministic_systems).',
      ]);
    } finally {
      reopened.close();
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
