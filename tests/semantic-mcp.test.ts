import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { createServer } from '../src/mcp/server.js';
import { createSemanticLedger } from '../src/ledger/semantic-ledger.js';
import { MemoryStore } from '../src/store/store.js';
import type { ChatMessage, LlmClient } from '../src/llm/client.js';

class NoopLlm implements LlmClient {
  async complete(_messages: ChatMessage[]): Promise<string> {
    throw new Error('LLM should not be called by semantic version tools');
  }
}

describe('semantic version MCP surface', () => {
  let DatabaseSync: typeof import('node:sqlite').DatabaseSync;

  beforeAll(async () => {
    ({ DatabaseSync } = await import('node:sqlite'));
  });

  it('captures, reviews, and promotes through a real MCP client/server round trip', async () => {
    const database = new DatabaseSync(':memory:');
    const store = new MemoryStore(mkdtempSync(join(tmpdir(), 'remembero-semantic-mcp-')));
    store.assert('default', 'status(atlas, active).', { opId: 'mcp-version-baseline' });
    const server = createServer({
      store,
      llm: new NoopLlm(),
      semanticLedger: createSemanticLedger(database),
    });
    const client = new Client({ name: 'remembero-semantic-test', version: '1.0.0' });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name)).toEqual(
        expect.arrayContaining([
          'capture_semantic_version',
          'list_semantic_versions',
          'inspect_semantic_version',
          'diff_semantic_versions',
          'review_semantic_version',
          'promote_semantic_version',
          'semantic_ref_history',
        ])
      );

      const capture = await client.callTool({
        name: 'capture_semantic_version',
        arguments: { label: 'mcp@baseline', ref: 'main' },
      });
      const captureText = capture.content.find((item) => item.type === 'text');
      const baseline = JSON.parse(captureText?.type === 'text' ? captureText.text : '');
      expect(baseline.version).toMatchObject({
        digest: expect.stringMatching(/^[a-f0-9]{64}$/),
        members: expect.arrayContaining([
          expect.objectContaining({ key: 'knowledge' }),
          expect.objectContaining({ key: 'documents' }),
        ]),
      });

      store.assert('default', 'owner(atlas, rahul).', { opId: 'mcp-version-candidate' });
      const candidateCall = await client.callTool({
        name: 'capture_semantic_version',
        arguments: { label: 'mcp@candidate', ref: 'main' },
      });
      const candidateText = candidateCall.content.find((item) => item.type === 'text');
      const candidatePayload = JSON.parse(candidateText?.type === 'text' ? candidateText.text : '');
      const candidateDigest = candidatePayload.version.digest as string;
      const reviewCall = await client.callTool({
        name: 'review_semantic_version',
        arguments: { candidate: candidateDigest },
      });
      const reviewText = reviewCall.content.find((item) => item.type === 'text');
      const review = JSON.parse(reviewText?.type === 'text' ? reviewText.text : '');
      expect(review).toMatchObject({
        candidateVersionDigest: candidateDigest,
        assessment: { digest: expect.stringMatching(/^[a-f0-9]{64}$/) },
      });
      const acceptedReviewDimensions = review.assessment.checks
        .filter((check: { status: string }) => check.status === 'review')
        .map((check: { dimension: string }) => check.dimension);
      const promotedCall = await client.callTool({
        name: 'promote_semantic_version',
        arguments: {
          ref: 'main',
          candidate: candidateDigest,
          assessment: review.assessment.digest,
          operationId: 'mcp-promote-semantic-candidate',
          acceptedReviewDimensions,
        },
      });
      const promotedText = promotedCall.content.find((item) => item.type === 'text');
      expect(JSON.parse(promotedText?.type === 'text' ? promotedText.text : '')).toMatchObject({
        outcome: 'accepted',
        candidateVersionDigest: candidateDigest,
      });
    } finally {
      await client.close();
      await server.close();
      database.close();
    }
  });
});
