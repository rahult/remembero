/**
 * The rows that have no product to pin: they live in the harness because there is nothing to
 * install. full-context hands back the whole haystack newest first and lets the context
 * budget do the cutting; bm25 is textbook BM25 over whole sessions; embed is cosine over
 * whole-session embeddings from a local nomic-embed-text through Ollama. None of them calls
 * a paid model. openMemorySystem also resolves an adapter manifest to a long-lived process.
 */
import { performance } from 'node:perf_hooks';
import type { EmbeddingClient } from '../llm/embeddings.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';
import { loadExternalAdapterManifest } from './memory-stack-external.js';
import { createMemorySystemProcess } from './memory-systems-process.js';
import {
  MEMORY_SYSTEM_MAX_SESSIONS,
  MEMORY_SYSTEMS_PROTOCOL_VERSION,
  type MemorySystemClient,
  type MemorySystemRequest,
  type MemorySystemResponse,
} from './memory-systems-protocol.js';

export const BUILTIN_MEMORY_SYSTEMS = [
  'builtin:full-context',
  'builtin:bm25',
  'builtin:embed',
] as const;

export type BuiltinMemorySystemId = (typeof BUILTIN_MEMORY_SYSTEMS)[number];

export function isBuiltinMemorySystem(
  value: string,
): value is BuiltinMemorySystemId {
  return (BUILTIN_MEMORY_SYSTEMS as readonly string[]).includes(value);
}

export interface MemorySystemFactoryOptions {
  embeddings?: EmbeddingClient;
  extractor?: LongMemEvalCompletionClient;
  extractionCacheDir?: string;
  extractionCharacters?: number;
  extractionMaxTokens?: number;
  timeoutMs?: number;
}

const ZERO_USAGE = {
  modelCalls: 0,
  inputTokens: 0,
  outputTokens: 0,
  costUsd: 0,
} as const;
const BM25_K1 = 1.2;
const BM25_B = 0.75;
/** nomic-embed-text takes 8192 tokens; 8,000 characters is comfortably inside it. */
const EMBED_CHARACTERS = 8_000;
const EMBED_BATCH = 16;

export function sessionText(
  session: MemorySystemRequest['sessions'][number],
): string {
  return session.turns
    .map(({ role, content }) => `${role}: ${content}`)
    .join('\n');
}

function words(text: string): string[] {
  return (
    text
      .toLowerCase()
      .normalize('NFKC')
      .match(/[\p{L}\p{N}_]+/gu) ?? []
  ).filter((word) => word.length > 1);
}

function fullContext(request: MemorySystemRequest): MemorySystemResponse {
  // newest first; the context builder's byte budget is what truncates the haystack. The cap
  // is the protocol's own limit, and also what buildLongMemEvalAnswerContext will accept.
  const ordered = [...request.sessions]
    .sort(
      (left, right) =>
        right.date.localeCompare(left.date) || left.id.localeCompare(right.id),
    )
    .slice(0, MEMORY_SYSTEM_MAX_SESSIONS);
  return {
    questionId: request.questionId,
    retrieved: ordered.map(({ id }, index) => ({
      sessionId: id,
      rank: index + 1,
      score: ordered.length - index,
    })),
    memories: [],
    unsupported: ['memories'],
    usage: { ...ZERO_USAGE },
    wallMs: { ingest: 0, search: 0 },
  };
}

function bm25(request: MemorySystemRequest): MemorySystemResponse {
  const started = performance.now();
  const documents = request.sessions.map((session) => ({
    id: session.id,
    terms: words(sessionText(session)),
  }));
  const averageLength =
    documents.reduce((total, { terms }) => total + terms.length, 0) /
    Math.max(1, documents.length);
  const documentFrequency = new Map<string, number>();
  for (const { terms } of documents) {
    for (const term of new Set(terms)) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
  }
  const query = [...new Set(words(request.question))];
  const ranked = documents
    .map(({ id, terms }) => {
      const counts = new Map<string, number>();
      for (const term of terms) counts.set(term, (counts.get(term) ?? 0) + 1);
      let score = 0;
      for (const term of query) {
        const frequency = counts.get(term);
        if (frequency === undefined) continue;
        const n = documentFrequency.get(term) ?? 0;
        const idf = Math.log(1 + (documents.length - n + 0.5) / (n + 0.5));
        score +=
          idf *
          ((frequency * (BM25_K1 + 1)) /
            (frequency +
              BM25_K1 *
                (1 -
                  BM25_B +
                  (BM25_B * terms.length) / Math.max(1, averageLength))));
      }
      return { id, score };
    })
    .filter(({ score }) => score > 0)
    .sort(
      (left, right) =>
        right.score - left.score || left.id.localeCompare(right.id),
    )
    .slice(0, request.topK);
  return {
    questionId: request.questionId,
    retrieved: ranked.map(({ id, score }, index) => ({
      sessionId: id,
      rank: index + 1,
      score,
    })),
    memories: [],
    unsupported: ['memories'],
    usage: { ...ZERO_USAGE },
    wallMs: { ingest: 0, search: performance.now() - started },
  };
}

function cosine(left: readonly number[], right: readonly number[]): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  const magnitude = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return magnitude === 0 ? 0 : dot / magnitude;
}

async function embedded(
  request: MemorySystemRequest,
  embeddings: EmbeddingClient,
): Promise<MemorySystemResponse> {
  const ingestStarted = performance.now();
  const texts = request.sessions.map((session) =>
    sessionText(session).slice(0, EMBED_CHARACTERS),
  );
  const vectors: number[][] = [];
  for (let index = 0; index < texts.length; index += EMBED_BATCH) {
    const batch = await embeddings.embed(
      texts.slice(index, index + EMBED_BATCH),
    );
    vectors.push(...batch.vectors);
  }
  const ingest = performance.now() - ingestStarted;
  const searchStarted = performance.now();
  const query = (await embeddings.embed([request.question])).vectors[0] ?? [];
  const ranked = request.sessions
    .map((session, index) => ({
      id: session.id,
      score: cosine(query, vectors[index] ?? []),
    }))
    .sort(
      (left, right) =>
        right.score - left.score || left.id.localeCompare(right.id),
    )
    .slice(0, request.topK);
  return {
    questionId: request.questionId,
    retrieved: ranked.map(({ id, score }, index) => ({
      sessionId: id,
      rank: index + 1,
      score,
    })),
    memories: [],
    unsupported: ['memories'],
    // local Ollama: tokens are not billed and the provider reports no cost
    usage: { ...ZERO_USAGE },
    wallMs: { ingest, search: performance.now() - searchStarted },
  };
}

export function createBuiltinMemorySystem(
  id: BuiltinMemorySystemId,
  options: MemorySystemFactoryOptions,
): MemorySystemClient {
  const embeddings = options.embeddings;
  if (id === 'builtin:embed' && embeddings === undefined) {
    throw new Error('builtin:embed needs an embedding client');
  }
  return {
    id,
    async request(request) {
      if (id === 'builtin:full-context') return fullContext(request);
      if (id === 'builtin:bm25') return bm25(request);
      return await embedded(request, embeddings!);
    },
    async close() {},
  };
}

/**
 * `builtin:*` or a path to an adapter manifest whose `protocol` is
 * rembero.memory-systems.v1. The manifest's own timeout and output cap are honoured.
 */
export async function openMemorySystem(
  spec: string,
  options: MemorySystemFactoryOptions,
): Promise<MemorySystemClient> {
  if (spec.startsWith('builtin:')) {
    if (!isBuiltinMemorySystem(spec)) {
      throw new Error(
        `unknown built-in memory system ${spec}; known: ${BUILTIN_MEMORY_SYSTEMS.join(', ')}`,
      );
    }
    return createBuiltinMemorySystem(spec, options);
  }
  const manifest = await loadExternalAdapterManifest(spec);
  if (manifest.protocol !== MEMORY_SYSTEMS_PROTOCOL_VERSION) {
    throw new Error(
      `${spec} declares protocol ${String(manifest.protocol)}; --memory-system needs ${MEMORY_SYSTEMS_PROTOCOL_VERSION}`,
    );
  }
  return createMemorySystemProcess({
    id: manifest.descriptor.id,
    executable: manifest.command.executable,
    ...(manifest.command.args === undefined
      ? {}
      : { args: manifest.command.args }),
    ...(manifest.command.workingDirectory === undefined
      ? {}
      : { workingDirectory: manifest.command.workingDirectory }),
    ...(manifest.command.env === undefined
      ? {}
      : { env: manifest.command.env }),
    ...(options.timeoutMs === undefined
      ? manifest.command.timeoutMs === undefined
        ? {}
        : { timeoutMs: manifest.command.timeoutMs }
      : { timeoutMs: options.timeoutMs }),
    ...(manifest.command.maxOutputBytes === undefined
      ? {}
      : { maxResponseBytes: manifest.command.maxOutputBytes }),
  });
}
