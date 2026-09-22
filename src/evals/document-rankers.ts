/**
 * The page rankers the benchmark compares, behind one interface.
 *
 * A ranker is built once per document (that is the point of item 3: the product's search is
 * rebuilt per question) and then answers `rank(question, limit)` with window ids best first.
 *
 *   product          the product's own retrieveSessions ranking, the baseline
 *   bm25             Okapi BM25 over pages
 *   dense            cosine over local nomic-embed-text vectors
 *   hybrid           reciprocal-rank fusion of bm25 and dense
 *   hybrid+rerank    hybrid's top 30 re-scored by TypeSafe (0.7 evidence + 0.3 relevance)
 *   decomposed       hybrid run once per sub-query an LLM split the question into, fused
 *   decomposed+rerank  both
 *
 * Decomposition exists because 73% of XL-DocBench questions cite more than one page — "which
 * standard is in section X but not section Y" needs X and Y, and one query ranks one of them.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { questionKindFromText } from '../knowledge/question-kind.js';
import { retrieveSessions, type RetrievableSession } from '../knowledge/session-retrieval.js';
import { Bm25Index, DenseIndex, reciprocalRankFusion, type Embed } from './document-index.js';
import { pageRangeFromId } from './document-corpus.js';
import { rerankSessionOrder, type TypesafeNouls } from './typesafe-rerank.js';
import type { LongMemEvalCompletionClient } from './longmemeval-answer.js';

export const RANKERS = [
  'product',
  'bm25',
  'dense',
  'hybrid',
  'hybrid+rerank',
  'decomposed',
  'decomposed+rerank',
  'facts',
  'hybrid+facts',
] as const;
export type RankerName = (typeof RANKERS)[number];

export interface DocumentRanker {
  name: RankerName;
  rank(question: string, limit: number): Promise<string[]>;
}

export interface RankerDeps {
  embed?: Embed;
  /** Where dense vectors for this document are cached. */
  denseCachePath?: string;
  nouls?: TypesafeNouls;
  decomposer?: QuestionDecomposer;
  /** Facts extracted from each page (page number → facts), for the fact-pointer rankers. */
  factsByPage?: ReadonlyMap<number, readonly string[]>;
}

/** How deep each fused list goes before fusion or re-ranking. */
const CANDIDATES = 100;
const RERANK_POOL = 30;

/** Build the named ranker over one document's windows. */
export async function buildDocumentRanker(
  name: RankerName,
  sessions: readonly RetrievableSession[],
  deps: RankerDeps = {},
): Promise<DocumentRanker> {
  const windows = sessions.map((session) => ({ id: session.id, text: session.turns.map((t) => t.text).join('\n') }));
  const byId = new Map(sessions.map((session) => [session.id, session]));

  if (name === 'product') {
    const askedAt = new Date(sessions[sessions.length - 1]?.date ?? Date.now());
    return {
      name,
      async rank(question, limit) {
        const { ranked } = await retrieveSessions(question, askedAt, sessions, {
          kind: questionKindFromText(question),
          // the same depth whatever the question's kind, so rankers are compared like for like
          topK: limit,
          aggregationTopK: limit,
          temporalTopK: limit,
          contextBytes: 24_576,
          dateDistances: false,
          computedNotes: false,
        });
        return ranked.slice(0, limit);
      },
    };
  }

  const bm25 = new Bm25Index(windows);
  if (name === 'bm25') return { name, rank: async (question, limit) => bm25.rank(question, limit) };

  const factPointer = () => {
    if (deps.factsByPage === undefined) throw new Error(`ranker ${name} needs extracted facts`);
    return new FactPointerIndex(sessions.map((session) => session.id), deps.factsByPage);
  };
  if (name === 'facts') {
    const facts = factPointer();
    return { name, rank: async (question, limit) => facts.rank(question, limit) };
  }

  if (deps.embed === undefined) throw new Error(`ranker ${name} needs an embedding function`);
  const dense = await DenseIndex.build(windows, deps.embed, { cachePath: deps.denseCachePath });
  if (name === 'dense') return { name, rank: (question, limit) => dense.rank(question, limit) };

  const hybrid = async (query: string, limit: number) =>
    reciprocalRankFusion([bm25.rank(query, CANDIDATES), await dense.rank(query, CANDIDATES)], limit);

  const decomposed = async (question: string, limit: number) => {
    if (deps.decomposer === undefined) throw new Error(`ranker ${name} needs a question decomposer`);
    const parts = await deps.decomposer(question);
    // the whole question always votes too, so a bad split cannot lose what plain hybrid found
    const lists = await Promise.all([question, ...parts].map((query) => hybrid(query, CANDIDATES)));
    return interleaveFirst(lists, reciprocalRankFusion(lists, CANDIDATES), limit);
  };

  const reranked = async (question: string, ordered: string[], limit: number) => {
    if (deps.nouls === undefined) throw new Error(`ranker ${name} needs a TypeSafe client`);
    const { order } = await rerankSessionOrder(ordered, {
      question,
      questionDate: '2020-01-01',
      pool: RERANK_POOL,
      sessionChars: 8_000,
      contextRoles: 'user',
      sessionFor: (id) => {
        const session = byId.get(id);
        return session === undefined
          ? undefined
          : { date: session.date, turns: session.turns.map((turn) => ({ role: turn.role, content: turn.text })) };
      },
      nouls: deps.nouls,
    });
    return order.slice(0, limit);
  };

  if (name === 'hybrid+facts') {
    const facts = factPointer();
    return {
      name,
      rank: async (question, limit) =>
        reciprocalRankFusion([await hybrid(question, CANDIDATES), facts.rank(question, CANDIDATES)], limit),
    };
  }

  switch (name) {
    case 'hybrid':
      return { name, rank: hybrid };
    case 'hybrid+rerank':
      return { name, rank: async (question, limit) => reranked(question, await hybrid(question, CANDIDATES), limit) };
    case 'decomposed':
      return { name, rank: decomposed };
    case 'decomposed+rerank':
      return {
        name,
        rank: async (question, limit) => reranked(question, await decomposed(question, CANDIDATES), limit),
      };
  }
}

/**
 * Fusion alone lets the whole-question list dominate, because it votes on every item; a
 * comparison question needs the *top* hit of each part in the context. So the first slots go to
 * each list's best page, then the fused order fills the rest.
 */
export function interleaveFirst(lists: ReadonlyArray<readonly string[]>, fused: readonly string[], limit: number): string[] {
  const out: string[] = [];
  for (const list of lists) {
    const top = list.find((id) => !out.includes(id));
    if (top !== undefined && out.length < limit) out.push(top);
  }
  for (const id of fused) {
    if (out.length >= limit) break;
    if (!out.includes(id)) out.push(id);
  }
  return out;
}

/** Splits a question into the separate searches it needs. */
export type QuestionDecomposer = (question: string) => Promise<string[]>;

export const DECOMPOSE_PROMPT =
  'You write search queries for finding pages in a long document. Split the question into ' +
  'the separate passages it needs: one short query per section, item, table or figure it ' +
  'refers to (a comparison of two sections needs two queries). Keep the document\'s own words ' +
  'and any section names. Output 1 to 3 queries, one per line, and nothing else.';

/** An LLM decomposer with an on-disk cache, so a sweep pays for each question once. */
export function llmDecomposer(client: LongMemEvalCompletionClient, cachePath: string): QuestionDecomposer {
  let cache: Record<string, string[]> = existsSync(cachePath)
    ? (JSON.parse(readFileSync(cachePath, 'utf8')) as Record<string, string[]>)
    : {};
  return async (question) => {
    const key = createHash('sha256').update(`${client.model}\u0000${question}`).digest('hex');
    const hit = cache[key];
    if (hit !== undefined) return hit;
    const reply = await client.completeWithUsage([
      { role: 'system', content: DECOMPOSE_PROMPT },
      { role: 'user', content: question },
    ]);
    const parts = reply.content
      .split('\n')
      .map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').trim())
      .filter((line) => line.length > 0)
      .slice(0, 3);
    // re-read before writing: concurrent questions share the file
    cache = existsSync(cachePath) ? (JSON.parse(readFileSync(cachePath, 'utf8')) as Record<string, string[]>) : {};
    cache[key] = parts;
    mkdirSync(dirname(cachePath), { recursive: true });
    writeFileSync(cachePath, JSON.stringify(cache, null, 1));
    return parts;
  };
}

/**
 * Search the extracted facts, then answer with the pages they came from.
 *
 * Each fact is a BM25 document (`budget_dollars(kestrel, 4200000).` tokenizes to budget, dollars,
 * kestrel, 4200000). The top facts vote for their page with 1 / log2(rank + 1), the same
 * aggregation the chat retrieval uses to roll turns up into sessions, so a page with several
 * matching facts beats a page with one. The fact is a pointer; the reader still reads the page.
 */
export class FactPointerIndex {
  private readonly index: Bm25Index;
  private readonly windowOfFact: string[] = [];

  constructor(windowIds: readonly string[], factsByPage: ReadonlyMap<number, readonly string[]>, private readonly topFacts = 60) {
    const windowOfPage = new Map<number, string>();
    for (const id of windowIds) {
      const range = pageRangeFromId(id);
      if (range === undefined) continue;
      for (let page = range.firstPage; page <= range.lastPage; page += 1) windowOfPage.set(page, id);
    }
    const docs: Array<{ id: string; text: string }> = [];
    for (const [page, facts] of factsByPage) {
      const windowId = windowOfPage.get(page);
      if (windowId === undefined) continue;
      for (const fact of facts) {
        docs.push({ id: String(docs.length), text: fact });
        this.windowOfFact.push(windowId);
      }
    }
    this.index = new Bm25Index(docs);
  }

  get facts(): number {
    return this.windowOfFact.length;
  }

  rank(question: string, limit: number): string[] {
    const votes = new Map<string, number>();
    const firstSeen = new Map<string, number>();
    this.index.rank(question, this.topFacts).forEach((factId, rank) => {
      const windowId = this.windowOfFact[Number(factId)]!;
      votes.set(windowId, (votes.get(windowId) ?? 0) + 1 / Math.log2(rank + 2));
      if (!firstSeen.has(windowId)) firstSeen.set(windowId, rank);
    });
    return [...votes.entries()]
      .sort((left, right) => right[1] - left[1] || firstSeen.get(left[0])! - firstSeen.get(right[0])!)
      .slice(0, limit)
      .map(([id]) => id);
  }
}
