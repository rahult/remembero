/**
 * A document's page index, built once and queried many times.
 *
 * The product's lexical search (`searchKnowledge`) scores a matching word a flat 45 points and
 * is rebuilt from the clauses on every call. Over a chat history that is fine; over a 1000-page
 * manual it is both slow (ranking time grew 0.1s → 0.8s per question from 100 to 1000 pages) and
 * blind to rarity — "section" counts as much as "Brady". This module holds the alternatives the
 * benchmark compares, each built once per document:
 *
 *   bm25   — term frequency x inverse document frequency, length-normalised (k1 1.2, b 0.75)
 *   dense  — cosine over local embeddings (nomic-embed-text via Ollama), cached on disk
 *   hybrid — reciprocal-rank fusion of the two, so either can rescue the other's misses
 *
 * Every ranker returns window ids best first; the caller cuts to depth.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { STOPWORDS } from '../knowledge/computed-notes.js';

export interface IndexedWindow {
  id: string;
  text: string;
}

/** Lower-cased word tokens, digits kept (a year or a figure is often the distinctive term). */
export function tokenize(text: string): string[] {
  const words = text.toLowerCase().normalize('NFKC').match(/[\p{L}\p{N}]+/gu) ?? [];
  return words.filter((word) => word.length > 1 && !STOPWORDS.has(word));
}

/** Okapi BM25 over whole windows. */
export class Bm25Index {
  private readonly postings = new Map<string, Array<{ doc: number; tf: number }>>();
  private readonly lengths: number[] = [];
  private readonly averageLength: number;

  constructor(
    readonly windows: readonly IndexedWindow[],
    private readonly k1 = 1.2,
    private readonly b = 0.75,
  ) {
    windows.forEach((window, doc) => {
      const counts = new Map<string, number>();
      const tokens = tokenize(window.text);
      for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
      this.lengths.push(tokens.length);
      for (const [term, tf] of counts) {
        const list = this.postings.get(term);
        if (list === undefined) this.postings.set(term, [{ doc, tf }]);
        else list.push({ doc, tf });
      }
    });
    const total = this.lengths.reduce((sum, length) => sum + length, 0);
    this.averageLength = this.lengths.length === 0 ? 0 : total / this.lengths.length;
  }

  /** Inverse document frequency, the BM25+ form that never goes negative. */
  idf(term: string): number {
    const df = this.postings.get(term)?.length ?? 0;
    const n = this.windows.length;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  scores(query: string): Map<number, number> {
    const scores = new Map<number, number>();
    for (const term of new Set(tokenize(query))) {
      const list = this.postings.get(term);
      if (list === undefined) continue;
      const idf = this.idf(term);
      for (const { doc, tf } of list) {
        const norm = 1 - this.b + (this.b * this.lengths[doc]!) / (this.averageLength || 1);
        const part = (idf * tf * (this.k1 + 1)) / (tf + this.k1 * norm);
        scores.set(doc, (scores.get(doc) ?? 0) + part);
      }
    }
    return scores;
  }

  rank(query: string, limit: number): string[] {
    return [...this.scores(query).entries()]
      .sort((left, right) => right[1] - left[1] || left[0] - right[0])
      .slice(0, limit)
      .map(([doc]) => this.windows[doc]!.id);
  }
}

/** Turns text into vectors; injected so tests need no model. */
export type Embed = (texts: string[], kind: 'document' | 'query') => Promise<number[][]>;

/** nomic-embed-text through a local Ollama, with the task prefixes that model is trained on. */
export function ollamaEmbed(
  model = 'nomic-embed-text',
  baseUrl = 'http://127.0.0.1:11434',
  maxChars = 6_000,
): Embed {
  return async (texts, kind) => {
    const prefix = kind === 'query' ? 'search_query: ' : 'search_document: ';
    const response = await fetch(`${baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        input: texts.map((text) => prefix + text.slice(0, maxChars)),
        // the default 2048-token window truncates silently; say so explicitly
        options: { num_ctx: 2048 },
        truncate: true,
      }),
    });
    if (!response.ok) throw new Error(`embedding request failed with status ${response.status}`);
    const body = (await response.json()) as { embeddings?: number[][] };
    if (!Array.isArray(body.embeddings) || body.embeddings.length !== texts.length) {
      throw new Error('embedding response did not return one vector per input');
    }
    return body.embeddings;
  };
}

function normalise(vector: number[]): number[] {
  const length = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / length);
}

/** Cosine ranking over window embeddings, computed once and cached against the text. */
export class DenseIndex {
  private constructor(
    readonly windows: readonly IndexedWindow[],
    private readonly vectors: number[][],
    private readonly embed: Embed,
  ) {}

  /**
   * Build, or load from `cachePath` when the cache was made from exactly these windows (the key
   * is a hash of every window's id and text, so a re-extracted PDF never reuses stale vectors).
   */
  static async build(
    windows: readonly IndexedWindow[],
    embed: Embed,
    options: { cachePath?: string; batch?: number; onProgress?: (done: number, total: number) => void } = {},
  ): Promise<DenseIndex> {
    const key = createHash('sha256')
      .update(windows.map((window) => `${window.id}\u0000${window.text}`).join('\u0001'))
      .digest('hex');
    if (options.cachePath !== undefined && existsSync(options.cachePath)) {
      const cached = JSON.parse(readFileSync(options.cachePath, 'utf8')) as { key: string; vectors: number[][] };
      if (cached.key === key && cached.vectors.length === windows.length) {
        return new DenseIndex(windows, cached.vectors, embed);
      }
    }
    const batch = options.batch ?? 32;
    const vectors: number[][] = [];
    for (let start = 0; start < windows.length; start += batch) {
      const slice = windows.slice(start, start + batch).map((window) => window.text);
      vectors.push(...(await embed(slice, 'document')).map(normalise));
      options.onProgress?.(vectors.length, windows.length);
    }
    if (options.cachePath !== undefined) {
      mkdirSync(dirname(options.cachePath), { recursive: true });
      writeFileSync(options.cachePath, JSON.stringify({ key, vectors }));
    }
    return new DenseIndex(windows, vectors, embed);
  }

  async rank(query: string, limit: number): Promise<string[]> {
    const [raw] = await this.embed([query], 'query');
    const vector = normalise(raw!);
    return this.vectors
      .map((candidate, doc) => ({ doc, score: candidate.reduce((sum, value, i) => sum + value * vector[i]!, 0) }))
      .sort((left, right) => right.score - left.score || left.doc - right.doc)
      .slice(0, limit)
      .map(({ doc }) => this.windows[doc]!.id);
  }
}

/**
 * Reciprocal-rank fusion: each list votes 1 / (k + rank). k = 60 is the value from the original
 * paper and the one nearly everyone uses; it keeps one list's top hit from drowning the other.
 */
export function reciprocalRankFusion(lists: ReadonlyArray<readonly string[]>, limit: number, k = 60): string[] {
  const scores = new Map<string, number>();
  const firstSeen = new Map<string, number>();
  let order = 0;
  for (const list of lists) {
    list.forEach((id, rank) => {
      scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1));
      if (!firstSeen.has(id)) firstSeen.set(id, order++);
    });
  }
  return [...scores.entries()]
    .sort((left, right) => right[1] - left[1] || firstSeen.get(left[0])! - firstSeen.get(right[0])!)
    .slice(0, limit)
    .map(([id]) => id);
}
